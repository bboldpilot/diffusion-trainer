"""CPU only: python3 scripts/tests/test_sn56_dev_testlike.py (numpy).

AST-load the real file split functions without importing Axolotl/torch/models.
Legacy golden hashes were captured from the pre-change split_dataset source.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import sn56_plan as P
import sn56_dev_testlike as D
import val_split as V
from audit_sn56_dev_testlike import legacy_functions


def rows():
    rng = random.Random(106)
    result = []
    for i in range(120):
        text = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz ', k=350))
        for variant in range(4):
            result.append({'instruct': text, 'output': f'answer {variant}'})
    return result + result[::13] + [{'instruct': 'invalid', 'output': ''}]


def run_split(data, **kwargs):
    with tempfile.TemporaryDirectory() as td:
        paths = [Path(td)/name for name in ('source.json', 'train.json', 'dev.json')]
        paths[0].write_text(json.dumps(data, ensure_ascii=False))
        with contextlib.redirect_stdout(io.StringIO()):
            legacy_functions()['split_dataset'](*(str(p) for p in paths), **kwargs)
        return paths[1].read_bytes(), paths[2].read_bytes()


class TestDevTestlike(unittest.TestCase):
    def test_default_off_golden_bytes(self):
        for seed, cap, expected in GOLDENS:
            with self.subTest(seed=seed, cap=cap):
                result = run_split(rows(), seed=seed, max_data_size=cap)
                self.assertEqual(tuple(hashlib.sha256(b).hexdigest() for b in result), expected)
                self.assertEqual(result, run_split(rows(), seed=seed, max_data_size=cap, dev_testlike=False))

    def test_scope_and_master_switch(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(P.FLAGS['SN56_DEV_TESTLIKE'], ('0', '0'))
            self.assertFalse(D.enabled('InstructTextTask'))
            os.environ['SN56_DEV_TESTLIKE'] = '1'
            self.assertTrue(D.enabled('InstructTextTask'))
            self.assertFalse(D.enabled('InstructTextTask', True))
            for kind in ('ChatTask', 'DpoTask', 'GrpoTask'):
                self.assertFalse(D.enabled(kind))
            os.environ[P.MASTER] = '0'
            self.assertFalse(D.enabled('InstructTextTask'))

    def test_random_split_and_near_duplicate_rate(self):
        data = D.unique_rows(rows()[:-1])
        with contextlib.redirect_stdout(io.StringIO()):
            groups = V._near_dedup_groups(data, V.concat_all_text)
        group_of = {i: g for g, indices in enumerate(groups) for i in indices}
        key_index = {D.exact_key(r): i for i, r in enumerate(data)}
        for seed in (0, 42, 123):
            dev, train = D.split_rows(data, seed)
            # Independent row-random reference, same datasets RNG semantics.
            perm = np.random.default_rng(seed).permutation(len(data))
            ndev = (len(data)+9)//10
            self.assertEqual(dev, [data[i] for i in perm[:ndev]])
            self.assertEqual(train, [data[i] for i in perm[ndev:]])
            train_groups = {group_of[key_index[D.exact_key(r)]] for r in train}
            actual = sum(group_of[key_index[D.exact_key(r)]] in train_groups for r in dev)/len(dev)
            reference = sum(group_of[i] in {group_of[j] for j in perm[ndev:]} for i in perm[:ndev])/ndev
            self.assertEqual(actual, reference)
            self.assertGreater(actual, .8)
        with contextlib.redirect_stdout(io.StringIO()):
            olddev, oldtrain = V.stratified_split(data, 48, V.concat_all_text)
        old_groups = {group_of[key_index[D.exact_key(r)]] for r in oldtrain}
        self.assertFalse(any(group_of[key_index[D.exact_key(r)]] in old_groups for r in olddev))

    def test_exact_identity_and_file_path(self):
        data = rows()
        data += [{'system': None, 'instruct': ' '+data[0]['instruct']+' ',
                  'input': '', 'output': data[0]['output'], 'ignored_metadata': 'x'}]
        train_b, dev_b = run_split(data, dev_testlike=True)
        train, dev = json.loads(train_b), json.loads(dev_b)
        keys = [D.exact_key(r) for r in train+dev]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(dev), 48)
        self.assertEqual(len(train)+len(dev), 480)
        self.assertEqual((dev, train), D.split_rows(rows()[:-1]))

    def test_sizes_determinism_and_global_rng(self):
        for n in (0, 1, 2, 11, 9999, 10001):
            data = [{'instruct': str(i), 'output': 'ok'} for i in range(n)]
            state = random.getstate()
            dev, train = D.split_rows(data)
            self.assertEqual(state, random.getstate())
            self.assertEqual(len(dev), 0 if n<2 else min(1000, (n+9)//10))
            self.assertEqual(len(dev)+len(train), n)
            self.assertEqual((dev, train), D.split_rows(data))
        self.assertEqual(D.split_rows([{'output':'x'}]*10), ([], [{'output':'x'}]))

    def test_config_records_resolved_scope(self):
        # Execute the actual config's final opt-in block with resolved LoRA state.
        tree = ast.parse((SCRIPTS/'instruct_config.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='get_training_json')
        block = ast.Module(body=fn.body[-3:-1], type_ignores=[])
        code = compile(block, '<config-tail>', 'exec')
        for value, kind, lora, expected in [
            ('0', 'InstructTextTask', False, False),
            ('1', 'InstructTextTask', False, True),
            ('1', 'InstructTextTask', True, False),
            ('1', 'ChatTask', False, False),
        ]:
            ns = {'train_info': {'task_type': kind}, 'run_config': {'use_lora': lora},
                  'train_request': {'unchanged': 1}}
            with patch.dict(os.environ, {'SN56_DEV_TESTLIKE': value}, clear=True):
                exec(code, ns)
            self.assertEqual(ns['train_request'],
                             {'unchanged': 1, 'dev_testlike': True} if expected else {'unchanged': 1})

    def test_training_dedup_gate(self):
        for rate in (0, .2, .20001, 1):
            self.assertEqual(D.training_dedup_enabled(rate), rate>.2)
            self.assertFalse(D.training_dedup_enabled(rate, True))
        # Check the actual trainer consumes the request marker at the dedup gate.
        source = (SCRIPTS/'train_instruct.py').read_text()
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id=='training_dedup_enabled']
        self.assertEqual(len(calls), 1)
        self.assertEqual([n.id for n in calls[0].args], ['near_dup_rate', '_dev_testlike'])


GOLDENS = [(42, -1, ('401b33e2ada364900ca04360d00cda6854d2ac72f1499007e1399f3961874bbd', '9d4fe84a70b8577faa6e8ff87c1d7b4d4070c6e7574686d3f403c0c4f52dc681')), (7, -1, ('778dfd8f358ac5a0532e4bf692a6a3ad7a23893cd406dfe71bcd087231d395f5', '7a343999066b95dc48b2ffb4d1a23cdcca20fc00b7446bfcdeb6e5ecd9ebca64')), (42, 150, ('f2bd5c1b4b9128138578a1f9f08ef6212db546123fdf0fc16786bf91822a42aa', 'a6c266b8c5765863ea4ab303fbaf51c7c4ee43dfcd5b102a16ffb3e116aa0557'))]

if __name__ == '__main__':
    unittest.main()
