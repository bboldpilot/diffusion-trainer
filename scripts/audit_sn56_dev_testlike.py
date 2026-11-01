"""CPU audit: only explicit training_data.json inputs; emits aggregate JSON.

Usage: python scripts/audit_sn56_dev_testlike.py ROOT > /tmp/neardup-audit.json
Near-duplicate means membership of a cross-split val_split MinHash component.
"""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys

import numpy as np
import val_split as V
from sn56_dev_testlike import exact_key, split_rows, unique_rows


def legacy_functions():
    tree = ast.parse(Path(__file__).with_name('tokenize_instruct.py').read_text())
    wanted = {'remove_empty_output_items', '_instruct_output_length', 'split_dataset'}
    ns = {'json': json, 'random': random}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name in wanted], type_ignores=[]), '<split>', 'exec'), ns)
    return ns


def audit(path):
    assert path.name == 'training_data.json'
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    funcs = legacy_functions()
    clean = funcs['remove_empty_output_items'](raw)
    unique = unique_rows(clean)
    with contextlib.redirect_stdout(io.StringIO()):
        groups = V._near_dedup_groups(unique, V.concat_all_text)
    group_of = {exact_key(unique[i]): gi for gi, group in enumerate(groups) for i in group}

    def rate(dev, train):
        train_groups = {group_of[exact_key(r)] for r in train}
        count = sum(group_of[exact_key(r)] in train_groups for r in dev)
        return {'dev': len(dev), 'train': len(train), 'near': count,
                'rate': count / len(dev) if dev else 0}

    old_rows = raw.copy()
    random.Random(42).shuffle(old_rows)
    old_rows = funcs['remove_empty_output_items'](old_rows)
    size = min(1000, max(min(400, len(old_rows)//5), len(old_rows)//33))
    with contextlib.redirect_stdout(io.StringIO()):
        dev0, train0 = V.stratified_split(old_rows, size, V.concat_all_text,
                                         funcs['_instruct_output_length'])
    dev, train = split_rows(clean)
    indices = np.random.default_rng(42).permutation(len(unique))
    floor = min(1000, int(len(unique)*.1))
    floor_dev = [unique[i] for i in indices[:floor]]
    floor_train = [unique[i] for i in indices[floor:]]
    rates = [rate(*split_rows(clean, seed))['rate'] for seed in range(10)]
    return {'path': str(path), 'sha256': hashlib.sha256(raw_bytes).hexdigest(),
            'raw': len(raw), 'invalid': len(raw)-len(clean),
            'exact_removed': len(clean)-len(unique),
            'grouped_rows': sum(len(g) for g in groups if len(g)>1),
            'legacy': rate(dev0, train0), 'testlike': rate(dev, train),
            'validator_floor': rate(floor_dev, floor_train),
            'seeds_0_9_mean': float(np.mean(rates)),
            'seeds_0_9_min': min(rates), 'seeds_0_9_max': max(rates)}


if __name__ == '__main__':
    root = Path(sys.argv[1])
    results = {}
    for task in ('451995f2', '1ee59641', 'b36ede07', 'bd68b438', '7923f3a2'):
        matches = list(root.glob('*/tasks/'+task+'*/training_data.json'))
        if len(matches) != 1:
            raise ValueError(f'{task}: expected one training file, got {len(matches)}')
        print(f'Auditing {task}', file=sys.stderr, flush=True)
        results[task] = audit(matches[0])
        print(json.dumps({task: results[task]}), file=sys.stderr, flush=True)
    print(json.dumps(results, indent=2))
