"""Training-only, opt-in row-random InstructText development split.

Exact identity matches validator #1412. Use the requested ceil(10%), capped
at 1000; pinned upstream a57d9008 uses floor instead (at most one row less).
No MinHash grouping: a dev row may have a near-duplicate in optimization data.
"""
from math import ceil

import numpy as np

from sn56_plan import flag_on


def enabled(task_type, use_lora=False):
    return (task_type == "InstructTextTask" and not use_lora
            and flag_on("SN56_DEV_TESTLIKE"))


def exact_key(row):
    return tuple(str(row.get(col) or "").strip()
                 for col in ("system", "instruct", "input", "output"))


def unique_rows(rows):
    seen = set()
    result = []
    for row in rows:
        key = exact_key(row)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def split_rows(rows, seed=42):
    """Return (dev, train), preserving the first occurrence of exact rows.

    default_rng/permutation matches datasets.Dataset.train_test_split's RNG.
    Empty/singleton inputs keep all available rows in train.
    """
    rows = unique_rows(rows)
    n = len(rows)
    if n < 2:
        return [], rows
    size = min(1000, ceil(n * 0.1), n - 1)
    indices = np.random.default_rng(seed).permutation(n)
    return ([rows[i] for i in indices[:size]],
            [rows[i] for i in indices[size:]])


def training_dedup_enabled(near_dup_rate, testlike=False):
    # Token equality can collapse distinct source rows (normalization/truncation).
    # Keep their frequency in this mode; source-exact rows were already removed.
    return near_dup_rate > 0.2 and not testlike
