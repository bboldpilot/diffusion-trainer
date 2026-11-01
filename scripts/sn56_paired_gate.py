"""The validator's paired boss-round gate, VERBATIM (K5 support; the training box has no upstream checkout).

Copied byte-for-byte from G.O.D origin/main bfa699ad validator/tournament/thresholds.py:
  compare_paired_losses          :16-149
  challenger_takes_paired_task   :203-228
  _bootstrap_lower_bounds        :262-298
Only the surroundings are substituted: `t_cst` carries the constants of
validator/tournament/constants.py:237-287 (values below), and PairedLossComparison
is a plain attribute holder with the same fields as validator/tournament/models.py.
scripts/tests/test_sn56_dpo_select.py re-extracts the three functions from the
upstream checkout and asserts this copy is identical, and asserts the constants.
"""

from types import SimpleNamespace

import numpy as np

t_cst = SimpleNamespace(
    BOSS_ROUND_WIN_MARGIN=0.01,
    BOSS_ROUND_TIE_DEADZONE_NATS=0.01,
    BOSS_ROUND_MIN_WIN_RATE=0.55,
    BOSS_ROUND_MIN_MEAN_GAP_NATS=0.01,
    BOSS_ROUND_BOOTSTRAP_CONFIDENCE=0.99,
    BOSS_ROUND_BOOTSTRAP_RESAMPLES=10_000,
    BOSS_ROUND_BOOTSTRAP_SEED=20260808,
)


class PairedLossComparison(SimpleNamespace):
    """Stand-in for validator.tournament.models.PairedLossComparison (same fields)."""

    def model_dump(self):
        return dict(self.__dict__)


# ---- verbatim from validator/tournament/thresholds.py @ bfa699ad (do not edit) ----


def compare_paired_losses(
    boss_losses: list[float],
    challenger_losses: list[float],
    deadzone_nats: float = t_cst.BOSS_ROUND_TIE_DEADZONE_NATS,
    min_win_rate: float = t_cst.BOSS_ROUND_MIN_WIN_RATE,
    min_mean_gap_nats: float = t_cst.BOSS_ROUND_MIN_MEAN_GAP_NATS,
    confidence: float = t_cst.BOSS_ROUND_BOOTSTRAP_CONFIDENCE,
    resamples: int = t_cst.BOSS_ROUND_BOOTSTRAP_RESAMPLES,
    seed: int = t_cst.BOSS_ROUND_BOOTSTRAP_SEED,
) -> PairedLossComparison:
    """Decide a boss-round task by comparing losses example by example on the same held-out set.

    Lower-is-better log-likelihood losses only (instruct, DPO). Both models must have been scored
    on the identical examples in the identical order — index i is the same example for both.

    The challenger takes the task only if it wins a clear majority of decided examples AND is
    better on average, with both statistics clearing their bar at the one-sided bootstrap bound
    rather than on the point estimate. Ties and tasks with too few decided examples are not wins.
    """
    if len(boss_losses) != len(challenger_losses):
        raise ValueError(
            f"Paired comparison needs equal-length loss vectors, got boss={len(boss_losses)} "
            f"challenger={len(challenger_losses)}"
        )

    boss = np.asarray(boss_losses, dtype=np.float64)
    challenger = np.asarray(challenger_losses, dtype=np.float64)

    # An example is only usable if both sides scored it; a non-finite loss on either side means
    # that example carries no comparison and pairing requires dropping it from both.
    usable = np.isfinite(boss) & np.isfinite(challenger)
    boss, challenger = boss[usable], challenger[usable]
    n_examples = int(boss.size)

    if n_examples == 0:
        return PairedLossComparison(
            n_examples=0,
            n_decided=0,
            challenger_example_wins=0,
            boss_example_wins=0,
            win_rate=0.0,
            win_rate_lower_bound=0.0,
            mean_gap_nats=0.0,
            mean_gap_lower_bound=0.0,
            challenger_won=False,
            is_draw=False,
            reason="No comparable examples between the two vectors",
            required_win_rate=min_win_rate,
            required_mean_gap_nats=min_mean_gap_nats,
            deadzone_nats=deadzone_nats,
        )

    # Positive gap = challenger assigned more probability to that example than the boss did.
    gaps = boss - challenger
    challenger_wins = gaps > deadzone_nats
    boss_wins = gaps < -deadzone_nats
    decided = challenger_wins | boss_wins

    n_decided = int(decided.sum())
    n_challenger_wins = int(challenger_wins.sum())
    win_rate = n_challenger_wins / n_decided if n_decided else 0.0
    # Mean gap is over every comparable example, ties included: it asks whether the challenger is
    # better overall, and near-identical examples legitimately dilute that. The win rate above uses
    # only decided examples, where the two models actually differ.
    mean_gap = float(gaps.mean())

    # Never a weaker requirement than the relative margin it replaces. A flat floor is looser than
    # 1% of the loss once the loss exceeds 2.0, which would have let a uniformly-but-slightly
    # better challenger take a high-loss task it would have lost under the old rule.
    required_mean_gap = max(min_mean_gap_nats, t_cst.BOSS_ROUND_WIN_MARGIN * abs(float(boss.mean())))

    win_rate_lb, mean_gap_lb = _bootstrap_lower_bounds(
        gaps=gaps,
        challenger_wins=challenger_wins,
        decided=decided,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )

    # A draw needs as much evidence as a difference does. "Everything landed in the dead zone" only
    # says the models are equivalent if there were enough examples that a real gap would have shown
    # up - on a handful of examples a genuinely better model lands inside the dead zone by luck.
    # Below the floor this is the same "cannot tell" case as too few decided, not a finding.
    # No minimum on the decided count. The mean gap is taken over every example, so a handful of
    # decided ones cannot move it: 20 of 800 improved by 0.5 nats averages 0.014 and fails, while
    # 20 improved by 1.5 averages 0.039 and passes. That makes the gate a requirement on total
    # improvement rather than on a count, and it already rejects the saturated case a decided-count
    # floor was meant to catch - tiny decided set AND tiny gaps means a tiny mean gap. A floor on
    # top only threw out tasks where the challenger did move the average decisively.
    is_draw = n_decided == 0
    if is_draw:
        reason = (
            f"Draw - all {n_examples} comparable examples fell inside the {deadzone_nats} nat dead "
            f"zone, so the two models are indistinguishable on this task"
        )
        challenger_won = False
    elif win_rate_lb < min_win_rate:
        reason = (
            f"Challenger win rate {win_rate:.1%} (bound {win_rate_lb:.1%}) below required "
            f"{min_win_rate:.1%} of {n_decided} decided examples"
        )
        challenger_won = False
    elif mean_gap_lb < required_mean_gap:
        reason = (
            f"Challenger mean gap {mean_gap:.4f} nats (bound {mean_gap_lb:.4f}) below required "
            f"{required_mean_gap:.4f} nats"
        )
        challenger_won = False
    else:
        reason = (
            f"Challenger won {win_rate:.1%} of {n_decided} decided examples (bound "
            f"{win_rate_lb:.1%}) by {mean_gap:.4f} nats (bound {mean_gap_lb:.4f})"
        )
        challenger_won = True

    return PairedLossComparison(
        n_examples=n_examples,
        n_decided=n_decided,
        challenger_example_wins=n_challenger_wins,
        boss_example_wins=int(boss_wins.sum()),
        win_rate=win_rate,
        win_rate_lower_bound=win_rate_lb,
        mean_gap_nats=mean_gap,
        mean_gap_lower_bound=mean_gap_lb,
        challenger_won=challenger_won,
        is_draw=is_draw,
        reason=reason,
        required_win_rate=min_win_rate,
        # The bar actually applied, not the floor it was derived from - on a high-loss task the
        # scaled requirement is the one the challenger had to clear.
        required_mean_gap_nats=required_mean_gap,
        deadzone_nats=deadzone_nats,
    )


def challenger_takes_paired_task(
    comparison: PairedLossComparison,
    boss_scalar_loss: float,
    challenger_scalar_loss: float,
) -> tuple[bool, str]:
    """Final verdict for a paired boss-round task. Returns (challenger_won, reason).

    Two conditions. The paired comparison runs on RAW per-example losses; the ranking scalar can
    additionally carry a per-model KL penalty, which is a constant and so cannot be folded into the
    vector without fabricating a 100% win rate. So the challenger must win on the examples AND not
    be worse once its penalty is counted. For non-KL tasks the scalar is the vector mean and the
    second condition is already implied by the first.

    Shared by crowning and by the emission projection: if these two disagreed, analytics would
    report a challenger win on a task the tournament recorded as a loss.
    """
    if comparison.is_draw:
        return False, comparison.reason
    if not comparison.challenger_won:
        return False, comparison.reason
    if challenger_scalar_loss > boss_scalar_loss:
        return False, (
            f"won the paired comparison but is worse on the ranking scalar "
            f"({challenger_scalar_loss:.6f} vs {boss_scalar_loss:.6f})"
        )
    return True, comparison.reason


def _bootstrap_lower_bounds(
    gaps: np.ndarray,
    challenger_wins: np.ndarray,
    decided: np.ndarray,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """One-sided lower bounds on the win rate and mean gap, resampling examples with replacement.

    Resamples the full example set rather than only the decided ones: which examples land in the
    dead zone is itself a property of the sample, so holding it fixed would understate the spread.
    Seeded, so two validators scoring the same boss round reach the same verdict.
    """
    n = gaps.size
    rng = np.random.default_rng(seed)
    win_arr = challenger_wins.astype(np.float64)
    decided_arr = decided.astype(np.float64)

    win_rates = np.empty(resamples, dtype=np.float64)
    mean_gaps = np.empty(resamples, dtype=np.float64)

    # Chunked so the index matrix stays bounded regardless of eval-set size.
    chunk = max(1, min(resamples, 2_000_000 // max(n, 1)))
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        decided_counts = decided_arr[idx].sum(axis=1)
        win_counts = win_arr[idx].sum(axis=1)
        # A resample that decided nothing is not evidence for the challenger.
        win_rates[start : start + size] = np.divide(
            win_counts, decided_counts, out=np.zeros(size), where=decided_counts > 0
        )
        mean_gaps[start : start + size] = gaps[idx].mean(axis=1)

    percentile = (1.0 - confidence) * 100.0
    return float(np.percentile(win_rates, percentile)), float(np.percentile(mean_gaps, percentile))
