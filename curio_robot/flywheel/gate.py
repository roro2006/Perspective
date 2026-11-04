"""The promotion gates of tasks/T5-24 and tasks/T5-26 as pure functions over per-item
results, so the rules are tested without a served model. A result is {suite: [{"id", "kind", "hit", ...}]}; the
same ids must appear for both adapters (the paired intervals rely on it)."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

FB_MIN = 0.85  # front/behind held-out accuracy floor
FB_GAIN = 0.25  # and at least this much above the current adapter
OVERALL_DROP = 0.01  # bench5c overall may fall at most this much
KIND_DROP = 0.05  # no bench5c kind (nor deictic) may fall more than this
BOOTSTRAP = 10_000
Z95 = 1.959964
EPS = 1e-9  # float slack on the >= comparisons

Results = dict[str, list[dict]]


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def paired_bootstrap(old: list[bool], new: list[bool], n: int = BOOTSTRAP,
                     seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of mean(new) - mean(old), items resampled together."""
    a, b = np.asarray(old, float), np.asarray(new, float)
    if a.shape != b.shape:
        raise ValueError("paired samples differ in length")
    if a.size == 0:
        return 0.0, 0.0
    idx = np.random.default_rng(seed).integers(0, a.size, size=(n, a.size))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def paired(old: list[dict], new: list[dict]) -> tuple[list[bool], list[bool]]:
    """Hits of both adapters in the same id order. ValueError if the ids differ."""
    by_id = {r["id"]: r for r in new}
    if set(by_id) != {r["id"] for r in old}:
        raise ValueError("the two adapters were scored on different items")
    return [bool(r["hit"]) for r in old], [bool(by_id[r["id"]]["hit"]) for r in old]


def by_kind(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
    return dict(groups)


def compare(name: str, old: list[dict], new: list[dict]) -> dict:
    """One table row: n, both accuracies with Wilson intervals, the paired difference."""
    a, b = paired(old, new)
    n = len(a)
    ka, kb = sum(a), sum(b)
    return {"name": name, "n": n, "v2": ka / n if n else 0.0, "v3": kb / n if n else 0.0,
            "v2_ci": wilson(ka, n), "v3_ci": wilson(kb, n),
            "diff": (kb - ka) / n if n else 0.0, "diff_ci": paired_bootstrap(a, b)}


def decide(v2: Results, v3: Results) -> dict:
    """{"promote", "reasons", "rows"} by the card's three criteria."""
    fb = compare("front-behind (held-out)", v2["front-behind"], v3["front-behind"])
    overall = compare("bench5c overall", v2["bench5c"], v3["bench5c"])
    kinds_old, kinds_new = by_kind(v2["bench5c"]), by_kind(v3["bench5c"])
    kinds = [compare(k, kinds_old[k], kinds_new.get(k, [])) for k in kinds_old]
    kinds.append(compare("deictic (target-2 held-out)", v2["deictic"], v3["deictic"]))

    reasons, ok = [], True
    c1 = fb["v3"] >= FB_MIN - EPS and fb["v3"] >= fb["v2"] + FB_GAIN - EPS
    ok &= c1
    reasons.append(f"{'PASS' if c1 else 'FAIL'} 1: front/behind {fb['v3']:.3f} "
                   f"(needs >= {FB_MIN:.2f} and >= v2 {fb['v2']:.3f} + {FB_GAIN:.2f})")
    c2 = overall["v3"] >= overall["v2"] - OVERALL_DROP - EPS
    ok &= c2
    reasons.append(f"{'PASS' if c2 else 'FAIL'} 2: bench5c overall {overall['v3']:.3f} "
                   f"(needs >= v2 {overall['v2']:.3f} - {OVERALL_DROP:.2f})")
    drops = [k for k in kinds if k["v3"] < k["v2"] - KIND_DROP - EPS]
    c3 = not drops
    ok &= c3
    worst = min(kinds, key=lambda k: k["diff"])
    detail = ", ".join(f"{k['name']} {k['v2']:.2f}->{k['v3']:.2f}" for k in drops) or (
        f"largest drop {worst['name']} {worst['diff']:+.3f}")
    reasons.append(f"{'PASS' if c3 else 'FAIL'} 3: per-kind drops <= {KIND_DROP:.2f} ({detail})")
    return {"promote": bool(ok), "reasons": reasons, "rows": [fb, overall, *kinds]}


def format_table(rows: list[dict]) -> str:
    """Markdown table of compare() rows."""
    def ci(c):
        return f"[{c[0]:.2f}, {c[1]:.2f}]"

    lines = ["| set | n | v2 | v2 95% CI | v3 | v3 95% CI | v3 - v2 | paired 95% CI |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['n']} | {r['v2']:.3f} | {ci(r['v2_ci'])} | "
                     f"{r['v3']:.3f} | {ci(r['v3_ci'])} | {r['diff']:+.3f} | "
                     f"[{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}] |")
    return "\n".join(lines)


# ---------------------------------------------------------------- T5-26 (flywheel turn 2)
# T5-26's line 1 replaces T5-24's +0.25: the candidate must remove at least half of the
# current adapter's errors on a fresh front/behind set, and reach FB2_MIN. Lines 2 and 3 are
# T5-24's, with the fresh deictic set as one more kind. T5-24's decide() stays as it was.

FB2_MIN = 0.90  # fresh front/behind accuracy floor
FB2_ERROR_SHARE = 0.5  # share of the current adapter's errors the candidate must remove
DEICTIC_TRAIN_SEED = 31  # grounding_data seed of the deictic replay rows
DEICTIC_TEST_SEED = 41  # grounding_data seed of the fresh deictic test (d2)
FB2_SEEDS = range(625000, 626000)  # the fresh front/behind test (b2)
ADAPTERS = ("v2", "v3", "v4")


def removes_half_the_errors(old: float, new: float) -> bool:
    """T5-26 line 1: new >= FB2_MIN and new >= old + FB2_ERROR_SHARE * (1 - old)."""
    return new >= FB2_MIN - EPS and new >= old + FB2_ERROR_SHARE * (1 - old) - EPS


def compare3(name: str, v2: list[dict], v3: list[dict], v4: list[dict]) -> dict:
    """One row for three adapters: accuracies with Wilson intervals, and v4 - v2 and
    v4 - v3 with paired bootstrap intervals."""
    a, c = paired(v2, v4)
    b, c3 = paired(v3, v4)
    n = len(a)
    hits = {"v2": a, "v3": b, "v4": c}
    acc = {k: (sum(h) / n if n else 0.0) for k, h in hits.items()}
    return {"name": name, "n": n, "acc": acc,
            "ci": {k: wilson(sum(h), n) for k, h in hits.items()},
            "d_v2": acc["v4"] - acc["v2"], "d_v2_ci": paired_bootstrap(a, c),
            "d_v3": acc["v4"] - acc["v3"], "d_v3_ci": paired_bootstrap(b, c3)}


def decide_v4(v2: Results, v3: Results, v4: Results) -> dict:
    """{"promote", "reasons", "rows"} by T5-26's three criteria. v4 is judged against v2;
    v3 is reported alongside and decides nothing."""
    def row(name: str, suite: str) -> dict:
        return compare3(name, v2[suite], v3[suite], v4[suite])

    fb = row("front-behind fresh (b2)", "front-behind-2")
    overall = row("bench5c overall", "bench5c")
    k2, k3, k4 = by_kind(v2["bench5c"]), by_kind(v3["bench5c"]), by_kind(v4["bench5c"])
    kinds = [compare3(k, k2[k], k3.get(k, []), k4.get(k, [])) for k in k2]
    kinds.append(row("deictic d1 (target-2 held-out)", "deictic"))
    kinds.append(row("deictic d2 (fresh, seed 41)", "deictic-2"))

    old, new = fb["acc"]["v2"], fb["acc"]["v4"]
    need = max(FB2_MIN, old + FB2_ERROR_SHARE * (1 - old))
    c1 = removes_half_the_errors(old, new)
    reasons = [(f"{'PASS' if c1 else 'FAIL'} 1: fresh front/behind {new:.3f} (needs >= "
               f"{FB2_MIN:.2f} and >= v2 {old:.3f} + {FB2_ERROR_SHARE} * (1 - v2), "
               f"so >= {need:.3f})")]
    c2 = overall["acc"]["v4"] >= overall["acc"]["v2"] - OVERALL_DROP - EPS
    reasons.append(f"{'PASS' if c2 else 'FAIL'} 2: bench5c overall {overall['acc']['v4']:.3f} "
                   f"(needs >= v2 {overall['acc']['v2']:.3f} - {OVERALL_DROP:.2f})")
    drops = [k for k in kinds if k["acc"]["v4"] < k["acc"]["v2"] - KIND_DROP - EPS]
    c3 = not drops
    worst = min(kinds, key=lambda k: k["d_v2"])
    detail = ", ".join(f"{k['name']} {k['acc']['v2']:.2f}->{k['acc']['v4']:.2f}"
                       for k in drops) or f"largest drop {worst['name']} {worst['d_v2']:+.3f}"
    reasons.append(f"{'PASS' if c3 else 'FAIL'} 3: per-kind drops <= {KIND_DROP:.2f} ({detail})")
    return {"promote": bool(c1 and c2 and c3), "reasons": reasons,
            "rows": [fb, overall, *kinds]}


def format_table3(rows: list[dict]) -> str:
    """Markdown table of compare3() rows."""
    def ci(c):
        return f"[{c[0]:.2f}, {c[1]:.2f}]"

    def diff(r, k):
        lo, hi = r[k + "_ci"]
        return f"{r[k]:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    lines = [("| set | n | v2 (95% CI) | v3 (95% CI) | v4 (95% CI) | v4 - v2 (paired 95% CI) "
             "| v4 - v3 (paired 95% CI) |"), "|---|---|---|---|---|---|---|"]
    for r in rows:
        accs = " | ".join(f"{r['acc'][k]:.3f} {ci(r['ci'][k])}" for k in ADAPTERS)
        lines.append(f"| {r['name']} | {r['n']} | {accs} | {diff(r, 'd_v2')} | "
                     f"{diff(r, 'd_v3')} |")
    return "\n".join(lines)
