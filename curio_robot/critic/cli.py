"""curio-sim critic: make reaction episodes, train the critic, score it against
the gaze rule, and run the whole phase 1 split from docs/target4-critic.md.
Task T4-07."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from curio_robot.critic import episodes as eps_io
from curio_robot.critic.baseline import rule_scores
from curio_robot.critic.episodes import Episode
from curio_robot.critic.evaluate import GATE, Scored, auroc, evaluate, gate, threshold_for_precision
from curio_robot.critic.model import Critic
from curio_robot.critic.reactions import FAMILIES

TRAIN_FAMILIES = ("attentive", "slow", "distracted")  # silent is held out
BASELINE_THRESHOLD = 0.5  # the rule's p_wrong is 0 or 1, so any threshold in (0, 1] is the same
SEEDS = {"train": 0, "val": 1, "test": 2}
ROWS = ("precision", "recall", "false_alarm_rate", "lead_s", "target_acc", "auroc", "threshold")


def add_parser(sub: argparse._SubParsersAction) -> None:
    c = sub.add_parser("critic", help="reaction critic: episodes, training, scoring, phase 1")
    cs = c.add_subparsers(dest="critic_cmd", required=True)

    g = cs.add_parser("gen", help="write reaction episodes as JSON lines")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--families", default=",".join(FAMILIES), help="comma-separated")
    g.add_argument("--p-wrong", type=float, default=0.5)
    g.add_argument("--out", type=Path, required=True)

    t = cs.add_parser("train", help="train the critic on an episode file")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--seed", type=int, default=0)

    e = cs.add_parser("eval", help="score the critic or the gaze rule on an episode file")
    e.add_argument("--data", type=Path, required=True)
    who = e.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", type=Path)
    who.add_argument("--baseline", action="store_true")
    e.add_argument("--threshold", type=float, default=0.5)
    e.add_argument("--val", type=Path, default=None, help="pick the threshold for 90%% precision here")
    e.add_argument("--json", type=Path, default=None)

    p = cs.add_parser("phase1", help="train / val / test split, critic vs gaze rule, the gate")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out", type=Path, default=Path("out/critic"))


def run(args: argparse.Namespace) -> None:
    if args.critic_cmd == "gen":
        families = tuple(f for f in args.families.split(",") if f)
        episodes = eps_io.generate(args.n, args.seed, families, args.p_wrong)
        eps_io.save(episodes, args.out)
        print(f"{len(episodes)} episodes ({sum(e.wrong for e in episodes)} wrong) -> {args.out}")
    elif args.critic_cmd == "train":
        critic = Critic(seed=args.seed)
        losses = critic.fit(eps_io.load(args.data), args.epochs, args.lr, seed=args.seed)
        critic.save(args.out)
        print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} epochs -> {args.out}")
    elif args.critic_cmd == "eval":
        _eval(args)
    else:
        summary = phase1(args.n_train, args.n_val, args.n_test, args.epochs, args.out, lr=args.lr)
        print(format_table(summary))


def score_all(episodes: list[Episode], scorer: Callable[[Episode], Scored]) -> list[Scored]:
    return [scorer(ep) for ep in episodes]


def report(episodes: list[Episode], scored: list[Scored], threshold: float) -> dict:
    """Metrics overall and per family, each with AUROC (None if one class is missing)."""

    def one(idx: list[int]) -> dict:
        sub_e, sub_s = [episodes[i] for i in idx], [scored[i] for i in idx]
        m = evaluate(sub_e, sub_s, threshold)
        try:
            m["auroc"] = auroc(sub_e, sub_s)
        except ValueError:
            m["auroc"] = None
        return m

    families = sorted({e.family for e in episodes})
    return {
        "overall": one(list(range(len(episodes)))),
        "by_family": {f: one([i for i, e in enumerate(episodes) if e.family == f]) for f in families},
    }


def phase1(
    n_train: int, n_val: int, n_test: int, epochs: int, out: Path, lr: float = 0.01
) -> dict:
    """Train on seed 0 (attentive, slow, distracted), pick the threshold for 90%
    precision on seed 1 (same families), test the critic and the gaze rule on
    seed 2 with all four families. Writes out/model.npz and out/summary.json."""
    train = eps_io.generate(n_train, SEEDS["train"], TRAIN_FAMILIES)
    val = eps_io.generate(n_val, SEEDS["val"], TRAIN_FAMILIES)
    test = eps_io.generate(n_test, SEEDS["test"], FAMILIES)

    critic = Critic(seed=0)
    losses = critic.fit(train, epochs, lr, seed=0)
    threshold = threshold_for_precision(val, score_all(val, critic.predict))
    used = threshold if threshold is not None else BASELINE_THRESHOLD
    critic_report = report(test, score_all(test, critic.predict), used)
    baseline_report = report(test, score_all(test, rule_scores), BASELINE_THRESHOLD)

    summary = {
        "sizes": {"train": n_train, "val": n_val, "test": n_test},
        "epochs": epochs,
        "losses": [losses[0], losses[-1]] if losses else [],
        "threshold": threshold,
        "critic": critic_report,
        "baseline": baseline_report,
        "gate": gate(critic_report["overall"]),
        "gate_baseline": gate(baseline_report["overall"]),
    }
    out.mkdir(parents=True, exist_ok=True)
    critic.save(out / "model.npz")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _cell(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def format_table(summary: dict) -> str:
    """Markdown: one row per metric for the critic and the gaze rule, then per
    family, then the gate."""
    c, b = summary["critic"]["overall"], summary["baseline"]["overall"]
    lines = ["| metric | critic | gaze rule | gate |", "|---|---|---|---|"]
    for row in ROWS:
        bar = f">= {GATE[row]}" if row in GATE else ""
        lines.append(f"| {row} | {_cell(c.get(row))} | {_cell(b.get(row))} | {bar} |")
    lines += [
        "",
        "| family | critic recall | critic lead_s | critic target_acc | rule recall | rule lead_s |",
        "|---|---|---|---|---|---|",
    ]
    for fam, m in summary["critic"]["by_family"].items():
        r = summary["baseline"]["by_family"][fam]
        cells = (m["recall"], m["lead_s"], m["target_acc"], r["recall"], r["lead_s"])
        lines.append(f"| {fam} | " + " | ".join(_cell(v) for v in cells) + " |")
    passed = summary["gate"]
    verdict = "PASS" if all(passed.values()) else "FAIL"
    detail = ", ".join(f"{k} {'ok' if v else 'missed'}" for k, v in passed.items())
    lines += ["", f"Gate: {verdict} ({detail})"]
    return "\n".join(lines)


def _eval(args: argparse.Namespace) -> None:
    episodes = eps_io.load(args.data)
    if args.baseline:
        scorer: Callable[[Episode], Scored] = rule_scores
    else:
        scorer = Critic.load(args.model).predict
    threshold = args.threshold
    if args.val is not None:
        val = eps_io.load(args.val)
        picked = threshold_for_precision(val, score_all(val, scorer))
        threshold = picked if picked is not None else threshold
    result = report(episodes, score_all(episodes, scorer), threshold)
    result["gate"] = gate(result["overall"])
    text = json.dumps(result, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
