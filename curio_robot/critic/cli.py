"""curio-sim critic: make reaction episodes, train the critic, score it against
the gaze rule, and run the whole phase 1 split from docs/target4-critic.md.
Phase 2a (critic labels as a training signal) is `critic phase2`.
Tasks T4-07, T4-09, T4-10."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from curio_robot.critic import episodes as eps_io
from curio_robot.critic import learn
from curio_robot.critic.baseline import rule_scores
from curio_robot.critic.episodes import Episode
from curio_robot.critic.evaluate import GATE, Scored, auroc, evaluate, gate, threshold_for_precision
from curio_robot.critic.model import Critic
from curio_robot.critic.reactions import FAMILIES

TRAIN_FAMILIES = ("attentive", "slow", "distracted")  # silent is held out
BASELINE_THRESHOLD = 0.5  # the rule's p_wrong is 0 or 1, so any threshold in (0, 1] is the same
SEEDS = {"train": 0, "val": 1, "test": 2}
ROWS = ("precision", "recall", "false_alarm_rate", "lead_s", "target_acc", "auroc", "threshold")
# A second operating point for a cheap action (slow the arm and ask), which can
# afford more false alarms than stopping can. Not part of the gate.
SLOW_PRECISION = 0.75
# Phase 2a gate, fixed before the first run (docs/target4-critic.md, section 7).
PHASE2_GATE = {"relative_cut": 0.25, "type_slack": 0.02, "label_acc": 0.90}


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

    q = cs.add_parser("phase2", help="critic labels vs spoken corrections as a training signal")
    q.add_argument("--people", type=int, default=25, help="per person type and reaction family")
    q.add_argument("--interactions", type=int, default=15)
    q.add_argument("--seed", type=int, default=3)
    q.add_argument("--model", type=Path, default=Path("out/critic/model.npz"))
    q.add_argument("--phase1", type=Path, default=Path("out/critic/summary.json"),
                   help="phase 1 summary, for the stop threshold")
    q.add_argument("--out", type=Path, default=Path("out/critic/phase2"))


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
    elif args.critic_cmd == "phase2":
        threshold = json.loads(args.phase1.read_text(encoding="utf-8")).get("threshold")
        summary = phase2(args.people, args.interactions, args.seed, Critic.load(args.model),
                         threshold if threshold is not None else BASELINE_THRESHOLD, args.out)
        print(format_phase2(summary))
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
    val_scored, test_scored = score_all(val, critic.predict), score_all(test, critic.predict)
    threshold = threshold_for_precision(val, val_scored)
    used = threshold if threshold is not None else BASELINE_THRESHOLD
    critic_report = report(test, test_scored, used)
    slow_threshold = threshold_for_precision(val, val_scored, SLOW_PRECISION)
    slow_used = slow_threshold if slow_threshold is not None else BASELINE_THRESHOLD
    slow_report = report(test, test_scored, slow_used)
    baseline_report = report(test, score_all(test, rule_scores), BASELINE_THRESHOLD)

    summary = {
        "sizes": {"train": n_train, "val": n_val, "test": n_test},
        "epochs": epochs,
        "losses": [losses[0], losses[-1]] if losses else [],
        "threshold": threshold,
        "threshold_slow": slow_threshold,
        "critic": critic_report,
        "critic_slow": slow_report,
        "baseline": baseline_report,
        "gate": gate(critic_report["overall"]),
        "gate_baseline": gate(baseline_report["overall"]),
    }
    out.mkdir(parents=True, exist_ok=True)
    critic.save(out / "model.npz")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def phase2(
    people: int, n: int, seed: int, critic: Critic, threshold: float, out: Path
) -> dict:
    """Every feedback source on the same simulated people; writes out/summary.json."""
    steps = learn.run(people, n, seed, critic, threshold)
    sources = learn.summarize(steps)
    summary = {"people_per_cell": people, "interactions": n, "seed": seed,
               "threshold": threshold, "sources": sources, "gate": phase2_gate(sources)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def phase2_gate(sources: dict) -> dict[str, bool]:
    """critic+spoken against spoken alone, on the last interactions per person."""
    spoken, both = sources["spoken"], sources["critic+spoken"]
    cut = 1.0 - PHASE2_GATE["relative_cut"]
    return {
        "fewer_errors": both["overall"]["wrong_last"] <= cut * spoken["overall"]["wrong_last"],
        "no_type_worse": all(
            both["by_type"][t]["wrong_last"]
            <= spoken["by_type"][t]["wrong_last"] + PHASE2_GATE["type_slack"]
            for t in spoken["by_type"]
        ),
        "labels_accurate": (both["overall"]["label_acc"] or 0.0) >= PHASE2_GATE["label_acc"],
    }


def format_phase2(summary: dict) -> str:
    """Markdown: one row per source (overall), then wrong-reach rate over the last
    interactions by person type and by reaction family, then the gate."""
    src = summary["sources"]
    lines = ["| source | wrong | wrong, last 5 | labels per reach | labels correct |",
             "|---|---|---|---|---|"]
    for name, s in src.items():
        o = s["overall"]
        cells = (o["wrong"], o["wrong_last"], o["coverage"], o["label_acc"])
        lines.append(f"| {name} | " + " | ".join(_cell(v) for v in cells) + " |")
    for split in ("by_type", "by_family"):
        keys = list(next(iter(src.values()))[split])
        lines += ["", f"| wrong, last 5 ({split[3:]}) | " + " | ".join(keys) + " |",
                  "|---" * (len(keys) + 1) + "|"]
        for name, s in src.items():
            lines.append(f"| {name} | "
                         + " | ".join(_cell(s[split][k]["wrong_last"]) for k in keys) + " |")
    passed = summary["gate"]
    verdict = "PASS" if all(passed.values()) else "FAIL"
    detail = ", ".join(f"{k} {'ok' if v else 'missed'}" for k, v in passed.items())
    lines += ["", f"Gate: {verdict} ({detail})"]
    return "\n".join(lines)


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
    s = summary["critic_slow"]["overall"]
    lines = ["| metric | critic | critic, slow-and-ask | gaze rule | gate |",
             "|---|---|---|---|---|"]
    for row in ROWS:
        bar = f">= {GATE[row]}" if row in GATE else ""
        cells = (c.get(row), s.get(row), b.get(row))
        lines.append(f"| {row} | " + " | ".join(_cell(v) for v in cells) + f" | {bar} |")
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
