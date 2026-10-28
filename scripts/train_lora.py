"""LoRA fine-tune of a ~4B Qwen-VL on the grounding dataset (T2-07 output). Task T2-09.

    uv run python scripts/train_lora.py --data out/grounding/data.jsonl \\
        --model Qwen/Qwen3-VL-4B-Instruct --out out/lora/<name> [--dry-run]

The model sees exactly what serving sends (the user message rendered with the
generation prompt) and is trained on the single answer letter that follows, so
the loss sits on one token. torch, transformers and peft are imported inside
`train` only: `--dry-run` and the tests run without them (`uv sync --extra train`
installs them on the training machine)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from curio_robot.grounding.prompt import PROMPT_VERSION
from curio_robot.sim.grounding_data import load_dataset

IGNORE_INDEX = -100  # label value the HF loss skips
EVAL_FRACTION = 0.1
EVAL_ID_BUCKETS = 1000  # id hash buckets; the first 10% are held out
# Attention projections of the language model. Qwen-VL vision blocks name theirs
# "qkv"/"proj", so these names leave the vision tower frozen.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyperparameters, defaults sized for a 20 GB card (RX 7900 XT)."""

    epochs: int = 3
    lr: float = 2e-4
    batch_size: int = 1  # one image per step; no padding logic needed
    grad_accum: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = LORA_TARGETS
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 0


# ---------------------------------------------------------------- pure helpers


def is_eval_id(example_id: str, fraction: float = EVAL_FRACTION) -> bool:
    """Stable by-id split: the same id lands on the same side for any dataset
    size, order or run, so eval_ids.json from one run stays valid for the next."""
    digest = hashlib.sha256(example_id.encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % EVAL_ID_BUCKETS
    return bucket < round(fraction * EVAL_ID_BUCKETS)


def split_rows(rows: list[dict], fraction: float = EVAL_FRACTION) -> tuple[list[dict], list[dict]]:
    """(train, eval) rows, order kept."""
    train = [r for r in rows if not is_eval_id(r["id"], fraction)]
    held = [r for r in rows if is_eval_id(r["id"], fraction)]
    return train, held


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_images(rows: list[dict], root: Path) -> list[str]:
    """Image paths (relative to the data.jsonl folder) that don't exist."""
    return [r["image"] for r in rows if not (root / r["image"]).is_file()]


def summarize(rows: list[dict]) -> dict:
    """Counts the dry run prints: by kind, by answer letter, train/eval split."""
    train, held = split_rows(rows)
    return {
        "n": len(rows),
        "kinds": dict(sorted(Counter(r["kind"] for r in rows).items())),
        "answers": dict(sorted(Counter(r["answer"] for r in rows).items())),
        "train": len(train),
        "eval": len(held),
    }


def to_chat(row: dict) -> tuple[list[dict], str]:
    """The JSONL messages (OpenAI format, image as a path) as an HF chat with an
    image placeholder for the processor, plus the answer letter. Only the user
    turn goes in: the answer is appended after the generation prompt, exactly
    as serving decodes it."""
    user = next(m for m in row["messages"] if m["role"] == "user")
    content: list[dict] = []
    for part in user["content"]:
        if part["type"] == "image_url":
            content.append({"type": "image"})
        elif part["type"] == "text":
            content.append({"type": "text", "text": part["text"]})
        else:
            raise ValueError(f"{row['id']}: unexpected content part {part['type']!r}")
    answers = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
    answer = answers[-1] if answers else row["answer"]
    if answer != row["answer"]:
        raise ValueError(f"{row['id']}: assistant turn {answer!r} != answer {row['answer']!r}")
    return [{"role": "user", "content": content}], answer


def answer_labels(prompt_len: int, answer_ids: list[int]) -> list[int]:
    """Labels for input_ids = prompt + answer: IGNORE_INDEX over the prompt, the
    ids over the answer. The answer must be one token, the one serving reads."""
    if len(answer_ids) != 1:
        raise ValueError(f"answer tokenizes to {len(answer_ids)} tokens, expected 1")
    return [IGNORE_INDEX] * prompt_len + list(answer_ids)


def build_config(
    model: str, data: Path, cfg: TrainConfig, n_train: int, n_eval: int
) -> dict:
    """train_config.json: enough for evaluation to attribute a result."""
    return {
        "model": model,
        "data": str(data),
        "data_sha256": file_sha256(data),
        "prompt_version": PROMPT_VERSION,
        "n_train": n_train,
        "n_eval": n_eval,
        "eval_fraction": EVAL_FRACTION,
        "hyperparameters": asdict(cfg),
    }


def check_rows(rows: list[dict], root: Path) -> None:
    """Fail early with a clear message instead of mid-epoch."""
    if not rows:
        raise SystemExit("error: dataset is empty")
    missing = missing_images(rows, root)
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise SystemExit(f"error: {len(missing)} missing image(s) under {root}: {shown}")
    stale = sorted({r["prompt_version"] for r in rows} - {PROMPT_VERSION})
    if stale:
        raise SystemExit(
            f"error: dataset prompt_version {stale} != PROMPT_VERSION {PROMPT_VERSION!r};"
            " regenerate the data"
        )


# ---------------------------------------------------------------- training


def _encode(processor, row: dict, root: Path, device) -> dict:
    """One example as model inputs: prompt via the chat template with the
    generation prompt, then the answer token, labels on that token only."""
    import torch
    from PIL import Image

    chat, answer = to_chat(row)
    prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    image = Image.open(root / row["image"]).convert("RGB")
    enc = processor(text=[prompt], images=[image], return_tensors="pt")
    answer_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
    prompt_len = enc["input_ids"].shape[1]
    labels = answer_labels(prompt_len, answer_ids)
    ans = torch.tensor([answer_ids], dtype=enc["input_ids"].dtype)
    enc["input_ids"] = torch.cat([enc["input_ids"], ans], dim=1)
    enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(ans)], dim=1)
    # Other per-token tensors (Qwen3-VL's mm_token_type_ids: 0 text, 1 image) must
    # grow with input_ids, or rope indexing fails; the answer is a text token.
    for key, value in enc.items():
        if key not in ("input_ids", "attention_mask") and value.ndim == 2 and (
            value.shape[1] == prompt_len
        ):
            enc[key] = torch.cat([value, torch.zeros_like(ans, dtype=value.dtype)], dim=1)
    enc["labels"] = torch.tensor([labels], dtype=enc["input_ids"].dtype)
    return {k: v.to(device) for k, v in enc.items()}


def _letter_ids(processor, letters: list[str]) -> dict[str, int]:
    ids = {}
    for letter in letters:
        tok = processor.tokenizer.encode(letter, add_special_tokens=False)
        answer_labels(0, tok)  # raises if a letter is not a single token
        ids[letter] = tok[0]
    return ids


def _evaluate(model, processor, rows: list[dict], root: Path) -> dict:
    """Mean answer-token loss and accuracy of the argmax over the example's own
    letters (what serving reads from top_logprobs)."""
    import torch

    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for row in rows:
            batch = _encode(processor, row, root, model.device)
            out = model(**batch)
            total_loss += float(out.loss)
            ids = _letter_ids(processor, sorted(row["letters"]))
            logits = out.logits[0, -2]  # position predicting the answer token
            pred = max(ids, key=lambda letter: float(logits[ids[letter]]))
            correct += pred == row["answer"]
    model.train()
    n = max(len(rows), 1)
    return {"eval_loss": total_loss / n, "eval_acc": correct / n, "n_eval": len(rows)}


def train(args: argparse.Namespace, cfg: TrainConfig, rows: list[dict], root: Path) -> None:
    """The real run (GPU machine). Imports the train extra here, never at top level."""
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        get_linear_schedule_with_warmup,
    )

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = split_rows(rows)
    (out_dir / "eval_ids.json").write_text(json.dumps([r["id"] for r in eval_rows], indent=1))
    config = build_config(args.model, Path(args.data), cfg, len(train_rows), len(eval_rows))
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2))

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm builds report as cuda
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets), bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, cfg.epochs * len(train_rows) // cfg.grad_accum)
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_ratio * steps), steps)

    log = (out_dir / "train_log.jsonl").open("w")
    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        order = train_rows[:]
        random.shuffle(order)
        running = 0.0
        for i, row in enumerate(order, 1):
            loss = model(**_encode(processor, row, root, device)).loss / cfg.grad_accum
            loss.backward()
            running += float(loss)
            if i % cfg.grad_accum == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                log.write(json.dumps({"epoch": epoch, "step": step, "loss": running}) + "\n")
                log.flush()
                running = 0.0
        metrics = {"epoch": epoch, "step": step, **_evaluate(model, processor, eval_rows, root)}
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        print(json.dumps(metrics))
    log.close()
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"saved adapter to {out_dir}")


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, TrainConfig]:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="LoRA fine-tune a Qwen-VL grounder")
    p.add_argument("--data", required=True, help="data.jsonl from curio_robot.sim.grounding_data")
    p.add_argument("--model", required=True, help="HF id or local path of the base Qwen-VL")
    p.add_argument("--out", required=True, help="output dir for the adapter and configs")
    p.add_argument("--dry-run", action="store_true", help="check the data and exit; no torch")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--lora-r", type=int, default=d.lora_r)
    p.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    args = p.parse_args(argv)
    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, grad_accum=args.grad_accum, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    return args, cfg


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_args(argv)
    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"error: no dataset at {data}")
    rows = load_dataset(data)
    root = data.parent
    check_rows(rows, root)
    s = summarize(rows)
    print(f"examples: {s['n']} (train {s['train']}, eval {s['eval']})")
    print("kinds: " + ", ".join(f"{k}={v}" for k, v in s["kinds"].items()))
    print("answers: " + ", ".join(f"{k}={v}" for k, v in s["answers"].items()))
    if s["eval"] == 0:
        print("warning: no eval examples at this size", file=sys.stderr)
    if args.dry_run:
        print("dry run: data ok, not training")
        return 0
    train(args, cfg, rows, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
