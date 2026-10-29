"""Fine-tune SmolVLA (lerobot/smolvla_base) on Curio pick demonstrations. Task T5-07.

Runs in the lerobot venv on the RX 7900 XT, offline:

    set HF_HUB_OFFLINE=1, HF_HUB_DISABLE_SYMLINKS_WARNING=1, WANDB_MODE=disabled
    C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe scripts/smolvla_train.py \\
        --demos out/demos300 --out out/smolvla/run1 --steps 20000 --batch-size 32

The loop is lerobot-train's (preprocessor -> policy.forward -> AdamW with the policy's preset,
grad clip, cosine schedule with warmup) written out here because lerobot-train needs
lerobot[dataset], which the venv lacks (see smolvla_dataset.py). The model is the base
checkpoint loaded from the local snapshot with load_vlm_weights=False, so nothing is fetched.

Cameras: the base was trained on camera1..3; ours are camera1 (robot view with the target ring)
and camera2 (person's view). input_features lists only those two, so prepare_images feeds two
images; with empty_cameras=0 no padded third image is added (a masked one would not be
attended to anyway). The VLM stays frozen (train_expert_only, the base config).

Outputs under --out: config.json (arguments, split, marker), train_log.jsonl (one line per
log step: loss, lr, step time, GPU memory; eval lines with the held-out loss), and
checkpoints/<step>/ with pretrained_model/ (policy + processors) and training_state.pt."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch

# smolvla_dataset puts the repo root on sys.path, so it comes before curio_robot.
from smolvla_dataset import CAMERA_KEYS, STATE, CurioPickFrames, marker_settings, split

from curio_robot.sim import demos

SNAPSHOT_GLOB = "C:/Users/Rohan/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/*"
EVAL_SAMPLES = 256  # held-out frames scored at each eval (a fixed, evenly spaced subset)
EVAL_SEED = 0  # same flow-matching noise and times at every eval, so the numbers compare


def _require_offline() -> None:
    for key, value in (("HF_HUB_OFFLINE", "1"), ("WANDB_MODE", "disabled")):
        if os.environ.get(key) != value:
            raise SystemExit(f"set {key}={value} (no downloads, no wandb)")


def build_policy(snapshot: str, image_hw: tuple[int, int], device: str, ckpt: Path | None):
    """SmolVLA from the base snapshot (or a checkpoint of this run) with our two cameras."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    source = str(ckpt) if ckpt else snapshot
    cfg = PreTrainedConfig.from_pretrained(source)
    cfg.load_vlm_weights = False  # the checkpoint holds the backbone; don't fetch SmolVLM2
    cfg.device = device
    cfg.push_to_hub = False
    cfg.empty_cameras = 0
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_hw))
    cfg.input_features = {
        STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        CAMERA_KEYS["robot"]: image,
        CAMERA_KEYS["human"]: image,
    }
    return SmolVLAPolicy.from_pretrained(source, config=cfg)


def _loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=True, drop_last=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def _cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(policy, preprocessor, loader) -> float:
    policy.eval()
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(EVAL_SEED)
    total = n = 0
    for batch in loader:
        loss, _ = policy.forward(preprocessor(batch))
        total += loss.item() * len(batch["task"])
        n += len(batch["task"])
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state(rng[1])
    policy.train()
    return total / n


def save_checkpoint(out: Path, step: int, policy, preprocessor, postprocessor, optimizer,
                    scheduler) -> Path:
    path = out / "checkpoints" / f"{step:06d}"
    model_dir = path / "pretrained_model"
    policy.save_pretrained(model_dir)
    preprocessor.save_pretrained(model_dir)
    postprocessor.save_pretrained(model_dir)
    torch.save({"step": step, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}, path / "training_state.pt")
    (out / "checkpoints" / "last.txt").write_text(path.name, encoding="utf-8")
    return path


def _log(f, record: dict) -> None:
    line = json.dumps(record)
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


def train(args: argparse.Namespace) -> None:
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    _require_offline()
    torch.manual_seed(args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    info = demos.load_info(args.demos)
    train_eps, eval_eps = split(info["total_episodes"])
    image_hw = (info["image_size"]["height"], info["image_size"]["width"])

    ckpt = None
    if args.resume:
        last = (out / "checkpoints" / "last.txt").read_text(encoding="utf-8").strip()
        ckpt = out / "checkpoints" / last
    snapshot = max(glob.glob(SNAPSHOT_GLOB))
    policy = build_policy(snapshot, image_hw, "cuda", ckpt and ckpt / "pretrained_model")
    policy.train()
    cfg = policy.config

    train_set = CurioPickFrames(args.demos, train_eps, cfg.chunk_size, args.marker)
    eval_set = CurioPickFrames(args.demos, eval_eps, cfg.chunk_size, args.marker)
    eval_subset = torch.utils.data.Subset(
        eval_set, list(range(0, len(eval_set), max(1, len(eval_set) // EVAL_SAMPLES))))
    preprocessor, postprocessor = make_smolvla_pre_post_processors(cfg, train_set.stats())

    if args.lr:
        cfg.optimizer_lr = args.lr
    optim_cfg = cfg.get_optimizer_preset()
    optimizer = optim_cfg.build(policy.get_optim_params())
    scheduler = cfg.get_scheduler_preset().build(optimizer, args.steps)
    start = 0
    if ckpt:
        state = torch.load(ckpt / "training_state.pt", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["step"]

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    (out / "config.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "snapshot": snapshot,
        "train_episodes": [train_eps[0], train_eps[-1]],
        "eval_episodes": [eval_eps[0], eval_eps[-1]],
        "train_frames": len(train_set),
        "eval_frames": len(eval_set),
        "marker": marker_settings(args.marker),
        "trainable_params": trainable,
        "lr": cfg.optimizer_lr,
        "chunk_size": cfg.chunk_size,
    }, indent=1), encoding="utf-8")

    batches = _cycle(_loader(train_set, args.batch_size, args.workers, True, args.seed))
    eval_loader = _loader(eval_subset, args.batch_size, 0, False, args.seed)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    with (out / "train_log.jsonl").open("a", encoding="utf-8") as log:
        _log(log, {"event": "start", "step": start, "steps": args.steps,
                   "train_frames": len(train_set), "trainable_params": trainable})
        t_last, data_time = time.perf_counter(), 0.0
        for step in range(start + 1, args.steps + 1):
            t0 = time.perf_counter()
            batch = next(batches)
            data_time += time.perf_counter() - t0
            loss, _ = policy.forward(preprocessor(batch))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), optim_cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_freq == 0 or step == start + 1:
                now = time.perf_counter()
                n = args.log_freq if step != start + 1 else 1
                _log(log, {
                    "step": step, "loss": round(loss.item(), 5),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": scheduler.get_last_lr()[0],
                    "step_s": round((now - t_last) / n, 4),
                    "data_s": round(data_time / n, 4),
                    "gpu_max_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "gpu_max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                    "elapsed_min": round((now - t_start) / 60, 2),
                })
                t_last, data_time = time.perf_counter(), 0.0
            if args.eval_freq and step % args.eval_freq == 0:
                _log(log, {"step": step, "eval_loss": round(
                    evaluate(policy, preprocessor, eval_loader), 5)})
                t_last = time.perf_counter()
            if step % args.save_freq == 0 or step == args.steps:
                path = save_checkpoint(out, step, policy, preprocessor, postprocessor,
                                       optimizer, scheduler)
                _log(log, {"step": step, "checkpoint": str(path)})
                t_last = time.perf_counter()
        _log(log, {"event": "done", "step": args.steps,
                   "elapsed_min": round((time.perf_counter() - t_start) / 60, 2)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None, help="default: the policy preset")
    parser.add_argument("--marker", choices=demos.MARKER_MODES, default="start")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=0, help="0: never")
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="from checkpoints/last.txt")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
