"""Fine-tune pi0 (lerobot/pi0_base, Physical Intelligence) on the relabelled run-5 demos.
Task T5-28.

Runs in the lerobot venv on the RX 7900 XT, offline (HF_HUB_OFFLINE=1, WANDB_MODE=disabled):

    C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe scripts/pi0_train.py \\
        --demos C:/Users/Rohan/dev/Perspective/out/demos800b \\
        --relabel C:/Users/Rohan/dev/Perspective/out/t5_28/data/relabel.jsonl \\
        --out C:/Users/Rohan/dev/Perspective/out/t5_28/pi0 --steps N --max-hours 8

The loop is smolvla_train's (preprocessor -> expert_loss -> AdamW with the policy's preset,
grad clip, the preset's cosine schedule with warmup, scaled to --steps), because lerobot-train
needs lerobot[dataset]. Inputs (pi0_data): the robot view as base_0_rgb, the wrist view as
left_wrist_0_rgb, the 6 joints, the robot-words task text. pi0 resizes 256 px to 224 px itself.

Precision: train_expert_only=True, so PaliGemma (SigLIP + Gemma 2B) is frozen and only the
Gemma-300M action expert and the state/action/time projections train. The frozen VLM is held
in bf16 (dtype="bfloat16", lerobot's own setting; the vision path and norms stay fp32 as lerobot
keeps them). With --expert-dtype float32 (default) the trainable expert keeps fp32 master
weights and the forward runs under bf16 autocast: at the preset's peak lr of 2.5e-5 an AdamW
step is about 1e-3 of a typical weight, under bf16's half-ULP (2^-9), so pure-bf16 weights would
drop most updates. Checkpoints keep the fp32 expert; pi0_eval loads them in bf16.

The chunk is run 5's: 30 actions predicted, 10 run per model call (the base's 50 is longer than
a 42-frame demo; the expert places action tokens by position, so any length works).

Outputs under --out: config.json, train_log.jsonl (loss, lr, step time, GPU memory; eval lines
with the held-out loss), checkpoints/<step>/ (pretrained_model/ with policy and processors,
training_state.pt) and checkpoints/last.txt. --resume continues from last.txt (the one rerun
the card allows after a crash, with the identical command plus --resume)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch
from pi0_data import ACTION, CAMERA_KEYS, STATE, Pi0Frames, load_tasks, split

# smolvla_train puts the repo root on sys.path (via smolvla_dataset) before curio_robot.
from smolvla_train import _Cycle, _loader, _log, autocast, hard_exit, release_gpu

from curio_robot.sim import demos

SNAPSHOT_GLOB = "C:/Users/Rohan/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/*"
EVAL_SAMPLES = 256  # held-out frames scored at each eval (a fixed, evenly spaced subset)
EVAL_SEED = 0  # same flow-matching noise and times at every eval, so the numbers compare
CHUNK_SIZE = 30  # run 5's
N_ACTION_STEPS = 10


def _require_offline() -> None:
    for key, value in (("HF_HUB_OFFLINE", "1"), ("WANDB_MODE", "disabled")):
        if os.environ.get(key) != value:
            raise SystemExit(f"set {key}={value} (no downloads, no wandb)")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")  # docs/bringup-7900xt.md


def expert_modules(policy) -> list[torch.nn.Module]:
    """The trainable part under train_expert_only: the action expert and the projections."""
    m = policy.model
    # gemma_expert.model, not its lm_head: the expert's 263M-parameter vocabulary head is
    # never used (pi0 reads actions out through action_out_proj) and gets no gradient.
    return [m.paligemma_with_expert.gemma_expert.model, m.action_in_proj, m.action_out_proj,
            m.state_proj, m.action_time_mlp_in, m.action_time_mlp_out]


def build_policy(source: str, image_hw: tuple[int, int], device: str,
                 expert_dtype: str = "float32", cameras: list[str] | None = None,
                 chunk_size: int = CHUNK_SIZE, n_action_steps: int = N_ACTION_STEPS,
                 weights: Path | None = None):
    """pi0 from the base snapshot or a checkpoint dir, with our inputs, bf16 frozen VLM and the
    expert in expert_dtype. weights: a checkpoint's model.safetensors loaded after the cast,
    so a resumed expert keeps its fp32 values."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from safetensors.torch import load_file

    cfg = PreTrainedConfig.from_pretrained(source)
    cfg.device = device
    cfg.push_to_hub = False
    cfg.dtype = "bfloat16"
    cfg.train_expert_only = True
    cfg.empty_cameras = 0
    cfg.chunk_size = chunk_size
    cfg.n_action_steps = n_action_steps
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_hw))
    cfg.input_features = {
        STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        **{CAMERA_KEYS[c]: image for c in (cameras or list(CAMERA_KEYS))},
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    policy = PI0Policy.from_pretrained(source, config=cfg)
    assert_loaded(policy, source)
    if expert_dtype == "float32":
        for module in expert_modules(policy):
            module.to(torch.float32)
    if weights is not None:
        policy.load_state_dict(load_file(str(weights)), strict=True)
    return policy


def utf8_stdout() -> None:
    """PI0Policy.from_pretrained prints a check mark after reading the weights; on a cp1252
    console that print raises inside its try block and it returns the model with random
    weights (seen in the first smoke run). UTF-8 stdout avoids it; assert_loaded catches it."""
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")


def assert_loaded(policy, source: str) -> None:
    """Fail unless the policy holds source's weights (one tensor compared exactly)."""
    from safetensors import safe_open

    key = "model.action_out_proj.weight"
    with safe_open(str(Path(source) / "model.safetensors"), "pt") as f:
        names = list(f.keys())
        name = key if key in names else key.removeprefix("model.")
        stored = f.get_tensor(name)
    mine = policy.model.action_out_proj.weight.detach().cpu()
    if not torch.equal(stored.to(mine.dtype), mine):
        raise RuntimeError(f"pi0 weights were not loaded from {source}")


def expert_loss(policy, batch: dict) -> torch.Tensor:
    """PI0Policy.forward's flow-matching loss with the frozen prefix run once under no_grad.

    lerobot's training forward runs prefix and suffix through every layer together, so with a
    frozen VLM autograd still keeps (and back-propagates zeros through) the 2B model's
    activations for all ~560 prefix tokens: the second smoke run ran out of memory at batch 16.
    The prefix never attends to the suffix, so the suffix output is the same when the prefix
    is computed first and handed over as a KV cache; that is pi0's own inference path
    (sample_actions, then denoise_step), reused here for training."""
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks, prepare_attention_masks_4d
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    m = policy.model
    images, img_masks = policy._preprocess_images(batch)
    state, actions = policy.prepare_state(batch), policy.prepare_action(batch)
    noise = m.sample_noise(actions.shape, actions.device)
    t = m.sample_time(actions.shape[0], actions.device)
    with torch.no_grad():
        embs, pad, att = m.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                        batch[OBS_LANGUAGE_ATTENTION_MASK])
        vlm = m.paligemma_with_expert.paligemma.model.language_model
        vlm.config._attn_implementation = "eager"  # as sample_actions sets it
        _, past = m.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pad, att)),
            position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True)
    x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
    v_t = m.denoise_step(state, pad, past, x_t, t)
    n_dims = policy.config.output_features[ACTION].shape[0]
    err = torch.nn.functional.mse_loss(noise - actions, v_t, reduction="none")
    return err[..., :n_dims].mean()


def trainable(policy) -> list[torch.nn.Parameter]:
    return [p for p in policy.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate(policy, preprocessor, loader) -> float:
    policy.eval()
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(EVAL_SEED)
    total = n = 0
    for batch in loader:
        with autocast("bf16"):
            loss = expert_loss(policy, preprocessor(batch))
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


def _stats(train_set: Pi0Frames) -> dict:
    return {k: {s: torch.from_numpy(v) for s, v in d.items()} for k, d in train_set.stats().items()}


def train(args: argparse.Namespace) -> None:
    from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

    _require_offline()
    torch.manual_seed(args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    info = demos.load_info(args.demos)
    tasks = load_tasks(args.relabel)
    train_eps, eval_eps = split(info["total_episodes"])
    image_hw = (info["image_size"]["height"], info["image_size"]["width"])

    ckpt = None
    if args.resume:
        last = (out / "checkpoints" / "last.txt").read_text(encoding="utf-8").strip()
        ckpt = out / "checkpoints" / last
    snapshot = max(glob.glob(SNAPSHOT_GLOB))
    source = str(ckpt / "pretrained_model") if ckpt else snapshot
    t_load = time.perf_counter()
    policy = build_policy(source, image_hw, "cuda", args.expert_dtype,
                          weights=ckpt and ckpt / "pretrained_model" / "model.safetensors")
    if args.grad_checkpointing:
        policy.model.gradient_checkpointing_enable()
    policy.train()
    load_s = time.perf_counter() - t_load
    cfg = policy.config

    train_set = Pi0Frames(args.demos, train_eps, cfg.chunk_size, tasks)
    eval_set = Pi0Frames(args.demos, eval_eps, cfg.chunk_size, tasks)
    eval_subset = torch.utils.data.Subset(
        eval_set, list(range(0, len(eval_set), max(1, len(eval_set) // EVAL_SAMPLES))))
    preprocessor, postprocessor = make_pi0_pre_post_processors(cfg, _stats(train_set))

    optim_cfg = cfg.get_optimizer_preset()
    params = trainable(policy)
    optimizer = optim_cfg.build(params)
    scheduler = cfg.get_scheduler_preset().build(optimizer, args.steps)
    start = 0
    if ckpt:
        state = torch.load(ckpt / "training_state.pt", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["step"]

    n_train = sum(p.numel() for p in params)
    (out / "config.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "snapshot": snapshot,
        "train_episodes": [train_eps[0], train_eps[-1]],
        "eval_episodes": [eval_eps[0], eval_eps[-1]],
        "train_frames": len(train_set),
        "eval_frames": len(eval_set),
        "trainable_params": n_train,
        "total_params": sum(p.numel() for p in policy.parameters()),
        "lr": cfg.optimizer_lr,
        "chunk_size": cfg.chunk_size,
        "n_action_steps": cfg.n_action_steps,
        "cameras": CAMERA_KEYS,
        "dtype": cfg.dtype,
        "expert_dtype": args.expert_dtype,
        "train_expert_only": cfg.train_expert_only,
        "load_s": round(load_s, 1),
    }, indent=1), encoding="utf-8")

    batches = _Cycle(_loader(train_set, args.batch_size, args.workers, True, args.seed))
    eval_loader = _loader(eval_subset, args.batch_size, 0, False, args.seed)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    limit_s = args.max_hours * 3600
    with (out / "train_log.jsonl").open("a", encoding="utf-8") as log:
        _log(log, {"event": "start", "step": start, "steps": args.steps,
                   "train_frames": len(train_set), "trainable_params": n_train,
                   "load_s": round(load_s, 1)})
        t_last, data_time, step = time.perf_counter(), 0.0, start
        for step in range(start + 1, args.steps + 1):
            t0 = time.perf_counter()
            batch = next(batches)
            data_time += time.perf_counter() - t0
            with autocast("bf16"):
                loss = expert_loss(policy, preprocessor(batch))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, optim_cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            out_of_time = time.perf_counter() - t_start > limit_s
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
            if args.eval_freq and (step % args.eval_freq == 0 or step == args.steps):
                _log(log, {"step": step, "eval_loss": round(
                    evaluate(policy, preprocessor, eval_loader), 5)})
                t_last = time.perf_counter()
            if step % args.save_freq == 0 or step == args.steps or out_of_time:
                path = save_checkpoint(out, step, policy, preprocessor, postprocessor,
                                       optimizer, scheduler)
                _log(log, {"step": step, "checkpoint": str(path)})
                t_last = time.perf_counter()
            if out_of_time:
                _log(log, {"event": "time_limit", "step": step, "max_hours": args.max_hours})
                break
        _log(log, {"event": "done", "step": step,
                   "elapsed_min": round((time.perf_counter() - t_start) / 60, 2)})
    closer = threading.Thread(target=batches.close, daemon=True)
    closer.start()
    closer.join(timeout=60)  # worker shutdown must not block the exit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--relabel", type=Path, required=True, help="pi0_data relabel.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-hours", type=float, default=8.0,
                        help="stop (after a checkpoint) once this much training time has passed")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=0, help="0: never")
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="from checkpoints/last.txt")
    parser.add_argument("--expert-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--grad-checkpointing", action="store_true")
    utf8_stdout()
    train(parser.parse_args())
    release_gpu()
    hard_exit(0)


if __name__ == "__main__":
    main()
