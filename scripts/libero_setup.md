# LIBERO venv for T5-31 (Windows 11, RX 7900 XT, ROCm 7.2.1)

Every step taken on 2026-11-05 to run `lerobot/smolvla_libero` and
`lerobot/pi05_libero_finetuned_v044` in LIBERO-Spatial, in the order done. The existing
lerobot venv (`C:\Users\Rohan\dev\lerobot-rocm\.venv`) was only read, never changed.

## 1. The venv: a copy of the lerobot venv
The lerobot venv was itself built as a copy of the `rocm-train` venv plus
`uv pip install "lerobot[smolvla]==0.6.1" "mujoco==3.17.0"` (docs/bringup-7900xt.md). Copying it
keeps the same `torch 2.9.1+rocm7.2.1` / `torchvision 0.24.1+rocm7.2.1` wheels from
`repo.radeon.com/rocm/windows/rocm-rel-7.2.1` and `lerobot==0.6.1`, with no PyTorch download.

```bash
cp -a C:/Users/Rohan/dev/lerobot-rocm/.venv C:/Users/Rohan/dev/libero-venv   # 6.3 GB
C:/Users/Rohan/dev/libero-venv/Scripts/python.exe -c "import torch; print(torch.__version__)"
# 2.9.1+rocm7.2.1
```
Use `python.exe -m pip` or `uv pip --python ...`; the copied `Scripts/*.exe` entry points still
name the old path.

## 2. hf-libero 0.1.4, robosuite and LeRobot's LIBERO extras
`lerobot[libero]` installs `hf-libero` only on Linux (its marker is `sys_platform == "linux"`),
so it is named explicitly. Its dependency `hf-egl-probe` (and robomimic's `egl-probe`) is a
Linux EGL GPU probe built from source with CMake; the first attempt failed with
`RuntimeError: CMake must be installed.` while building `hf-egl-probe` 1.0.2. Nothing on
Windows uses it (robomimic imports `egl_probe` only for Linux EGL rendering), so the second
attempt keeps both probes Linux-only through an override file:

```bash
printf 'hf-egl-probe>=1.0.1 ; sys_platform == "linux"\negl-probe>=1.0.1 ; sys_platform == "linux"\n' > egl_override.txt
uv pip install --python C:/Users/Rohan/dev/libero-venv/Scripts/python.exe \
  --override egl_override.txt "hf-libero==0.1.4" "lerobot[smolvla,pi,libero]==0.6.1"
```
82 packages installed, 2 replaced: `mujoco 3.17.0 -> 3.8.1` (hf-libero needs `mujoco<3.9`) and
`fsspec 2026.9.0 -> 2026.2.0`. New: `robosuite 1.4.0`, `robomimic 0.2.0`, `bddl 1.0.1`,
`datasets 4.8.5`, `torchcodec 0.11.1`, `scipy 1.18.1`, `opencv-python 5.0.0.93` (beside the
existing headless 4.13), `wandb`, `hydra-core` and their dependencies. Unchanged: torch,
torchvision, `lerobot 0.6.1`, `transformers 5.5.4`, `numpy 2.2.6`, `gymnasium 1.4.0`.

## 3. LIBERO config (no interactive prompt)
LIBERO asks on first import where its datasets go. The config file is written up front in the
venv, and every run sets `LIBERO_CONFIG_PATH` (scripts/libero_eval.py does it):

```python
# C:/Users/Rohan/dev/libero-venv/libero_config/config.yaml, keys from LIBERO's
# get_default_path_dict(): benchmark_root, bddl_files, init_states, datasets, assets, all
# under <venv>/Lib/site-packages/libero/libero (bddl_files and init_files ship in the wheel)
```

## 4. LIBERO assets
hf-libero's wheel has no meshes; it fetches them from the Hub dataset `lerobot/libero-assets`
into `~/.cache/libero/assets` the first time a scene is built. Done once ahead of time:

```bash
LIBERO_CONFIG_PATH=C:/Users/Rohan/dev/libero-venv/libero_config \
  C:/Users/Rohan/dev/libero-venv/Scripts/python.exe -I -c \
  "from libero.libero.utils.download_utils import download_assets_from_huggingface as d; d()"
# 586 files, 406 MB, 25 s
```

## 5. robosuite on Windows (two documented fixes inside the venv)
- `robosuite/utils/binding_utils.py` loads `mujoco.dll` from its own folder on Windows:
  `cp <venv>/Lib/site-packages/mujoco/mujoco.dll <venv>/Lib/site-packages/robosuite/utils/`.
- robosuite's private macros (what `robosuite/scripts/setup_macros.py` creates) at
  `<venv>/Lib/site-packages/robosuite/macros_private.py`: `MUJOCO_GPU_RENDERING = False`
  (True forces `MUJOCO_GL=egl`, which robosuite rejects on Windows; rendering then uses GLFW)
  and `FILE_LOGGING_LEVEL = None` (it logs to `/tmp/robosuite.log`, absent here). The file also
  sets both on `robosuite.macros`, because binding_utils and log_utils read that module.

Check: a LIBERO-Spatial `OffScreenRenderEnv` builds in 6 s; a step with two 256 x 256 camera
renders takes about 14 ms; a hard reset 0.4 s.

## 6. Checkpoints (Hub cache, pinned revisions)
```python
from huggingface_hub import snapshot_download
snapshot_download("lerobot/smolvla_libero", revision="a24f1f380461b665158d52acd140152c79e8b36d")
snapshot_download("lerobot/pi05_libero_finetuned_v044",
                  revision="7152f13e34c6c115d808aea0e5d08aca74eac14b")
```
- smolvla_libero (Apache-2.0): 865 MB (`model.safetensors` 907 MB as listed). Its tokenizer
  and config, `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, were already cached (4.9 MB, T5-08).
- pi05_libero_finetuned_v044 (Gemma terms, accepted by the user): `model.safetensors`
  7.47 GB. Its tokenizer, `google/paligemma-3b-pt-224`, was already cached (T5-28).
The token in the Hub cache was used as is and never printed.

## 7. Running
All runs are offline (`HF_HUB_OFFLINE=1`; scripts/libero_eval.py sets it, plus
`MIOPEN_FIND_MODE=FAST` and `LIBERO_CONFIG_PATH`), from the repo root:
```bash
C:/Users/Rohan/dev/libero-venv/Scripts/python.exe scripts/libero_eval.py <step> [--policy ...]
```
Two `--policy.*` overrides on top of the checkpoints' configs, the same for P and Curio runs:
- SmolVLA `--policy.load_vlm_weights=false`: the checkpoint holds the whole model; true would
  download SmolVLM2's 2 GB backbone and then overwrite it with the checkpoint's weights.
- pi0.5 `--policy.compile_model=false`: the config asks for `torch.compile(max-autotune)`,
  and Inductor has no Triton backend on Windows ROCm. Same weights and numerics, eager mode.
