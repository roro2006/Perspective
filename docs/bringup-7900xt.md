# Bring-up on the RX 7900 XT (T2-10)

Measured 2026-10-28 on the dev desktop. Route: **llama.cpp (Vulkan) on Windows**,
not vLLM in WSL2: WSL isn't installed on this machine, and llama.cpp was
already here. The vLLM route stays open; nothing below depends on it.

## Machine and versions
| Item | Value |
|---|---|
| GPU | AMD Radeon RX 7900 XT (20 GB), driver 32.0.31041.1004 |
| Host | Windows 11 Home 10.0.26200, 31 GB RAM |
| Server | llama.cpp `llama-server` 0.6.0-dev, build 11512, commit a11f57ba9, Vulkan |
| Model | `Qwen/Qwen3-VL-4B-Instruct-GGUF`: `Qwen3VL-4B-Instruct-Q8_0.gguf` (4.28 GB) |
| Vision projector | `mmproj-Qwen3VL-4B-Instruct-F16.gguf` (0.84 GB) |

## Serve command
```powershell
$d = 'C:\Users\Rohan\dev\llama'
& "$d\vulkan\llama-server.exe" -m "$d\models\Qwen3VL-4B-Instruct-Q8_0.gguf" `
  --mmproj "$d\models\mmproj-Qwen3VL-4B-Instruct-F16.gguf" -dev Vulkan0 -ngl 99 -fa on `
  -c 8192 --jinja --parallel 1 --alias qwen-vl --host 127.0.0.1 --port 8081
```
Loads in about 4 s. VRAM: 6.5 GB for the server process (7.7 GB on the card in
total with the desktop). Grounder config: `VllmGrounder("http://127.0.0.1:8081/v1", "qwen-vl")`.

## Contract checks (the [VERIFY] items)
| Question | Answer on this stack |
|---|---|
| Capital letters single tokens? | **Yes.** `A`..`Z` are token ids 32..57 (`" A"` is a different token, 362; the prompt makes the model emit the bare letter). |
| Image as a data URL in `image_url`? | **Yes**, with `--mmproj`. |
| `top_logprobs` returned? | **Yes**, OpenAI shape (`choices[0].logprobs.content[0].top_logprobs`); `letter_probs` and `VllmGrounder.score` work unchanged. |
| Guided-choice parameter | llama-server has no `guided_choice`; it **ignores the key silently**, so the default `choice_param` is harmless. Its constraint is `grammar` (GBNF, e.g. `root ::= "A" \| "B"`), which works but expects a string, not the letter list `VllmGrounder` sends. Not needed in practice: with `max_tokens: 1` the top 10 tokens were letters plus filler like "The" at about e^-10. |
| `top_logprobs: 10` enough? | Fine up to 10 visible objects; the model's mass is on letters. Above 10 marks, letters outside the top 10 get 0. |

## Latency and accuracy (`scripts/bench_grounding.py --n 50`, seed 0)
Fresh sim renders, one request at a time, 2 warm-up calls discarded. Latency is
the client's wall time per request (image encode + prompt + one token).

| Condition | p50 ms | p95 ms | Errors |
|---|---|---|---|
| GPU otherwise idle | 100 | 115 | 0 |
| With whisper-server, Kokoro and the detector loaded | not run: those services aren't installed here | | |

Accuracy of the **base model** (no LoRA), same run:

| Kind | n | Accuracy | Chance (about) |
|---|---|---|---|
| color | 16 | 1.00 | 0.2–0.25 |
| relation | 17 | 0.47 | 0.5 (pair) |
| deictic | 17 | 0.18 | 0.2 |
| all | 50 | 0.54 | |

"emitted" and "argmax" accuracy were identical. Reading: the base model reads
colours perfectly, guesses on left/right, and ignores where the person is
looking. Deictic and relation are what the LoRA has to teach; the latency gate
(p95 ≤ 800 ms) has ample room.

## LoRA training on the card (native Windows ROCm, no WSL)
AMD's "PyTorch on Windows" release, in its own venv so `uv sync` in the repo
can't replace the ROCm torch with the PyPI build.

| Item | Value |
|---|---|
| ROCm | 7.2.1 Windows wheels from `repo.radeon.com/rocm/windows/rocm-rel-7.2.1` (HIP 7.2.53211) |
| PyTorch | `torch 2.9.1+rocm7.2.1`, `torchvision 0.24.1+rocm7.2.1`, Python 3.12.10 |
| Driver | Adrenalin 26.8.1 (AMD's page asks for 26.2.2; the newer one works) |
| Libraries | transformers 5.21.0, peft 0.21.2, accelerate 1.15.0 |
| Base weights | `Qwen/Qwen3-VL-4B-Instruct` safetensors, bf16, 8.89 GB, Apache-2.0 |

```powershell
py -3.12 -m venv C:\Users\Rohan\dev\rocm-train\.venv
$py = 'C:\Users\Rohan\dev\rocm-train\.venv\Scripts\python.exe'
$b = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1'
& $py -m pip install --no-cache-dir "$b/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
  "$b/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
  "$b/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" "$b/rocm-7.2.1.tar.gz"
& $py -m pip install --no-cache-dir "$b/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" `
  "$b/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
& $py -m pip install "transformers>=4.57" "peft>=0.13" "accelerate>=1.0" numpy mujoco pillow
& $py -m pip install --no-deps -e C:\Users\Rohan\dev\Perspective
```

Run (from the repo root). **`MIOPEN_FIND_MODE=FAST` is required**: MIOpen's
default exhaustive kernel search takes about 40 s per new input shape for the
vision tower's Conv3d, and the cropped images change shape every example, so
without it every step pays that. With it the same conv takes about 1 ms.
```powershell
$env:MIOPEN_FIND_MODE = 'FAST'
& $py scripts\train_lora.py --data out\grounding\data.jsonl `
  --model C:\Users\Rohan\dev\rocm-train\models\Qwen3-VL-4B-Instruct --out out\lora\<name>
```

Measured: 40 sim examples (37 train, 3 eval), 1 epoch, batch 1, grad accum 1,
LoRA r=16 on q/k/v/o (11.8M trainable of 4.45B), bf16, gradient checkpointing.

| Metric | Value |
|---|---|
| Model load | about 2 s |
| First step (incl. kernel setup) | 10.3 s |
| Step time, median / worst | **0.285 s / 0.47 s** |
| Peak VRAM, allocated / reserved | **9.67 GB / 11.3 GB** of 20 |
| Whole epoch + eval + save | 21.5 s |

A default run (3,000 examples, 3 epochs, about 8,100 train steps) is about
40 minutes at this rate.

**4B vs 8B:** train the **4B** here. Qwen3-VL-8B's bf16 weights alone are
about 17 GB, which leaves no room for activations on a 20 GB card; the 8B
would need 4-bit QLoRA (bitsandbytes on ROCm/Windows, unverified) or a rented
GPU. The 4B leaves about 9 GB free.

Bug found and fixed here: `_encode` appended the answer token to `input_ids`
and `attention_mask` but not to Qwen3-VL's `mm_token_type_ids`, so the first
forward failed in rope indexing (mask 211 vs tensor 210). Every per-token
tensor now grows with the answer, which is a text token. Separately, a crashed
run can hang on exit under HIP on Windows, burning a CPU core; kill it.

## Full training run (2026-10-28)
`out/grounding`: 3,000 sim examples (seed 0, 1,000 per kind), 2,698 train and
302 held out by id hash. Defaults: 3 epochs, batch 1, grad accum 16 (507
optimizer steps), lr 2e-4, LoRA r=16 on q/k/v/o. **41 minutes** on the card,
4.6 s per optimizer step (16 examples), peak VRAM 9.7 GB allocated.

Held-out eval during training: 80.5% after epoch 1, 82.5% after 2, 84.1% after 3.

Base vs adapter on the same 302 held-out examples (argmax over each example's
letters, scored in PyTorch with `train_lora`'s own encoding):

| Kind | n | Base Qwen3-VL-4B | With the LoRA |
|---|---|---|---|
| color | 105 | 0.990 | **1.000** |
| relation | 96 | 0.469 | **0.917** |
| deictic | 101 | 0.208 | **0.604** |
| all | 302 | 0.563 | **0.841** |

The base model's 0.563 here agrees with llama.cpp's 0.54 above, so the two
stacks see the same model. Relation is near its ceiling: in unmarked requests
the right answer follows a frame the simulated person drew at random with the
habit shown in the prompt (0.15 or 0.85), so the best possible is about 0.92.
Deictic ("hand me that one", from where the person looks) is where the
remaining error is. These are simulated scenes and simulated people. Per-request
latency in PyTorch rose from 116 to 135 ms with the adapter unmerged.

Adapter: `out/lora/qwen3vl4b-r16` (not in git; `heldout_scores.json` and
`train_log.jsonl` next to it).

## Not done yet
- **Contention run**: whisper-server, Kokoro and the detector aren't installed
  on this machine.
- **Word timestamps** (`anchor.parse_words` vs whisper-server's JSON): needs
  whisper-server. `WhisperCppServer.transcribe_words` still uses guessed fields.
- **Serving the adapter**: llama-server needs it merged and converted to GGUF
  (or served with `--lora`), so `curio-sim ground-eval` and the live runtime
  still use the base model.
- **8B comparison**: not run (see above).
