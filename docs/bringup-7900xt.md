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

## Not done yet
- **Contention run**: whisper-server, Kokoro and the detector aren't installed
  on this machine.
- **Word timestamps** (`anchor.parse_words` vs whisper-server's JSON): needs
  whisper-server. `WhisperCppServer.transcribe_words` still uses guessed fields.
- **LoRA training step** (peak VRAM, step time, 4B vs 8B): needs PyTorch with
  ROCm, which isn't installed (no torch on the host Python).
- **8B comparison**: not run.
