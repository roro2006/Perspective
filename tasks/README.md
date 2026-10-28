# Task board

Work is split into cards so several agents (Claude sessions and local Qwen
agents) can build in parallel without stepping on each other. Rules for agents:
[`AGENTS.md`](../AGENTS.md). Design: [`docs/target1-conventions.md`](../docs/target1-conventions.md).

```bash
uv run python scripts/tasks.py list          # every card, status, agent, deps
uv run python scripts/tasks.py ready         # todo cards whose deps are done
uv run python scripts/tasks.py check         # ownership overlaps, deps, pack sizes
uv run python scripts/tasks.py pack T1-01    # one-file context for an agent
```

## Target 1 waves

| Wave | Cards (parallel within a wave) | Agent |
|---|---|---|
| 0 | T1-00 contracts, tests, board | done |
| 1 | T1-01 readings · T1-02 belief · T1-03 noise · T1-04 decide · T1-06 sim users · T1-07 recorder | Qwen ×6 (or any) |
| 2 | T1-05 profiles (needs 02) · T1-09 learning curves (needs 01–04, 06) | Qwen · any |
| 3 | T1-08 runtime integration (needs 01–05, 07) | Claude |
| 4 | T1-10 live learning demo · T1-11 results write-up | Claude · any |

Wave 1 is six independent modules with frozen contracts and acceptance tests,
each one file. That's the bulk of the work and the right size for a local model.

## Card lifecycle
`todo` → `doing` → `review` → `done` (or `blocked`, with a reason in the notes).
Only the coordinator changes `status`, on `main`. An agent's branch never
touches a card, so parallel branches don't conflict on the board.

1. Coordinator picks a ready card, sets `doing`, starts an agent on branch
   `task/<id>` in its own worktree:
   `git worktree add ../curio-t1-01 -b task/t1-01`
2. The agent works only in `owns`, passes `accept` + full suite + ruff, writes
   `tasks/notes/<ID>.md`, commits.
3. Coordinator reviews (notes first, then diff), merges to `main`, sets `done`,
   runs `check`, starts whatever became ready.

`check` fails if two open cards own the same file, unless one depends on the
other (then they can't run at the same time anyway).

## Card format
Frontmatter keys: `id`, `title`, `status`, `agent` (`qwen` | `claude` | `any`),
`depends`, `owns`, `reads`, `accept`, and optional `budget` (max tokens for
the context pack; defaults: qwen 16000, any 24000, claude 60000). Lists are
`[a, b]` on one line. The body says the goal, gives hints and says when it's done; the
spec itself lives in the stub's docstrings, so the code and the card can't drift.

## Running Qwen agents on the 7900 XT
One Qwen 27B-class model at 4-bit is ~15-17 GB on the 20 GB card, leaving a
few GB for KV cache. That's one or two agents at 32k context at a time,
served by one vLLM instance (continuous batching shares the KV pool), so give
Qwen the wave-1 cards one or two at a time, not all six.

Example (WSL2 + ROCm; **unverified**, check flags against your vLLM build):

```bash
vllm serve <qwen-27b-awq-or-gptq> --served-model-name qwen-dev \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --enable-prefix-caching --port 8000
```

Point any OpenAI-compatible coding agent (Qwen Code, OpenCode, Aider, ...) at
`http://localhost:8000/v1`, model `qwen-dev`, with its working directory set
to the card's worktree. Start the session by giving it the pack:
`out/context/<ID>.md` contains AGENTS.md, the card, every file it needs, and
the instructions repeated at the end, so it works whatever context file the
tool loads by default.

The same GPU serves the runtime's VLM, Whisper and Kokoro. Don't run dev
agents and a latency measurement at the same time.
