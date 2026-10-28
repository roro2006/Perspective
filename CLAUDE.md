@AGENTS.md

## Claude sessions
- **Working a card**: same rules as every agent above. Use a worktree on branch
  `task/<id>` so parallel sessions never share a checkout.
- **Coordinating** (when asked to): pick ready cards with
  `uv run python scripts/tasks.py ready`, review finished branches against the
  card and its notes, merge to `main`, then set the card's `status` and run
  `uv run python scripts/tasks.py check`. Only the coordinator edits statuses
  and contracts; contract changes get their own commit and a note to every card
  that reads the changed file.
- Cards marked `agent: claude` need judgement across modules (integration,
  harnesses). Cards marked `agent: qwen` are sized for a 32k-context local
  model; prefer giving those to Qwen agents.
