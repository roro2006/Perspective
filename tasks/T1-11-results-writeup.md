---
id: T1-11
title: Write up target-1 results for the pitch
status: todo
agent: any
depends: [T1-09]
owns: [product/results.md, README.md, docs/target1-conventions.md]
reads: [docs/target1-conventions.md, product/results.md, README.md]
accept: uv run curio-sim learn --user all --baseline --noise
---

## Goal
Run the learning curves (with and without noise, with the baseline) and add a
"Learning from corrections" section to `product/results.md` with a small table
and a two-sentence takeaway. Update the README module-status table and the
"Next" list.

## Rules for claims
- Say "simulated users" every time; these are not people.
- Mark the perspective-taking literature claim **[VERIFY]** as in the design doc.
- Report the baseline next to every learned number.

## Done when
The numbers in the doc match a fresh run of the accept command (state its seed).
