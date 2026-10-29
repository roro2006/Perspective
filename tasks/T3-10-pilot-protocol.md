---
id: T3-10
title: Pilot study protocol for real people
status: done
agent: claude
depends: [T3-02]
owns: [docs/pilot-protocol.md]
reads: [docs/target3-collaboration.md, docs/pivot-shared-perspective.md, curio_robot/collab/metrics.py]
accept: uv run pytest -q
---

## Goal
`docs/pilot-protocol.md`: how to run the same fetch task with real people
once the arm and head camera are set up, so the sim scoreboard has a real
counterpart.

## Contents
Design (within-subjects, view-off / view-on / view-on+gaze, counterbalanced
order), task script and scene, participants and consent (what is recorded,
how it is stored, deletion on request), measures (the session metrics plus
NASA-TLX and a short trust scale after each condition), sample size for a
pilot (and why it is a pilot, not a result), analysis plan, and the claims we
will and will not make from it.

## Done when
The doc is reviewed by the coordinator; the full suite and ruff still pass.
