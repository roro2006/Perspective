"""What the console says about the system (T5-33): a preflight of this machine and the
downstream drop-ins as the repo records them.

The drop-in lines read each card's `status` from tasks/ at startup, so the console never
claims more than the board does. The one-line results are copied from the cards' notes with
their source named, and a card without a recorded result says "not run"."""

from __future__ import annotations

import platform
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "tasks"

# (card id, card file, what it tests, recorded result or None, source)
DROPINS = (
    ("T5-31", "T5-31-libero-dropin.md",
     "LIBERO-Spatial: SmolVLA and pi0.5 trained by others, Curio in front, no retraining",
     None, "tasks/T5-31-libero-dropin.md"),
    ("T5-32", "T5-32-go2-nav-dropin.md",
     "Go2: pretrained Isaac Lab navigation policy, Curio pose goals",
     "FAIL on the port: 0/20 goals reached, 13 falls; Curio not evaluated",
     "tasks/notes/T5-32.md"),
    ("T5-28/30", "T5-30-pi0-fix.md",
     "pi0 fine-tuned on our demos, Curio robot-words",
     "blocked: loader race fixed, retrain stopped at ~8.9k steps, gate not run",
     "tasks/notes/T5-30.md"),
    ("T5-17", "T5-17-run5.md",
     "SmolVLA fine-tuned on our demos, Curio ring goal",
     "gate missed: ring-on-target 0.57 < 0.60",
     "tasks/notes/T5-17.md"),
)


def card_status(name: str) -> str:
    path = TASKS / name
    if not path.exists():
        return "missing"
    m = re.search(r"^status:\s*(\S+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1) if m else "unknown"


def dropins() -> list[dict]:
    return [{"card": cid, "status": card_status(name), "what": what,
             "result": result or "not run: no result recorded", "source": src}
            for cid, name, what, result, src in DROPINS]


def preflight(render_ms: float | None, sim_error: str | None) -> list[dict]:
    """Checks for this machine. `ok` is True, False, or None (not needed by this demo)."""
    checks = []
    try:
        import mujoco

        checks.append({"name": "MuJoCo", "ok": True, "detail": mujoco.__version__})
    except ImportError as e:
        checks.append({"name": "MuJoCo", "ok": False, "detail": str(e)})
    if sim_error:
        checks.append({"name": "Simulator", "ok": False, "detail": sim_error})
    elif render_ms is None:
        checks.append({"name": "Simulator", "ok": False, "detail": "starting"})
    else:
        checks.append({"name": "Simulator", "ok": True,
                       "detail": f"SO-101 scene up; one render {render_ms} ms"})
    checks.append({"name": "Curio pipeline", "ok": True,
                   "detail": "detector, fusion, perspective, rule resolver (pipeline.step)"})
    checks.append({"name": "Trained grounder (VLM + LoRA)", "ok": None,
                   "detail": "not on this machine; this demo uses the geometry pipeline"})
    checks.append({"name": "GPU", "ok": None,
                   "detail": ("nvidia-smi found" if shutil.which("nvidia-smi") else
                              "none visible") + "; not needed for this demo"})
    checks.append({"name": "Python", "ok": True,
                   "detail": f"{sys.version.split()[0]} on {platform.system()}"})
    return checks
