"""uv run python -m curio_robot.demo [--port 8765] [--host 127.0.0.1] [--scene perspective:0]"""

from __future__ import annotations

import argparse
from pathlib import Path

from curio_robot.demo.server import serve


def main() -> None:
    p = argparse.ArgumentParser(prog="curio_robot.demo", description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 exposes it to the network")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--scene", default="perspective:0", help="first scene, kind:seed")
    p.add_argument("--record-dir", type=Path, default=Path("out/demo/recordings"))
    p.add_argument("--no-record", action="store_true")
    args = p.parse_args()
    kind, _, seed = args.scene.partition(":")
    serve(args.host, args.port, None if args.no_record else args.record_dir,
          (kind, int(seed or 0)))


if __name__ == "__main__":
    main()
