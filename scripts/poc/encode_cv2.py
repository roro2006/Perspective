"""Raw RGB frames on stdin -> mp4v file. Run with an interpreter that has cv2 (the project env
doesn't), e.g. C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe.

    python encode_cv2.py OUT.mp4 WIDTH HEIGHT FPS < frames.rgb"""

from __future__ import annotations

import sys

import cv2
import numpy as np


def main() -> None:
    out, w, h, fps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cv2 could not open {out}")
    size = w * h * 3
    n = 0
    while True:
        buf = sys.stdin.buffer.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        n += 1
    writer.release()
    print(f"{out}: {n} frames", file=sys.stderr)


if __name__ == "__main__":
    main()
