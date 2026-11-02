"""PIL overlays for the PoC video: captions, subtitles, the head-camera inset, grounder chips,
check and X marks, the intro cards and the end grid."""

from __future__ import annotations

from functools import cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
GREEN = (46, 204, 113)
RED = (231, 76, 60)
FONT_DIR = "C:/Windows/Fonts/"


@cache
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(FONT_DIR + name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def alpha_of(t: float, t0: float, t1: float, fade: float = 0.25) -> float:
    if t < t0 or t > t1:
        return 0.0
    return float(min(1.0, (t - t0) / fade, (t1 - t) / fade))


def _layer() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def shadow_text(d: ImageDraw.ImageDraw, xy, text, f, fill=(255, 255, 255), a=1.0,
                anchor="la") -> None:
    x, y = xy
    for dx, dy in ((2, 2), (3, 3)):
        d.text((x + dx, y + dy), text, font=f, fill=(0, 0, 0, int(150 * a)), anchor=anchor)
    d.text((x, y), text, font=f, fill=(*fill, int(255 * a)), anchor=anchor)


def pill(d, box, a=1.0, fill=(12, 14, 18), alpha=170, radius=14) -> None:
    d.rounded_rectangle(box, radius=radius, fill=(*fill, int(alpha * a)))


def check_mark(d, c, r, good: bool, a=1.0) -> None:
    x, y = c
    col = GREEN if good else RED
    d.ellipse((x - r, y - r, x + r, y + r), fill=(*col, int(235 * a)),
              outline=(255, 255, 255, int(255 * a)), width=max(2, r // 9))
    w = max(3, r // 4)
    if good:
        pts = [(x - 0.5 * r, y + 0.02 * r), (x - 0.12 * r, y + 0.4 * r), (x + 0.52 * r,
                                                                            y - 0.38 * r)]
        d.line(pts, fill=(255, 255, 255, int(255 * a)), width=w, joint="curve")
    else:
        k = 0.42 * r
        d.line((x - k, y - k, x + k, y + k), fill=(255, 255, 255, int(255 * a)), width=w)
        d.line((x - k, y + k, x + k, y - k), fill=(255, 255, 255, int(255 * a)), width=w)


def letter_disc(d, c, letter: str, a=1.0, ring=(255, 255, 255)) -> None:
    x, y = c
    d.ellipse((x - 19, y - 19, x + 19, y + 19), fill=(255, 255, 255, int(235 * a)),
              outline=(*ring, int(255 * a)), width=3)
    d.text((x, y - 1), letter, font=font(24, True), fill=(10, 10, 10, int(255 * a)),
           anchor="mm")


def arrow(d, p0, p1, col, width=10, a=1.0) -> None:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    n = v / max(1e-6, np.linalg.norm(v))
    perp = np.array([-n[1], n[0]])
    head = p1 - 34 * n
    d.line((*p0, *head), fill=(*col, int(255 * a)), width=width)
    tri = [tuple(p1), tuple(head + 20 * perp), tuple(head - 20 * perp)]
    d.polygon(tri, fill=(*col, int(255 * a)))


def inset(base: Image.Image, img: np.ndarray, label: str, a: float = 1.0) -> None:
    """Head-camera inset, top right: what the grounder saw, marks and all."""
    if a <= 0:
        return
    pic = Image.fromarray(img)
    pic.thumbnail((400, 300))
    x1, y0 = W - 40, 40
    x0 = x1 - pic.width
    frame = Image.new("RGBA", (pic.width + 8, pic.height + 8), (255, 255, 255, int(230 * a)))
    base.alpha_composite(frame, (x0 - 4, y0 - 4))
    pic = pic.convert("RGBA")
    pic.putalpha(int(255 * a))
    base.alpha_composite(pic, (x0, y0))
    lay, d = _layer()
    f = font(20)
    tw = d.textlength(label, font=f)
    pill(d, (x1 - tw - 24, y0 + pic.height + 10, x1, y0 + pic.height + 44), a, alpha=190,
         radius=8)
    d.text((x1 - 12, y0 + pic.height + 27), label, font=f, fill=(255, 255, 255, int(255 * a)),
           anchor="rm")
    base.alpha_composite(lay)


def chip(d, text: str, good: bool, a: float, y: float) -> None:
    f = font(24, True)
    tw = d.textlength(text, font=f)
    x1 = W - 40
    pill(d, (x1 - tw - 54, y, x1, y + 50), a, alpha=200, radius=10)
    col = GREEN if good else RED
    d.ellipse((x1 - tw - 40, y + 17, x1 - tw - 24, y + 33), fill=(*col, int(255 * a)))
    d.text((x1 - 14, y + 25), text, font=f, fill=(255, 255, 255, int(255 * a)), anchor="rm")


def subtitle(d, text: str, a: float, small: str = "") -> None:
    f = font(50, True)
    q = f"\u201c{text}\u201d"
    tw = d.textlength(q, font=f)
    y = H - 170
    pill(d, (W / 2 - tw / 2 - 30, y - 8, W / 2 + tw / 2 + 30, y + 72), a, alpha=175)
    d.text((W / 2, y + 32), q, font=f, fill=(255, 255, 255, int(255 * a)), anchor="mm")
    if small:
        d.text((W / 2, y - 26), small, font=font(24), fill=(190, 225, 255, int(255 * a)),
               anchor="mm")


def caption(d, text: str, a: float) -> None:
    shadow_text(d, (56, 46), text, font(52, True), a=a)


def big_center(d, text: str, a: float, col=(255, 255, 255)) -> None:
    f = font(110, True)
    tw = d.textlength(text, font=f)
    yc = H * 0.6
    pill(d, (W / 2 - tw / 2 - 40, yc - 85, W / 2 + tw / 2 + 40, yc + 85), a, alpha=170)
    d.text((W / 2, yc - 10), text, font=f, fill=(*col, int(255 * a)), anchor="mm")


def title_card(lines: list[tuple[str, float, float]], t: float) -> np.ndarray:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    d = ImageDraw.Draw(img)
    for text, t0, t1 in lines:
        a = alpha_of(t, t0, t1, 0.4)
        if a > 0:
            d.text((W / 2, H / 2), text, font=font(120, True), fill=(255, 255, 255, int(255 * a)),
                   anchor="mm")
    return np.asarray(img.convert("RGB"))


def end_grid(tiles: list[tuple[np.ndarray, str, bool, str]], t: float) -> np.ndarray:
    """tiles: (final frame, scene label, grounder right?, outcome text)."""
    img = Image.new("RGBA", (W, H), (10, 11, 14, 255))
    tw, th, gap = 576, 324, 24
    x0 = (W - 3 * tw - 2 * gap) // 2
    y0 = 170
    for i, (frame, label, good, outcome) in enumerate(tiles):
        a = alpha_of(t, 0.15 * i, 99, 0.4)
        pic = Image.fromarray(frame).resize((tw, th), Image.LANCZOS).convert("RGBA")
        pic.putalpha(int(255 * a))
        x, y = x0 + (i % 3) * (tw + gap), y0 + (i // 3) * (th + gap + 40)
        img.alpha_composite(pic, (x, y))
        d = ImageDraw.Draw(img)
        check_mark(d, (x + tw - 40, y + 40), 28, good, a)
        d.text((x, y + th + 8), label, font=font(26, True), fill=(255, 255, 255, int(255 * a)))
        d.text((x + tw, y + th + 12), outcome, font=font(22),
               fill=(*(GREEN if good else RED), int(255 * a)), anchor="ra")
    d = ImageDraw.Draw(img)
    a = alpha_of(t, 0.6, 99, 0.5)
    d.text((W / 2, 92), "Curio: robots that see it your way.", font=font(72, True),
           fill=(255, 255, 255, int(255 * a)), anchor="mm")
    return np.asarray(img.convert("RGB"))
