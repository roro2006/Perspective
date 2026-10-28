"""Combine the VLM's probabilities with the target-1 belief. Task T2-06."""

from __future__ import annotations

VLM_WEIGHT = 1.0
VLM_FLOOR = 0.05  # a VLM "0" never vetoes on its own


def fuse(
    belief: dict[str, float],
    vlm: dict[str, float],
    *,
    weight: float = VLM_WEIGHT,
    floor: float = VLM_FLOOR,
) -> dict[str, float]:
    """Product of experts, normalized to sum to 1.

    - belief empty (no readings: the rules couldn't parse it) -> normalized
      vlm (or {} if that's empty too).
    - vlm empty (unavailable) -> belief unchanged.
    - Otherwise over the keys of `belief` only (the VLM can't add candidates):
      p(k) ∝ belief[k] * (floor + vlm.get(k, 0)) ** weight. If every product
      is 0, return belief unchanged.
    weight 0 returns the belief unchanged."""
    # 1. belief empty: VLM decides
    if not belief:
        if not vlm:
            return {}
        total = sum(vlm.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in vlm.items()}

    # 2. vlm empty: belief unchanged
    if not vlm:
        return dict(belief)

    # 3. weight == 0: belief unchanged
    if weight == 0:
        return dict(belief)

    # Product of experts over belief keys only
    products = {}
    for k, b in belief.items():
        vlm_val = vlm.get(k, 0.0)
        products[k] = b * (floor + vlm_val) ** weight

    total = sum(products.values())
    if total == 0:
        return dict(belief)

    return {k: p / total for k, p in products.items()}
