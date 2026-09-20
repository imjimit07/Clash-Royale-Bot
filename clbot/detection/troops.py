"""Enemy troop detection (Part 3.2)."""

from __future__ import annotations

from clbot.bot.coords import PLAYABLE_PLAY_REGION_LTRB

LEFT_LANE = "left"
RIGHT_LANE = "right"


def detect_enemy_troops(frame, logger=None, templates: dict | None = None, regions=None) -> list[dict]:
    """Template-match known troop icons; falls back to empty list (no crash)."""
    if frame is None or templates is None:
        return []
    try:
        from clbot.detection.image_rec import find_image_single
    except Exception:
        return []
    threats: list[dict] = []
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    mid_x = (left + right) // 2
    lane_regions = {
        LEFT_LANE: (left, top, mid_x - left, bottom - top),
        RIGHT_LANE: (mid_x, top, right - mid_x, bottom - top),
    }
    for troop_name, template in (templates or {}).items():
        for lane, region in lane_regions.items():
            try:
                hit = find_image_single(frame, template, region=region)
            except Exception:
                hit = None
            if hit:
                threats.append({"type": troop_name, "lane": lane, "pos": hit.get("location")})
    return threats
