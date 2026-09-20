"""Tactical decision agents for battle play selection and placement.

Two policies live here:

- :class:`TacticalAIEngine` — threat-aware CV policy. Reads enemy
  health bars from the live frame, reacts to the threatened lane with pulls,
  gates damage spells behind real clusters, and banks elixir (>= 9) before
  starting a push. Superseded by :class:`AdvancedCombatAI`, kept working.
- :class:`BattleMLAgent` (legacy) — heuristic/utility scorer over card roles
  and elixir state, kept as a fallback when no frame-based decision fires.
- :class:`AdvancedCombatAI` — combo engine (cross-lane kiting, push sync,
  locked-lane offense) reading card data from the global registry
  (:mod:`clbot.bot.card_database`).

Coordinates come from :mod:`clbot.bot.coords` (fixed 419x633 resolution);
this module holds no raw pixel positions.
"""

from __future__ import annotations

import random
import time

import cv2
import numpy as np

from clbot.bot.card_database import (
    ROLE_BUILDING,
    ROLE_CYCLE,
    ROLE_SPELL_BUFF,
    ROLE_SPELL_DMG,
    ROLE_SUPPORT,
    ROLE_WIN_CON,
    ROLE_SWARM as COMBAT_SWARM,
    ROLE_TANK as COMBAT_TANK,
    TARGET_BACK,
    TARGET_BRIDGE,
    TARGET_DEFENSE_PULL,
    TARGET_KING_CYCLE,
    TARGET_POCKET,
    TARGET_TOWER,
    TARGET_THREAT,
    lookup_card,
    resolve_card,
)
from clbot.bot.card_detection import PLAY_COORDS
from clbot.bot.coords import (
    KITE_PULL_LEFT_THREAT,
    ML_CENTER_POCKET,
    ML_ENEMY_LEFT_TOWER,
    ML_ENEMY_RIGHT_TOWER,
    ML_FALLBACK_POCKET,
    ML_KING_CENTER,
    ML_LEFT_BACK,
    ML_LEFT_BRIDGE,
    ML_RIGHT_BACK,
    ML_RIGHT_BRIDGE,
    TACT_CENTER_ANTI_AIR,
    TACT_CENTER_PULL,
    TACT_ENEMY_LEFT_TOWER,
    TACT_ENEMY_RIGHT_TOWER,
    TACT_KING_TOWER,
    TACT_LEFT_BACK,
    TACT_LEFT_BRIDGE,
    TACT_RIGHT_BACK,
    TACT_RIGHT_BRIDGE,
)

ROLE_WIN_CONDITION = "win_condition"
ROLE_TANK = "tank"
ROLE_SWARM = "swarm"
ROLE_SUPPORT_RANGED = "ranged"
ROLE_SPELL = "spell"
ROLE_DEFENSIVE_BUILDING = "building"
ROLE_UNKNOWN = "unknown"

def _legacy_role_for_card(card_name: str) -> str:
    """Map the global registry role to the legacy BattleMLAgent role names."""
    meta = lookup_card(card_name)
    if meta is None:
        return ROLE_UNKNOWN
    role = meta.get("role")
    return {
        COMBAT_TANK: ROLE_TANK,
        ROLE_WIN_CON: ROLE_WIN_CONDITION,
        COMBAT_SWARM: ROLE_SWARM,
        ROLE_SUPPORT: ROLE_SUPPORT_RANGED,
        ROLE_BUILDING: ROLE_DEFENSIVE_BUILDING,
        ROLE_SPELL_DMG: ROLE_SPELL,
        ROLE_SPELL_BUFF: ROLE_SPELL,
        ROLE_CYCLE: ROLE_SWARM,
    }.get(role, ROLE_UNKNOWN)


_LEGACY_ROLE_UTILITY = {
    ROLE_WIN_CONDITION: 0.90,
    ROLE_TANK: 0.82,
    ROLE_SUPPORT_RANGED: 0.72,
    ROLE_SWARM: 0.66,
    ROLE_SPELL: 0.62,
    ROLE_DEFENSIVE_BUILDING: 0.60,
    ROLE_UNKNOWN: 0.10,
}


class BattleMLAgent:
    """Utility-scoring agent for card choice + placement.

    Weights over ``[elixir_readiness, role_utility, counter_potential,
    lane_positioning]``. Deterministic scoring; lane side ties are broken
    with :mod:`random` (seedable in tests).
    """

    def __init__(self) -> None:
        # [Elixir Advantage, Aggression Factor, Defense Urgency, Synergy]
        self.weights = np.array([0.35, 0.25, 0.25, 0.15])

    def classify_card(self, card_name: str) -> str:
        """Return the tactical role using the global card registry."""
        return _legacy_role_for_card(card_name)

    def placement_for_role(
        self,
        role: str,
        card_name: str = "",
        side_preference: str | None = None,
    ) -> tuple[int, int]:
        """Return tactical placement coords for a role.

        Args:
            role: One of the ``ROLE_*`` constants.
            card_name: Original card id (used for the rage-vs-damage spell tweak).
            side_preference: ``"left"``/``"right"`` to force a lane, else random.
        """
        _ = card_name  # reserved for per-card tweaks (e.g. rage follows the push)
        pick_right = random.choice([True, False]) if side_preference is None else side_preference == "right"

        if role == ROLE_TANK:
            return ML_RIGHT_BACK if pick_right else ML_LEFT_BACK
        if role == ROLE_WIN_CONDITION:
            return ML_RIGHT_BRIDGE if pick_right else ML_LEFT_BRIDGE
        if role == ROLE_SUPPORT_RANGED:
            return ML_KING_CENTER
        if role == ROLE_SPELL:
            return ML_ENEMY_RIGHT_TOWER if pick_right else ML_ENEMY_LEFT_TOWER
        if role == ROLE_SWARM:
            return ML_CENTER_POCKET
        if role == ROLE_DEFENSIVE_BUILDING:
            return ML_KING_CENTER
        return ML_FALLBACK_POCKET

    def evaluate_best_play(
        self,
        available_cards: list[dict],
        current_elixir: int,
        match_time_elapsed: float,
        side_preference: str | None = None,
    ) -> tuple[int, tuple[int, int], int] | None:
        """Return the best affordable legacy-policy move.

        This is the deterministic baseline used when no learned checkpoint is
        installed.  It now uses actual registry costs instead of role-level
        pseudo-costs, so an unaffordable card is never deliberately selected.
        """
        if not available_cards:
            return None
        elixir = max(0, min(10, int(current_elixir)))
        phase_bonus = 0.10 if match_time_elapsed >= 120.0 else 0.0
        best_score = float("-inf")
        best_play: tuple[int, tuple[int, int], int] | None = None

        for card in available_cards:
            try:
                index = int(card["index"])
                name = str(card["name"])
            except (KeyError, TypeError, ValueError):
                continue
            meta = lookup_card(name)
            if meta is None:
                continue
            role = self.classify_card(name)
            cost = meta.get("cost")
            if not isinstance(cost, (int, float)):
                continue
            cost = int(cost)
            if cost > elixir:
                continue

            readiness = (elixir - cost + 1) / 10.0
            efficiency = 1.0 - cost / 12.0
            role_utility = _LEGACY_ROLE_UTILITY.get(role, 0.1)
            score = (
                0.35 * readiness
                + 0.30 * role_utility
                + 0.20 * efficiency
                + 0.15 * phase_bonus
            )
            if score > best_score:
                best_score = score
                coords = self.placement_for_role(role, name, side_preference)
                best_play = (index, coords, cost)
        return best_play


def get_current_elixir(emulator, max_elixir: int = 10) -> int:
    """Estimate current elixir (0-10) by scanning elixir pips top-down.

    Uses :func:`clbot.bot.state_detect.count_elixir` so the BGR/pixel
    logic stays in one place. Returns 0 when no pip matches.
    """
    from clbot.bot.state_detect import count_elixir

    for amount in range(max_elixir, 0, -1):
        try:
            if count_elixir(emulator, amount):
                return amount
        except (IndexError, TypeError):
            continue
    return 0


# --- Threat-aware tactical engine -------------------------------------------

# Card archetypes for the threat-aware policy.
TYPE_WIN_CON = "WIN_CONDITION"
TYPE_TANK = "TANK"
TYPE_SPLASH = "SPLASH_RANGED"
TYPE_SWARM = "SWARM"
TYPE_SPELL_DAMAGE = "SPELL_DAMAGE"
TYPE_SPELL_BUFF = "SPELL_BUFF"
TYPE_BUILDING = "BUILDING"
TYPE_CHEAP_CYCLE = "CYCLE"

# Placement anchors for the threat-aware policy (all named coords).
TACT_TILES: dict[str, tuple[int, int]] = {
    "left_bridge": TACT_LEFT_BRIDGE,
    "right_bridge": TACT_RIGHT_BRIDGE,
    "left_back": TACT_LEFT_BACK,
    "right_back": TACT_RIGHT_BACK,
    "center_pull": TACT_CENTER_PULL,
    "center_anti_air": TACT_CENTER_ANTI_AIR,
    "king_tower": TACT_KING_TOWER,
    "enemy_left_tower": TACT_ENEMY_LEFT_TOWER,
    "enemy_right_tower": TACT_ENEMY_RIGHT_TOWER,
}

# Offense is locked until elixir banks this high (bar reads 0-10 ints).
OFFENSE_ELIXIR_GATE = 9


class VisionThreatDetector:
    """Detect active enemy troops via red health bars on the mid-field.

    The scan window is deliberately tight (y 44-76%, x 14-86%) so static
    red arena dressing — tower roofs, flags, banners — and our own hand
    row can never read as a "push". Candidates must also be bright red
    and wider than tall (health-bar shaped); anything else is noise.
    Operates on a BGR frame (raw ``emulator.screenshot()`` channel order).
    Never raises — vision failures report "no threat" so the fight loop
    holds elixir instead of misfiring.
    """

    @staticmethod
    def detect_enemy_threats(
        screen_image: np.ndarray | None,
    ) -> tuple[bool, str, tuple[int, int], int]:
        """Return (has_threat, lane, threat_center, count).

        Lane is ``"left"``/``"right"``/``"none"``; ``threat_center`` is the
        (x, y) of the push closest to our towers in full-frame coordinates.
        """
        if screen_image is None:
            return False, "none", TACT_TILES["center_pull"], 0
        try:
            if screen_image.ndim != 3 or screen_image.shape[2] != 3:
                return False, "none", TACT_TILES["center_pull"], 0
            h, w = screen_image.shape[:2]
            if h <= 0 or w <= 0:
                return False, "none", TACT_TILES["center_pull"], 0

            # Mid-field playfield only — excludes tower roofs (top) and
            # our hand row (bottom), where static red objects live.
            crop_y1, crop_y2 = int(h * 0.44), int(h * 0.76)
            crop_x1, crop_x2 = int(w * 0.14), int(w * 0.86)
            if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
                return False, "none", TACT_TILES["center_pull"], 0
            playfield = screen_image[crop_y1:crop_y2, crop_x1:crop_x2]

            hsv = cv2.cvtColor(playfield, cv2.COLOR_BGR2HSV)

            # Bright-red health bars only (both hue wrap sides; S/V floor
            # sits above dim red dressing but below real bars, which render
            # near-full saturation on emulator captures).
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 130, 130]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([170, 130, 130]), np.array([180, 255, 255])),
            )
            # Drop isolated noise pixels before contouring.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_threats: list[tuple[int, int]] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Health-bar sized blobs (contourArea runs ~1px under pixel
                # area, so the lower bound is inclusive).
                if 12 <= area <= 250:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if float(bw) / max(bh, 1) >= 1.2:  # bars run wide, not tall
                        valid_threats.append((x + bw // 2 + crop_x1, y + bh // 2 + crop_y1))

            if not valid_threats:
                return False, "none", TACT_TILES["center_pull"], 0

            # Push closest to our towers first.
            valid_threats.sort(key=lambda pt: pt[1], reverse=True)
            primary_threat = valid_threats[0]
            lane = "left" if primary_threat[0] < (w // 2) else "right"
            return True, lane, primary_threat, len(valid_threats)
        except Exception:
            return False, "none", TACT_TILES["center_pull"], 0


class TacticalAIEngine:
    """Threat-aware play policy: defend first, bank elixir, push at >= 9."""

    def __init__(self) -> None:
        self.threat_detector = VisionThreatDetector()

    def get_card_meta(self, card_name: str) -> dict[str, object]:
        """Return tactical metadata from the single global card registry."""
        meta = lookup_card(card_name)
        if meta is None:
            # Preserve the old public fallback for callers/tests, but unknown
            # identities must still be rejected by the live learned-action validator.
            return {"type": TYPE_SPLASH, "cost": 3, "target": "defense"}
        role = meta.get("role")
        role_to_type = {
            ROLE_WIN_CON: TYPE_WIN_CON,
            COMBAT_TANK: TYPE_TANK,
            COMBAT_SWARM: TYPE_SWARM,
            ROLE_SUPPORT: TYPE_SPLASH,
            ROLE_BUILDING: TYPE_BUILDING,
            ROLE_SPELL_DMG: TYPE_SPELL_DAMAGE,
            ROLE_SPELL_BUFF: TYPE_SPELL_BUFF,
            ROLE_CYCLE: TYPE_CHEAP_CYCLE,
        }
        target_map = {
            TARGET_BRIDGE: "bridge",
            TARGET_BACK: "back",
            TARGET_DEFENSE_PULL: "defense",
            TARGET_POCKET: "defense",
            TARGET_TOWER: "tower",
            TARGET_THREAT: "threat",
            TARGET_KING_CYCLE: "center",
        }
        return {
            "type": role_to_type.get(role, TYPE_SPLASH),
            "cost": int(meta.get("cost", 3)),
            "target": target_map.get(str(meta.get("target")), "defense"),
        }

    def select_best_move(
        self,
        screen_frame: np.ndarray | None,
        available_cards: list[dict],
        current_elixir: int,
        match_time_elapsed: float,
    ) -> tuple[int, tuple[int, int], str] | None:
        """Pick the tactical play, or ``None`` to hold elixir.

        Args:
            screen_frame: Live BGR frame for threat detection.
            available_cards: ``[{'index': int, 'name': str}, ...]``.
            current_elixir: Current elixir bar (0-10).
            match_time_elapsed: Seconds since battle start (reserved for
                future double-elixir aggression; currently unused).

        Returns:
            ``(card_index, (x, y), reason)`` or ``None`` when holding.
        """
        _ = match_time_elapsed
        has_threat, _threat_lane, threat_coord, threat_count = self.threat_detector.detect_enemy_threats(screen_frame)

        cards_with_meta: list[dict] = []
        for card in available_cards:
            try:
                index = int(card["index"])
                name = str(card["name"])
            except (KeyError, TypeError, ValueError):
                continue
            cards_with_meta.append({"index": index, "name": name, **self.get_card_meta(name)})

        def affordable(card: dict) -> bool:
            try:
                return current_elixir >= int(card["cost"])
            except (KeyError, TypeError, ValueError):
                return False

        # 1. Defensive reaction (priority #1).
        if has_threat:
            if threat_count >= 2:
                # Swarm/cluster: splash pulls to the pocket first ...
                for card in cards_with_meta:
                    if card["type"] == TYPE_SPLASH and affordable(card):
                        return int(card["index"]), TACT_TILES["center_pull"], "Defending swarm with splash"
                # ... then a damage spell straight onto the cluster.
                for card in cards_with_meta:
                    if card["type"] == TYPE_SPELL_DAMAGE and affordable(card):
                        return int(card["index"]), threat_coord, "Sniping cluster with spell"
            # Single heavy threat: kite it into the center pocket.
            for card in cards_with_meta:
                if card["type"] in (TYPE_SWARM, TYPE_TANK, TYPE_SPLASH, TYPE_BUILDING) and affordable(card):
                    return int(card["index"]), TACT_TILES["center_pull"], "Pulling threat to center"
            # Nothing affordable — wait for elixir, never force a bad play.
            return None

        # 2. Offensive initiation — only safe (no threat) and banked (>= 9).
        if current_elixir >= OFFENSE_ELIXIR_GATE:
            for card in cards_with_meta:
                if card["type"] == TYPE_TANK and affordable(card):
                    back_tile = random.choice([TACT_TILES["left_back"], TACT_TILES["right_back"]])
                    return int(card["index"]), back_tile, "Building slow push in back"
            for card in cards_with_meta:
                if card["type"] == TYPE_WIN_CON and affordable(card):
                    bridge_tile = random.choice([TACT_TILES["left_bridge"], TACT_TILES["right_bridge"]])
                    return int(card["index"]), bridge_tile, "Attacking tower with win condition"
            for card in cards_with_meta:
                if card["type"] in (TYPE_CHEAP_CYCLE, TYPE_SPLASH) and affordable(card):
                    try:
                        cheap = int(card["cost"]) <= 3
                    except (TypeError, ValueError):
                        cheap = False
                    if cheap:
                        return int(card["index"]), TACT_TILES["king_tower"], "Cycling cheap card"

        # 3. Hold elixir.
        return None


# --- High-winrate strategic combat AI ---------------------------------------
# Roles, costs, and targets come from the global registry
# (bot/card_database.py). The legacy BattleMLAgent ROLE_TANK / ROLE_SWARM
# bindings above are untouched; the registry's same-named roles are
# imported as COMBAT_TANK / COMBAT_SWARM to avoid clobbering them.

# Siege win conditions play from their own tiles, never the bridge.
_SIEGE_IDS = frozenset({"xbow", "mortar"})
# Tower-target win conditions with dedicated tile groups in PLAY_COORDS.
_TOWER_TILE_GROUPS = frozenset({"goblin_barrel", "miner", "graveyard", "goblin_drill"})


def _base_card_id(card_name: object) -> str:
    """Normalize a detector id for tile-group lookup (strips evo_/hero_)."""
    key = str(card_name).lower().replace(" ", "_").replace("-", "_")
    for prefix in ("evo_", "hero_"):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def grid_coord(group_name: str, lane: str | None) -> tuple[int, int]:
    """Pick a random tile from a PLAY_COORDS group for a lane.

    Unknown groups or lanes fall back to back_support/left, so a bad key
    can never produce a missing or unplayable coordinate.
    """
    group = PLAY_COORDS.get(group_name) or PLAY_COORDS["back_support"]
    tiles = group.get(lane or "") or group.get("left") or next(iter(group.values()))
    return random.choice(tiles)


class AdvancedCombatAI:
    """Combo engine: cross-lane kiting, push sync, weakest-lane lock-on.

    Threat sensing reuses :class:`VisionThreatDetector` so both policies
    share one tuned HSV threshold set. Placement resolves through the
    tuned ``PLAY_COORDS`` grid in :mod:`clbot.bot.card_detection`
    (``grid_coord``) — never fixed single anchors. Target lane defaults
    to ``"left"`` and stays locked — spreading damage across two healthy
    towers loses games. Call :meth:`reset` when a new battle starts so
    tank-walk tracking never leaks across matches.
    """

    # Anti-freeze: longest gap between deploys before the failsafe fires.
    FAILSAFE_IDLE_S = 3.5
    # Failsafe only second-guesses low-elixir stalls; banked hands (>= 6)
    # are governed by the offense gate below, never overridden.
    FAILSAFE_ELIXIR_MAX = 6

    def __init__(self) -> None:
        self.target_lane: str | None = None
        self.active_push_lane: str | None = None
        self.tank_start_time: float = 0.0
        self.is_tank_walking: bool = False
        self.last_deploy_time: float = time.time()

    def reset(self) -> None:
        """Clear per-battle push tracking (call on battle start)."""
        self.active_push_lane = None
        self.tank_start_time = 0.0
        self.is_tank_walking = False
        self.last_deploy_time = time.time()

    def get_meta(self, card_name: str) -> dict[str, object]:
        """Return ``{"role", "cost", "target"}`` from the global registry."""
        return resolve_card(card_name)

    def detect_board_state(self, screen: np.ndarray | None) -> tuple[bool, str, tuple[int, int], int, str]:
        """Return (has_threat, threat_lane, threat_pos, threat_count, target_lane).

        Lanes are ``"left"``/``"right"``/``"none"``; ``threat_pos`` is the
        push closest to our towers in full-frame coordinates.
        """
        if self.target_lane is None:
            self.target_lane = "left"
        if screen is None:
            return False, "none", KITE_PULL_LEFT_THREAT, 0, self.target_lane
        has_threat, lane, pos, count = VisionThreatDetector.detect_enemy_threats(screen)
        if not has_threat:
            return False, "none", KITE_PULL_LEFT_THREAT, 0, self.target_lane
        return has_threat, lane, pos, count, self.target_lane

    def plan_move(
        self,
        screen: np.ndarray | None,
        hand_cards: list[dict],
        elixir: int,
        elapsed_sec: float,
    ) -> tuple[int, tuple[int, int], str] | None:
        """Run the decision tree; ``None`` means hold elixir."""
        has_threat, threat_lane, threat_pos, threat_count, target_lane = self.detect_board_state(screen)

        cards: list[dict] = []
        for card in hand_cards:
            try:
                index = int(card["index"])
                name = str(card["name"])
            except (KeyError, TypeError, ValueError):
                continue
            cards.append({"index": index, "name": name, **self.get_meta(name)})

        def affordable(card: dict) -> bool:
            try:
                return elixir >= int(card["cost"])
            except (KeyError, TypeError, ValueError):
                return False

        def cost_of(card: dict) -> int:
            try:
                return int(card["cost"])
            except (KeyError, TypeError, ValueError):
                return 99

        def deploy(index: int, coords: tuple[int, int], reason: str) -> tuple[int, tuple[int, int], str]:
            self.last_deploy_time = time.time()
            return index, coords, reason

        _ = elapsed_sec  # reserved for future double-elixir aggression tuning.

        # 0. Anti-freeze failsafe: a long stall with low elixir means the
        # vision/elixir pipeline is lying (ghost lock) — force the cheapest
        # troop at the bridge rather than sit out the match. Spells and
        # buildings are excluded: blind spells/buildings lose more than
        # they save, and an unaffordable tap is a harmless no-op.
        if time.time() - self.last_deploy_time > self.FAILSAFE_IDLE_S and elixir < self.FAILSAFE_ELIXIR_MAX:
            troops = [card for card in cards if card["role"] not in (ROLE_SPELL_DMG, ROLE_SPELL_BUFF, ROLE_BUILDING)]
            if troops:
                cheapest = min(troops, key=cost_of)
                return deploy(
                    int(cheapest["index"]),
                    grid_coord("bridge_rush", target_lane),
                    f"Failsafe anti-freeze play: {cheapest.get('name', 'card')}",
                )

        # Expire stale tank walks (a push that never got support is over).
        if self.is_tank_walking and (time.time() - self.tank_start_time > 14):
            self.is_tank_walking = False

        # 1. Immediate defense (highest priority) on the tuned grid tiles.
        if has_threat:
            if threat_count >= 2:
                # Cluster: damage spell straight onto it first ...
                for card in cards:
                    if card["role"] == ROLE_SPELL_DMG and affordable(card):
                        return deploy(int(card["index"]), threat_pos, f"Spell defense on cluster at {threat_pos}")
                # ... then splash in the back field ...
                for card in cards:
                    if card["role"] == ROLE_SUPPORT and affordable(card):
                        return deploy(
                            int(card["index"]),
                            grid_coord("back_support", threat_lane),
                            "Placing splash to wipe swarm",
                        )
                # ... then a building on the defensive grid.
                for card in cards:
                    if card["role"] == ROLE_BUILDING and affordable(card):
                        return deploy(
                            int(card["index"]),
                            grid_coord("defense_building", threat_lane),
                            "Planting building to absorb swarm",
                        )
            # Single threat: kite troops to the center tiles, buildings
            # onto the defensive grid ...
            for card in cards:
                if card["role"] in (COMBAT_SWARM, COMBAT_TANK) and affordable(card):
                    return deploy(
                        int(card["index"]),
                        grid_coord("spirit", threat_lane),
                        f"Cross-lane kiting {threat_lane} threat",
                    )
            for card in cards:
                if card["role"] == ROLE_BUILDING and affordable(card):
                    return deploy(
                        int(card["index"]),
                        grid_coord("defense_building", threat_lane),
                        f"Cross-lane kiting {threat_lane} threat",
                    )
            # ... or ranged support in the back-field pocket when no
            # swarm/tank/building is held.
            for card in cards:
                if card["role"] == ROLE_SUPPORT and affordable(card):
                    return deploy(
                        int(card["index"]),
                        grid_coord("back_support", threat_lane),
                        "Support defense in pocket",
                    )
            # Nothing affordable — wait for the right counter, never force it.
            return None

        # 2. Push synergy: tank is walking, drop support behind it at the bridge.
        if self.is_tank_walking:
            walk_time = time.time() - self.tank_start_time
            if 4 <= walk_time <= 9:
                for card in cards:
                    if card["role"] in (ROLE_SUPPORT, ROLE_WIN_CON) and affordable(card):
                        self.is_tank_walking = False
                        return deploy(
                            int(card["index"]),
                            grid_coord("bridge_rush", self.active_push_lane),
                            "Following up tank push with support",
                        )

        # 3. Proactive offense in the locked lane (elixir >= 6).
        if elixir >= 6:
            for card in cards:
                if card["role"] == COMBAT_TANK and affordable(card):
                    self.is_tank_walking = True
                    self.tank_start_time = time.time()
                    self.active_push_lane = target_lane
                    return deploy(
                        int(card["index"]),
                        grid_coord("king_lane", target_lane),
                        f"Starting Tank push in {target_lane} lane",
                    )
            for card in cards:
                if card["role"] == ROLE_WIN_CON and affordable(card):
                    base_id = _base_card_id(card.get("name", ""))
                    if base_id in _SIEGE_IDS:
                        return deploy(
                            int(card["index"]),
                            grid_coord("siege_building", target_lane),
                            f"Sieging from {target_lane} side",
                        )
                    if card.get("target") == TARGET_TOWER and base_id in _TOWER_TILE_GROUPS:
                        return deploy(
                            int(card["index"]),
                            grid_coord(base_id, target_lane),
                            f"Direct Tower attack down {target_lane} lane",
                        )
                    return deploy(
                        int(card["index"]),
                        grid_coord("bridge_rush", target_lane),
                        f"Sending Win Condition down {target_lane} lane",
                    )
            for card in cards:
                try:
                    cheap = int(card["cost"]) <= 3
                except (TypeError, ValueError):
                    cheap = False
                if card["role"] in (ROLE_CYCLE, ROLE_SUPPORT, COMBAT_SWARM) and cheap and affordable(card):
                    group = "princess" if card["role"] == ROLE_CYCLE else "bridge_rush"
                    return deploy(int(card["index"]), grid_coord(group, target_lane), "Proactive push/cycle")

        # Hold elixir.
        return None
