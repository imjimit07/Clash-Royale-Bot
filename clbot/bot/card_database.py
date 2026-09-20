"""Global Clash Royale card registry.

Single source of truth mapping every card id to its elixir cost, tactical
role, and placement anchor kind. Consumers look cards up with
:func:`resolve_card`, which sanitizes names, follows :data:`CARD_ALIASES`,
and tolerates ``evo_``/``hero_`` prefixes via substring match.
"""

from __future__ import annotations

# Archetype roles.
ROLE_WIN_CON = "WIN_CONDITION"
ROLE_TANK = "TANK"
ROLE_SUPPORT = "SUPPORT_SPLASH"
ROLE_SWARM = "SWARM"
ROLE_BUILDING = "BUILDING"
ROLE_SPELL_DMG = "SPELL_DAMAGE"
ROLE_SPELL_BUFF = "SPELL_BUFF"
ROLE_CYCLE = "CYCLE"

# Strategic placement anchor kinds (resolved to screen coords by the caller).
TARGET_BRIDGE = "bridge"
TARGET_BACK = "back"
TARGET_DEFENSE_PULL = "center_pull"
TARGET_POCKET = "center_pocket"
TARGET_TOWER = "enemy_tower"
TARGET_THREAT = "threat"
TARGET_KING_CYCLE = "king_back"

_ALL_ROLES = frozenset(
    {
        ROLE_WIN_CON,
        ROLE_TANK,
        ROLE_SUPPORT,
        ROLE_SWARM,
        ROLE_BUILDING,
        ROLE_SPELL_DMG,
        ROLE_SPELL_BUFF,
        ROLE_CYCLE,
    }
)

ALL_CARDS: dict[str, dict[str, object]] = {
    # 1. Win conditions (tower rushers / primary damage).
    "hog_rider": {"role": ROLE_WIN_CON, "cost": 4, "target": TARGET_BRIDGE},
    "royal_giant": {"role": ROLE_WIN_CON, "cost": 6, "target": TARGET_BRIDGE},
    "balloon": {"role": ROLE_WIN_CON, "cost": 5, "target": TARGET_BRIDGE},
    "goblin_barrel": {"role": ROLE_WIN_CON, "cost": 3, "target": TARGET_TOWER},
    "ram_rider": {"role": ROLE_WIN_CON, "cost": 5, "target": TARGET_BRIDGE},
    "battle_ram": {"role": ROLE_WIN_CON, "cost": 4, "target": TARGET_BRIDGE},
    "wall_breakers": {"role": ROLE_WIN_CON, "cost": 2, "target": TARGET_BRIDGE},
    "miner": {"role": ROLE_WIN_CON, "cost": 3, "target": TARGET_TOWER},
    "goblin_drill": {"role": ROLE_WIN_CON, "cost": 4, "target": TARGET_TOWER},
    "skeleton_barrel": {"role": ROLE_WIN_CON, "cost": 3, "target": TARGET_BRIDGE},
    "royal_hogs": {"role": ROLE_WIN_CON, "cost": 5, "target": TARGET_BRIDGE},
    "elixir_golem": {"role": ROLE_WIN_CON, "cost": 3, "target": TARGET_BRIDGE},
    "graveyard": {"role": ROLE_WIN_CON, "cost": 5, "target": TARGET_TOWER},
    "lava_hound": {"role": ROLE_WIN_CON, "cost": 7, "target": TARGET_BACK},
    "x_bow": {"role": ROLE_WIN_CON, "cost": 6, "target": TARGET_BRIDGE},
    "mortar": {"role": ROLE_WIN_CON, "cost": 4, "target": TARGET_BRIDGE},
    "goblin_giant": {"role": ROLE_WIN_CON, "cost": 6, "target": TARGET_BACK},
    "electro_giant": {"role": ROLE_WIN_CON, "cost": 7, "target": TARGET_BACK},
    # 2. Tanks & mini-tanks (high-HP frontline / pullers).
    "giant": {"role": ROLE_TANK, "cost": 5, "target": TARGET_BACK},
    "golem": {"role": ROLE_TANK, "cost": 8, "target": TARGET_BACK},
    "pekka": {"role": ROLE_TANK, "cost": 7, "target": TARGET_DEFENSE_PULL},
    "mega_knight": {"role": ROLE_TANK, "cost": 7, "target": TARGET_DEFENSE_PULL},
    "giant_skeleton": {"role": ROLE_TANK, "cost": 6, "target": TARGET_DEFENSE_PULL},
    "knight": {"role": ROLE_TANK, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "valkyrie": {"role": ROLE_TANK, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "mini_pekka": {"role": ROLE_TANK, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "prince": {"role": ROLE_TANK, "cost": 5, "target": TARGET_DEFENSE_PULL},
    "dark_prince": {"role": ROLE_TANK, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "ice_golem": {"role": ROLE_TANK, "cost": 2, "target": TARGET_DEFENSE_PULL},
    "royal_ghost": {"role": ROLE_TANK, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "bandit": {"role": ROLE_TANK, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "lumberjack": {"role": ROLE_TANK, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "fisherman": {"role": ROLE_TANK, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "cannon_cart": {"role": ROLE_TANK, "cost": 5, "target": TARGET_DEFENSE_PULL},
    "bowler": {"role": ROLE_TANK, "cost": 5, "target": TARGET_DEFENSE_PULL},
    # 3. Champions & special heroes.
    "skeleton_king": {"role": ROLE_TANK, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "golden_knight": {"role": ROLE_TANK, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "mighty_miner": {"role": ROLE_TANK, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "archer_queen": {"role": ROLE_SUPPORT, "cost": 5, "target": TARGET_POCKET},
    "monk": {"role": ROLE_TANK, "cost": 5, "target": TARGET_DEFENSE_PULL},
    "little_prince": {"role": ROLE_SUPPORT, "cost": 3, "target": TARGET_POCKET},
    "goblin_machine": {"role": ROLE_TANK, "cost": 5, "target": TARGET_DEFENSE_PULL},
    # 4. Support, ranged, splash & air units.
    "musketeer": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "three_musketeers": {"role": ROLE_SUPPORT, "cost": 9, "target": TARGET_KING_CYCLE},
    "wizard": {"role": ROLE_SUPPORT, "cost": 5, "target": TARGET_POCKET},
    "ice_wizard": {"role": ROLE_SUPPORT, "cost": 3, "target": TARGET_POCKET},
    "electro_wizard": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "magic_archer": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "princess": {"role": ROLE_SUPPORT, "cost": 3, "target": TARGET_KING_CYCLE},
    "firecracker": {"role": ROLE_SUPPORT, "cost": 3, "target": TARGET_POCKET},
    "dart_goblin": {"role": ROLE_SUPPORT, "cost": 3, "target": TARGET_POCKET},
    "bomber": {"role": ROLE_SUPPORT, "cost": 2, "target": TARGET_POCKET},
    "archers": {"role": ROLE_SUPPORT, "cost": 3, "target": TARGET_POCKET},
    "spear_goblins": {"role": ROLE_SUPPORT, "cost": 2, "target": TARGET_POCKET},
    "witch": {"role": ROLE_SUPPORT, "cost": 5, "target": TARGET_POCKET},
    "night_witch": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_BACK},
    "mother_witch": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "executioner": {"role": ROLE_SUPPORT, "cost": 5, "target": TARGET_POCKET},
    "hunter": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "sparky": {"role": ROLE_SUPPORT, "cost": 6, "target": TARGET_BACK},
    "zappies": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "flying_machine": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "baby_dragon": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "inferno_dragon": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "electro_dragon": {"role": ROLE_SUPPORT, "cost": 5, "target": TARGET_POCKET},
    "skeleton_dragons": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "phoenix": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "mega_minion": {"role": ROLE_SUPPORT, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "goblin_demolisher": {"role": ROLE_SUPPORT, "cost": 4, "target": TARGET_POCKET},
    "suspicious_bush": {"role": ROLE_SUPPORT, "cost": 2, "target": TARGET_BRIDGE},
    "battle_healer": {"role": ROLE_SUPPORT, "cost": 3, "target": TARGET_POCKET},
    # 5. Swarms & squad cards.
    "skeleton_army": {"role": ROLE_SWARM, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "goblins": {"role": ROLE_SWARM, "cost": 2, "target": TARGET_DEFENSE_PULL},
    "bats": {"role": ROLE_SWARM, "cost": 2, "target": TARGET_DEFENSE_PULL},
    "minions": {"role": ROLE_SWARM, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "minion_horde": {"role": ROLE_SWARM, "cost": 5, "target": TARGET_DEFENSE_PULL},
    "guards": {"role": ROLE_SWARM, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "goblin_gang": {"role": ROLE_SWARM, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "barbarians": {"role": ROLE_SWARM, "cost": 5, "target": TARGET_DEFENSE_PULL},
    "elite_barbarians": {"role": ROLE_TANK, "cost": 6, "target": TARGET_DEFENSE_PULL},
    "rascals": {"role": ROLE_SWARM, "cost": 5, "target": TARGET_DEFENSE_PULL},
    "royal_recruits": {"role": ROLE_SWARM, "cost": 7, "target": TARGET_DEFENSE_PULL},
    # 6. Cheap cycles (1-elixir spirits & skeletons).
    "skeletons": {"role": ROLE_CYCLE, "cost": 1, "target": TARGET_KING_CYCLE},
    "ice_spirit": {"role": ROLE_CYCLE, "cost": 1, "target": TARGET_KING_CYCLE},
    "fire_spirit": {"role": ROLE_CYCLE, "cost": 1, "target": TARGET_KING_CYCLE},
    "electro_spirit": {"role": ROLE_CYCLE, "cost": 1, "target": TARGET_KING_CYCLE},
    "heal_spirit": {"role": ROLE_CYCLE, "cost": 1, "target": TARGET_KING_CYCLE},
    # 7. Buildings & spawners.
    "cannon": {"role": ROLE_BUILDING, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "tesla": {"role": ROLE_BUILDING, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "inferno_tower": {"role": ROLE_BUILDING, "cost": 5, "target": TARGET_DEFENSE_PULL},
    "bomb_tower": {"role": ROLE_BUILDING, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "tombstone": {"role": ROLE_BUILDING, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "goblin_cage": {"role": ROLE_BUILDING, "cost": 4, "target": TARGET_DEFENSE_PULL},
    "furnace": {"role": ROLE_BUILDING, "cost": 4, "target": TARGET_POCKET},
    "goblin_hut": {"role": ROLE_BUILDING, "cost": 5, "target": TARGET_POCKET},
    "barbarian_hut": {"role": ROLE_BUILDING, "cost": 6, "target": TARGET_POCKET},
    "elixir_collector": {"role": ROLE_BUILDING, "cost": 6, "target": TARGET_KING_CYCLE},
    # 8. Damage spells.
    "zap": {"role": ROLE_SPELL_DMG, "cost": 2, "target": TARGET_THREAT},
    "the_log": {"role": ROLE_SPELL_DMG, "cost": 2, "target": TARGET_THREAT},
    "arrows": {"role": ROLE_SPELL_DMG, "cost": 3, "target": TARGET_THREAT},
    "fireball": {"role": ROLE_SPELL_DMG, "cost": 4, "target": TARGET_THREAT},
    "poison": {"role": ROLE_SPELL_DMG, "cost": 4, "target": TARGET_THREAT},
    "rocket": {"role": ROLE_SPELL_DMG, "cost": 6, "target": TARGET_TOWER},
    "lightning": {"role": ROLE_SPELL_DMG, "cost": 6, "target": TARGET_THREAT},
    "tornado": {"role": ROLE_SPELL_DMG, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "giant_snowball": {"role": ROLE_SPELL_DMG, "cost": 2, "target": TARGET_THREAT},
    "earthquake": {"role": ROLE_SPELL_DMG, "cost": 3, "target": TARGET_TOWER},
    "barbarian_barrel": {"role": ROLE_SPELL_DMG, "cost": 2, "target": TARGET_THREAT},
    "royal_delivery": {"role": ROLE_SPELL_DMG, "cost": 3, "target": TARGET_DEFENSE_PULL},
    "void": {"role": ROLE_SPELL_DMG, "cost": 3, "target": TARGET_THREAT},
    "goblin_curse": {"role": ROLE_SPELL_DMG, "cost": 2, "target": TARGET_THREAT},
    # 9. Buff / special spells.
    "rage": {"role": ROLE_SPELL_BUFF, "cost": 2, "target": TARGET_BRIDGE},
    "freeze": {"role": ROLE_SPELL_BUFF, "cost": 4, "target": TARGET_TOWER},
    "clone": {"role": ROLE_SPELL_BUFF, "cost": 3, "target": TARGET_BRIDGE},
    "mirror": {"role": ROLE_SPELL_BUFF, "cost": 1, "target": TARGET_KING_CYCLE},
}

# Aliases: detector ids / OCR variants / shorthand -> registry keys.
CARD_ALIASES: dict[str, str] = {
    "hog": "hog_rider",
    "xbow": "x_bow",
    "barb_barrel": "barbarian_barrel",
    "barb_hut": "barbarian_hut",
    "skarmy": "skeleton_army",
    "log": "the_log",
    "fire_cracker": "firecracker",
    "snowball": "giant_snowball",
    "ebarbs": "elite_barbarians",
    "p_e_k_k_a": "pekka",
    "mini_p_e_k_k_a": "mini_pekka",
    "3m": "three_musketeers",
    "delivery": "royal_delivery",
    "collector": "elixir_collector",
    "pump": "elixir_collector",
}

_FALLBACK_META: dict[str, object] = {"role": ROLE_SUPPORT, "cost": 3, "target": TARGET_POCKET}


def normalize_card_id(card_name: str) -> str:
    """Normalize detector/OCR ids without applying the unknown fallback."""
    clean = str(card_name or "").lower().strip().replace(" ", "_").replace("-", "_")
    for prefix in ("evo_", "hero_"):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
            break
    return CARD_ALIASES.get(clean, clean)


def lookup_card(card_name: str) -> dict[str, object] | None:
    """Return registry metadata for a known card, else ``None``."""
    clean = normalize_card_id(card_name)
    if not clean:
        return None
    if clean in ALL_CARDS:
        return dict(ALL_CARDS[clean])
    # Preserve support for detector variants that contain the canonical id.
    for key, data in ALL_CARDS.items():
        if key in clean:
            return dict(data)
    return None


__all__ = ["lookup_card", "resolve_card", "normalize_card_id", "ALL_CARDS", "CARD_ALIASES"]


def resolve_card(card_name: str) -> dict[str, object]:
    """Look up ``{"role", "cost", "target"}`` with name sanitation.

    Follows :data:`CARD_ALIASES`, then exact match, then substring match
    (covers ``evo_``/``hero_`` detector prefixes), then a safe fallback.
    Returned dicts are copies — callers may merge freely.
    """
    meta = lookup_card(card_name)
    return meta if meta is not None else dict(_FALLBACK_META)
