from __future__ import annotations

from enum import StrEnum


class StatField(StrEnum):
    WINS = "wins"
    LOSSES = "losses"
    CARDS_PLAYED = "cards_played"
    CLASSIC_1V1_FIGHTS = "classic_1v1_fights"
    CLASSIC_2V2_FIGHTS = "classic_2v2_fights"
    TROPHY_ROAD_1V1_FIGHTS = "trophy_road_1v1_fights"
    CARD_RANDOMIZATIONS = "card_randomizations"
    CARD_CYCLES = "card_cycles"


class DerivedStatField(StrEnum):
    WINRATE = "winrate"
    CURRENT_WIN_STREAK = "current_win_streak"
    BEST_WIN_STREAK = "best_win_streak"


class BotStatField(StrEnum):
    RESTARTS_AFTER_FAILURE = "restarts_after_failure"
    TIME_SINCE_START = "time_since_start"


class UIField(StrEnum):
    CLASSIC_1V1_USER_TOGGLE = "classic_1v1_user_toggle"
    CLASSIC_2V2_USER_TOGGLE = "classic_2v2_user_toggle"
    TROPHY_ROAD_USER_TOGGLE = "trophy_road_user_toggle"
    RANDOM_DECKS_USER_TOGGLE = "random_decks_user_toggle"
    DECK_NUMBER_SELECTION = "deck_number_selection"
    CYCLE_DECKS_USER_TOGGLE = "cycle_decks_user_toggle"
    MAX_DECK_SELECTION = "max_deck_selection"
    RANDOM_PLAYS_USER_TOGGLE = "random_plays_user_toggle"
    DISABLE_WIN_TRACK_TOGGLE = "disable_win_track_toggle"
    RECORD_FIGHTS_TOGGLE = "record_fights_toggle"
    RECORDING_FOLDER_PATH = "recording_folder_path"
    GOOGLE_PLAY_EMULATOR_TOGGLE = "google_play_emulator_toggle"
    DISCORD_RPC_TOGGLE = "discord_rpc_toggle"
    GP_ANGLE = "gp_angle"
    GP_VULKAN = "gp_vulkan"
    GP_GLES = "gp_gles"
    GP_SURFACELESS = "gp_surfaceless"
    GP_EGL = "gp_egl"
    GP_BACKEND = "gp_backend"
    GP_WSI = "gp_wsi"
    GP_DEVICE_SERIAL = "gp_device_serial"


WIN_RATE_STAT_LABELS: dict[StatField, str] = {
    StatField.WINS: "Wins",
    StatField.LOSSES: "Losses",
}

WIN_RATE_STAT_FIELDS: tuple[StatField, ...] = tuple(WIN_RATE_STAT_LABELS.keys())

BATTLE_STAT_LABELS: dict[StatField, str] = {
    StatField.CARDS_PLAYED: "Cards played",
    StatField.CLASSIC_1V1_FIGHTS: "Classic 1v1",
    StatField.CLASSIC_2V2_FIGHTS: "Classic 2v2",
    StatField.TROPHY_ROAD_1V1_FIGHTS: "Trophy Road",
    StatField.CARD_RANDOMIZATIONS: "Decks randomized",
    StatField.CARD_CYCLES: "Decks cycled",
}

BATTLE_STAT_FIELDS: tuple[StatField, ...] = tuple(BATTLE_STAT_LABELS.keys())

COLLECTION_STAT_LABELS: dict[StatField, str] = {}

COLLECTION_STAT_FIELDS: tuple[StatField, ...] = tuple(COLLECTION_STAT_LABELS.keys())

BOT_STAT_LABELS: dict[BotStatField, str] = {
    BotStatField.RESTARTS_AFTER_FAILURE: "Recovery restarts",
    BotStatField.TIME_SINCE_START: "Runtime",
}

BOT_STAT_FIELDS: tuple[BotStatField, ...] = tuple(BOT_STAT_LABELS.keys())

# Jobs that count toward starting the bot — battle-only (1v1, 2v2, Trophy Road).
START_JOB_GROUPS: tuple[tuple[UIField, ...], ...] = (
    (
        UIField.CLASSIC_1V1_USER_TOGGLE,
        UIField.CLASSIC_2V2_USER_TOGGLE,
        UIField.TROPHY_ROAD_USER_TOGGLE,
    ),
)


def has_start_ready_job(values: dict[str, object]) -> bool:
    """True when at least one battle job is enabled."""
    return any(any(bool(values.get(field.value, False)) for field in group) for group in START_JOB_GROUPS)
