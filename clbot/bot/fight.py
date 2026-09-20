"""random module for randomizing fight plays"""

import collections
import os
import random
import time
from typing import Literal

from clbot.bot.card_database import lookup_card, resolve_card
from clbot.bot.card_detection import (
    check_which_cards_are_available,
    create_default_bridge_iar,
    get_play_coords_for_card,
    identify_hand_cards,
    is_hero_champion_ability_visible,
    trigger_hero_champion_ability,
)
from clbot.bot.coords import (
    CLOSE_BATTLE_LOG_BUTTON,
    EMOTE_BUTTON_COORD,
    EMOTE_ICON_COORDS,
    HAND_CARDS_COORDS,
    PLAYABLE_PLAY_REGION_LTRB,
    QUICKMATCH_POPUP_BUTTON_COORD,
    START_FIGHT_BUTTON_COORD,
)
from clbot.bot.elixir import ElixirScanner, ElixirTracker
from clbot.bot.ml_agent import AdvancedCombatAI, BattleMLAgent, TacticalAIEngine, get_current_elixir
from clbot.bot.ml.inference import LearnedPolicy
from clbot.bot.ml.state import EnemyThreat, GameState, HandCard
from clbot.bot.nav import (
    check_for_in_battle_with_delay,
    get_to_activity_log,
    get_to_main_after_fight,
    wait_for_battle_start,
    wait_for_clash_main_menu,
)
from clbot.bot.recorder import (
    finish_fight_recording,
    is_recording,
    log_decision,
    log_play,
    start_fight_recording,
    stop_fight_capture,
)
from clbot.bot.state_detect import (
    check_if_battle_has_ended,
    check_if_in_battle,
    check_if_on_clash_main_menu,
    check_pixels_for_win_in_battle_log,
    count_elixir,
)
from clbot.utils.logger import Logger
from clbot.utils.versioning import __version__

ELIXIR_WAIT_TIMEOUT = 20
ELIXIR_POLL_INTERVAL = 0.3
ABILITY_TRIGGER_DELAY_S = 3

_ML_AGENT = BattleMLAgent()
_TACTICAL_AI = TacticalAIEngine()
_COMBAT_AI = AdvancedCombatAI()
_ELIXIR_TRACKER = ElixirTracker()
_LEARNED_POLICY = LearnedPolicy()
_recent_card_names: collections.deque[str] = collections.deque(maxlen=3)
_last_hold_record_time = 0.0
_last_hold_record_key: tuple | None = None
_VERIFY_DEPLOYS = os.getenv("CLBOT_VERIFY_DEPLOYS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _read_elixir_optional(frame):
    """Support the new optional scanner API and old/test scanner doubles."""
    reader = getattr(ElixirScanner, "read_elixir_optional", None)
    if reader is not None:
        return reader(frame)
    value = ElixirScanner.read_elixir(frame)
    return int(value) if value is not None else None


def _maybe_start_fight_recording(
    emulator, logger, recording_flag: bool, fight_mode_chosen: str, custom_path: str | None = None
) -> None:
    """Begin opt-in training-data capture for 1v1-type fights only (Trophy Road / Classic 1v1, never 2v2)."""
    if recording_flag and fight_mode_chosen in ["Classic 1v1", "Trophy Road"]:
        pack_mode = "1v1_trophy" if fight_mode_chosen == "Trophy Road" else "1v1_classic"
        start_fight_recording(emulator, pack_mode, __version__, logger=logger, custom_path=custom_path)


def do_fight_state(
    emulator,
    logger: Logger,
    random_fight_mode,
    fight_mode_chosen,
    called_from_launching=False,
    recording_flag: bool = False,
    custom_path: str | None = None,
) -> bool:
    """Handle the entirety of a battle state (start fight, do fight, end fight)."""

    logger.change_status("Waiting for battle to start")

    # Wait for battle start
    if wait_for_battle_start(emulator, logger) is False:
        logger.change_status("Timed out waiting for battle to start")
        return False

    logger.change_status("Starting fight loop")
    logger.log(f'This is the fight mode: "{fight_mode_chosen}"')

    # Recording is started/stopped inside the fight loops themselves so each pack
    # brackets exactly the in-battle play loop (no pre-battle wait or post-fight nav).
    # Run regular fight loop if random mode not toggled
    if not random_fight_mode and _fight_loop(emulator, logger, recording_flag, fight_mode_chosen, custom_path) is False:
        logger.change_status("Fight loop failed")
        return False

    # Run random fight loop if random mode toggled
    if (
        random_fight_mode
        and _random_fight_loop(emulator, logger, recording_flag, fight_mode_chosen, custom_path) is False
    ):
        logger.change_status("Fight loop failed")
        return False

    # Only log the fight if not called from the start
    if not called_from_launching:
        if fight_mode_chosen in ["Classic 1v1", "Trophy Road"]:
            logger.add_1v1_fight()
        elif fight_mode_chosen == "Classic 2v2":
            logger.increment_2v2_fights()

        if fight_mode_chosen == "Trophy Road":
            logger.increment_trophy_road_fights()
        elif fight_mode_chosen == "Classic 1v1":
            logger.increment_classic_1v1_fights()
        elif fight_mode_chosen == "Classic 2v2":
            logger.increment_classic_2v2_fights()

    time.sleep(10)
    return True


def start_fight(emulator, logger, mode) -> bool:
    """Start a fight with the specified mode.

    Args:
        emulator: The emulator controller
        logger: Logger instance
        mode: Fight mode - must be one of "Classic 1v1", "Classic 2v2", or "Trophy Road"

    Returns:
        bool: True if fight started successfully, False otherwise
    """
    # Validate mode parameter
    logger.log(f'Input mode type: "{type(mode)}"')
    logger.log(f"Input mode value: {mode}")
    valid_modes = ["Classic 1v1", "Classic 2v2", "Trophy Road"]
    logger.log(f"Valid modes: {valid_modes}")
    if mode not in valid_modes:
        logger.log(f"The valid modes for start_fight() are: {valid_modes}")
        logger.log(f"But start_fight() got an invalid mode: '{mode}'")
        return False

    logger.change_status(f"Starting a {mode} fight")

    # Check if on clash main menu
    logger.log("Checking if on main menu before starting fight...")
    if not check_if_on_clash_main_menu(emulator):
        logger.change_status("Not on main menu — cannot start fight")
        return False

    # For all modes (1v1 and 2v2), use the same start button
    # Mode is already set by select_mode() in states.py, just click start button
    emulator.click(START_FIGHT_BUTTON_COORD[0], START_FIGHT_BUTTON_COORD[1])
    logger.log(f"Clicked Start button at {START_FIGHT_BUTTON_COORD}")

    # 2v2 needs a second popup after Start
    if mode == "Classic 2v2":
        logger.change_status("Classic 2v2 — clicking Quick Match popup...")
        time.sleep(3)
        emulator.click(QUICKMATCH_POPUP_BUTTON_COORD[0], QUICKMATCH_POPUP_BUTTON_COORD[1])
        logger.log(f"Clicked Quickmatch button at {QUICKMATCH_POPUP_BUTTON_COORD}")

    return True


def send_emote(emulator, logger: Logger):
    """Method to do an emote in a fight"""
    logger.change_status("Sending emote")

    # click emote button
    emulator.click(EMOTE_BUTTON_COORD[0], EMOTE_BUTTON_COORD[1])
    time.sleep(0.33)

    emote_coord = random.choice(EMOTE_ICON_COORDS)
    emulator.click(emote_coord[0], emote_coord[1])


RANDOM_PLAY_ELIXIR_MIN = 3
RANDOM_PLAY_ELIXIR_MAX = 9


def play_random_available_card(emulator, logger, recording_flag: bool, elapsed_s: float) -> bool:
    """Play one card that is actually available, at a random friendly-half coord.

    Mirrors the rl-bot RandomPlayer so recorded plays are never polluted with cards
    that didn't deploy: read the available hand slots (affordable, non-empty) and, if
    any, play a random one; if none are available, do nothing. Returns True if a card
    was played. The caller gates this behind a random elixir wait.
    """
    card_indices = check_which_cards_are_available(emulator, check_side=True)
    if not card_indices:
        return False  # nothing playable -> no play, nothing recorded (caller re-rolls)

    card_index = random.choice(card_indices)
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    play_coord = (random.randint(left, right), random.randint(top, bottom))

    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])
    time.sleep(0.1)
    emulator.click(play_coord[0], play_coord[1])
    time.sleep(0.1)

    if recording_flag:
        # Card identity isn't computed for random plays; home re-derives it from pixels.
        log_play(card_index, play_coord[0], play_coord[1], elapsed_s)
    logger.add_card_played()
    return True


def wait_for_elixir(
    emulator,
    logger,
    elixir_wait_amount,
    WAIT_THRESHOLD=5000,  # noqa: N803
    PLAY_THRESHOLD=10000,  # noqa: N803
    recording_flag: bool = False,
) -> Literal["restart", "no battle"] | bool:
    """Wait until the target elixir count is reached.

    Pacing fix: premature pixel-activity triggers ("Battle too active — playing
    now" / "All cards are available!") are intentionally disabled. The old
    ``switch_side`` offset check aborted the wait after 0-4s whenever troops
    moved on screen, so the bot never accumulated to its planned 7-9 elixir
    push. The wait is now purely elixir-gated (plus timeout / battle-end
    handling). ``WAIT_THRESHOLD``/``PLAY_THRESHOLD`` are kept for backwards
    compatibility with existing callers but are no longer consulted.
    """
    _ = (WAIT_THRESHOLD, PLAY_THRESHOLD, recording_flag)
    start_time = time.time()
    battle_detection_lost_count = 0
    last_logged_second = -1
    last_lost_detection_log_second = -1
    last_threat_check = 0.0
    ability_available_since = None

    while True:
        match_elapsed = time.time() - start_time
        try:
            frame_now = emulator.screenshot()
            _ELIXIR_TRACKER.accrue(time.time(), match_elapsed)
            observed_now = _read_elixir_optional(frame_now)
            if observed_now is not None:
                _ELIXIR_TRACKER.correct(observed_now)
            observed_amount = _effective_elixir(observed_now, match_elapsed)
        except Exception:
            observed_amount = int(round(_ELIXIR_TRACKER.estimate))
        if observed_amount >= elixir_wait_amount:
            break
        # debug screenshot saving removed from production
        wait_time = time.time() - start_time
        elapsed_second = int(wait_time)
        if elapsed_second != last_logged_second:
            logger.change_status(
                f"Waiting for {elixir_wait_amount} elixir for {elapsed_second}s...",
            )
            last_logged_second = elapsed_second

        if is_hero_champion_ability_visible(emulator):
            if ability_available_since is None:
                ability_available_since = time.time()
                logger.change_status(
                    f"Hero/Champion ability ready — triggering in {ABILITY_TRIGGER_DELAY_S}s",
                )
            elif time.time() - ability_available_since >= ABILITY_TRIGGER_DELAY_S:
                trigger_hero_champion_ability(emulator, logger)
                ability_available_since = None
        else:
            ability_available_since = None

        if wait_time > ELIXIR_WAIT_TIMEOUT:
            logger.change_status(status="Waited too long for elixir")
            return "restart"

        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                logger.change_status(status="Battle ended — stopping elixir wait")
                return "no battle"

            battle_detection_lost_count += 1
            lost_detection_second = int(time.time())
            if lost_detection_second != last_lost_detection_log_second:
                logger.change_status(
                    status="Lost battle detection while waiting for elixir — assuming still in battle",
                )
                last_lost_detection_log_second = lost_detection_second
            if battle_detection_lost_count >= 4:
                logger.change_status(
                    status="Lost battle detection repeatedly — assuming battle ended",
                )
                return "no battle"

            time.sleep(0.5)
            continue

        battle_detection_lost_count = 0

        # Defense urgency break: a confirmed push with an affordable answer
        # interrupts banking — but only for vision-confirmed threats, never
        # bare pixel activity (see docstring). Checked at most once a second
        # to bound screenshot cost.
        if time.time() - last_threat_check >= 1.0:
            last_threat_check = time.time()
            try:
                frame = emulator.screenshot()
                threat, _lane, _pos, _count, _locked = _COMBAT_AI.detect_board_state(frame)
                answers = check_which_cards_are_available(emulator) if threat else []
            except Exception:
                threat, answers = False, []
            if threat and answers:
                logger.change_status("Breaking elixir wait to defend the push")
                return True

        # Throttle screenshot rate while elixir accumulates (each check above
        # screenshots). Keeps the wait purely elixir-gated without ADB spam.
        time.sleep(ELIXIR_POLL_INTERVAL)

    logger.change_status(
        f"Took {str(time.time() - start_time)[:4]}s for {elixir_wait_amount} elixir.",
    )

    return True


def wait_for_elixir_with_timeout(scanner, target, logger, start_time, timeout: float = ELIXIR_WAIT_TIMEOUT):
    """Poll a scanner with clock fallback until target elixir or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        elapsed = time.time() - start_time
        try:
            current = scanner.get_elixir_with_fallback(elapsed)
        except Exception:
            current = 0
        if current is not None and current >= target:
            return True
        time.sleep(ELIXIR_POLL_INTERVAL)
    try:
        logger.log(f"Elixir wait timed out (target={target})")
    except Exception:
        pass
    return False


def end_fight_state(
    emulator,
    logger: Logger,
    recording_flag,
    disable_win_tracker_toggle=True,
):
    """Method to handle the time after a fight and before the next state"""
    # count the crown score on this end-battle screen

    # get to clash main after this fight
    logger.log("Returning to main menu after fight")
    if get_to_main_after_fight(emulator, logger) is False:
        logger.log("Failed to return to main menu after fight")
        finish_fight_recording(None)
        return False

    logger.log("Returned to main menu after fight")
    time.sleep(3)

    # Determine the outcome. Force the win check when a recording is active so the
    # pack gets a real win/loss; otherwise honor the user's win-tracker toggle.
    if is_recording() or not disable_win_tracker_toggle:
        win_check_return = check_if_previous_game_was_win(emulator, logger)

        if win_check_return == "restart":
            logger.log("Failed while checking if previous game was a win")
            finish_fight_recording(None)
            return False

        outcome = "win" if win_check_return else "loss"

        # Only touch the user's win/loss stats when their tracker is enabled.
        if not disable_win_tracker_toggle:
            if win_check_return:
                logger.add_win()
            else:
                logger.add_loss()

        finish_fight_recording(outcome)
    else:
        logger.log("Not checking win/loss because check is disabled")
        finish_fight_recording(None)

    return True


def check_if_previous_game_was_win(
    emulator,
    logger: Logger,
) -> bool | Literal["restart"]:
    """Method to handle the checking if the previous game was a win or loss"""
    logger.change_status(status="Checking last game result")

    # Use wait_for_clash_main_menu to ensure we are on the main menu.
    if not wait_for_clash_main_menu(emulator, logger, deadspace_click=True):
        logger.change_status(status="Not on main menu — cannot check last game result")
        return "restart"

    # get to clash main options menu
    if get_to_activity_log(emulator, logger, printmode=False) == "restart":
        logger.change_status(status="Failed to open battle log")

        return "restart"

    logger.change_status(status="Checking battle log for win...")
    is_a_win = check_pixels_for_win_in_battle_log(emulator)
    result = "win" if is_a_win else "loss"
    logger.change_status(status=f"Last game result: {result}")

    # close battle log
    logger.change_status(status="Returning to main menu")
    emulator.click(CLOSE_BATTLE_LOG_BUTTON[0], CLOSE_BATTLE_LOG_BUTTON[1])
    if wait_for_clash_main_menu(emulator, logger) is False:
        logger.change_status(status="Timed out returning to main menu after battle log")
        return "restart"
    time.sleep(2)

    return is_a_win


# main fight loops

# Initialize a deque with a maximum length of 3 to store the last three chosen cards
last_three_cards = collections.deque(maxlen=3)


def select_card_index(card_indices, last_three_cards):
    if not card_indices:
        raise ValueError("card_indices cannot be empty")

    # First preference: Cards not in the last_three_cards queue
    preferred_cards = [index for index in card_indices if index not in last_three_cards]

    # Second preference: Cards not among the last two added to the queue
    if not preferred_cards and len(last_three_cards) == 3:
        preferred_cards = [index for index in card_indices if index not in list(last_three_cards)[-2:]]

    # Third preference: Any card except the most recently added one
    if not preferred_cards and last_three_cards:
        preferred_cards = [index for index in card_indices if index != last_three_cards[-1]]

    # Fallback: If all else fails, consider all cards
    if not preferred_cards:
        preferred_cards = card_indices

    return random.choice(preferred_cards)


def _effective_elixir(raw_elixir: int | None, elapsed_s: float) -> int:
    """Use a valid bar reading; otherwise use the clock tracker estimate."""
    _ = elapsed_s
    if raw_elixir is not None:
        return max(0, min(10, int(raw_elixir)))
    return max(0, min(10, int(round(_ELIXIR_TRACKER.estimate))))


def _deduct_played_cost(card_index: int, hand: list[dict]) -> float | None:
    """Account for a confirmed deployment using the canonical card registry."""
    try:
        name = next(str(card["name"]) for card in hand if int(card.get("index", -1)) == int(card_index))
    except (StopIteration, KeyError, TypeError, ValueError):
        return None
    meta = lookup_card(name)
    cost = meta.get("cost") if meta is not None else None
    if isinstance(cost, (int, float)):
        _ELIXIR_TRACKER.deduct(cost)
        return float(cost)
    return None


def _hand_slot_roi(frame, card_index: int):
    if frame is None or getattr(frame, "ndim", 0) != 3:
        return None
    try:
        cx, cy = HAND_CARDS_COORDS[int(card_index)]
    except (IndexError, TypeError, ValueError):
        return None
    h, w = frame.shape[:2]
    x1, x2 = max(0, cx - 28), min(w, cx + 28)
    y1, y2 = max(0, cy - 34), min(h, cy + 20)
    return frame[y1:y2, x1:x2]


def _confirm_deployment(
    emulator,
    selected_frame,
    card_index: int,
    expected_cost: float | None,
    *,
    before_frame=None,
) -> bool:
    """Confirm a click changed the selected hand slot or reduced visible elixir."""
    if not _VERIFY_DEPLOYS:
        return True
    before_roi = _hand_slot_roi(selected_frame, card_index)
    before_elixir = _read_elixir_optional(before_frame)
    deadline = time.time() + 0.55
    while time.time() < deadline:
        try:
            frame = emulator.screenshot()
        except Exception:
            frame = None
        if frame is not None:
            after_roi = _hand_slot_roi(frame, card_index)
            if selected_frame is not None and before_roi is not None and after_roi is not None and before_roi.shape == after_roi.shape:
                try:
                    import numpy as _np
                    delta_selected = float(
                        _np.mean(_np.abs(after_roi.astype(_np.int16) - before_roi.astype(_np.int16)))
                    )
                    original_roi = _hand_slot_roi(before_frame, card_index)
                    delta_original = 0.0
                    if original_roi is not None and original_roi.shape == after_roi.shape:
                        delta_original = float(
                            _np.mean(_np.abs(after_roi.astype(_np.int16) - original_roi.astype(_np.int16)))
                        )
                    # A failed placement often just deselects the card, which makes
                    # the slot differ from the selected frame. Requiring a change
                    # from both selected and original states reduces that false positive.
                    if delta_selected >= 4.0 and delta_original >= 6.0:
                        return True
                except Exception:
                    pass
            after_elixir = _read_elixir_optional(frame)
            if before_elixir is not None and after_elixir is not None and expected_cost is not None:
                if after_elixir <= max(0, int(before_elixir - expected_cost + 0.5)):
                    return True
        time.sleep(0.08)
    return False


def _build_game_state(frame, hand: list[dict], elixir: int | float, elapsed: float) -> GameState:
    has_threat, lane, pos, count, target_lane = _COMBAT_AI.detect_board_state(frame)
    h, w = frame.shape[:2] if frame is not None else (633, 419)
    threat = EnemyThreat(
        count=count if has_threat else 0,
        lane=lane if has_threat else "none",
        x=(pos[0] / max(1, w)) if has_threat else 0.5,
        y=(pos[1] / max(1, h)) if has_threat else 0.5,
    )
    state_hand = tuple(
        HandCard(index=int(card["index"]), name=str(card["name"]), confidence=float(card.get("confidence", 1.0)))
        for card in hand
        if card.get("name") not in {"UNKNOWN", "unknown", ""}
    )
    return GameState(
        elapsed_s=float(elapsed),
        elixir=float(elixir),
        hand=state_hand,
        enemy_threat=threat,
        target_lane=target_lane if target_lane in {"left", "right"} else "left",
        recent_cards=tuple(_recent_card_names),
        source="live",
    )


def _execute_confirmed_play(
    emulator, logger, card_index: int, x: int, y: int, hand: list[dict], elapsed: float,
    recording_flag: bool, reason: str, source: str, decision_confidence: float = 1.0,
    placement_confidence: float = 1.0, decision_elixir: float | None = None,
) -> bool:
    """Perform one deployment and mutate model state only after confirmation."""
    try:
        before_frame = emulator.screenshot()
    except Exception:
        before_frame = None
    try:
        meta_name = next(
            str(card.get("name")) for card in hand if int(card.get("index", -1)) == int(card_index)
        )
    except (StopIteration, TypeError, ValueError):
        return False
    meta = lookup_card(meta_name)
    if meta is None:
        logger.change_status(f"Unknown card identity {meta_name!r} — refusing deployment")
        return False
    expected_cost = meta.get("cost")
    cost = float(expected_cost) if isinstance(expected_cost, (int, float)) else None
    if cost is None:
        logger.change_status(f"No known elixir cost for {meta_name!r} — refusing deployment")
        return False
    recent_before = list(_recent_card_names)
    try:
        has_threat, pre_lane, pre_pos, pre_count, pre_target = _COMBAT_AI.detect_board_state(before_frame)
    except Exception:
        has_threat, pre_lane, pre_pos, pre_count, pre_target = False, "none", (0, 0), 0, (_COMBAT_AI.target_lane or "left")

    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])
    time.sleep(0.1)
    try:
        selected_frame = emulator.screenshot()
    except Exception:
        selected_frame = None
    emulator.click(int(x), int(y))

    if not _confirm_deployment(emulator, selected_frame, card_index, cost, before_frame=before_frame):
        logger.change_status(f"Deployment not confirmed for card {card_index} — state not updated")
        return False

    _deduct_played_cost(card_index, hand)
    if card_index not in last_three_cards:
        last_three_cards.append(card_index)
    _recent_card_names.append(meta_name)
    if recording_flag:
        try:
            log_play(
                card_index, x, y, elapsed, hand=hand,
                elixir=(float(decision_elixir) if decision_elixir is not None else None),
                threat={"count": pre_count if has_threat else 0, "lane": pre_lane if has_threat else "none",
                        "x": pre_pos[0] / max(1, before_frame.shape[1]) if before_frame is not None else 0.5,
                        "y": pre_pos[1] / max(1, before_frame.shape[0]) if before_frame is not None else 0.5},
                target_lane=pre_target, recent_cards=recent_before, policy_source=source,
                confidence=decision_confidence, placement_confidence=placement_confidence, confirmed=True,
            )
        except Exception as exc:
            logger.log(f"Training record state capture failed: {exc}")
    logger.add_card_played()
    logger.change_status(f"{source.title()} play: {reason} — card {card_index} at {(x, y)}")
    return True


def _record_hold_decision(
    logger,
    frame,
    hand: list[dict],
    elixir: float,
    elapsed: float,
    source: str,
    reason: str,
    confidence: float | None = None,
) -> None:
    global _last_hold_record_time, _last_hold_record_key
    if not is_recording():
        return
    hand_key = tuple(sorted((int(c.get("index", -1)), str(c.get("name", "unknown"))) for c in hand))
    try:
        has_threat, lane, pos, count, target = _COMBAT_AI.detect_board_state(frame)
        key = (source, round(float(elixir)), bool(has_threat), lane, target, hand_key)
        now = time.time()
        if key == _last_hold_record_key and now - _last_hold_record_time < 1.5:
            return
        _last_hold_record_key = key
        _last_hold_record_time = now
        h, w = frame.shape[:2]
        log_decision(
            elapsed,
            hand=hand,
            elixir=float(elixir),
            threat={"count": count if has_threat else 0, "lane": lane if has_threat else "none",
                    "x": pos[0] / max(1, w) if has_threat else 0.5,
                    "y": pos[1] / max(1, h) if has_threat else 0.5},
            target_lane=target,
            recent_cards=list(_recent_card_names),
            policy_source=source,
            confidence=confidence,
            reason=reason,
        )
    except Exception as exc:
        logger.log(f"Hold decision recording failed: {exc}")


def _try_tactical_play(
    emulator,
    logger,
    card_indices: list[int],
    battle_strategy: "BattleStrategy",
    recording_flag: bool,
) -> Literal["played", "hold", "fail"]:
    """Attempt one threat-aware tactical play.

    Returns ``"played"`` when a card was deployed, ``"hold"`` when the AI
    deliberately banks elixir (defense with nothing affordable, or no threat
    below the offensive gate), and ``"fail"`` when vision/identification
    broke down and the caller should fall back to the legacy rotation.
    """
    if not check_if_in_battle(emulator):
        return "fail"
    try:
        frame = emulator.screenshot()
    except Exception:
        return "fail"
    if frame is None:
        return "fail"
    # Fused estimate: game clock accrued + bar scan on the cached frame
    # (no extra pip screenshots), floored against 0-read deadlocks.
    elapsed = battle_strategy.get_elapsed_time()
    now = time.time()
    raw_elixir = _read_elixir_optional(frame)
    _ELIXIR_TRACKER.accrue(now, elapsed)
    if raw_elixir is not None:
        _ELIXIR_TRACKER.correct(raw_elixir)
    current_elixir = _effective_elixir(raw_elixir, elapsed)
    if raw_elixir is None:
        logger.log(f"Elixir scanner unavailable — using clock estimate {current_elixir}")
    elif current_elixir != raw_elixir:
        logger.log(f"Elixir scanner read {raw_elixir}, fused estimate {current_elixir}")

    hand: list[dict] = []
    for index in card_indices:
        try:
            name = identify_hand_cards(emulator, index)
        except Exception:
            continue
        if name and name != "UNKNOWN":
            hand.append({"index": index, "name": name})
    if not hand:
        return "fail"

    # Prefer variety (avoid the last-three) but fall back to the full hand.
    preferred = [c for c in hand if c["index"] not in last_three_cards] or hand
    try:
        decision = _COMBAT_AI.plan_move(
            screen=frame,
            hand_cards=preferred,
            elixir=current_elixir,
            elapsed_sec=elapsed,
        )
    except Exception as exc:
        logger.log(f"Tactical AI failed: {exc}")
        return "fail"

    if decision is None:
        try:
            has_threat, threat_lane, _coord, _count, _locked = _COMBAT_AI.detect_board_state(frame)
        except Exception:
            has_threat, threat_lane = False, "none"
        if has_threat:
            reason = f"hold against {threat_lane} push"
            logger.change_status(f"Holding for elixir against {threat_lane} push ({current_elixir}/10)")
        else:
            reason = "banking elixir — no threat"
            logger.change_status(f"Banking elixir ({current_elixir}/10) — no threat, no push yet")
        _record_hold_decision(logger, frame, hand, current_elixir, elapsed, "tactical", reason)
        return "hold"

    card_index, coords, reason = decision
    if card_index not in card_indices or not check_if_in_battle(emulator):
        return "fail"
    x, y = coords
    if not _execute_confirmed_play(
        emulator, logger, card_index, x, y, hand, elapsed, recording_flag, reason, "tactical",
        decision_elixir=float(current_elixir),
    ):
        return "fail"
    return "played"


def _try_ml_play(
    emulator, logger, card_indices: list[int], battle_strategy: "BattleStrategy", recording_flag: bool
) -> Literal["played", "hold", "fail"]:
    """Use the trained policy when a valid checkpoint is available."""
    if not _LEARNED_POLICY.available or not check_if_in_battle(emulator):
        return "fail"
    try:
        frame = emulator.screenshot()
    except Exception:
        return "fail"
    if frame is None:
        return "fail"
    elapsed = battle_strategy.get_elapsed_time()
    raw_elixir = _read_elixir_optional(frame)
    _ELIXIR_TRACKER.accrue(time.time(), elapsed)
    if raw_elixir is not None:
        _ELIXIR_TRACKER.correct(raw_elixir)
    current_elixir = _effective_elixir(raw_elixir, elapsed)

    hand: list[dict] = []
    for index in range(4):
        try:
            name = identify_hand_cards(emulator, index)
        except Exception:
            continue
        if name and name != "UNKNOWN":
            hand.append({"index": index, "name": name, "confidence": 1.0})
    if not hand or not card_indices:
        return "fail"

    state = _build_game_state(frame, hand, current_elixir, elapsed)
    decision = _LEARNED_POLICY.predict_decision(state)
    if decision.hold:
        logger.change_status(f"Learned policy holds ({decision.confidence:.2f})")
        _record_hold_decision(
            logger, frame, hand, current_elixir, elapsed, "learned", decision.reason, decision.confidence
        )
        return "hold"

    action = decision.action
    if action is None or action.card_index not in card_indices:
        logger.log(f"Learned policy did not produce a usable action: {decision.reason}")
        return "fail"
    valid, reason = _LEARNED_POLICY.validator.validate(action, state)
    if not valid:
        logger.log(f"Learned action rejected: {reason}")
        return "fail"
    played = _execute_confirmed_play(
        emulator, logger, action.card_index, action.x, action.y, hand, elapsed, recording_flag,
        f"confidence {action.confidence:.2f}/{action.placement_confidence:.2f}", "learned",
        action.confidence, action.placement_confidence, float(current_elixir),
    )
    return "played" if played else "fail"

def _try_legacy_ml_play(
    emulator, logger, card_indices: list[int], battle_strategy: "BattleStrategy", recording_flag: bool
) -> bool:
    """Deterministic heuristic scorer retained as the second fallback."""
    if not check_if_in_battle(emulator):
        return False
    try:
        frame = emulator.screenshot()
        raw_elixir = _read_elixir_optional(frame)
    except Exception:
        return False
    elapsed = battle_strategy.get_elapsed_time()
    _ELIXIR_TRACKER.accrue(time.time(), elapsed)
    if raw_elixir is not None:
        _ELIXIR_TRACKER.correct(raw_elixir)
    current_elixir = _effective_elixir(raw_elixir, elapsed)

    hand: list[dict] = []
    for index in range(4):
        try:
            name = identify_hand_cards(emulator, index)
        except Exception:
            continue
        if name and name != "UNKNOWN":
            hand.append({"index": index, "name": name, "confidence": 1.0})
    if not hand:
        return False

    preferred = [c for c in hand if c["index"] not in last_three_cards] or hand
    best = _ML_AGENT.evaluate_best_play(preferred, current_elixir, elapsed)
    if best is None:
        return False
    card_index, coords, cost = best
    if card_index not in card_indices or current_elixir < cost:
        return False
    return _execute_confirmed_play(
        emulator, logger, card_index, coords[0], coords[1], hand, elapsed, recording_flag,
        f"heuristic score, cost {cost}", "heuristic",
        decision_elixir=float(current_elixir),
    )

def play_a_card(emulator, logger, recording_flag: bool, battle_strategy: "BattleStrategy") -> bool:
    print("\n")

    # check which cards are available
    logger.change_status("Looking at which cards are available")
    available_card_check_start_time = time.time()
    card_indices = check_which_cards_are_available(emulator, check_side=True)

    if not card_indices:
        logger.change_status("No cards ready yet...")
        return False

    available_card_check_time_taken = str(
        time.time() - available_card_check_start_time,
    )[:3]

    logger.change_status(
        f"These cards are available: {card_indices} ({available_card_check_time_taken}s)",
    )

    # Learned policy gets first opportunity only when a checkpoint is installed.
    try:
        learned = _try_ml_play(emulator, logger, card_indices, battle_strategy, recording_flag)
        if learned == "played":
            if random.randint(0, 9) == 1:
                send_emote(emulator, logger)
            return True
        if learned == "hold":
            return True
    except Exception as exc:
        logger.log(f"Learned policy failed, falling back: {exc}")

    # Deterministic threat-aware engine remains the safety-oriented fallback.
    try:
        tactical = _try_tactical_play(emulator, logger, card_indices, battle_strategy, recording_flag)
    except Exception as exc:
        logger.log(f"Tactical AI failed, falling back: {exc}")
        tactical = "fail"
    if tactical == "played":
        if random.randint(0, 9) == 1:
            send_emote(emulator, logger)
        return True
    if tactical == "hold":
        return True

    # Heuristic scorer; finally fall back to the existing placement resolver.
    try:
        if _try_legacy_ml_play(emulator, logger, card_indices, battle_strategy, recording_flag):
            if random.randint(0, 9) == 1:
                send_emote(emulator, logger)
            return True
    except Exception as exc:
        logger.log(f"Legacy policy failed, falling back: {exc}")

    card_index = select_card_index(card_indices, last_three_cards)
    logger.change_status(f"Choosing this card index: {card_index}")

    # get a coord based on the selected side
    play_coord_calculation_start_time = time.time()
    card_id, play_coord = get_play_coords_for_card(emulator, logger, card_index, battle_strategy.get_elapsed_time())
    play_coord_calculation_time_taken = str(
        time.time() - play_coord_calculation_start_time,
    )[:3]

    logger.change_status(
        f"Calculated play for: {card_id} at {play_coord} ({play_coord_calculation_time_taken}s)",
    )

    if play_coord is None or card_index not in range(4):
        logger.change_status("Non-fatal error: play coordinates are invalid")
        return False
    play_start = time.time()
    hand_for_record: list[dict] = []
    try:
        name = identify_hand_cards(emulator, card_index)
    except Exception:
        name = "UNKNOWN"
    if name and name != "UNKNOWN":
        hand_for_record.append({"index": card_index, "name": name, "confidence": 1.0})
    else:
        # Legacy path cannot safely account or train from an unknown identity.
        hand_for_record.append({"index": card_index, "name": str(card_id), "confidence": 0.5})
    if not _execute_confirmed_play(
        emulator, logger, card_index, play_coord[0], play_coord[1], hand_for_record,
        battle_strategy.get_elapsed_time(), recording_flag, "legacy rotation", "legacy"
    ):
        return False
    click_and_play_card_time_taken = str(time.time() - play_start)[:3]
    logger.change_status(f"Made the play {click_and_play_card_time_taken}s")

    if random.randint(0, 9) == 1:
        send_emote(emulator, logger)
    return True


_PHASE_STRATEGIES = {
    "early": [0, 0, 0, 0, 0.3, 0.3, 0.4],  # 0-7s: Conservative, wait for more elixir
    "single": [0.05, 0.05, 0.1, 0.15, 0.15, 0.3, 0.2],  # 7-90s: Balanced distribution
    "double": [0.05, 0.05, 0.1, 0.15, 0.25, 0.3, 0.1],  # 90-200s: Favor 7-8 elixir
    "triple": [0.05, 0.05, 0.1, 0.1, 0.3, 0.4, 0],  # 200s+: Heavy favor 7-8, never 9
}

_PHASE_THRESHOLDS_1V1 = {
    "early": (6000, 9000),
    "single": (6000, 9000),
    "double": (7000, 10000),
    "triple": (8000, 11000),
}

_PHASE_THRESHOLDS_2V2 = {
    "early": (7000, 13000),
    "single": (7000, 14000),
    "double": (8000, 15000),
    "triple": (9000, 16000),
}


class BattleStrategy:
    """Manages battle timing and elixir selection strategy.

    Encapsulates the sophisticated elixir selection logic that changes
    based on battle phase, eliminating the need for global variables.
    """

    def __init__(self, fight_mode: str = "Classic 1v1"):
        self.start_time = None
        self.elixir_amounts = [3, 4, 5, 6, 7, 8, 9]

        is_2v2 = fight_mode == "Classic 2v2"
        self.phase_strategies = _PHASE_STRATEGIES
        self.phase_thresholds = _PHASE_THRESHOLDS_2V2 if is_2v2 else _PHASE_THRESHOLDS_1V1

    def start_battle(self):
        """Call when battle begins to start timing."""
        self.start_time = time.time()

    def get_elapsed_time(self):
        """Get seconds elapsed since battle start."""
        return time.time() - self.start_time if self.start_time else 0

    def get_battle_phase(self):
        """Determine current battle phase based on elapsed time."""
        elapsed = self.get_elapsed_time()
        if elapsed < 7:
            return "early"
        elif elapsed < 120:
            return "single"
        elif elapsed < 180:
            return "double"
        else:
            return "triple"

    def select_elixir_amount(self):
        """Select elixir amount to wait for based on current battle phase."""
        phase = self.get_battle_phase()
        weights = self.phase_strategies[phase]
        return random.choices(self.elixir_amounts, weights=weights, k=1)[0]

    def get_thresholds(self):
        """Get (WAIT_THRESHOLD, PLAY_THRESHOLD) for current battle phase."""
        phase = self.get_battle_phase()
        return self.phase_thresholds[phase]


def _fight_loop(
    emulator, logger: Logger, recording_flag: bool, fight_mode: str = "Classic 1v1", custom_path: str | None = None
) -> bool:
    """Method for handling dynamically timed fight"""
    global _last_hold_record_time, _last_hold_record_key
    create_default_bridge_iar(emulator)
    _maybe_start_fight_recording(emulator, logger, recording_flag, fight_mode, custom_path)
    collections.deque(maxlen=3)
    prev_cards_played = logger.get_cards_played()
    battle_detection_lost_count = 0

    # Initialize battle strategy and start timing
    battle_strategy = BattleStrategy(fight_mode)
    battle_strategy.start_battle()
    # Fresh push/lane tracking every battle — never leak a tank walk across matches.
    _COMBAT_AI.reset()
    last_three_cards.clear()
    _recent_card_names.clear()
    _LEARNED_POLICY.reset()
    _last_hold_record_time = 0.0
    _last_hold_record_key = None
    # Fresh elixir clock every battle — starts banked at 5 like a real match.
    _ELIXIR_TRACKER.start_match(time.time())

    while True:
        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                break

            battle_detection_lost_count += 1
            logger.change_status(
                f"Lost battle detection mid-fight ({battle_detection_lost_count}) — waiting it out",
            )

            # If we've lost detection several times in a row, assume the battle
            # ended even if we couldn't confirm it (prevents infinite loops if UI changes).
            if battle_detection_lost_count >= 4:
                logger.change_status(
                    "Lost battle detection repeatedly — assuming battle ended",
                )
                break

            time.sleep(1)
            continue

        battle_detection_lost_count = 0
        # debug screenshot saving removed from production

        # Get elixir amount and thresholds based on current battle phase
        elixir_amount = battle_strategy.select_elixir_amount()
        wait_threshold, play_threshold = battle_strategy.get_thresholds()

        wait_output = wait_for_elixir(
            emulator,
            logger,
            elixir_amount,
            wait_threshold,
            play_threshold,
            recording_flag,
        )

        if wait_output == "restart":
            logger.change_status("Failed while waiting for elixir")
            return False

        if wait_output == "no battle":
            logger.change_status("Not in battle anymore!")
            break

        if not check_if_in_battle(emulator):
            if check_if_battle_has_ended(emulator):
                logger.change_status("Battle ended (confirmed)")
                break

            logger.change_status("Lost battle detection — continuing fight loop")
            continue

        play_start_time = time.time()
        if play_a_card(emulator, logger, recording_flag, battle_strategy) is False:
            logger.change_status("Failed to play a card, retrying...")
        # play_time_taken = str(time.time() - play_start_time)[:4]
        logger.change_status(
            f"Made a play in {str(time.time() - play_start_time)[:4]}s",
        )

    # Fight over: freeze capture so the pack excludes post-fight nav (manifest/outcome written later).
    stop_fight_capture()
    logger.change_status("Fight complete")
    time.sleep(2.13)
    cards_played = logger.get_cards_played()
    logger.change_status(f"Played ~{cards_played - prev_cards_played} cards this fight")

    return True


def _random_fight_loop(
    emulator,
    logger,
    recording_flag: bool = False,
    fight_mode_chosen: str = "Trophy Road",
    custom_path: str | None = None,
) -> bool:
    """Method for handling dynamically timed fight with random plays"""
    logger.change_status(status="Starting battle with random plays")
    _maybe_start_fight_recording(emulator, logger, recording_flag, fight_mode_chosen, custom_path)
    create_default_bridge_iar(emulator)
    fight_timeout = 5 * 60  # 5 minutes
    start_time = time.time()
    battle_detection_lost_count = 0

    # while in battle:
    while True:
        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                break

            battle_detection_lost_count += 1
            logger.change_status(
                f"Lost battle detection mid-fight ({battle_detection_lost_count}) — waiting it out",
            )

            if battle_detection_lost_count >= 4:
                logger.change_status(
                    "Lost battle detection repeatedly — assuming battle ended",
                )
                break

            time.sleep(1)
            continue

        battle_detection_lost_count = 0
        if time.time() - start_time > fight_timeout:
            logger.change_status("Random fight loop timed out after 5 minutes")
            return False

        # Clean random-play flow (mirrors rl-bot RandomPlayer): wait for a random
        # elixir gate, then play only if a card is actually available.
        target = random.randint(RANDOM_PLAY_ELIXIR_MIN, RANDOM_PLAY_ELIXIR_MAX)
        elixir_result = wait_for_elixir(emulator, logger, target, recording_flag=recording_flag)
        if elixir_result == "no battle":
            break
        if elixir_result == "restart":
            return False
        play_random_available_card(emulator, logger, recording_flag, time.time() - start_time)

    # Fight over: freeze capture so the pack excludes post-fight nav (manifest/outcome written later).
    stop_fight_capture()
    logger.change_status("Random-plays fight complete")
    return True


if __name__ == "__main__":
    pass
