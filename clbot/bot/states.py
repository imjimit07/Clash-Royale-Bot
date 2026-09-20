"""time module for timing functions and controling pacing"""

import random
import time

from clbot.bot.deck import randomize_deck_state, select_deck_state
from clbot.bot.fight import (
    do_fight_state,
    end_fight_state,
    start_fight,
)
from clbot.bot.nav import select_mode
from clbot.bot.recording_archiver import maybe_archive_recordings
from clbot.bot.state_detect import check_if_battle_mode_is_selected
from clbot.interface.enums import UIField
from clbot.utils.caching import (
    get_deck_number_for_battle_mode,
    set_deck_number_for_battle_mode,
)
from clbot.utils.logger import Logger


def handle_state_failure(logger: Logger, state_name: str, function_name: str, error_msg: str | None = None) -> str:
    """Helper function to standardize error logging when states fail.

    Args:
        logger: The logger instance
        state_name: Name of the current state
        function_name: Name of the function that failed
        error_msg: Optional additional error message

    Returns:
        "restart" to trigger a restart
    """
    full_msg = f"State '{state_name}' failed in function '{function_name}'"
    if error_msg:
        full_msg += f": {error_msg}"

    logger.error(full_msg)
    logger.change_status(f"Error in {state_name} — restarting")
    print(f"[ERROR] {full_msg}")

    return "restart"


mode_used_in_1v1 = None
fight_mode_cycle_index = 0

# --- State-machine hardening (Part 1.2 / 1.6 / Part 8) ---
VALID_STATES = {
    "start",
    "restart",
    "select_battle_mode",
    "randomize_deck",
    "cycle_deck",
    "start_fight",
    "1v1_fight",
    "2v2_fight",
    "end_fight",
    "restart_emulator",
    "press_back_and_retry",
    "unknown",
    "idle",
    "fail",
}

# Per-state internal budgets (seconds). Long loops should bail to restart.
STATE_BUDGETS = {
    "select_battle_mode": 60,
    "randomize_deck": 60,
    "cycle_deck": 60,
    "start_fight": 45,
    "1v1_fight": 180,
    "2v2_fight": 180,
    "end_fight": 45,
}

ERROR_MESSAGES = {
    "black_frame": "Emulator render failure — check rendering mode (OpenGL/DirectX/Vulkan).",
    "adb_disconnect": "ADB connection lost — check emulator ADB port and firewall.",
    "elixir_not_found": "Elixir bar not detected — recalibrate HSV in config.yaml.",
    "unknown_screen": "Unrecognized screen — game may have updated; templates need refreshing.",
    "window_not_focused": "Emulator window lost focus — click on it and retry.",
}

UNKNOWN_SCREEN_LIMIT = 3
_unknown_screen_count = 0


def safe_next_state(state_order, current, logger=None):
    """Validate transitions; never propagate an invalid sentinel."""
    try:
        nxt = state_order.next_state(current)
    except Exception:
        return "restart"
    if nxt not in VALID_STATES:
        if logger is not None:
            try:
                logger.error(f"Invalid state transition {current} -> {nxt}")
            except Exception:
                pass
        return "restart"
    return nxt


def note_unknown_screen(logger=None) -> str:
    """Count consecutive unknown screens; escalate to back-and-retry."""
    global _unknown_screen_count
    _unknown_screen_count += 1
    if logger is not None:
        try:
            logger.log(f"Unknown screen (#{_unknown_screen_count})")
        except Exception:
            pass
    if _unknown_screen_count >= UNKNOWN_SCREEN_LIMIT:
        _unknown_screen_count = 0
        return "press_back_and_retry"
    return "unknown"


def note_known_screen() -> None:
    global _unknown_screen_count
    _unknown_screen_count = 0


def any_fight_mode_enabled(job_list) -> bool:
    return any(
        job_list.get(field.value, False)
        for field in (
            UIField.CLASSIC_1V1_USER_TOGGLE,
            UIField.CLASSIC_2V2_USER_TOGGLE,
            UIField.TROPHY_ROAD_USER_TOGGLE,
        )
    )


def get_next_fight_mode(job_list):
    """Get the next fight mode to use, cycling through enabled modes."""
    global fight_mode_cycle_index

    # Get all enabled fight modes
    enabled_modes = []
    if job_list.get(UIField.CLASSIC_1V1_USER_TOGGLE, False):
        enabled_modes.append("Classic 1v1")
    if job_list.get(UIField.CLASSIC_2V2_USER_TOGGLE, False):
        enabled_modes.append("Classic 2v2")
    if job_list.get(UIField.TROPHY_ROAD_USER_TOGGLE, False):
        enabled_modes.append("Trophy Road")

    if not enabled_modes:
        return None

    # Get the current mode and increment the cycle index
    current_mode = enabled_modes[fight_mode_cycle_index % len(enabled_modes)]
    fight_mode_cycle_index += 1

    return current_mode


class StateHistory:
    def __init__(self, logger):
        self.time_history_string_list = []
        self.logger = logger

        # This increment time is hard-coded to be as
        # low as possible while not spamming slow states

        # Battle-only bot: no throttled collection states.
        self.state2time_increment = {}
        self.randomize_state2time_increment()

    def randomize_state2time_increment(self):
        percent_diff = 40
        for state, time_increment in self.state2time_increment.items():
            adjustment_factor = random.randint(100 - percent_diff, 100 + percent_diff) / 100
            new_value = time_increment * adjustment_factor
            self.state2time_increment[state] = new_value

    def print_time_increments(self):
        def hours2readable(hours):
            def format_digit(digit):
                digit = str(digit)
                while len(digit) < 2:
                    digit = "0" + str(digit)

                return str(digit)

            remainder = hours * 60 * 60

            hours = int(remainder // 3600)
            remainder = remainder % 3600

            minutes = int(remainder // 60)
            remainder = remainder % 60

            seconds = int(remainder)

            return f"{format_digit(hours)}:{format_digit(minutes)}:{format_digit(seconds)}"

        for state, time_increment in self.state2time_increment.items():
            print(f"{state:>20} : {hours2readable(time_increment)}")

    def print(self):
        print("State history:")
        for i, line in enumerate(self.time_history_string_list):
            print("\t", i, line)
        print("\n")

    def add_state(self, state):
        time_history_string = f"{state} {time.time()} {int(self.logger.current_account)}"
        self.time_history_string_list.append(time_history_string)

    def get_time_of_last_state(self, state: str) -> int:
        most_recent_time = -1
        for line in self.time_history_string_list:
            # filter by state
            if state in line:
                # split line
                try:
                    # split by account index
                    state, time, this_account_index = line.split(" ")
                    time = float(time)
                    this_account_index = int(this_account_index)
                    if int(this_account_index) != int(self.logger.current_account):
                        continue

                    # handling negative time for whatever reason
                    most_recent_time = max(most_recent_time, time)
                except Exception as e:
                    print(f"Got an exception in StateHistory.get_time_of_last_state()\n{e}")
                    pass

        return int(most_recent_time)

    def state_is_ready(self, state: str) -> bool:
        def to_wrap():
            # if the state isnt in the state time increment dictionary, return True
            if state not in self.state2time_increment:
                print(f"The time increment for {state} isn't specified, so defaulting to True (ready)")
                return True

            # get the time of the last state
            last_time = self.get_time_of_last_state(state)

            # if the last time is -1, then the state has never been run before
            if last_time == -1:
                print(f"{state} has never been run before, so it is ready")
                return True

            # retrieve the time increment for this state
            time_increment = self.state2time_increment[state]

            # convert the time increment from hours to seconds
            time_increment = time_increment * 60 * 60

            # time since last state
            time_since_last_state = time.time() - last_time
            print(f"It's been {str(time_since_last_state)[:5]}s since this state has been ran")

            # if the time since the last state is greater than the time increment, return True
            if time_since_last_state > time_increment:
                print(f"{state} is ready to run")
                return True

            # otherwise
            print(f"{state} is not ready to run")
            return False

        # add ready states to history because they always happen after True returns
        if to_wrap():
            self.add_state(state)
            return True

        return False


class StateOrder:
    def __init__(self):
        self.states = [
            "select_battle_mode",
            "randomize_deck",
            "cycle_deck",
            "start_fight",
            "1v1_fight",
            "2v2_fight",
            "end_fight",
        ]

    def next_state(self, curr_state):
        if curr_state in ["restart", "start", "restart_emulator", "press_back_and_retry", "unknown"]:
            return self.states[0]

        if curr_state not in self.states:
            print(f'[!] Fatal error: state "{curr_state}" not in state order')
            return "restart"

        this_index = self.states.index(curr_state)

        # if last, loop
        if this_index == len(self.states) - 1:
            return self.states[0]

        # else, return next state
        nxt = self.states[this_index + 1]
        if nxt not in VALID_STATES:
            print(f"[!] Fatal error: invalid next state {curr_state} -> {nxt}")
            return "restart"
        return nxt


# States whose lines are kept out of the terminal (still written to the log
# file). These are the noisy emulator-boot / restart dumps; the logger is told
# console=False for them via set_current_state, so it never needs to know any
# state names itself.
QUIET_CONSOLE_STATES = {"No state", "restart"}


def state_tree(
    emulator,
    logger: Logger,
    state,
    job_list,
    state_history: StateHistory,
    state_order: StateOrder,
) -> str:
    """Method to handle and loop between the various states of the bot"""
    global mode_used_in_1v1, fight_mode_cycle_index  # noqa: PLW0602
    logger.log(f'Set the current state to "{state}"')
    logger.set_current_state(state, console=state not in QUIET_CONSOLE_STATES)
    time.sleep(0.1)

    # header in the log file to split the log by state loop iterations
    logger.log(f"\n\n------------------------------\nTHIS STATE IS: {state} ")

    if state is None:
        logger.error("Error! State is None!!")
        raise ValueError("State is None - critical error in state machine")

    if state == "fail":
        logger.error("State machine entered 'fail' state - stopping execution")
        logger.add_restart_after_failure()
        raise RuntimeError("State machine entered fail state - unrecoverable error")

    if state not in VALID_STATES and state not in (None, "fail"):
        logger.error(f"Failure in state tree: unknown state '{state}' — restarting")
        return "restart"

    if state == "start":
        return state_order.next_state(state)

    if state == "restart":
        logger.change_status("Restarting emulator...")
        logger.add_restart_after_failure()
        emulator.restart()
        return state_order.next_state(state)

    if state == "randomize_deck":
        # if randomize deck isn't toggled, return next state
        if not job_list[UIField.RANDOM_DECKS_USER_TOGGLE]:
            logger.log("deck randomization isn't toggled. skipping this state")
            return state_order.next_state(state)

        # make sure there's a relevant fight job toggled, else just skip deck randomization
        if (
            not job_list.get(UIField.CLASSIC_1V1_USER_TOGGLE, False)
            and not job_list.get(UIField.CLASSIC_2V2_USER_TOGGLE, False)
            and not job_list.get(UIField.TROPHY_ROAD_USER_TOGGLE, False)
        ):
            print("No fight jobs are toggled, so skipping random deck state.")
            return state_order.next_state(state)

        # Get the selected deck number from job_list, default to 2 if not found
        deck_number = job_list.get(UIField.DECK_NUMBER_SELECTION.value, 2)
        if randomize_deck_state(emulator, logger, deck_number) is False:
            return handle_state_failure(logger, "randomize_deck", "randomize_deck_state")

        return state_order.next_state(state)

    if state == "cycle_deck":
        if not job_list[UIField.CYCLE_DECKS_USER_TOGGLE]:
            logger.log("deck cycling isn't toggled. skipping this state")
            return state_order.next_state(state)

        if mode_used_in_1v1 is None:
            logger.log("No battle mode selected, skipping deck cycling.")
            return state_order.next_state(state)

        account_index = max(0, int(logger.current_account))
        deck_cycle_index = get_deck_number_for_battle_mode(mode_used_in_1v1, account_index)

        deck_count = job_list.get(UIField.MAX_DECK_SELECTION.value, 10)

        success, selected_deck_number = select_deck_state(emulator, logger, deck_cycle_index, deck_count)

        if not success or selected_deck_number is None:
            return handle_state_failure(logger, "cycle_deck", "select_deck_state")

        next_deck = selected_deck_number + 1 if selected_deck_number < deck_count else 1

        set_deck_number_for_battle_mode(mode_used_in_1v1, next_deck, account_index)

        return state_order.next_state(state)

    if state == "select_battle_mode":
        # Get all enabled fight modes
        enabled_modes = []
        if job_list.get(UIField.CLASSIC_1V1_USER_TOGGLE, False):
            enabled_modes.append("Classic 1v1")
        if job_list.get(UIField.CLASSIC_2V2_USER_TOGGLE, False):
            enabled_modes.append("Classic 2v2")
        if job_list.get(UIField.TROPHY_ROAD_USER_TOGGLE, False):
            enabled_modes.append("Trophy Road")

        if not enabled_modes:
            logger.log("No fight modes enabled, skipping fight chain")
            return state_order.next_state("end_fight")

        # if more than one mode is selected, just cycle through them
        if len(enabled_modes) > 1:
            selected_mode = get_next_fight_mode(job_list)
            logger.change_status(f"Selected {selected_mode} as the next battle mode")
            mode_used_in_1v1 = selected_mode
            if select_mode(emulator, selected_mode, logger) is False:
                return handle_state_failure(
                    logger, "select_battle_mode", "select_mode", f"Failed to select mode: {selected_mode}"
                )
        else:
            # if only one mode is selected, check if it's already selected
            selected_mode = enabled_modes[0]
            mode_used_in_1v1 = selected_mode
            if not check_if_battle_mode_is_selected(emulator, selected_mode):
                if select_mode(emulator, selected_mode, logger) is False:
                    return handle_state_failure(
                        logger, "select_battle_mode", "select_mode", f"Failed to select mode: {selected_mode}"
                    )
            else:
                logger.change_status(f"{selected_mode} is already selected")

        return state_order.next_state(state)

    if state == "start_fight":
        if mode_used_in_1v1 is None:
            print("No battle mode selected. Skipping this state")
            return state_order.next_state(state)

        # Start fight using the selected mode directly
        if start_fight(emulator, logger, mode_used_in_1v1) is False:
            return handle_state_failure(logger, "start_fight", "start_fight", "Failed while starting fight")

        # go to next state
        return state_order.next_state(state)

    if state == "1v1_fight":
        # Check if the current mode is a 1v1 type (Classic 1v1 or Trophy Road)
        if mode_used_in_1v1 not in ["Classic 1v1", "Trophy Road"]:
            print(f"Current mode '{mode_used_in_1v1}' is not a 1v1 type. Skipping this state")
            return state_order.next_state(state)

        random_plays_flag = job_list.get(UIField.RANDOM_PLAYS_USER_TOGGLE, False)

        recording_flag = job_list.get(UIField.RECORD_FIGHTS_TOGGLE, False)
        custom_path = job_list.get(UIField.RECORDING_FOLDER_PATH, None)
        if (
            do_fight_state(
                emulator,
                logger,
                random_plays_flag,
                mode_used_in_1v1,
                False,
                recording_flag=recording_flag,
                custom_path=custom_path,
            )
            is False
        ):
            return handle_state_failure(
                logger, "1v1_fight", "do_fight_state", f"1v1 fight failed in mode: {mode_used_in_1v1}"
            )

        return state_order.next_state(state)

    if state == "2v2_fight":
        # Check if the current mode is a 2v2 type (Classic 2v2)
        if mode_used_in_1v1 != "Classic 2v2":
            print(f"Current mode '{mode_used_in_1v1}' is not a 2v2 type. Skipping this state")
            return state_order.next_state(state)

        random_plays_flag = job_list.get(UIField.RANDOM_PLAYS_USER_TOGGLE, False)

        custom_path = job_list.get(UIField.RECORDING_FOLDER_PATH, None)
        # 2v2 fights are never recorded (training data is 1v1-only: Trophy Road + Classic 1v1).
        if (
            do_fight_state(
                emulator,
                logger,
                random_plays_flag,
                "Classic 2v2",
                called_from_launching=False,
                recording_flag=False,
                custom_path=custom_path,
            )
            is False
        ):
            return handle_state_failure(logger, "2v2_fight", "do_fight_state", "2v2 fight failed")

        return state_order.next_state(state)

    if state == "end_fight":
        if not any_fight_mode_enabled(job_list):
            logger.log("No fight modes enabled, skipping end_fight")
            return state_order.next_state(state)

        recording_flag = job_list.get(UIField.RECORD_FIGHTS_TOGGLE, False)
        if (
            end_fight_state(
                emulator,
                logger,
                recording_flag,
                job_list[UIField.DISABLE_WIN_TRACK_TOGGLE],
            )
            is False
        ):
            return handle_state_failure(logger, "end_fight", "end_fight_state", "Failed to end fight properly")

        # When recording, periodically bundle ~5 GB of the oldest packs into a zip
        # so the loose recordings folder doesn't grow without bound.
        if recording_flag:
            custom_path = job_list.get(UIField.RECORDING_FOLDER_PATH, None)
            maybe_archive_recordings(logger=logger, custom_path=custom_path)

        return state_order.next_state(state)

    if state == "press_back_and_retry":
        # Lightweight recovery for Python exceptions: BACK out to a known
        # screen instead of rebooting the emulator. Falls through to normal
        # mode selection on the next loop.
        try:
            press = getattr(emulator, "press_back", None)
            if callable(press):
                press()
            else:
                emulator.click(35, 500)
        except Exception as e:
            logger.log(f"Menu recovery tap failed: {e}")
        time.sleep(2)
        return state_order.next_state("restart")

    logger.error("Failure in state tree")
    return "fail"


if __name__ == "__main__":
    pass
