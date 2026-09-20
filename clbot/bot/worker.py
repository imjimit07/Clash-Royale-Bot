import threading
import time
import traceback
from multiprocessing import Process, Queue
from multiprocessing.synchronize import Event
from typing import Any, Protocol

from clbot.bot.states import StateHistory, StateOrder, state_tree
from clbot.emulators import EmulatorType, get_emulator_registry
from clbot.emulators.base import EmulatorNotReadyError
from clbot.interface.enums import UIField
from clbot.utils.logger import ProcessLogger, attach_worker_file_logging

# --- Watchdog / recovery tuning (Part 1 + Part 8) ---
STATE_TIMEOUT_SECONDS = 120
STATE_WARN_SECONDS = 60
MAX_CONSECUTIVE_RESTARTS = 5
RESTART_BACKOFF_BASE = 5
MAX_EMULATOR_FAILS = 3


def safe_state_call(func, *args, **kwargs):
    """Global exception guard: never let one state crash the loop."""
    try:
        return func(*args, **kwargs)
    except Exception:
        traceback.print_exc()
        return "restart"


class LoopDetector:
    """Detect 1-2 state oscillation indicating a stuck machine."""

    def __init__(self, limit: int = 8) -> None:
        self.history: list[str] = []
        self.limit = limit

    def push(self, state: str) -> None:
        self.history.append(str(state))
        if len(self.history) > self.limit * 2:
            self.history.pop(0)

    def is_stuck(self) -> bool:
        if len(self.history) < self.limit:
            return False
        last = self.history[-self.limit :]
        return len(set(last)) <= 2

    def clear(self) -> None:
        self.history.clear()


class WorkerProcess(Process):
    """Worker process for running the bot.

    Uses multiprocessing instead of threading for reliable force termination.
    Communicates stats back to main process via Queue.
    """

    def __init__(
        self,
        jobs: dict[str, Any],
        stats_queue: Queue,
        shutdown_event: Event,
        session_log_path: str,
    ) -> None:
        super().__init__(daemon=True)
        self.jobs = jobs
        self.stats_queue = stats_queue
        self.shutdown_event = shutdown_event
        self.session_log_path = session_log_path

    def _setup_emulator(self, jobs: dict[str, Any], logger: ProcessLogger):
        """Set up Google Play Games PC (Developer Mode) — the only supported backend."""
        try:
            from clbot.utils.admin import warn_if_not_admin

            warn_if_not_admin(logger)
        except Exception:
            pass
        # Battle-only bot: always use Google Play Games, ignoring any legacy selection.
        emulator_selection = EmulatorType.GOOGLE_PLAY
        registry = get_emulator_registry()

        if emulator_selection not in registry:
            print("[!] Fatal error: Google Play Games emulator is not available on this platform!")
            logger.change_status("Google Play Games is not available on this platform.")
            return None

        controller_class = registry[emulator_selection]

        try:
            print("Creating Google Play Games PC (Developer Mode) emulator")
            gp_device_serial = jobs.get(UIField.GP_DEVICE_SERIAL.value) or None
            emu = controller_class(logger=logger, device_serial=gp_device_serial)

            # Construction is cheap; restart() boots the emulator and launches Clash.
            # First-boot failure stops the bot — there is no setup retry.
            emu.restart()
            return emu

        except EmulatorNotReadyError as e:
            print(f"{emulator_selection} emulator is not ready: {e}")
            logger.change_status(f"{emulator_selection} is not ready — fix it and start the bot again.")
            return None
        except Exception as e:
            print(f"Failed to create {emulator_selection} emulator: {e}")
            logger.change_status(f"Failed to start {emulator_selection}. Verify its installation!")
            return None

    def _run_state_with_timeout(self, emulator, logger, state, jobs, state_history, state_order, timeout: float):
        """Run one state_tree call in a daemon thread; return None on timeout."""
        result: dict[str, Any] = {"value": None, "done": False}

        def target() -> None:
            try:
                result["value"] = state_tree(emulator, logger, state, jobs, state_history, state_order)
            except Exception as e:
                try:
                    logger.error(f"State thread crashed in '{state}': {e}")
                except Exception:
                    pass
                result["value"] = "restart"
            finally:
                result["done"] = True

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)
        if not result["done"]:
            return None
        return result["value"]

    def _restart_emulator(self, emulator, logger) -> bool:
        """Full emulator + ADB restart. Returns True on success."""
        try:
            logger.change_status("Full emulator restart for recovery...")
            try:
                emulator.stop()
            except Exception:
                pass
            time.sleep(5)
            try:
                emulator.start()
            except Exception as e:
                logger.error(f"Emulator restart failed at start(): {e}")
                return False
            time.sleep(15)
            reconnect = getattr(emulator, "ensure_adb_connection", None)
            if callable(reconnect):
                try:
                    if not reconnect():
                        logger.error("Emulator restart failed: ADB reconnect failed")
                        return False
                except Exception as e:
                    logger.error(f"Emulator restart failed: ADB reconnect raised {e}")
                    return False
            else:
                connect = getattr(emulator, "_connect", None)
                if callable(connect):
                    try:
                        connect()
                    except Exception:
                        pass
            return True
        except Exception as e:
            try:
                logger.error(f"Emulator restart failed: {e}")
            except Exception:
                pass
            traceback.print_exc()
            return False

    def _run_bot_loop(self, emulator, jobs: dict[str, Any], logger: ProcessLogger) -> None:
        """Run the main bot state loop with watchdog, backoff and escalation."""
        state = "start"
        state_history = StateHistory(logger)
        state_order = StateOrder()
        loop_detector = LoopDetector()
        consecutive_restarts = 0
        emulator_fail_count = 0

        while not self.shutdown_event.is_set():
            # --- Emulator crash recovery (Part 2.4) ---
            try:
                running_check = getattr(emulator, "_is_emulator_running", None)
                if callable(running_check) and not running_check():
                    logger.error("Emulator process missing — attempting recovery restart")
                    if self._restart_emulator(emulator, logger):
                        consecutive_restarts = 0
                        continue
                    emulator_fail_count += 1
                    if emulator_fail_count >= MAX_EMULATOR_FAILS:
                        logger.error("Emulator unrecoverable — stopping bot")
                        try:
                            self.shutdown_event.set()
                        except Exception:
                            pass
                        return
                    time.sleep(RESTART_BACKOFF_BASE)
                    continue
                adb_check = getattr(emulator, "ensure_adb_connection", None)
                if callable(adb_check):
                    try:
                        adb_check(retries=1)
                    except Exception:
                        pass
            except Exception:
                pass

            state_start = time.time()
            try:
                new_state = self._run_state_with_timeout(
                    emulator, logger, state, jobs, state_history, state_order, STATE_TIMEOUT_SECONDS
                )
                elapsed = time.time() - state_start
                if elapsed > STATE_WARN_SECONDS:
                    try:
                        logger.log(f"State '{state}' took {elapsed:.1f}s (warn threshold {STATE_WARN_SECONDS}s)")
                    except Exception:
                        pass

                if new_state is None:
                    logger.error(f"WATCHDOG: State '{state}' exceeded {STATE_TIMEOUT_SECONDS}s — forcing restart")
                    new_state = "restart"

                loop_detector.push(str(new_state))
                if loop_detector.is_stuck():
                    logger.error(f"Loop detected in states {loop_detector.history[-8:]} — restarting emulator")
                    if self._restart_emulator(emulator, logger):
                        consecutive_restarts = 0
                    else:
                        emulator_fail_count += 1
                        if emulator_fail_count >= MAX_EMULATOR_FAILS:
                            logger.error("Emulator unrecoverable after loop — stopping bot")
                            return
                    loop_detector.clear()
                    state = "restart"
                    continue

                if new_state == "restart":
                    consecutive_restarts += 1
                    backoff = min(RESTART_BACKOFF_BASE * (2 ** (consecutive_restarts - 1)), 60)
                    logger.log(f"Restart #{consecutive_restarts} — backoff {backoff}s")
                    time.sleep(backoff)
                    if consecutive_restarts >= MAX_CONSECUTIVE_RESTARTS:
                        logger.error("Too many restarts — attempting full emulator restart")
                        if self._restart_emulator(emulator, logger):
                            emulator_fail_count = 0
                        else:
                            emulator_fail_count += 1
                            if emulator_fail_count >= MAX_EMULATOR_FAILS:
                                logger.error("Emulator unrecoverable — stopping bot")
                                return
                        consecutive_restarts = 0
                else:
                    consecutive_restarts = 0

                if new_state in ["fail", None]:
                    logger.error(f"Critical error: state_tree returned '{new_state}'")
                    logger.add_restart_after_failure()
                    new_state = "restart"

                state = safe_state_call(lambda: new_state)

            except Exception as e:
                logger.error(f"Exception in state_tree: {e}")
                logger.log(f"Current state was: {state}")
                print(f"[ERROR] Exception in state_tree: {e}")
                traceback.print_exc()
                state = "restart"
                consecutive_restarts += 1
                if consecutive_restarts >= MAX_CONSECUTIVE_RESTARTS:
                    logger.error("Too many consecutive exceptions — attempting full emulator restart")
                    if not self._restart_emulator(emulator, logger):
                        emulator_fail_count += 1
                        if emulator_fail_count >= MAX_EMULATOR_FAILS:
                            logger.error("Emulator unrecoverable — stopping bot")
                            return
                    consecutive_restarts = 0

            # Note: Pause functionality removed in multiprocessing version
            # If pause is needed, use a separate mp.Event

    def run(self) -> None:
        """Main worker process execution."""
        print("WorkerProcess run()...")
        attach_worker_file_logging(self.session_log_path)

        # Create logger that sends stats through queue
        logger = ProcessLogger(self.stats_queue)

        try:
            emulator = self._setup_emulator(self.jobs, logger)
            if emulator is None:
                return

            self._run_bot_loop(emulator, self.jobs, logger)
        except Exception as err:
            logger.error(str(err))
            traceback.print_exc()
        finally:
            logger.change_status("Bot stopped")


class _StoppableProcess(Protocol):
    """Structural interface for the process-lifecycle methods ``stop_worker_process`` needs.

    Satisfied by ``multiprocessing.Process`` (and ``WorkerProcess``), spawned context
    processes, and test fakes alike — so the helper depends on the contract, not the
    concrete class.
    """

    def is_alive(self) -> bool: ...
    def terminate(self) -> None: ...
    def join(self, timeout: float | None = ...) -> None: ...
    def kill(self) -> None: ...


class _Signal(Protocol):
    """Structural interface for the shutdown event — only ``set()`` is used here."""

    def set(self) -> None: ...


def stop_worker_process(
    process: _StoppableProcess | None,
    shutdown_event: _Signal | None = None,
    *,
    graceful_timeout: float = 2.0,
) -> None:
    """Stop the worker process via OS-level termination.

    Signals shutdown (belt-and-suspenders), then terminates and hard-kills if the
    process does not exit within ``graceful_timeout``. The OS interrupts any blocking
    call in the worker (sleeps, ADB/socket waits), so this does not depend on the
    worker cooperatively noticing the shutdown event.
    """
    if shutdown_event is not None:
        shutdown_event.set()
    if process is None or not process.is_alive():
        return
    process.terminate()  # SIGTERM / TerminateProcess
    process.join(timeout=graceful_timeout)
    if process.is_alive():
        process.kill()  # SIGKILL
        process.join(timeout=graceful_timeout)
