"""Main entry point for the Clash Royale Bot ttkbootstrap interface."""

from __future__ import annotations

import locale
import logging
import multiprocessing as mp
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from multiprocessing.synchronize import Event

_original_setlocale = locale.setlocale
_LocaleError = locale.Error


# Parameter name must match the stdlib's (callers may pass locale= by keyword);
# it shadows the locale module inside the body, hence the module-level aliases.
def _setlocale_safe(category: int, locale: str | Iterable[str | None] | None = None) -> str:
    """Fallback to the C locale if the requested locale is unsupported."""
    try:
        return _original_setlocale(category, locale)
    except _LocaleError:
        return _original_setlocale(category, "C")


# Deliberate monkeypatch; ty rejects reassigning a module-level function even with
# a matching signature, so suppress rather than launder the type.
locale.setlocale = _setlocale_safe  # ty: ignore[invalid-assignment]

try:
    # Force a portable locale so ttkbootstrap does not raise unsupported locale errors on non-English systems.
    locale.setlocale(locale.LC_ALL, "C")
except locale.Error:
    # If even the C locale is unavailable, continue with the default to avoid crashing on import.
    pass

from clbot.bot.worker import WorkerProcess, stop_worker_process
from clbot.emulators import EmulatorType
from clbot.emulators.adb_base import validate_device_serial
from clbot.emulators.google_play import GooglePlayEmulatorController
from clbot.interface.enums import UIField, has_start_ready_job
from clbot.interface.ui import BotUI, no_jobs_popup
from clbot.utils.caching import USER_SETTINGS_CACHE
from clbot.utils.cli_config import arg_parser
from clbot.utils.discord_rpc import DiscordRPCManager
from clbot.utils.logger import Logger, initialize_pylogging, log_dir, log_name
from clbot.utils.open_folder import open_folder
from clbot.utils.platform import get_recordings_dir

initialize_pylogging()


def make_job_dictionary(values: dict[str, Any]) -> dict[str, Any]:
    """Create a dictionary of battle-only job toggles based on UI values."""

    def as_bool(field: UIField) -> bool:
        return bool(values.get(field.value, False))

    def as_int(field: UIField, default: int) -> int:
        try:
            return int(values.get(field.value, default))
        except (TypeError, ValueError):
            return default

    def as_str(field: UIField, default: str = "") -> str:
        return str(values.get(field.value, default) or default)

    job_dictionary: dict[str, Any] = {
        UIField.CLASSIC_1V1_USER_TOGGLE.value: as_bool(UIField.CLASSIC_1V1_USER_TOGGLE),
        UIField.CLASSIC_2V2_USER_TOGGLE.value: as_bool(UIField.CLASSIC_2V2_USER_TOGGLE),
        UIField.TROPHY_ROAD_USER_TOGGLE.value: as_bool(UIField.TROPHY_ROAD_USER_TOGGLE),
        UIField.RANDOM_DECKS_USER_TOGGLE.value: as_bool(UIField.RANDOM_DECKS_USER_TOGGLE),
        UIField.DECK_NUMBER_SELECTION.value: as_int(UIField.DECK_NUMBER_SELECTION, 2),
        UIField.CYCLE_DECKS_USER_TOGGLE.value: as_bool(UIField.CYCLE_DECKS_USER_TOGGLE),
        UIField.MAX_DECK_SELECTION.value: as_int(UIField.MAX_DECK_SELECTION, 2),
        UIField.RANDOM_PLAYS_USER_TOGGLE.value: as_bool(UIField.RANDOM_PLAYS_USER_TOGGLE),
        UIField.DISABLE_WIN_TRACK_TOGGLE.value: as_bool(UIField.DISABLE_WIN_TRACK_TOGGLE),
        UIField.RECORD_FIGHTS_TOGGLE.value: as_bool(UIField.RECORD_FIGHTS_TOGGLE),
        UIField.RECORDING_FOLDER_PATH.value: as_str(UIField.RECORDING_FOLDER_PATH),
    }

    # Battle-only bot: Google Play Games PC (Developer Mode) is the only backend.
    job_dictionary["emulator"] = EmulatorType.GOOGLE_PLAY

    gp_serial = values.get(UIField.GP_DEVICE_SERIAL.value)
    job_dictionary[UIField.GP_DEVICE_SERIAL.value] = gp_serial.strip() if gp_serial else None

    return job_dictionary


def has_no_jobs_selected(job_dict: dict[str, Any]) -> bool:
    """Check if no start-ready jobs are selected."""
    return not has_start_ready_job(job_dict)


def save_current_settings(values: dict[str, Any]) -> None:
    """Cache the user's current settings."""
    USER_SETTINGS_CACHE.cache_data(values)


def load_settings(settings: dict[str, Any] | None, ui: BotUI) -> dict[str, Any] | None:
    """Load settings into the UI from CLI args or cached data."""
    loaded = settings
    if not loaded and USER_SETTINGS_CACHE.exists():
        loaded = USER_SETTINGS_CACHE.load_data()
    if loaded:
        ui.set_all_values(loaded)
    return loaded


def start_button_event(
    logger: Logger,
    ui: BotUI,
    values: dict[str, Any],
    stats_queue: Queue,
    shutdown_event: Event,
) -> WorkerProcess | None:
    """Start the worker process with the current configuration."""
    job_dictionary = make_job_dictionary(values)

    if has_no_jobs_selected(job_dictionary):
        no_jobs_popup()
        logger.log("No jobs are selected!")
        return None

    # Google Play - serial optional; validate format only, worker boots and connects.
    device_serial = job_dictionary.get(UIField.GP_DEVICE_SERIAL.value)
    if device_serial and not validate_device_serial(device_serial):
        logger.change_status(f"Start cancelled: invalid device serial '{device_serial}'.")
        return None

    logger.log("Start Button Event")
    logger.change_status("Starting the bot!")
    save_current_settings(values)
    logger.log_job_dictionary(job_dictionary)

    # Single-pane dashboard has no notebook; tabbed UIs switch to Stats instead.
    notebook = getattr(ui, "notebook", None)
    stats_tab = getattr(ui, "stats_tab", None)
    if notebook is not None and stats_tab is not None:
        try:
            notebook.select(stats_tab)
        except Exception:
            pass

    process = WorkerProcess(job_dictionary, stats_queue, shutdown_event, log_name)
    process.start()
    return process


def update_layout(
    ui: BotUI,
    logger: Logger,
    stats: dict[str, Any] | None = None,
) -> None:
    """Update UI widgets from the logger's statistics."""
    if stats is None:
        stats = logger.get_stats()
    if stats is None:
        return
    ui.update_stats(stats)
    status_text = logger.current_status
    if "current_status" in stats:
        status_text = str(stats["current_status"])
    ui.set_status(status_text)
    if ui.get_button_state() == "idle" and not ui.can_start():
        return
    ui.append_log(status_text)


def exit_button_event(process: WorkerProcess | None, shutdown_event: Event | None) -> None:
    """Force stop the worker process on application exit."""
    stop_worker_process(process, shutdown_event)


def handle_process_finished(
    ui: BotUI,
    process: WorkerProcess | None,
    logger: Logger,
) -> tuple[WorkerProcess | None, Logger]:
    """Check if the worker process has finished and reset UI state if so."""
    if process is not None and not process.is_alive():
        ui.set_button_state("idle")
        logger = Logger(timed=False)
        logger.change_status("Idle")
        process = None
    return process, logger


def open_recordings_folder() -> None:
    custom_path = None
    if USER_SETTINGS_CACHE.exists():
        settings = USER_SETTINGS_CACHE.load_data()
        custom_path = settings.get(UIField.RECORDING_FOLDER_PATH.value) or None
    open_folder(get_recordings_dir(custom_path=custom_path))


def open_logs_folder() -> None:
    folder_path = log_dir
    open_folder(folder_path)


class BotApplication:
    """Main application class for the ttkbootstrap GUI."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.ui = BotUI()
        self.ui.main_btn.configure(command=self._on_main_button)
        self.ui.register_config_callback(self._on_config_change)
        self.ui.register_open_logs_callback(self._on_open_logs_clicked)
        self.ui.register_open_recordings_callback(self._on_open_recordings_clicked)
        self.ui.protocol("WM_DELETE_WINDOW", self._on_close)
        # Auto-refresh Google Play device list when dropdown opens
        # (battle-only UI has no visible device dropdown — guard for headless state).
        combo = getattr(self.ui, "gp_device_serial_combo", None)
        if combo is not None:
            try:
                combo.configure(postcommand=self._refresh_gp_devices)
            except Exception:
                pass

        # Multiprocessing primitives
        self.process: WorkerProcess | None = None
        self.stats_queue: Queue | None = None
        self.shutdown_event: Event | None = None
        self.last_stats: dict[str, Any] = {}

        self.logger = Logger(timed=False)
        self.discord_rpc = DiscordRPCManager()
        self.discord_rpc_enabled = False
        self._closing = False
        self._suppress_persist = True
        self.current_values = self.ui.get_all_values()

        loaded = load_settings(settings, self.ui)
        if loaded:
            self.current_values = self.ui.get_all_values()
            self.discord_rpc_enabled = bool(self.current_values.get(UIField.DISCORD_RPC_TOGGLE.value, False))
        self._suppress_persist = False

        self.ui.set_button_state("idle")
        self._poll()

    def _on_main_button(self) -> None:
        """Handle the unified Start/Stop button."""
        state = self.ui.get_button_state()
        if state == "idle":
            self._on_start()
        elif state == "running":
            self._on_stop()

    def _on_start(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        values = self.ui.get_all_values()

        # Create fresh multiprocessing primitives for each run
        self.stats_queue = mp.Queue(maxsize=100)
        self.shutdown_event = mp.Event()
        self.last_stats = {}

        new_logger = Logger(timed=True)
        process = start_button_event(new_logger, self.ui, values, self.stats_queue, self.shutdown_event)
        if process is not None:
            self.logger = new_logger
            self.process = process
            self.current_values = values.copy()
            if self.discord_rpc_enabled:
                self.discord_rpc.enable()
            self.ui.set_button_state("running")
        else:
            self.ui.set_button_state("idle")

    def _on_stop(self) -> None:
        """Stop the worker via OS-level termination (single press)."""
        self.logger.change_status("Stopping")
        stop_worker_process(self.process, self.shutdown_event)
        self._snapshot_session_stats()
        self.process = None
        self.logger.change_status("Idle")
        self.ui.set_button_state("idle")

    def _on_config_change(self, values: dict[str, Any]) -> None:
        changed = {key for key, value in values.items() if self.current_values.get(key) != value}
        random_toggle = UIField.RANDOM_DECKS_USER_TOGGLE.value
        cycle_toggle = UIField.CYCLE_DECKS_USER_TOGGLE.value
        if values.get(random_toggle) and values.get(cycle_toggle):
            if random_toggle in changed:
                self.ui.set_all_values({cycle_toggle: False})
                values[cycle_toggle] = False
            elif cycle_toggle in changed:
                self.ui.set_all_values({random_toggle: False})
                values[random_toggle] = False
            else:
                self.ui.set_all_values({cycle_toggle: False})
                values[cycle_toggle] = False
        self.current_values = values.copy()
        self.discord_rpc_enabled = bool(values.get(UIField.DISCORD_RPC_TOGGLE.value, False))
        if not self._suppress_persist:
            save_current_settings(values)

    def _snapshot_session_stats(self) -> None:
        """Freeze the current session stats for display while idle."""
        # Counters live in the worker and arrive via the stats queue. The UI
        # logger's change_status() can rebuild logger.stats from zeroed instance
        # fields, so prefer the latest queue snapshot and only borrow runtime
        # from get_stats().
        frozen = dict(self.last_stats) if self.last_stats else {}
        live_stats = self.logger.get_stats()
        if live_stats:
            if not frozen:
                frozen = dict(live_stats)
            else:
                frozen["time_since_start"] = live_stats["time_since_start"]
        if frozen:
            self.last_stats = frozen

    def _get_display_stats(self) -> dict[str, Any] | None:
        """Return stats for the UI: frozen session totals when idle, live stats while running."""
        if self.process is None and self.last_stats:
            stats = self.last_stats.copy()
            stats["current_status"] = self.logger.current_status
            return stats
        return self.logger.get_stats()

    def _update_ui_from_stats(self) -> None:
        update_layout(self.ui, self.logger, self._get_display_stats())

    def _poll(self) -> None:
        if self._closing:
            return

        # Read all available stats from the queue
        if self.stats_queue is not None:
            try:
                while True:
                    stats = self.stats_queue.get_nowait()
                    if self.process is not None:
                        self.last_stats = stats
                        # Update local logger with stats from worker process
                        if "current_status" in stats:
                            self.logger.current_status = stats["current_status"]
                        if self.logger.stats is not None:
                            self.logger.stats.update(stats)
            except Exception:
                pass  # Queue empty

        if self.process is not None and not self.process.is_alive():
            self._snapshot_session_stats()

        self.process, self.logger = handle_process_finished(self.ui, self.process, self.logger)
        self._update_ui_from_stats()
        self.discord_rpc.sync(self.discord_rpc_enabled, self._get_display_stats())
        self.ui.after(100, self._poll)

    def _on_open_logs_clicked(self) -> None:
        open_logs_folder()

    def _on_open_recordings_clicked(self) -> None:
        open_recordings_folder()

    def _on_close(self) -> None:
        self._closing = True
        exit_button_event(self.process, self.shutdown_event)
        self.discord_rpc.disable()
        self.ui.destroy()

    def _refresh_device_dropdown(self, combo_widget, controller_class, emulator_name: str) -> None:
        """Refresh device list for a dropdown (called on open)."""
        try:
            devices = controller_class.discover_devices()
            combo_widget.configure(values=["", *devices])  # Empty option for default
            if not devices:
                self.logger.change_status(f"No ADB devices found for {emulator_name}")
        except Exception:
            logging.exception(f"Failed to refresh {emulator_name} device list")
            self.logger.change_status(f"Failed to discover devices for {emulator_name}")

    def _refresh_gp_devices(self) -> None:
        """Refresh device list for Google Play dropdown (called on open)."""
        combo = getattr(self.ui, "gp_device_serial_combo", None)
        if combo is None:
            return
        self._refresh_device_dropdown(combo, GooglePlayEmulatorController, "Google Play")

    def run(self, start_on_run: bool = False) -> None:
        if start_on_run:
            self.ui.after(200, self._on_start)
        self.ui.mainloop()


def main_gui(start_on_run: bool = False, settings: dict[str, Any] | None = None) -> None:
    app = BotApplication(settings)
    app.run(start_on_run)


if __name__ == "__main__":
    from multiprocessing import freeze_support, set_start_method

    freeze_support()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # Already set
    cli_args = arg_parser()
    main_gui(start_on_run=cli_args.start)
