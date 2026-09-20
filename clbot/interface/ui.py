from __future__ import annotations

import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING

import ttkbootstrap as ttk
from ttkbootstrap.constants import LEFT, READONLY, RIGHT, X
from ttkbootstrap.widgets import ToolTip

from clbot.emulators import EmulatorType
from clbot.interface.config import (
    GOOGLE_PLAY_DEVICE_CONFIG,
    GOOGLE_PLAY_SETTINGS,
    JOBS,
    JobConfig,
)
from clbot.interface.enums import (
    BATTLE_STAT_LABELS,
    BOT_STAT_LABELS,
    BotStatField,
    DerivedStatField,
    StatField,
    UIField,
    has_start_ready_job,
)
from clbot.utils.platform import get_recordings_dir, is_windows

if TYPE_CHECKING:
    from collections.abc import Callable

# Dashboard card layout (display only — JOBS in config.py keeps bot state-machine order).
_DASHBOARD_CARDS: tuple[tuple[str, tuple[UIField, ...]], ...] = (
    (
        "Battle Mode",
        (
            UIField.CLASSIC_1V1_USER_TOGGLE,
            UIField.CLASSIC_2V2_USER_TOGGLE,
            UIField.TROPHY_ROAD_USER_TOGGLE,
        ),
    ),
    (
        "Deck Options",
        (
            UIField.RANDOM_DECKS_USER_TOGGLE,
            UIField.CYCLE_DECKS_USER_TOGGLE,
        ),
    ),
    (
        "Options",
        (
            UIField.RANDOM_PLAYS_USER_TOGGLE,
            UIField.DISABLE_WIN_TRACK_TOGGLE,
        ),
    ),
)

# Stored disable_win_track_toggle is inverted in the UI — see _job_toggle_default().
_WIN_TRACK_UI_FIELD = UIField.DISABLE_WIN_TRACK_TOGGLE
_JOB_SPINBOX_LABELS: dict[UIField, str] = {
    UIField.DECK_NUMBER_SELECTION: "Deck slot:",
    UIField.MAX_DECK_SELECTION: "Decks:",
}
_JOB_TOGGLE_ON_STYLE = "round-toggle"
_JOB_TOGGLE_OFF_STYLE = "secondary-round-toggle"
_CARD_STYLE = "TabSection.TLabelframe"
_CARD_PADDING = (10, 8)
_IDLE_STATUS = "Idle"
_START_BLOCKED_MESSAGE = "Enable at least one battle job (1v1, 2v2, or Trophy Road) to start."
_ACTION_BUTTON_IPAD = (18, 4)
_APP_ICON_NAME = "pixel-pycb.ico"
_TERMINAL_MAX_LINES = 500
_HERO_START_TEXT = "▶ Start Bot"
_HERO_STOP_TEXT = "⏹ Stop Bot"


def no_jobs_popup() -> None:
    messagebox.showerror("Cannot start", _START_BLOCKED_MESSAGE)


class BotUI(ttk.Window):
    """Single-pane dark dashboard (header + stat cards + controls + terminal).

    Same contract as the old tabbed window: never talks to the worker
    directly, reports changes through ``config_callback`` as a dict keyed by
    ``UIField.value`` strings, and exposes the members ``__main__.py``
    drives (``main_btn``, ``get/set_all_values``, ``set_button_state``,
    ``can_start``, ``append_log``, ``update_stats``, ``set_status``).
    """

    DEFAULT_THEME = "darkly"
    DEFAULT_WIDTH = 860
    DEFAULT_HEIGHT = 680
    MIN_WIDTH = 800
    MIN_HEIGHT = 600

    def __init__(self) -> None:
        super().__init__(themename=self.DEFAULT_THEME)
        self.title("Clash Royale Bot")
        self._set_window_icon()
        self.geometry(f"{self.DEFAULT_WIDTH}x{self.DEFAULT_HEIGHT}")
        self.resizable(True, True)
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        self._style = ttk.Style()
        self.discord_rpc_var = ttk.BooleanVar(value=False)
        self.record_fights_var = ttk.BooleanVar(value=False)
        # Show the resolved default location in the box until the user picks a custom one.
        self.recording_folder_path_var = ttk.StringVar(value=get_recordings_dir())
        self.advanced_settings_var = ttk.BooleanVar(value=False)
        self._config_callback: Callable[[dict[str, object]], None] | None = None
        self._open_logs_callback: Callable[[], None] | None = None
        self._open_recordings_callback: Callable[[], None] | None = None
        self._config_widgets: dict[str, tk.Widget] = {}
        self._job_toggle_checkbuttons: dict[UIField, ttk.Checkbutton] = {}
        self._job_row_extras: dict[UIField, list[tk.Widget]] = {}
        self._job_extra_spinboxes: dict[UIField, ttk.Spinbox] = {}
        self._traces: list[tuple[tk.Variable, str]] = []
        self._suspend_traces = 0
        self._button_state = "idle"
        self.deck_var: ttk.StringVar | None = None
        self.max_deck_var: ttk.StringVar | None = None
        self._log_last_message: str | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)  # Header stays compact
        self.rowconfigure(1, weight=0)  # Stat cards stay compact
        self.rowconfigure(2, weight=1)  # Controls + terminal absorb height
        self.rowconfigure(3, weight=0)  # Hero button stays compact

        self._init_headless_emulator_state()
        self._build_header()
        self._build_stats_ribbon()
        self._build_body()
        self._build_hero_button()
        self._style_terminal()

    def register_config_callback(self, callback: Callable[[dict[str, object]], None]) -> None:
        self._config_callback = callback

    def register_open_logs_callback(self, callback: Callable[[], None]) -> None:
        self._open_logs_callback = callback

    def register_open_recordings_callback(self, callback: Callable[[], None]) -> None:
        self._open_recordings_callback = callback

    def get_all_values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for field, var in self.jobs_vars.items():
            stored = bool(var.get())
            if field == _WIN_TRACK_UI_FIELD:
                stored = not stored
            values[field.value] = stored

        values[UIField.DECK_NUMBER_SELECTION.value] = self._safe_int(
            self.deck_var.get() if self.deck_var is not None else "2", fallback=2
        )
        values[UIField.CYCLE_DECKS_USER_TOGGLE.value] = bool(self.jobs_vars[UIField.CYCLE_DECKS_USER_TOGGLE].get())
        values[UIField.MAX_DECK_SELECTION.value] = self._safe_int(
            self.max_deck_var.get() if self.max_deck_var is not None else "2", fallback=2
        )
        # Battle-only bot: Google Play Games PC (Developer Mode) is the only backend.
        values[UIField.GOOGLE_PLAY_EMULATOR_TOGGLE.value] = True

        for field, var in self.gp_vars.items():
            values[field.value] = var.get()

        values[UIField.GP_DEVICE_SERIAL.value] = self.gp_device_serial_var.get()

        values[UIField.DISCORD_RPC_TOGGLE.value] = bool(self.discord_rpc_var.get())
        values[UIField.RECORD_FIGHTS_TOGGLE.value] = bool(self.record_fights_var.get())
        values[UIField.RECORDING_FOLDER_PATH.value] = self.recording_folder_path_var.get() or ""
        return values

    def set_all_values(self, values: dict[str, object]) -> None:
        self._suspend_traces += 1
        try:
            for field, var in self.jobs_vars.items():
                if field.value in values:
                    ui_value = bool(values[field.value])
                    if field == _WIN_TRACK_UI_FIELD:
                        ui_value = not ui_value
                    var.set(ui_value)

            if UIField.DECK_NUMBER_SELECTION.value in values and self.deck_var is not None:
                self.deck_var.set(str(values[UIField.DECK_NUMBER_SELECTION.value]))
            if UIField.MAX_DECK_SELECTION.value in values and self.max_deck_var is not None:
                self.max_deck_var.set(str(values[UIField.MAX_DECK_SELECTION.value]))

            # Battle-only bot: emulator is always Google Play Games.
            try:
                self.emulator_var.set(EmulatorType.GOOGLE_PLAY)
            except (AttributeError, tk.TclError):
                pass

            for field, var in self.gp_vars.items():
                config = next((c for c in GOOGLE_PLAY_SETTINGS if c.key == field), None)
                if field.value in values and values[field.value] is not None:
                    var.set(str(values[field.value]))
                elif config:
                    var.set(str(config.default))

            if UIField.GP_DEVICE_SERIAL.value in values:
                self.gp_device_serial_var.set(str(values[UIField.GP_DEVICE_SERIAL.value]))

            self._update_google_play_comboboxes()

        finally:
            self._suspend_traces -= 1

        if self._job_extra_spinboxes:
            self._sync_job_extra_spinboxes()
        if self._job_toggle_checkbuttons:
            self._sync_all_job_toggle_appearances()
        if self._button_state == "idle":
            self._sync_start_button_readiness()

        if UIField.DISCORD_RPC_TOGGLE.value in values:
            self.discord_rpc_var.set(bool(values[UIField.DISCORD_RPC_TOGGLE.value]))

        if UIField.RECORD_FIGHTS_TOGGLE.value in values:
            self.record_fights_var.set(bool(values[UIField.RECORD_FIGHTS_TOGGLE.value]))

        if UIField.RECORDING_FOLDER_PATH.value in values:
            saved_path = str(values[UIField.RECORDING_FOLDER_PATH.value])
            # An empty saved value means "default" -- show the resolved default path.
            self.recording_folder_path_var.set(saved_path or get_recordings_dir())

        self._show_current_emulator_settings()

    def set_button_state(self, state: str) -> None:
        """Set the main button state: 'idle' or 'running'."""
        self._button_state = state
        if state == "idle":
            self._sync_start_button_readiness()
            self.status_badge.configure(text="● STOPPED", bootstyle="danger")
        elif state == "running":
            self.main_btn.configure(text=_HERO_STOP_TEXT, bootstyle="danger", state=tk.NORMAL)
            self.status_badge.configure(text="● RUNNING", bootstyle="success")

        # Disable/enable config widgets based on running state
        running = state == "running"
        for key, widget in self._config_widgets.items():
            if key == "main_btn":
                continue
            try:
                if isinstance(widget, ttk.Combobox):
                    if widget == getattr(self, "gp_device_serial_combo", None):
                        widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                    else:
                        widget.configure(state=tk.DISABLED if running else READONLY)
                elif isinstance(widget, ttk.Spinbox):
                    if widget in self._job_extra_spinboxes.values():
                        continue
                    widget.configure(state=tk.DISABLED if running else READONLY)
                elif isinstance(widget, ttk.Checkbutton):
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                elif isinstance(widget, ttk.Button):
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)

            except tk.TclError:
                continue
        self._sync_job_extra_spinboxes()
        if self._job_toggle_checkbuttons:
            self._sync_all_job_toggle_appearances()

    def get_button_state(self) -> str:
        """Get the current button state: 'idle' or 'running'."""
        return self._button_state

    def append_log(self, message: str) -> None:
        """Append a timestamped line to the terminal (consecutive dupes collapse)."""
        if message == self._log_last_message:
            return
        self._log_last_message = message
        try:
            self.log_box.configure(state="normal")
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_box.insert("end", f"[{timestamp}] {message}\n")
            lines = int(self.log_box.index("end-1c").split(".")[0])
            if lines > _TERMINAL_MAX_LINES:
                self.log_box.delete("1.0", f"{lines - _TERMINAL_MAX_LINES}.0")
            self.log_box.configure(state="disabled")
            self.log_box.see("end")
        except tk.TclError:
            pass

    def clear_logs(self) -> None:
        """Empty the terminal."""
        try:
            self._log_last_message = None
            self.log_box.configure(state="normal")
            self.log_box.delete("1.0", "end")
            self.log_box.configure(state="disabled")
        except tk.TclError:
            pass

    def set_status(self, text: str) -> None:
        self._status_text = text

    def update_stats(self, stats: dict[str, object] | None) -> None:
        if not stats:
            return

        def as_string(field: StatField, default: str = "0") -> str:
            value = stats.get(field.value, default)
            return str(value)

        def as_int(field: StatField) -> int:
            value = stats.get(field.value)
            try:
                return int(str(value))
            except (TypeError, ValueError):
                return 0

        for field, var in self.stat_labels.items():
            var.set(as_string(field))

        runtime = stats.get(BotStatField.TIME_SINCE_START.value)
        if runtime is not None:
            self.bot_labels[BotStatField.TIME_SINCE_START].set(str(runtime))
        failures = stats.get(BotStatField.RESTARTS_AFTER_FAILURE.value)
        if failures is not None:
            self.bot_labels[BotStatField.RESTARTS_AFTER_FAILURE].set(str(failures))

        winrate_raw = stats.get(DerivedStatField.WINRATE.value)
        wins = as_int(StatField.WINS)
        losses = as_int(StatField.LOSSES)
        parsed_winrate = self._parse_winrate_value(winrate_raw)
        winrate = parsed_winrate if parsed_winrate is not None else self._calculate_winrate_percentage(wins, losses)
        self.winrate_var.set(f"{winrate:.1f}%")

        # Update win streak stats
        current_streak = stats.get(DerivedStatField.CURRENT_WIN_STREAK.value, 0)
        best_streak = stats.get(DerivedStatField.BEST_WIN_STREAK.value, 0)
        if hasattr(self, "current_streak_var"):
            self.current_streak_var.set(str(current_streak))
        if hasattr(self, "best_streak_var"):
            self.best_streak_var.set(str(best_streak))

    def can_start(self) -> bool:
        """True when at least one battle job is enabled."""
        return has_start_ready_job(self.get_all_values())

    def _init_headless_emulator_state(self) -> None:
        """Backend defaults for the removed Emulator tab (Google Play only)."""
        self.emulator_var = ttk.StringVar(value=EmulatorType.GOOGLE_PLAY)
        self.gp_vars: dict[UIField, ttk.StringVar] = {
            config.key: ttk.StringVar(value=str(config.default)) for config in GOOGLE_PLAY_SETTINGS
        }
        self.gp_device_serial_var = ttk.StringVar(value=str(GOOGLE_PLAY_DEVICE_CONFIG.default))
        # No visible combobox in battle-only UI; kept as None for __main__ compat.
        self.gp_device_serial_combo = None  # type: ignore[assignment]
        self.emulator_settings_frames: dict[str, ttk.Frame] = {}
        self.settings_container = ttk.Frame(self)
        self.google_play_frame = ttk.Frame(self.settings_container)

    def _dashboard_card(self, parent: tk.Misc, title: str) -> ttk.Labelframe:
        """Section box with a title — same look across the dashboard."""
        return ttk.Labelframe(parent, text=title, padding=_CARD_PADDING)

    def _build_header(self) -> None:
        header = ttk.Frame(self, padding=(16, 12, 16, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title_box = ttk.Frame(header)
        title_box.grid(row=0, column=0, sticky="w")
        ttk.Label(title_box, text="CLASH ROYALE BOT", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(title_box, text="Google Play Games PC • Threat-Aware Engine", bootstyle="secondary").pack(anchor="w")

        right_box = ttk.Frame(header)
        right_box.grid(row=0, column=1, sticky="e")

        self.status_badge = ttk.Label(right_box, text="● STOPPED", font=("Segoe UI", 10, "bold"))
        self.status_badge.pack(side=LEFT, ipadx=10, ipady=4)

    def _build_stats_ribbon(self) -> None:
        ribbon = ttk.Frame(self, padding=(16, 2, 16, 6))
        ribbon.grid(row=1, column=0, sticky="ew")
        for index in range(4):
            ribbon.columnconfigure(index, weight=1)

        # Kept as an attribute for __main__ compat (it used to be a tab).
        self.stats_tab = ttk.Frame(ribbon)
        self.stats_tab.grid(row=0, column=0, columnspan=4, sticky="ew")
        for index in range(4):
            self.stats_tab.columnconfigure(index, weight=1)

        self.stat_labels: dict[StatField, ttk.StringVar] = {}
        hero_cards: tuple[tuple[str, StatField | None, str], ...] = (
            ("WINS", StatField.WINS, "success"),
            ("LOSSES", StatField.LOSSES, "danger"),
            ("WIN RATE", None, "info"),
            ("CARDS PLAYED", StatField.CARDS_PLAYED, "warning"),
        )
        for column, (title, field, style) in enumerate(hero_cards):
            card = self._dashboard_card(self.stats_tab, "")
            card.grid(row=0, column=column, sticky="ew", padx=4)
            ttk.Label(card, text=title, font=("Segoe UI", 8, "bold"), bootstyle="secondary").pack()
            if field is None:
                self.winrate_var = ttk.StringVar(value="0.0%")
                ttk.Label(card, textvariable=self.winrate_var, font=("Segoe UI", 16, "bold"), bootstyle=style).pack()
            else:
                var = ttk.StringVar(value="0")
                ttk.Label(card, textvariable=var, font=("Segoe UI", 16, "bold"), bootstyle=style).pack()
                self.stat_labels[field] = var

        self.bot_labels = {
            BotStatField.RESTARTS_AFTER_FAILURE: ttk.StringVar(value="0"),
            BotStatField.TIME_SINCE_START: ttk.StringVar(value="00:00:00"),
        }
        self.current_streak_var = ttk.StringVar(value="0")
        self.best_streak_var = ttk.StringVar(value="0")
        detail_specs: tuple[tuple[str, ttk.StringVar], ...] = (
            (BATTLE_STAT_LABELS[StatField.CLASSIC_1V1_FIGHTS], self._battle_stat_var(StatField.CLASSIC_1V1_FIGHTS)),
            (BATTLE_STAT_LABELS[StatField.CLASSIC_2V2_FIGHTS], self._battle_stat_var(StatField.CLASSIC_2V2_FIGHTS)),
            (
                BATTLE_STAT_LABELS[StatField.TROPHY_ROAD_1V1_FIGHTS],
                self._battle_stat_var(StatField.TROPHY_ROAD_1V1_FIGHTS),
            ),
            (BATTLE_STAT_LABELS[StatField.CARD_RANDOMIZATIONS], self._battle_stat_var(StatField.CARD_RANDOMIZATIONS)),
            (BATTLE_STAT_LABELS[StatField.CARD_CYCLES], self._battle_stat_var(StatField.CARD_CYCLES)),
            ("Current streak", self.current_streak_var),
            ("Best streak", self.best_streak_var),
            (
                BOT_STAT_LABELS[BotStatField.RESTARTS_AFTER_FAILURE],
                self.bot_labels[BotStatField.RESTARTS_AFTER_FAILURE],
            ),
            (BOT_STAT_LABELS[BotStatField.TIME_SINCE_START], self.bot_labels[BotStatField.TIME_SINCE_START]),
        )
        details = ttk.Frame(ribbon)
        details.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        for index in range(3):
            details.columnconfigure(index, weight=1)
        for index, (title, var) in enumerate(detail_specs):
            cell = ttk.Frame(details)
            cell.grid(row=index // 3, column=index % 3, sticky="ew", padx=4)
            ttk.Label(cell, text=f"{title}:").pack(side=LEFT)
            ttk.Label(cell, textvariable=var, bootstyle="info").pack(side=RIGHT)

    def _battle_stat_var(self, field: StatField) -> ttk.StringVar:
        var = ttk.StringVar(value="0")
        self.stat_labels[field] = var
        return var

    def _build_body(self) -> None:
        body = ttk.Frame(self, padding=(16, 2, 16, 6))
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        controls = ttk.Frame(body, width=300)
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        self._build_control_cards(controls)

        terminal_card = self._dashboard_card(body, "Live Battle Feed")
        terminal_card.grid(row=0, column=1, sticky="nsew")
        terminal_card.columnconfigure(0, weight=1)
        terminal_card.rowconfigure(0, weight=1)

        terminal_text_frame = ttk.Frame(terminal_card)
        terminal_text_frame.grid(row=0, column=0, sticky="nsew")
        terminal_text_frame.columnconfigure(0, weight=1)
        terminal_text_frame.rowconfigure(0, weight=1)

        self.log_box = tk.Text(terminal_text_frame, wrap=tk.WORD, height=12, state="disabled")
        self.log_box.grid(row=0, column=0, sticky="nsew")
        terminal_scroll = ttk.Scrollbar(terminal_text_frame, orient="vertical", command=self.log_box.yview)
        terminal_scroll.grid(row=0, column=1, sticky="ns")
        self.log_box.configure(yscrollcommand=terminal_scroll.set)
        self._style_terminal()

        terminal_bar = ttk.Frame(terminal_card)
        terminal_bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(terminal_bar, text="Clear", bootstyle="secondary", command=self.clear_logs).pack(side=LEFT)
        ttk.Button(
            terminal_bar, text="View Recordings", bootstyle="secondary", command=self._on_open_recordings_clicked
        ).pack(side=RIGHT, padx=(6, 0))
        ttk.Button(terminal_bar, text="View Logs", bootstyle="secondary", command=self._on_open_logs_clicked).pack(
            side=RIGHT
        )

    def _build_control_cards(self, parent: tk.Misc) -> None:
        job_defaults = {job.key: job.default for job in JOBS}
        jobs_by_key = {job.key: job for job in JOBS}
        self.jobs_vars: dict[UIField, ttk.BooleanVar] = {}

        for title, job_keys in _DASHBOARD_CARDS:
            card = self._dashboard_card(parent, title)
            card.pack(fill=X, pady=(0, 8))
            for job_key in job_keys:
                self._add_job_row(card, jobs_by_key[job_key], job_defaults)

        self._sync_all_job_toggle_appearances()

    def _add_job_row(self, parent: tk.Misc, job: JobConfig, job_defaults: dict[UIField, bool]) -> None:
        """One toggle row; deck-number pickers sit inline on the right."""
        row = ttk.Frame(parent)
        row.pack(fill=X, pady=2)
        row.columnconfigure(0, weight=1)

        var = ttk.BooleanVar(value=self._job_toggle_default(job.key, job_defaults))

        def on_toggle() -> None:
            self._sync_job_toggle_appearance(job.key)
            if job.extras:
                self._on_job_with_extra_toggle(job.key)
            else:
                self._notify_config_change()

        checkbox = ttk.Checkbutton(
            row,
            text=job.title,
            variable=var,
            bootstyle=_JOB_TOGGLE_ON_STYLE,
            command=on_toggle,
        )
        checkbox.grid(row=0, column=0, sticky="w")
        if job.tooltip:
            ToolTip(checkbox, job.tooltip)
        self.jobs_vars[job.key] = var
        self._job_toggle_checkbuttons[job.key] = checkbox
        self._trace_variable(var)
        self._register_config_widget(job.key.value, checkbox)
        self._sync_job_toggle_appearance(job.key)

        if job.extras:
            combo_config = next(iter(job.extras.values()))
            combo_field = combo_config.key
            extras = ttk.Frame(row)
            extras.grid(row=0, column=1, sticky="e")
            short_label = _JOB_SPINBOX_LABELS.get(combo_field, combo_config.label)
            label = ttk.Label(extras, text=short_label)
            label.pack(side=LEFT, padx=(0, 4))
            self._job_row_extras.setdefault(job.key, []).append(label)

            spin_var = ttk.StringVar(value=str(combo_config.default))
            spinbox = ttk.Spinbox(
                extras,
                from_=int(min(combo_config.values)),
                to=int(max(combo_config.values)),
                width=3,
                textvariable=spin_var,
                command=self._notify_config_change,
                state=READONLY,
            )
            spinbox.pack(side=LEFT)
            self._trace_variable(spin_var)
            self._register_config_widget(combo_field.value, spinbox)
            self._job_extra_spinboxes[job.key] = spinbox
            self._job_row_extras.setdefault(job.key, []).append(spinbox)

            if combo_config.tooltip:
                info_label = ttk.Label(extras, text="ⓘ", bootstyle="info")
                info_label.pack(side=LEFT, padx=(4, 0))
                ToolTip(info_label, combo_config.tooltip)
                self._job_row_extras.setdefault(job.key, []).append(info_label)

            if combo_field == UIField.DECK_NUMBER_SELECTION:
                self.deck_var = spin_var
            elif combo_field == UIField.MAX_DECK_SELECTION:
                self.max_deck_var = spin_var
            self._sync_job_extra_spinboxes()

    def _build_hero_button(self) -> None:
        bottom = ttk.Frame(self, padding=(16, 0, 16, 14))
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ipadx, ipady = _ACTION_BUTTON_IPAD

        self.main_btn = ttk.Button(bottom, text=_HERO_START_TEXT, bootstyle="success")
        self.main_btn.grid(row=0, column=0, sticky="ew", ipadx=ipadx, ipady=ipady)
        self._register_config_widget("main_btn", self.main_btn)

    def _sync_start_button_readiness(self) -> None:
        """Enable Start only when at least one battle job is selected."""
        if self._button_state != "idle":
            return
        if self.can_start():
            self.main_btn.configure(text=_HERO_START_TEXT, bootstyle="success", state=tk.NORMAL)
            self.append_log(_IDLE_STATUS)
        else:
            self.main_btn.configure(text=_HERO_START_TEXT, bootstyle="secondary", state=tk.DISABLED)
            self.append_log(_START_BLOCKED_MESSAGE)

    def _on_job_with_extra_toggle(self, job_key: UIField) -> None:
        self._sync_job_toggle_appearance(job_key)
        self._sync_job_extra_spinboxes()
        self._notify_config_change()

    def _sync_job_extra_spinboxes(self) -> None:
        bot_running = self._button_state == "running"
        for job_key, spinbox in self._job_extra_spinboxes.items():
            job_on = bool(self.jobs_vars[job_key].get())
            if bot_running or not job_on:
                spinbox.configure(state=tk.DISABLED)
            else:
                spinbox.configure(state=READONLY)

    def _sync_job_toggle_appearance(self, field: UIField) -> None:
        """Swap toggle bootstyle when turned off."""
        checkbox = self._job_toggle_checkbuttons.get(field)
        var = self.jobs_vars.get(field)
        if not checkbox or not var:
            return
        try:
            checkbox.configure(bootstyle=_JOB_TOGGLE_ON_STYLE if bool(var.get()) else _JOB_TOGGLE_OFF_STYLE)
        except tk.TclError:
            pass

    def _sync_all_job_toggle_appearances(self) -> None:
        for field in self._job_toggle_checkbuttons:
            self._sync_job_toggle_appearance(field)

    @staticmethod
    def _job_toggle_default(field: UIField, job_defaults: dict[UIField, bool]) -> bool:
        """Map stored job defaults to what the toggle should show."""
        stored = job_defaults.get(field, False)
        if field == _WIN_TRACK_UI_FIELD:
            return not stored
        return stored

    def _register_config_widget(self, key: str, widget: tk.Widget) -> None:
        self._config_widgets[key] = widget

    def _notify_config_change(self, *_: object) -> None:
        if self._suspend_traces > 0 or self._config_callback is None:
            return
        if self._button_state == "idle":
            self._sync_start_button_readiness()
        callback = self._config_callback
        self.after_idle(lambda: callback(self.get_all_values()))

    def _trace_variable(self, var: tk.Variable) -> None:
        trace_id = var.trace_add("write", self._notify_config_change)
        self._traces.append((var, trace_id))

    def _update_google_play_comboboxes(self) -> None:
        for field, var in self.gp_vars.items():
            widget = self._config_widgets.get(field.value)
            if not widget:
                continue
            values = [str(option) for option in widget.cget("values")]
            if var.get() not in values and values:
                var.set(values[0])

    def _show_current_emulator_settings(self) -> None:
        """Headless no-op: Google Play Games is the only backend (no tab)."""
        show_advanced = bool(self.advanced_settings_var.get())

        for frame in self.emulator_settings_frames.values():
            frame.pack_forget()

        frame_to_show = self.emulator_settings_frames.get(EmulatorType.GOOGLE_PLAY)

        if frame_to_show and show_advanced:
            self.settings_container.pack(fill=X, anchor="n", pady=(0, 6))
            frame_to_show.pack(fill=X, anchor="n")
        else:
            self.settings_container.pack_forget()

        self.settings_container.update_idletasks()

    def _on_open_logs_clicked(self) -> None:
        if self._open_logs_callback:
            self._open_logs_callback()

    def _on_open_recordings_clicked(self) -> None:
        if self._open_recordings_callback:
            self._open_recordings_callback()

    def _stat_accent_foreground(self) -> str:
        colors = getattr(self._style, "colors", None)
        if colors is not None:
            accent = getattr(colors, "info", "")
            if accent:
                return accent
        try:
            return self._style.lookup("TLabel", "foreground") or "#202020"
        except tk.TclError:
            return "#202020"

    def _style_terminal(self) -> None:
        if not hasattr(self, "log_box"):
            return
        try:
            background = self._style.lookup("TEntry", "fieldbackground")
            foreground = self._style.lookup("TEntry", "foreground")
        except tk.TclError:
            return

        colors = getattr(self._style, "colors", None)
        if not background and colors is not None:
            background = getattr(colors, "input", "") or getattr(colors, "bg", "")
        if not foreground and colors is not None:
            foreground = getattr(colors, "fg", "")

        if not background or not foreground:
            return

        try:
            self.log_box.configure(
                background=background,
                foreground=foreground,
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                insertbackground=foreground,
            )
        except tk.TclError:
            pass

    def _app_icon_path(self) -> Path | None:
        relative = Path("assets") / _APP_ICON_NAME
        candidates: list[Path] = []
        if getattr(sys, "frozen", False):
            bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
            candidates.extend((bundle_root / relative, Path(sys.executable).parent / relative))
        candidates.append(Path(__file__).resolve().parents[2] / relative)
        return next((path for path in candidates if path.is_file()), None)

    def _set_window_icon(self) -> None:
        icon_path = self._app_icon_path()
        if icon_path is None:
            return
        if is_windows():
            try:
                self.iconbitmap(str(icon_path))
            except tk.TclError:
                return
            return
        try:
            from PIL import Image, ImageTk

            with Image.open(icon_path) as source:
                resized = source.resize((64, 64), Image.Resampling.LANCZOS)
                self._window_icon = ImageTk.PhotoImage(resized)
            # PIL ImageTk.PhotoImage is _PhotoImageLike-compatible but not in tkinter stubs.
            self.iconphoto(True, self._window_icon)  # ty: ignore[invalid-argument-type]
        except (ImportError, OSError, tk.TclError):
            return

    @staticmethod
    def _safe_int(value: object, fallback: int = 0) -> int:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _parse_winrate_value(raw: object) -> float | None:
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped.endswith("%"):
                stripped = stripped[:-1]
            try:
                return float(stripped)
            except ValueError:
                return None
        if isinstance(raw, int | float):
            return float(raw)
        return None

    @staticmethod
    def _calculate_winrate_percentage(wins: int, losses: int) -> float:
        total = wins + losses
        if total <= 0:
            return 0.0
        return wins / total * 100
