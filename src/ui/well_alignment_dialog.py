"""Well alignment dialog — monthly production vs months since first production.

Each well's X-axis starts at month 1 = its own first month with positive
production.  This lets users compare decline profiles regardless of when each
well was brought online.

New in this version:
  - Scenario persistence: save / load / manage well-analysis scenarios.
  - Prev / next navigation buttons for single-well step-through.
  - Eraser: draw a lasso around data points to exclude them from analysis
    and P90/P50/P10.  Each lasso operation is a single undo step.
    Excluded points are shown as red x markers and stored per scenario.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.data.models import (
    COL_DATE, COL_GAS, COL_HOURS_WORK, COL_OIL, COL_WATER, COL_WELL,
    COL_WORK_TYPE, WellAnalysisScenario, WORK_TYPE_OIL,
)

# 20-colour palette
try:
    from matplotlib import colormaps as _CMS
    _COLORS: list = list(_CMS["tab20"].colors)
except Exception:
    _COLORS = [f"C{i}" for i in range(10)]

# Petroleum convention: P10 = optimistic (high, 90th pctl), P90 = conservative (low, 10th pctl)
_PCT_COLORS = {10: "#d62728", 50: "#222222", 90: "#1f77b4", -1: "#2ca02c"}
_PCT_LABELS = {10: "P90", 50: "P50", 90: "P10", -1: "Среднее"}


class WellAlignmentDialog(QDialog):
    """Adjusted production dialog with scenarios, prev/next nav, and eraser."""

    def __init__(
        self,
        df: pd.DataFrame,
        well_analysis_scenarios: list | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Приведённая добыча по скважинам")
        self.resize(1340, 720)
        self._df = df

        # Scenario state
        self._scenarios: list[WellAnalysisScenario] = list(
            well_analysis_scenarios or []
        )
        self._active_idx: int = 0

        # Per-session working state
        self._phase: str = "oil"
        self._wells: list[str] = []
        self._excluded: set[tuple[str, str]] = set()
        # Each entry is one undo batch (all keys excluded by one lasso operation)
        self._exclusion_history: list[list[tuple[str, str]]] = []

        # Percentile state
        self._show_pct: bool = False
        self._pct_months: np.ndarray | None = None
        self._pct_data: dict | None = None
        self._pct_trends: dict = {}

        # Canvas lookup rebuilt on every _draw()
        # well -> list of (x_month, y_raw_value, iso_date_str)
        self._well_points: dict[str, list[tuple[float, float, str]]] = {}

        # Eraser state
        self._eraser_active: bool = False
        self._lasso_selector = None  # matplotlib LassoSelector when active

        self._build_ui()

        self._wells = self._producing_wells()
        self._populate_well_list()
        if self._scenarios:
            self._load_scenario(0)
        else:
            self._draw()

    # ── Public API ──────────────────────────────────────────────────────────

    def result_scenarios(self) -> list[WellAnalysisScenario]:
        return self._scenarios

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setFixedWidth(276)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(5)
        splitter.addWidget(left)

        # Scenario controls
        sc_row = QHBoxLayout()
        self._btn_scenarios = QPushButton("Сценарии\u2026")
        self._btn_save_sc = QPushButton("Сохранить")
        sc_row.addWidget(self._btn_scenarios)
        sc_row.addWidget(self._btn_save_sc)
        left_lay.addLayout(sc_row)

        self._lbl_sc = QLabel("")
        self._lbl_sc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_sc.setStyleSheet("font-size: 9px; color: #555;")
        left_lay.addWidget(self._lbl_sc)

        # Phase
        phase_row = QHBoxLayout()
        phase_row.addWidget(QLabel("Флюид:"))
        self._cmb_phase = QComboBox()
        self._cmb_phase.addItem("Нефть", "oil")
        self._cmb_phase.addItem("Газ",   "gas")
        phase_row.addWidget(self._cmb_phase)
        phase_row.addStretch()
        left_lay.addLayout(phase_row)

        # Data mode (production vs daily rate)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Режим:"))
        self._cmb_mode = QComboBox()
        self._cmb_mode.addItem("Добыча", "production")
        self._cmb_mode.addItem("Дебит", "rate")
        self._cmb_mode.setToolTip(
            "Добыча — месячная добыча (т/мес или м³/мес).\n"
            "Дебит — суточный дебит = добыча / (часы работы / 24)."
        )
        mode_row.addWidget(self._cmb_mode)
        mode_row.addStretch()
        left_lay.addLayout(mode_row)

        # Well list group
        grp_wells = QGroupBox("Скважины")
        gw_lay = QVBoxLayout(grp_wells)
        self._lst = QListWidget()
        self._lst.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        gw_lay.addWidget(self._lst)

        row_sel = QHBoxLayout()
        btn_all = QPushButton("Все")
        btn_none = QPushButton("Снять")
        btn_all.clicked.connect(self._lst.selectAll)
        btn_none.clicked.connect(self._lst.clearSelection)
        row_sel.addWidget(btn_all)
        row_sel.addWidget(btn_none)
        row_sel.addStretch()
        gw_lay.addLayout(row_sel)

        # Prev/next navigation
        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("\u25c0")
        self._btn_next = QPushButton("\u25b6")
        self._btn_prev.setFixedWidth(36)
        self._btn_next.setFixedWidth(36)
        self._btn_prev.setToolTip("Предыдущая скважина")
        self._btn_next.setToolTip("Следующая скважина")
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        nav_row.addStretch()
        btn_filter = QPushButton("Список\u2026")
        btn_filter.clicked.connect(self._load_filter)
        nav_row.addWidget(btn_filter)
        gw_lay.addLayout(nav_row)

        left_lay.addWidget(grp_wells)

        self._chk_log = QCheckBox("Log шкала (Y)")
        left_lay.addWidget(self._chk_log)

        self._chk_normalize = QCheckBox("Нормализовать")
        self._chk_normalize.setToolTip(
            "Разделить данные каждой скважины на первое\n"
            "ненулевое значение (после фильтров)."
        )
        left_lay.addWidget(self._chk_normalize)

        # Eraser
        eraser_row = QHBoxLayout()
        self._btn_eraser = QPushButton("Ластик")
        self._btn_eraser.setCheckable(True)
        self._btn_undo_excl = QPushButton("Отменить")
        self._btn_undo_excl.setEnabled(False)
        eraser_row.addWidget(self._btn_eraser)
        eraser_row.addWidget(self._btn_undo_excl)
        eraser_row.addStretch()
        left_lay.addLayout(eraser_row)

        self._lbl_excl = QLabel("")
        self._lbl_excl.setStyleSheet("font-size: 9px; color: #a00;")
        left_lay.addWidget(self._lbl_excl)

        # DCA model choice for percentile trendlines
        pct_model_row = QHBoxLayout()
        pct_model_row.addWidget(QLabel("Модель:"))
        self._cmb_pct_model = QComboBox()
        self._cmb_pct_model.addItem("Экспоненциальная", "exp")
        self._cmb_pct_model.addItem("Гиперболическая", "hyp")
        self._cmb_pct_model.addItem("Гармоническая", "har")
        self._cmb_pct_model.addItem("Лучший R\u00b2", "best")
        pct_model_row.addWidget(self._cmb_pct_model)
        pct_model_row.addStretch()
        left_lay.addLayout(pct_model_row)

        # Profile checkboxes
        prof_row = QHBoxLayout()
        self._chk_p10 = QCheckBox("P10")
        self._chk_p50 = QCheckBox("P50")
        self._chk_avg = QCheckBox("Средн.")
        self._chk_p90 = QCheckBox("P90")
        self._chk_p10.setChecked(True)
        self._chk_p50.setChecked(True)
        self._chk_avg.setChecked(True)
        self._chk_p90.setChecked(True)
        for cb in (self._chk_p10, self._chk_p50, self._chk_avg, self._chk_p90):
            prof_row.addWidget(cb)
        prof_row.addStretch()
        left_lay.addLayout(prof_row)

        self._btn_pct = QPushButton("Аппроксимировать")
        left_lay.addWidget(self._btn_pct)

        # Results box
        grp_res = QGroupBox("Результаты")
        res_lay = QVBoxLayout(grp_res)
        self._txt_result = QTextEdit()
        self._txt_result.setReadOnly(True)
        self._txt_result.setMaximumHeight(140)
        res_lay.addWidget(self._txt_result)
        left_lay.addWidget(grp_res)

        # Data filters
        grp_flt = QGroupBox("Фильтр данных")
        flt_lay = QVBoxLayout(grp_flt)
        flt_lay.setSpacing(3)

        row_rate = QHBoxLayout()
        row_rate.addWidget(QLabel("Мин. дебит:"))
        self._spn_min_rate = QDoubleSpinBox()
        self._spn_min_rate.setRange(0.0, 999999.0)
        self._spn_min_rate.setDecimals(1)
        self._spn_min_rate.setValue(1.0)
        self._spn_min_rate.setToolTip(
            "Месяцы с дебитом ниже этого значения исключаются из анализа."
        )
        row_rate.addWidget(self._spn_min_rate)
        flt_lay.addLayout(row_rate)

        row_days = QHBoxLayout()
        row_days.addWidget(QLabel("Мин. дней добычи:"))
        self._spn_min_days = QDoubleSpinBox()
        self._spn_min_days.setRange(0.0, 31.0)
        self._spn_min_days.setDecimals(1)
        self._spn_min_days.setValue(1.0)
        self._spn_min_days.setToolTip(
            "Месяцы с числом рабочих дней (часы работы / 24) "
            "ниже этого значения исключаются из анализа."
        )
        row_days.addWidget(self._spn_min_days)
        flt_lay.addLayout(row_days)

        left_lay.addWidget(grp_flt)

        left_lay.addStretch()

        # Right (plot)
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._fig = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._nav = NavigationToolbar2QT(self._canvas, right)
        right_lay.addWidget(self._nav)
        right_lay.addWidget(self._canvas)

        from src.ui.hover_tooltip import install_hover_tooltip
        install_hover_tooltip(self._canvas, self._fig)

        btns = QHBoxLayout()
        btn_clip = QPushButton("Копировать график")
        btn_save_img = QPushButton("Сохранить картинку\u2026")
        btn_data = QPushButton("Скопировать данные")
        btn_clip.clicked.connect(self._to_clipboard)
        btn_save_img.clicked.connect(self._save_image)
        btn_data.clicked.connect(self._copy_data)
        for b in (btn_clip, btn_save_img, btn_data):
            btns.addWidget(b)
        btns.addStretch()
        right_lay.addLayout(btns)

        # Connections
        self._lst.itemSelectionChanged.connect(self._on_selection_changed)
        self._chk_log.stateChanged.connect(self._draw)
        self._chk_normalize.stateChanged.connect(self._draw)
        self._cmb_phase.currentIndexChanged.connect(self._on_phase_changed)
        self._cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self._btn_pct.clicked.connect(self._generate_percentiles)
        self._btn_prev.clicked.connect(self._on_prev_well)
        self._btn_next.clicked.connect(self._on_next_well)
        self._btn_eraser.toggled.connect(self._on_eraser_toggled)
        self._btn_undo_excl.clicked.connect(self._on_undo_exclusion)
        self._btn_scenarios.clicked.connect(self._on_open_scenarios)
        self._btn_save_sc.clicked.connect(self._on_save_scenario)
        self._spn_min_rate.valueChanged.connect(self._on_filter_changed)
        self._spn_min_days.valueChanged.connect(self._on_filter_changed)

    # ── Scenario management ─────────────────────────────────────────────────

    def _scenario_label(self) -> str:
        if not self._scenarios or self._active_idx >= len(self._scenarios):
            return ""
        return self._scenarios[self._active_idx].name

    def _update_scenario_label(self) -> None:
        name = self._scenario_label()
        self._lbl_sc.setText(f"Сценарий: {name}" if name else "")

    def _load_scenario(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._scenarios):
            return
        self._active_idx = idx
        sc = self._scenarios[idx]

        self._phase = sc.phase
        ci = self._cmb_phase.findData(sc.phase)
        self._cmb_phase.blockSignals(True)
        self._cmb_phase.setCurrentIndex(max(0, ci))
        self._cmb_phase.blockSignals(False)

        self._wells = self._producing_wells()
        self._populate_well_list()

        well_set = set(sc.wells)
        self._lst.blockSignals(True)
        for i in range(self._lst.count()):
            item = self._lst.item(i)
            if item:
                item.setSelected(item.text() in well_set)
        self._lst.blockSignals(False)

        self._excluded = {tuple(pair) for pair in sc.excluded}
        self._exclusion_history = []

        if sc.pct_months:
            self._pct_months = np.array(sc.pct_months, dtype=float)
            self._pct_data = {
                int(k): np.array(v, dtype=float)
                for k, v in sc.pct_data.items()
            }
            self._pct_trends = {
                int(k): (tuple(v) if v else None)
                for k, v in sc.pct_trends.items()
            }
            self._show_pct = True
        else:
            self._pct_months = None
            self._pct_data = None
            self._pct_trends = {}
            self._show_pct = False

        self._update_scenario_label()
        self._update_exclusion_ui()
        self._update_nav_buttons()
        self._draw()

    def _collect_current_state(self) -> dict:
        selected = [item.text() for item in self._lst.selectedItems()]
        excl_list = [list(pair) for pair in sorted(self._excluded)]
        pct_months_list: list[float] = []
        pct_data_dict: dict = {}
        pct_trends_dict: dict = {}
        if self._pct_months is not None and self._pct_data is not None:
            pct_months_list = self._pct_months.tolist()
            pct_data_dict = {str(k): v.tolist() for k, v in self._pct_data.items()}
            pct_trends_dict = {
                str(k): list(v) if v else None
                for k, v in self._pct_trends.items()
            }
        return dict(
            wells=selected, phase=self._phase, excluded=excl_list,
            pct_months=pct_months_list, pct_data=pct_data_dict,
            pct_trends=pct_trends_dict,
        )

    def _apply_state_to_scenario(self, sc: WellAnalysisScenario) -> None:
        state = self._collect_current_state()
        sc.wells = state["wells"]
        sc.phase = state["phase"]
        sc.excluded = state["excluded"]
        sc.pct_months = state["pct_months"]
        sc.pct_data = state["pct_data"]
        sc.pct_trends = state["pct_trends"]

    def _on_open_scenarios(self) -> None:
        from src.ui.well_analysis_scenario_dialog import WellAnalysisScenarioDialog
        dlg = WellAnalysisScenarioDialog(
            scenarios=self._scenarios,
            active_idx=self._active_idx,
            parent=self,
        )
        dlg.scenario_activated.connect(self._on_scenario_activated)
        dlg.exec()
        self._scenarios = dlg.result_scenarios()
        self._active_idx = min(
            dlg.result_active_idx(), max(0, len(self._scenarios) - 1)
        )
        self._update_scenario_label()

    def _on_scenario_activated(self, idx: int) -> None:
        # Ask to save the current scenario before switching
        if self._scenarios and self._active_idx < len(self._scenarios):
            selected = [item.text() for item in self._lst.selectedItems()]
            has_work = bool(selected) or bool(self._excluded) or self._show_pct
            if has_work:
                sc_name = self._scenario_label() or "Текущий"
                reply = QMessageBox.question(
                    self, "Смена сценария",
                    f"Сохранить сценарий \u00ab{sc_name}\u00bb?",
                    QMessageBox.StandardButton.Save |
                    QMessageBox.StandardButton.Discard,
                    QMessageBox.StandardButton.Save,
                )
                if reply == QMessageBox.StandardButton.Save:
                    self._apply_state_to_scenario(
                        self._scenarios[self._active_idx]
                    )
        self._load_scenario(idx)

    def _scenario_summary(self) -> str:
        """Build a short text describing the current scenario setup."""
        selected = [item.text() for item in self._lst.selectedItems()]
        phase_lbl = "Нефть" if self._phase == "oil" else "Газ"
        mode_lbl = "Добыча" if self._cmb_mode.currentData() == "production" else "Дебит"
        lines = [
            f"Скважины: {len(selected)}",
            f"Флюид: {phase_lbl}",
            f"Режим: {mode_lbl}",
            f"Исключено точек: {len(self._excluded)}",
            f"Нормализация: {'\u0414\u0430' if self._chk_normalize.isChecked() else '\u041d\u0435\u0442'}",
            f"P10/P50/P90: {'\u0414\u0430' if self._show_pct else '\u041d\u0435\u0442'}",
        ]
        if self._show_pct:
            model_names = {"exp": "Эксп.", "hyp": "Гиперб.", "har": "Гарм."}
            lines.append(f"  Модель: {model_names.get(self._cmb_pct_model.currentData(), '?')}")
        flt_parts = []
        if self._spn_min_rate.value() > 0:
            flt_parts.append(f"мин. дебит {self._spn_min_rate.value():.1f}")
        if self._spn_min_days.value() > 0:
            flt_parts.append(f"мин. дней {self._spn_min_days.value():.1f}")
        if flt_parts:
            lines.append(f"Фильтры: {', '.join(flt_parts)}")
        return "\n".join(lines)

    def _on_save_scenario(self) -> None:
        if not self._scenarios:
            name, ok = QInputDialog.getText(
                self, "Сохранить сценарий", "Название:", text="Анализ 1"
            )
            if not ok or not name.strip():
                return
            sc = WellAnalysisScenario(name=name.strip())
            self._scenarios.append(sc)
            self._active_idx = len(self._scenarios) - 1
        else:
            sc = self._scenarios[self._active_idx]

        # Show confirmation with scenario summary
        summary = self._scenario_summary()
        reply = QMessageBox.question(
            self, f"Сохранить сценарий \u00ab{sc.name}\u00bb?",
            f"Параметры сценария:\n\n{summary}",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply != QMessageBox.StandardButton.Save:
            # If it was a new scenario, remove it
            if len(self._scenarios) > 0 and self._scenarios[-1] is sc and not sc.wells:
                self._scenarios.pop()
                self._active_idx = max(0, len(self._scenarios) - 1)
            return

        self._apply_state_to_scenario(sc)
        self._update_scenario_label()
        QMessageBox.information(
            self, "Сохранено",
            f"Сценарий \u00ab{sc.name}\u00bb сохранён."
        )

    def closeEvent(self, event) -> None:
        """Ask to save the current scenario on dialog close."""
        selected = [item.text() for item in self._lst.selectedItems()]
        has_work = bool(selected) or bool(self._excluded) or self._show_pct
        if not has_work:
            event.accept()
            return

        sc_name = self._scenario_label() or "Текущий сценарий"
        summary = self._scenario_summary()
        reply = QMessageBox.question(
            self, "Закрытие",
            f"Сохранить сценарий \u00ab{sc_name}\u00bb перед закрытием?\n\n{summary}",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if reply == QMessageBox.StandardButton.Save:
            if not self._scenarios:
                name, ok = QInputDialog.getText(
                    self, "Сохранить сценарий", "Название:", text="Анализ 1"
                )
                if not ok or not name.strip():
                    event.ignore()
                    return
                sc = WellAnalysisScenario(name=name.strip())
                self._scenarios.append(sc)
                self._active_idx = len(self._scenarios) - 1
            else:
                sc = self._scenarios[self._active_idx]
            self._apply_state_to_scenario(sc)
        event.accept()

    # ── Filter criteria ─────────────────────────────────────────────────────

    def _on_filter_changed(self) -> None:
        """Redraw when min-rate or min-days criteria change."""
        self._clear_percentiles()
        self._draw()

    # ── Mode ────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, _idx: int) -> None:
        self._clear_percentiles()
        self._draw()

    # ── Phase ───────────────────────────────────────────────────────────────

    def _on_phase_changed(self, _idx: int) -> None:
        self._phase = self._cmb_phase.currentData()
        self._wells = self._producing_wells()
        self._populate_well_list()
        self._excluded.clear()
        self._exclusion_history = []
        self._clear_percentiles()
        self._update_exclusion_ui()
        self._draw()

    # ── Well list helpers ───────────────────────────────────────────────────

    def _populate_well_list(self) -> None:
        self._lst.blockSignals(True)
        self._lst.clear()
        for w in self._wells:
            self._lst.addItem(QListWidgetItem(w))
        self._lst.blockSignals(False)

    def _on_selection_changed(self) -> None:
        self._clear_percentiles()
        self._update_nav_buttons()
        self._draw()

    def _update_nav_buttons(self) -> None:
        enabled = len(self._lst.selectedItems()) == 1
        self._btn_prev.setEnabled(enabled)
        self._btn_next.setEnabled(enabled)

    def _on_prev_well(self) -> None:
        self._step_well(-1)

    def _on_next_well(self) -> None:
        self._step_well(+1)

    def _step_well(self, delta: int) -> None:
        items = self._lst.selectedItems()
        if len(items) != 1:
            return
        n = self._lst.count()
        if n == 0:
            return
        new_row = (self._lst.row(items[0]) + delta) % n
        self._lst.blockSignals(True)
        self._lst.clearSelection()
        item = self._lst.item(new_row)
        if item:
            item.setSelected(True)
        self._lst.blockSignals(False)
        self._lst.setCurrentRow(new_row)
        self._on_selection_changed()

    # ── Exclusion / eraser ──────────────────────────────────────────────────

    def _update_exclusion_ui(self) -> None:
        n = len(self._excluded)
        self._lbl_excl.setText(f"Исключено: {n} точек" if n else "")
        self._btn_undo_excl.setEnabled(bool(self._exclusion_history))

    def _on_eraser_toggled(self, active: bool) -> None:
        self._eraser_active = active
        if active:
            # Deactivate any active toolbar tool (pan / zoom)
            try:
                if self._nav.mode:
                    self._nav.mode = ""
            except Exception:
                pass
            self._attach_eraser_lasso()
        else:
            self._detach_eraser_lasso()

    def _attach_eraser_lasso(self) -> None:
        """Attach a LassoSelector to the current plot axes."""
        self._detach_eraser_lasso()
        if not self._fig.axes:
            return
        ax = self._fig.axes[0]
        from matplotlib.widgets import LassoSelector
        self._lasso_selector = LassoSelector(
            ax, self._on_lasso_select_eraser, useblit=True
        )

    def _detach_eraser_lasso(self) -> None:
        """Disconnect and clear the active LassoSelector."""
        if self._lasso_selector is not None:
            try:
                self._lasso_selector.disconnect_events()
            except Exception:
                pass
            self._lasso_selector = None

    def _on_lasso_select_eraser(self, vertices) -> None:
        """Exclude all data points that fall inside the drawn lasso polygon."""
        if len(vertices) < 3:
            return
        from matplotlib.path import Path
        path = Path(vertices)

        batch: list[tuple[str, str]] = []
        for well, pts in self._well_points.items():
            for x_m, y_v, iso_date in pts:
                key = (well, iso_date)
                if key not in self._excluded and path.contains_point([x_m, y_v]):
                    self._excluded.add(key)
                    batch.append(key)

        if batch:
            self._exclusion_history.append(batch)
            self._update_exclusion_ui()
            self._clear_percentiles()
            self._draw()   # _draw() re-attaches the lasso when eraser is active

    def _on_undo_exclusion(self) -> None:
        """Undo the last lasso-exclusion batch."""
        if not self._exclusion_history:
            return
        batch = self._exclusion_history.pop()
        for key in batch:
            self._excluded.discard(key)
        self._update_exclusion_ui()
        self._clear_percentiles()
        self._draw()   # _draw() re-attaches the lasso when eraser is active

    # ── Filter ──────────────────────────────────────────────────────────────

    def _load_filter(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить список скважин", "",
            "Текстовые файлы (*.txt);;Все файлы (*)",
        )
        if not path:
            return
        names = self._read_well_list(path)
        if not names:
            return
        name_set = {n.lower() for n in names}
        self._lst.blockSignals(True)
        self._lst.clearSelection()
        for i in range(self._lst.count()):
            item = self._lst.item(i)
            if item and item.text().lower() in name_set:
                item.setSelected(True)
        self._lst.blockSignals(False)
        self._lst.itemSelectionChanged.emit()

    @staticmethod
    def _read_well_list(path: str) -> list[str]:
        names: list[str] = []
        for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                with open(path, encoding=enc) as fh:
                    for line in fh:
                        name = line.strip()
                        if name and not name.startswith("#"):
                            names.append(name)
                return names
            except (UnicodeDecodeError, LookupError):
                continue
        return names

    # ── Percentile generation ───────────────────────────────────────────────

    def _clear_percentiles(self) -> None:
        self._show_pct = False
        self._pct_months = None
        self._pct_data = None
        self._pct_trends = {}
        self._btn_pct.setText("Аппроксимировать")
        self._txt_result.clear()

    def _generate_percentiles(self) -> None:
        """Toggle P90/P50/P10.  Excluded and zero values are skipped.

        The percentiles are computed on the same data the user sees on the
        plot: daily rate when mode is 'rate', and normalised values when
        the normalisation checkbox is active.
        """
        if self._show_pct:
            self._clear_percentiles()
            self._draw()
            return
        selected = [item.text() for item in self._lst.selectedItems()]
        if len(selected) < 3:
            QMessageBox.warning(
                self, "Недостаточно скважин",
                f"Необходимо не менее 3 скважин.\nВыбрано: {len(selected)}.",
            )
            return
        do_normalize = self._chk_normalize.isChecked()
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        max_month = 0
        for well in selected:
            result = self._aligned_series(well)
            if result is None:
                continue
            x, y, _ = result
            if not np.any(np.isfinite(y) & (y > 0)):
                continue
            # Apply normalization so percentiles match the plotted data
            if do_normalize:
                finite = y[np.isfinite(y) & (y > 0)]
                if len(finite) > 0:
                    y = y / finite[0]
            series[well] = (x, y)
            max_month = max(max_month, int(x[-1]))
        if not series:
            QMessageBox.warning(self, "Нет данных",
                                "Нет скважин с ненулевой добычей после исключений.")
            return
        # Determine which profiles to generate
        gen_p10 = self._chk_p10.isChecked()  # P10 = 90th percentile (high)
        gen_p50 = self._chk_p50.isChecked()
        gen_avg = self._chk_avg.isChecked()
        gen_p90 = self._chk_p90.isChecked()  # P90 = 10th percentile (low)
        if not (gen_p10 or gen_p50 or gen_avg or gen_p90):
            QMessageBox.warning(self, "Нет профилей",
                                "Выберите хотя бы один профиль для генерации.")
            return

        months_list: list[float] = []
        p_vals: dict[int, list[float]] = {}
        if gen_p90: p_vals[10] = []
        if gen_p50: p_vals[50] = []
        if gen_p10: p_vals[90] = []
        if gen_avg: p_vals[-1] = []
        for m in range(1, max_month + 1):
            vals = []
            for x, y in series.values():
                if m <= len(y):
                    v = y[m - 1]
                    if np.isfinite(v) and v > 0:
                        vals.append(v)
            if len(vals) >= 3:
                arr = np.array(vals)
                months_list.append(float(m))
                if gen_p90: p_vals[10].append(float(np.percentile(arr, 10)))
                if gen_p50: p_vals[50].append(float(np.percentile(arr, 50)))
                if gen_p10: p_vals[90].append(float(np.percentile(arr, 90)))
                if gen_avg: p_vals[-1].append(float(arr.mean()))
        if not months_list:
            QMessageBox.warning(self, "Нет данных",
                                "Нет месяцев с \u2265 3 скважинами с ненулевой добычей.")
            return
        self._pct_months = np.array(months_list)
        self._pct_data = {k: np.array(v) for k, v in p_vals.items()}
        self._pct_trends = {
            k: self._fit_trend(self._pct_months, self._pct_data[k])
            for k in p_vals
        }
        self._show_pct = True
        self._btn_pct.setText("Отменить")
        self._update_results_text()
        self._draw()

    def _fit_single_model(self, months: np.ndarray, values: np.ndarray, model: str) -> tuple | None:
        """Fit one DCA model. Returns params tuple or None."""
        mask = values > 0
        if np.sum(mask) < 3:
            return None
        t = months[mask]
        q = values[mask]
        from scipy.optimize import curve_fit
        try:
            if model == "hyp":
                def _hyp(t, qi, Di, b):
                    base = 1.0 + b * Di * t
                    return qi / np.power(np.where(base > 0, base, 1e-12), 1.0 / b)
                popt, _ = curve_fit(
                    _hyp, t, q, p0=[float(q[0]), 0.02, 0.5],
                    bounds=([0, 0, 0.01], [np.inf, 10, 0.99]), maxfev=10000,
                )
                return float(popt[0]), float(popt[1]), float(popt[2])
            elif model == "har":
                def _har(t, qi, Di):
                    return qi / (1.0 + Di * t)
                popt, _ = curve_fit(
                    _har, t, q, p0=[float(q[0]), 0.01],
                    bounds=([0, 0], [np.inf, 10]), maxfev=5000,
                )
                return float(popt[0]), float(popt[1])
            else:  # exp
                popt, _ = curve_fit(
                    lambda t, qi, Di: qi * np.exp(-Di * t),
                    t, q, p0=[float(q[0]), 0.02],
                    bounds=([0.0, 0.0], [np.inf, 5.0]), maxfev=5000,
                )
                return float(popt[0]), float(np.clip(popt[1], 0.0, 5.0))
        except Exception:
            pass
        if model == "exp":
            try:
                log_q = np.log(np.clip(q, 1e-12, None))
                coeffs = np.polyfit(t, log_q, 1)
                return float(np.exp(coeffs[1])), float(np.clip(-coeffs[0], 0.0, 5.0))
            except Exception:
                pass
        return None

    def _fit_trend(self, months: np.ndarray, values: np.ndarray) -> tuple | None:
        """Fit a DCA trend using the selected model or best R².

        Returns (qi, Di) for exp/har, (qi, Di, b) for hyp, or None.
        In 'best' mode, tries all 3 and picks the one with highest R².
        """
        mask = values > 0
        if np.sum(mask) < 3:
            return None
        model = self._cmb_pct_model.currentData()

        if model == "best":
            return self._fit_best(months, values)

        return self._fit_single_model(months, values, model)

    def _fit_best(self, months: np.ndarray, values: np.ndarray) -> tuple | None:
        """Try exp, hyp, har and return the params with the highest R²."""
        mask = values > 0
        t = months[mask]
        q = values[mask]
        best_r2 = -1.0
        best_params = None
        for mdl in ("exp", "hyp", "har"):
            params = self._fit_single_model(months, values, mdl)
            if params is None:
                continue
            pred = self._predict_from_params(params, mdl, t)
            ss_res = float(np.sum((q - pred) ** 2))
            ss_tot = float(np.sum((q - np.mean(q)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            if r2 > best_r2:
                best_r2 = r2
                best_params = params
        return best_params

    @staticmethod
    def _predict_from_params(params: tuple, model: str, t: np.ndarray) -> np.ndarray:
        """Evaluate a DCA model at given t from a params tuple."""
        if model == "hyp" and len(params) == 3:
            qi, Di, b = params
            base = 1.0 + b * Di * t
            return qi / np.power(np.where(base > 0, base, 1e-12), 1.0 / b)
        elif model == "har":
            qi, Di = params[0], params[1]
            return qi / (1.0 + Di * t)
        else:  # exp
            qi, Di = params[0], params[1]
            return qi * np.exp(-Di * t)


    def _update_results_text(self) -> None:
        """Show R² and trend coefficients for P10/P50/P90 in the results box."""
        if not self._show_pct or self._pct_months is None or self._pct_data is None:
            self._txt_result.clear()
            return
        model = self._cmb_pct_model.currentData()
        model_names = {"exp": "Экспоненциальная", "hyp": "Гиперболическая", "har": "Гармоническая"}
        lines = [f"Модель: {model_names.get(model, model)}"]
        for pct_key in (90, 50, -1, 10):  # P10(high) P50 Avg P90(low)
            label = _PCT_LABELS[pct_key]
            trend = self._pct_trends.get(pct_key)
            data = self._pct_data.get(pct_key)
            if trend is None or data is None:
                lines.append(f"{label}: не удалось подобрать")
                continue
            t = self._pct_months
            q = data
            mask = q > 0
            t_m, q_m = t[mask], q[mask]
            # Detect actual model from params shape
            if len(trend) == 3:
                qi, Di, b = trend
                base = 1.0 + b * Di * t_m
                pred = qi / np.power(np.where(base > 0, base, 1e-12), 1.0 / b)
                params_str = f"qi={qi:.4g}, Di={Di:.4g}, b={b:.3f} [гип.]"
            else:
                qi, Di = trend[0], trend[1]
                # Distinguish exp vs har by checking which fits better
                pred_exp = qi * np.exp(-Di * t_m)
                pred_har = qi / (1.0 + Di * t_m)
                r2_exp = 1.0 - float(np.sum((q_m - pred_exp)**2)) / max(float(np.sum((q_m - np.mean(q_m))**2)), 1e-12)
                r2_har = 1.0 - float(np.sum((q_m - pred_har)**2)) / max(float(np.sum((q_m - np.mean(q_m))**2)), 1e-12)
                if model in ("har",) or (model == "best" and r2_har > r2_exp):
                    pred = pred_har
                    tag = "гарм."
                else:
                    pred = pred_exp
                    tag = "эксп."
                params_str = f"qi={qi:.4g}, Di={Di:.4g} [{tag}]"
            ss_res = float(np.sum((q_m - pred) ** 2))
            ss_tot = float(np.sum((q_m - np.mean(q_m)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            lines.append(f"{label}: R\u00b2={r2:.4f}  {params_str}")
        self._txt_result.setPlainText("\n".join(lines))

    # ── Data helpers ────────────────────────────────────────────────────

    def _producing_wells(self) -> list[str]:
        prod_col = COL_GAS if self._phase == "gas" else COL_OIL
        if COL_WELL not in self._df.columns or prod_col not in self._df.columns:
            return []
        sub = self._df
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        totals = sub.groupby(COL_WELL)[prod_col].sum()
        return sorted(totals[totals > 0].index.tolist())

    def _aligned_series(
        self, well: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
        """Return (x_months, y_with_nans_for_excluded_or_filtered, iso_dates).

        y carries NaN where:
          - the month is in the user exclusion set, OR
          - the rate is below the min-rate filter, OR
          - the working days are below the min-days filter.
        iso_dates are ISO-formatted strings for each month row.

        When mode is 'rate', y = production / (hours_work / 24) per well.
        """
        prod_col = COL_GAS if self._phase == "gas" else COL_OIL
        is_rate_mode = self._cmb_mode.currentData() == "rate"

        sub = self._df[self._df[COL_WELL] == well].copy()
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        if COL_DATE not in sub.columns or prod_col not in sub.columns:
            return None

        if is_rate_mode and COL_HOURS_WORK in sub.columns:
            # Daily rate = production / producing_days
            agg = sub.groupby(COL_DATE).agg(
                {prod_col: "sum", COL_HOURS_WORK: "sum"}
            ).sort_index()
            days = agg[COL_HOURS_WORK].values / 24.0
            raw_vals = np.divide(
                agg[prod_col].values, days,
                out=np.zeros(len(days), dtype=float), where=days > 0,
            )
            agg_series = pd.Series(raw_vals, index=agg.index)
        else:
            agg_series = sub.groupby(COL_DATE)[prod_col].sum().sort_index()

        if len(agg_series) == 0:
            return None
        positive_dates = agg_series[agg_series > 0].index
        if len(positive_dates) == 0:
            return None
        agg_series = agg_series[agg_series.index >= positive_dates[0]]
        x = np.arange(1, len(agg_series) + 1, dtype=float)
        y_raw = agg_series.values.astype(float)
        iso_dates: list[str] = [str(d)[:10] for d in agg_series.index]
        y = y_raw.copy()

        # ── Criteria filters (applied before user exclusions) ────────────────
        min_rate = self._spn_min_rate.value()
        if min_rate > 0:
            for i, v in enumerate(y_raw):
                if np.isfinite(v) and v < min_rate:
                    y[i] = np.nan

        min_days = self._spn_min_days.value()
        if min_days > 0 and COL_HOURS_WORK in sub.columns:
            hours_agg = sub.groupby(COL_DATE)[COL_HOURS_WORK].sum().sort_index()
            hours_aligned = hours_agg.reindex(agg_series.index, fill_value=0.0)
            for i, hours in enumerate(hours_aligned.values):
                if float(hours) / 24.0 < min_days:
                    y[i] = np.nan

        # ── User exclusions ──────────────────────────────────────────────
        for i, iso in enumerate(iso_dates):
            if (well, iso) in self._excluded:
                y[i] = np.nan

        # ── Trim leading NaNs so month 1 = first valid (non-filtered) value ──
        first_valid = -1
        for i in range(len(y)):
            if np.isfinite(y[i]) and y[i] > 0:
                first_valid = i
                break
        if first_valid < 0:
            return None  # all values filtered out
        if first_valid > 0:
            y = y[first_valid:]
            iso_dates = iso_dates[first_valid:]
        x = np.arange(1, len(y) + 1, dtype=float)

        return x, y, iso_dates

    # ── Drawing ─────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        self._well_points.clear()

        selected = [item.text() for item in self._lst.selectedItems()]
        well_alpha = 0.25 if self._show_pct else 0.85
        well_lw    = 0.8  if self._show_pct else 1.3

        prod_col = COL_GAS if self._phase == "gas" else COL_OIL

        for ki, well in enumerate(selected):
            result = self._aligned_series(well)
            if result is None:
                continue
            x, y, iso_dates = result
            color = _COLORS[ki % len(_COLORS)]

            # Normalize to first non-NaN value if requested
            if self._chk_normalize.isChecked():
                finite = y[np.isfinite(y) & (y > 0)]
                if len(finite) > 0:
                    y = y / finite[0]

            # Store non-NaN plotted points for eraser hit-testing
            # (coordinates must match what is visible on the axes)
            self._well_points[well] = [
                (float(x[i]), float(y[i]), iso_dates[i])
                for i in range(len(y))
                if np.isfinite(y[i]) and y[i] > 0
            ]

            # Plot line (NaN creates visible gaps at excluded months)
            ax.plot(x, y, color=color, linewidth=well_lw,
                    label=well, alpha=well_alpha)


        # Percentile trendlines
        if self._show_pct and self._pct_months is not None:
            m = self._pct_months
            t_max = float(m[-1]) * 1.5
            model = self._cmb_pct_model.currentData()
            for pct in (10, 50, -1, 90):
                col = _PCT_COLORS[pct]
                trend = self._pct_trends.get(pct)
                if trend is None:
                    continue
                x_ext = np.linspace(float(m[0]), t_max, 400)
                if model == "hyp" and len(trend) == 3:
                    qi, Di, b = trend
                    base = 1.0 + b * Di * x_ext
                    y_ext = qi / np.power(np.where(base > 0, base, 1e-12), 1.0 / b)
                elif model == "har":
                    qi, Di = trend[0], trend[1]
                    y_ext = qi / (1.0 + Di * x_ext)
                else:
                    qi, Di = trend[0], trend[1]
                    y_ext = qi * np.exp(-Di * x_ext)
                ax.plot(x_ext, y_ext,
                        color=col, linewidth=2.0, linestyle="--",
                        label=_PCT_LABELS[pct], zorder=5)

        ax.set_xlabel("Месяц от начала добычи")
        _mode = self._cmb_mode.currentData()
        if self._chk_normalize.isChecked():
            y_lbl = "Относительное значение"
        elif _mode == "rate":
            y_lbl = "Дебит газа, м\u00b3/сут" if self._phase == "gas" else "Дебит нефти, т/сут"
        else:
            y_lbl = "Газ, м\u00b3/мес" if self._phase == "gas" else "Нефть, т/мес"
        ax.set_ylabel(y_lbl)
        title = "Приведённая добыча по скважинам"
        lbl = self._scenario_label()
        if lbl:
            title += f"  [{lbl}]"
        ax.set_title(title)

        if self._chk_log.isChecked():
            try:
                ax.set_yscale("log")
            except Exception:
                pass

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if self._show_pct:
                pct_lbls = set(_PCT_LABELS.values())
                handles = [h for h, lbl in zip(handles, labels) if lbl in pct_lbls]
                labels  = [lbl for lbl in labels if lbl in pct_lbls]
            from src.ui.legend_helper import fit_legend
            fit_legend(ax, handles, labels, loc="upper right")

        ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()

        # Keep the lasso selector alive on the new axes after every redraw
        if self._eraser_active:
            self._attach_eraser_lasso()

    # ── Export helpers ──────────────────────────────────────────────────────

    def _copy_data(self) -> None:
        from PySide6.QtWidgets import QApplication
        selected = [item.text() for item in self._lst.selectedItems()]
        well_data: dict[str, np.ndarray] = {}
        max_month_w = 0
        for well in selected:
            result = self._aligned_series(well)
            if result is not None:
                _, y, _ = result
                well_data[well] = y
                max_month_w = max(max_month_w, len(y))
        has_pct = self._pct_data is not None and self._pct_months is not None
        has_trend = has_pct and any(v is not None for v in self._pct_trends.values())
        max_month = max_month_w
        if has_pct and len(self._pct_months) > 0:
            max_month = max(max_month, int(self._pct_months[-1] * 1.5))
        if max_month == 0:
            return
        pct_lookup: dict[int, list[float]] = {}
        if has_pct:
            for i, mo in enumerate(self._pct_months):
                pct_lookup[int(mo)] = [
                    float(self._pct_data[10][i]),
                    float(self._pct_data[50][i]),
                    float(self._pct_data[90][i]),
                ]
        hdr = ["Месяц"] + list(well_data.keys())
        if has_pct:
            hdr += ["P10 факт", "P50 факт", "P90 факт"]
        if has_trend:
            for pct in (10, 50, 90):
                if self._pct_trends.get(pct) is not None:
                    hdr.append(f"P{pct} тренд")
        rows = ["\t".join(hdr)]
        for m in range(1, max_month + 1):
            row = [str(m)]
            for y in well_data.values():
                v = y[m - 1] if m <= len(y) else np.nan
                row.append(f"{v:.4g}" if np.isfinite(v) else "")
            if has_pct:
                row += (
                    [f"{v:.4g}" for v in pct_lookup[m]]
                    if m in pct_lookup else ["", "", ""]
                )
            if has_trend:
                model = self._cmb_pct_model.currentData()
                for pct in (10, 50, 90):
                    tr = self._pct_trends.get(pct)
                    if tr is not None:
                        qi, Di = tr[0], tr[1]
                        if model == "hyp" and len(tr) == 3:
                            b = tr[2]
                            base = 1.0 + b * Di * m
                            val = qi / (base ** (1.0 / b)) if base > 0 else 0.0
                        elif model == "har":
                            val = qi / (1.0 + Di * m)
                        else:
                            val = qi * np.exp(-Di * m)
                        row.append(f"{val:.4g}")
            rows.append("\t".join(row))
        QApplication.instance().clipboard().setText("\n".join(rows))

    def _to_clipboard(self) -> None:
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        QApplication.instance().clipboard().setImage(QImage.fromData(buf.getvalue()))

    def _save_image(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить картинку", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
