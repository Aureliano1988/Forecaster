"""Forecast plots dialog — interactive multi-method forecast charts.

Allows the user to choose X/Y variables, select which methods to compare,
toggle log scales, and export the plot as an image or copy it to clipboard.
"""

from __future__ import annotations

import io

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ── Variable catalogue ────────────────────────────────────────────────────────

_FAMILY_SHORT: dict[str, str] = {
    "Характеристики вытеснения":   "ХВ",
    "Кривые падения добычи (DCA)": "DCA",
    "Фракционный поток":           "ФП",
}

# (display_name,  series_attr,  axis_label)
_X_VARS = [
    ("Месяц прогноза",         "month",   "Месяц"),
    ("Нак. нефть, т",          "Qo",      "Нак. нефть, т"),
    ("ВНФ",                    "WOR",     "ВНФ"),
    ("Нак. жидкость, т",       "Ql",      "Нак. жидкость, т"),
    ("Нефть, т/мес",           "qo",      "Нефть, т/мес"),
    ("КИН (RF)",               "RF",      "КИН (RF)"),
    ("HCPVI",                  "HCPVI",   "HCPVI"),
]

_Y_VARS = [
    ("Нефть, т/мес",           "qo",    "Нефть, т/мес"),
    ("ВНФ",                    "WOR",   "ВНФ"),
    ("Нак. нефть, т",          "Qo",    "Нак. нефть, т"),
    ("Жидкость, т/мес",        "ql",    "Жидкость, т/мес"),
    ("Нак. жидкость, т",       "Ql",    "Нак. жидкость, т"),
    ("КИН (RF)",               "RF",    "КИН (RF)"),
    ("Закачка, т/мес",        "qi_inj", "Закачка, т/мес"),
    ("Нак. закачка, т",       "Qi_inj", "Нак. закачка, т"),
    ("HCPVI",                  "HCPVI",  "HCPVI"),
]

# default (enabled, axis 1 or 2)
_Y_DEFAULTS: dict[str, tuple[bool, int]] = {
    "qo":     (True,  1),
    "WOR":    (True,  2),
    "Qo":     (False, 1),
    "ql":     (False, 1),
    "Ql":     (False, 1),
    "RF":     (False, 1),
    "qi_inj": (False, 1),
    "Qi_inj": (False, 1),
    "HCPVI":  (False, 1),
}

# 20-colour palette -----------------------------------------------------------
try:
    from matplotlib import colormaps as _CMS
    _COLORS: list = list(_CMS["tab20"].colors)
except Exception:
    _COLORS = [f"C{i}" for i in range(10)]

_LSTYLES   = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
_PCT_COLORS = {10: "#1f77b4", 50: "#222222", 90: "#d62728"}


def _arr(result, attr: str) -> np.ndarray:
    """Extract a numpy array from a SavedMethodResult by variable name."""
    s = result.monthly
    if attr == "month":
        return np.arange(1, s.duration + 1, dtype=float)
    return np.asarray(getattr(s, attr), dtype=float)


# Attrs that require external reservoir/injection parameters to compute
_SPECIAL_ATTRS = {"RF", "HCPVI", "qi_inj", "Qi_inj"}


# ── Y-variable row ────────────────────────────────────────────────────────────

class _YRow(QWidget):
    """One row inside the Y-axis control group."""

    def __init__(self, label: str, enabled: bool, axis: int) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        self.chk = QCheckBox(label)
        self.chk.setChecked(enabled)
        self.cmb = QComboBox()
        self.cmb.addItems(["Y1", "Y2"])
        self.cmb.setCurrentIndex(0 if axis == 1 else 1)
        self.cmb.setFixedWidth(52)
        lay.addWidget(self.chk, 1)
        lay.addWidget(self.cmb)

    @property
    def enabled(self) -> bool:
        return self.chk.isChecked()

    @property
    def axis(self) -> int:
        return 1 if self.cmb.currentIndex() == 0 else 2

    def connect_changed(self, slot) -> None:
        self.chk.stateChanged.connect(slot)
        self.cmb.currentIndexChanged.connect(slot)


# ── Main dialog ───────────────────────────────────────────────────────────────

class ForecastPlotsDialog(QDialog):
    """Interactive multi-method forecast plot window."""

    def __init__(
        self,
        saved_results: dict,
        project_name: str = "",
        hist_data: dict | None = None,
        stoiip: float = 0.0,
        hcpv: float = 0.0,
        qi_hist_last: float = 0.0,
        qi_const: float = 0.0,
        scenarios=None,           # list[ForecastScenario] | None
        scenarios_hist=None,      # list[dict | None] | None  — per-scenario hist_data
        parent=None,
    ) -> None:
        super().__init__(parent)

        # Percentile state: maps pct level → method key of the closest method
        self._pct_methods: dict[int, str] | None = None
        self._show_pct: bool = False

        # Reservoir parameters for RF / HCPVI computation
        self._stoiip = stoiip
        self._hcpv = hcpv
        self._qi_hist_last = qi_hist_last
        self._qi_const = qi_const

        # All scenarios (for scenario-comparison mode)
        self._scenarios = list(scenarios) if scenarios else []
        # Per-scenario historical data (parallel list to self._scenarios)
        self._scenarios_hist: list = list(scenarios_hist) if scenarios_hist else []

        # Keep only results that have a computed monthly forecast
        self._data = {
            k: v for k, v in saved_results.items()
            if v.monthly is not None and v.monthly.duration > 0
        }
        # Historical aggregated data: keys qo, ql, Qo, Ql, WOR  (np.ndarrays)
        # May also contain RF and HCPVI if STOIIP/HCPV are set in main window
        self._hist: dict | None = hist_data

        title = "Графики прогнозов"
        if project_name:
            title += f" — {project_name}"
        self.setWindowTitle(title)
        self.resize(1400, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel (controls) ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ctrl = QWidget()
        ctrl_lay = QVBoxLayout(ctrl)
        ctrl_lay.setSpacing(6)
        scroll.setWidget(ctrl)
        splitter.addWidget(scroll)

        # Mode selector
        mode_grp = QGroupBox("Режим сравнения")
        mode_lay = QVBoxLayout(mode_grp)
        self._rb_methods   = QRadioButton("По методам")
        self._rb_scenarios = QRadioButton("По сценариям")
        self._rb_methods.setChecked(True)
        if len(self._scenarios) < 2:
            self._rb_scenarios.setEnabled(False)
        mode_lay.addWidget(self._rb_methods)
        mode_lay.addWidget(self._rb_scenarios)
        ctrl_lay.addWidget(mode_grp)

        # Methods group
        grp_m = QGroupBox("Методы")
        gm_lay = QVBoxLayout(grp_m)
        self._lst = QListWidget()
        self._lst.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for key, result in self._data.items():
            family, mname = key.split("|", 1)
            short = _FAMILY_SHORT.get(family, family[:3])
            item = QListWidgetItem(f"{mname}  [{short}]")
            item.setData(Qt.ItemDataRole.UserRole, key)
            self._lst.addItem(item)
        self._lst.selectAll()
        gm_lay.addWidget(self._lst)
        ctrl_lay.addWidget(grp_m)
        self._grp_methods = grp_m   # kept for show/hide

        # Scenario-comparison panel (hidden by default)
        sc_panel = QGroupBox("По сценариям")
        sc_lay = QVBoxLayout(sc_panel)
        sc_lay.addWidget(QLabel("Метод:"))
        self._cmb_sc_method = QComboBox()
        # Populate with all method keys that have a monthly forecast in any scenario
        sc_keys_seen: set[str] = set()
        for _sc in self._scenarios:
            for _key, _res in _sc.results.items():
                if (
                    _key not in sc_keys_seen
                    and _res.monthly is not None
                    and _res.monthly.duration > 0
                ):
                    _fam, _mn = _key.split("|", 1)
                    _sh = _FAMILY_SHORT.get(_fam, _fam[:3])
                    self._cmb_sc_method.addItem(f"{_mn}  [{_sh}]", userData=_key)
                    sc_keys_seen.add(_key)
        sc_lay.addWidget(self._cmb_sc_method)
        sc_lay.addWidget(QLabel("Сценарии:"))
        self._lst_sc = QListWidget()
        self._lst_sc.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for _i, _sc in enumerate(self._scenarios):
            _item = QListWidgetItem(_sc.name)
            _item.setData(Qt.ItemDataRole.UserRole, _i)
            self._lst_sc.addItem(_item)
        self._lst_sc.selectAll()
        sc_lay.addWidget(self._lst_sc)
        ctrl_lay.addWidget(sc_panel)
        sc_panel.setVisible(False)
        self._sc_panel = sc_panel

        # History toggle
        self._chk_hist = QCheckBox("Показать исторические данные")
        self._chk_hist.setChecked(hist_data is not None)
        self._chk_hist.setEnabled(hist_data is not None)
        ctrl_lay.addWidget(self._chk_hist)

        # X-axis group
        grp_x = QGroupBox("Ось X")
        gx_lay = QVBoxLayout(grp_x)
        self._cmb_x = QComboBox()
        self._cmb_x.addItems([d for d, _, _ in _X_VARS])
        gx_lay.addWidget(self._cmb_x)
        self._chk_xlog = QCheckBox("Log шкала (X)")
        gx_lay.addWidget(self._chk_xlog)
        ctrl_lay.addWidget(grp_x)

        # Y-axis group
        grp_y = QGroupBox("Ось Y (выберите переменные)")
        gy_lay = QVBoxLayout(grp_y)
        log_row = QHBoxLayout()
        self._chk_y1log = QCheckBox("Log Y1")
        self._chk_y2log = QCheckBox("Log Y2")
        log_row.addWidget(self._chk_y1log)
        log_row.addWidget(self._chk_y2log)
        gy_lay.addLayout(log_row)
        self._y_rows: list[_YRow] = []
        for disp, attr, _ in _Y_VARS:
            en, ax = _Y_DEFAULTS.get(attr, (False, 1))
            row = _YRow(disp, en, ax)
            gy_lay.addWidget(row)
            self._y_rows.append(row)
        ctrl_lay.addWidget(grp_y)

        btn_pct = QPushButton("Выбрать P10/P50/P90")
        btn_pct.clicked.connect(self._generate_pct)
        ctrl_lay.addWidget(btn_pct)

        # Reservoir parameters info
        res_parts: list[str] = []
        if stoiip > 0:
            res_parts.append(f"STOIIP={stoiip:,.0f} т")
        else:
            res_parts.append("STOIIP=—")
        if hcpv > 0:
            res_parts.append(f"HCPV={hcpv:,.0f} м\u00b3")
        else:
            res_parts.append("HCPV=—")
        lbl_res = QLabel(", ".join(res_parts))
        lbl_res.setStyleSheet("color: gray; font-size: 10px;")
        lbl_res.setWordWrap(True)
        ctrl_lay.addWidget(lbl_res)

        ctrl_lay.addStretch()

        # ── Right panel (plot) ────────────────────────────────────────────
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

        btns = QHBoxLayout()
        btn_clip = QPushButton("Копировать график")
        btn_clip.clicked.connect(self._to_clipboard)
        btn_save = QPushButton("Сохранить картинку…")
        btn_save.clicked.connect(self._save)
        btn_data = QPushButton("Скопировать данные")
        btn_data.clicked.connect(self._copy_data)
        btns.addWidget(btn_clip)
        btns.addWidget(btn_save)
        btns.addWidget(btn_data)
        btns.addStretch()
        right_lay.addLayout(btns)

        # ── Connect all controls → redraw (clears stale percentiles first) ─────
        self._rb_methods.toggled.connect(self._on_mode_changed)
        self._lst.itemSelectionChanged.connect(self._on_control_changed)
        self._cmb_x.currentIndexChanged.connect(self._on_control_changed)
        self._chk_xlog.stateChanged.connect(self._on_control_changed)
        self._chk_y1log.stateChanged.connect(self._on_control_changed)
        self._chk_y2log.stateChanged.connect(self._on_control_changed)
        for row in self._y_rows:
            row.connect_changed(self._on_control_changed)
        self._chk_hist.stateChanged.connect(self._on_control_changed)
        self._cmb_sc_method.currentIndexChanged.connect(self._on_control_changed)
        self._lst_sc.itemSelectionChanged.connect(self._on_control_changed)

        self._draw()

    # ── RF / HCPVI helpers ────────────────────────────────────────────

    def _x_for(
        self, s, x_attr: str, qo_last: float, ql_last: float,
        stoiip: float | None = None,
        hcpv: float | None = None,
        qi_hist_last: float | None = None,
        qi_const: float | None = None,
    ) -> np.ndarray:
        """Compute x-array for a forecast series, handling all x-axis types.

        Optional override parameters allow scenarios mode to supply per-scenario
        reservoir values instead of the dialog-level defaults.
        """
        eff_stoiip       = stoiip       if stoiip       is not None else self._stoiip
        eff_hcpv         = hcpv         if hcpv         is not None else self._hcpv
        eff_qi_hist_last = qi_hist_last if qi_hist_last is not None else self._qi_hist_last
        eff_qi_const     = qi_const     if qi_const     is not None else self._qi_const
        if x_attr == "month":
            return np.arange(1, s.duration + 1, dtype=float)
        elif x_attr == "Qo":
            return qo_last + np.asarray(s.Qo, dtype=float)
        elif x_attr == "Ql":
            return ql_last + np.asarray(s.Ql, dtype=float)
        elif x_attr == "RF":
            if eff_stoiip > 0:
                return (qo_last + np.asarray(s.Qo, dtype=float)) / eff_stoiip
            return np.zeros(s.duration, dtype=float)
        elif x_attr == "HCPVI":
            m_arr = np.arange(1, s.duration + 1, dtype=float)
            if eff_hcpv > 0:
                return (eff_qi_hist_last + eff_qi_const * m_arr) / eff_hcpv
            return np.zeros(s.duration, dtype=float)
        return np.asarray(getattr(s, x_attr), dtype=float)

    def _y_for(
        self, s, y_attr: str, qo_last: float, ql_last: float,
        stoiip: float | None = None,
        hcpv: float | None = None,
        qi_hist_last: float | None = None,
        qi_const: float | None = None,
    ) -> np.ndarray:
        """Compute y-array for a forecast series, handling all y-axis types.

        Optional override parameters allow scenarios mode to supply per-scenario
        reservoir values instead of the dialog-level defaults.
        """
        eff_stoiip       = stoiip       if stoiip       is not None else self._stoiip
        eff_hcpv         = hcpv         if hcpv         is not None else self._hcpv
        eff_qi_hist_last = qi_hist_last if qi_hist_last is not None else self._qi_hist_last
        eff_qi_const     = qi_const     if qi_const     is not None else self._qi_const
        if y_attr == "Qo":
            return qo_last + np.asarray(s.Qo, dtype=float)
        elif y_attr == "Ql":
            return ql_last + np.asarray(s.Ql, dtype=float)
        elif y_attr == "RF":
            if eff_stoiip > 0:
                return (qo_last + np.asarray(s.Qo, dtype=float)) / eff_stoiip
            return np.zeros(s.duration, dtype=float)
        elif y_attr == "qi_inj":
            return np.full(s.duration, eff_qi_const, dtype=float)
        elif y_attr == "Qi_inj":
            m_arr = np.arange(1, s.duration + 1, dtype=float)
            return eff_qi_hist_last + eff_qi_const * m_arr
        elif y_attr == "HCPVI":
            m_arr = np.arange(1, s.duration + 1, dtype=float)
            if eff_hcpv > 0:
                return (eff_qi_hist_last + eff_qi_const * m_arr) / eff_hcpv
            return np.zeros(s.duration, dtype=float)
        return np.asarray(getattr(s, y_attr), dtype=float)

    # ── Mode toggle ───────────────────────────────────────────────────

    def _on_mode_changed(self, methods_active: bool) -> None:
        """Show/hide panels and redraw when the comparison mode radio changes."""
        self._grp_methods.setVisible(methods_active)
        self._sc_panel.setVisible(not methods_active)
        self._pct_methods = None
        self._show_pct = False
        self._draw()

    # ── Control-change handler ────────────────────────────────────────

    def _on_control_changed(self) -> None:
        """Clear stale percentile selection and redraw."""
        self._pct_methods = None
        self._show_pct = False
        self._draw()

    # ── Percentile generation ──────────────────────────────────────

    def _generate_pct(self) -> None:
        """Select P10/P50/P90 methods by cumulative oil proximity.

        For each selected method, computes total remaining reserves.
        Finds the P10/P50/P90 percentile values of those reserves, then
        picks the method whose reserves is closest to each level.
        Acts as a toggle: second click clears the highlight.
        """
        from PySide6.QtWidgets import QMessageBox

        if self._show_pct:
            self._pct_methods = None
            self._show_pct = False
            self._draw()
            return

        selected_keys = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._lst.selectedItems()
        ]
        if len(selected_keys) < 3:
            QMessageBox.warning(
                self, "Недостаточно методов",
                f"Для отбора P10/P50/P90 необходимо не менее 3 методов.\n"
                f"Выбрано: {len(selected_keys)}.",
            )
            return

        # Cumulative remaining reserves for each method
        reserves: dict[str, float] = {
            key: float(self._data[key].monthly.remain_reserves)
            for key in selected_keys
        }
        reserves_arr = np.array(list(reserves.values()))

        # Percentile targets over the distribution of reserves
        targets = {
            10: float(np.percentile(reserves_arr, 10)),
            50: float(np.percentile(reserves_arr, 50)),
            90: float(np.percentile(reserves_arr, 90)),
        }

        # Pick the method closest to each percentile target
        pct_methods: dict[int, str] = {}
        for pct, target_val in targets.items():
            closest = min(selected_keys, key=lambda k: abs(reserves[k] - target_val))
            pct_methods[pct] = closest

        self._pct_methods = pct_methods
        self._show_pct = True
        self._draw()

    # ── Data clipboard ──────────────────────────────────────────────

    def _copy_data(self) -> None:
        """Copy forecast data as TSV.  Handles both comparison modes."""
        from PySide6.QtWidgets import QApplication

        if self._rb_scenarios.isChecked():
            self._copy_data_scenarios(QApplication)
            return

        selected_keys = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._lst.selectedItems()
        ]
        if not selected_keys:
            return

        _, x_attr, x_label = _X_VARS[self._cmb_x.currentIndex()]
        hist = self._hist
        _hQo = np.asarray(hist["Qo"], dtype=float) if hist and "Qo" in hist else np.array([])
        _hQl = np.asarray(hist["Ql"], dtype=float) if hist and "Ql" in hist else np.array([])
        qo_last = float(_hQo[-1]) if len(_hQo) else 0.0
        ql_last = float(_hQl[-1]) if len(_hQl) else 0.0

        method_names = {k: k.split("|", 1)[1] for k in selected_keys}
        enabled_yvars = [
            (y_attr, ylbl)
            for row, (_, y_attr, ylbl) in zip(self._y_rows, _Y_VARS)
            if row.enabled
        ]
        if not enabled_yvars:
            return

        max_dur = max(self._data[k].monthly.duration for k in selected_keys)

        # Pre-compute x and y arrays per method (using helpers that handle RF/HCPVI)
        method_x: dict[str, np.ndarray] = {}
        method_y: dict[str, dict[str, np.ndarray]] = {}
        for key in selected_keys:
            s = self._data[key].monthly
            method_x[key] = self._x_for(s, x_attr, qo_last, ql_last)
            method_y[key] = {
                y_attr: self._y_for(s, y_attr, qo_last, ql_last)
                for y_attr, _ in enabled_yvars
            }

        # Build Y-arrays for selected P10/P50/P90 methods
        pct_method_y: dict[int, dict[str, np.ndarray]] = {}
        if self._pct_methods:
            for pct, key in self._pct_methods.items():
                s = self._data[key].monthly
                pct_method_y[pct] = {
                    y_attr: self._y_for(s, y_attr, qo_last, ql_last)
                    for y_attr, _ in enabled_yvars
                }

        # Header
        hdr = ["Месяц"]
        for y_attr, ylbl in enabled_yvars:
            for key in selected_keys:
                hdr.append(f"{method_names[key]} — {ylbl}")
        if x_attr != "month":
            for key in selected_keys:
                hdr.append(f"{method_names[key]} — {x_label}")
        if pct_method_y:
            for y_attr, ylbl in enabled_yvars:
                for pct in (10, 50, 90):
                    hdr.append(f"P{pct} — {ylbl}")

        # Rows
        rows = ["\t".join(hdr)]
        for m in range(1, max_dur + 1):
            row = [str(m)]
            for y_attr, _ in enabled_yvars:
                for key in selected_keys:
                    ya = method_y[key][y_attr]
                    row.append(f"{ya[m-1]:.4g}" if m <= len(ya) else "")
            if x_attr != "month":
                for key in selected_keys:
                    xa = method_x[key]
                    row.append(f"{xa[m-1]:.4g}" if m <= len(xa) else "")
            if pct_method_y:
                for y_attr, _ in enabled_yvars:
                    for pct in (10, 50, 90):
                        if pct in pct_method_y and y_attr in pct_method_y[pct]:
                            ya_p = pct_method_y[pct][y_attr]
                            row.append(f"{ya_p[m-1]:.4g}" if m <= len(ya_p) else "")
                        else:
                            row.append("")
            rows.append("\t".join(row))

        QApplication.instance().clipboard().setText("\n".join(rows))

    def _copy_data_scenarios(self, QApplication) -> None:
        """Copy scenario-comparison data as TSV (month-indexed)."""
        method_key = self._cmb_sc_method.currentData(Qt.ItemDataRole.UserRole)
        if not method_key:
            return
        selected_sc_idx = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._lst_sc.selectedItems()
        ]
        if not selected_sc_idx:
            return

        _, x_attr, x_label = _X_VARS[self._cmb_x.currentIndex()]
        hist = self._hist
        _hQl = np.asarray(hist["Ql"], dtype=float) if hist and "Ql" in hist else np.array([])
        ql_last_global = float(_hQl[-1]) if len(_hQl) > 0 else 0.0

        enabled_yvars = [
            (y_attr, ylbl)
            for row, (_, y_attr, ylbl) in zip(self._y_rows, _Y_VARS)
            if row.enabled
        ]
        if not enabled_yvars:
            return

        # Build per-scenario arrays
        sc_names: list[str] = []
        sc_x: list[np.ndarray] = []
        sc_y: list[dict[str, np.ndarray]] = []
        max_dur = 0
        for sc_idx in selected_sc_idx:
            sc = self._scenarios[sc_idx]
            result = sc.results.get(method_key)
            if result is None or result.monthly is None or result.monthly.duration == 0:
                continue
            s = result.monthly
            qo_last = result.qo_hist_last
            # Per-scenario STOIIP / HCPV
            _sc_stoiip = sc.stoiip if sc.stoiip > 0 else self._stoiip
            _sc_hcpv   = sc.hcpv   if sc.hcpv   > 0 else self._hcpv
            _sc_qi_hist_last = self._qi_hist_last
            _sc_qi_const     = self._qi_const
            if sc_idx < len(self._scenarios_hist) and self._scenarios_hist[sc_idx] is not None:
                _sh = self._scenarios_hist[sc_idx]
                if "Qi_inj" in _sh and len(_sh["Qi_inj"]) > 0:
                    _sc_qi_hist_last = float(_sh["Qi_inj"][-1])
                if "qi_inj" in _sh and len(_sh["qi_inj"]) > 0:
                    _qi = np.asarray(_sh["qi_inj"], dtype=float)
                    _pos = _qi[_qi > 0]
                    _sc_qi_const = float(_pos[-min(3, len(_pos)):].mean()) if len(_pos) > 0 else 0.0
            sc_names.append(sc.name)
            sc_x.append(self._x_for(
                s, x_attr, qo_last, ql_last_global,
                stoiip=_sc_stoiip,
                hcpv=_sc_hcpv,
                qi_hist_last=_sc_qi_hist_last,
                qi_const=_sc_qi_const,
            ))
            sc_y.append({
                y_attr: self._y_for(
                    s, y_attr, qo_last, ql_last_global,
                    stoiip=_sc_stoiip,
                    hcpv=_sc_hcpv,
                    qi_hist_last=_sc_qi_hist_last,
                    qi_const=_sc_qi_const,
                )
                for y_attr, _ in enabled_yvars
            })
            max_dur = max(max_dur, s.duration)

        if not sc_names:
            return

        hdr = ["Месяц"]
        for y_attr, ylbl in enabled_yvars:
            for name in sc_names:
                hdr.append(f"{name} — {ylbl}")
        if x_attr != "month":
            for name in sc_names:
                hdr.append(f"{name} — {x_label}")

        rows = ["\t".join(hdr)]
        for m in range(1, max_dur + 1):
            row = [str(m)]
            for y_attr, _ in enabled_yvars:
                for i in range(len(sc_names)):
                    ya = sc_y[i][y_attr]
                    row.append(f"{ya[m-1]:.4g}" if m <= len(ya) else "")
            if x_attr != "month":
                for i in range(len(sc_names)):
                    xa = sc_x[i]
                    row.append(f"{xa[m-1]:.4g}" if m <= len(xa) else "")
            rows.append("\t".join(row))

        QApplication.instance().clipboard().setText("\n".join(rows))

    # ── Drawing ──────────────────────────────────────────────

    def _draw(self) -> None:
        """Dispatch to the appropriate drawing method based on mode."""
        if self._rb_scenarios.isChecked():
            self._draw_scenarios()
        else:
            self._draw_methods()

    # ── By-scenarios drawing ──────────────────────────────────────

    def _draw_scenarios(self) -> None:
        """Draw one forecast line per selected scenario for the chosen method."""
        self._fig.clear()
        ax1 = self._fig.add_subplot(111)
        ax2 = None

        method_key = self._cmb_sc_method.currentData(Qt.ItemDataRole.UserRole)
        if not method_key:
            self._canvas.draw_idle()
            return

        selected_sc_idx = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._lst_sc.selectedItems()
        ]
        if not selected_sc_idx:
            self._canvas.draw_idle()
            return

        needs_y2 = any(r.enabled and r.axis == 2 for r in self._y_rows)
        if needs_y2:
            ax2 = ax1.twinx()

        _, x_attr, x_label = _X_VARS[self._cmb_x.currentIndex()]
        hist = self._hist
        _hQl = np.asarray(hist["Ql"], dtype=float) if hist and "Ql" in hist else np.array([])
        ql_last_global = float(_hQl[-1]) if len(_hQl) > 0 else 0.0
        n_hist = len(hist["qo"]) if hist and "qo" in hist else 0

        y1_labels: list[str] = []
        y2_labels: list[str] = []

        for ki, sc_idx in enumerate(selected_sc_idx):
            sc = self._scenarios[sc_idx]
            result = sc.results.get(method_key)
            if result is None or result.monthly is None or result.monthly.duration == 0:
                continue
            s = result.monthly
            color = _COLORS[ki % len(_COLORS)]
            # Per-scenario qo_hist_last for cumulative-Qo offset
            qo_last = result.qo_hist_last
            ql_last = ql_last_global
            # Per-scenario STOIIP / HCPV (fall back to dialog defaults when 0)
            sc_stoiip = sc.stoiip if sc.stoiip > 0 else self._stoiip
            sc_hcpv   = sc.hcpv   if sc.hcpv   > 0 else self._hcpv
            # Per-scenario injection parameters derived from historical data
            sc_qi_hist_last = self._qi_hist_last
            sc_qi_const     = self._qi_const
            if sc_idx < len(self._scenarios_hist) and self._scenarios_hist[sc_idx] is not None:
                _sh = self._scenarios_hist[sc_idx]
                if "Qi_inj" in _sh and len(_sh["Qi_inj"]) > 0:
                    sc_qi_hist_last = float(_sh["Qi_inj"][-1])
                if "qi_inj" in _sh and len(_sh["qi_inj"]) > 0:
                    _qi = np.asarray(_sh["qi_inj"], dtype=float)
                    _pos = _qi[_qi > 0]
                    sc_qi_const = float(_pos[-min(3, len(_pos)):].mean()) if len(_pos) > 0 else 0.0
            x = self._x_for(s, x_attr, qo_last, ql_last,
                            stoiip=sc_stoiip,
                            hcpv=sc_hcpv,
                            qi_hist_last=sc_qi_hist_last,
                            qi_const=sc_qi_const)
            for yi, (row, (_, attr, ylbl)) in enumerate(zip(self._y_rows, _Y_VARS)):
                if not row.enabled:
                    continue
                y = self._y_for(s, attr, qo_last, ql_last,
                                stoiip=sc_stoiip,
                                hcpv=sc_hcpv,
                                qi_hist_last=sc_qi_hist_last,
                                qi_const=sc_qi_const)
                n = min(len(x), len(y))
                if n == 0:
                    continue
                ls = _LSTYLES[yi % len(_LSTYLES)]
                target = ax1 if row.axis == 1 else ax2
                target.plot(
                    x[:n], y[:n],
                    color=color, linestyle=ls, linewidth=1.5,
                    label=sc.name, alpha=0.85,
                )
                bucket = y1_labels if row.axis == 1 else y2_labels
                if ylbl not in bucket:
                    bucket.append(ylbl)

        # Historical overlay: one line per selected scenario (color-matched to forecast)
        if self._chk_hist.isChecked() and self._scenarios_hist:
            for ki, sc_idx in enumerate(selected_sc_idx):
                sc_hist_i = (
                    self._scenarios_hist[sc_idx]
                    if sc_idx < len(self._scenarios_hist)
                    else None
                )
                if sc_hist_i is None:
                    continue
                n_hist_i = len(sc_hist_i.get("qo", []))
                if n_hist_i == 0:
                    continue
                color_i = _COLORS[ki % len(_COLORS)]
                sc_i = self._scenarios[sc_idx]
                if x_attr == "month":
                    xh = np.arange(-(n_hist_i - 1), 1, dtype=float)
                elif x_attr in sc_hist_i:
                    xh = np.asarray(sc_hist_i[x_attr], dtype=float)
                else:
                    continue
                if len(xh) == 0:
                    continue
                for yi, (row, (_, attr, ylbl)) in enumerate(zip(self._y_rows, _Y_VARS)):
                    if not row.enabled or attr not in sc_hist_i:
                        continue
                    yh = np.asarray(sc_hist_i[attr], dtype=float)
                    n = min(len(xh), len(yh))
                    if n == 0:
                        continue
                    ls = _LSTYLES[yi % len(_LSTYLES)]
                    target = ax1 if row.axis == 1 else ax2
                    target.plot(
                        xh[:n], yh[:n],
                        color=color_i, linestyle=ls, linewidth=0.9,
                        marker=".", markersize=2.5,
                        label=f"{sc_i.name} (факт)", alpha=0.55,
                    )
                    bucket = y1_labels if row.axis == 1 else y2_labels
                    if ylbl not in bucket:
                        bucket.append(ylbl)

        self._finish_axes(ax1, ax2, x_label, y1_labels, y2_labels, n_items=len(selected_sc_idx))

    # ── Shared axes finaliser ─────────────────────────────────────

    def _finish_axes(
        self, ax1, ax2, x_label: str,
        y1_labels: list, y2_labels: list, n_items: int,
    ) -> None:
        """Apply labels, log scales, legend and grid, then redraw."""
        ax1.set_xlabel(x_label)
        if y1_labels:
            ax1.set_ylabel(" / ".join(y1_labels))
        if ax2 and y2_labels:
            ax2.set_ylabel(" / ".join(y2_labels))

        for chk, ax in [
            (self._chk_xlog, ax1),
            (self._chk_y1log, ax1),
            (self._chk_y2log, ax2),
        ]:
            if ax is None or not chk.isChecked():
                continue
            try:
                if chk is self._chk_xlog:
                    ax.set_xscale("log")
                else:
                    ax.set_yscale("log")
            except Exception:
                pass

        h, l = ax1.get_legend_handles_labels()
        if ax2:
            h2, l2 = ax2.get_legend_handles_labels()
            h += h2; l += l2
        if h:
            fsize = 6 if n_items > 5 else 7
            leg = ax1.legend(h, l, fontsize=fsize, loc="best")
            if leg is not None:
                leg.set_draggable(True)

        ax1.grid(True, alpha=0.3)
        self._canvas.draw_idle()

    # ── By-methods drawing ─────────────────────────────────────────

    def _draw_methods(self) -> None:
        self._fig.clear()
        ax1 = self._fig.add_subplot(111)
        ax2 = None

        selected_keys = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._lst.selectedItems()
        ]
        if not selected_keys:
            self._canvas.draw_idle()
            return

        needs_y2 = any(r.enabled and r.axis == 2 for r in self._y_rows)
        if needs_y2:
            ax2 = ax1.twinx()

        _, x_attr, x_label = _X_VARS[self._cmb_x.currentIndex()]
        hist = self._hist

        # Cumulative offsets so forecast continues from end of history
        _hQo = np.asarray(hist["Qo"], dtype=float) if hist and "Qo" in hist else np.array([])
        _hQl = np.asarray(hist["Ql"], dtype=float) if hist and "Ql" in hist else np.array([])
        qo_last = float(_hQo[-1]) if len(_hQo) > 0 else 0.0
        ql_last = float(_hQl[-1]) if len(_hQl) > 0 else 0.0
        n_hist  = len(hist["qo"]) if hist and "qo" in hist else 0

        y1_labels: list[str] = []
        y2_labels: list[str] = []

        # ── Forecast lines ──────────────────────────────────────────
        for ki, key in enumerate(selected_keys):
            result = self._data[key]
            _, mname = key.split("|", 1)
            color = _COLORS[ki % len(_COLORS)]
            s = result.monthly

            # X for forecast — offset cumulative variables to continue from history
            x = self._x_for(s, x_attr, qo_last, ql_last)

            for yi, (row, (_, attr, ylbl)) in enumerate(zip(self._y_rows, _Y_VARS)):
                if not row.enabled:
                    continue
                # Y for forecast — offset cumulative / compute special metrics
                y = self._y_for(s, attr, qo_last, ql_last)
                n = min(len(x), len(y))
                if n == 0:
                    continue
                ls = _LSTYLES[yi % len(_LSTYLES)]
                target = ax1 if row.axis == 1 else ax2
                target.plot(
                    x[:n], y[:n],
                    color=color, linestyle=ls, linewidth=1.5,
                    label=mname, alpha=0.85,
                )
                bucket = y1_labels if row.axis == 1 else y2_labels
                if ylbl not in bucket:
                    bucket.append(ylbl)

        # ── Historical data (one shared set, dark colour) ───────────────
        if hist is not None and self._chk_hist.isChecked() and n_hist > 0:
            if x_attr == "month":
                xh = np.arange(-(n_hist - 1), 1, dtype=float)  # … -1, 0
            elif x_attr in hist:
                xh = np.asarray(hist[x_attr], dtype=float)
            else:
                xh = np.array([])

            if len(xh) > 0:
                for yi, (row, (_, attr, ylbl)) in enumerate(zip(self._y_rows, _Y_VARS)):
                    if not row.enabled or attr not in hist:
                        continue
                    yh = np.asarray(hist[attr], dtype=float)
                    n = min(len(xh), len(yh))
                    if n == 0:
                        continue
                    ls = _LSTYLES[yi % len(_LSTYLES)]
                    target = ax1 if row.axis == 1 else ax2
                    target.plot(
                        xh[:n], yh[:n],
                        color="0.25", linestyle=ls, linewidth=1.1,
                        marker=".", markersize=3,
                        label="Факт", alpha=0.65,
                    )

        # ── P10/P50/P90 selected-method highlights ─────────────────
        if self._show_pct and self._pct_methods:
            for pct, key in self._pct_methods.items():
                if key not in self._data:
                    continue
                col = _PCT_COLORS[pct]
                s   = self._data[key].monthly
                xp  = self._x_for(s, x_attr, qo_last, ql_last)

                for yi, (row, (_, attr, ylbl)) in enumerate(zip(self._y_rows, _Y_VARS)):
                    if not row.enabled:
                        continue
                    yp = self._y_for(s, attr, qo_last, ql_last)
                    n = min(len(xp), len(yp))
                    if n == 0:
                        continue
                    ls = _LSTYLES[yi % len(_LSTYLES)]
                    tgt = ax1 if row.axis == 1 else ax2
                    tgt.plot(
                        xp[:n], yp[:n],
                        color=col, linestyle=ls, linewidth=2.8,
                        label=f"P{pct}", zorder=6,
                    )
                    bucket = y1_labels if row.axis == 1 else y2_labels
                    if ylbl not in bucket:
                        bucket.append(ylbl)

        self._finish_axes(ax1, ax2, x_label, y1_labels, y2_labels, n_items=len(selected_keys))

    # ── Export ────────────────────────────────────────────────────────────

    def _to_clipboard(self) -> None:
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        QApplication.instance().clipboard().setImage(
            QImage.fromData(buf.getvalue())
        )

    def _save(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить картинку", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
