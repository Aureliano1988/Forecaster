"""Well alignment dialog — monthly oil vs months since first production.

Each well's X-axis starts at month 1 = its own first month with positive
oil production.  This lets users compare decline profiles regardless of
when each well was brought online.
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.data.models import COL_DATE, COL_GAS, COL_OIL, COL_WELL, COL_WORK_TYPE, WORK_TYPE_OIL

# 20-colour palette (same as other dialogs)
try:
    from matplotlib import colormaps as _CMS
    _COLORS: list = list(_CMS["tab20"].colors)
except Exception:
    _COLORS = [f"C{i}" for i in range(10)]

# Percentile colours, labels, and line widths
_PCT_COLORS  = {10: "#1f77b4", 50: "#222222", 90: "#d62728"}
_PCT_LABELS  = {10: "P10",     50: "P50",     90: "P90"}
_PCT_LW      = {10: 2.0,       50: 2.5,        90: 2.0}


class WellAlignmentDialog(QDialog):
    """Monthly oil production aligned to each well's first production month."""

    def __init__(self, df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Приведённая добыча по скважинам")
        self.resize(1300, 700)
        self._df = df

        self._phase: str = "oil"  # "oil" | "gas"
        self._wells: list[str] = self._producing_wells()

        # Percentile state
        self._show_pct: bool = False
        self._pct_months: np.ndarray | None = None
        self._pct_data:   dict | None = None
        self._pct_trends: dict = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel (controls) ─────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(260)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(6)
        splitter.addWidget(left)

        # Fluid phase selector
        phase_row = QHBoxLayout()
        phase_row.addWidget(QLabel("Флюид:"))
        self._cmb_phase = QComboBox()
        self._cmb_phase.addItem("Нефть", "oil")
        self._cmb_phase.addItem("Газ", "gas")
        phase_row.addWidget(self._cmb_phase)
        phase_row.addStretch()
        left_lay.addLayout(phase_row)

        grp_wells = QGroupBox("Скважины")
        gw_lay = QVBoxLayout(grp_wells)
        self._lst = QListWidget()
        self._lst.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for w in self._wells:
            self._lst.addItem(QListWidgetItem(w))
        # No default selection — user chooses explicitly
        gw_lay.addWidget(self._lst)
        btn_all = QPushButton("Выбрать все")
        btn_all.clicked.connect(self._lst.selectAll)
        gw_lay.addWidget(btn_all)
        btn_none = QPushButton("Снять выбор")
        btn_none.clicked.connect(self._lst.clearSelection)
        gw_lay.addWidget(btn_none)
        btn_filter = QPushButton("Загрузить список из файла…")
        btn_filter.clicked.connect(self._load_filter)
        gw_lay.addWidget(btn_filter)
        left_lay.addWidget(grp_wells)

        self._chk_log = QCheckBox("Log шкала (Y)")
        left_lay.addWidget(self._chk_log)

        btn_pct = QPushButton("Генерировать P90/P50/P10")
        btn_pct.clicked.connect(self._generate_percentiles)
        left_lay.addWidget(btn_pct)

        left_lay.addStretch()

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

        # Connections
        self._lst.itemSelectionChanged.connect(self._on_selection_changed)
        self._chk_log.stateChanged.connect(self._draw)
        self._cmb_phase.currentIndexChanged.connect(self._on_phase_changed)

        self._draw()

    # ── Phase selector ───────────────────────────────────────────────

    def _on_phase_changed(self, _index: int) -> None:
        """Switch fluid phase and rebuild well list + plot."""
        self._phase = self._cmb_phase.currentData()
        # Rebuild well list for the new phase
        new_wells = self._producing_wells()
        self._lst.blockSignals(True)
        self._lst.clear()
        for w in new_wells:
            self._lst.addItem(QListWidgetItem(w))
        self._lst.blockSignals(False)
        self._wells = new_wells
        # Clear percentile overlay
        self._show_pct = False
        self._pct_months = None
        self._pct_data = None
        self._pct_trends = {}
        self._draw()

    # ── Selection ────────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        """Clear percentile overlay whenever the well selection changes."""
        self._show_pct = False
        self._pct_months = None
        self._pct_data   = None
        self._pct_trends = {}
        self._draw()

    # ── Filter ────────────────────────────────────────────────────────────

    def _load_filter(self) -> None:
        """Open a .txt file and select wells whose names appear in it."""
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
        found = 0
        for i in range(self._lst.count()):
            item = self._lst.item(i)
            if item and item.text().lower() in name_set:
                item.setSelected(True)
                found += 1
        self._lst.blockSignals(False)
        # Emit one selection-changed event
        self._lst.itemSelectionChanged.emit()

    @staticmethod
    def _read_well_list(path: str) -> list[str]:
        """Read a plain-text file with one well name per line."""
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

    # ── Percentile generation ─────────────────────────────────────────

    def _generate_percentiles(self) -> None:
        """Compute P10/P50/P90 cross-well profiles and fit exponential trends.

        Acts as a toggle: a second click clears the overlay.
        """
        from PySide6.QtWidgets import QMessageBox

        # Toggle off
        if self._show_pct:
            self._show_pct = False
            self._pct_months = None
            self._pct_data   = None
            self._pct_trends = {}
            self._draw()
            return

        selected = [item.text() for item in self._lst.selectedItems()]
        if len(selected) < 3:
            QMessageBox.warning(
                self, "Недостаточно скважин",
                f"Для расчёта перцентилей необходимо не менее 3 скважин.\n"
                f"Выбрано: {len(selected)}.",
            )
            return

        # Build aligned series for all selected wells
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        max_month = 0
        for well in selected:
            result = self._aligned_series(well)
            if result is None:
                continue
            x, y = result
            series[well] = (x, y)
            max_month = max(max_month, int(x[-1]))

        if not series:
            return

        MIN_WELLS = 3   # minimum non-zero wells required per month bucket
        months_list: list[float] = []
        p10_list: list[float] = []
        p50_list: list[float] = []
        p90_list: list[float] = []

        for m in range(1, max_month + 1):
            vals = [
                y[m - 1]
                for x, y in series.values()
                if m <= len(y) and y[m - 1] > 0
            ]
            if len(vals) >= MIN_WELLS:
                arr = np.array(vals)
                months_list.append(float(m))
                p10_list.append(float(np.percentile(arr, 10)))
                p50_list.append(float(np.percentile(arr, 50)))
                p90_list.append(float(np.percentile(arr, 90)))

        if not months_list:
            QMessageBox.warning(
                self, "Нет данных",
                "Не найдено ни одного месяца с ≥ 3 скважинами с ненулевой добычей.",
            )
            return

        self._pct_months = np.array(months_list)
        self._pct_data = {
            10: np.array(p10_list),
            50: np.array(p50_list),
            90: np.array(p90_list),
        }

        # Fit exponential decline to each percentile curve
        self._pct_trends = {
            pct: self._fit_exp(self._pct_months, vals)
            for pct, vals in self._pct_data.items()
        }

        self._show_pct = True
        self._draw()

    @staticmethod
    def _fit_exp(
        months: np.ndarray, values: np.ndarray
    ) -> tuple[float, float] | None:
        """Fit q(t) = qi · exp(−Di · t).  Returns (qi, Di) or None on failure."""
        mask = values > 0
        if np.sum(mask) < 3:
            return None
        t = months[mask]
        q = values[mask]
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(
                lambda t, qi, Di: qi * np.exp(-Di * t),
                t, q,
                p0=[float(q[0]), 0.02],
                bounds=([0.0, 0.0], [np.inf, 5.0]),
                maxfev=5000,
            )
            qi, Di = float(popt[0]), float(np.clip(popt[1], 0.0, 5.0))
            return qi, Di
        except Exception:
            pass
        # Fallback: log-linear regression
        try:
            log_q = np.log(np.clip(q, 1e-12, None))
            coeffs = np.polyfit(t, log_q, 1)
            Di = float(np.clip(-coeffs[0], 0.0, 5.0))
            qi = float(np.exp(coeffs[1]))
            return qi, Di
        except Exception:
            return None

    # ── Data helpers ───────────────────────────────────────────────

    def _producing_wells(self) -> list[str]:
        """Return sorted list of wells that have production of the selected fluid."""
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
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (x_months, y_production) with x starting at 1 = first production month."""
        prod_col = COL_GAS if self._phase == "gas" else COL_OIL
        sub = self._df[self._df[COL_WELL] == well].copy()
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        if COL_DATE not in sub.columns or prod_col not in sub.columns:
            return None

        agg = sub.groupby(COL_DATE)[prod_col].sum().sort_index()
        if len(agg) == 0:
            return None

        # Trim everything before the first month with positive production
        positive_dates = agg[agg > 0].index
        if len(positive_dates) == 0:
            return None
        agg = agg[agg.index >= positive_dates[0]]

        x = np.arange(1, len(agg) + 1, dtype=float)
        y = agg.values.astype(float)
        return x, y

    # ── Drawing ─────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        selected = [item.text() for item in self._lst.selectedItems()]

        # Individual well lines — faded when percentile overlay is active
        well_alpha = 0.25 if self._show_pct else 0.85
        well_lw    = 0.8  if self._show_pct else 1.3

        for ki, well in enumerate(selected):
            result = self._aligned_series(well)
            if result is None:
                continue
            x, y = result
            color = _COLORS[ki % len(_COLORS)]
            ax.plot(x, y, color=color, linewidth=well_lw, label=well, alpha=well_alpha)

        # ── Percentile overlay — trendlines only ────────────────────────
        if self._show_pct and self._pct_months is not None:
            m = self._pct_months
            t_max = float(m[-1]) * 1.5   # extend trendlines beyond data

            for pct in (10, 50, 90):
                col = _PCT_COLORS[pct]
                lbl = _PCT_LABELS[pct]
                trend = self._pct_trends.get(pct)
                if trend is not None:
                    qi, Di = trend
                    x_ext = np.linspace(float(m[0]), t_max, 400)
                    y_ext = qi * np.exp(-Di * x_ext)
                    ax.plot(x_ext, y_ext, color=col, linewidth=2.0,
                            linestyle="--", label=lbl, zorder=5)

        ax.set_xlabel("Месяц от начала добычи")
        _y_label = "Газ, м\u00b3/мес" if self._phase == "gas" else "Нефть, т/мес"
        ax.set_ylabel(_y_label)
        ax.set_title("Приведённая добыча по скважинам")

        if self._chk_log.isChecked():
            try:
                ax.set_yscale("log")
            except Exception:
                pass

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if self._show_pct:
                # Show only the P10/P50/P90 trendline entries
                pct_lbls = set(_PCT_LABELS.values())
                handles = [h for h, l in zip(handles, labels) if l in pct_lbls]
                labels  = [l for l in labels if l in pct_lbls]
            fsize = 6 if len(selected) > 15 else 7
            leg = ax.legend(handles, labels, fontsize=fsize, loc="upper right",
                            ncol=max(1, len(labels) // 8))
            if leg is not None:
                leg.set_draggable(True)

        ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()

    # ── Export ────────────────────────────────────────────────────────────

    def _copy_data(self) -> None:
        """Copy selected wells + P90/P50/P10 data as tab-separated text.

        Columns: Месяц | Well1 | Well2 | … | P10 факт | P50 факт | P90 факт |
                 P10 тренд | P50 тренд | P90 тренд
        Rows cover month 1 … max(well_length, trend_extent).
        """
        from PySide6.QtWidgets import QApplication

        selected = [item.text() for item in self._lst.selectedItems()]

        # Build well arrays (indexed 0-based by month position)
        well_data: dict[str, np.ndarray] = {}
        max_month_w = 0
        for well in selected:
            result = self._aligned_series(well)
            if result is not None:
                _, y = result
                well_data[well] = y
                max_month_w = max(max_month_w, len(y))

        has_pct = self._pct_data is not None and self._pct_months is not None
        has_trend = has_pct and any(
            v is not None for v in self._pct_trends.values()
        )

        # Determine total row count (extend for trend forecast)
        max_month = max_month_w
        trend_max = max_month_w
        if has_pct and len(self._pct_months) > 0:
            trend_max = int(self._pct_months[-1] * 1.5)
            max_month = max(max_month, trend_max)

        if max_month == 0:
            return

        # Percentile raw data lookup
        pct_lookup: dict[int, list[float]] = {}
        if has_pct:
            for i, mo in enumerate(self._pct_months):
                pct_lookup[int(mo)] = [
                    float(self._pct_data[10][i]),
                    float(self._pct_data[50][i]),
                    float(self._pct_data[90][i]),
                ]

        # Header
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
            # Well production
            for y in well_data.values():
                row.append(f"{y[m-1]:.4g}" if m <= len(y) else "")
            # Raw percentile
            if has_pct:
                if m in pct_lookup:
                    row += [f"{v:.4g}" for v in pct_lookup[m]]
                else:
                    row += ["", "", ""]
            # Trend forecast
            if has_trend:
                for pct in (10, 50, 90):
                    tr = self._pct_trends.get(pct)
                    if tr is not None:
                        qi, Di = tr
                        row.append(f"{qi * np.exp(-Di * m):.4g}")
            rows.append("\t".join(row))

        QApplication.instance().clipboard().setText("\n".join(rows))

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
