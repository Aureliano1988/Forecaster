"""Total oil production distribution dialog — histogram of cumulative oil per well.

For each selected well the total (cumulative) oil production is computed.
The result is rendered as a histogram.  When log-scale is checked the
histogram uses log10(Qo) bins but labels show actual Qo ranges.
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
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.data.models import (
    COL_DATE, COL_GAS, COL_HOURS_WORK, COL_LIQUID, COL_OIL, COL_WATER,
    COL_WELL, COL_WORK_TYPE, WORK_TYPE_OIL,
)

# (display_name, internal_key, x_label, unit)
_DIST_PARAMS = [
    ("Нак. нефть",            "total_oil",     "Нак. нефть, т"),
    ("Нач. дебит нефти",     "init_oil_rate", "Нач. дебит нефти, т/сут"),
    ("Макс. дебит нефти",    "max_oil_rate",  "Макс. дебит нефти, т/сут"),
    ("Нач. дебит жидкости",   "init_liq_rate", "Нач. дебит жидк., т/сут"),
    ("Макс. дебит жидкости",  "max_liq_rate",  "Макс. дебит жидк., т/сут"),
    ("Нач. обводнённость",     "init_wcut",     "Нач. обводнённость"),
]


class ProductionDistributionDialog(QDialog):
    """Histogram of per-well metrics distribution."""

    def __init__(
        self,
        df: pd.DataFrame,
        parent=None,
        well_coords_path: str = "",
        well_coords_mapping: dict | None = None,
        well_coords_delimiters: dict | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Распределение")
        self.resize(1100, 650)
        self._df = df
        self._coords_path = well_coords_path
        self._coords_mapping = well_coords_mapping or {}
        self._coords_delimiters = well_coords_delimiters or {}

        # Per-well metrics: {param_key: {well: value}}
        self._well_metrics: dict[str, dict[str, float]] = {}
        self._build_cache()

        self._build_ui()

        # Populate well list from wells that have at least total_oil > 0
        all_wells = sorted(self._well_metrics.get("total_oil", {}).keys())
        for w in all_wells:
            self._lst.addItem(QListWidgetItem(w))

        self._draw()

    # ── Cache ──────────────────────────────────────────────────────────────

    def _build_cache(self) -> None:
        sub = self._df
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        if COL_WELL not in sub.columns or COL_OIL not in sub.columns:
            return

        grp = sub.groupby(COL_WELL)

        # Total oil
        totals = grp[COL_OIL].sum()
        wells = [str(w) for w in totals.index if pd.notna(w) and totals[w] > 0]
        self._well_metrics["total_oil"] = {
            w: float(totals[w]) for w in wells
        }

        # Compute liquid per row if not present
        has_water = COL_WATER in sub.columns
        has_liquid = COL_LIQUID in sub.columns
        has_gas = COL_GAS in sub.columns
        has_hours = COL_HOURS_WORK in sub.columns

        # Per-well first-month and max-month rates
        init_oil: dict[str, float] = {}
        max_oil: dict[str, float] = {}
        init_liq: dict[str, float] = {}
        max_liq: dict[str, float] = {}
        init_wcut: dict[str, float] = {}

        for w in wells:
            try:
                ws = grp.get_group(w).sort_values(COL_DATE)
            except KeyError:
                continue
            oil_m = ws[COL_OIL].values.astype(float)
            hours_m = ws[COL_HOURS_WORK].values.astype(float) if has_hours else np.full(len(ws), 730.5)
            days = hours_m / 24.0
            days = np.where(days > 0, days, np.nan)

            oil_rate = oil_m / days  # t/day
            valid_oil = np.where(np.isfinite(oil_rate) & (oil_rate > 0), oil_rate, np.nan)

            # First valid rate
            first_idx = np.nanargmin(np.where(np.isfinite(valid_oil), np.arange(len(valid_oil)), np.inf))
            if np.isfinite(valid_oil[first_idx]):
                init_oil[w] = float(valid_oil[first_idx])
            max_v = np.nanmax(valid_oil) if np.any(np.isfinite(valid_oil)) else np.nan
            if np.isfinite(max_v):
                max_oil[w] = float(max_v)

            # Liquid rate
            if has_liquid:
                liq_m = ws[COL_LIQUID].values.astype(float)
            elif has_water:
                liq_m = oil_m + ws[COL_WATER].values.astype(float)
            else:
                liq_m = oil_m
            liq_rate = liq_m / days
            valid_liq = np.where(np.isfinite(liq_rate) & (liq_rate > 0), liq_rate, np.nan)
            if np.isfinite(valid_liq[first_idx]):
                init_liq[w] = float(valid_liq[first_idx])
            max_lv = np.nanmax(valid_liq) if np.any(np.isfinite(valid_liq)) else np.nan
            if np.isfinite(max_lv):
                max_liq[w] = float(max_lv)

            # Initial watercut
            if has_water:
                water_m = ws[COL_WATER].values.astype(float)
                if liq_m[first_idx] > 0:
                    init_wcut[w] = float(water_m[first_idx] / liq_m[first_idx])

        self._well_metrics["init_oil_rate"] = init_oil
        self._well_metrics["max_oil_rate"] = max_oil
        self._well_metrics["init_liq_rate"] = init_liq
        self._well_metrics["max_liq_rate"] = max_liq
        self._well_metrics["init_wcut"] = init_wcut

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(240)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(6)
        splitter.addWidget(left)

        # Well list
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

        btn_filter = QPushButton("Список из файла\u2026")
        btn_filter.clicked.connect(self._load_filter)
        gw_lay.addWidget(btn_filter)

        btn_criteria = QPushButton("Выбрать по критерию\u2026")
        btn_criteria.clicked.connect(self._on_criteria)
        gw_lay.addWidget(btn_criteria)

        if self._coords_path:
            btn_map = QPushButton("Выбрать на карте\u2026")
            btn_map.clicked.connect(self._on_select_on_map)
            gw_lay.addWidget(btn_map)

        left_lay.addWidget(grp_wells)

        # Parameter selector
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Параметр:"))
        self._cmb_param = QComboBox()
        for disp, key, _ in _DIST_PARAMS:
            self._cmb_param.addItem(disp, key)
        param_row.addWidget(self._cmb_param, 1)
        left_lay.addLayout(param_row)

        # Bins
        bins_row = QHBoxLayout()
        bins_row.addWidget(QLabel("Кол-во бинов:"))
        self._spn_bins = QSpinBox()
        self._spn_bins.setRange(3, 100)
        self._spn_bins.setValue(15)
        bins_row.addWidget(self._spn_bins)
        bins_row.addStretch()
        left_lay.addLayout(bins_row)

        # Log-scale
        self._chk_log = QCheckBox("Log-шкала")
        self._chk_log.setToolTip(
            "Использовать log₁₀(Qo) для разбиения,\n"
            "но показывать диапазоны Qo на оси X."
        )
        left_lay.addWidget(self._chk_log)

        # Min total production
        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Мин. значение:"))
        self._spn_min = QDoubleSpinBox()
        self._spn_min.setRange(0.0, 1e9)
        self._spn_min.setDecimals(0)
        self._spn_min.setValue(1000.0)
        min_row.addWidget(self._spn_min)
        min_row.addStretch()
        left_lay.addLayout(min_row)

        # Cumulative curve
        self._chk_cum = QCheckBox("Показать накопленную кривую")
        left_lay.addWidget(self._chk_cum)

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

        from src.ui.hover_tooltip import install_hover_tooltip
        install_hover_tooltip(self._canvas, self._fig)

        btns = QHBoxLayout()
        btn_clip = QPushButton("Копировать график")
        btn_save = QPushButton("Сохранить картинку…")
        btn_data = QPushButton("Скопировать данные")
        btn_clip.clicked.connect(self._to_clipboard)
        btn_save.clicked.connect(self._save_image)
        btn_data.clicked.connect(self._copy_data)
        for b in (btn_clip, btn_save, btn_data):
            btns.addWidget(b)
        btns.addStretch()
        right_lay.addLayout(btns)

        # ── Connections ──────────────────────────────────────────────────────────
        self._lst.itemSelectionChanged.connect(self._draw)
        self._cmb_param.currentIndexChanged.connect(self._draw)
        self._spn_bins.valueChanged.connect(self._draw)
        self._chk_log.stateChanged.connect(self._draw)
        self._spn_min.valueChanged.connect(self._draw)
        self._chk_cum.stateChanged.connect(self._draw)

    # ── Selection helpers ────────────────────────────────────

    def _apply_well_selection(self, names: list[str]) -> None:
        """Select list items whose name (case-insensitive) is in *names*."""
        name_set = {n.lower() for n in names}
        self._lst.blockSignals(True)
        self._lst.clearSelection()
        for i in range(self._lst.count()):
            item = self._lst.item(i)
            if item and item.text().lower() in name_set:
                item.setSelected(True)
        self._lst.blockSignals(False)
        self._lst.itemSelectionChanged.emit()

    def _on_criteria(self) -> None:
        from src.ui.well_criteria_dialog import WellCriteriaDialog
        cur = [item.text() for item in self._lst.selectedItems()]
        dlg = WellCriteriaDialog(self._df, current_wells=cur, parent=self)
        if dlg.exec() != WellCriteriaDialog.DialogCode.Accepted:
            return
        self._apply_well_selection(dlg.matched_wells())

    def _on_select_on_map(self) -> None:
        from src.ui.well_location_dialog import WellLocationDialog
        cur = [item.text() for item in self._lst.selectedItems()]
        dlg = WellLocationDialog(
            self._coords_path, self._coords_mapping, self._coords_delimiters,
            parent=self, selection_mode=True, initial_selection=cur,
        )
        if dlg.exec() != WellLocationDialog.DialogCode.Accepted:
            return
        self._apply_well_selection(dlg.result_selected_wells())

    # ── Filter from file ──────────────────────────────────────

    def _load_filter(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить список скважин", "",
            "Текстовые файлы (*.txt);;Все файлы (*)",
        )
        if not path:
            return
        names: list[str] = []
        for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                with open(path, encoding=enc) as fh:
                    for line in fh:
                        name = line.strip()
                        if name and not name.startswith("#"):
                            names.append(name)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if not names:
            return
        self._apply_well_selection(names)

    # ── Drawing ──────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        selected = [item.text() for item in self._lst.selectedItems()]
        param_key = self._cmb_param.currentData()
        param_label = _DIST_PARAMS[self._cmb_param.currentIndex()][2]
        metrics = self._well_metrics.get(param_key, {})
        min_val = self._spn_min.value()

        # For ratio parameters (watercut 0-1), skip the min-value filter
        # but still exclude zeros for initial rates and watercut
        skip_min = param_key in ("init_wcut",)
        exclude_zero = param_key in ("init_oil_rate", "init_liq_rate", "init_wcut")
        values = np.array([
            metrics[w] for w in selected
            if w in metrics
            and (skip_min or metrics[w] > min_val)
            and (not exclude_zero or metrics[w] > 0)
        ])

        if len(values) == 0:
            ax.set_title("Нет данных")
            self._canvas.draw_idle()
            return

        n_bins = self._spn_bins.value()
        use_log = self._chk_log.isChecked()
        x_lbl = param_label

        if use_log:
            log_vals = np.log10(np.clip(values, 1e-6, None))
            counts, bin_edges = np.histogram(log_vals, bins=n_bins)
            real_edges = 10.0 ** bin_edges
            widths = np.diff(bin_edges)
            ax.bar(bin_edges[:-1], counts, width=widths, align="edge",
                   color="steelblue", edgecolor="white", linewidth=0.5)
            tick_pos = bin_edges
            tick_labels = [f"{v:,.1f}" if max(real_edges) < 100 else f"{v:,.0f}" for v in real_edges]
            if len(tick_pos) > 12:
                step = max(1, len(tick_pos) // 8)
                tick_pos = tick_pos[::step]
                tick_labels = tick_labels[::step]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            ax.set_xlabel(f"{x_lbl} (log-шкала)")
        else:
            counts, bin_edges, _patches = ax.hist(
                values, bins=n_bins,
                color="steelblue", edgecolor="white", linewidth=0.5,
            )
            ax.set_xlabel(x_lbl)
            ax.ticklabel_format(axis="x", style="plain")
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")
                lbl.set_fontsize(7)

        # P50 vertical line
        p50 = float(np.median(values))
        if use_log:
            p50_pos = float(np.log10(max(p50, 1e-6)))
        else:
            p50_pos = p50
        ax.axvline(p50_pos, color="red", linestyle="--", linewidth=1.2, zorder=5)
        ax.text(
            p50_pos, ax.get_ylim()[1] * 0.95,
            f"  P50={p50:,.0f}",
            color="red", fontsize=8, va="top",
        )

        ax.set_ylabel("Кол-во скважин")
        ax.set_title(f"Распределение: {param_label} ({len(values)} скв.)")
        ax.grid(True, axis="y", alpha=0.3)

        # Cumulative curve on Y2
        if self._chk_cum.isChecked() and len(counts) > 0:
            cum = np.cumsum(counts).astype(float)
            cum_pct = cum / cum[-1] * 100.0
            mid = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            ax2 = ax.twinx()
            ax2.plot(mid, cum_pct, color="red", linewidth=1.5,
                     marker=".", markersize=4, label="Накопленная, %")
            ax2.set_ylabel("Накопл. %", fontsize=9)
            ax2.set_ylim(0, 105)
            ax2.legend(fontsize=7, loc="center right")

        # Store for clipboard
        self._last_counts = counts
        self._last_edges = bin_edges if not use_log else real_edges

        self._canvas.draw_idle()

    # ── Export ────────────────────────────────────────────────────────────────

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

    def _copy_data(self) -> None:
        from PySide6.QtWidgets import QApplication
        counts = getattr(self, "_last_counts", None)
        edges = getattr(self, "_last_edges", None)
        if counts is None or edges is None:
            return
        hdr = ["От", "До", "Кол-во скв."]
        rows = ["\t".join(hdr)]
        for i in range(len(counts)):
            rows.append(f"{edges[i]:,.0f}\t{edges[i+1]:,.0f}\t{int(counts[i])}")
        QApplication.instance().clipboard().setText("\n".join(rows))
