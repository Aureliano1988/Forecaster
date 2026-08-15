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
    COL_DATE, COL_OIL, COL_WELL, COL_WORK_TYPE, WORK_TYPE_OIL,
)


class ProductionDistributionDialog(QDialog):
    """Histogram of total oil production per well."""

    def __init__(self, df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Распределение НДН")
        self.resize(1100, 650)
        self._df = df

        # Pre-compute per-well cumulative oil
        self._well_qo: dict[str, float] = {}
        self._build_cache()

        self._build_ui()

        # Populate well list
        for w in sorted(self._well_qo.keys()):
            self._lst.addItem(QListWidgetItem(w))

        self._draw()

    # ── Cache ────────────────────────────────────────────────────────────────

    def _build_cache(self) -> None:
        sub = self._df
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        if COL_WELL not in sub.columns or COL_OIL not in sub.columns:
            return
        totals = sub.groupby(COL_WELL)[COL_OIL].sum()
        self._well_qo = {
            str(w): float(v) for w, v in totals.items()
            if pd.notna(w) and v > 0
        }

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

        left_lay.addWidget(grp_wells)

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
        min_row.addWidget(QLabel("Мин. НДН, т:"))
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

        # ── Connections ──────────────────────────────────────────────────────
        self._lst.itemSelectionChanged.connect(self._draw)
        self._spn_bins.valueChanged.connect(self._draw)
        self._chk_log.stateChanged.connect(self._draw)
        self._spn_min.valueChanged.connect(self._draw)
        self._chk_cum.stateChanged.connect(self._draw)

    # ── Criteria selection ─────────────────────────────────────────────────

    def _on_criteria(self) -> None:
        from src.ui.well_criteria_dialog import WellCriteriaDialog
        dlg = WellCriteriaDialog(self._df, parent=self)
        if dlg.exec() != WellCriteriaDialog.DialogCode.Accepted:
            return
        matched = dlg.matched_wells()
        name_set = {n.lower() for n in matched}
        self._lst.blockSignals(True)
        self._lst.clearSelection()
        for i in range(self._lst.count()):
            item = self._lst.item(i)
            if item and item.text().lower() in name_set:
                item.setSelected(True)
        self._lst.blockSignals(False)
        self._lst.itemSelectionChanged.emit()

    # ── Filter from file ─────────────────────────────────────────────────────

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
        name_set = {n.lower() for n in names}
        self._lst.blockSignals(True)
        self._lst.clearSelection()
        for i in range(self._lst.count()):
            item = self._lst.item(i)
            if item and item.text().lower() in name_set:
                item.setSelected(True)
        self._lst.blockSignals(False)
        self._lst.itemSelectionChanged.emit()

    # ── Drawing ──────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        selected = [item.text() for item in self._lst.selectedItems()]
        min_qo = self._spn_min.value()
        values = np.array([
            self._well_qo[w] for w in selected
            if w in self._well_qo and self._well_qo[w] > min_qo
        ])

        if len(values) == 0:
            ax.set_title("Нет данных")
            self._canvas.draw_idle()
            return

        n_bins = self._spn_bins.value()
        use_log = self._chk_log.isChecked()

        if use_log:
            log_vals = np.log10(np.clip(values, 1e-6, None))
            counts, bin_edges = np.histogram(log_vals, bins=n_bins)
            # Convert log edges back to real Qo for display
            real_edges = 10.0 ** bin_edges
            widths = np.diff(bin_edges)
            ax.bar(bin_edges[:-1], counts, width=widths, align="edge",
                   color="steelblue", edgecolor="white", linewidth=0.5)
            # X-axis: show real Qo values at log positions
            tick_pos = bin_edges
            tick_labels = [f"{v:,.0f}" for v in real_edges]
            # Show every other tick if too many
            if len(tick_pos) > 12:
                step = max(1, len(tick_pos) // 8)
                tick_pos = tick_pos[::step]
                tick_labels = tick_labels[::step]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            ax.set_xlabel("Нак. нефть, т (log-шкала)")
        else:
            counts, bin_edges, _patches = ax.hist(
                values, bins=n_bins,
                color="steelblue", edgecolor="white", linewidth=0.5,
            )
            ax.set_xlabel("Нак. нефть, т")
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
        ax.set_title(f"Распределение НДН ({len(values)} скв.)")
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
