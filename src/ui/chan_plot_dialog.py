"""Chan diagnostic plot — WOR and WOR' vs elapsed time (log-log).

For each selected well:
  - X-axis: elapsed time since first production, in days (log scale)
  - Y-axis: WOR = qw/qo  and  WOR' = ΔWOR/Δt  (log scale)
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.data.models import (
    COL_DATE, COL_OIL, COL_WATER, COL_WELL, COL_WORK_TYPE, WORK_TYPE_OIL,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
try:
    from matplotlib import colormaps as _CMS
    _COLORS: list = list(_CMS["tab20"].colors)
except Exception:
    _COLORS = [f"C{i}" for i in range(10)]


class ChanPlotDialog(QDialog):
    """Chan diagnostic plot: WOR and WOR' vs elapsed days (log-log)."""

    def __init__(self, df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("График Чена")
        self.resize(1300, 700)
        self._df = df

        self._build_ui()

        self._wells = self._producing_wells()
        for w in self._wells:
            self._lst.addItem(QListWidgetItem(w))

        self._draw()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ──────────────────────────────────────────────────────
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
        btn_all  = QPushButton("Все")
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

        left_lay.addWidget(grp_wells)

        # Smoothing slider
        grp_smooth = QGroupBox("Сглаживание")
        gs_lay = QVBoxLayout(grp_smooth)
        self._lbl_smooth = QLabel("Окно: 1 (без сглаживания)")
        gs_lay.addWidget(self._lbl_smooth)
        self._sld_smooth = QSlider(Qt.Orientation.Horizontal)
        self._sld_smooth.setRange(1, 12)
        self._sld_smooth.setValue(1)
        self._sld_smooth.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._sld_smooth.setTickInterval(1)
        gs_lay.addWidget(self._sld_smooth)
        left_lay.addWidget(grp_smooth)

        left_lay.addStretch()

        # ── Right panel (plot) ───────────────────────────────────────────────
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
        btn_clip     = QPushButton("Копировать график")
        btn_save_img = QPushButton("Сохранить картинку\u2026")
        btn_data     = QPushButton("Скопировать данные")
        btn_clip.clicked.connect(self._to_clipboard)
        btn_save_img.clicked.connect(self._save_image)
        btn_data.clicked.connect(self._copy_data)
        for b in (btn_clip, btn_save_img, btn_data):
            btns.addWidget(b)
        btns.addStretch()
        right_lay.addLayout(btns)

        # ── Connections ────────────────────────────────────────────────────
        self._lst.itemSelectionChanged.connect(self._draw)
        self._sld_smooth.valueChanged.connect(self._on_smooth_changed)

    def _on_smooth_changed(self, val: int) -> None:
        if val <= 1:
            self._lbl_smooth.setText("Окно: 1 (без сглаживания)")
        else:
            self._lbl_smooth.setText(f"Окно: {val} мес.")
        self._draw()

    # ── Filter ─────────────────────────────────────────────────────────────────

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

    # ── Data helpers ───────────────────────────────────────────────────────────

    def _producing_wells(self) -> list[str]:
        if COL_WELL not in self._df.columns or COL_OIL not in self._df.columns:
            return []
        sub = self._df
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        totals = sub.groupby(COL_WELL)[COL_OIL].sum()
        return sorted(totals[totals > 0].index.tolist())

    def _chan_series(
        self, well: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (t_wor, wor, t_wor_prime, wor_prime) for one well.

        t values are elapsed days since first production.
        Only months with qo > 0 and qw > 0 contribute to WOR.
        WOR' = ΔWOR / Δt (forward finite difference).
        """
        sub = self._df[self._df[COL_WELL] == well].copy()
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        if COL_DATE not in sub.columns or COL_OIL not in sub.columns:
            return None
        water_col = COL_WATER if COL_WATER in sub.columns else None
        if water_col is None:
            return None

        agg = sub.groupby(COL_DATE).agg({COL_OIL: "sum", water_col: "sum"}).sort_index()
        qo = agg[COL_OIL].values.astype(float)
        qw = agg[water_col].values.astype(float)
        dates = agg.index

        # First positive oil date
        pos_mask = qo > 0
        if not np.any(pos_mask):
            return None
        first_idx = int(np.argmax(pos_mask))
        first_date = pd.Timestamp(dates[first_idx])

        # Elapsed months from first production (1-based index)
        elapsed = np.arange(len(dates), dtype=float) - first_idx + 1

        # WOR: only where both qo > 0 and qw >= 0
        wor_mask = (qo > 0) & (elapsed >= 0)
        t_wor = elapsed[wor_mask]
        wor = np.divide(qw[wor_mask], qo[wor_mask])

        if len(wor) < 2:
            return None

        # Filter to positive WOR for log-scale
        pos_wor = wor > 0
        t_wor = t_wor[pos_wor]
        wor = wor[pos_wor]
        if len(wor) < 2:
            return None

        # Optional moving-average smoothing of WOR before derivative
        win = self._sld_smooth.value()
        if win > 1 and len(wor) >= win:
            kernel = np.ones(win) / win
            wor_smooth = np.convolve(wor, kernel, mode="valid")
            # Align t to the centre of each window
            half = (win - 1) // 2
            t_wor = t_wor[half: half + len(wor_smooth)]
            wor = wor_smooth
        if len(wor) < 3:
            return t_wor, wor, np.array([]), np.array([])

        # WOR' = (WOR(i+1) - WOR(i-1)) / (t(i+1) - t(i-1))  — symmetric derivative
        n = len(wor)
        t_wor_prime = t_wor[1:n-1]
        dt = t_wor[2:] - t_wor[:n-2]
        d_wor = wor[2:] - wor[:n-2]
        valid = dt > 0
        t_wor_prime = t_wor_prime[valid]
        wor_prime = np.abs(d_wor[valid] / dt[valid])

        # Keep only positive WOR' for log scale
        pos_prime = wor_prime > 0
        t_wor_prime = t_wor_prime[pos_prime]
        wor_prime = wor_prime[pos_prime]

        return t_wor, wor, t_wor_prime, wor_prime

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        selected = [item.text() for item in self._lst.selectedItems()]
        if not selected:
            ax.set_title("График Чена")
            self._canvas.draw_idle()
            return

        has_data = False
        for ki, well in enumerate(selected):
            result = self._chan_series(well)
            if result is None:
                continue
            t_wor, wor, t_wp, wor_prime = result
            color = _COLORS[ki % len(_COLORS)]
            if len(t_wor) > 0:
                ax.plot(t_wor, wor, "o-", color=color, ms=3,
                        linewidth=1.2, label=f"{well} WOR", alpha=0.85)
                has_data = True
            if len(t_wp) > 0:
                ax.plot(t_wp, wor_prime, "s--", color=color, ms=3,
                        linewidth=1.0, label=f"{well} WOR'", alpha=0.65)

        if has_data:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel("Месяц от начала добычи")
        ax.set_ylabel("WOR / WOR'")
        ax.set_title(f"График Чена  ({len(selected)} скв.)")

        from src.ui.legend_helper import fit_legend
        fit_legend(ax, loc="upper left")

        ax.grid(True, alpha=0.3, which="both")
        self._canvas.draw_idle()

    # ── Export ─────────────────────────────────────────────────────────────────

    def _copy_data(self) -> None:
        from PySide6.QtWidgets import QApplication

        selected = [item.text() for item in self._lst.selectedItems()]
        rows: list[str] = ["Скважина\tМесяц\tWOR\tWOR'"]
        for well in selected:
            result = self._chan_series(well)
            if result is None:
                continue
            t_wor, wor, t_wp, wor_prime = result
            # Build a dict for WOR' keyed by t for easy lookup
            wp_dict: dict[float, float] = {}
            for t_v, wp_v in zip(t_wp, wor_prime):
                wp_dict[float(t_v)] = float(wp_v)
            for t_v, w_v in zip(t_wor, wor):
                wp_str = f"{wp_dict[float(t_v)]:.6g}" if float(t_v) in wp_dict else ""
                rows.append(f"{well}\t{t_v:.0f}\t{w_v:.6g}\t{wp_str}")

        QApplication.instance().clipboard().setText("\n".join(rows))

    def _to_clipboard(self) -> None:
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        QApplication.instance().clipboard().setImage(
            QImage.fromData(buf.getvalue())
        )

    def _save_image(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить картинку", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
