"""Well Vintage dialog — stacked area chart grouped by first-production year.

For each selected well the year of its first positive production is identified
(the "vintage year").  Wells are then grouped by that year and each group's
total monthly production is summed onto a shared calendar-date axis.
The result is rendered as a stacked area chart so the contribution of each
vintage class to total production is immediately visible over calendar time.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.data.models import (
    COL_DATE, COL_GAS, COL_LIQUID, COL_OIL, COL_WATER, COL_WELL,
    COL_WORK_TYPE, WORK_TYPE_OIL,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
try:
    from matplotlib import colormaps as _CMS
    _COLORS: list = list(_CMS["tab20"].colors)
except Exception:
    _COLORS = [f"C{i}" for i in range(10)]


class WellVintageDialog(QDialog):
    """Monthly production stacked by well vintage year (first-production year)."""

    def __init__(self, df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Группировка скважин по годам ввода")
        self.resize(1300, 700)
        self._df = df
        self._phase: str = "oil"

        self._build_ui()

        # ── Pre-computed state (rebuilt on phase change) ─────────────────────
        # Production-only sub-frame and vintage-year lookup table.
        # Both are computed once per phase via _build_cache() so that _draw()
        # never performs per-well DataFrame scans.
        self._df_prod: pd.DataFrame = pd.DataFrame()
        self._vintage_cache: dict[str, int] = {}   # well name → vintage year
        self._build_cache()

        # Populate well list after UI is built
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

        # Fluid
        phase_row = QHBoxLayout()
        phase_row.addWidget(QLabel("Флюид:"))
        self._cmb_phase = QComboBox()
        self._cmb_phase.addItem("Нефть", "oil")
        self._cmb_phase.addItem("Газ",   "gas")
        phase_row.addWidget(self._cmb_phase)
        phase_row.addStretch()
        left_lay.addLayout(phase_row)

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

        btn_criteria = QPushButton("Выбрать по критерию\u2026")
        btn_criteria.clicked.connect(self._on_criteria)
        gw_lay.addWidget(btn_criteria)

        left_lay.addWidget(grp_wells)

        # Display mode
        self._rb_production = QRadioButton("Добыча")
        self._rb_well_count = QRadioButton("Кол-во скважин")
        self._rb_qo_wor = QRadioButton("Qo vs ВНФ")
        self._rb_production.setChecked(True)
        left_lay.addWidget(self._rb_production)
        left_lay.addWidget(self._rb_well_count)
        left_lay.addWidget(self._rb_qo_wor)

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
        btn_data     = QPushButton("Скопировать добычу")
        btn_wells    = QPushButton("Скопировать список")
        btn_clip.clicked.connect(self._to_clipboard)
        btn_save_img.clicked.connect(self._save_image)
        btn_data.clicked.connect(self._copy_data)
        btn_wells.clicked.connect(self._copy_well_list)
        for b in (btn_clip, btn_save_img, btn_data, btn_wells):
            btns.addWidget(b)
        btns.addStretch()
        right_lay.addLayout(btns)

        # ── Connections ───────────────────────────────────────────────────────────────
        self._lst.itemSelectionChanged.connect(self._draw)
        self._cmb_phase.currentIndexChanged.connect(self._on_phase_changed)
        self._rb_production.toggled.connect(self._draw)
        self._rb_well_count.toggled.connect(self._draw)
        self._rb_qo_wor.toggled.connect(self._draw)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _build_cache(self) -> None:
        """Pre-filter to production rows and compute vintage years for ALL wells.

        Uses a single ``groupby(COL_WELL)`` across the entire dataset so that
        _draw() never needs per-well DataFrame scans.  Call once at init and
        again whenever the fluid phase changes.
        """
        prod_col = COL_GAS if self._phase == "gas" else COL_OIL

        # ── Pre-filtered production frame ────────────────────────────────────
        sub = self._df
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        self._df_prod = sub

        # ── Vectorised vintage-year computation (one groupby for all wells) ──
        if (
            prod_col not in sub.columns
            or COL_DATE not in sub.columns
            or COL_WELL not in sub.columns
        ):
            self._vintage_cache = {}
            return

        positive = sub[sub[prod_col] > 0]
        first_dates = positive.groupby(COL_WELL)[COL_DATE].min()
        self._vintage_cache = {
            well: int(pd.Timestamp(d).year)
            for well, d in first_dates.items()
            if pd.notna(d)
        }

    # ── Phase ───────────────────────────────────────────────────────────────

    def _on_phase_changed(self, _idx: int) -> None:
        self._phase = self._cmb_phase.currentData()
        # Rebuild the pre-filtered frame and vintage cache for the new phase
        self._build_cache()
        # Rebuild well list for the new phase
        new_wells = self._producing_wells()
        self._lst.blockSignals(True)
        self._lst.clear()
        for w in new_wells:
            self._lst.addItem(QListWidgetItem(w))
        self._lst.blockSignals(False)
        self._wells = new_wells
        self._draw()

    # ── Criteria selection ─────────────────────────────────────────────────

    def _on_criteria(self) -> None:
        from src.ui.well_criteria_dialog import WellCriteriaDialog
        cur = [item.text() for item in self._lst.selectedItems()]
        dlg = WellCriteriaDialog(self._df, current_wells=cur, parent=self)
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
        """Return sorted list of wells that have any positive production."""
        prod_col = COL_GAS if self._phase == "gas" else COL_OIL
        if COL_WELL not in self._df.columns or prod_col not in self._df.columns:
            return []
        sub = self._df
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        totals = sub.groupby(COL_WELL)[prod_col].sum()
        return sorted(totals[totals > 0].index.tolist())

    def _build_vintage_data(
        self, selected: list[str]
    ) -> tuple[list[pd.Timestamp], dict[int, np.ndarray]] | None:
        """
        Build a shared calendar-date axis and a production array per vintage year.

        Returns (dates, {year: array}) where the arrays are aligned to *dates*
        and zeros fill months with no production for a given group.
        Returns None when there is no usable data.
        """
        prod_col = COL_GAS if self._phase == "gas" else COL_OIL

        # ── Assign each well to a vintage year via O(1) cache lookups ────────
        vintage: dict[str, int] = {
            w: self._vintage_cache[w]
            for w in selected
            if w in self._vintage_cache
        }
        if not vintage:
            return None

        # ── Sum production per vintage per calendar month ──────────────────
        # Use pre-filtered production frame — no work-type re-filtering needed
        sub = self._df_prod[self._df_prod[COL_WELL].isin(list(vintage.keys()))].copy()
        if COL_DATE not in sub.columns or prod_col not in sub.columns:
            return None

        # Attach vintage year to each row and aggregate
        sub["_vintage"] = sub[COL_WELL].map(vintage).astype(int)

        # Pivot: index = date, columns = vintage year, values = sum of prod_col
        pivot = (
            sub.groupby([COL_DATE, "_vintage"])[prod_col]
            .sum()
            .unstack(fill_value=0.0)
            .sort_index()
        )
        if pivot.empty:
            return None

        # Build aligned arrays (all years_sorted are guaranteed pivot columns)
        all_dates: list[pd.Timestamp] = [pd.Timestamp(d) for d in pivot.index]
        years_sorted = sorted(pivot.columns.tolist())
        arrays: dict[int, np.ndarray] = {
            yr: pivot[yr].values.astype(float)
            for yr in years_sorted
        }
        return all_dates, arrays

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        selected = [item.text() for item in self._lst.selectedItems()]
        if not selected:
            ax.set_title("Группировка скважин по годам ввода")
            self._canvas.draw_idle()
            return

        if self._rb_well_count.isChecked():
            self._draw_well_count(ax, selected)
        elif self._rb_qo_wor.isChecked():
            self._draw_qo_wor(ax, selected)
        else:
            self._draw_production(ax, selected)

        ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()

    def _draw_production(self, ax, selected: list[str]) -> None:
        """Stacked area chart of production by vintage year."""
        result = self._build_vintage_data(selected)
        if result is None:
            ax.set_title("Нет данных")
            return

        dates, arrays = result
        years = sorted(arrays.keys())

        date_nums = mdates.date2num(dates)
        layers = [arrays[yr] for yr in years]
        colors = [_COLORS[i % len(_COLORS)] for i in range(len(years))]

        ax.stackplot(
            date_nums, layers,
            labels=[str(yr) for yr in years],
            colors=colors, alpha=0.85,
        )

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        self._fig.autofmt_xdate(rotation=45, ha="right")

        y_lbl = "Газ, м\u00b3/мес" if self._phase == "gas" else "Нефть, т/мес"
        ax.set_ylabel(y_lbl)
        ax.set_xlabel("Дата")
        n_wells = len([w for w in selected if w in self._vintage_cache])
        ax.set_title(
            f"Группировка скважин по годам ввода  ({n_wells} скв., {len(years)} групп)"
        )

        from src.ui.legend_helper import fit_legend
        fit_legend(ax, loc="upper right")

    def _draw_well_count(self, ax, selected: list[str]) -> None:
        """Bar chart: number of wells per vintage year."""
        vintage = {
            w: self._vintage_cache[w]
            for w in selected if w in self._vintage_cache
        }
        if not vintage:
            ax.set_title("Нет данных")
            return

        from collections import Counter
        counts = Counter(vintage.values())
        years = sorted(counts.keys())
        vals = [counts[yr] for yr in years]

        colors = [_COLORS[i % len(_COLORS)] for i in range(len(years))]
        ax.bar([str(yr) for yr in years], vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Год ввода")
        ax.set_ylabel("Кол-во скважин")
        ax.set_title(f"Кол-во скважин по годам ввода ({sum(vals)} скв.)")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")
            lbl.set_fontsize(8)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def _draw_qo_wor(self, ax, selected: list[str]) -> None:
        """Cumulative Qo vs WOR: each vintage group includes all wells from that year and earlier."""
        prod_col = COL_GAS if self._phase == "gas" else COL_OIL
        vintage: dict[str, int] = {
            w: self._vintage_cache[w]
            for w in selected if w in self._vintage_cache
        }
        if not vintage:
            ax.set_title("Нет данных")
            return

        sub = self._df_prod[self._df_prod[COL_WELL].isin(list(vintage.keys()))].copy()
        if COL_DATE not in sub.columns or prod_col not in sub.columns:
            ax.set_title("Нет данных")
            return

        water_col = COL_WATER if COL_WATER in sub.columns else None
        if water_col is None:
            ax.set_title("Нет данных по воде")
            return

        years_sorted = sorted(set(vintage.values()))

        # For each vintage year, include all wells started in that year or earlier,
        # but only plot from 01.01 of that year onward (where new wells diverge)
        for ki, yr in enumerate(years_sorted):
            wells_up_to = [w for w, v in vintage.items() if v <= yr]
            if not wells_up_to:
                continue
            ws = sub[sub[COL_WELL].isin(wells_up_to)]
            agg = ws.groupby(COL_DATE).agg({prod_col: "sum", water_col: "sum"}).sort_index()
            qo_m = agg[prod_col].values.astype(float)
            qw_m = agg[water_col].values.astype(float)
            qo_cum = np.cumsum(qo_m)
            wor = np.divide(qw_m, qo_m, out=np.full_like(qo_m, np.nan), where=qo_m > 0)
            # For the first group, plot everything; for subsequent groups,
            # start from 01.01 of the vintage year (where new wells join)
            if ki == 0:
                start_idx = 0
            else:
                start_date = pd.Timestamp(f"{yr}-01-01")
                dates_arr = agg.index
                later = np.where(dates_arr >= start_date)[0]
                start_idx = int(later[0]) if len(later) > 0 else len(qo_m)
            qo_plot = qo_cum[start_idx:]
            wor_plot = wor[start_idx:]
            qo_m_plot = qo_m[start_idx:]
            mask = qo_m_plot > 0
            if not np.any(mask):
                continue
            color = _COLORS[ki % len(_COLORS)]
            ax.plot(qo_plot[mask], wor_plot[mask], color=color, linewidth=1.0,
                    label=f"≤{yr}", alpha=0.85)

        ax.set_xlabel("Нак. нефть, т")
        ax.set_ylabel("ВНФ")
        n_wells = len(vintage)
        ax.set_title(f"Qo vs ВНФ по годам ввода ({n_wells} скв., {len(years_sorted)} групп)")
        ax.set_yscale("log")

        from src.ui.legend_helper import fit_legend
        fit_legend(ax, loc="upper left")

    # ── Export ─────────────────────────────────────────────────────────────

    def _copy_data(self) -> None:
        """Copy tab-separated table: Date | Year1 | Year2 | … | Total."""
        from PySide6.QtWidgets import QApplication

        selected = [item.text() for item in self._lst.selectedItems()]
        result = self._build_vintage_data(selected)
        if result is None:
            return

        dates, arrays = result
        years = sorted(arrays.keys())

        header = ["Дата"] + [str(yr) for yr in years] + ["Итого"]
        rows = ["\t".join(header)]
        for i, dt in enumerate(dates):
            vals = [arrays[yr][i] for yr in years]
            total = sum(vals)
            row = [dt.strftime("%Y-%m-%d")]
            row += [f"{v:.4g}" for v in vals]
            row.append(f"{total:.4g}")
            rows.append("\t".join(row))

        QApplication.instance().clipboard().setText("\n".join(rows))

    def _copy_well_list(self) -> None:
        """Copy well-name / vintage-year table to clipboard."""
        from PySide6.QtWidgets import QApplication

        selected = [item.text() for item in self._lst.selectedItems()]
        rows: list[str] = ["\u0421\u043a\u0432\u0430\u0436\u0438\u043d\u0430\t\u0413\u043e\u0434 \u0432\u0432\u043e\u0434\u0430"]
        for well in sorted(selected):
            yr = self._vintage_cache.get(well)
            if yr is not None:
                rows.append(f"{well}\t{yr}")
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
