"""Object info card — summary dashboard with plots and KPI metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter as _FuncFormatter
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.data.models import (
    COL_DATE, COL_GAS, COL_LIQUID, COL_OIL, COL_WATER,
    COL_WATER_CUT, COL_WELL, COL_WORK_TYPE, WORK_TYPE_INJ, WORK_TYPE_OIL,
)


# ── Metric card widget ────────────────────────────────────────────────────────

class _MetricCard(QFrame):
    """A small card showing a label and a value."""

    def __init__(self, label: str, value: str = "—", parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #f5f5f5; border: 1px solid #ccc; border-radius: 4px; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        self._lbl = QLabel(label)
        self._lbl.setStyleSheet("font-size: 9px; color: #666;")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val = QLabel(value)
        self._val.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._lbl)
        lay.addWidget(self._val)

    def set_value(self, v: str) -> None:
        self._val.setText(v)


# ── Main dialog ──────────────────────────────────────────────────────────────

class ObjectInfoDialog(QDialog):
    """Dashboard: production plots + 12 KPI metric cards."""

    def __init__(
        self,
        df: pd.DataFrame,
        selected_wells: list[str],
        project_name: str = "",
        stoiip: float = 0.0,
        eur: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        title = "Карточка объекта"
        if project_name:
            title += f" — {project_name}"
        self.setWindowTitle(title)
        self.resize(1100, 750)

        self._df = df
        self._wells = selected_wells
        self._stoiip = stoiip
        self._eur = eur

        # Stored after _populate for clipboard export
        self._plot_dates: list = []
        self._plot_qo: np.ndarray = np.array([])
        self._plot_ql: np.ndarray = np.array([])
        self._plot_qi: np.ndarray = np.array([])
        self._plot_fw: np.ndarray = np.array([])
        self._plot_comp: np.ndarray = np.array([])

        self._build_ui()
        self._populate()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter)

        # ── Top: two plots side by side + copy button ─────────────────
        top_w = QWidget()
        top_lay = QVBoxLayout(top_w)
        top_lay.setContentsMargins(0, 0, 0, 0)

        plots_row = QHBoxLayout()
        self._fig1 = Figure(tight_layout=True)
        self._canvas1 = FigureCanvasQTAgg(self._fig1)
        plots_row.addWidget(self._canvas1)

        self._fig2 = Figure(tight_layout=True)
        self._canvas2 = FigureCanvasQTAgg(self._fig2)
        plots_row.addWidget(self._canvas2)
        top_lay.addLayout(plots_row)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("Скопировать данные")
        btn_copy.clicked.connect(self._copy_data)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        top_lay.addLayout(btn_row)

        splitter.addWidget(top_w)

        # ── Bottom: metric cards 4×3 ──────────────────────────────────────────
        cards_w = QWidget()
        cards_lay = QGridLayout(cards_w)
        cards_lay.setSpacing(8)
        cards_lay.setContentsMargins(8, 8, 8, 8)

        labels = [
            "Нак. нефть, тыс.т",       "Нак. жидкость, тыс.т",   "Нак. закачка, тыс.т",
            "КИН (RF)",                 "ГФ (GOR)",                "Комп. нак.",
            "Выработка ОИЗ",            "Обводнённость",           "Комп. тек.",
            "Акт. доб. скв.",           "Всего акт. скв.",         "Акт. наг. скв.",
        ]
        self._cards: list[_MetricCard] = []
        for i, lbl in enumerate(labels):
            card = _MetricCard(lbl)
            r, c = divmod(i, 3)
            cards_lay.addWidget(card, r, c)
            self._cards.append(card)

        splitter.addWidget(cards_w)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    # ── Data + drawing ───────────────────────────────────────────────────

    def _populate(self) -> None:
        df = self._df
        wells = self._wells
        if df is None or not wells:
            return

        # ── Production sub-frame (oil rows) ──────────────────────────────
        sub = df[df[COL_WELL].isin(wells)].copy()
        sub_oil = (
            sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
            if COL_WORK_TYPE in sub.columns else sub
        )

        if COL_DATE not in sub_oil.columns or COL_OIL not in sub_oil.columns:
            return

        agg_cols = {COL_OIL: "sum"}
        if COL_WATER in sub_oil.columns:
            agg_cols[COL_WATER] = "sum"
        if COL_LIQUID in sub_oil.columns:
            agg_cols[COL_LIQUID] = "sum"
        if COL_GAS in sub_oil.columns:
            agg_cols[COL_GAS] = "sum"
        if COL_WELL in sub_oil.columns:
            agg_cols[COL_WELL] = "nunique"

        prod = sub_oil.groupby(COL_DATE).agg(agg_cols).sort_index()
        dates = prod.index
        qo = prod[COL_OIL].values.astype(float)
        qw = prod[COL_WATER].values.astype(float) if COL_WATER in prod.columns else np.zeros_like(qo)
        ql = prod[COL_LIQUID].values.astype(float) if COL_LIQUID in prod.columns else qo + qw
        qg = prod[COL_GAS].values.astype(float) if COL_GAS in prod.columns else np.zeros_like(qo)
        n_prod_wells = prod[COL_WELL].values.astype(int) if COL_WELL in prod.columns else np.zeros(len(qo), dtype=int)

        Qo = np.cumsum(qo)
        Ql = np.cumsum(ql)

        # ── Per-date active producing wells (non-zero oil) ──────────────
        last_date = dates[-1]
        n_act_prod = np.zeros(len(dates), dtype=float)
        if COL_WELL in sub_oil.columns:
            for di, dt in enumerate(dates):
                dt_sub = sub_oil[sub_oil[COL_DATE] == dt]
                if len(dt_sub) > 0:
                    wo = dt_sub.groupby(COL_WELL)[COL_OIL].sum()
                    n_act_prod[di] = float((wo > 0).sum())
        n_prod_last = int(n_act_prod[-1])

        # ── Injection sub-frame ──────────────────────────────────────────
        qi = np.zeros_like(qo)
        Qi = np.zeros_like(qo)
        n_act_inj = np.zeros(len(dates), dtype=float)
        n_inj_wells = 0
        if COL_WORK_TYPE in sub.columns and COL_WATER in sub.columns:
            inj = sub[sub[COL_WORK_TYPE] == WORK_TYPE_INJ]
            if len(inj) > 0:
                inj_monthly = inj.groupby(COL_DATE)[COL_WATER].sum().sort_index()
                qi_aligned = inj_monthly.reindex(dates, fill_value=0.0).values.astype(float)
                qi = qi_aligned
                Qi = np.cumsum(qi)
                # Per-date active injection wells
                if COL_WELL in inj.columns:
                    for di, dt in enumerate(dates):
                        dt_inj = inj[inj[COL_DATE] == dt]
                        if len(dt_inj) > 0:
                            wi = dt_inj.groupby(COL_WELL)[COL_WATER].sum()
                            n_act_inj[di] = float((wi > 0).sum())
                n_inj_wells = int(n_act_inj[-1])

        # ── Watercut & compensation ──────────────────────────────────────
        fw = np.divide(qw, ql, out=np.zeros_like(ql), where=ql > 0)
        comp_cur = np.divide(qi, ql, out=np.zeros_like(ql), where=ql > 0)
        comp_tot = np.divide(Qi, Ql, out=np.zeros_like(Ql), where=Ql > 0)

        # Total active wells per date
        n_act_total = n_act_prod + n_act_inj

        # Store for clipboard export
        self._plot_dates = [str(d)[:10] for d in dates]
        self._plot_qo = qo
        self._plot_ql = ql
        self._plot_qi = qi
        self._plot_fw = fw
        self._plot_comp = comp_cur
        self._plot_n_prod = n_act_prod
        self._plot_n_inj = n_act_inj

        # ── Date formatter ───────────────────────────────────────────────
        date_vals = pd.to_datetime(dates)
        x = np.arange(len(dates))

        def _fmt(val, _pos):
            idx = int(round(val))
            if 0 <= idx < len(date_vals):
                return date_vals[idx].strftime("%m.%Y")
            return ""

        # ── Plot 1: production + injection + active wells (Y2) ─────────
        ax1 = self._fig1.add_subplot(111)
        ax1.plot(x, qo, color="red", linewidth=1.0, label="Нефть, т/мес")
        ax1.plot(x, ql, color="green", linewidth=0.8, alpha=0.7, label="Жидкость, т/мес")
        ax1.plot(x, qi, color="purple", linewidth=0.8, alpha=0.7, label="Закачка, т/мес")
        ax1.set_ylabel("т/мес")
        ax1.set_title("Добыча и закачка")
        ax1.xaxis.set_major_formatter(_FuncFormatter(_fmt))
        # Active wells on secondary Y-axis
        ax1b = ax1.twinx()
        ax1b.step(x, n_act_prod, where="mid", color="steelblue", linewidth=0.9,
                  linestyle="--", alpha=0.6, label="Доб. скв.")
        ax1b.step(x, n_act_inj, where="mid", color="darkorange", linewidth=0.9,
                  linestyle=":", alpha=0.6, label="Наг. скв.")
        ax1b.set_ylabel("Скважины", fontsize=8)
        ax1b.set_ylim(bottom=0)
        from matplotlib.ticker import MaxNLocator as _MNL
        ax1b.yaxis.set_major_locator(_MNL(integer=True, nbins=5))
        ax1b.tick_params(axis="y", labelsize=7)
        self._fig1.autofmt_xdate(rotation=45, ha="right")
        ax1.grid(True, alpha=0.3)
        # Combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1b.get_legend_handles_labels()
        ax1b.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
        self._canvas1.draw_idle()

        # ── Plot 2: watercut + compensation ──────────────────────────────
        ax2 = self._fig2.add_subplot(111)
        ax2.plot(x, fw, color="blue", linewidth=1.0, label="Обводнённость")
        ax2.plot(x, comp_cur, color="purple", linewidth=0.8, alpha=0.7, label="Комп. тек.")
        ax2.set_ylabel("доли ед.")
        ax2.set_ylim(-0.05, 1.5)
        ax2.set_title("Обводнённость и компенсация")
        ax2.xaxis.set_major_formatter(_FuncFormatter(_fmt))
        self._fig2.autofmt_xdate(rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7, loc="best")
        self._canvas2.draw_idle()

        # ── Metric cards ─────────────────────────────────────────────────
        total_oil = float(Qo[-1]) if len(Qo) else 0.0
        total_liq = float(Ql[-1]) if len(Ql) else 0.0
        total_inj = float(Qi[-1]) if len(Qi) else 0.0

        # RF
        rf_val = total_oil / self._stoiip if self._stoiip > 0 else 0.0
        # GOR (last month)
        gor_val = float(qg[-1] / qo[-1]) if len(qo) > 0 and qo[-1] > 0 else 0.0
        # Total compensation
        comp_tot_val = total_inj / total_liq if total_liq > 0 else 0.0
        # Depleted PDP reserves
        depl_val = total_oil / self._eur if self._eur > 0 else 0.0
        # Current watercut (last month)
        fw_last = float(fw[-1]) if len(fw) else 0.0
        # Current compensation (last month)
        comp_cur_last = float(comp_cur[-1]) if len(comp_cur) else 0.0
        # Total active (producing + injection)
        n_total = n_prod_last + n_inj_wells

        values = [
            f"{total_oil / 1000:,.1f}",
            f"{total_liq / 1000:,.1f}",
            f"{total_inj / 1000:,.1f}",
            f"{rf_val:.4f}" if self._stoiip > 0 else "—",
            f"{gor_val:,.0f}" if gor_val > 0 else "—",
            f"{comp_tot_val:.3f}",
            f"{depl_val:.1%}" if self._eur > 0 else "—",
            f"{fw_last:.3f}",
            f"{comp_cur_last:.3f}",
            str(n_prod_last),
            str(n_total),
            str(n_inj_wells),
        ]
        for card, val in zip(self._cards, values):
            card.set_value(val)

    # ── Clipboard ─────────────────────────────────────────────────────────

    def _copy_data(self) -> None:
        """Copy monthly time-series to clipboard as TSV."""
        from PySide6.QtWidgets import QApplication
        if len(self._plot_dates) == 0:
            return
        hdr = ["Дата", "Нефть, т/мес", "Жидкость, т/мес", "Закачка, т/мес",
               "Обводнённость", "Комп. тек.", "Доб. скв.", "Наг. скв."]
        rows = ["\t".join(hdr)]
        for i, dt in enumerate(self._plot_dates):
            row = [
                dt,
                f"{self._plot_qo[i]:.4g}",
                f"{self._plot_ql[i]:.4g}",
                f"{self._plot_qi[i]:.4g}",
                f"{self._plot_fw[i]:.4f}",
                f"{self._plot_comp[i]:.4f}",
                f"{int(self._plot_n_prod[i])}",
                f"{int(self._plot_n_inj[i])}",
            ]
            rows.append("\t".join(row))
        QApplication.instance().clipboard().setText("\n".join(rows))
