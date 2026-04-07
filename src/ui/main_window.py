"""Main application window — orchestrates all panels and forecasting logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from src.data.loader import load_file
from src.data.models import (
    COL_CUM_LIQUID,
    COL_CUM_OIL,
    COL_DATE,
    COL_LIQUID,
    COL_OIL,
    COL_WATER_CUT,
    COL_WELL,
    COL_WORK_TYPE,
    WORK_TYPE_OIL,
)
from src.data.validation import validate
from src.export.exporter import export_forecast_csv, export_plot
from src.forecasting.base import ForecastMethod
from src.ui.data_panel import DataPanel
from src.ui.method_panel import MethodPanel
from src.ui.plot_widget import PlotWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Displacement Forecaster")
        self.resize(1280, 720)

        # ── State ────────────────────────────────────────────────────────────
        self.df: pd.DataFrame | None = None
        self._selected_wells: list[str] = []
        self._span_range: tuple[float, float] | None = None
        self._current_method: ForecastMethod | None = None

        # ── Widgets ──────────────────────────────────────────────────────────
        self.data_panel = DataPanel()
        self.plot_widget = PlotWidget()
        self.method_panel = MethodPanel()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.data_panel)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.method_panel)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # ── Menu ─────────────────────────────────────────────────────────────
        menu = self.menuBar()
        file_menu = menu.addMenu("Файл")
        file_menu.addAction("Открыть…", self._on_load)
        file_menu.addAction("Экспорт графика…", self._on_export_plot)
        file_menu.addAction("Экспорт прогноза…", self._on_export_forecast)
        file_menu.addSeparator()
        file_menu.addAction("Выход", self.close)

        # ── Connections ──────────────────────────────────────────────────────
        self.data_panel.btn_load.clicked.connect(self._on_load)
        self.data_panel.wells_changed.connect(self._on_wells_changed)
        self.method_panel.fit_requested.connect(self._on_fit)
        self.method_panel.forecast_requested.connect(self._on_forecast)
        self.method_panel.cmb_family.currentIndexChanged.connect(self._on_plot_data)
        self.method_panel.cmb_method.currentIndexChanged.connect(self._on_plot_data)

    # ── Load ─────────────────────────────────────────────────────────────────

    def _on_load(self) -> None:
        path = self.data_panel.get_file_path()
        if not path:
            return
        try:
            self.df = load_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        vr = validate(self.df)
        if not vr.is_valid:
            QMessageBox.critical(self, "Ошибка данных", "\n".join(vr.errors))
            return
        if vr.warnings:
            self.status.showMessage("; ".join(vr.warnings), 8000)

        wells = sorted(self.df[COL_WELL].unique().tolist())
        self.data_panel.populate_wells(wells)
        n = len(self.df)
        self.data_panel.lbl_info.setText(
            f"Загружено {n} строк, {len(wells)} скважин"
        )
        self.status.showMessage("Данные загружены", 3000)

    # ── Well selection → plot ────────────────────────────────────────────────

    def _on_wells_changed(self, wells: list[str]) -> None:
        self._selected_wells = wells
        self._span_range = None
        self._on_plot_data()

    def _on_plot_data(self) -> None:
        """Plot historical data for selected wells and method family."""
        if self.df is None or not self._selected_wells:
            self.plot_widget.clear()
            return

        sub = self.df[self.df[COL_WELL].isin(self._selected_wells)].copy()
        # Filter to oil-producing rows only
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]

        family = self.method_panel.get_family_name()
        self.plot_widget.clear()
        ax = self.plot_widget.ax

        if family == "Характеристики вытеснения":
            x, y = self._agg_cumulative(sub, COL_CUM_LIQUID), self._agg_cumulative(sub, COL_CUM_OIL)
            ax.scatter(x, y, s=4, label="Qo vs Ql (факт)")
            ax.set_xlabel("Накопленная жидкость, т")
            ax.set_ylabel("Накопленная нефть, т")

        elif family == "Кривые падения добычи (DCA)":
            ts = self._monthly_series(sub, COL_OIL)
            if ts is not None:
                ax.plot(range(len(ts)), ts.values, "o-", ms=3, label="Добыча нефти (факт)")
                ax.set_xlabel("Месяц")
                ax.set_ylabel("Добыча нефти, т/мес")

        elif family == "Фракционный поток":
            x = self._agg_cumulative(sub, COL_CUM_OIL)
            y = self._agg_watercut(sub)
            if x is not None and y is not None:
                ax.scatter(x, y, s=4, label="fw vs Qo (факт)")
                ax.set_xlabel("Накопленная нефть, т")
                ax.set_ylabel("Обводнённость")

        ax.set_title(", ".join(self._selected_wells[:5]))
        self.plot_widget.enable_span_selector(self._on_span_select)
        self.plot_widget.redraw()

    # ── Span selector callback ───────────────────────────────────────────────

    def _on_span_select(self, xmin: float, xmax: float) -> None:
        self._span_range = (xmin, xmax)
        self.status.showMessage(
            f"Выбран диапазон: {xmin:.1f} – {xmax:.1f}", 5000
        )

    # ── Fit trend ────────────────────────────────────────────────────────────

    def _on_fit(self) -> None:
        if self.df is None or not self._selected_wells:
            return

        method_cls = self.method_panel.get_method_class()
        if method_cls is None:
            return

        x, y = self._get_xy()
        if x is None or len(x) < 3:
            self.status.showMessage("Недостаточно данных для аппроксимации", 4000)
            return

        # Apply span selection
        if self._span_range:
            mask = (x >= self._span_range[0]) & (x <= self._span_range[1])
            x_sel, y_sel = x[mask], y[mask]
        else:
            x_sel, y_sel = x, y

        if len(x_sel) < 2:
            self.status.showMessage("Слишком мало точек в выбранном диапазоне", 4000)
            return

        method: ForecastMethod = method_cls()
        method.fit(x_sel, y_sel)
        self._current_method = method

        # Overlay fit line
        ax = self.plot_widget.ax
        x_line = np.linspace(x.min(), x.max(), 300)
        y_line = method.predict(x_line)
        ax.plot(x_line, y_line, "r-", lw=2, label=f"Тренд: {method.get_name()}")
        self.plot_widget.redraw()

        r2 = method.r_squared(x_sel, y_sel)
        params = method.get_parameters()
        lines = [f"Метод: {method.get_name()}", f"R² = {r2:.4f}"]
        for k, v in params.items():
            lines.append(f"  {k} = {v}")
        self.method_panel.show_result("\n".join(lines))
        self.status.showMessage(f"Тренд построен, R² = {r2:.4f}", 5000)

    # ── Forecast ─────────────────────────────────────────────────────────────

    def _on_forecast(self) -> None:
        if self._current_method is None:
            self.status.showMessage("Сначала постройте тренд", 4000)
            return

        x, y = self._get_xy()
        if x is None:
            return

        horizon = self.method_panel.get_horizon()
        family = self.method_panel.get_family_name()

        if family == "Кривые падения добычи (DCA)":
            x_fc = np.arange(len(x), len(x) + horizon, dtype=float)
        else:
            # Extrapolate x by extending linearly
            dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
            x_fc = np.linspace(x[-1], x[-1] + dx * horizon, horizon)

        y_fc = self._current_method.predict(x_fc)
        ax = self.plot_widget.ax
        ax.plot(x_fc, y_fc, "g--", lw=2, label="Прогноз")
        self.plot_widget.redraw()
        self.status.showMessage(f"Прогноз рассчитан на {horizon} мес.", 5000)

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export_plot(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "", "PNG (*.png);;SVG (*.svg)"
        )
        if path:
            export_plot(self.plot_widget.figure, path)
            self.status.showMessage(f"График сохранён: {path}", 3000)

    def _on_export_forecast(self) -> None:
        if self._current_method is None:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить прогноз", "", "CSV (*.csv);;Excel (*.xlsx)"
        )
        if not path:
            return
        x, y = self._get_xy()
        if x is None:
            return
        horizon = self.method_panel.get_horizon()
        family = self.method_panel.get_family_name()
        if family == "Кривые падения добычи (DCA)":
            x_fc = np.arange(len(x), len(x) + horizon, dtype=float)
        else:
            dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
            x_fc = np.linspace(x[-1], x[-1] + dx * horizon, horizon)
        y_fc = self._current_method.predict(x_fc)
        export_forecast_csv(x_fc, y_fc, self._current_method.get_name(), path)
        self.status.showMessage(f"Прогноз экспортирован: {path}", 3000)

    # ── Data helpers ─────────────────────────────────────────────────────────

    def _get_xy(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (x, y) arrays for the current family + wells."""
        if self.df is None or not self._selected_wells:
            return None, None

        sub = self.df[self.df[COL_WELL].isin(self._selected_wells)].copy()
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]

        family = self.method_panel.get_family_name()

        if family == "Характеристики вытеснения":
            x = self._agg_cumulative(sub, COL_CUM_LIQUID)
            y = self._agg_cumulative(sub, COL_CUM_OIL)

        elif family == "Кривые падения добычи (DCA)":
            ts = self._monthly_series(sub, COL_OIL)
            if ts is None:
                return None, None
            x = np.arange(len(ts), dtype=float)
            y = ts.values.astype(float)

        elif family == "Фракционный поток":
            x = self._agg_cumulative(sub, COL_CUM_OIL)
            y = self._agg_watercut(sub)

        else:
            return None, None

        if x is None or y is None or len(x) == 0:
            return None, None
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    @staticmethod
    def _agg_cumulative(sub: pd.DataFrame, col: str) -> np.ndarray | None:
        """Aggregate cumulative column across selected wells by date."""
        if col not in sub.columns or COL_DATE not in sub.columns:
            return None
        agg = sub.groupby(COL_DATE)[col].sum().sort_index()
        return agg.values.astype(float)

    @staticmethod
    def _monthly_series(sub: pd.DataFrame, col: str) -> pd.Series | None:
        if col not in sub.columns or COL_DATE not in sub.columns:
            return None
        agg = sub.groupby(COL_DATE)[col].sum().sort_index()
        agg = agg[agg > 0]
        return agg if len(agg) > 0 else None

    @staticmethod
    def _agg_watercut(sub: pd.DataFrame) -> np.ndarray | None:
        if COL_WATER_CUT not in sub.columns or COL_DATE not in sub.columns:
            return None
        # Weighted average water-cut by liquid volume
        if COL_LIQUID not in sub.columns:
            return None
        from src.data.models import COL_LIQUID

        grp = sub.groupby(COL_DATE).apply(
            lambda g: (g[COL_WATER_CUT] * g[COL_LIQUID]).sum()
            / g[COL_LIQUID].sum()
            if g[COL_LIQUID].sum() > 0
            else 0.0
        )
        return grp.sort_index().values.astype(float)
