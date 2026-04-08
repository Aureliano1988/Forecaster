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
    COL_WATER,
    COL_WATER_CUT,
    COL_WELL,
    COL_WORK_TYPE,
    ForecastSeries,
    SavedMethodResult,
    WORK_TYPE_OIL,
)
from src.data.validation import validate
from src.export.exporter import export_forecast_csv, export_plot
from src.forecasting.base import ForecastMethod
from src.forecasting.displacement import LinearDisplacement
from src.forecasting.monthly import (
    build_displacement_forecast,
    build_dca_forecast,
    build_fractional_forecast,
)
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
        self._excluded_ranges: list[tuple[float, float]] = []
        self._eraser_active: bool = False
        self._fit_result_text: str = ""           # text produced by the last fit
        self._source_file: str = ""               # path of last loaded file
        self._saved_results: dict[str, SavedMethodResult] = {}  # keyed by family|method

        # Plot overlay references (for in-place update)
        self._trend_line = None
        self._forecast_line = None
        self._excl_patches: list = []

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
        self.method_panel.build_requested.connect(self._on_build_forecast)
        self.method_panel.discard_requested.connect(self._on_discard)
        self.method_panel.eraser_toggled.connect(self._on_eraser_toggle)
        self.method_panel.cmb_family.currentIndexChanged.connect(self._on_plot_data)
        self.method_panel.cmb_method.currentIndexChanged.connect(self._on_plot_data)
        self.method_panel.save_requested.connect(self._on_save)

    # ── Load ─────────────────────────────────────────────────────────────────

    def _on_load(self) -> None:
        path = self.data_panel.get_file_path()
        if not path:
            return
        self._source_file = path
        self._saved_results.clear()
        self._fit_result_text = ""
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
        self._saved_results.clear()   # underlying data changed
        self._fit_result_text = ""
        self._on_plot_data()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_sub(self) -> pd.DataFrame | None:
        """Return filtered sub-frame for selected wells (oil rows only)."""
        if self.df is None or not self._selected_wells:
            return None
        sub = self.df[self.df[COL_WELL].isin(self._selected_wells)].copy()
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        return sub

    def _result_key(self) -> str:
        """Unique key for the currently-selected method."""
        return (
            f"{self.method_panel.get_family_name()}"
            f"|{self.method_panel.cmb_method.currentText()}"
        )

    # ── Plot data ────────────────────────────────────────────────────────────

    def _on_plot_data(self) -> None:
        """Plot historical data for selected wells and method family."""
        if self.df is None or not self._selected_wells:
            self.plot_widget.clear()
            return

        sub = self._get_sub()
        if sub is None:
            self.plot_widget.clear()
            return

        family = self.method_panel.get_family_name()
        self.plot_widget.clear()
        # ax.clear() invalidates any previously added artists — drop stale refs
        self._trend_line = None
        self._forecast_line = None
        ax = self.plot_widget.ax

        if family == "Характеристики вытеснения":
            method_cls = self.method_panel.get_method_class()
            prod = self._get_displacement_data(sub)
            if method_cls is not None and prod is not None:
                x, y = method_cls.prepare_xy(*prod)
                ax.scatter(x, y, s=4, label="Факт")
                ax.set_xlabel(method_cls.x_label)
                ax.set_ylabel(method_cls.y_label)

        elif family == "Кривые падения добычи (DCA)":
            ts = self._monthly_series(sub, COL_OIL)
            if ts is not None:
                ax.plot(range(len(ts)), ts.values, "o-", ms=3, label="Добыча нефти (факт)")
                ax.set_xlabel("Месяц")
                ax.set_ylabel("Добыча нефти, т/мес")
                ax.set_yscale("log")

        elif family == "Фракционный поток":
            x = self._agg_cumulative(sub, COL_CUM_OIL)
            y = self._agg_watercut(sub)
            if x is not None and y is not None:
                ax.scatter(x, y, s=4, label="fw vs Qo (факт)")
                ax.set_xlabel("Накопленная нефть, т")
                ax.set_ylabel("Обводнённость")

        ax.set_title(", ".join(self._selected_wells[:5]))

        # ── Restore saved overlay for this technique, if available ───────────────
        key = self._result_key()
        if key in self._saved_results:
            saved = self._saved_results[key]
            x_tr = np.array(saved.x_trend, dtype=float)
            y_tr = np.array(saved.y_trend, dtype=float)
            if len(x_tr):
                self._trend_line, = ax.plot(
                    x_tr, y_tr, "r-", lw=2, label=f"Тренд: {saved.method_name}"
                )
            x_fc = np.array(saved.x_forecast, dtype=float)
            y_fc = np.array(saved.y_forecast, dtype=float)
            if len(x_fc):
                self._forecast_line, = ax.plot(
                    x_fc, y_fc, "g--", lw=2, label="Прогноз"
                )
            self.method_panel.show_result(saved.params_text)
        else:
            self.method_panel.show_result("")

        self.plot_widget.enable_span_selector(self._on_span_select)
        self.plot_widget.redraw()

    # ── Span selector callback ───────────────────────────────────────────────

    def _on_span_select(self, xmin: float, xmax: float) -> None:
        if self._eraser_active:
            self._excluded_ranges.append((xmin, xmax))
            self._draw_exclusion_patches()
            self.status.showMessage(
                f"Исключён диапазон: {xmin:.1f} – {xmax:.1f}  "
                f"(всего исключений: {len(self._excluded_ranges)})", 5000
            )
        else:
            self._span_range = (xmin, xmax)
            self.status.showMessage(
                f"Выбран диапазон: {xmin:.1f} – {xmax:.1f}", 5000
            )

    # ── Eraser toggle ─────────────────────────────────────────────────────────

    def _on_eraser_toggle(self, active: bool) -> None:
        self._eraser_active = active
        if active:
            self.status.showMessage(
                "Ластик: выделите диапазон на графике для исключения", 5000
            )
        else:
            self.status.showMessage("Ластик выключен", 2000)

    def _draw_exclusion_patches(self) -> None:
        """Draw red shading for all excluded ranges."""
        # Remove old patches
        for p in self._excl_patches:
            p.remove()
        self._excl_patches.clear()

        ax = self.plot_widget.ax
        for xmin, xmax in self._excluded_ranges:
            patch = ax.axvspan(xmin, xmax, alpha=0.15, color="red")
            self._excl_patches.append(patch)
        self.plot_widget.canvas.draw_idle()

    def _apply_exclusions(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove points falling inside any excluded range."""
        if not self._excluded_ranges:
            return x, y
        mask = np.ones(len(x), dtype=bool)
        for xmin, xmax in self._excluded_ranges:
            mask &= ~((x >= xmin) & (x <= xmax))
        return x[mask], y[mask]
    def _on_build_forecast(self) -> None:
        """Fit trend on the selected interval and immediately build forecast."""
        if not self._on_fit():
            return
        self._on_forecast()

    # ── Fit trend ────────────────────────────────────────────────────────────
    def _on_fit(self) -> bool:
        if self.df is None or not self._selected_wells:
            return False

        method_cls = self.method_panel.get_method_class()
        if method_cls is None:
            return False

        x, y = self._get_xy()
        if x is None or len(x) < 3:
            self.status.showMessage("Недостаточно данных для аппроксимации", 4000)
            return False

        # Apply span selection
        if self._span_range:
            mask = (x >= self._span_range[0]) & (x <= self._span_range[1])
            x_sel, y_sel = x[mask], y[mask]
        else:
            x_sel, y_sel = x, y

        # Apply exclusion zones
        x_sel, y_sel = self._apply_exclusions(x_sel, y_sel)

        if len(x_sel) < 2:
            self.status.showMessage("Слишком мало точек в выбранном диапазоне", 4000)
            return False

        method: ForecastMethod = method_cls()
        method.fit(x_sel, y_sel)
        self._current_method = method

        # Invalidate any previously saved forecast for this technique
        key = self._result_key()
        self._saved_results.pop(key, None)

        # Remove old trend / forecast lines before drawing
        self._remove_overlay_lines()

        # Overlay fit line only within the selected range
        ax = self.plot_widget.ax
        x_line = np.linspace(x_sel.min(), x_sel.max(), 300)
        y_line = method.predict(x_line)
        self._trend_line, = ax.plot(
            x_line, y_line, "r-", lw=2, label=f"Тренд: {method.get_name()}"
        )
        self.plot_widget.redraw()

        r2 = method.r_squared(x_sel, y_sel)
        params = method.get_parameters()
        lines = [f"Метод: {method.get_name()}", f"R² = {r2:.4f}"]
        for k, v in params.items():
            lines.append(f"  {k} = {float(v):.6g}")
        self._fit_result_text = "\n".join(lines)
        self.method_panel.show_result(self._fit_result_text)
        self.status.showMessage(f"Тренд построен, R² = {r2:.4f}", 5000)
        return True

    # ── Remove / discard helpers ──────────────────────────────────────────────

    def _remove_overlay_lines(self) -> None:
        """Remove existing trend and forecast lines from the axes."""
        if self._trend_line is not None:
            self._trend_line.remove()
            self._trend_line = None
        if self._forecast_line is not None:
            self._forecast_line.remove()
            self._forecast_line = None

    def _on_discard(self) -> None:
        key = self._result_key()
        self._saved_results.pop(key, None)
        self._current_method = None
        self._span_range = None
        self._excluded_ranges.clear()
        self._fit_result_text = ""
        self.method_panel.show_result("")
        self.method_panel.btn_eraser.setChecked(False)
        self._on_plot_data()
        self.status.showMessage("Тренд сброшен", 3000)

    # ── Forecast ─────────────────────────────────────────────────────────────

    def _on_forecast(self) -> None:
        if self._current_method is None:
            self.status.showMessage("Сначала постройте тренд", 4000)
            return

        x, y = self._get_xy()
        if x is None:
            return

        horizon    = self.method_panel.get_horizon()
        wor_limit  = self.method_panel.get_wor_limit()
        family     = self.method_panel.get_family_name()
        x_last     = float(x[-1])

        # ── Physical last values for monthly forecast builders ─────────────────
        Qo_last = Ql_last = Qw_last = ql_last = 0.0
        sub = self._get_sub()
        if sub is not None:
            prod = self._get_displacement_data(sub)
            if prod is not None:
                Qo_a, Ql_a, Qw_a, _qo_a, ql_a, _qw_a = prod
                if len(Qo_a):
                    Qo_last = float(Qo_a[-1])
                    Ql_last = float(Ql_a[-1])
                    Qw_last = float(Qw_a[-1])
                nz = ql_a[ql_a > 0]
                ql_last = float(nz[-1]) if len(nz) else float(ql_a[-1]) if len(ql_a) else 1.0

        # ── Build monthly series ──────────────────────────────────────────
        monthly: ForecastSeries | None = None
        try:
            if family == "Характеристики вытеснения" and isinstance(
                self._current_method, LinearDisplacement
            ):
                monthly = build_displacement_forecast(
                    self._current_method,
                    Qo_last, Ql_last, Qw_last, ql_last, horizon, wor_limit,
                )
            elif family == "Кривые падения добычи (DCA)":
                monthly = build_dca_forecast(
                    self._current_method, x_last, ql_last, horizon, wor_limit,
                )
            elif family == "Фракционный поток":
                monthly = build_fractional_forecast(
                    self._current_method, Qo_last, ql_last, horizon, wor_limit,
                )
        except Exception as exc:
            self.status.showMessage(f"Ошибка расчёта прогноза: {exc}", 5000)

        actual_duration = monthly.duration if monthly else horizon

        # ── Forecast line in method-space (visual only) ───────────────────────
        n = max(actual_duration, 1)
        if family == "Кривые падения добычи (DCA)":
            x_fc = np.arange(x_last + 1, x_last + 1 + n, dtype=float)
        else:
            dx = (x[-1] - x[0]) / max(len(x) - 1, 1) if len(x) > 1 else 1.0
            x_fc = np.linspace(x_last, x_last + dx * n, n)
        y_fc = self._current_method.predict(x_fc) if len(x_fc) else np.array([])

        # Remove old forecast line before drawing
        if self._forecast_line is not None:
            self._forecast_line.remove()
            self._forecast_line = None

        ax = self.plot_widget.ax
        if len(x_fc):
            self._forecast_line, = ax.plot(x_fc, y_fc, "g--", lw=2, label="Прогноз")
        self.plot_widget.redraw()

        # ── Results text ─────────────────────────────────────────────────
        result_text = self._fit_result_text
        if monthly and monthly.duration > 0:
            stopped_by = (
                "WOR"
                if monthly.WOR and monthly.WOR[-1] >= wor_limit
                else "горизонт"
            )
            result_text += (
                f"\n{'\u2500'*22}\n"
                f"Прогноз (стоп: {stopped_by}):\n"
                f"  Горизонт: {monthly.duration} мес.\n"
                f"  Ост. запасы: {monthly.remain_reserves:,.0f} т\n"
                f"  WOR (посл.): {monthly.wor_last:.2f}"
            )
        self.method_panel.show_result(result_text)

        msg = f"Прогноз: {actual_duration} мес."
        if monthly:
            msg += f", ост. запасы {monthly.remain_reserves:,.0f} т"
        self.status.showMessage(msg, 6000)

        # ── Save state for this technique ─────────────────────────────────
        key = self._result_key()
        self._saved_results[key] = SavedMethodResult(
            method_name=self._current_method.get_name(),
            params_text=result_text,
            parameters={k: float(v) for k, v in self._current_method.get_parameters().items()},
            x_trend=(
                self._trend_line.get_xdata().tolist() if self._trend_line is not None else []
            ),
            y_trend=(
                self._trend_line.get_ydata().tolist() if self._trend_line is not None else []
            ),
            x_forecast=x_fc.tolist(),
            y_forecast=y_fc.tolist(),
            monthly=monthly,
        )

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
        x_last = float(x[-1])
        if family == "Кривые падения добычи (DCA)":
            x_fc = np.arange(x_last + 1, x_last + 1 + horizon, dtype=float)
        else:
            dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
            x_fc = np.linspace(x_last, x_last + dx * horizon, horizon)
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
            method_cls = self.method_panel.get_method_class()
            prod = self._get_displacement_data(sub)
            if method_cls is None or prod is None:
                return None, None
            x, y = method_cls.prepare_xy(*prod)

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
    def _get_displacement_data(
        sub: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (Qo, Ql, Qw, qo, ql, qw) aggregated by date."""
        if COL_DATE not in sub.columns:
            return None
        cols = {COL_OIL: "sum", COL_WATER: "sum"}
        for c in cols:
            if c not in sub.columns:
                return None
        agg = sub.groupby(COL_DATE).agg(cols).sort_index()
        qo = agg[COL_OIL].values.astype(float)
        qw = agg[COL_WATER].values.astype(float)
        ql = qo + qw
        Qo = np.cumsum(qo)
        Ql = np.cumsum(ql)
        Qw = np.cumsum(qw)
        return Qo, Ql, Qw, qo, ql, qw

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

    # ── Save project ────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        if not self._saved_results:
            self.status.showMessage("Нет сохранённых трендов для записи", 3000)
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект", "",
            "Forecast file (*.fcst);;All files (*)"
        )
        if not path:
            return
        from src.export.exporter import save_fcst_file
        try:
            save_fcst_file(
                path, self._saved_results,
                self._selected_wells, self._source_file
            )
            self.status.showMessage(f"Проект сохранён: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {exc}")

    # ── Data helpers ───────────────────────────────────────────────────

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
