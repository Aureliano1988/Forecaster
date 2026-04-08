"""Main application window — orchestrates all panels and forecasting logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

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
from src.data.loader import load_file, recompute_derived
from src.export.exporter import export_forecast_csv, export_plot
from src.forecasting.base import ForecastMethod
from src.forecasting.displacement import LinearDisplacement
from src.forecasting.monthly import (
    anchor_displacement_method,
    build_displacement_forecast,
    build_dca_forecast,
    build_fractional_forecast,
    dca_time_shift,
    fractional_qo_anchor,
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
        self._lasso_include_path: Path | None = None   # inclusion lasso polygon
        self._lasso_exclude_paths: list[Path] = []    # exclusion lasso polygons
        self._current_method: ForecastMethod | None = None
        self._eraser_active: bool = False
        self._fit_result_text: str = ""           # text produced by the last fit
        self._source_file: str = ""               # path of last loaded file
        self._saved_results: dict[str, SavedMethodResult] = {}  # keyed by family|method

        # Plot overlay references (for in-place update)
        self._trend_line = None
        self._forecast_line = None
        self._excl_patches: list = []
        self._include_patch = None          # blue polygon for current selection

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
        file_menu.addAction("Открыть данные…", self._on_load)
        file_menu.addAction("Открыть проект…", self._on_load_project)
        file_menu.addSeparator()
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
        self.method_panel.autofit_requested.connect(self._on_autofit)
        self.method_panel.autofit_all_requested.connect(self._on_autofit_all)

    # ── Load ────────────────────────────────────────────────────────────

    def _on_load(self) -> None:
        paths = self.data_panel.get_file_paths()
        if not paths:
            return

        # ── Load and validate every selected file ────────────────────────
        valid_dfs: list[tuple[str, pd.DataFrame]] = []
        for path in paths:
            try:
                df_i = load_file(path)
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка", f"{path}:\n{exc}")
                continue
            vr = validate(df_i)
            if not vr.is_valid:
                QMessageBox.critical(
                    self, "Ошибка данных",
                    f"{path}:\n" + "\n".join(vr.errors)
                )
                continue
            if vr.warnings:
                self.status.showMessage("; ".join(vr.warnings), 3000)
            valid_dfs.append((path, df_i))

        if not valid_dfs:
            return

        # ── Merge all selected files into one new dataframe ───────────────
        if len(valid_dfs) == 1:
            new_path, new_df = valid_dfs[0]
        else:
            concat_df = pd.concat([df for _, df in valid_dfs], ignore_index=True)
            concat_df = concat_df.drop_duplicates(
                subset=[COL_WELL, COL_DATE], keep="first"
            )
            new_df = recompute_derived(concat_df)
            new_path = " + ".join(p for p, _ in valid_dfs)

        # ── Decide: load fresh or append to existing ───────────────────
        if self.df is not None:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Данные уже загружены")
            msg_box.setText(
                f"В приложении уже есть данные ({len(self.df)} строк).\n"
                "Что сделать с данными из нового файла?"
            )
            from PySide6.QtWidgets import QPushButton
            btn_append  = msg_box.addButton("Добавить",  QMessageBox.ButtonRole.AcceptRole)
            btn_replace = msg_box.addButton("Заменить", QMessageBox.ButtonRole.DestructiveRole)
            msg_box.addButton("Отмена",  QMessageBox.ButtonRole.RejectRole)
            msg_box.exec()
            clicked = msg_box.clickedButton()
            if clicked is None or clicked not in (btn_append, btn_replace):
                return
            append = clicked is btn_append
        else:
            append = False

        # ── Apply the chosen action ───────────────────────────────────
        self._saved_results.clear()
        self._fit_result_text = ""
        if append:
            combined = pd.concat([self.df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=[COL_WELL, COL_DATE], keep="first"
            )
            self.df = recompute_derived(combined)
            self._source_file = self._source_file + " + " + new_path
            action_msg = "Данные добавлены"
        else:
            self.df = new_df
            self._source_file = new_path
            action_msg = "Данные загружены"

        wells = sorted(self.df[COL_WELL].unique().tolist())
        self.data_panel.populate_wells(wells)
        n = len(self.df)
        self.data_panel.lbl_info.setText(
            f"Загружено {n} строк, {len(wells)} скважин"
        )
        self.status.showMessage(action_msg, 3000)

    # ── Well selection → plot ────────────────────────────────────────────────

    def _on_wells_changed(self, wells: list[str]) -> None:
        self._selected_wells = wells
        self._lasso_include_path = None
        self._lasso_exclude_paths.clear()
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
        """Plot historical data (if available) and saved overlay for the current method."""
        # Always clear lasso on any plot refresh — avoids stale selection across methods
        self._lasso_include_path = None
        self._lasso_exclude_paths.clear()

        family = self.method_panel.get_family_name()
        self.plot_widget.clear()
        self._trend_line = None
        self._forecast_line = None
        self._include_patch = None
        self._excl_patches.clear()
        ax = self.plot_widget.ax

        # ── Historical data (only if data is loaded) ────────────────────────
        if self.df is not None and self._selected_wells:
            sub = self._get_sub()
            if sub is not None:
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

        ax.set_title(", ".join(self._selected_wells[:5]) if self._selected_wells else "")

        # ── Restore saved overlay (works even without source data) ─────────
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

        self.plot_widget.enable_lasso_selector(self._on_lasso_select)
        self.plot_widget.redraw()

    # ── Lasso selector callback ────────────────────────────────────────────────────────────

    def _on_lasso_select(self, vertices: list) -> None:
        """Called when the user finishes drawing a lasso polygon."""
        if len(vertices) < 3:
            return
        path = Path(vertices)
        if self._eraser_active:
            self._lasso_exclude_paths.append(path)
            self._draw_exclusion_polygons()
            self.status.showMessage(
                f"Исключена область "
                f"(всего зон: {len(self._lasso_exclude_paths)})", 5000
            )
        else:
            self._lasso_include_path = path
            self._draw_inclusion_polygon()
            self.status.showMessage("Область для тренда задана", 5000)

    # ── Eraser toggle ───────────────────────────────────────────────────────────────

    def _on_eraser_toggle(self, active: bool) -> None:
        self._eraser_active = active
        if active:
            self.status.showMessage(
                "Ластик: нарисуйте лассо вокруг точек для исключения", 5000
            )
        else:
            self.status.showMessage("Ластик выключён", 2000)

    def _draw_inclusion_polygon(self) -> None:
        """Draw a blue filled polygon for the current inclusion lasso."""
        if self._include_patch is not None:
            try:
                self._include_patch.remove()
            except Exception:
                pass
            self._include_patch = None
        if self._lasso_include_path is not None:
            patch = PathPatch(
                self._lasso_include_path,
                alpha=0.20, facecolor="tab:blue", edgecolor="tab:blue", lw=0.5, zorder=2
            )
            self.plot_widget.ax.add_patch(patch)
            self._include_patch = patch
        self.plot_widget.canvas.draw_idle()

    def _draw_exclusion_polygons(self) -> None:
        """Draw red filled polygons for all excluded lasso regions."""
        for p in self._excl_patches:
            try:
                p.remove()
            except Exception:
                pass
        self._excl_patches.clear()
        ax = self.plot_widget.ax
        for path in self._lasso_exclude_paths:
            patch = PathPatch(
                path, alpha=0.20, facecolor="red", edgecolor="red", lw=0.5, zorder=2
            )
            ax.add_patch(patch)
            self._excl_patches.append(patch)
        self.plot_widget.canvas.draw_idle()

    def _apply_exclusions(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove points falling inside any excluded lasso polygon."""
        if not self._lasso_exclude_paths:
            return x, y
        points = np.column_stack([x, y])
        mask = np.ones(len(x), dtype=bool)
        for path in self._lasso_exclude_paths:
            mask &= ~path.contains_points(points)
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

        # Apply lasso inclusion filter
        if self._lasso_include_path is not None:
            points = np.column_stack([x, y])
            inside = self._lasso_include_path.contains_points(points)
            x_sel, y_sel = x[inside], y[inside]
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
        self._lasso_include_path = None
        self._lasso_exclude_paths.clear()
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
        # Last historical oil rate — used to anchor DCA forecast
        q_last_oil = float(y[-1]) if len(y) else 0.0

        # ── Physical last values for monthly forecast builders ─────────────────
        Qo_last = Ql_last = Qw_last = ql_last = 0.0
        qo_last_monthly = qw_last_monthly = 0.0
        sub = self._get_sub()
        if sub is not None:
            prod = self._get_displacement_data(sub)
            if prod is not None:
                Qo_a, Ql_a, Qw_a, qo_a, ql_a, qw_a = prod
                if len(Qo_a):
                    Qo_last = float(Qo_a[-1])
                    Ql_last = float(Ql_a[-1])
                    Qw_last = float(Qw_a[-1])
                    qo_last_monthly = float(qo_a[-1])
                    qw_last_monthly = float(qw_a[-1])
                nz = ql_a[ql_a > 0]
                ql_last = float(nz[-1]) if len(nz) else float(ql_a[-1]) if len(ql_a) else 1.0
        # Water cut at last historical month (for fractional-flow anchoring)
        fw_last = (ql_last - q_last_oil) / ql_last if ql_last > 0 else 0.0

        # ── Build monthly series ──────────────────────────────────────────
        monthly: ForecastSeries | None = None
        dca_t_shift: float = 0.0   # computed below for DCA, reused for plot
        Qo_eff_frac: float = Qo_last  # anchor Qo for fractional-flow visual
        try:
            if family == "Характеристики вытеснения" and isinstance(
                self._current_method, LinearDisplacement
            ):
                monthly = build_displacement_forecast(
                    self._current_method,
                    Qo_last, Ql_last, Qw_last,
                    qo_last_monthly, qw_last_monthly,
                    ql_last, horizon, wor_limit,
                )
            elif family == "Кривые падения добычи (DCA)":
                # Anchor curve to last historical rate before projecting
                dca_t_shift = dca_time_shift(self._current_method, q_last_oil)
                monthly = build_dca_forecast(
                    self._current_method, x_last, q_last_oil, ql_last, horizon, wor_limit,
                )
            elif family == "Фракционный поток":
                # Find Qo_eff where fw(Qo_eff) = fw_last, then anchor from there
                Qo_eff_frac = fractional_qo_anchor(self._current_method, fw_last, Qo_last)
                monthly = build_fractional_forecast(
                    self._current_method, Qo_eff_frac, fw_last, ql_last, horizon, wor_limit,
                )
        except Exception as exc:
            self.status.showMessage(f"Ошибка расчёта прогноза: {exc}", 5000)

        actual_duration = monthly.duration if monthly else horizon

        # ── Forecast line in method-space (visual only) ───────────────────────
        n = max(actual_duration, 1)
        dx = (x[-1] - x[0]) / max(len(x) - 1, 1) if len(x) > 1 else 1.0
        if family == "Кривые падения добычи (DCA)":
            # x positions = natural month indices; y uses time-shifted model
            x_fc = np.arange(x_last + 1, x_last + 1 + n, dtype=float)
            y_fc = self._current_method.predict(dca_t_shift + np.arange(1, n + 1, dtype=float))
        elif family == "Характеристики вытеснения":
            # Use anchored method copy so the forecast line starts at Y_last
            method_vis = anchor_displacement_method(
                self._current_method,
                Qo_last, Ql_last, Qw_last,
                qo_last_monthly, ql_last, qw_last_monthly,
            )
            x_fc = np.linspace(x_last, x_last + dx * n, n)
            y_fc = method_vis.predict(x_fc) if len(x_fc) else np.array([])
        elif family == "Фракционный поток":
            # Start forecast line from Qo_eff so it opens at fw_last
            x_fc = np.linspace(Qo_eff_frac, Qo_eff_frac + dx * n, n)
            y_fc = self._current_method.predict(x_fc) if len(x_fc) else np.array([])
        else:
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
            uur = Qo_last + monthly.remain_reserves
            result_text += (
                f"\n{'\u2500'*22}\n"
                f"Прогноз (стоп: {stopped_by}):\n"
                f"  Горизонт: {monthly.duration} мес.\n"
                f"  Нак. нефть (факт): {Qo_last:,.0f} т\n"
                f"  Ост. запасы: {monthly.remain_reserves:,.0f} т\n"
                f"  УИН: {uur:,.0f} т\n"
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
        """Return (x, y) for the currently selected family + method."""
        family = self.method_panel.get_family_name()
        method_cls = (
            self.method_panel.get_method_class()
            if family == "Характеристики вытеснения" else None
        )
        return self._get_xy_for(family, method_cls)

    def _get_xy_for(
        self, family: str, method_cls=None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (x, y) for arbitrary family + method class (panel-independent)."""
        if self.df is None or not self._selected_wells:
            return None, None
        sub = self._get_sub()
        if sub is None:
            return None, None

        if family == "Характеристики вытеснения":
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

    # ── Autofit ────────────────────────────────────────────────────────────

    def _draw_span_selection(self, xmin: float, xmax: float) -> None:
        """Create a rectangular inclusion lasso covering [xmin, xmax] and draw it.

        Used by autofit to mark the selected fitting interval.
        """
        ylim = self.plot_widget.ax.get_ylim()
        eps = abs(ylim[1] - ylim[0]) * 0.05 or 1.0
        self._lasso_include_path = Path([
            (xmin, ylim[0] - eps),
            (xmax, ylim[0] - eps),
            (xmax, ylim[1] + eps),
            (xmin, ylim[1] + eps),
            (xmin, ylim[0] - eps),
        ])
        self._draw_inclusion_polygon()

    def _run_autofit(
        self, family: str, method_cls
    ) -> tuple | None:
        """Find the optimal fitting interval from the END of history.

        Adds points from the last data point backwards one by one until
        R² ≥ 0.99 or 120 points are used.  Returns (method, x_sel, y_sel, r2)
        or None if there is not enough data.
        """
        x, y = self._get_xy_for(family, method_cls)
        if x is None or len(x) < 3:
            return None

        best: tuple | None = None
        for n_pts in range(2, min(121, len(x) + 1)):
            x_sel = x[-n_pts:]
            y_sel = y[-n_pts:]
            x_sel, y_sel = self._apply_exclusions(x_sel, y_sel)
            if len(x_sel) < 2:
                continue
            m = method_cls()
            try:
                m.fit(x_sel, y_sel)
            except Exception:
                continue
            r2 = m.r_squared(x_sel, y_sel)
            best = (m, x_sel, y_sel, r2)
            if r2 >= 0.99:
                break  # threshold reached — stop adding points
        return best

    def _autofit_method(
        self, family: str, method_cls
    ) -> tuple[SavedMethodResult, np.ndarray, object] | None:
        """Full autofit + forecast for one method.  Returns
        (SavedMethodResult, x_sel, method_obj) or None.
        """
        result = self._run_autofit(family, method_cls)
        if result is None:
            return None
        method, x_sel, y_sel, r2 = result

        x, y = self._get_xy_for(family, method_cls)
        if x is None or len(x) == 0:
            return None

        # ─ Fit result text
        params = method.get_parameters()
        lines = [
            f"Метод: {method.get_name()}",
            f"R² = {r2:.4f}",
            f"  Точек (авто): {len(x_sel)}",
        ]
        for k, v in params.items():
            lines.append(f"  {k} = {float(v):.6g}")
        fit_text = "\n".join(lines)

        # ─ Trend plot arrays
        x_trend = np.linspace(float(x_sel.min()), float(x_sel.max()), 300)
        y_trend = method.predict(x_trend)

        # ─ Physical last values
        sub = self._get_sub()
        Qo_last = Ql_last = Qw_last = ql_last = 0.0
        qo_last_m = qw_last_m = 0.0
        if sub is not None:
            prod = self._get_displacement_data(sub)
            if prod is not None:
                Qo_a, Ql_a, Qw_a, qo_a, ql_a, qw_a = prod
                if len(Qo_a):
                    Qo_last = float(Qo_a[-1])
                    Ql_last = float(Ql_a[-1])
                    Qw_last = float(Qw_a[-1])
                    qo_last_m = float(qo_a[-1])
                    qw_last_m = float(qw_a[-1])
                nz = ql_a[ql_a > 0]
                ql_last = float(nz[-1]) if len(nz) else float(ql_a[-1]) if len(ql_a) else 1.0
        q_last_oil = qo_last_m if family != "Кривые падения добычи (DCA)" else (
            float(y[-1]) if len(y) else 0.0
        )
        fw_last = (ql_last - q_last_oil) / ql_last if ql_last > 0 else 0.0

        horizon   = self.method_panel.get_horizon()
        wor_limit = self.method_panel.get_wor_limit()
        x_last    = float(x[-1])
        dx        = (x[-1] - x[0]) / max(len(x) - 1, 1) if len(x) > 1 else 1.0

        # ─ Monthly forecast
        monthly = None
        dca_t_shift = 0.0
        Qo_eff_frac = Qo_last
        try:
            if family == "Характеристики вытеснения" and isinstance(method, LinearDisplacement):
                monthly = build_displacement_forecast(
                    method, Qo_last, Ql_last, Qw_last,
                    qo_last_m, qw_last_m, ql_last, horizon, wor_limit,
                )
            elif family == "Кривые падения добычи (DCA)":
                dca_t_shift = dca_time_shift(method, q_last_oil)
                monthly = build_dca_forecast(
                    method, x_last, q_last_oil, ql_last, horizon, wor_limit
                )
            elif family == "Фракционный поток":
                Qo_eff_frac = fractional_qo_anchor(method, fw_last, Qo_last)
                monthly = build_fractional_forecast(
                    method, Qo_eff_frac, fw_last, ql_last, horizon, wor_limit
                )
        except Exception:
            pass

        # ─ Forecast plot arrays
        n = max(monthly.duration if monthly else horizon, 1)
        if family == "Кривые падения добычи (DCA)":
            x_fc = np.arange(x_last + 1, x_last + 1 + n, dtype=float)
            y_fc = method.predict(dca_t_shift + np.arange(1, n + 1, dtype=float))
        elif family == "Характеристики вытеснения":
            mv = anchor_displacement_method(
                method, Qo_last, Ql_last, Qw_last, qo_last_m, ql_last, qw_last_m
            )
            x_fc = np.linspace(x_last, x_last + dx * n, n)
            y_fc = mv.predict(x_fc)
        else:  # fractional
            x_fc = np.linspace(Qo_eff_frac, Qo_eff_frac + dx * n, n)
            y_fc = method.predict(x_fc) if len(x_fc) else np.array([])

        # ─ Result text
        result_text = fit_text
        if monthly and monthly.duration > 0:
            stopped_by = (
                "WOR" if monthly.WOR and monthly.WOR[-1] >= wor_limit else "горизонт"
            )
            uur = Qo_last + monthly.remain_reserves
            result_text += (
                f"\n{'\u2500'*22}\n"
                f"Прогноз (стоп: {stopped_by}):\n"
                f"  Горизонт: {monthly.duration} мес.\n"
                f"  Нак. нефть (факт): {Qo_last:,.0f} т\n"
                f"  Ост. запасы: {monthly.remain_reserves:,.0f} т\n"
                f"  УИН: {uur:,.0f} т\n"
                f"  WOR (посл.): {monthly.wor_last:.2f}"
            )

        saved = SavedMethodResult(
            method_name=method.get_name(),
            params_text=result_text,
            parameters={k: float(v) for k, v in method.get_parameters().items()},
            x_trend=x_trend.tolist(),
            y_trend=y_trend.tolist(),
            x_forecast=x_fc.tolist(),
            y_forecast=y_fc.tolist(),
            monthly=monthly,
        )
        return saved, x_sel, method

    def _on_autofit(self) -> None:
        """Autofit the currently selected method and build the forecast."""
        family     = self.method_panel.get_family_name()
        method_cls = self.method_panel.get_method_class()
        if method_cls is None:
            return

        self.status.showMessage("Автоподбор...", 0)
        ret = self._autofit_method(family, method_cls)
        if ret is None:
            self.status.showMessage("Недостаточно данных для автоподбора", 4000)
            return

        saved, x_sel, method = ret
        key = self._result_key()
        self._saved_results[key] = saved
        self._current_method = method
        self._fit_result_text = saved.params_text

        # Set span range so manual "Build" also uses the autofit interval
        self._span_range = (float(x_sel.min()), float(x_sel.max()))

        # Redraw historical + overlay from saved_results
        self._on_plot_data()

        # Draw autofit interval highlight (after plot_data cleared it)
        self._draw_span_selection(float(x_sel.min()), float(x_sel.max()))
        self.plot_widget.redraw()

        r2 = saved.parameters.get("a", None)  # just use the status bar
        dur = saved.monthly.duration if saved.monthly else 0
        self.status.showMessage(
            f"Автоподбор: {len(x_sel)} точ., {dur} мес. прогноза", 6000
        )

    def _on_autofit_all(self) -> None:
        """Run autofit + forecast for every method in every family."""
        if self.df is None or not self._selected_wells:
            self.status.showMessage("Нет данных для автоподбора", 4000)
            return

        from src.forecasting.displacement import DISPLACEMENT_METHODS
        from src.forecasting.dca import DCA_METHODS
        from src.forecasting.fractional import FRACTIONAL_METHODS
        from PySide6.QtWidgets import QApplication

        families = {
            "Характеристики вытеснения": DISPLACEMENT_METHODS,
            "Кривые падения добычи (DCA)": DCA_METHODS,
            "Фракционный поток": FRACTIONAL_METHODS,
        }
        total = sum(len(v) for v in families.values())
        done = 0

        for family, methods in families.items():
            for method_cls in methods:
                self.status.showMessage(
                    f"Автоподбор всех: {done}/{total} — {method_cls().get_name()}", 0
                )
                QApplication.processEvents()
                try:
                    ret = self._autofit_method(family, method_cls)
                except Exception:
                    ret = None
                if ret is not None:
                    saved, _x_sel, _method = ret
                    key = f"{family}|{method_cls().get_name()}"
                    self._saved_results[key] = saved
                done += 1

        self.status.showMessage(f"Автоподбор всех: готово ({done} методов)", 6000)
        # Refresh to show current method's saved result
        self._on_plot_data()

    # ── Save / load project ───────────────────────────────────────────────

    def _on_load_project(self) -> None:
        """Open a .fcst project file and restore all saved results."""
        from PySide6.QtWidgets import QFileDialog
        import os
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть проект", "",
            "Forecast file (*.fcst);;All files (*)"
        )
        if not path:
            return
        from src.export.exporter import load_fcst_file
        try:
            project = load_fcst_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть проект: {exc}")
            return

        wells = project["wells"]
        source_file = project["source_file"]

        # Try to load the original data file (best-effort)
        if source_file and os.path.exists(source_file):
            try:
                from src.data.loader import load_file
                from src.data.validation import validate
                new_df = load_file(source_file)
                if validate(new_df).is_valid:
                    self.df = new_df
                    self._source_file = source_file
            except Exception:
                pass

        # Restore project state (don't clear saved_results here — we're setting them)
        self._saved_results = project["results"]
        self._selected_wells = list(wells)
        self._fit_result_text = ""
        self._current_method = None
        self._lasso_include_path = None
        self._lasso_exclude_paths.clear()

        # Populate the well list without triggering wells_changed
        if self.df is not None:
            all_wells = sorted(self.df[COL_WELL].unique().tolist())
        else:
            all_wells = sorted(set(wells))
        self.data_panel.populate_wells(all_wells)
        self.data_panel.well_list.blockSignals(True)
        for i in range(self.data_panel.well_list.count()):
            item = self.data_panel.well_list.item(i)
            if item and item.text() in set(wells):
                item.setSelected(True)
        self.data_panel.well_list.blockSignals(False)

        n_rows = len(self.df) if self.df is not None else 0
        self.data_panel.lbl_info.setText(
            f"Проект: {len(wells)} скважин"
            + (f", {n_rows} строк данных" if n_rows else " (данные не загружены)")
        )
        self._on_plot_data()
        self.status.showMessage(
            f"Проект загружен: {len(project['results'])} результатов", 5000
        )

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
        # COL_LIQUID is already imported at module level — no local import needed

        grp = sub.groupby(COL_DATE).apply(
            lambda g: (g[COL_WATER_CUT] * g[COL_LIQUID]).sum()
            / g[COL_LIQUID].sum()
            if g[COL_LIQUID].sum() > 0
            else 0.0
        )
        return grp.sort_index().values.astype(float)
