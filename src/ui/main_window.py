"""Main application window — orchestrates all panels and forecasting logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator as _MaxNLocator
from PySide6.QtGui import QAction, QKeySequence
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
    COL_CUM_GAS,
    COL_DATE,
    COL_GAS,
    COL_LIQUID,
    COL_OIL,
    COL_WATER,
    COL_WATER_CUT,
    COL_WELL,
    COL_WORK_TYPE,
    ForecastScenario,
    ForecastSeries,
    SavedMethodResult,
    WORK_TYPE_OIL,
)
from src.data.validation import validate
from src.data.loader import apply_manual_mapping, load_file, read_raw, recompute_derived
from src.export.exporter import export_forecast_csv, export_plot
from src.forecasting.base import ForecastMethod
from src.forecasting.displacement import LinearDisplacement
from src.forecasting.dca import ExponentialDecline, HarmonicDecline, HyperbolicDecline
from src.forecasting.fractional import BuckleyLeverettSemiLog, WaterCutVsCumOil
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
        self._project_name: str = ""              # display title for the project
        self._fit_result_text: str = ""           # text produced by the last fit
        self._source_files: list[str] = []        # paths of loaded data files
        self._current_save_path: str = ""         # path of last saved .fcst file
        self._saved_results: dict[str, SavedMethodResult] = {}  # keyed by family|method

        # ── Scenario management ───────────────────────────────────────────────
        self._scenarios: list[ForecastScenario] = []   # all named scenarios
        self._active_scenario_idx: int = 0

        # Reservoir parameters for RF / HCPVI computation
        self._stoiip: float = 0.0   # initial oil in place, tonnes
        self._hcpv:   float = 0.0   # hydrocarbon pore volume, m³

        # Well alignment analysis scenarios (persisted in the project file)
        self._well_analysis_scenarios: list = []

        # Plot overlay references (for in-place update)
        self._trend_line = None
        self._forecast_line = None
        self._excl_patches: list = []
        self._include_patch = None          # blue polygon for current selection

        # Trend editor state
        self._edit_trend_active: bool = False
        self._edit_handles: list = []       # [start, mid, end] Line2D markers
        self._edit_pressed: str | None = None  # 'start' | 'mid' | 'end'
        self._edit_x_start: float = 0.0    # fixed x of trend left endpoint
        self._edit_x_end: float   = 0.0    # fixed x of trend right endpoint
        self._edit_a: float = 0.0           # intercept in linear/log space
        self._edit_b: float = 0.0           # slope   in linear/log space
        self._edit_cids: list = []          # canvas mpl_connect IDs
        self._trend_param_dlg = None        # TrendParamDialog (lazy init)
        self._inspector_dlg   = None        # ForecastInspectorDialog (persistent)

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

        def _act(title: str, handler, shortcut: str = "") -> QAction:
            """Helper: create a QAction with optional shortcut and add to this window."""
            a = QAction(title, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(handler)
            return a

        # Файл
        file_menu = menu.addMenu("Файл")
        file_menu.addAction(_act("Открыть данные…",       self._on_load,            "Ctrl+Shift+O"))
        file_menu.addAction(_act("Открыть проект…",       self._on_load_project,    "Ctrl+O"))
        file_menu.addSeparator()
        file_menu.addAction(_act("Сохранить проект",        self._on_save,            "Ctrl+S"))
        file_menu.addAction(_act("Сохранить проект как…",   self._on_save_as,         "Ctrl+Shift+S"))
        file_menu.addSeparator()
        file_menu.addAction(_act("Закрыть проект",          self._on_close_project,   "Ctrl+W"))
        file_menu.addSeparator()
        file_menu.addAction(_act("Экспорт графика…",        self._on_export_plot,     "Ctrl+E"))
        file_menu.addAction(_act("Экспорт прогноза…",       self._on_export_forecast, "Ctrl+Shift+E"))
        file_menu.addSeparator()
        file_menu.addAction(_act("Выход",                     self.close))

        # Прогноз
        forecast_menu = menu.addMenu("Прогноз")
        forecast_menu.addAction(_act("Инспектор прогнозов…", self._on_forecast_inspector, "Ctrl+I"))
        forecast_menu.addSeparator()
        forecast_menu.addAction(_act("Ввести данные…",       self._on_enter_reservoir_data))
        forecast_menu.addSeparator()
        forecast_menu.addAction(_act("Сводка прогнозов…",  self._on_forecast_summary))
        forecast_menu.addAction(_act("Графики прогнозов…", self._on_forecast_plots))

        # Скважины
        wells_menu = menu.addMenu("Скважины")
        wells_menu.addAction(_act("Приведённая добыча по скважинам…", self._on_well_alignment))

        # ── Connections ──────────────────────────────────────────────────────
        self.data_panel.btn_load.clicked.connect(self._on_load)
        self.data_panel.wells_changed.connect(self._on_wells_changed)
        self.method_panel.build_requested.connect(self._on_build_forecast)
        self.method_panel.discard_requested.connect(self._on_discard)
        self.method_panel.eraser_toggled.connect(self._on_eraser_toggle)
        self.method_panel.cmb_family.currentIndexChanged.connect(self._on_plot_data)
        self.method_panel.cmb_method.currentIndexChanged.connect(self._on_plot_data)
        self.method_panel.autofit_requested.connect(self._on_autofit)
        self.method_panel.autofit_all_requested.connect(self._on_autofit_all)
        self.method_panel.edit_toggled.connect(self._on_edit_toggle)
        self.data_panel.filter_applied.connect(self._on_filter_applied)
        self.data_panel.chk_active_wells.stateChanged.connect(self._on_plot_data)

    # ── Application close

    def closeEvent(self, event) -> None:
        """Ask to save before quitting (handles both the X button and Quit action)."""
        has_work = bool(self._saved_results) or any(s.results for s in self._scenarios)
        if not has_work:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Выход",
            "Сохранить изменения перед выходом?",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if reply == QMessageBox.StandardButton.Save:
            if self._current_save_path:
                if not self._do_save(self._current_save_path):
                    event.ignore()
                    return
            else:
                from PySide6.QtWidgets import QFileDialog
                path, _ = QFileDialog.getSaveFileName(
                    self, "Сохранить проект", "",
                    "Forecast file (*.fcst);;All files (*)"
                )
                if not path:
                    event.ignore()
                    return
                if not self._do_save(path):
                    event.ignore()
                    return
                self._current_save_path = path
        event.accept()

    # ── Close project ──────────────────────────────────

    def _on_close_project(self) -> None:
        """Return to the initial empty state, prompting to save if needed."""
        has_work = bool(self._saved_results) or any(s.results for s in self._scenarios)
        if has_work:
            reply = QMessageBox.question(
                self,
                "Закрыть проект",
                "Сохранить изменения перед закрытием проекта?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Save:
                if self._current_save_path:
                    if not self._do_save(self._current_save_path):
                        return
                else:
                    from PySide6.QtWidgets import QFileDialog
                    path, _ = QFileDialog.getSaveFileName(
                        self, "Сохранить проект", "",
                        "Forecast file (*.fcst);;All files (*)"
                    )
                    if not path:
                        return  # user cancelled the save dialog
                    if not self._do_save(path):
                        return
                    self._current_save_path = path

        # ── Reset to initial state ──────────────────────────────
        self._exit_edit_mode()

        self.df                     = None
        self._selected_wells        = []
        self._lasso_include_path    = None
        self._lasso_exclude_paths.clear()
        self._current_method        = None
        self._eraser_active         = False
        self._project_name          = ""
        self._fit_result_text       = ""
        self._source_files          = []
        self._current_save_path     = ""
        self._saved_results         = {}
        self._scenarios             = []
        self._active_scenario_idx   = 0
        self._stoiip                = 0.0
        self._hcpv                  = 0.0
        self._well_analysis_scenarios = []

        # Reset UI
        self._update_window_title()
        self.data_panel.well_list.blockSignals(True)
        self.data_panel.well_list.clear()
        self.data_panel.well_list.blockSignals(False)
        self.data_panel.lbl_info.setText("Файл не загружен")
        self.data_panel.lbl_filter.setText("")
        self.method_panel.show_result("")
        self.method_panel.set_edit_enabled(False)
        self.method_panel.btn_eraser.setChecked(False)

        self._on_plot_data()
        self.status.showMessage("Проект закрыт", 3000)

    # ── Load ───────────────────────────────────────────────────────────────

    def _on_load(self) -> None:
        paths = self.data_panel.get_file_paths()
        if not paths:
            return

        # ── Read raw data from every selected file ───────────────────
        raw_dfs: list[tuple[str, pd.DataFrame]] = []
        for path in paths:
            try:
                raw_dfs.append((path, read_raw(path)))
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка чтения", f"{path}:\n{exc}")
        if not raw_dfs:
            return

        # ── Show import dialog on the first file ─────────────────────
        from src.ui.data_import_dialog import DataImportDialog
        dlg = DataImportDialog(
            raw_dfs[0][1],
            n_files=len(raw_dfs),
            parent=self,
        )
        if dlg.exec() != DataImportDialog.DialogCode.Accepted:
            return
        col_mapping = dlg.result_mapping()

        # ── Apply mapping + validate every file ─────────────────────
        valid_dfs: list[tuple[str, pd.DataFrame]] = []
        for path, raw_df in raw_dfs:
            # Build per-file mapping: reuse the dialog mapping for columns that
            # exist in this file; additional columns are silently ignored.
            file_mapping = {
                col: col_mapping.get(col, "")
                for col in raw_df.columns
            }
            try:
                df_i = apply_manual_mapping(raw_df, file_mapping)
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка обработки", f"{path}:\n{exc}")
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
        self._current_save_path = ""        # data changed; clear associated project file
        new_paths = [p for p, _ in valid_dfs]
        if append:
            combined = pd.concat([self.df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=[COL_WELL, COL_DATE], keep="first"
            )
            self.df = recompute_derived(combined)
            self._source_files.extend(new_paths)
            action_msg = "Данные добавлены"
        else:
            self.df = new_df
            self._source_files = new_paths
            action_msg = "Данные загружены"

        wells = sorted(self.df[COL_WELL].unique().tolist())
        self.data_panel.populate_wells(wells)
        n = len(self.df)
        self.data_panel.lbl_info.setText(
            f"Загружено {n} строк, {len(wells)} скважин"
        )
        self.status.showMessage(action_msg, 3000)

        # Ask user for project name after data load
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "Название проекта",
            "Введите название проекта:",
            text=self._project_name,
        )
        if ok:
            self._project_name = name.strip()

        # On a fresh replace, initialise a single default scenario
        if not append:
            self._scenarios = [ForecastScenario(name="Сценарий 1", wells=[], results={})]
            self._active_scenario_idx = 0

        self._update_window_title()

    # ── Well selection → plot

    def _on_filter_applied(self, found: list[str], missing: list[str]) -> None:
        """Show filter result in the status bar."""
        n_found   = len(found)
        n_missing = len(missing)
        msg = f"Фильтр применён: {n_found} скважин выбрано"
        if n_missing:
            if n_missing <= 5:
                names_str = ", ".join(missing)
                msg += f", {n_missing} не найдено: {names_str}"
            else:
                msg += f", {n_missing} не найдено"
        self.status.showMessage(msg, 8000)

    def _on_wells_changed(self, wells: list[str]) -> None:
        self._selected_wells = wells
        self._lasso_include_path = None
        self._lasso_exclude_paths.clear()
        self._saved_results.clear()   # underlying data changed
        self._fit_result_text = ""
        self._on_plot_data()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _reconstruct_method(self, key: str, saved: SavedMethodResult) -> ForecastMethod | None:
        """Rebuild a ForecastMethod object from a SavedMethodResult's parameters.

        Used to restore ``_current_method`` after a project load or scenario
        switch, so that the trend-edit button becomes available immediately.
        """
        from src.forecasting.displacement import DISPLACEMENT_METHODS
        from src.forecasting.dca import DCA_METHODS
        from src.forecasting.fractional import FRACTIONAL_METHODS

        family = key.split("|", 1)[0]
        method_classes = {
            "Характеристики вытеснения": DISPLACEMENT_METHODS,
            "Кривые падения добычи (DCA)": DCA_METHODS,
            "Фракционный поток": FRACTIONAL_METHODS,
        }.get(family, [])

        method_cls = None
        for cls in method_classes:
            if cls().get_name() == saved.method_name:
                method_cls = cls
                break
        if method_cls is None:
            return None

        m = method_cls()
        # Apply saved parameters; handle key case differences (e.g. "Di" → attr "di")
        for attr_key, val in saved.parameters.items():
            attr_lower = (attr_key[0].lower() + attr_key[1:]) if attr_key else attr_key
            if hasattr(m, attr_lower):
                setattr(m, attr_lower, float(val))
            elif hasattr(m, attr_key):
                setattr(m, attr_key, float(val))
        return m

    def _update_window_title(self) -> None:
        """Reflect the current project name + active scenario in the title bar."""
        title = "Displacement Forecaster"
        if self._project_name:
            title += f" — {self._project_name}"
        if len(self._scenarios) > 1 and 0 <= self._active_scenario_idx < len(self._scenarios):
            title += f" [{self._scenarios[self._active_scenario_idx].name}]"
        self.setWindowTitle(title)

    # ── Scenario helpers ────────────────────────────────────────────────

    def _commit_active_scenario(self) -> None:
        """Write working buffers into the active scenario slot."""
        if not self._scenarios:
            return
        sc = self._scenarios[self._active_scenario_idx]
        sc.wells   = list(self._selected_wells)
        sc.results = dict(self._saved_results)

    def _load_scenario_into_buffers(self, idx: int) -> None:
        """Load scenario *idx* into working buffers and repopulate the well list.

        Does NOT commit the current state first — call _commit_active_scenario()
        before this when switching interactively.
        """
        self._active_scenario_idx = idx
        sc = self._scenarios[idx]

        self._selected_wells  = list(sc.wells)
        self._saved_results   = dict(sc.results)
        self._fit_result_text = ""
        self._current_method  = None
        self._lasso_include_path = None
        self._lasso_exclude_paths.clear()

        # Apply phase to the method panel (restricts families for gas)
        self.method_panel.set_phase(getattr(sc, "phase", "oil"))

        well_set  = set(sc.wells)
        all_wells = (
            sorted(self.df[COL_WELL].unique().tolist())
            if self.df is not None else sorted(well_set)
        )
        self.data_panel.well_list.blockSignals(True)
        self.data_panel.populate_wells(all_wells)
        for i in range(self.data_panel.well_list.count()):
            item = self.data_panel.well_list.item(i)
            if item and item.text() in well_set:
                item.setSelected(True)
        self.data_panel.well_list.blockSignals(False)

        self._update_window_title()
        self._on_plot_data()

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
        self._exit_edit_mode()              # exit editor before redrawing
        self.method_panel.set_edit_enabled(False)
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

        # ── Historical data (only if data is loaded) ──────────────────────────────────────
        sub: pd.DataFrame | None = None
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
                    _phase = self._active_phase()
                    _dca_col = COL_GAS if _phase == "gas" else COL_OIL
                    ts = self._monthly_series(sub, _dca_col)
                    if ts is not None:
                        _dca_lbl = "Добыча газа (факт)" if _phase == "gas" else "Добыча нефти (факт)"
                        _dca_y_lbl = "Добыча газа, м\u00b3/мес" if _phase == "gas" else "Добыча нефти, т/мес"
                        ax.plot(range(len(ts)), ts.values, "o-", ms=3, label=_dca_lbl)
                        ax.set_xlabel("Месяц")
                        ax.set_ylabel(_dca_y_lbl)
                        ax.set_yscale("log")
                elif family == "Фракционный поток":
                    x = self._agg_cumulative(sub, COL_CUM_OIL)
                    y = self._agg_watercut(sub)
                    if x is not None and y is not None:
                        ax.scatter(x, y, s=4, label="fw vs Qo (факт)")
                        ax.set_xlabel("Накопленная нефть, т")
                        ax.set_ylabel("Обводнённость")

                # ── Active-wells overlay (secondary Y-axis) ───────────────────────
                if self.data_panel.chk_active_wells.isChecked():
                    aw_xy = self._compute_active_wells_xy(sub, family, self._active_phase())
                    if aw_xy is not None:
                        aw_x, aw_counts = aw_xy
                        if len(aw_counts) > 0 and int(aw_counts.max()) > 0:
                            ax2 = ax.twinx()
                            ax2.step(aw_x, aw_counts, where='post',
                                     color='steelblue', alpha=0.45,
                                     linewidth=1.3, linestyle='--')
                            ax2.fill_between(
                                aw_x, aw_counts, step='post',
                                alpha=0.08, color='steelblue',
                            )
                            ax2.set_ylabel("Акт. скв.", color='steelblue', fontsize=9)
                            ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=8)
                            ax2.set_ylim(bottom=0)
                            ax2.yaxis.set_major_locator(
                                _MaxNLocator(integer=True, nbins=5)
                            )

        # Plot title: project name (bold) + well list
        wells_str = ""
        if self._selected_wells:
            wells_str = ", ".join(self._selected_wells[:4])
            if len(self._selected_wells) > 4:
                wells_str += f" +{len(self._selected_wells) - 4}"
        if self._project_name:
            title = self._project_name + (f"\n{wells_str}" if wells_str else "")
        else:
            title = wells_str
        ax.set_title(title, fontsize=10)

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
        # Always reconstruct the method object from the current saved result so
        # that the edit button reflects the method actually shown — even when
        # _current_method was set by a previously selected method family.
        if self._trend_line is not None:
            _key = self._result_key()
            if _key in self._saved_results:
                _m = self._reconstruct_method(_key, self._saved_results[_key])
                if _m is not None:
                    self._current_method = _m
        self.method_panel.set_edit_enabled(
            self._current_method is not None and self._trend_line is not None
        )
        self.plot_widget.redraw()

    # ── Lasso selector callback

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

    # ── Fit trend ────────────────────────────────────────────────────────
    def _on_fit(self) -> bool:
        self._exit_edit_mode()   # leave edit mode before fitting a new trend
        self.method_panel.set_edit_enabled(False)
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
        self.method_panel.set_edit_enabled(True)  # trend exists — enable editor
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
        self._exit_edit_mode()
        self.method_panel.set_edit_enabled(False)
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
        min_oil    = self.method_panel.get_min_oil()
        n_avg      = self.method_panel.get_n_avg()
        family     = self.method_panel.get_family_name()
        x_last     = float(x[-1])
        # Last historical oil rate — used to anchor DCA forecast
        # For DCA: average last n_avg rates from the method-space y-series
        if n_avg > 1 and len(y) >= 1:
            q_last_oil, _used = self._avg_last(y, n_avg)
            if _used < n_avg:
                self.status.showMessage(
                    f"Среднее DCA: запрошено {n_avg} мес., доступно {_used}", 5000
                )
        else:
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
                    # Cumulative totals stay as actual last-row endpoints
                    Qo_last = float(Qo_a[-1])
                    Ql_last = float(Ql_a[-1])
                    Qw_last = float(Qw_a[-1])
                    # Monthly rates: average last n_avg months
                    qo_last_monthly, used_qo = self._avg_last(qo_a, n_avg)
                    qw_last_monthly, _       = self._avg_last(qw_a, n_avg)
                    ql_val,          used_ql = self._avg_last(ql_a, n_avg)
                    ql_last = max(ql_val, 1.0)  # guard against zero liquid
                    if used_qo < n_avg or used_ql < n_avg:
                        avail = min(used_qo, used_ql)
                        self.status.showMessage(
                            f"Среднее: запрошено {n_avg} мес., доступно {avail} — используется {avail}", 5000
                        )
                    if ql_val <= 0:
                        self.status.showMessage(
                            "Предупреждение: средняя жидкость равна 0 — значение скорректировано до 1 т/мес", 6000
                        )
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
                    ql_last, horizon, wor_limit, min_oil,
                )
            elif family == "Кривые падения добычи (DCA)":
                # Anchor curve to last historical rate before projecting
                dca_t_shift = dca_time_shift(self._current_method, q_last_oil)
                monthly = build_dca_forecast(
                    self._current_method, x_last, q_last_oil, ql_last, horizon, wor_limit, min_oil,
                )
            elif family == "Фракционный поток":
                # Find Qo_eff where fw(Qo_eff) = fw_last, then anchor from there
                Qo_eff_frac = fractional_qo_anchor(self._current_method, fw_last, Qo_last)
                monthly = build_fractional_forecast(
                    self._current_method, Qo_eff_frac, fw_last, ql_last, horizon, wor_limit, min_oil,
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
            stopped_by = monthly.stop_reason or "горизонт"
            uur = Qo_last + monthly.remain_reserves
            result_text += (
                f"\n{'\u2500'*22}\n"
                f"Прогноз (стоп: {stopped_by}):\n"
                f"  Горизонт: {monthly.duration} мес.\n"
                f"  Нак. нефть (факт): {Qo_last:,.0f} т\n"
                f"  Ост. запасы: {monthly.remain_reserves:,.0f} т\n"
                f"  НТИК: {uur:,.0f} т\n"
                f"  ВНФ (посл.): {monthly.wor_last:.2f}"
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
            qo_hist_last=Qo_last,
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
            _dca_col = COL_GAS if self._active_phase() == "gas" else COL_OIL
            ts = self._monthly_series(sub, _dca_col)
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

    def _compute_active_wells_xy(
        self, sub: pd.DataFrame, family: str, phase: str = "oil",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (x_values, active_well_counts) aligned to the main plot's X-axis.

        Active wells = number of wells with non-zero production (oil or gas
        depending on *phase*) at each X-axis position.
        x_values are the same coordinates used for the primary plot data so the
        step overlay sits directly on top of the scatter/line data.
        """
        prod_col = COL_GAS if phase == "gas" else COL_OIL
        if COL_WELL not in sub.columns or prod_col not in sub.columns or COL_DATE not in sub.columns:
            return None

        # Per-date active-well count (wells with non-zero production)
        aw = (
            sub[sub[prod_col] > 0]
            .groupby(COL_DATE)[COL_WELL]
            .nunique()
            .sort_index()
        )

        if family == "Кривые падения добычи (DCA)":
            # X-axis = month index 0,1,2,... for dates with total production > 0
            ts = self._monthly_series(sub, prod_col)
            if ts is None:
                return None
            counts = aw.reindex(ts.index, fill_value=0).values.astype(int)
            x = np.arange(len(ts), dtype=float)

        elif family == "Характеристики вытеснения":
            method_cls = self.method_panel.get_method_class()
            prod = self._get_displacement_data(sub)
            if prod is None or method_cls is None:
                return None
            date_idx = sub.groupby(COL_DATE).size().sort_index().index
            counts = aw.reindex(date_idx, fill_value=0).values.astype(int)
            x_raw, _ = method_cls.prepare_xy(*prod)
            x = np.asarray(x_raw, dtype=float)

        elif family == "Фракционный поток":
            x_cum = self._agg_cumulative(sub, COL_CUM_OIL)
            if x_cum is None:
                return None
            date_idx = sub.groupby(COL_DATE)[COL_CUM_OIL].sum().sort_index().index
            counts = aw.reindex(date_idx, fill_value=0).values.astype(int)
            x = np.asarray(x_cum, dtype=float)

        else:
            return None

        n = min(len(x), len(counts))
        if n == 0:
            return None
        return x[:n], counts[:n]

    @staticmethod
    def _monthly_series(sub: pd.DataFrame, col: str) -> pd.Series | None:
        if col not in sub.columns or COL_DATE not in sub.columns:
            return None
        agg = sub.groupby(COL_DATE)[col].sum().sort_index()
        agg = agg[agg > 0]
        return agg if len(agg) > 0 else None

    # ── Trend editor ────────────────────────────────────────────────────────

    # ── Linear-space helpers (works for all supported method families) ─────

    def _edit_get_ab(self) -> tuple[float, float]:
        """Extract (a, b) from the current method in ‘linear space’.
        For LinearDisplacement  : a = method.a, b = method.b
        For DCA                 : a = log(qi),  b = -Di  (log-space)
        For BuckleyLeverettSemiLog: a = method.a, b = method.b
        """
        m = self._current_method
        if isinstance(m, LinearDisplacement):
            return m.a, m.b
        elif isinstance(m, (ExponentialDecline, HyperbolicDecline, HarmonicDecline)):
            return float(np.log(max(m.qi, 1e-12))), float(-m.di)
        elif isinstance(m, BuckleyLeverettSemiLog):
            return m.a, m.b
        return 0.0, 0.0

    def _edit_predict(self, x: np.ndarray) -> np.ndarray:
        """Fast preview prediction using stored (_edit_a, _edit_b)."""
        a, b = self._edit_a, self._edit_b
        m = self._current_method
        if isinstance(m, LinearDisplacement):
            return a + b * x
        elif isinstance(m, (ExponentialDecline, HyperbolicDecline, HarmonicDecline)):
            return np.exp(np.clip(a + b * x, -300, 300))
        elif isinstance(m, BuckleyLeverettSemiLog):
            return np.clip(1.0 - np.exp(np.clip(a + b * x, -300, 300)), 0.0, 1.0)
        return self._current_method.predict(x)

    def _edit_apply_ab(self, a: float, b: float) -> None:
        """Write (a, b) back into the current method’s parameters."""
        self._edit_a = a
        self._edit_b = b
        m = self._current_method
        if isinstance(m, LinearDisplacement):
            m.a = a
            m.b = b
        elif isinstance(m, (ExponentialDecline, HarmonicDecline)):
            m.qi = float(np.exp(np.clip(a, -300, 300)))
            m.di = float(max(-b, 0.0))
        elif isinstance(m, HyperbolicDecline):
            m.qi = float(np.exp(np.clip(a, -300, 300)))
            m.di = float(max(-b, 0.0))    # method.b (exponent) unchanged
        elif isinstance(m, BuckleyLeverettSemiLog):
            m.a = a
            m.b = b

    def _edit_y_to_lin(self, y: float) -> float:
        """Convert data-space y to the corresponding linear-space value."""
        m = self._current_method
        if isinstance(m, LinearDisplacement):
            return float(y)
        elif isinstance(m, (ExponentialDecline, HyperbolicDecline, HarmonicDecline)):
            return float(np.log(max(y, 1e-12)))
        elif isinstance(m, BuckleyLeverettSemiLog):
            fw = max(0.0, min(y, 1.0 - 1e-9))
            return float(np.log(1.0 - fw))
        return float(y)

    # ── Drag constraint math ───────────────────────────────────────────────

    def _compute_drag(self, handle: str, y_new: float) -> None:
        """Update _edit_a and _edit_b from a drag event.

        All arithmetic is done in the linear space appropriate for the
        current method family.  x-coordinates of endpoints are fixed.
        """
        y_lin = self._edit_y_to_lin(y_new)
        x_s   = self._edit_x_start
        x_e   = self._edit_x_end
        x_m   = (x_s + x_e) / 2.0
        dx    = x_e - x_s

        y_s_lin = self._edit_a + self._edit_b * x_s   # current start in lin-space
        y_e_lin = self._edit_a + self._edit_b * x_e   # current end   in lin-space

        if handle == 'start':
            if abs(dx) < 1e-12:
                return
            b_new = (y_e_lin - y_lin) / dx
            a_new = y_lin - b_new * x_s
        elif handle == 'end':
            if abs(dx) < 1e-12:
                return
            b_new = (y_lin - y_s_lin) / dx
            a_new = y_s_lin - b_new * x_s
        else:
            return

        self._edit_apply_ab(a_new, b_new)

    # ── Handle display ────────────────────────────────────────────────────

    def _update_edit_handles(self) -> None:
        """Remove old handle markers and place them at their current positions."""
        ax = self.plot_widget.ax
        for h in self._edit_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._edit_handles.clear()

        x_s = self._edit_x_start
        x_e = self._edit_x_end
        y_s = float(self._edit_predict(np.array([x_s]))[0])
        y_e = float(self._edit_predict(np.array([x_e]))[0])

        hs, = ax.plot([x_s], [y_s], 'o', ms=11, color='limegreen', zorder=11)
        he, = ax.plot([x_e], [y_e], 'o', ms=11, color='crimson',   zorder=11)
        self._edit_handles = [hs, he]
        self.plot_widget.canvas.draw_idle()

    # ── Enter / leave ──────────────────────────────────────────────────────

    def _on_edit_toggle(self, active: bool) -> None:
        if active:
            self._enter_edit_mode()
        else:
            self._exit_edit_mode()

    def _enter_edit_mode(self) -> None:
        """Start interactive trend editing."""
        m = self._current_method
        if m is None or self._trend_line is None:
            self.method_panel.set_edit_enabled(False)
            return
        if isinstance(m, WaterCutVsCumOil):
            self.status.showMessage(
                "Редактирование не поддерживается для логистической модели fw(Нак.нефть)", 5000
            )
            self.method_panel.set_edit_enabled(False)
            return

        self._edit_trend_active = True
        xdata = self._trend_line.get_xdata()
        self._edit_x_start = float(xdata[0])
        self._edit_x_end   = float(xdata[-1])
        self._edit_a, self._edit_b = self._edit_get_ab()

        self.plot_widget.disable_lasso_selector()
        self._update_edit_handles()

        canvas = self.plot_widget.canvas
        self._edit_cids = [
            canvas.mpl_connect('button_press_event',   self._on_trend_press),
            canvas.mpl_connect('motion_notify_event',  self._on_trend_motion),
            canvas.mpl_connect('button_release_event', self._on_trend_release),
        ]

        # ── Floating parameter dialog ──────────────────────────────────────
        if self._trend_param_dlg is None:
            from src.ui.trend_param_dialog import TrendParamDialog
            self._trend_param_dlg = TrendParamDialog(parent=self)
            self._trend_param_dlg.params_changed.connect(self._on_trend_params_changed)
        self._trend_param_dlg.set_method(
            m.get_name(), m.get_parameters()
        )
        # Position the dialog just to the right of the main window,
        # clamped to the available screen area so it is never off-screen.
        from PySide6.QtWidgets import QApplication
        screen_rect = QApplication.primaryScreen().availableGeometry()
        dlg = self._trend_param_dlg
        dlg.adjustSize()
        geo = self.frameGeometry()
        x = geo.right() + 8
        y = geo.top() + 120
        x = min(x, screen_rect.right()  - dlg.width()  - 4)
        y = min(y, screen_rect.bottom() - dlg.height() - 4)
        x = max(x, screen_rect.left())
        y = max(y, screen_rect.top())
        dlg.move(x, y)
        dlg.show()
        dlg.raise_()

        self.status.showMessage(
            "Редактирование тренда: зелёный = начало; красный = конец. Тяните маркеры или введите значения в окне параметров.", 0
        )

    def _exit_edit_mode(self) -> None:
        """Leave trend editing mode, restore lasso."""
        if not self._edit_trend_active:
            return
        self._edit_trend_active = False
        self._edit_pressed = None

        for h in self._edit_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._edit_handles.clear()

        canvas = self.plot_widget.canvas
        for cid in self._edit_cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._edit_cids.clear()

        self.plot_widget.enable_lasso_selector(self._on_lasso_select)
        # Uncheck button silently
        self.method_panel.btn_edit.blockSignals(True)
        self.method_panel.btn_edit.setChecked(False)
        self.method_panel.btn_edit.blockSignals(False)
        canvas.draw_idle()
        # Hide parameter dialog
        if self._trend_param_dlg is not None:
            self._trend_param_dlg.hide()

    # ── Mouse event handlers ───────────────────────────────────────────────

    def _on_trend_press(self, event) -> None:
        """Pick the nearest handle within PICK_RADIUS display pixels."""
        if event.button != 1 or event.inaxes != self.plot_widget.ax:
            return
        ax = self.plot_widget.ax
        click = np.array([event.x, event.y])
        PICK_RADIUS = 14

        x_s = self._edit_x_start
        x_e = self._edit_x_end
        candidates = {
            'start': (x_s, float(self._edit_predict(np.array([x_s]))[0])),
            'end':   (x_e, float(self._edit_predict(np.array([x_e]))[0])),
        }
        best, best_d = None, PICK_RADIUS
        for name, (xd, yd) in candidates.items():
            try:
                disp = ax.transData.transform([xd, yd])
                d = float(np.linalg.norm(click - disp))
                if d < best_d:
                    best_d, best = d, name
            except Exception:
                pass
        if best:
            self._edit_pressed = best

    def _on_trend_motion(self, event) -> None:
        """Live-preview the drag — update trend line and handles without rebuilding forecast."""
        if self._edit_pressed is None or event.inaxes != self.plot_widget.ax:
            return
        if event.ydata is None:
            return

        self._compute_drag(self._edit_pressed, float(event.ydata))

        # Redraw trend line
        x_line = np.linspace(self._edit_x_start, self._edit_x_end, 300)
        y_line = self._edit_predict(x_line)
        if self._trend_line is not None:
            self._trend_line.set_data(x_line, y_line)

        self._update_edit_handles()

    def _on_trend_release(self, event) -> None:
        """On mouse release: commit the new parameters and rebuild the forecast."""
        if self._edit_pressed is None:
            return
        self._edit_pressed = None

        # Redraw trend with exact method.predict (corrects the log-linear approx)
        if self._trend_line is not None:
            x_line = np.linspace(self._edit_x_start, self._edit_x_end, 300)
            self._trend_line.set_data(x_line, self._current_method.predict(x_line))

        # Re-sync edit params from the committed method state
        self._edit_a, self._edit_b = self._edit_get_ab()

        # Rebuild forecast with updated method
        self._on_forecast()

        # Sync parameter dialog after drag
        if self._trend_param_dlg is not None and self._trend_param_dlg.isVisible():
            self._trend_param_dlg.update_params(self._current_method.get_parameters())

        # Refresh handles after forecast
        if self._edit_trend_active:
            self._update_edit_handles()
            self.plot_widget.redraw()

    # ── Trend parameter dialog handler ────────────────────────────────

    def _on_trend_params_changed(self, params: dict) -> None:
        """Apply parameters typed in the floating dialog, redraw, rebuild forecast."""
        if self._current_method is None or self._trend_line is None:
            return

        # Write values directly onto the method object
        m = self._current_method
        for attr_key, val in params.items():
            attr_lower = (attr_key[0].lower() + attr_key[1:]) if attr_key else attr_key
            if hasattr(m, attr_lower):
                setattr(m, attr_lower, float(val))
            elif hasattr(m, attr_key):
                setattr(m, attr_key, float(val))

        # Sync the internal (a, b) representation used by drag handles
        self._edit_a, self._edit_b = self._edit_get_ab()

        # Recompute and redraw the trend line
        x_line = np.linspace(self._edit_x_start, self._edit_x_end, 300)
        self._trend_line.set_data(x_line, m.predict(x_line))
        if self._edit_trend_active:
            self._update_edit_handles()

        # Rebuild the forecast with the updated method
        self._on_forecast()
        self.plot_widget.canvas.draw_idle()

    # ── Autofit ──────────────────────────────────────────────────

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

        Strategy (descending search):
          1. Start with the maximum window of up to 120 points.
          2. Reduce by one point at a time (dropping the oldest point).
          3. Stop at the FIRST (largest) window where R² ≥ 0.99.
          4. If the threshold is never reached, return the window with
             the highest R² found (minimum MIN_PTS points).

        This avoids the trivial R² = 1 that occurs with only 2 points.
        """
        _MIN_PTS = 5       # never use fewer than this many points
        _R2_THRESHOLD = 0.99

        x, y = self._get_xy_for(family, method_cls)
        if x is None or len(x) < _MIN_PTS:
            return None

        max_pts = min(120, len(x))
        best: tuple | None = None

        for n_pts in range(max_pts, _MIN_PTS - 1, -1):  # 120 → 5
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
            # Keep track of best result so far
            if best is None or r2 > best[3]:
                best = (m, x_sel, y_sel, r2)
            if r2 >= _R2_THRESHOLD:
                break   # largest window achieving required precision found
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

        # ─ Physical last values (with n_avg-month averaging for rates)
        n_avg     = self.method_panel.get_n_avg()
        sub = self._get_sub()
        Qo_last = Ql_last = Qw_last = ql_last = 0.0
        qo_last_m = qw_last_m = 0.0
        if sub is not None:
            prod = self._get_displacement_data(sub)
            if prod is not None:
                Qo_a, Ql_a, Qw_a, qo_a, ql_a, qw_a = prod
                if len(Qo_a):
                    # Cumulative totals: exact last-row values (not averaged)
                    Qo_last = float(Qo_a[-1])
                    Ql_last = float(Ql_a[-1])
                    Qw_last = float(Qw_a[-1])
                    # Monthly rates: average last n_avg months
                    qo_last_m, _ = self._avg_last(qo_a, n_avg)
                    qw_last_m, _ = self._avg_last(qw_a, n_avg)
                    ql_val,    _ = self._avg_last(ql_a, n_avg)
                    ql_last = max(ql_val, 1.0)
        # q_last_oil: average DCA y-series, or use qo_last_m for others
        if family == "Кривые падения добычи (DCA)":
            q_last_oil, _ = self._avg_last(y, n_avg) if len(y) else (0.0, 0)
        else:
            q_last_oil = qo_last_m
        fw_last = (ql_last - q_last_oil) / ql_last if ql_last > 0 else 0.0

        horizon   = self.method_panel.get_horizon()
        wor_limit = self.method_panel.get_wor_limit()
        min_oil   = self.method_panel.get_min_oil()
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
                    qo_last_m, qw_last_m, ql_last, horizon, wor_limit, min_oil,
                )
            elif family == "Кривые падения добычи (DCA)":
                dca_t_shift = dca_time_shift(method, q_last_oil)
                monthly = build_dca_forecast(
                    method, x_last, q_last_oil, ql_last, horizon, wor_limit, min_oil,
                )
            elif family == "Фракционный поток":
                Qo_eff_frac = fractional_qo_anchor(method, fw_last, Qo_last)
                monthly = build_fractional_forecast(
                    method, Qo_eff_frac, fw_last, ql_last, horizon, wor_limit, min_oil,
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
            stopped_by = monthly.stop_reason or "горизонт"
            uur = Qo_last + monthly.remain_reserves
            result_text += (
                f"\n{'\u2500'*22}\n"
                f"Прогноз (стоп: {stopped_by}):\n"
                f"  Горизонт: {monthly.duration} мес.\n"
                f"  Нак. нефть (факт): {Qo_last:,.0f} т\n"
                f"  Ост. запасы: {monthly.remain_reserves:,.0f} т\n"
                f"  НТИК: {uur:,.0f} т\n"
                f"  ВНФ (посл.): {monthly.wor_last:.2f}"
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
            qo_hist_last=Qo_last,
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

    def _on_well_alignment(self) -> None:
        """Open the well alignment (adjusted production) window."""
        if self.df is None:
            self.status.showMessage("Данные не загружены", 3000)
            return
        from src.ui.well_alignment_dialog import WellAlignmentDialog
        dlg = WellAlignmentDialog(
            self.df,
            well_analysis_scenarios=self._well_analysis_scenarios,
            parent=self,
        )
        dlg.exec()
        # Retrieve updated scenarios list
        self._well_analysis_scenarios = dlg.result_scenarios()

    def _compute_scenario_hist(self, sc) -> "dict | None":
        """Compute hist_data dict for a scenario's well selection.

        Returns a dict with the same keys as the active-selection hist_data
        (qo, ql, Qo, Ql, WOR, and optionally RF, qi_inj, Qi_inj, HCPVI).
        Returns None when the dataframe is missing or the scenario has no wells.
        """
        if self.df is None or not sc.wells:
            return None

        sc_sub = self.df[self.df[COL_WELL].isin(sc.wells)].copy()
        sc_sub_oil = (
            sc_sub[sc_sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
            if COL_WORK_TYPE in sc_sub.columns
            else sc_sub
        )
        prod = self._get_displacement_data(sc_sub_oil)
        if prod is None:
            return None

        Qo, Ql, _Qw, qo, ql, qw = prod
        sc_hist: dict = {
            "qo": qo,
            "ql": ql,
            "Qo": Qo,
            "Ql": Ql,
            "WOR": np.where(qo > 0, qw / qo, 0.0),
        }

        eff_stoiip = sc.stoiip if sc.stoiip > 0 else self._stoiip
        if eff_stoiip > 0:
            sc_hist["RF"] = Qo / eff_stoiip

        if COL_WORK_TYPE in self.df.columns and COL_WATER in self.df.columns:
            from src.data.models import WORK_TYPE_INJ
            inj = self.df[
                (self.df[COL_WELL].isin(sc.wells)) &
                (self.df[COL_WORK_TYPE] == WORK_TYPE_INJ)
            ]
            if len(inj) > 0:
                try:
                    prod_dates = (
                        sc_sub_oil.groupby(COL_DATE)[COL_OIL]
                        .sum().sort_index().index
                    )
                    inj_monthly = inj.groupby(COL_DATE)[COL_WATER].sum().sort_index()
                    inj_aligned = inj_monthly.reindex(prod_dates, fill_value=0.0)
                    qi_arr = inj_aligned.values.astype(float)
                    qi_cum = np.cumsum(qi_arr)
                    sc_hist["qi_inj"] = qi_arr
                    sc_hist["Qi_inj"] = qi_cum
                    eff_hcpv = sc.hcpv if sc.hcpv > 0 else self._hcpv
                    if eff_hcpv > 0 and len(qi_cum) > 0:
                        sc_hist["HCPVI"] = qi_cum / eff_hcpv
                except Exception:
                    pass

        return sc_hist

    def _active_phase(self) -> str:
        """Return the phase ("oil" | "gas") for the active scenario."""
        if self._scenarios and 0 <= self._active_scenario_idx < len(self._scenarios):
            return getattr(self._scenarios[self._active_scenario_idx], "phase", "oil")
        return "oil"

    def _sc_stoiip(self) -> float:
        """Effective STOIIP for the active scenario (falls back to project default)."""
        if self._scenarios:
            sc = self._scenarios[self._active_scenario_idx]
            if sc.stoiip > 0:
                return sc.stoiip
        return self._stoiip

    def _sc_hcpv(self) -> float:
        """Effective HCPV for the active scenario (falls back to project default)."""
        if self._scenarios:
            sc = self._scenarios[self._active_scenario_idx]
            if sc.hcpv > 0:
                return sc.hcpv
        return self._hcpv

    def _on_enter_reservoir_data(self) -> None:
        """Open the reservoir parameters dialog to enter STOIIP and HCPV."""
        from src.ui.reservoir_data_dialog import ReservoirDataDialog
        from PySide6.QtWidgets import QPushButton

        # Pre-fill with the active scenario's effective values
        dlg = ReservoirDataDialog(
            stoiip=self._sc_stoiip(),
            hcpv=self._sc_hcpv(),
            parent=self,
        )
        if dlg.exec() != ReservoirDataDialog.DialogCode.Accepted:
            return

        new_stoiip = dlg.get_stoiip()
        new_hcpv   = dlg.get_hcpv()

        # Ask scope when there are scenarios to distinguish between
        apply_to_all = True
        if self._scenarios:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Применить данные пласта")
            sc_name = self._scenarios[self._active_scenario_idx].name
            msg_box.setText(
                f"Применить STOIIP / HCPV:"
            )
            btn_sc  = msg_box.addButton(
                f"Только для сценария «{sc_name}»",
                QMessageBox.ButtonRole.AcceptRole,
            )
            btn_all = msg_box.addButton(
                "Всему проекту (всем сценариям)",
                QMessageBox.ButtonRole.ActionRole,
            )
            msg_box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
            msg_box.exec()
            clicked = msg_box.clickedButton()
            if clicked is None or clicked not in (btn_sc, btn_all):
                return
            apply_to_all = (clicked is btn_all)

        if apply_to_all:
            self._stoiip = new_stoiip
            self._hcpv   = new_hcpv
            for sc in self._scenarios:
                sc.stoiip = new_stoiip
                sc.hcpv   = new_hcpv
            scope_msg = "всему проекту"
        else:
            sc = self._scenarios[self._active_scenario_idx]
            sc.stoiip = new_stoiip
            sc.hcpv   = new_hcpv
            scope_msg = f"сценарию «{sc.name}»"

        parts = []
        if new_stoiip > 0:
            parts.append(f"STOIIP={new_stoiip:,.0f} т")
        if new_hcpv > 0:
            parts.append(f"HCPV={new_hcpv:,.0f} м\u00b3")
        vals = ", ".join(parts) if parts else "нулевые значения"
        self.status.showMessage(f"Данные пласта сохранены — {vals} — применено {scope_msg}", 6000)

    def _on_forecast_plots(self) -> None:
        """Open the interactive forecast plots window."""
        has_forecasts = any(
            v.monthly is not None and v.monthly.duration > 0
            for v in self._saved_results.values()
        )
        if not has_forecasts:
            self.status.showMessage("Нет рассчитанных прогнозов для отображения", 3000)
            return

        # Aggregate historical production for the current well selection
        hist_data: dict | None = None
        qi_hist_last: float = 0.0
        qi_const: float = 0.0
        sub = self._get_sub()
        if sub is not None and self.df is not None:
            prod = self._get_displacement_data(sub)
            if prod is not None:
                Qo, Ql, _Qw, qo, ql, qw = prod
                hist_data = {
                    "qo": qo,
                    "ql": ql,
                    "Qo": Qo,
                    "Ql": Ql,
                    "WOR": np.where(qo > 0, qw / qo, 0.0),
                }
                # RF and HCPVI use the active scenario's effective values
                eff_stoiip_hist = self._sc_stoiip()
                eff_hcpv_hist   = self._sc_hcpv()
                if eff_stoiip_hist > 0:
                    hist_data["RF"] = Qo / eff_stoiip_hist

                # Injection history aligned to production dates
                if COL_WORK_TYPE in self.df.columns and COL_WATER in self.df.columns:
                    from src.data.models import WORK_TYPE_INJ
                    inj_df = self.df[
                        (self.df[COL_WELL].isin(self._selected_wells)) &
                        (self.df[COL_WORK_TYPE] == WORK_TYPE_INJ)
                    ]
                    if len(inj_df) > 0:
                        prod_dates = (
                            sub.groupby(COL_DATE)[COL_OIL].sum()
                            .sort_index().index
                        )
                        inj_monthly = (
                            inj_df.groupby(COL_DATE)[COL_WATER].sum()
                            .sort_index()
                        )
                        inj_aligned = inj_monthly.reindex(prod_dates, fill_value=0.0)
                        qi_arr = inj_aligned.values.astype(float)
                        qi_cum = np.cumsum(qi_arr)
                        qi_hist_last = float(qi_cum[-1]) if len(qi_cum) else 0.0
                        n_avg = self.method_panel.get_n_avg()
                        qi_const, _ = self._avg_last(qi_arr, n_avg)
                        # Historical injection series for plot overlay
                        hist_data["qi_inj"] = qi_arr       # monthly injection
                        hist_data["Qi_inj"] = qi_cum       # cumulative injection
                        if eff_hcpv_hist > 0 and len(qi_cum) > 0:
                            hist_data["HCPVI"] = qi_cum / eff_hcpv_hist

        # Commit current scenario so all scenarios are up-to-date
        self._commit_active_scenario()

        # Use the effective STOIIP/HCPV for the active scenario
        eff_stoiip = self._sc_stoiip()
        eff_hcpv   = self._sc_hcpv()

        # Compute historical data for every scenario (for the scenario-comparison mode)
        scenarios_hist = [self._compute_scenario_hist(sc) for sc in self._scenarios]

        from src.ui.forecast_plots_dialog import ForecastPlotsDialog
        dlg = ForecastPlotsDialog(
            self._saved_results,
            project_name=self._project_name,
            hist_data=hist_data,
            stoiip=eff_stoiip,
            hcpv=eff_hcpv,
            qi_hist_last=qi_hist_last,
            qi_const=qi_const,
            scenarios=self._scenarios,
            scenarios_hist=scenarios_hist,
            parent=self,
        )
        dlg.exec()

    def _on_forecast_summary(self) -> None:
        """Open the forecasts summary window."""
        if not self._saved_results:
            self.status.showMessage("Нет сохранённых результатов для отображения", 3000)
            return
        from src.ui.summary_dialog import ForecastSummaryDialog
        dlg = ForecastSummaryDialog(
            self._saved_results,
            project_name=self._project_name,
            parent=self,
        )
        dlg.exec()

    # ── Save / load project ───────────────────────────────────────────────

    def _on_load_project(self) -> None:
        """Open a .fcst project file and restore all saved results."""
        from PySide6.QtWidgets import QFileDialog
        import os

        fcst_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть проект", "",
            "Forecast file (*.fcst);;All files (*)"
        )
        if not fcst_path:
            return

        from src.export.exporter import load_fcst_file
        try:
            project = load_fcst_file(fcst_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть проект: {exc}")
            return

        src_paths: list[str] = project["source_files"]
        scenarios: list[ForecastScenario] = project["scenarios"]
        # Scenario 0 provides the initial well selection
        sc0_wells = scenarios[0].wells if scenarios else []

        # ── Load every stored source file; offer to locate missing ones ───────
        loaded_dfs: list[pd.DataFrame] = []
        loaded_paths: list[str] = []
        for src in src_paths:
            resolved = src
            if not os.path.exists(resolved):
                reply = QMessageBox.question(
                    self,
                    "Файл данных не найден",
                    f"Файл данных не найден:\n{resolved}\n\n"
                    "Указать другое расположение?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    new_path, _ = QFileDialog.getOpenFileName(
                        self, f"Найти файл данных", "",
                        "CSV / Excel (*.csv *.txt *.xls *.xlsx);;Все файлы (*)",
                    )
                    if new_path:
                        resolved = new_path
                    else:
                        continue
                else:
                    continue
            try:
                df_i = load_file(resolved)
                if validate(df_i).is_valid:
                    loaded_dfs.append(df_i)
                    loaded_paths.append(resolved)
            except Exception:
                pass

        # Merge all loaded dataframes
        if loaded_dfs:
            if len(loaded_dfs) == 1:
                self.df = loaded_dfs[0]
            else:
                combined = pd.concat(loaded_dfs, ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=[COL_WELL, COL_DATE], keep="first"
                )
                self.df = recompute_derived(combined)
            self._source_files = loaded_paths
        else:
            self.df = None
            self._source_files = []

        # ── Restore project state ───────────────────────
        self._scenarios                = scenarios
        self._active_scenario_idx      = 0
        self._project_name             = project.get("project_name", "")
        self._current_save_path        = fcst_path
        self._stoiip                   = project.get("stoiip", 0.0)
        self._hcpv                     = project.get("hcpv", 0.0)
        self._well_analysis_scenarios  = project.get("well_analysis_scenarios", [])

        # Load scenario 0 into working buffers (no commit needed — nothing to commit)
        self._load_scenario_into_buffers(0)
        self.data_panel.lbl_filter.setText("")  # clear any stale filter label

        n_rows        = len(self.df) if self.df is not None else 0
        missing_files = len(src_paths) - len(loaded_paths)
        data_note = (
            f", {n_rows} строк данных"
            if n_rows else
            " (данные не загружены)"
        )
        if missing_files:
            data_note += f" — {missing_files} файл(ов) не найдено"
        n_sc = len(scenarios)
        total_results = sum(len(s.results) for s in scenarios)
        self.data_panel.lbl_info.setText(
            f"Проект: {n_sc} сцен., {len(sc0_wells)} скв. (sc. 1){data_note}"
        )

        self.status.showMessage(
            f"Проект загружен: {n_sc} сценариев, {total_results} рез-в."
            + (f", {n_rows} стр. данных" if n_rows else ""), 6000
        )

    def _on_save(self) -> None:
        """Save to the current project file (silent); open dialog if no file set."""
        has_data = bool(self._saved_results) or bool(self._scenarios)
        if not has_data:
            self.status.showMessage("Нет данных для сохранения", 3000)
            return
        if self._current_save_path:
            self._do_save(self._current_save_path)
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        """Always open Save-As dialog and update the current save path."""
        has_data = bool(self._saved_results) or bool(self._scenarios)
        if not has_data:
            self.status.showMessage("Нет данных для сохранения", 3000)
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект как", "",
            "Forecast file (*.fcst);;All files (*)"
        )
        if not path:
            return
        if self._do_save(path):
            self._current_save_path = path

    def _do_save(self, path: str) -> bool:
        """Commit working state, then write all scenarios to *path*."""
        from src.export.exporter import save_fcst_file
        # Commit working buffers into the active scenario before writing
        self._commit_active_scenario()
        # Ensure there is at least one scenario to save
        if not self._scenarios:
            self._scenarios = [
                ForecastScenario(
                    name="Сценарий 1",
                    wells=list(self._selected_wells),
                    results=dict(self._saved_results),
                )
            ]
        try:
            save_fcst_file(
                path,
                self._scenarios,
                self._source_files,
                project_name=self._project_name,
                stoiip=self._stoiip,
                hcpv=self._hcpv,
                well_analysis_scenarios=self._well_analysis_scenarios,
            )
            self.status.showMessage(f"Проект сохранён: {path}", 5000)
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {exc}")
            return False

    # ── Forecast Inspector ──────────────────────────────────────────────

    def _on_forecast_inspector(self) -> None:
        """Открыть Инспектор прогнозов (non-modal, persistent reference)."""
        if not self._scenarios:
            self.status.showMessage("Сначала загрузите данные или откройте проект", 3000)
            return

        # Commit working state so the inspector sees up-to-date scenario data
        self._commit_active_scenario()

        from src.ui.forecast_inspector_dialog import ForecastInspectorDialog

        if self._inspector_dlg is None:
            # First open: create, connect signals, keep reference
            self._inspector_dlg = ForecastInspectorDialog(
                self._scenarios,
                active_idx=self._active_scenario_idx,
                parent=self,
            )
            self._inspector_dlg.scenario_activated.connect(self._on_inspector_activate)
            self._inspector_dlg.finished.connect(self._on_inspector_finished)
        else:
            # Already exists: refresh with current data and bring to front
            self._inspector_dlg._scenarios  = list(self._scenarios)
            self._inspector_dlg._active_idx = self._active_scenario_idx
            self._inspector_dlg._refresh_list()

        self._inspector_dlg.show()
        self._inspector_dlg.raise_()
        self._inspector_dlg.activateWindow()

    def _on_inspector_activate(self, idx: int) -> None:
        """Handle scenario activation emitted by the inspector dialog."""
        if self._inspector_dlg is None:
            return
        # Apply any structural changes (create/delete) from the dialog first
        new_scenarios = self._inspector_dlg.result_scenarios()
        if new_scenarios:
            self._scenarios = new_scenarios
        # Clamp idx in case a deletion shifted the list
        idx = max(0, min(idx, len(self._scenarios) - 1))
        self._commit_active_scenario()
        self._load_scenario_into_buffers(idx)
        # Update the active indicator in the dialog
        self._inspector_dlg.refresh_active(idx)
        self.status.showMessage(f"Активирован сценарий: {self._scenarios[idx].name}", 3000)

    def _on_inspector_finished(self) -> None:
        """Apply the inspector's scenario list after the dialog is closed."""
        if self._inspector_dlg is None:
            return
        new_scenarios = self._inspector_dlg.result_scenarios()
        new_active    = self._inspector_dlg.result_active_idx()
        if not new_scenarios:
            return
        old_active    = self._active_scenario_idx
        self._scenarios = new_scenarios
        # Clamp active index in case scenarios were deleted
        new_active = max(0, min(new_active, len(self._scenarios) - 1))
        if new_active != old_active:
            self._load_scenario_into_buffers(new_active)
        else:
            self._update_window_title()

    # ── Data helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _avg_last(arr: np.ndarray, n: int) -> tuple[float, int]:
        """Return (mean_of_last_n_positive_values, actual_count_used).

        Uses only positive values in the tail so that shut-in months
        (zero production) do not artificially depress the average.
        Falls back to the last non-positive value if no positives exist.
        """
        if len(arr) == 0:
            return 0.0, 0
        tail = arr[-n:]                     # at most n elements from the end
        pos  = tail[tail > 0]
        if len(pos) == 0:
            return float(tail[-1]), len(tail)
        return float(pos.mean()), len(tail)

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
