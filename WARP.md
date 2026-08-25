# Displacement Forecaster — Project Rules

## Purpose
Desktop application for petroleum engineers to load well production history and generate oil/gas production forecasts using standard reservoir engineering methods. Supports multi-scenario projects saved as `.fcst` JSON files.

## Tech Stack
- **Language**: Python 3.14 (CPython)
- **GUI**: PySide6 (Qt for Python)
- **Data processing**: pandas, NumPy, SciPy
- **Plotting**: Matplotlib embedded in Qt via `FigureCanvasQTAgg`
- **File I/O**: openpyxl (Excel), built-in CSV
- **Packaging**: PyInstaller (`main.spec`)

## Project Structure
```
forecaster/
├── main.py                              # Entry point — QApplication + MainWindow
├── requirements.txt                     # PySide6, pandas, numpy, scipy, matplotlib, openpyxl
├── main.spec                            # PyInstaller spec
├── PLAN.md                              # Original phased implementation plan (historical; largely superseded)
├── src/
│   ├── data/
│   │   ├── models.py                    # Column constants, HEADER_MAP, dataclasses:
│   │   │                                #   ForecastResult, ForecastSeries, SavedMethodResult,
│   │   │                                #   ForecastScenario, WellAnalysisScenario
│   │   ├── loader.py                    # load_file(), read_raw(), apply_manual_mapping(),
│   │   │                                #   recompute_derived() — CSV/Excel → normalised DataFrame
│   │   └── validation.py               # validate() → ValidationResult (errors + warnings)
│   ├── forecasting/
│   │   ├── base.py                      # Abstract ForecastMethod ABC (fit/predict/get_parameters/r_squared)
│   │   ├── displacement.py              # 11 LinearDisplacement methods (Камбаров → IFP)
│   │   ├── dca.py                       # Arps decline: Exponential, Hyperbolic, Harmonic
│   │   ├── fractional.py               # Fractional flow: Logistic fw(Qo), Buckley-Leverett
│   │   └── monthly.py                  # Month-by-month forecast builders:
│   │                                    #   build_displacement_forecast(), build_dca_forecast(),
│   │                                    #   build_fractional_forecast(), anchor helpers, time/Ql shifts
│   ├── ui/
│   │   ├── main_window.py               # Orchestrator: data → plot → fit → forecast → export → save/load
│   │   ├── data_panel.py                # Left panel: file loader, well list, filter, active-wells toggle
│   │   ├── method_panel.py              # Right panel: family/method dropdowns, horizon, WOR limit,
│   │   │                                #   min oil, n_avg, action buttons, result display
│   │   ├── plot_widget.py               # Matplotlib canvas + NavigationToolbar + LassoSelector
│   │   ├── data_import_dialog.py        # Preview raw data and assign columns to parameters
│   │   ├── forecast_inspector_dialog.py # Manage multiple named forecast scenarios
│   │   ├── forecast_plots_dialog.py     # Multi-method interactive forecast comparison charts
│   │   ├── summary_dialog.py            # Tabular summary of all built method results
│   │   ├── export_data_dialog.py        # Cross-scenario metrics table builder (copy-to-clipboard)
│   │   ├── trend_param_dialog.py        # Floating dialog for numeric trend parameter editing
│   │   ├── reservoir_data_dialog.py     # Enter STOIIP and HCPV reservoir parameters
│   │   ├── well_alignment_dialog.py     # Adjusted production: monthly vs months-since-first-production
│   │   ├── well_analysis_scenario_dialog.py  # Create/rename/duplicate/delete well-analysis scenarios
│   │   ├── well_vintage_dialog.py       # Stacked area chart grouped by first-production year
│   │   ├── well_criteria_dialog.py      # Select wells by computed criteria (first-prod year, avg KE, etc.)
│   │   ├── chan_plot_dialog.py          # Chan diagnostic plot: WOR / WOR' vs elapsed time (log-log)
│   │   ├── production_distribution_dialog.py  # Histogram of per-well metrics (cum. oil, rates, watercut)
│   │   ├── object_info_dialog.py        # Dashboard: production plots + 12 KPI metric cards for selection
│   │   ├── hover_tooltip.py             # install_hover_tooltip() — shared matplotlib hover-label helper
│   │   └── legend_helper.py             # fit_legend() — shared compact/draggable legend builder
│   └── export/
│       └── exporter.py                  # export_forecast_csv(), export_plot(),
│                                        #   save_fcst_file(), load_fcst_file() — project persistence
└── tests/
    └── __init__.py                      # No tests implemented yet
```

## Key Concepts

### Data layer
- Input: CSV (semicolon/comma/tab) or Excel files with **Russian-language headers** from MER (monthly exploitation report) systems.
- `HEADER_MAP` in `models.py` maps Russian column names → internal language-neutral constants (`COL_OIL`, `COL_WELL`, etc.).
- `load_file()` auto-detects encoding (utf-8-sig → cp1251 → latin-1) and delimiter, then computes derived columns: `liquid_t`, `cum_oil_t`, `cum_water_t`, `cum_liquid_t`, `cum_gas_m3`, `water_cut`.
- `read_raw()` + `apply_manual_mapping()` support manual column assignment via `data_import_dialog.py`.
- `COL_WATER_DUAL` is auto-split into production water (`work_type == "НЕФ"`) vs injection (`work_type == "НАГ"`) based on oil > 0.
- Oil-producing rows are filtered by `work_type == "НЕФ"`.

### Forecasting engine
All methods implement the `ForecastMethod` ABC (`base.py`):
- `fit(x, y)` — fits model to selected data window
- `predict(x)` — evaluates model
- `get_parameters()` — returns dict of fitted coefficients
- `r_squared(x, y)` — computes R² on fit

**Displacement characteristics** (`displacement.py`): 11 `LinearDisplacement` subclasses — Камбаров, Пирвердян, Назаров, Говоров, Гусейнов, Мовмыга, Варукшин, ВНФ(WOR), Сазонов, Максимов, IFP. Each defines `prepare_xy(Qo, Ql, Qw, qo, ql, qw)` for coordinate transformation and `compute_Qo()` for monthly stepping. Fit via `np.polyfit(..., deg=1)`.

**Decline Curve Analysis** (`dca.py`): Arps exponential (b=0), hyperbolic (0<b<1), harmonic (b=1). Parameters `qi`, `Di`, `b` fitted via `scipy.optimize.curve_fit` with fallback to log-linear regression.

**Fractional flow** (`fractional.py`): Logistic `fw = c/(1+exp(-(a·Qo+b)))` and Buckley-Leverett semi-log `ln(1−fw) = a + b·Qo`.

**Monthly forecast builders** (`monthly.py`): Convert trend model → physical month-by-month `ForecastSeries` (qo, qw, ql, Qo, Qw, Ql, WOR). Driving assumption: `ql_last = const`. Stop conditions: horizon reached, monthly oil ≤ 0, WOR ≥ limit, or min oil threshold. Anchoring helpers (`anchor_displacement_method`, `dca_time_shift`, `fractional_qo_anchor`, `displacement_ql_shift`) ensure the forecast starts from the last historical point.

### Scenario system
- A project contains multiple `ForecastScenario` objects, each with a name, well selection, fitted results (`dict[str, SavedMethodResult]`), and optional STOIIP/HCPV overrides.
- Scenarios are managed via the **Forecast Inspector** dialog.
- Separate `WellAnalysisScenario` objects store well-alignment analysis with P10/P50/P90 percentile profiles.

### Project persistence (`.fcst`)
- `save_fcst_file()` / `load_fcst_file()` in `exporter.py` serialise all scenarios, well-analysis scenarios, source file paths, and reservoir parameters to JSON (v2.0 format with backward compat for v1.0/v1.1).
- Source data file paths are preserved for auto-reload on project open.

### UI workflow
1. Load file(s) → wells populate left panel (multi-file supported).
2. Select well(s) manually, via a saved well list, or via **well criteria** (`well_criteria_dialog.py` — filter by first-production year, average KE, months of production, etc.).
3. Historical scatter/line appears in centre plot; choose method family and variant in right panel.
4. Draw lasso on plot to select fitting range (blue polygon) or use eraser for exclusion zones (red).
5. Click **"Построить прогноз"** → trend + forecast overlaid, parameters shown.
6. Optionally use **"Редактировать тренд"** for drag-handle slope/intercept adjustment or **"Автоподбор"** / **"Автоподбор всех"** for automated fitting.
7. Use **Forecast Inspector** (Ctrl+I) to manage scenarios, **Сводка прогнозов** for summary table, **Графики прогнозов** for comparison charts, **Экспорт данных** (`export_data_dialog.py`) for a cross-scenario metrics table.
8. Enter reservoir data (STOIIP, HCPV) for RF/HCPVI computation.
9. Use standalone diagnostic/analysis windows as needed: **Карточка объекта** (`object_info_dialog.py` — KPI dashboard), **График Чена** (`chan_plot_dialog.py` — WOR/WOR' diagnostic), **Распределение** (`production_distribution_dialog.py` — per-well metric histograms), **Выравнивание скважин** (`well_alignment_dialog.py`), **Группировка по годам** (`well_vintage_dialog.py`).
10. Export plot (PNG/SVG) or forecast table (CSV/Excel) via File menu.
11. Save/load project as `.fcst` file (Ctrl+S / Ctrl+O).

### Shared UI helpers
- `hover_tooltip.py` — `install_hover_tooltip()` attaches a cached, hover-driven label tooltip to any matplotlib canvas; used across the diagnostic dialogs.
- `legend_helper.py` — `fit_legend()` builds a deduplicated, size-adaptive, draggable legend shared by multi-series plots.

### Export
- `export_forecast_csv()` saves `x`, `forecast`, `method` columns to CSV (`;`-delimited, UTF-8-BOM) or Excel.
- `export_plot()` saves the matplotlib figure to PNG or SVG at 150 dpi.
- `export_data_dialog.py` builds a configurable cross-scenario summary table (RF, HCPVI, WOR, reserves, etc.) for copy-to-clipboard export.

## Conventions
- UI labels, menu items, and user-facing strings are in **Russian**.
- Internal column names and code identifiers are in **English**.
- Gas-phase forecasting restricts method families to DCA only (via `_GAS_FAMILIES` in `method_panel.py`).
- Methods are registered in module-level lists: `DISPLACEMENT_METHODS`, `DCA_METHODS`, `FRACTIONAL_METHODS`.

## Running the Application
```
pip install -r requirements.txt
python main.py
```

## Input File Requirements
CSV or Excel with Russian headers matching `HEADER_MAP` in `src/data/models.py`. Minimum required columns after mapping: `well`, `date`, and `oil_t`. Dates in `DD.MM.YYYY` format. Comma or period decimal separators are handled automatically.

## Current Limitations
- No unit tests beyond the `tests/__init__.py` stub.
- `README.md` project structure is outdated — the actual structure is as documented above.
