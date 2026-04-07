# Implementation Plan — Displacement Forecaster

## 1. Problem Statement

Petroleum engineers need a desktop tool to load well production history and generate oil production forecasts using standard reservoir engineering techniques: displacement characteristics, decline curve analysis, and fractional flow. The tool must support interactive trend selection and visualization.

## 2. Phases

### Phase 1 — Project Skeleton and Data Layer

**Goal**: Establish project structure, dependencies, and data import pipeline.

- Initialize Python project with `requirements.txt` (PySide6, pandas, numpy, scipy, matplotlib, openpyxl).
- Define data models:
  - `ProductionRecord`: date, oil, water, gas, liquid, days.
  - `Well`: name/id, list of `ProductionRecord`, computed cumulative columns.
  - `ProductionDataset`: collection of wells with aggregation helpers.
- Implement `loader.py`:
  - Read CSV and Excel files via pandas.
  - Auto-detect or map columns (well, date, oil, water, gas, liquid, days).
  - Compute cumulative values (cum_oil, cum_water, cum_liquid) if absent.
- Implement `validation.py`:
  - Check for required columns, missing values, non-negative production.
  - Return structured error/warning messages.
- Write unit tests for loader and validation.

### Phase 2 — Forecasting Engine

**Goal**: Implement the three forecasting method families behind a common interface.

#### 2a. Base Interface

- Define abstract `ForecastMethod` class:
  - `fit(x, y, selection_range)` — fit trend to selected data.
  - `predict(x_future)` — generate forecast values.
  - `get_parameters()` — return fitted parameters dict.
  - `get_name()` — human-readable method name.

#### 2b. Displacement Characteristics (`displacement.py`)

Relationship: cumulative oil (`Qo`) vs. cumulative liquid (`Ql`) or cumulative water (`Qw`).

Methods to implement:
- **Simple cumulative plot** — `Qo = f(Ql)` with polynomial/logarithmic trend extrapolation.
- **Nazarov-Sipachev** — linearized form: `Ql/Qo = a + b·Ql`. Fit `a`, `b` by linear regression on selected range; forecast `Qo` from predicted `Ql`.
- **Sazonov** — linearized form: `Ql/Qo = a + b·ln(Ql)`. Same workflow.
- **Maksimov** — `ln(Qo) = a + b·ln(Ql)`. Log-log linear regression.

Each method:
1. Takes cumulative production arrays.
2. Fits linear regression on the user-selected interval.
3. Extrapolates to a user-specified future `Ql` range.
4. Returns forecast `Qo` array and R² metric.

#### 2c. Decline Curve Analysis (`dca.py`)

Arps decline equation: `q(t) = qi / (1 + b·Di·t)^(1/b)`

Models:
- **Exponential** (b = 0): `q = qi · exp(-Di·t)`
- **Hyperbolic** (0 < b < 1): general Arps form
- **Harmonic** (b = 1): `q = qi / (1 + Di·t)`

Implementation:
1. Input: time series of production rate (`q` vs `t`).
2. User selects decline period on the rate-time plot.
3. Fit `qi`, `Di`, `b` via `scipy.optimize.curve_fit` on the selected interval.
4. Extrapolate rate for future time steps.
5. Integrate rate to get cumulative production forecast.
6. Also support **rate vs. cumulative** diagnostic plot (`q` vs `Np`).

#### 2d. Fractional Flow (`fractional.py`)

Water cut (`fw`) vs. cumulative oil or recovery factor.

Methods:
- **fw vs. Recovery Factor** — plot `fw = f(RF)` or `fw = f(Qo)`, fit trend (logistic, polynomial), extrapolate to `fw → 1.0`.
- **Buckley-Leverett style** — `1 - fw` vs `Qo` on semi-log scale; linear fit on selected range; extrapolate to economic limit.

Each method:
1. Compute water cut: `fw = Qw / (Qo + Qw)` (volumetric) from monthly or cumulative data.
2. User selects fitting interval.
3. Fit selected model (linear on transformed axes or curve_fit).
4. Forecast `Qo` at target `fw` or over time.

#### 2e. Unit Tests

- Test each method with synthetic data where the analytical answer is known.
- Test edge cases (zero production months, single-well, multi-well aggregation).

### Phase 3 — GUI Shell

**Goal**: Build the main application window with navigation and layout.

- `main.py` — entry point, launch `QApplication`.
- `main_window.py` — top-level window with:
  - Menu bar (File → Open, Export; Help → About).
  - Left panel: data/well selector.
  - Center: plot area (Matplotlib canvas via `FigureCanvasQTAgg`).
  - Right panel: method selection and parameters.
  - Bottom: status bar with messages/warnings.
- `data_panel.py`:
  - "Load Data" button → file dialog (CSV/XLSX).
  - Well list (checkboxes to select wells or "All wells" aggregate).
  - Data preview table (first N rows).
- `method_panel.py`:
  - Dropdown: Displacement / DCA / Fractional Flow.
  - Sub-dropdown for specific method variant.
  - Parameter inputs (e.g., forecast horizon, economic limit).
  - "Fit Trend" and "Calculate Forecast" buttons.

### Phase 4 — Interactive Plotting and Trend Selection

**Goal**: Let users visualize data, select fitting ranges, and see forecasts overlaid.

- `plot_widget.py`:
  - Embed `matplotlib.figure.Figure` in Qt widget.
  - Plot historical data as scatter/line.
  - **Range selector**: use `matplotlib.widgets.SpanSelector` or custom click-drag to select X-axis range for trend fitting.
  - After fitting: overlay trend line and forecast curve in a different color/style.
  - Legend, axis labels, title auto-generated from method context.
  - Toolbar for zoom, pan, reset.
- Workflow:
  1. User loads data → wells appear in left panel.
  2. User selects well(s) and method → historical plot appears.
  3. User drags on plot to select fitting range → highlighted region.
  4. User clicks "Fit Trend" → trend line overlaid, parameters displayed.
  5. User sets forecast horizon → clicks "Calculate Forecast" → forecast curve appended.

### Phase 5 — Export and Polish

**Goal**: Export results and refine UX.

- `exporter.py`:
  - Export forecast table (date, predicted oil, water, liquid) to CSV/Excel.
  - Export plot to PNG/SVG.
- Settings dialog:
  - Default units (tonnes/m³ vs barrels).
  - Plot style preferences.
- Error handling:
  - Graceful messages for bad input files, failed curve fits, insufficient data.
- Tooltips and brief help text for each forecasting method.

### Phase 6 — Testing and Packaging

- Integration tests: load sample file → run forecast → verify output.
- Package with PyInstaller or cx_Freeze for standalone `.exe` distribution.
- Write user-facing quick-start guide.

## 3. Dependencies

- Python 3.10+
- PySide6
- pandas
- numpy
- scipy
- matplotlib
- openpyxl

## 4. Risks and Considerations

- **Column mapping**: Real production files vary widely in column naming; the loader must handle flexible mapping or provide a UI for manual mapping.
- **Curve fitting convergence**: Arps hyperbolic fit can fail with poor initial guesses; provide sensible defaults and fallback to exponential.
- **Large datasets**: Multi-field data with hundreds of wells; ensure aggregation and plotting remain responsive (consider lazy loading or background threads).
- **Unit consistency**: Oil in tonnes vs. barrels, liquid in m³ vs. barrels — must be explicit throughout.
