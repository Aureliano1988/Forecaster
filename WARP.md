# Displacement Forecaster — Project Description

## Purpose
Desktop application for petroleum engineers to load well production history and generate oil production forecasts using standard reservoir engineering methods.

## Tech Stack
- **Language**: Python 3.14 (CPython)
- **GUI**: PySide6 (Qt for Python)
- **Data processing**: pandas, NumPy, SciPy
- **Plotting**: Matplotlib embedded in Qt via `FigureCanvasQTAgg`
- **File I/O**: openpyxl (Excel), built-in CSV

## Project Structure
```
forecaster/
├── main.py                        # Entry point — launches QApplication + MainWindow
├── requirements.txt               # PySide6, pandas, numpy, scipy, matplotlib, openpyxl
├── src/
│   ├── data/
│   │   ├── models.py              # Column constants, HEADER_MAP, ForecastResult dataclass
│   │   ├── loader.py              # load_file() — CSV/Excel → normalised DataFrame
│   │   └── validation.py         # validate() → ValidationResult (errors + warnings)
│   ├── forecasting/
│   │   ├── base.py                # Abstract ForecastMethod (fit / predict / get_parameters / r_squared)
│   │   ├── displacement.py        # 11 linearised displacement-characteristic methods
│   │   ├── dca.py                 # Arps decline: Exponential, Hyperbolic, Harmonic
│   │   └── fractional.py         # Fractional flow: Logistic fw(Qo), Buckley-Leverett semi-log
│   ├── ui/
│   │   ├── main_window.py         # Orchestrator: load → plot → fit → forecast → export
│   │   ├── data_panel.py          # Left panel: file loader + multi-select well list
│   │   ├── method_panel.py        # Right panel: family/method dropdowns, horizon spinbox, result display
│   │   └── plot_widget.py         # Matplotlib canvas + NavigationToolbar + SpanSelector
│   └── export/
│       └── exporter.py            # export_forecast_csv() and export_plot()
└── tests/
    └── __init__.py
```

## Key Concepts

### Data layer
- Input: CSV (semicolon/comma/tab) or Excel files with **Russian-language headers** from MER (monthly exploitation report) systems.
- `HEADER_MAP` in `models.py` maps Russian column names to internal language-neutral constants (`COL_OIL`, `COL_WELL`, etc.).
- `load_file()` auto-detects encoding (utf-8-sig → cp1251 → latin-1) and delimiter, then computes derived columns: `liquid_t`, `cum_oil_t`, `cum_water_t`, `cum_liquid_t`, `cum_gas_m3`, `water_cut`.
- All filtering for oil-producing rows uses `work_type == "НЕФ"`.

### Forecasting engine
All methods implement the `ForecastMethod` ABC:
- `fit(x, y)` — fits model to selected data window
- `predict(x)` — evaluates model
- `get_parameters()` — returns dict of fitted coefficients
- `r_squared(x, y)` — computes R² on fit

**Displacement characteristics** (`displacement.py`): 11 methods (Камбаров, Пирвердян, Назаров, Говоров, Гусейнов, Мовмыга, Варукшин, ВНО/WOR, Сазонов, Максимов, IFP). Each defines `prepare_xy(Qo, Ql, Qw, qo, ql, qw)` to transform cumulative production arrays into linearised X–Y coordinates, then `LinearDisplacement.fit()` runs `np.polyfit(..., deg=1)`.

**Decline Curve Analysis** (`dca.py`): Arps exponential (b=0), hyperbolic (0<b<1), harmonic (b=1). Parameters `qi`, `Di`, `b` are fitted via `scipy.optimize.curve_fit` with fallback to log-linear regression.

**Fractional flow** (`fractional.py`): Logistic fit of `fw = f(Qo)` and Buckley-Leverett semi-log `ln(1 − fw) = a + b·Qo`.

### UI workflow
1. Load file → wells populate left panel.
2. Select well(s) → historical scatter/line appears in centre plot.
3. Choose method family and variant in right panel.
4. Drag on plot to select fitting range (blue highlight via `SpanSelector`).
5. Optionally activate **eraser** mode to mark exclusion zones (red shading).
6. Click **"Построить тренд"** → trend overlaid in red, R² and parameters shown.
7. Set forecast horizon → click **"Рассчитать прогноз"** → forecast curve in green dashes.
8. Export plot (PNG/SVG) or forecast table (CSV/Excel) via File menu or buttons.

### Export
- `export_forecast_csv()` saves `x`, `forecast`, `method` columns to CSV (`;`-delimited, UTF-8-BOM) or Excel.
- `export_plot()` saves the matplotlib figure to PNG or SVG at 150 dpi.

## Running the Application
```bash
pip install -r requirements.txt
python main.py
```

## Input File Requirements
CSV or Excel with Russian headers matching `HEADER_MAP` in `src/data/models.py`. The minimum required columns after mapping are `well`, `date`, and `oil_t`. Dates must be in `DD.MM.YYYY` format. Decimal separator can be comma or period — both are handled automatically.

## Current Limitations / Notes
- No unit tests are implemented yet beyond the `tests/__init__.py` stub.
- No packaging script (PyInstaller/cx_Freeze) exists yet.
- `README.md` lists some files (`app.py`, `settings_dialog.py`, test files) that do not exist — the actual structure is as documented above.
