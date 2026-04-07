# Displacement Forecaster

Desktop application for oil production forecasting based on historical production data.

## Overview

Displacement Forecaster is a Python-based desktop tool designed for petroleum engineers to analyze well production history and generate production forecasts using established reservoir engineering methods.

## Key Features

- **Data Import**: Load monthly production data (oil, gas, water) by wells from CSV and Excel files.
- **Multiple Forecasting Methods**:
  - **Displacement Characteristics** — forecast oil recovery based on cumulative production relationships (Nazarov-Sipachev, Sazonov, Maksimov, cumulative oil vs. cumulative liquid).
  - **Decline Curve Analysis (DCA)** — Arps decline models (exponential, hyperbolic, harmonic) for rate-time and rate-cumulative analysis.
  - **Fractional Flow Analysis** — water cut vs. recovery factor relationships based on Buckley-Leverett theory.
- **Interactive Plotting**: Visualize historical data and forecasts with interactive charts.
- **Trend Selection**: Manually select data ranges for trend fitting and adjust forecast parameters.
- **Export**: Save forecast results and plots for reporting.

## Tech Stack

- **Language**: Python 3.10+
- **GUI Framework**: PySide6 (Qt for Python)
- **Data Processing**: Pandas, NumPy, SciPy
- **Plotting**: Matplotlib (embedded in Qt)
- **File I/O**: openpyxl (Excel), built-in csv

## Project Structure

```
Displacement_forecaster/
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── README.md
├── PLAN.md
├── src/
│   ├── __init__.py
│   ├── app.py               # Main application window
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py        # CSV/Excel import logic
│   │   ├── models.py        # Data models (Well, ProductionRecord)
│   │   └── validation.py    # Input data validation
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract forecasting interface
│   │   ├── displacement.py  # Displacement characteristic methods
│   │   ├── dca.py           # Decline curve analysis (Arps)
│   │   └── fractional.py    # Fractional flow analysis
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py   # Main window layout
│   │   ├── data_panel.py    # Data loading and well selection panel
│   │   ├── method_panel.py  # Forecasting method selection panel
│   │   ├── plot_widget.py   # Matplotlib canvas widget
│   │   └── settings_dialog.py
│   └── export/
│       ├── __init__.py
│       └── exporter.py      # Export results to CSV/Excel/PNG
└── tests/
    ├── __init__.py
    ├── test_loader.py
    ├── test_displacement.py
    ├── test_dca.py
    └── test_fractional.py
```

## Getting Started

```bash
pip install -r requirements.txt
python main.py
```

## Input Data Format

The application accepts CSV (semicolon- or comma-delimited) and Excel files with Russian-language headers as exported from standard MER (monthly exploitation report) systems.

**Expected columns (Russian headers, auto-mapped on load):**

- `имя скважины` — well name
- `дата(дд.мм.гггг)` — date (DD.MM.YYYY)
- `пласт` — formation
- `характер работы` — work type (НЕФ = oil, НАГ = injection)
- `состояние` — well status (РАБ., ЛИК, КОНС, etc.)
- `способ эксплуатации` — exploitation method
- `причина простоя` — downtime reason
- `время работы, ч` — operating hours
- `время накопления, ч` — accumulation hours
- `время простоя, ч` — downtime hours
- `нефть, т` — monthly oil production (tonnes)
- `вода, т/Закачка, Водозабор, м3` — water / injection (t or m³)
- `газ, м3` — gas production (m³)
- `газ из ГШ, м3` — gas-cap gas (m³)
- `Конденсат, т` — condensate (tonnes)
- `доп.параметр` — additional parameter

Liquid and cumulative values are computed automatically.

## License

Internal use only.
