"""Export forecast data and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def export_forecast_csv(
    x: np.ndarray,
    y: np.ndarray,
    method_name: str,
    path: str | Path,
) -> None:
    """Save forecast x/y arrays to CSV or Excel."""
    path = Path(path)
    df = pd.DataFrame({"x": x, "forecast": y, "method": method_name})

    if path.suffix.lower() in (".xls", ".xlsx"):
        df.to_excel(path, index=False, engine="openpyxl")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig", sep=";")


def export_plot(fig: Figure, path: str | Path, dpi: int = 150) -> None:
    """Save the matplotlib figure to an image file."""
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")


def save_fcst_file(
    path: str | Path,
    saved_results: dict,
    wells: list[str],
    source_file: str = "",
) -> None:
    """Save built trends and forecasts to a .fcst JSON text file.

    The file stores, for every fitted technique:
    - method family and name
    - fitted parameters
    - trend and forecast line arrays (method-space coordinates)
    - month-by-month physical forecast table (qo, qw, ql, Qo, Qw, Ql, WOR)
    """
    import json
    from datetime import datetime, timezone

    records: dict = {}
    for key, result in saved_results.items():
        family, method_name = key.split("|", 1)
        entry: dict = {
            "family": family,
            "method_name": result.method_name,
            "parameters": result.parameters,
            "x_trend": result.x_trend,
            "y_trend": result.y_trend,
            "x_forecast": result.x_forecast,
            "y_forecast": result.y_forecast,
        }
        m = result.monthly
        if m is not None and m.duration > 0:
            entry["monthly_forecast"] = {
                "duration": m.duration,
                "remain_reserves_t": round(m.remain_reserves, 2),
                "wor_last": round(m.wor_last, 4),
                "qo":  [round(v, 4) for v in m.qo],
                "qw":  [round(v, 4) for v in m.qw],
                "ql":  [round(v, 4) for v in m.ql],
                "Qo":  [round(v, 2) for v in m.Qo],
                "Qw":  [round(v, 2) for v in m.Qw],
                "Ql":  [round(v, 2) for v in m.Ql],
                "WOR": [round(v, 4) for v in m.WOR],
            }
        records[key] = entry

    data = {
        "version": "1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "wells": wells,
        "source_file": source_file,
        "results": records,
    }
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
