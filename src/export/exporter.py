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
            "params_text": result.params_text,
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


def load_fcst_file(path: str | Path) -> dict:
    """Load a .fcst project file.

    Returns a dict with keys:
      ``wells``        — list of well names that were selected
      ``source_file``  — path of the original production data file
      ``results``      — ``dict[str, SavedMethodResult]`` keyed by family|method
    """
    import json
    from src.data.models import ForecastSeries, SavedMethodResult

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    results: dict = {}
    for key, entry in data.get("results", {}).items():
        # Reconstruct ForecastSeries
        monthly: ForecastSeries | None = None
        mf = entry.get("monthly_forecast")
        if mf:
            monthly = ForecastSeries(
                qo=mf.get("qo", []),
                qw=mf.get("qw", []),
                ql=mf.get("ql", []),
                Qo=mf.get("Qo", []),
                Qw=mf.get("Qw", []),
                Ql=mf.get("Ql", []),
                WOR=mf.get("WOR", []),
            )

        # Restore params_text — or rebuild a basic version if not stored
        params_text = entry.get("params_text", "")
        if not params_text:
            mn = entry.get("method_name", "?")
            lines = [f"Метод: {mn}"]
            for k, v in entry.get("parameters", {}).items():
                lines.append(f"  {k} = {float(v):.6g}")
            if monthly and monthly.duration > 0:
                sw = "WOR" if monthly.WOR and monthly.WOR[-1] >= 99 else "горизонт"
                lines += [
                    f"{'\u2500'*22}",
                    f"Прогноз (стоп: {sw}):",
                    f"  Горизонт: {monthly.duration} мес.",
                    f"  Ост. запасы: {monthly.remain_reserves:,.0f} т",
                    f"  WOR (посл.): {monthly.wor_last:.2f}",
                ]
            params_text = "\n".join(lines)

        results[key] = SavedMethodResult(
            method_name=entry.get("method_name", ""),
            params_text=params_text,
            parameters=entry.get("parameters", {}),
            x_trend=entry.get("x_trend", []),
            y_trend=entry.get("y_trend", []),
            x_forecast=entry.get("x_forecast", []),
            y_forecast=entry.get("y_forecast", []),
            monthly=monthly,
        )

    return {
        "wells": data.get("wells", []),
        "source_file": data.get("source_file", ""),
        "results": results,
    }
