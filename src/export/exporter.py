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


def _serialise_results(saved_results: dict) -> dict:
    """Convert a ``dict[str, SavedMethodResult]`` to a JSON-serialisable dict."""
    records: dict = {}
    for key, result in saved_results.items():
        entry: dict = {
            "family": key.split("|", 1)[0],
            "method_name": result.method_name,
            "params_text": result.params_text,
            "parameters": result.parameters,
            "qo_hist_last": result.qo_hist_last,
            "x_trend": result.x_trend,
            "y_trend": result.y_trend,
            "x_forecast": result.x_forecast,
            "y_forecast": result.y_forecast,
        }
        m = result.monthly
        if m is not None and m.duration > 0:
            entry["monthly_forecast"] = {
                "duration": m.duration,
                "stop_reason": m.stop_reason,
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
    return records


def _serialise_well_analysis_scenarios(well_analysis_scenarios: list) -> list:
    """Serialise a list of WellAnalysisScenario objects to plain dicts."""
    result = []
    for sc in well_analysis_scenarios:
        entry: dict = {
            "name":       sc.name,
            "wells":      sc.wells,
            "phase":      getattr(sc, "phase", "oil"),
            "excluded":   sc.excluded,   # [[well, iso_date], ...]
            "pct_months": sc.pct_months,
            "pct_data":   sc.pct_data,   # {"10": [...], "50": [...], "90": [...]}
            "pct_trends": sc.pct_trends, # {"10": [qi, Di] | null, ...}
        }
        result.append(entry)
    return result


def _deserialise_well_analysis_scenarios(raw: list) -> list:
    """Restore WellAnalysisScenario objects from a plain-dict list."""
    from src.data.models import WellAnalysisScenario
    scenarios = []
    for i, entry in enumerate(raw):
        sc = WellAnalysisScenario(
            name       = entry.get("name", f"Анализ {i + 1}"),
            wells      = entry.get("wells", []),
            phase      = entry.get("phase", "oil"),
            excluded   = entry.get("excluded", []),
            pct_months = entry.get("pct_months", []),
            pct_data   = entry.get("pct_data", {}),
            pct_trends = entry.get("pct_trends", {}),
        )
        scenarios.append(sc)
    return scenarios


def save_fcst_file(
    path: str | Path,
    scenarios,                                # list[ForecastScenario]
    source_files: list[str] | str = "",
    project_name: str = "",
    stoiip: float = 0.0,
    hcpv: float = 0.0,
    well_analysis_scenarios=None,             # list[WellAnalysisScenario] | None
) -> None:
    """Save all forecast scenarios to a .fcst v2.0 JSON file.

    Each scenario stores its name, selected wells, and all fitted method
    results.  Source data file paths are preserved so they can be reloaded
    automatically when the project is opened.
    """
    import json
    from datetime import datetime, timezone

    # Normalise to list (accept legacy callers passing a plain string)
    if isinstance(source_files, str):
        source_files = [source_files] if source_files else []

    serialised_scenarios = [
        {
            "name": sc.name,
            "wells": sc.wells,
            "stoiip": sc.stoiip,
            "hcpv":   sc.hcpv,
            "phase":  getattr(sc, "phase", "oil"),
            "results": _serialise_results(sc.results),
        }
        for sc in scenarios
    ]

    data = {
        "version": "2.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "project_name": project_name,
        "source_files": source_files,
        "stoiip": stoiip,
        "hcpv": hcpv,
        "scenarios": serialised_scenarios,
        "well_analysis_scenarios": _serialise_well_analysis_scenarios(
            well_analysis_scenarios or []
        ),
    }
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _deserialise_results(raw_results: dict) -> dict:
    """Convert raw JSON dict → ``dict[str, SavedMethodResult]``."""
    from src.data.models import ForecastSeries, SavedMethodResult

    results: dict = {}
    for key, entry in raw_results.items():
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
                stop_reason=mf.get("stop_reason", ""),
            )

        # Restore params_text — or rebuild a basic version if not stored
        params_text = entry.get("params_text", "")
        if not params_text:
            mn = entry.get("method_name", "?")
            lines = [f"Метод: {mn}"]
            for k, v in entry.get("parameters", {}).items():
                lines.append(f"  {k} = {float(v):.6g}")
            if monthly and monthly.duration > 0:
                sw = "ВНФ" if monthly.WOR and monthly.WOR[-1] >= 99 else "горизонт"
                lines += [
                    f"{'\u2500'*22}",
                    f"Прогноз (стоп: {sw}):",
                    f"  Горизонт: {monthly.duration} мес.",
                    f"  Ост. запасы: {monthly.remain_reserves:,.0f} т",
                    f"  ВНФ (посл.): {monthly.wor_last:.2f}",
                ]
            params_text = "\n".join(lines)

        results[key] = SavedMethodResult(
            method_name=entry.get("method_name", ""),
            params_text=params_text,
            parameters=entry.get("parameters", {}),
            qo_hist_last=entry.get("qo_hist_last", 0.0),
            x_trend=entry.get("x_trend", []),
            y_trend=entry.get("y_trend", []),
            x_forecast=entry.get("x_forecast", []),
            y_forecast=entry.get("y_forecast", []),
            monthly=monthly,
        )
    return results


def load_fcst_file(path: str | Path) -> dict:
    """Load a .fcst project file (v1.0 / v1.1 / v2.0).

    Returns a dict with keys:
      ``project_name``  — str
      ``source_files``  — list[str]
      ``scenarios``     — list[ForecastScenario]
    """
    import json
    from src.data.models import ForecastScenario

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # Support both v1.1 (source_files list) and v1.0 (source_file string)
    src_files: list[str] = data.get("source_files", [])
    if not src_files:
        legacy = data.get("source_file", "")
        src_files = [legacy] if legacy else []
    # Project-level stoiip/hcpv (used as fallback for scenarios that don't have their own)
    proj_stoiip = float(data.get("stoiip", 0.0))
    proj_hcpv   = float(data.get("hcpv",   0.0))

    # ── v2.0: scenarios array ───────────────────────────────────────────────
    if "scenarios" in data:
        scenarios = [
            ForecastScenario(
                name=sc.get("name", f"Сценарий {i + 1}"),
                wells=sc.get("wells", []),
                results=_deserialise_results(sc.get("results", {})),
                # Per-scenario values; fall back to project-level for old files
                stoiip=float(sc.get("stoiip", 0.0)) or proj_stoiip,
                hcpv=float(sc.get("hcpv",   0.0)) or proj_hcpv,
                phase=sc.get("phase", "oil"),
            )
            for i, sc in enumerate(data["scenarios"])
        ]
    else:
        # ── Backward compat: v1.0 / v1.1 single scenario ──────────────────
        scenarios = [
            ForecastScenario(
                name="Сценарий 1",
                wells=data.get("wells", []),
                results=_deserialise_results(data.get("results", {})),
                stoiip=proj_stoiip,
                hcpv=proj_hcpv,
            )
        ]

    raw_wa = data.get("well_analysis_scenarios", [])
    well_analysis_scenarios = _deserialise_well_analysis_scenarios(raw_wa)

    return {
        "project_name": data.get("project_name", ""),
        "source_files": src_files,
        "stoiip": proj_stoiip,
        "hcpv":   proj_hcpv,
        "scenarios": scenarios,
        "well_analysis_scenarios": well_analysis_scenarios,
    }
