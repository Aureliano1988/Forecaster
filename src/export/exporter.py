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
