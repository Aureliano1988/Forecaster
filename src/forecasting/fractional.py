"""Fractional flow forecasting methods.

x = cumulative oil (Qo), y = water cut (fw, 0–1).
Goal: determine Qo at target fw or extrapolate fw trend.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from src.forecasting.base import ForecastMethod


# ── Model functions ──────────────────────────────────────────────────────────

def _logistic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Logistic curve: fw = c / (1 + exp(-(a·x + b)))"""
    z = a * x + b
    return c / (1.0 + np.exp(-z))


# ─────────────────────────────────────────────────────────────────────────────
class WaterCutVsCumOil(ForecastMethod):
    """Logistic fit of fw = f(Qo).

    fw = c / (1 + exp(-(a·Qo + b)))
    """

    def __init__(self):
        self.a: float = 0.0
        self.b: float = 0.0
        self.c: float = 1.0

    def get_name(self) -> str:
        return "Обводнённость от накопленной нефти (логистическая)"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = (x > 0) & np.isfinite(y)
        x, y = x[mask], y[mask]
        try:
            popt, _ = curve_fit(
                _logistic, x, y,
                p0=[1e-4, -1.0, 1.0],
                bounds=([-np.inf, -np.inf, 0.5], [np.inf, np.inf, 1.0]),
                maxfev=10000,
            )
            self.a, self.b, self.c = (
                float(popt[0]), float(popt[1]), float(popt[2])
            )
        except RuntimeError:
            # Fallback: linear on raw data
            coeffs = np.polyfit(x, y, 1)
            self.a, self.b, self.c = float(coeffs[0]), float(coeffs[1]), 1.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _logistic(x, self.a, self.b, self.c)

    def get_parameters(self) -> dict:
        return {"a": self.a, "b": self.b, "c": self.c}


# ─────────────────────────────────────────────────────────────────────────────
class BuckleyLeverettSemiLog(ForecastMethod):
    """Semi-log method: ln(1 − fw) = a + b · Qo

    Linear fit on selected range; extrapolate to economic limit.
    """

    def __init__(self):
        self.a: float = 0.0
        self.b: float = 0.0

    def get_name(self) -> str:
        return "Баклей-Леверетт (полулогарифмический)"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = (x > 0) & (y > 0) & (y < 1.0)
        x, y = x[mask], y[mask]
        Y = np.log(1.0 - y)
        coeffs = np.polyfit(x, Y, 1)
        self.b = float(coeffs[0])
        self.a = float(coeffs[1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted water cut fw."""
        ln_one_minus_fw = self.a + self.b * x
        fw = 1.0 - np.exp(ln_one_minus_fw)
        return np.clip(fw, 0.0, 1.0)

    def get_parameters(self) -> dict:
        return {"a": self.a, "b": self.b}


# ── Registry ─────────────────────────────────────────────────────────────────
FRACTIONAL_METHODS: list[type[ForecastMethod]] = [
    WaterCutVsCumOil,
    BuckleyLeverettSemiLog,
]
