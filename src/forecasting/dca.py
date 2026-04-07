"""Decline Curve Analysis — Arps models.

Input: x = time (months from start), y = oil rate (t/month).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from src.forecasting.base import ForecastMethod


# ── Model functions ──────────────────────────────────────────────────────────

def _exponential(t: np.ndarray, qi: float, di: float) -> np.ndarray:
    return qi * np.exp(-di * t)


def _hyperbolic(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    base = 1.0 + b * di * t
    base = np.where(base > 0, base, 1e-12)
    return qi / np.power(base, 1.0 / b)


def _harmonic(t: np.ndarray, qi: float, di: float) -> np.ndarray:
    return qi / (1.0 + di * t)


# ─────────────────────────────────────────────────────────────────────────────
class ExponentialDecline(ForecastMethod):
    """q(t) = qi · exp(−Di · t)"""

    def __init__(self):
        self.qi: float = 0.0
        self.di: float = 0.0

    def get_name(self) -> str:
        return "Экспоненциальное падение (Arps b=0)"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = y > 0
        x, y = x[mask], y[mask]
        try:
            popt, _ = curve_fit(
                _exponential, x, y,
                p0=[y[0], 0.01],
                bounds=([0, 0], [np.inf, 10]),
                maxfev=5000,
            )
            self.qi, self.di = float(popt[0]), float(popt[1])
        except RuntimeError:
            # Fallback: simple log-linear regression
            coeffs = np.polyfit(x, np.log(np.clip(y, 1e-12, None)), 1)
            self.di = float(-coeffs[0])
            self.qi = float(np.exp(coeffs[1]))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _exponential(x, self.qi, self.di)

    def get_parameters(self) -> dict:
        return {"qi": self.qi, "Di": self.di, "b": 0}


# ─────────────────────────────────────────────────────────────────────────────
class HyperbolicDecline(ForecastMethod):
    """q(t) = qi / (1 + b·Di·t)^(1/b)"""

    def __init__(self):
        self.qi: float = 0.0
        self.di: float = 0.0
        self.b: float = 0.5

    def get_name(self) -> str:
        return "Гиперболическое падение (Arps)"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = y > 0
        x, y = x[mask], y[mask]
        try:
            popt, _ = curve_fit(
                _hyperbolic, x, y,
                p0=[y[0], 0.01, 0.5],
                bounds=([0, 0, 0.01], [np.inf, 10, 0.99]),
                maxfev=10000,
            )
            self.qi, self.di, self.b = (
                float(popt[0]), float(popt[1]), float(popt[2])
            )
        except RuntimeError:
            # Fallback to exponential
            exp = ExponentialDecline()
            exp.fit(x, y)
            self.qi, self.di, self.b = exp.qi, exp.di, 0.01

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _hyperbolic(x, self.qi, self.di, self.b)

    def get_parameters(self) -> dict:
        return {"qi": self.qi, "Di": self.di, "b": self.b}


# ─────────────────────────────────────────────────────────────────────────────
class HarmonicDecline(ForecastMethod):
    """q(t) = qi / (1 + Di·t)   (Arps b=1)"""

    def __init__(self):
        self.qi: float = 0.0
        self.di: float = 0.0

    def get_name(self) -> str:
        return "Гармоническое падение (Arps b=1)"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = y > 0
        x, y = x[mask], y[mask]
        try:
            popt, _ = curve_fit(
                _harmonic, x, y,
                p0=[y[0], 0.01],
                bounds=([0, 0], [np.inf, 10]),
                maxfev=5000,
            )
            self.qi, self.di = float(popt[0]), float(popt[1])
        except RuntimeError:
            self.qi = float(y[0]) if len(y) else 1.0
            self.di = 0.01

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _harmonic(x, self.qi, self.di)

    def get_parameters(self) -> dict:
        return {"qi": self.qi, "Di": self.di, "b": 1}


# ── Registry ─────────────────────────────────────────────────────────────────
DCA_METHODS: list[type[ForecastMethod]] = [
    ExponentialDecline,
    HyperbolicDecline,
    HarmonicDecline,
]
