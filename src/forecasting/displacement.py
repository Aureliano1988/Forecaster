"""Displacement characteristic forecasting methods.

All methods work on cumulative data:
  x = cumulative liquid (Ql)
  y = cumulative oil   (Qo)
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import polynomial as P

from src.forecasting.base import ForecastMethod


# ─────────────────────────────────────────────────────────────────────────────
class SimpleCumulative(ForecastMethod):
    """Polynomial fit of Qo = f(Ql)."""

    def __init__(self, degree: int = 3):
        self.degree = degree
        self._coeffs: np.ndarray | None = None

    def get_name(self) -> str:
        return f"Кумулятивная зависимость (степень {self.degree})"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._coeffs = P.polyfit(x, y, self.degree)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self._coeffs is not None, "Call fit() first"
        return P.polyval(x, self._coeffs)

    def get_parameters(self) -> dict:
        return {"degree": self.degree, "coeffs": self._coeffs.tolist()}


# ─────────────────────────────────────────────────────────────────────────────
class NazarovSipachev(ForecastMethod):
    """Linearised form:  Ql / Qo = a + b · Ql

    Fit a, b by OLS → Qo = Ql / (a + b · Ql)
    """

    def __init__(self):
        self.a: float = 0.0
        self.b: float = 0.0

    def get_name(self) -> str:
        return "Назаров-Сипачёв"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = y > 0
        x, y = x[mask], y[mask]
        Y = x / y  # Ql / Qo
        X = x       # Ql
        coeffs = np.polyfit(X, Y, 1)  # Y = b*X + a
        self.b = float(coeffs[0])
        self.a = float(coeffs[1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        denom = self.a + self.b * x
        denom = np.where(denom > 0, denom, np.nan)
        return x / denom

    def get_parameters(self) -> dict:
        return {"a": self.a, "b": self.b}


# ─────────────────────────────────────────────────────────────────────────────
class Sazonov(ForecastMethod):
    """Linearised form:  Ql / Qo = a + b · ln(Ql)

    Qo = Ql / (a + b · ln(Ql))
    """

    def __init__(self):
        self.a: float = 0.0
        self.b: float = 0.0

    def get_name(self) -> str:
        return "Сазонов"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = (y > 0) & (x > 0)
        x, y = x[mask], y[mask]
        Y = x / y
        X = np.log(x)
        coeffs = np.polyfit(X, Y, 1)
        self.b = float(coeffs[0])
        self.a = float(coeffs[1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        ln_x = np.log(np.where(x > 0, x, 1.0))
        denom = self.a + self.b * ln_x
        denom = np.where(denom > 0, denom, np.nan)
        return x / denom

    def get_parameters(self) -> dict:
        return {"a": self.a, "b": self.b}


# ─────────────────────────────────────────────────────────────────────────────
class Maksimov(ForecastMethod):
    """Log-log linear: ln(Qo) = a + b · ln(Ql)

    Qo = exp(a) · Ql^b
    """

    def __init__(self):
        self.a: float = 0.0
        self.b: float = 0.0

    def get_name(self) -> str:
        return "Максимов"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = (x > 0) & (y > 0)
        x, y = x[mask], y[mask]
        coeffs = np.polyfit(np.log(x), np.log(y), 1)
        self.b = float(coeffs[0])
        self.a = float(coeffs[1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.a) * np.power(np.where(x > 0, x, 1.0), self.b)

    def get_parameters(self) -> dict:
        return {"a": self.a, "b": self.b, "A": float(np.exp(self.a))}


# ── Registry ─────────────────────────────────────────────────────────────────
DISPLACEMENT_METHODS: list[type[ForecastMethod]] = [
    SimpleCumulative,
    NazarovSipachev,
    Sazonov,
    Maksimov,
]
