"""Abstract base class for all forecasting methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.data.models import ForecastResult


class ForecastMethod(ABC):
    """Common interface for every forecasting technique."""

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable method name."""

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fit trend to *selected* data window."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted model at given x values."""

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return dict of fitted parameters."""

    # ── Convenience helpers ──────────────────────────────────────────────────

    def r_squared(self, x: np.ndarray, y: np.ndarray) -> float:
        """Coefficient of determination for the fit on (x, y)."""
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    def build_result(
        self,
        x_hist: np.ndarray,
        y_hist: np.ndarray,
        x_forecast: np.ndarray,
    ) -> ForecastResult:
        """Produce a full ForecastResult after fitting."""
        y_fit = self.predict(x_hist)
        y_forecast = self.predict(x_forecast)
        return ForecastResult(
            method_name=self.get_name(),
            x_hist=x_hist.tolist(),
            y_hist=y_hist.tolist(),
            x_fit=x_hist.tolist(),
            y_fit=y_fit.tolist(),
            x_forecast=x_forecast.tolist(),
            y_forecast=y_forecast.tolist(),
            parameters=self.get_parameters(),
            r_squared=self.r_squared(x_hist, y_hist),
        )
