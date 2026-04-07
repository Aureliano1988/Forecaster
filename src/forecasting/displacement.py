"""Displacement characteristic forecasting methods.

All methods perform a linear fit on specifically transformed X–Y coordinates.
Each method defines ``prepare_xy()`` which converts raw production arrays
into the plotting / fitting coordinate system.

Naming convention (following user spec):
  Qi — cumulative i-fluid production  (i: o=oil, l=liquid, w=water, g=gas)
  qi — monthly i-fluid production
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from src.forecasting.base import ForecastMethod


# ─────────────────────────────────────────────────────────────────────────────
class LinearDisplacement(ForecastMethod):
    """Base class — linear fit Y = a + b·X on transformed coordinates."""

    x_label: str = ""
    y_label: str = ""

    def __init__(self):
        self.a: float = 0.0
        self.b: float = 0.0

    @staticmethod
    @abstractmethod
    def prepare_xy(
        Qo: np.ndarray,
        Ql: np.ndarray,
        Qw: np.ndarray,
        qo: np.ndarray,
        ql: np.ndarray,
        qw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) arrays from production data."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 2:
            return
        coeffs = np.polyfit(x, y, 1)
        self.b = float(coeffs[0])
        self.a = float(coeffs[1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.a + self.b * x

    def get_parameters(self) -> dict:
        return {"a": self.a, "b": self.b}


# ── 1. Камбаров ──────────────────────────────────────────────────────────────
class Kambarov(LinearDisplacement):
    """X = Ql,  Y = Ql·Qo"""

    x_label = "Ql, т"
    y_label = "Ql · Qo"

    def get_name(self) -> str:
        return "Камбаров"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Ql > 0) & (Qo > 0)
        return Ql[mask], (Ql * Qo)[mask]


# ── 2. Пирвердян ─────────────────────────────────────────────────────────────
class Pirverdyan(LinearDisplacement):
    """X = 1/√Ql,  Y = Qo"""

    x_label = "1 / √Ql"
    y_label = "Qo, т"

    def get_name(self) -> str:
        return "Пирвердян"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = Ql > 0
        return 1.0 / np.sqrt(Ql[mask]), Qo[mask]


# ── 3. Назаров ────────────────────────────────────────────────────────────────
class Nazarov(LinearDisplacement):
    """X = Qw,  Y = Qw/Qo"""

    x_label = "Qw, т"
    y_label = "Qw / Qo"

    def get_name(self) -> str:
        return "Назаров"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Qw > 0) & (Qo > 0)
        return Qw[mask], (Qw / Qo)[mask]


# ── 4. Говоров ────────────────────────────────────────────────────────────────
class Govorov(LinearDisplacement):
    """X = ln(Ql),  Y = ln(Qo)"""

    x_label = "ln(Ql)"
    y_label = "ln(Qo)"

    def get_name(self) -> str:
        return "Говоров"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Ql > 0) & (Qo > 0)
        return np.log(Ql[mask]), np.log(Qo[mask])


# ── 5. Гусейнов ──────────────────────────────────────────────────────────────
class Guseinov(LinearDisplacement):
    """X = 1/Ql,  Y = Qo"""

    x_label = "1 / Ql"
    y_label = "Qo, т"

    def get_name(self) -> str:
        return "Гусейнов"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = Ql > 0
        return 1.0 / Ql[mask], Qo[mask]


# ── 6. Мовмыга ───────────────────────────────────────────────────────────────
class Movmyga(LinearDisplacement):
    """X = qo/ql (monthly),  Y = Qo"""

    x_label = "qo / ql"
    y_label = "Qo, т"

    def get_name(self) -> str:
        return "Мовмыга"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = ql > 0
        return (qo / ql)[mask], Qo[mask]


# ── 7. Варукшин ──────────────────────────────────────────────────────────────
class Varukshin(LinearDisplacement):
    """X = ln(qo/ql),  Y = Ql"""

    x_label = "ln(qo / ql)"
    y_label = "Ql, т"

    def get_name(self) -> str:
        return "Варукшин"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        ratio = np.where(ql > 0, qo / ql, 0.0)
        mask = ratio > 0
        return np.log(ratio[mask]), Ql[mask]


# ── 8. ВНО (WOR) ─────────────────────────────────────────────────────────────
class WOR(LinearDisplacement):
    """X = Qo,  Y = ln(qw/qo)"""

    x_label = "Qo, т"
    y_label = "ln(qw / qo)"

    def get_name(self) -> str:
        return "ВНО (WOR)"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        ratio = np.where(qo > 0, qw / qo, 0.0)
        mask = ratio > 0
        return Qo[mask], np.log(ratio[mask])


# ── 9. Сазонов ────────────────────────────────────────────────────────────────
class Sazonov(LinearDisplacement):
    """X = Qo,  Y = ln(Ql)"""

    x_label = "Qo, т"
    y_label = "ln(Ql)"

    def get_name(self) -> str:
        return "Сазонов"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Qo > 0) & (Ql > 0)
        return Qo[mask], np.log(Ql[mask])


# ── 10. Максимов ──────────────────────────────────────────────────────────────
class Maksimov(LinearDisplacement):
    """X = Qo,  Y = ln(Qw)"""

    x_label = "Qo, т"
    y_label = "ln(Qw)"

    def get_name(self) -> str:
        return "Максимов"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Qo > 0) & (Qw > 0)
        return Qo[mask], np.log(Qw[mask])


# ── 11. IFP ───────────────────────────────────────────────────────────────────
class IFP(LinearDisplacement):
    """X = Qo,  Y = ln(Qw/Qo)"""

    x_label = "Qo, т"
    y_label = "ln(Qw / Qo)"

    def get_name(self) -> str:
        return "IFP"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Qo > 0) & (Qw > 0)
        return Qo[mask], np.log((Qw / Qo)[mask])


# ── Registry ─────────────────────────────────────────────────────────────────
DISPLACEMENT_METHODS: list[type[ForecastMethod]] = [
    Kambarov,
    Pirverdyan,
    Nazarov,
    Govorov,
    Guseinov,
    Movmyga,
    Varukshin,
    WOR,
    Sazonov,
    Maksimov,
    IFP,
]
