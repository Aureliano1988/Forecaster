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

    def compute_Qo(self, Qo_prev: float, Ql_next: float, ql_last: float) -> float:  # noqa: N802
        """Return cumulative Qo at Ql_next given previous cumulative state.

        Assumes monthly liquid = ql_last = const, so Ql advances by ql_last
        each step.  Each subclass implements the inverse of its own
        coordinate transformation.
        """
        raise NotImplementedError(f"{type(self).__name__}.compute_Qo")


# ── 1. Камбаров ────────────────────────────────────────────────────────────────────────
class Kambarov(LinearDisplacement):
    """X = Ql,  Y = Ql·Qo   →   Qo = (a + b·Ql) / Ql"""

    x_label = "Ql, т"
    y_label = "Ql · Qo"

    def get_name(self) -> str:
        return "Камбаров"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Ql > 0) & (Qo > 0)
        return Ql[mask], (Ql * Qo)[mask]

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        if Ql_next <= 0:
            return Qo_prev
        return float((self.a + self.b * Ql_next) / Ql_next)


# ── 2. Пирвердян ─────────────────────────────────────────────────────────────────────────
class Pirverdyan(LinearDisplacement):
    """X = 1/√Ql,  Y = Qo   →   Qo = a + b/√Ql"""

    x_label = "1 / √Ql"
    y_label = "Qo, т"

    def get_name(self) -> str:
        return "Пирвердян"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = Ql > 0
        return 1.0 / np.sqrt(Ql[mask]), Qo[mask]

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        if Ql_next <= 0:
            return Qo_prev
        return float(self.a + self.b / np.sqrt(Ql_next))


# ── 3. Назаров ────────────────────────────────────────────────────────────────────────────
class Nazarov(LinearDisplacement):
    """X = Qw,  Y = Qw/Qo   →   solve Ql = Qo + Qw"""

    x_label = "Qw, т"
    y_label = "Qw / Qo"

    def get_name(self) -> str:
        return "Назаров"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Qw > 0) & (Qo > 0)
        return Qw[mask], (Qw / Qo)[mask]

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        # predict(Qw) = a + b*Qw = Qw/Qo  →  Qo = Qw/(a + b*Qw)
        # Ql = Qo + Qw  →  f(Qw) = Qw/(a+b*Qw) + Qw - Ql_next = 0
        from scipy.optimize import brentq
        a, b = self.a, self.b

        def f(Qw):
            denom = a + b * Qw
            if denom <= 1e-12:
                return 1e10
            return Qw / denom + Qw - Ql_next

        try:
            hi = Ql_next * 0.9999
            if f(0.0) * f(hi) >= 0:
                return Qo_prev
            Qw_sol = brentq(f, 0.0, hi, maxiter=200)
            return max(float(Qo_prev), float(Ql_next - Qw_sol))
        except ValueError:
            return Qo_prev


# ── 4. Говоров ────────────────────────────────────────────────────────────────────────────
class Govorov(LinearDisplacement):
    """X = ln(Ql),  Y = ln(Qo)   →   Qo = exp(a + b·ln(Ql))"""

    x_label = "ln(Ql)"
    y_label = "ln(Qo)"

    def get_name(self) -> str:
        return "Говоров"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Ql > 0) & (Qo > 0)
        return np.log(Ql[mask]), np.log(Qo[mask])

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        if Ql_next <= 0:
            return Qo_prev
        return float(np.exp(np.clip(self.a + self.b * np.log(Ql_next), -300, 300)))


# ── 5. Гусейнов ──────────────────────────────────────────────────────────────────────────
class Guseinov(LinearDisplacement):
    """X = 1/Ql,  Y = Qo   →   Qo = a + b/Ql"""

    x_label = "1 / Ql"
    y_label = "Qo, т"

    def get_name(self) -> str:
        return "Гусейнов"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = Ql > 0
        return 1.0 / Ql[mask], Qo[mask]

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        if Ql_next <= 0:
            return Qo_prev
        return float(self.a + self.b / Ql_next)


# ── 6. Мовмыга ───────────────────────────────────────────────────────────────────────────
class Movmyga(LinearDisplacement):
    """X = qo/ql (monthly),  Y = Qo

    Qo = a + b·(qo/ql)  →  solve for qo given Qo_prev:
    qo_next = ql_last · (Qo_prev − a) / (b − ql_last)
    """

    x_label = "qo / ql"
    y_label = "Qo, т"

    def get_name(self) -> str:
        return "Мовмыга"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = ql > 0
        return (qo / ql)[mask], Qo[mask]

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        denom = self.b - ql_last
        if abs(denom) < 1e-12:
            return Qo_prev
        qo_next = ql_last * (Qo_prev - self.a) / denom
        return float(Qo_prev + max(0.0, qo_next))


# ── 7. Варукшин ────────────────────────────────────────────────────────────────────────────
class Varukshin(LinearDisplacement):
    """X = ln(qo/ql),  Y = Ql

    Ql = a + b·ln(qo/ql)  →  qo/ql = exp((Ql−a)/b)
    qo_next = ql_last · exp((Ql_next − a) / b)
    """

    x_label = "ln(qo / ql)"
    y_label = "Ql, т"

    def get_name(self) -> str:
        return "Варукшин"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        ratio = np.where(ql > 0, qo / ql, 0.0)
        mask = ratio > 0
        return np.log(ratio[mask]), Ql[mask]

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        if abs(self.b) < 1e-12:
            return Qo_prev
        exponent = np.clip((Ql_next - self.a) / self.b, -100, 100)
        qo_next = ql_last * float(np.exp(exponent))
        return float(Qo_prev + max(0.0, qo_next))


# ── 8. ВНО (WOR) ───────────────────────────────────────────────────────────────────────────
class WOR(LinearDisplacement):
    """X = Qo (cum.),  Y = ln(qw/qo) (monthly)

    monthly WOR = exp(a + b·Qo_prev)
    qo_next = ql_last / (1 + WOR)
    """

    x_label = "Qo, т"
    y_label = "ln(qw / qo)"

    def get_name(self) -> str:
        return "ВНО (WOR)"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        ratio = np.where(qo > 0, qw / qo, 0.0)
        mask = ratio > 0
        return Qo[mask], np.log(ratio[mask])

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        wor = float(np.exp(np.clip(self.a + self.b * Qo_prev, -100, 100)))
        qo_next = ql_last / (1.0 + wor)
        return float(Qo_prev + max(0.0, qo_next))


# ── 9. Сазонов ─────────────────────────────────────────────────────────────────────────────────────
class Sazonov(LinearDisplacement):
    """X = Qo,  Y = ln(Ql)   →   Qo = (ln(Ql) − a) / b"""

    x_label = "Qo, т"
    y_label = "ln(Ql)"

    def get_name(self) -> str:
        return "Сазонов"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Qo > 0) & (Ql > 0)
        return Qo[mask], np.log(Ql[mask])

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        if abs(self.b) < 1e-12 or Ql_next <= 0:
            return Qo_prev
        Qo_next = (np.log(Ql_next) - self.a) / self.b
        return max(float(Qo_prev), float(Qo_next))


# ── 10. Максимов ──────────────────────────────────────────────────────────────────────────────────────
class Maksimov(LinearDisplacement):
    """X = Qo,  Y = ln(Qw)   →   Ql = Qo + exp(a+b·Qo), solve with brentq"""

    x_label = "Qo, т"
    y_label = "ln(Qw)"

    def get_name(self) -> str:
        return "Максимов"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Qo > 0) & (Qw > 0)
        return Qo[mask], np.log(Qw[mask])

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        from scipy.optimize import brentq
        a, b = self.a, self.b

        def f(Qo):
            return Qo + float(np.exp(np.clip(a + b * Qo, -300, 300))) - Ql_next

        try:
            if f(0.0) >= 0:      # exp(a) >= Ql_next — no valid root
                return Qo_prev
            return float(brentq(f, 0.0, Ql_next * 0.9999, maxiter=200))
        except ValueError:
            return Qo_prev


# ── 11. IFP ───────────────────────────────────────────────────────────────────────────────────────────────
class IFP(LinearDisplacement):
    """X = Qo,  Y = ln(Qw/Qo)   →   Ql = Qo·(1+exp(a+b·Qo)), solve with brentq"""

    x_label = "Qo, т"
    y_label = "ln(Qw / Qo)"

    def get_name(self) -> str:
        return "IFP"

    @staticmethod
    def prepare_xy(Qo, Ql, Qw, qo, ql, qw):
        mask = (Qo > 0) & (Qw > 0)
        return Qo[mask], np.log((Qw / Qo)[mask])

    def compute_Qo(self, Qo_prev, Ql_next, ql_last):
        from scipy.optimize import brentq
        a, b = self.a, self.b

        def f(Qo):
            return Qo * (1.0 + float(np.exp(np.clip(a + b * Qo, -300, 300)))) - Ql_next

        try:
            # f(0) = -Ql_next < 0, f(Ql_next) > 0  →  bracket exists
            return float(brentq(f, 0.0, Ql_next * 0.9999, maxiter=200))
        except ValueError:
            return Qo_prev


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
