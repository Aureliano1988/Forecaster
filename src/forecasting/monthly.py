"""Build month-by-month physical forecast series.

Driving assumption for all methods:
    monthly liquid production = ql_last = const (last historical month)

Stopping conditions (whichever comes first):
    - Forecast horizon reached (max_months)
    - Monthly oil rate <= 0
    - WOR (qw/qo) >= wor_limit
"""

from __future__ import annotations

import copy

import numpy as np

from src.data.models import ForecastSeries
from src.forecasting.base import ForecastMethod


# ── Anchor helpers ─────────────────────────────────────────────────────────────────

def anchor_displacement_method(
    method,
    Qo0: float, Ql0: float, Qw0: float,
    qo_last_m: float, ql_last: float, qw_last_m: float,
):
    """Return a shallow copy of *method* with intercept `a` shifted so the
    fitted line passes through the last historical data point in the
    method's own coordinate space.

    Preserves the fitted slope `b`, ensuring the forecast starts from the
    correct cumulative position (Ql0, Qo0) with the right decline shape.
    """
    m = copy.copy(method)
    try:
        X_arr, Y_arr = m.prepare_xy(
            np.array([float(Qo0)]),
            np.array([float(Ql0)]),
            np.array([float(Qw0)]),
            np.array([float(qo_last_m)]),
            np.array([float(ql_last)]),
            np.array([float(qw_last_m)]),
        )
        if len(X_arr) > 0:
            m.a = float(Y_arr[0]) - m.b * float(X_arr[0])
    except Exception:
        pass  # can't anchor — use original parameters
    return m


def fractional_qo_anchor(method, fw_last: float, Qo0: float) -> float:
    """Find Qo_eff such that method.predict(Qo_eff) ≈ fw_last.

    Anchors the fractional-flow forecast so the first month's water cut
    equals the last historical value, giving:
        qo_step1 = ql_last * (1 − fw_last)  ≡  q_last_oil
    """
    if fw_last <= 0.0 or fw_last >= 1.0:
        return float(Qo0)

    from scipy.optimize import brentq

    def f(Qo: float) -> float:
        return float(np.clip(method.predict(np.array([Qo]))[0], 0.0, 1.0)) - fw_last

    lo = 0.0
    hi = max(float(Qo0) * 2.0, 1e6)
    try:
        if f(lo) * f(hi) >= 0:
            return float(Qo0)
        return float(brentq(f, lo, hi, maxiter=300))
    except Exception:
        return float(Qo0)


# ── Displacement characteristics ─────────────────────────────────────────────────────────

def build_displacement_forecast(
    method,
    Qo0: float,
    Ql0: float,
    Qw0: float,
    qo_last_monthly: float,
    qw_last_monthly: float,
    ql_last: float,
    max_months: int,
    wor_limit: float = 99.0,
) -> ForecastSeries:
    """Monthly forecast for LinearDisplacement methods.

    The model is anchored to (Ql0, Qo0) via an intercept shift before
    stepping forward.  Each step: Ql_next = Ql_prev + ql_last, then
    Qo_next from the anchored method.compute_Qo().
    """
    series = ForecastSeries()
    if ql_last <= 0 or max_months <= 0:
        return series

    # Shift intercept so model passes through last historical cumulative point
    method = anchor_displacement_method(
        method, Qo0, Ql0, Qw0, qo_last_monthly, ql_last, qw_last_monthly
    )

    Qo_cum = float(Qo0)
    Ql_cum = float(Ql0)
    fc_Qo = fc_Qw = fc_Ql = 0.0

    for _ in range(max_months):
        Ql_next = Ql_cum + ql_last
        try:
            Qo_next = float(method.compute_Qo(Qo_cum, Ql_next, ql_last))
        except Exception:
            break

        qo = Qo_next - Qo_cum
        if qo <= 0:
            break
        qo = min(qo, ql_last)      # can't exceed total liquid
        qw = max(0.0, ql_last - qo)
        wor = qw / qo               # qo > 0 guaranteed

        fc_Qo += qo
        fc_Qw += qw
        fc_Ql += ql_last

        series.qo.append(qo)
        series.qw.append(qw)
        series.ql.append(ql_last)
        series.Qo.append(fc_Qo)
        series.Qw.append(fc_Qw)
        series.Ql.append(fc_Ql)
        series.WOR.append(wor)

        Qo_cum = Qo_next
        Ql_cum = Ql_next

        if wor >= wor_limit:
            break

    return series


# ── Decline Curve Analysis ────────────────────────────────────────────────────────────────

def dca_time_shift(method: ForecastMethod, q0: float) -> float:
    """Find t_shift such that method.predict(t_shift) ≈ q0.

    Shifts the Arps curve in time so that the forecast starts at q0
    (last historical monthly oil rate) and declines at the fitted rate.
    All Arps models are monotonically decreasing in t, so brentq works.
    """
    if q0 <= 0:
        return 0.0

    q_at_0 = float(method.predict(np.array([0.0]))[0])
    if q0 >= q_at_0:
        # q0 ≥ model's own initial rate — anchor at t = 0
        return 0.0

    from scipy.optimize import brentq

    def f(t: float) -> float:
        return float(method.predict(np.array([t]))[0]) - q0

    try:
        # f(0) = q_at_0 - q0 > 0;  f(∞) → −q0 < 0  —  bracket is valid
        return float(brentq(f, 0.0, 1e6, maxiter=300))
    except ValueError:
        return 0.0


def build_dca_forecast(
    method: ForecastMethod,
    x_last: float,
    q_last: float,
    ql_last: float,
    max_months: int,
    wor_limit: float = 99.0,
) -> ForecastSeries:
    """Monthly forecast for Arps DCA methods.

    The forecast is anchored so that the first forecast month's oil rate
    equals q_last (last historical monthly oil production).  The decline
    curve is shifted in time until predict(t_shift) == q_last; subsequent
    months use predict(t_shift + 1), predict(t_shift + 2), …

    ql = ql_last = const (liquid rate assumption).
    """
    series = ForecastSeries()
    if ql_last <= 0 or max_months <= 0:
        return series

    t_shift = dca_time_shift(method, q_last)
    fc_Qo = fc_Qw = fc_Ql = 0.0

    for i in range(1, max_months + 1):
        qo = float(method.predict(np.array([t_shift + i]))[0])
        if qo <= 0:
            break
        qo = min(qo, ql_last)
        qw = max(0.0, ql_last - qo)
        wor = qw / qo

        fc_Qo += qo
        fc_Qw += qw
        fc_Ql += ql_last

        series.qo.append(qo)
        series.qw.append(qw)
        series.ql.append(ql_last)
        series.Qo.append(fc_Qo)
        series.Qw.append(fc_Qw)
        series.Ql.append(fc_Ql)
        series.WOR.append(wor)

        if wor >= wor_limit:
            break

    return series


# ── Fractional flow ───────────────────────────────────────────────────────────

def build_fractional_forecast(
    method: ForecastMethod,
    Qo0: float,
    fw_last: float,
    ql_last: float,
    max_months: int,
    wor_limit: float = 99.0,
) -> ForecastSeries:
    """Monthly forecast for fractional-flow methods.

    The forecast is anchored so the first month's water cut matches
    fw_last (last historical water cut), giving:
        qo_step1 = ql_last * (1 - fw_last) = q_last_oil

    fw[i] = method.predict(Qo_cumulative)  → water cut (0–1)
    """
    series = ForecastSeries()
    if ql_last <= 0 or max_months <= 0:
        return series

    # Shift starting Qo to where fw(Qo_eff) = fw_last
    Qo_cum = fractional_qo_anchor(method, fw_last, Qo0)
    fc_Qo = fc_Qw = fc_Ql = 0.0

    for _ in range(max_months):
        fw = float(np.clip(method.predict(np.array([Qo_cum]))[0], 0.0, 0.9999))
        qo = ql_last * (1.0 - fw)
        if qo <= 0:
            break
        qw = ql_last * fw
        wor = qw / qo

        fc_Qo += qo
        fc_Qw += qw
        fc_Ql += ql_last

        series.qo.append(qo)
        series.qw.append(qw)
        series.ql.append(ql_last)
        series.Qo.append(fc_Qo)
        series.Qw.append(fc_Qw)
        series.Ql.append(fc_Ql)
        series.WOR.append(wor)

        Qo_cum += qo

        if wor >= wor_limit:
            break

    return series
