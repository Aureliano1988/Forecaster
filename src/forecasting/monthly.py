"""Build month-by-month physical forecast series.

Driving assumption for all methods:
    monthly liquid production = ql_last = const (last historical month)

Stopping conditions (whichever comes first):
    - Forecast horizon reached (max_months)
    - Monthly oil rate <= 0
    - WOR (qw/qo) >= wor_limit
"""

from __future__ import annotations

import numpy as np

from src.data.models import ForecastSeries
from src.forecasting.base import ForecastMethod


# ── Displacement characteristics ─────────────────────────────────────────────

def build_displacement_forecast(
    method,
    Qo0: float,
    Ql0: float,
    Qw0: float,
    ql_last: float,
    max_months: int,
    wor_limit: float = 99.0,
) -> ForecastSeries:
    """Monthly forecast for LinearDisplacement methods.

    Each step: Ql_next = Ql_prev + ql_last, then Qo_next from method.compute_Qo().
    Monthly oil = Qo_next - Qo_prev; water = ql_last - oil.
    """
    series = ForecastSeries()
    if ql_last <= 0 or max_months <= 0:
        return series

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


# ── Decline Curve Analysis ────────────────────────────────────────────────────

def build_dca_forecast(
    method: ForecastMethod,
    x_last: float,
    ql_last: float,
    max_months: int,
    wor_limit: float = 99.0,
) -> ForecastSeries:
    """Monthly forecast for Arps DCA methods.

    qo[i] = method.predict(x_last + i)  (model drives oil rate)
    ql    = ql_last = const
    qw    = ql_last - qo  (capped to >= 0)
    """
    series = ForecastSeries()
    if ql_last <= 0 or max_months <= 0:
        return series

    fc_Qo = fc_Qw = fc_Ql = 0.0

    for i in range(1, max_months + 1):
        qo = float(method.predict(np.array([x_last + i]))[0])
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
    ql_last: float,
    max_months: int,
    wor_limit: float = 99.0,
) -> ForecastSeries:
    """Monthly forecast for fractional-flow methods.

    fw[i] = method.predict(Qo_cumulative)  → water cut (0–1)
    qo    = ql_last * (1 - fw)
    qw    = ql_last * fw
    """
    series = ForecastSeries()
    if ql_last <= 0 or max_months <= 0:
        return series

    Qo_cum = float(Qo0)
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
