"""Data models and column constants for production data."""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Standard internal column names (language-neutral) ────────────────────────
COL_WELL = "well"
COL_DATE = "date"
COL_FORMATION = "formation"
COL_WORK_TYPE = "work_type"
COL_STATUS = "status"
COL_EXPLOIT_METHOD = "exploit_method"
COL_DOWNTIME_REASON = "downtime_reason"
COL_HOURS_WORK = "hours_work"
COL_HOURS_ACCUM = "hours_accum"
COL_HOURS_DOWN = "hours_down"
COL_OIL = "oil_t"
COL_WATER = "water_t"
COL_GAS = "gas_m3"
COL_GAS_CAP = "gas_cap_m3"
COL_CONDENSATE = "condensate_t"
COL_EXTRA = "extra_param"

# Derived / computed columns
COL_LIQUID = "liquid_t"       # oil + water
COL_CUM_OIL = "cum_oil_t"
COL_CUM_WATER = "cum_water_t"
COL_CUM_LIQUID = "cum_liquid_t"
COL_CUM_GAS = "cum_gas_m3"
COL_WATER_CUT = "water_cut"   # water / liquid

# Mapping from Russian header (lowercase, stripped) → internal name.
# The loader normalises headers before lookup.
HEADER_MAP: dict[str, str] = {
    "имя скважины":                          COL_WELL,
    "дата(дд.мм.гггг)":                     COL_DATE,
    "пласт":                                 COL_FORMATION,
    "характер работы":                       COL_WORK_TYPE,
    "состояние":                             COL_STATUS,
    "способ эксплуатации":                   COL_EXPLOIT_METHOD,
    "причина простоя":                       COL_DOWNTIME_REASON,
    "время работы, ч":                       COL_HOURS_WORK,
    "время накопления, ч":                   COL_HOURS_ACCUM,
    "время простоя, ч":                      COL_HOURS_DOWN,
    "нефть, т":                              COL_OIL,
    "вода, т/закачка, водозабор, м3":        COL_WATER,
    "газ, м3":                               COL_GAS,
    "газ из гш, м3":                         COL_GAS_CAP,
    "конденсат, т":                          COL_CONDENSATE,
    "доп.параметр":                          COL_EXTRA,
}

# Numeric columns that must be coerced to float
NUMERIC_COLS = [
    COL_HOURS_WORK, COL_HOURS_ACCUM, COL_HOURS_DOWN,
    COL_OIL, COL_WATER, COL_GAS, COL_GAS_CAP, COL_CONDENSATE, COL_EXTRA,
]

# Work-type constants
WORK_TYPE_OIL = "НЕФ"
WORK_TYPE_INJ = "НАГ"


@dataclass
class ForecastResult:
    """Container for a single forecast run."""

    method_name: str
    x_hist: list[float] = field(default_factory=list)
    y_hist: list[float] = field(default_factory=list)
    x_fit: list[float] = field(default_factory=list)
    y_fit: list[float] = field(default_factory=list)
    x_forecast: list[float] = field(default_factory=list)
    y_forecast: list[float] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    r_squared: float | None = None


@dataclass
class ForecastSeries:
    """Month-by-month physical forecast output."""

    qo:  list[float] = field(default_factory=list)   # monthly oil, t
    qw:  list[float] = field(default_factory=list)   # monthly water, t
    ql:  list[float] = field(default_factory=list)   # monthly liquid, t
    Qo:  list[float] = field(default_factory=list)   # cumulative oil from forecast start, t
    Qw:  list[float] = field(default_factory=list)   # cumulative water from forecast start, t
    Ql:  list[float] = field(default_factory=list)   # cumulative liquid from forecast start, t
    WOR: list[float] = field(default_factory=list)   # monthly water-oil ratio qw/qo

    @property
    def duration(self) -> int:
        return len(self.qo)

    @property
    def remain_reserves(self) -> float:
        """Total forecasted oil production (t)."""
        return float(sum(self.qo))

    @property
    def wor_last(self) -> float:
        return float(self.WOR[-1]) if self.WOR else 0.0


@dataclass
class SavedMethodResult:
    """Persists trend + forecast for one technique across method switches."""

    method_name: str
    params_text: str                                   # formatted string shown in results panel
    parameters:  dict          = field(default_factory=dict)
    x_trend:     list[float]   = field(default_factory=list)
    y_trend:     list[float]   = field(default_factory=list)
    x_forecast:  list[float]   = field(default_factory=list)
    y_forecast:  list[float]   = field(default_factory=list)
    monthly:     ForecastSeries | None = None
