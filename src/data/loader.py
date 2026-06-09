"""Load production data from CSV or Excel files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.models import (
    COL_CUM_GAS,
    COL_CUM_LIQUID,
    COL_CUM_OIL,
    COL_CUM_WATER,
    COL_DATE,
    COL_GAS,
    COL_HOURS_WORK,
    COL_LIQUID,
    COL_LIQUID_RATE,
    COL_OIL,
    COL_OIL_RATE,
    COL_WATER,
    COL_WATER_CUT,
    COL_WATER_DUAL,
    COL_WATER_INJ,
    COL_WATER_RATE,
    COL_WELL,
    COL_WORK_TYPE,
    HEADER_MAP,
    NUMERIC_COLS,
    WORK_TYPE_INJ,
    WORK_TYPE_OIL,
)

# Encodings to try in order
_ENCODINGS = ["utf-8-sig", "utf-8", "cp1251", "latin-1"]
# Common CSV delimiters
_DELIMITERS = [";", ",", "\t"]

def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file and return the raw DataFrame (no renaming).

    Headers are kept exactly as they appear in the file so the caller can
    present them to the user for manual column assignment.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        return _read_excel(path)
    elif suffix in (".csv", ".txt"):
        return _read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def apply_manual_mapping(
    raw_df: pd.DataFrame,
    col_mapping: dict[str, str],
) -> pd.DataFrame:
    """Apply a user-defined column mapping and process the DataFrame.

    ``col_mapping`` maps original column names \u2192 internal column names
    (empty string means \u201cdo not use\u201d).  After renaming, date parsing,
    numeric coercion and derived-column computation are applied exactly
    as in :func:`load_file`.
    """
    rename = {k: v for k, v in col_mapping.items() if v}
    df = raw_df.rename(columns=rename)
    # Drop columns that were not assigned
    keep = set(rename.values())
    df = df[[c for c in df.columns if c in keep]].copy()
    df = _parse_dates(df)
    df = _coerce_numerics(df)
    df = _convert_rates(df)      # daily rate → monthly production
    df = _apply_water_split(df)  # resolve WATER_DUAL / WATER_INJ sentinels
    df = _compute_derived(df)
    return df


def load_file(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file and return a normalised DataFrame.

    Steps:
    1. Read raw file (auto-detect encoding and delimiter for CSV).
    2. Map Russian headers \u2192 internal column names via HEADER_MAP.
    3. Parse date column.
    4. Coerce numeric columns.
    5. Compute derived columns (liquid, cumulatives, water-cut).
    """
    df = read_raw(path)
    df = _rename_columns(df)
    df = _parse_dates(df)
    df = _coerce_numerics(df)
    df = _convert_rates(df)      # daily rate → monthly production
    df = _apply_water_split(df)  # resolve WATER_DUAL / WATER_INJ sentinels
    df = _compute_derived(df)
    return df


# ── Public helpers

_DERIVED_COLS = [
    COL_LIQUID, COL_CUM_OIL, COL_CUM_WATER,
    COL_CUM_LIQUID, COL_CUM_GAS, COL_WATER_CUT,
]


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Drop existing derived columns and recompute them from base columns.

    Use this after concatenating two loaded DataFrames so that cumulative
    sums are recalculated over the full combined history.
    """
    df = df.drop(columns=[c for c in _DERIVED_COLS if c in df.columns])
    return _compute_derived(df.copy())


# ── Private helpers ──────────────────────────────────────────────────────


def _convert_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily-rate columns to monthly production when production columns are absent.

    Conversion factor: if hours_work is available, producing_days = hours_work / 24;
    otherwise assume 30.4375 days/month (КЭ = 1).

    Rate columns are consumed (dropped) after conversion.
    """
    import numpy as np

    _RATE_TO_PROD = [
        (COL_OIL_RATE,    COL_OIL),
        (COL_WATER_RATE,  COL_WATER),
        (COL_LIQUID_RATE, COL_LIQUID),
    ]

    converted_any = False
    for rate_col, prod_col in _RATE_TO_PROD:
        if rate_col not in df.columns:
            continue
        if prod_col in df.columns:
            # Production column already exists — don't overwrite; just drop rate
            df = df.drop(columns=[rate_col])
            continue
        # Compute monthly production = rate × producing_days
        rate = df[rate_col].values.astype(float)
        if COL_HOURS_WORK in df.columns:
            days = df[COL_HOURS_WORK].values.astype(float) / 24.0
            days = np.where(days > 0, days, 30.4375)
        else:
            days = np.full(len(rate), 30.4375)
        df[prod_col] = rate * days
        df = df.drop(columns=[rate_col])
        converted_any = True

    # If oil was converted from rate and work_type is absent, mark as oil-producing
    if converted_any and COL_WORK_TYPE not in df.columns:
        df[COL_WORK_TYPE] = WORK_TYPE_OIL

    return df


def _apply_water_split(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve the COL_WATER_DUAL and COL_WATER_INJ sentinel columns.

    COL_WATER_DUAL (“вода (добыча/закачка)”):
        Row-by-row logic based on oil production:
        - oil > 0  → producing well  → WORK_TYPE = НЕФ, water = produced water
        - oil == 0 → injection well  → WORK_TYPE = НАГ, water = injected water
        If WORK_TYPE is already set the existing value is preserved.

    COL_WATER_INJ (“закачка воды, только закачка”):
        All rows flagged as injection (WORK_TYPE = НАГ).

    Both sentinels are renamed to COL_WATER so the rest of the pipeline
    sees a unified water column.
    """
    if COL_WATER_DUAL not in df.columns and COL_WATER_INJ not in df.columns:
        return df

    # Ensure WORK_TYPE column exists
    if COL_WORK_TYPE not in df.columns:
        df = df.copy()
        df[COL_WORK_TYPE] = ""

    # ─ COL_WATER_DUAL: split by oil production ─────────────────────
    if COL_WATER_DUAL in df.columns:
        oil = df[COL_OIL].values if COL_OIL in df.columns else None
        no_type = df[COL_WORK_TYPE].astype(str).str.strip() == ""
        if oil is not None:
            import numpy as np
            prod_mask = pd.Series(oil, index=df.index) > 0
            df.loc[prod_mask & no_type, COL_WORK_TYPE] = WORK_TYPE_OIL
            df.loc[~prod_mask & no_type, COL_WORK_TYPE] = WORK_TYPE_INJ
        else:
            # No oil column — cannot determine type; default to production
            df.loc[no_type, COL_WORK_TYPE] = WORK_TYPE_OIL
        # Rename sentinel → COL_WATER (handle rare case where COL_WATER already exists)
        if COL_WATER in df.columns:
            df[COL_WATER] = df[COL_WATER].where(df[COL_WATER].notna(), df[COL_WATER_DUAL])
            df = df.drop(columns=[COL_WATER_DUAL])
        else:
            df = df.rename(columns={COL_WATER_DUAL: COL_WATER})

    # ─ COL_WATER_INJ: all rows are injection ─────────────────────
    if COL_WATER_INJ in df.columns:
        no_type = df[COL_WORK_TYPE].astype(str).str.strip() == ""
        df.loc[no_type, COL_WORK_TYPE] = WORK_TYPE_INJ
        if COL_WATER in df.columns:
            df[COL_WATER] = df[COL_WATER].where(df[COL_WATER].notna(), df[COL_WATER_INJ])
            df = df.drop(columns=[COL_WATER_INJ])
        else:
            df = df.rename(columns={COL_WATER_INJ: COL_WATER})

    return df


def _read_csv(path: Path) -> pd.DataFrame:
    """Try combinations of encoding + delimiter until one works."""
    best: pd.DataFrame | None = None
    best_ncols = 0
    for enc in _ENCODINGS:
        for sep in _DELIMITERS:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str)
                nc = len(df.columns)
                if nc >= 10:
                    return df          # full MER-style file — use immediately
                if nc > best_ncols:
                    best = df
                    best_ncols = nc
            except Exception:
                continue
    # Accept files with fewer columns (e.g. 3-column unpivoted data)
    if best is not None and best_ncols >= 2:
        return best
    raise ValueError(
        f"Could not read {path} with any encoding/delimiter combination."
    )


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, dtype=str, engine="openpyxl")


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map original (Russian) column names to internal names."""
    mapping: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in HEADER_MAP:
            mapping[col] = HEADER_MAP[key]
    df = df.rename(columns=mapping)
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if COL_DATE not in df.columns:
        return df
    # Try dayfirst=True (DD.MM.YYYY / DD/MM/YYYY) first, then dayfirst=False
    # (M/D/YYYY).  Pick whichever produces fewer NaT values.
    raw = df[COL_DATE]
    dt_day = pd.to_datetime(raw, dayfirst=True, errors="coerce")
    dt_month = pd.to_datetime(raw, dayfirst=False, errors="coerce")
    nat_day = dt_day.isna().sum()
    nat_month = dt_month.isna().sum()
    df[COL_DATE] = dt_day if nat_day <= nat_month else dt_month
    return df


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add liquid, cumulative, and water-cut columns."""
    # Ensure water and gas columns exist (zero-filled) so downstream code
    # never has to handle their absence.
    if COL_WATER not in df.columns:
        df[COL_WATER] = 0.0
    if COL_GAS not in df.columns:
        df[COL_GAS] = 0.0
    if COL_OIL not in df.columns:
        df[COL_OIL] = 0.0

    oil = df[COL_OIL]
    water = df[COL_WATER]
    gas = df[COL_GAS]

    df[COL_LIQUID] = oil + water

    # Per-well cumulative sums (sorted by date)
    if COL_WELL in df.columns and COL_DATE in df.columns:
        df = df.sort_values([COL_WELL, COL_DATE])
        grp = df.groupby(COL_WELL)
        df[COL_CUM_OIL] = grp[COL_OIL].cumsum()
        df[COL_CUM_WATER] = grp[COL_WATER].cumsum()
        df[COL_CUM_LIQUID] = grp[COL_LIQUID].cumsum()
        df[COL_CUM_GAS] = grp[COL_GAS].cumsum()
    else:
        df[COL_CUM_OIL] = oil.cumsum()
        df[COL_CUM_WATER] = water.cumsum()
        df[COL_CUM_LIQUID] = df[COL_LIQUID].cumsum()
        df[COL_CUM_GAS] = gas.cumsum()

    # Water cut (fraction, 0–1)
    liquid = df[COL_LIQUID]
    df[COL_WATER_CUT] = (water / liquid).where(liquid > 0, 0.0)

    return df
