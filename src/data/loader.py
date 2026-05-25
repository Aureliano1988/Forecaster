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
    COL_LIQUID,
    COL_OIL,
    COL_WATER,
    COL_WATER_CUT,
    COL_WATER_DUAL,
    COL_WATER_INJ,
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
    for enc in _ENCODINGS:
        for sep in _DELIMITERS:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str)
                if len(df.columns) >= 10:
                    return df
            except Exception:
                continue
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
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], dayfirst=True, errors="coerce")
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
    oil = df.get(COL_OIL, 0.0)
    water = df.get(COL_WATER, 0.0)
    gas = df.get(COL_GAS, 0.0)

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
