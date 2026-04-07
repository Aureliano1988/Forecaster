"""Validate loaded production data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.models import COL_DATE, COL_OIL, COL_WATER, COL_WELL


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate(df: pd.DataFrame) -> ValidationResult:
    """Run validation checks on a loaded DataFrame."""
    result = ValidationResult()

    # Required columns
    for col in (COL_WELL, COL_DATE, COL_OIL):
        if col not in df.columns:
            result.errors.append(f"Missing required column: '{col}'")

    if result.errors:
        return result

    # Date parsing
    n_bad_dates = df[COL_DATE].isna().sum()
    if n_bad_dates > 0:
        result.warnings.append(
            f"{n_bad_dates} rows have unparseable dates."
        )

    # Non-negative production
    for col in (COL_OIL, COL_WATER):
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                result.warnings.append(
                    f"{n_neg} negative values in '{col}'."
                )

    # Empty dataset
    if len(df) == 0:
        result.errors.append("File contains no data rows.")

    return result
