"""Well selection by criteria dialog.

Lets users define up to 3 filter criteria (e.g. first-production year,
average efficiency factor) and returns the list of wells that satisfy
all of them (AND logic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.data.models import (
    COL_DATE, COL_HOURS_WORK, COL_OIL, COL_WELL,
    COL_WORK_TYPE, WORK_TYPE_OIL,
)

# ── Available criteria ────────────────────────────────────────────────────────

# (display_name, internal_key)
_CRITERIA = [
    ("Год начала добычи",              "first_year"),
    ("Средний КЭ",                     "avg_ke"),
    ("Месяцев добычи",                "prod_months"),
    ("Добыча каждый год",            "prod_every_yr"),
    ("КЭ > 0.95 каждый год",          "ke95_every_yr"),
]

_COMPARISONS = [
    ("<",  "lt"),
    (">",  "gt"),
    ("=",  "eq"),
]

_MAX_ROWS = 3


# ── One criteria row ─────────────────────────────────────────────────────────

class _CriteriaRow(QWidget):
    """A single criterion: type + comparison + value + remove button."""

    def __init__(self, on_remove, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)

        self.cmb_type = QComboBox()
        for label, _ in _CRITERIA:
            self.cmb_type.addItem(label)
        lay.addWidget(self.cmb_type, 2)

        self.cmb_cmp = QComboBox()
        for label, _ in _COMPARISONS:
            self.cmb_cmp.addItem(label)
        self.cmb_cmp.setFixedWidth(44)
        lay.addWidget(self.cmb_cmp)

        self.txt_value = QLineEdit()
        self.txt_value.setPlaceholderText("значение")
        self.txt_value.setFixedWidth(80)
        lay.addWidget(self.txt_value)

        self.btn_remove = QPushButton("−")
        self.btn_remove.setFixedWidth(28)
        self.btn_remove.clicked.connect(on_remove)
        lay.addWidget(self.btn_remove)

    def criterion_key(self) -> str:
        return _CRITERIA[self.cmb_type.currentIndex()][1]

    def comparison(self) -> str:
        return _COMPARISONS[self.cmb_cmp.currentIndex()][1]

    def value_text(self) -> str:
        return self.txt_value.text().strip()


# ── Main dialog ──────────────────────────────────────────────────────────────

class WellCriteriaDialog(QDialog):
    """Dialog for selecting wells by computed criteria."""

    def __init__(self, df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Выбрать скважины по критерию")
        self.resize(480, 220)
        self._df = df
        self._matched_wells: list[str] = []

        self._rows: list[_CriteriaRow] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self._lbl = QLabel("Задайте критерии отбора скважин (логика И):")
        root.addWidget(self._lbl)

        self._rows_layout = QVBoxLayout()
        root.addLayout(self._rows_layout)

        # First row (always present)
        self._add_row()

        # Add-row button
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("+ Добавить критерий")
        self._btn_add.clicked.connect(self._add_row)
        btn_row.addWidget(self._btn_add)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()

        # OK / Cancel
        bottom = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Отмена")
        btn_ok.clicked.connect(self._on_ok)
        btn_cancel.clicked.connect(self.reject)
        bottom.addStretch()
        bottom.addWidget(btn_ok)
        bottom.addWidget(btn_cancel)
        root.addLayout(bottom)

        self._update_ui()

    # ── Public ────────────────────────────────────────────────────────────

    def matched_wells(self) -> list[str]:
        """Return the list of well names that passed all criteria."""
        return self._matched_wells

    # ── Row management ────────────────────────────────────────────────────

    def _add_row(self) -> None:
        if len(self._rows) >= _MAX_ROWS:
            return
        row = _CriteriaRow(on_remove=lambda r=None: self._remove_row_widget())
        # Connect the remove button to remove THIS specific row
        row.btn_remove.clicked.disconnect()
        row.btn_remove.clicked.connect(lambda checked=False, w=row: self._remove_row(w))
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        self._update_ui()

    def _remove_row(self, row: _CriteriaRow) -> None:
        if len(self._rows) <= 1:
            return  # keep at least one
        self._rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self._update_ui()

    def _remove_row_widget(self) -> None:
        pass  # placeholder; overridden per-row in _add_row

    def _update_ui(self) -> None:
        if hasattr(self, "_btn_add"):
            self._btn_add.setEnabled(len(self._rows) < _MAX_ROWS)
        for r in self._rows:
            r.btn_remove.setEnabled(len(self._rows) > 1)

    # ── Compute & filter ──────────────────────────────────────────────────

    def _on_ok(self) -> None:
        """Parse criteria, compute per-well metrics, filter, accept."""
        df = self._df
        if df is None or df.empty:
            self.reject()
            return

        # Filter to oil-production rows
        sub = df
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        if COL_WELL not in sub.columns or COL_DATE not in sub.columns:
            QMessageBox.warning(self, "Ошибка", "Нет необходимых столбцов в данных.")
            return

        # Parse all criteria
        criteria: list[tuple[str, str, float]] = []
        for row in self._rows:
            val_str = row.value_text()
            if not val_str:
                QMessageBox.warning(
                    self, "Ошибка",
                    "Заполните значение для всех критериев.",
                )
                return
            try:
                val = float(val_str.replace(",", "."))
            except ValueError:
                QMessageBox.warning(
                    self, "Ошибка",
                    f"Некорректное число: «{val_str}»",
                )
                return
            criteria.append((row.criterion_key(), row.comparison(), val))

        # Compute per-well metrics (lazy — only compute what's needed)
        needed_keys = {c[0] for c in criteria}
        wells = sorted(str(w) for w in sub[COL_WELL].unique() if pd.notna(w))
        well_groups = sub.groupby(COL_WELL)

        metrics: dict[str, dict[str, float]] = {w: {} for w in wells}

        if "first_year" in needed_keys:
            first_dates = well_groups[COL_DATE].min()
            for w in wells:
                if w in first_dates.index:
                    metrics[w]["first_year"] = float(pd.Timestamp(first_dates[w]).year)
                else:
                    metrics[w]["first_year"] = float("nan")

        if "prod_months" in needed_keys:
            n_months_per_well = well_groups[COL_DATE].nunique()
            for w in wells:
                if w in n_months_per_well.index:
                    metrics[w]["prod_months"] = float(n_months_per_well[w])
                else:
                    metrics[w]["prod_months"] = 0.0

        if "avg_ke" in needed_keys:
            if COL_HOURS_WORK in sub.columns:
                total_hours = well_groups[COL_HOURS_WORK].sum()
                n_months = well_groups[COL_DATE].nunique()
                for w in wells:
                    if w in total_hours.index and w in n_months.index:
                        nm = float(n_months[w])
                        if nm > 0:
                            metrics[w]["avg_ke"] = float(total_hours[w]) / (nm * 24.0 * 30.4375)
                        else:
                            metrics[w]["avg_ke"] = float("nan")
                    else:
                        metrics[w]["avg_ke"] = float("nan")
            else:
                for w in wells:
                    metrics[w]["avg_ke"] = float("nan")

        # Per-well annual oil production (year → total oil)
        well_annual: dict[str, dict[int, float]] = {}
        need_annual = "prod_every_yr" in needed_keys
        if need_annual and COL_OIL in sub.columns:
            sub_c = sub.copy()
            sub_c["_year"] = pd.to_datetime(sub_c[COL_DATE]).dt.year
            annual = sub_c.groupby([COL_WELL, "_year"])[COL_OIL].sum()
            for w in wells:
                if w in annual.index.get_level_values(0):
                    well_annual[w] = {
                        int(yr): float(v)
                        for yr, v in annual.loc[w].items()
                    }
                else:
                    well_annual[w] = {}

        # Per-well annual efficiency factor (year → KE)
        well_annual_ke: dict[str, dict[int, float]] = {}
        need_ke_annual = "ke95_every_yr" in needed_keys
        if need_ke_annual and COL_HOURS_WORK in sub.columns:
            sub_c = sub.copy() if not need_annual else sub_c  # reuse if already copied
            if "_year" not in sub_c.columns:
                sub_c["_year"] = pd.to_datetime(sub_c[COL_DATE]).dt.year
            ann_hours = sub_c.groupby([COL_WELL, "_year"])[COL_HOURS_WORK].sum()
            ann_months = sub_c.groupby([COL_WELL, "_year"])[COL_DATE].nunique()
            for w in wells:
                ke_dict: dict[int, float] = {}
                if w in ann_hours.index.get_level_values(0):
                    for yr in ann_hours.loc[w].index:
                        nm = float(ann_months.loc[w].loc[yr]) if yr in ann_months.loc[w].index else 0.0
                        if nm > 0:
                            ke_dict[int(yr)] = float(ann_hours.loc[w].loc[yr]) / (nm * 24.0 * 30.4375)
                        else:
                            ke_dict[int(yr)] = 0.0
                well_annual_ke[w] = ke_dict

        # Apply criteria (AND logic)
        matched: list[str] = []
        for w in wells:
            ok = True
            for key, cmp, val in criteria:
                # Production every year: uses < (before year) or > (from year)
                if key == "prod_every_yr":
                    yr_thr = int(val)
                    wa = well_annual.get(w, {})
                    if cmp == "lt":
                        check_years = [y for y in wa if y < yr_thr]
                    else:  # gt or eq → treat as ≥
                        check_years = [y for y in wa if y >= yr_thr]
                    if not check_years:
                        ok = False
                        break
                    if not all(wa.get(y, 0.0) > 0 for y in check_years):
                        ok = False
                        break
                    continue

                # KE > 0.95 every year: uses < (before year) or > (from year)
                if key == "ke95_every_yr":
                    yr_thr = int(val)
                    wke = well_annual_ke.get(w, {})
                    if cmp == "lt":
                        check_years = [y for y in wke if y < yr_thr]
                    else:
                        check_years = [y for y in wke if y >= yr_thr]
                    if not check_years:
                        ok = False
                        break
                    if not all(wke.get(y, 0.0) > 0.95 for y in check_years):
                        ok = False
                        break
                    continue

                wv = metrics[w].get(key, float("nan"))
                if np.isnan(wv):
                    ok = False
                    break
                if cmp == "lt" and not (wv < val):
                    ok = False
                    break
                if cmp == "gt" and not (wv > val):
                    ok = False
                    break
                if cmp == "eq" and not (abs(wv - val) < 1e-9):
                    ok = False
                    break
            if ok:
                matched.append(w)

        self._matched_wells = matched
        self.accept()
