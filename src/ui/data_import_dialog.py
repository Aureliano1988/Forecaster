"""Data import dialog — preview raw data and assign columns to parameters."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.data.models import (
    COL_CONDENSATE,
    COL_DATE,
    COL_DOWNTIME_REASON,
    COL_EXPLOIT_METHOD,
    COL_EXTRA,
    COL_FORMATION,
    COL_GAS,
    COL_GAS_CAP,
    COL_HOURS_ACCUM,
    COL_HOURS_DOWN,
    COL_HOURS_WORK,
    COL_OIL,
    COL_STATUS,
    COL_WATER,
    COL_WATER_DUAL,
    COL_WATER_INJ,
    COL_WELL,
    COL_WORK_TYPE,
    HEADER_MAP,
)

# ── Assignment options presented to the user ────────────────────────────────
# (display label, internal column constant)
_ASSIGNMENTS: list[tuple[str, str]] = [
    ("(не использовать)", ""),
    ("Скважина",                   COL_WELL),
    ("Дата",                       COL_DATE),
    ("Пласт",                      COL_FORMATION),
    ("Характер работы",            COL_WORK_TYPE),
    ("Состояние",                  COL_STATUS),
    ("Способ эксплуатации",        COL_EXPLOIT_METHOD),
    ("Причина простоя",            COL_DOWNTIME_REASON),
    ("Время работы, ч",            COL_HOURS_WORK),
    ("Время накопления, ч",        COL_HOURS_ACCUM),
    ("Время простоя, ч",           COL_HOURS_DOWN),
    ("Нефть, т",                               COL_OIL),
    ("Вода, т (только добыча)",           COL_WATER),
    ("Вода (добыча/закачка) — автосплит", COL_WATER_DUAL),
    ("Закачка воды, м3 (только закачка)",  COL_WATER_INJ),
    ("Газ, м3",                                COL_GAS),
    ("Газ из ГШ, м3",              COL_GAS_CAP),
    ("Конденсат, т",               COL_CONDENSATE),
    ("Доп. параметр",              COL_EXTRA),
]

# Reverse map: internal column → display label (for duplicate detection messages)
_INTERNAL_TO_LABEL = {v: lbl for lbl, v in _ASSIGNMENTS if v}

# Columns that are semantically equivalent (cannot be mixed)
_WATER_GROUP = {COL_WATER, COL_WATER_DUAL, COL_WATER_INJ}

# Required columns for a valid dataset
_REQUIRED = {COL_WELL: "Скважина", COL_DATE: "Дата", COL_OIL: "Нефть, т"}

# Background colour for the assignment row
_ASSIGN_BG = QColor(230, 240, 255)   # light blue
_PREVIEW_N = 5                        # number of data rows shown


class DataImportDialog(QDialog):
    """Dialog that shows a raw-data preview with per-column assignment dropdowns.

    Usage::

        dlg = DataImportDialog(raw_df, n_files=len(paths), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            mapping = dlg.result_mapping()
            # mapping: {original_col_name: internal_col_name | ""}

    The user can override the auto-detected column assignments before
    accepting.  At least Скважина, Дата and Нефть, т must be assigned for
    the dialog to accept.
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        n_files: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Загрузка данных — назначение столбцов")
        self.resize(1100, 480)

        self._raw_df = raw_df
        self._combos: list[QComboBox] = []

        self._build_ui(n_files)
        self._populate_table()

    # ── Public API ──────────────────────────────────────────────────────────

    def result_mapping(self) -> dict[str, str]:
        """Return {original_col_name: internal_col_name} for every column.

        Columns whose combobox is set to "(не использовать)" have an empty
        string value and will be dropped by ``apply_manual_mapping``.
        """
        cols = [str(c) for c in self._raw_df.columns]
        return {
            col: self._combos[i].currentData() or ""
            for i, col in enumerate(cols)
        }

    # ── Construction ────────────────────────────────────────────────────────

    def _build_ui(self, n_files: int) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Info label ──────────────────────────────────────────────────────
        bold = QFont()
        bold.setBold(True)
        info = QLabel("Назначьте параметры для каждого столбца файла:")
        info.setFont(bold)
        root.addWidget(info)

        if n_files > 1:
            multi = QLabel(
                f"Назначение будет применено ко всем {n_files} выбранным файлам."
            )
            multi.setStyleSheet("color: #555;")
            root.addWidget(multi)

        # ── Table ───────────────────────────────────────────────────────────
        n_cols = len(self._raw_df.columns)
        n_data = min(_PREVIEW_N, len(self._raw_df))
        # Row 0 = assignment comboboxes; rows 1..n_data = preview data
        self._table = QTableWidget(1 + n_data, n_cols)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.verticalHeader().setDefaultSectionSize(26)

        # Vertical header labels
        row_labels = ["Параметр"] + [str(i + 1) for i in range(n_data)]
        self._table.setVerticalHeaderLabels(row_labels)

        # Horizontal header = original column names
        orig_cols = [str(c) for c in self._raw_df.columns]
        self._table.setHorizontalHeaderLabels(orig_cols)
        self._table.horizontalHeader().setMinimumSectionSize(120)

        root.addWidget(self._table, stretch=1)

        # ── Legend ──────────────────────────────────────────────────────────
        legend_row = QHBoxLayout()
        req_lbl = QLabel(
            "Обязательные: "
            "<b>Скважина</b>, <b>Дата</b>, <b>Нефть, т</b>"
        )
        legend_row.addWidget(req_lbl)
        legend_row.addStretch()
        root.addLayout(legend_row)

        # ── Buttons ─────────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Загрузить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _populate_table(self) -> None:
        """Fill comboboxes (row 0) and preview data (rows 1+)."""
        orig_cols = [str(c) for c in self._raw_df.columns]
        n_data = min(_PREVIEW_N, len(self._raw_df))

        for col_idx, col_name in enumerate(orig_cols):
            # ── Row 0: assignment combobox ────────────────────────────────
            combo = QComboBox()
            combo.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            for label, internal in _ASSIGNMENTS:
                combo.addItem(label, internal)

            # Auto-detect via HEADER_MAP
            auto_internal = HEADER_MAP.get(col_name.strip().lower(), "")
            if auto_internal:
                for i in range(combo.count()):
                    if combo.itemData(i) == auto_internal:
                        combo.setCurrentIndex(i)
                        break

            self._table.setCellWidget(0, col_idx, combo)
            self._combos.append(combo)

            # Colour the assignment row
            placeholder = QTableWidgetItem()
            placeholder.setBackground(_ASSIGN_BG)
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            # Note: setCellWidget on row 0 already covers the cell, but we
            # set the item background so the area around the combo looks right
            self._table.setItem(0, col_idx, placeholder)

            # ── Rows 1+: preview data ─────────────────────────────────────
            for row_idx in range(n_data):
                val = str(self._raw_df.iloc[row_idx, col_idx])
                if val == "nan":
                    val = ""
                item = QTableWidgetItem(val)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                self._table.setItem(1 + row_idx, col_idx, item)

        self._table.setRowHeight(0, 32)   # taller row for comboboxes
        self._table.resizeColumnsToContents()
        # Enforce minimum column width so comboboxes aren't squished
        for c in range(self._table.columnCount()):
            if self._table.columnWidth(c) < 130:
                self._table.setColumnWidth(c, 130)

    # ── Validation ──────────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        mapping = self.result_mapping()
        assigned_vals = [v for v in mapping.values() if v]

        # Check duplicates
        seen: set[str] = set()
        dups: set[str] = set()
        for v in assigned_vals:
            (dups if v in seen else seen).add(v)
        if dups:
            dup_labels = ", ".join(_INTERNAL_TO_LABEL.get(d, d) for d in dups)
            QMessageBox.warning(
                self, "Дублирование параметров",
                f"Следующие параметры назначены нескольким столбцам: {dup_labels}\n"
                "Пожалуйста, устраните дублирование."
            )
            return

        # Check water-column conflict (only one water variant allowed)
        assigned_water = [v for v in assigned_vals if v in _WATER_GROUP]
        if len(assigned_water) > 1:
            labels = ", ".join(_INTERNAL_TO_LABEL.get(v, v) for v in assigned_water)
            QMessageBox.warning(
                self, "Конфликт назначения воды",
                f"Назначено несколько вариантов воды: {labels}.\n"
                "Допустим один столбец для воды."
            )
            return

        # Check required columns
        missing = [lbl for col, lbl in _REQUIRED.items() if col not in assigned_vals]
        if missing:
            reply = QMessageBox.question(
                self, "Не все параметры назначены",
                f"Не назначены обязательные параметры: {', '.join(missing)}.\n"
                "Загрузка может завершиться ошибкой. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.accept()
