"""Well coordinates dialog — pick a *.txt file and assign well/X/Y columns.

Only the file location, chosen separator and column assignment are kept;
the actual coordinate values are never parsed into the project. The
preview shows only the first few lines of the file.
"""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

_PREVIEW_N = 5   # number of data rows shown in the preview

# Assignment options: (display label, role key)
_ROLES: list[tuple[str, str]] = [
    ("(не использовать)", ""),
    ("Скважина", "well"),
    ("X", "x"),
    ("Y", "y"),
]
_ROLE_LABELS = {key: label for label, key in _ROLES if key}

# Delimiter checkboxes: (display label, key, literal character)
_DELIMITERS: list[tuple[str, str, str]] = [
    ("Табуляция", "tab", "\t"),
    ("Пробел", "space", " "),
    ("Запятая", "comma", ","),
    ("Точка с запятой", "semicolon", ";"),
]
_DEFAULT_CHECKED = {"tab", "space"}


def read_text_lines(path: str) -> list[str]:
    """Read all lines from *path*, trying a few common encodings."""
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                return [line.rstrip("\r\n") for line in fh]
        except (UnicodeDecodeError, LookupError):
            continue
    return []


def build_separator_pattern(delimiters: dict[str, bool]) -> str:
    """Build a regex character-class pattern from checked delimiter keys.

    Falls back to tab+space when nothing is selected, so callers reading a
    previously-saved (possibly empty) delimiter set never get an empty
    pattern.
    """
    chars = "".join(
        ch for _label, key, ch in _DELIMITERS if delimiters.get(key)
    )
    if not chars:
        chars = "\t "
    return f"[{re.escape(chars)}]+"


def resolve_file_well_coords(
    path: str, mapping: dict[str, int], delimiters: dict[str, bool]
) -> dict[str, tuple[float, float]]:
    """Parse a well-coordinates text file into ``{well_name: (x, y)}``.

    Rows whose well/X/Y columns are missing, non-numeric, or whose
    coordinates are both zero are skipped. Re-reads *path* fresh on every
    call — no coordinate data is cached between calls.
    """
    coords: dict[str, tuple[float, float]] = {}
    idx_well = mapping.get("well")
    idx_x = mapping.get("x")
    idx_y = mapping.get("y")
    if not path or idx_well is None or idx_x is None or idx_y is None:
        return coords

    pattern = build_separator_pattern(delimiters)
    for line in read_text_lines(path):
        if not line.strip():
            continue
        parts = re.split(pattern, line.strip())
        if max(idx_well, idx_x, idx_y) >= len(parts):
            continue
        name = parts[idx_well].strip()
        if not name:
            continue
        try:
            x = float(parts[idx_x].replace(",", "."))
            y = float(parts[idx_y].replace(",", "."))
        except ValueError:
            continue
        if x == 0 and y == 0:
            continue
        coords[name] = (x, y)
    return coords


class WellCoordsDialog(QDialog):
    """Dialog to select a well-coordinates file and assign its columns.

    Usage::

        dlg = WellCoordsDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            path = dlg.result_file_path()
            mapping = dlg.result_mapping()          # {"well": i, "x": i, "y": i}
            delims = dlg.result_delimiters()        # {"tab": True, ...}

    Only the file path and the chosen separator/column assignment are
    returned — the dialog never reads more than a handful of preview rows
    and no coordinate data is loaded into the caller.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Загрузка координат скважин")
        self.resize(700, 420)

        self._file_path: str = ""
        self._raw_lines: list[str] = []
        self._combos: list[QComboBox] = []
        self._checks: dict[str, QCheckBox] = {}

        self._build_ui()
        self._update_ok_enabled()

    # ── Public API ────────────────────────────────────────────────────────

    def result_file_path(self) -> str:
        return self._file_path

    def result_mapping(self) -> dict[str, int]:
        """Return {"well": col_idx, "x": col_idx, "y": col_idx}."""
        mapping: dict[str, int] = {}
        for idx, combo in enumerate(self._combos):
            role = combo.currentData()
            if role:
                mapping[role] = idx
        return mapping

    def result_delimiters(self) -> dict[str, bool]:
        return {key: chk.isChecked() for key, chk in self._checks.items()}

    # ── Construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── File picker row ─────────────────────────────────────────────
        file_row = QHBoxLayout()
        btn_open = QPushButton("Открыть файл…")
        btn_open.clicked.connect(self._on_open_file)
        file_row.addWidget(btn_open)
        self._lbl_file = QLabel("Файл не выбран")
        self._lbl_file.setStyleSheet("color: #555;")
        file_row.addWidget(self._lbl_file, stretch=1)
        root.addLayout(file_row)

        # ── Separator group ─────────────────────────────────────────────
        grp = QGroupBox("Разделитель")
        grp_lay = QHBoxLayout(grp)
        for label, key, _ch in _DELIMITERS:
            chk = QCheckBox(label)
            chk.setChecked(key in _DEFAULT_CHECKED)
            chk.stateChanged.connect(self._on_delimiter_changed)
            grp_lay.addWidget(chk)
            self._checks[key] = chk
        grp_lay.addStretch()
        root.addWidget(grp)

        # ── Preview table ───────────────────────────────────────────────
        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.verticalHeader().setDefaultSectionSize(24)
        root.addWidget(self._table, stretch=1)

        info = QLabel(
            "Назначьте столбцы: <b>Скважина</b>, <b>X</b>, <b>Y</b>."
        )
        info.setStyleSheet("color: #555;")
        root.addWidget(info)

        # ── Buttons ─────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons = buttons
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Загрузить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ── File handling ────────────────────────────────────────────────────

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть файл координат скважин", "",
            "Текстовые файлы (*.txt);;Все файлы (*)",
        )
        if not path:
            return
        lines = read_text_lines(path)
        if not lines:
            QMessageBox.warning(
                self, "Ошибка чтения",
                "Не удалось прочитать файл (проверьте кодировку/формат).",
            )
            return
        self._file_path = path
        self._raw_lines = lines
        self._lbl_file.setText(path)
        self._rebuild_preview()
        self._update_ok_enabled()

    def _on_delimiter_changed(self) -> None:
        # Keep at least one delimiter checked
        if not any(chk.isChecked() for chk in self._checks.values()):
            self._checks["tab"].blockSignals(True)
            self._checks["space"].blockSignals(True)
            self._checks["tab"].setChecked(True)
            self._checks["space"].setChecked(True)
            self._checks["tab"].blockSignals(False)
            self._checks["space"].blockSignals(False)
        self._rebuild_preview()

    def _current_pattern(self) -> str:
        return build_separator_pattern(self.result_delimiters())

    # ── Preview table ────────────────────────────────────────────────────

    def _rebuild_preview(self) -> None:
        self._combos = []
        if not self._raw_lines:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return

        pattern = self._current_pattern()
        preview_src = [ln for ln in self._raw_lines if ln.strip() != ""][:_PREVIEW_N]
        split_rows = [re.split(pattern, ln.strip()) for ln in preview_src]
        n_cols = max((len(r) for r in split_rows), default=0)
        n_data = len(split_rows)

        self._table.setRowCount(1 + n_data)
        self._table.setColumnCount(n_cols)
        self._table.setVerticalHeaderLabels(
            ["Параметр"] + [str(i + 1) for i in range(n_data)]
        )
        self._table.setHorizontalHeaderLabels(
            [f"Столбец {i + 1}" for i in range(n_cols)]
        )

        for col_idx in range(n_cols):
            combo = QComboBox()
            combo.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            for label, role in _ROLES:
                combo.addItem(label, role)
            # Defaults: col0 -> well, col1 -> X, col2 -> Y
            default_role = {0: "well", 1: "x", 2: "y"}.get(col_idx, "")
            if default_role:
                for i in range(combo.count()):
                    if combo.itemData(i) == default_role:
                        combo.setCurrentIndex(i)
                        break
            self._table.setCellWidget(0, col_idx, combo)
            self._combos.append(combo)

            for row_idx, row in enumerate(split_rows):
                val = row[col_idx] if col_idx < len(row) else ""
                item = QTableWidgetItem(val)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                self._table.setItem(1 + row_idx, col_idx, item)

        self._table.setRowHeight(0, 30)
        self._table.resizeColumnsToContents()
        for c in range(self._table.columnCount()):
            if self._table.columnWidth(c) < 110:
                self._table.setColumnWidth(c, 110)

    def _update_ok_enabled(self) -> None:
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(self._file_path)
        )

    # ── Validation ───────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        mapping = self.result_mapping()
        missing = [
            _ROLE_LABELS[role] for role in ("well", "x", "y") if role not in mapping
        ]
        if missing:
            QMessageBox.warning(
                self, "Не все параметры назначены",
                f"Назначьте столбцы для: {', '.join(missing)}.",
            )
            return
        self.accept()
