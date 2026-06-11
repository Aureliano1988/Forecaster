"""Dialog for entering reservoir parameters (STOIIP and HCPV) per scenario."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.data.models import ForecastScenario

_COL_NAME   = 0
_COL_STOIIP = 1
_COL_HCPV   = 2


class ReservoirDataDialog(QDialog):
    """Table-based dialog for STOIIP / HCPV per scenario.

    Row 0 = project defaults; rows 1..N = individual scenarios.
    Columns: Сценарий (read-only) | STOIIP, т | HCPV, м³.
    Supports copy/paste from Excel into the STOIIP / HCPV columns.
    """

    def __init__(
        self,
        scenarios: list[ForecastScenario],
        default_stoiip: float = 0.0,
        default_hcpv: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Данные пласта — STOIIP / HCPV")
        self.resize(560, max(300, 100 + 28 * len(scenarios)))

        self._scenarios = scenarios

        root = QVBoxLayout(self)

        note = QLabel(
            "Нулевые значения отключают расчёт КИН (RF) и HCPVI.\n"
            "Можно вставлять данные из Excel (Ctrl+V)."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        # ── Table ────────────────────────────────────────────────────
        n_rows = 1 + len(scenarios)  # row 0 = defaults
        self._table = QTableWidget(n_rows, 3)
        self._table.setHorizontalHeaderLabels(["Сценарий", "STOIIP, т", "HCPV, м³"])
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_NAME, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_STOIIP, QHeaderView.ResizeMode.Interactive
        )
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_HCPV, QHeaderView.ResizeMode.Interactive
        )
        self._table.setColumnWidth(_COL_STOIIP, 130)
        self._table.setColumnWidth(_COL_HCPV, 130)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)

        # Row 0: project defaults
        def_name = QTableWidgetItem("По умолчанию (проект)")
        def_name.setFlags(def_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(0, _COL_NAME, def_name)
        self._table.setItem(0, _COL_STOIIP, QTableWidgetItem(
            f"{default_stoiip:.0f}" if default_stoiip else ""
        ))
        self._table.setItem(0, _COL_HCPV, QTableWidgetItem(
            f"{default_hcpv:.0f}" if default_hcpv else ""
        ))

        # Scenario rows
        for i, sc in enumerate(scenarios):
            r = i + 1
            name_item = QTableWidgetItem(sc.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, _COL_NAME, name_item)
            self._table.setItem(r, _COL_STOIIP, QTableWidgetItem(
                f"{sc.stoiip:.0f}" if sc.stoiip else ""
            ))
            self._table.setItem(r, _COL_HCPV, QTableWidgetItem(
                f"{sc.hcpv:.0f}" if sc.hcpv else ""
            ))

        root.addWidget(self._table, stretch=1)

        # ── Buttons ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_fill = QPushButton("Заполнить пустые из умолчания")
        btn_fill.setToolTip(
            "Копирует значения из строки «По умолчанию» во все\n"
            "пустые ячейки STOIIP / HCPV."
        )
        btn_fill.clicked.connect(self._fill_from_defaults)
        btn_row.addWidget(btn_fill)
        btn_row.addStretch()
        root.addLayout(btn_row)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Применить")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        # ── Paste shortcut ───────────────────────────────────────────
        paste_sc = QShortcut(QKeySequence.StandardKey.Paste, self._table)
        paste_sc.activated.connect(self._paste_from_clipboard)

    # ── Public API ──────────────────────────────────────────────────

    def get_default_stoiip(self) -> float:
        return self._cell_float(0, _COL_STOIIP)

    def get_default_hcpv(self) -> float:
        return self._cell_float(0, _COL_HCPV)

    def apply_to_scenarios(self) -> None:
        """Write table values back into scenario objects."""
        for i, sc in enumerate(self._scenarios):
            r = i + 1
            sc.stoiip = self._cell_float(r, _COL_STOIIP)
            sc.hcpv   = self._cell_float(r, _COL_HCPV)

    # ── Helpers ─────────────────────────────────────────────────────

    def _cell_float(self, row: int, col: int) -> float:
        item = self._table.item(row, col)
        if item is None:
            return 0.0
        txt = item.text().strip().replace(",", ".").replace(" ", "")
        try:
            return float(txt)
        except ValueError:
            return 0.0

    def _fill_from_defaults(self) -> None:
        def_stoiip = self._table.item(0, _COL_STOIIP)
        def_hcpv   = self._table.item(0, _COL_HCPV)
        s_text = def_stoiip.text().strip() if def_stoiip else ""
        h_text = def_hcpv.text().strip()   if def_hcpv   else ""
        for r in range(1, self._table.rowCount()):
            for col, default_text in [(_COL_STOIIP, s_text), (_COL_HCPV, h_text)]:
                item = self._table.item(r, col)
                if item is None or not item.text().strip():
                    if item is None:
                        item = QTableWidgetItem(default_text)
                        self._table.setItem(r, col, item)
                    else:
                        item.setText(default_text)

    def _paste_from_clipboard(self) -> None:
        """Paste tab-separated text from clipboard into the selected area."""
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text:
            return

        sel = self._table.selectedRanges()
        if not sel:
            return
        start_row = sel[0].topRow()
        start_col = sel[0].leftColumn()

        for ri, line in enumerate(text.split("\n")):
            if not line.strip():
                continue
            row = start_row + ri
            if row >= self._table.rowCount():
                break
            for ci, val in enumerate(line.split("\t")):
                col = start_col + ci
                if col < 1 or col >= self._table.columnCount():
                    continue  # skip name column
                item = self._table.item(row, col)
                if item is None:
                    item = QTableWidgetItem()
                    self._table.setItem(row, col, item)
                item.setText(val.strip())
