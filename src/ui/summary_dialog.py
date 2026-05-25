"""Forecasts summary dialog — table of all built method results."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class ForecastSummaryDialog(QDialog):
    """Modal window listing every saved method result in a sortable table."""

    _COLUMNS = [
        ("Семейство",            120),
        ("Метод",                160),
        ("Горизонт, мес.",        90),
        ("Стоп",                  90),
        ("ВНФ (посл.)",           80),
        ("Ост. запасы, т",       110),
        ("Нак. нефть факт, т",   130),
        ("НТИК, т",               110),
        ("Параметры",            220),
    ]

    def __init__(
        self,
        saved_results: dict,
        project_name: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        title = "Сводка прогнозов"
        if project_name:
            title += f" — {project_name}"
        self.setWindowTitle(title)
        self.resize(1280, 520)

        layout = QVBoxLayout(self)

        if project_name:
            lbl = QLabel(f"<b>{project_name}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)

        # ── Table ────────────────────────────────────────────────────────────
        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels([c for c, _ in self._COLUMNS])
        hdr = self._table.horizontalHeader()
        for i, (_, w) in enumerate(self._COLUMNS):
            hdr.resizeSection(i, w)
        hdr.setStretchLastSection(True)
        hdr.setSortIndicatorShown(True)
        self._table.setSortingEnabled(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_copy = QPushButton("Копировать в буфер")
        btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self._populate(saved_results)

    # ── Private ───────────────────────────────────────────────────────────────

    def _populate(self, saved_results: dict) -> None:
        self._table.setSortingEnabled(False)
        rows: list[list[str]] = []

        for key, saved in saved_results.items():
            family, _ = key.split("|", 1)
            m = saved.monthly

            if m is not None and m.duration > 0:
                duration   = str(m.duration)
                stop       = m.stop_reason or "горизонт"
                wor_last   = f"{m.wor_last:.2f}"
                remain     = f"{m.remain_reserves:.0f}"
                qo_hist    = (
                    f"{saved.qo_hist_last:.0f}" if saved.qo_hist_last > 0 else "—"
                )
                uur = (
                    f"{saved.qo_hist_last + m.remain_reserves:.0f}"
                    if saved.qo_hist_last > 0
                    else "—"
                )
            else:
                duration = wor_last = remain = qo_hist = uur = "—"
                stop = "—"

            params_str = ";  ".join(
                f"{k} = {float(v):.4g}" for k, v in saved.parameters.items()
            )

            rows.append([
                family,
                saved.method_name,
                duration,
                stop,
                wor_last,
                remain,
                qo_hist,
                uur,
                params_str,
            ])

        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignVCenter
                    | (
                        Qt.AlignmentFlag.AlignRight
                        if c in (2, 4, 5, 6, 7)
                        else Qt.AlignmentFlag.AlignLeft
                    )
                )
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(r, c, item)

        self._table.setSortingEnabled(True)

    def _copy_to_clipboard(self) -> None:
        """Copy the whole table as tab-separated text."""
        from PySide6.QtWidgets import QApplication
        headers = "\t".join(c for c, _ in self._COLUMNS)
        lines = [headers]
        for r in range(self._table.rowCount()):
            row_vals = []
            for c in range(self._table.columnCount()):
                item = self._table.item(r, c)
                row_vals.append(item.text() if item else "")
            lines.append("\t".join(row_vals))
        QApplication.clipboard().setText("\n".join(lines))
