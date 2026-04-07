"""Left-side panel: load data, select wells, preview table."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.data.models import COL_WELL


class DataPanel(QWidget):
    """Panel for loading files and selecting wells."""

    wells_changed = Signal(list)  # emits list of selected well names

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)

        layout = QVBoxLayout(self)

        # ── Load button ──────────────────────────────────────────────────────
        self.btn_load = QPushButton("Загрузить данные…")
        layout.addWidget(self.btn_load)

        # ── Info label ───────────────────────────────────────────────────────
        self.lbl_info = QLabel("Файл не загружен")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        # ── Well list ────────────────────────────────────────────────────────
        grp = QGroupBox("Скважины")
        grp_layout = QVBoxLayout(grp)
        self.well_list = QListWidget()
        self.well_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        grp_layout.addWidget(self.well_list)

        self.btn_select_all = QPushButton("Выбрать все")
        grp_layout.addWidget(self.btn_select_all)

        layout.addWidget(grp)
        layout.addStretch()

        # ── Connections ──────────────────────────────────────────────────────
        self.btn_select_all.clicked.connect(self._select_all)
        self.well_list.itemSelectionChanged.connect(self._on_selection)

    # ── Public ───────────────────────────────────────────────────────────────

    def populate_wells(self, wells: list[str]) -> None:
        self.well_list.clear()
        for w in sorted(wells):
            item = QListWidgetItem(w)
            self.well_list.addItem(item)

    def get_selected_wells(self) -> list[str]:
        return [item.text() for item in self.well_list.selectedItems()]

    def get_file_path(self) -> str | None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть файл данных",
            "",
            "CSV / Excel (*.csv *.txt *.xls *.xlsx);;Все файлы (*)",
        )
        return path or None

    # ── Slots ────────────────────────────────────────────────────────────────

    def _select_all(self) -> None:
        self.well_list.selectAll()

    def _on_selection(self) -> None:
        self.wells_changed.emit(self.get_selected_wells())
