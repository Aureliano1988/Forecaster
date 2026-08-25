"""Left-side panel: load data, select wells, preview table."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.data.models import COL_WELL


def _read_well_list(path: str) -> list[str]:
    """Read a plain-text well-name list (one name per line).

    Blank lines and lines starting with ``#`` are ignored.
    Whitespace around each name is stripped.
    """
    names: list[str] = []
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                for line in fh:
                    name = line.strip()
                    if name and not name.startswith("#"):
                        names.append(name)
            return names
        except (UnicodeDecodeError, LookupError):
            continue
    return names


class DataPanel(QWidget):
    """Panel for loading files and selecting wells."""

    wells_changed    = Signal(list)         # emits list of selected well names
    filter_applied   = Signal(list, list)   # emits (found_wells, missing_wells)
    scenario_changed = Signal(int)          # emits new scenario index
    criteria_requested = Signal()           # user clicked "select by criteria"
    map_selection_requested = Signal()      # user clicked "select on map"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)

        layout = QVBoxLayout(self)

        # ── Load buttons ──────────────────────────────────────
        load_row = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить из CSV…")
        self.btn_load_db = QPushButton("Загрузить из БД…")
        load_row.addWidget(self.btn_load)
        load_row.addWidget(self.btn_load_db)
        layout.addLayout(load_row)

        # ── Info label ───────────────────────────────────────────────────────
        self.lbl_info = QLabel("Файл не загружен")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        # ── Scenario selector ───────────────────────────────────────────────
        sc_row = QHBoxLayout()
        sc_row.addWidget(QLabel("Сценарий:"))
        self.cmb_scenario = QComboBox()
        self.cmb_scenario.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        sc_row.addWidget(self.cmb_scenario, 1)
        layout.addLayout(sc_row)
        self.cmb_scenario.currentIndexChanged.connect(self._on_scenario_selected)

        # ── Well list ────────────────────────────────────────────────────────
        grp = QGroupBox("Скважины")
        grp_layout = QVBoxLayout(grp)
        self.well_list = QListWidget()
        self.well_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        grp_layout.addWidget(self.well_list)

        row_sel = QHBoxLayout()
        self.btn_select_all = QPushButton("Все")
        self.btn_select_none = QPushButton("Снять")
        row_sel.addWidget(self.btn_select_all)
        row_sel.addWidget(self.btn_select_none)
        row_sel.addStretch()
        grp_layout.addLayout(row_sel)

        self.btn_filter = QPushButton("Фильтр скважин…")
        grp_layout.addWidget(self.btn_filter)

        self.btn_criteria = QPushButton("Выбрать по критерию…")
        grp_layout.addWidget(self.btn_criteria)

        self.btn_map = QPushButton("Выбрать на карте…")
        self.btn_map.setVisible(False)   # shown only when coordinates are loaded
        grp_layout.addWidget(self.btn_map)

        self.lbl_filter = QLabel("")
        self.lbl_filter.setWordWrap(True)
        grp_layout.addWidget(self.lbl_filter)

        layout.addWidget(grp)

        # ── Active-wells overlay toggle ────────────────────────────────────
        self.chk_active_wells = QCheckBox("Кол-во акт. скв. (Y2)")
        self.chk_active_wells.setToolTip(
            "Показывает количество скважин \u0441 \u043dенулевой добычей нефти \u043dа вторичной оси Y2"
        )
        layout.addWidget(self.chk_active_wells)
        layout.addStretch()

        # ── Connections ────────────────────────────────────────────────────────
        self.btn_select_all.clicked.connect(self._select_all)
        self.btn_select_none.clicked.connect(self._deselect_all)
        self.btn_filter.clicked.connect(self._on_load_filter)
        self.btn_criteria.clicked.connect(self.criteria_requested.emit)
        self.btn_map.clicked.connect(self.map_selection_requested.emit)
        self.well_list.itemSelectionChanged.connect(self._on_selection)

    # ── Public ───────────────────────────────────────────────────────────────

    def set_map_button_visible(self, visible: bool) -> None:
        self.btn_map.setVisible(visible)

    def populate_wells(self, wells: list[str]) -> None:
        self.well_list.clear()
        for w in sorted(wells):
            item = QListWidgetItem(w)
            self.well_list.addItem(item)

    def get_selected_wells(self) -> list[str]:
        return [item.text() for item in self.well_list.selectedItems()]

    def get_file_paths(self) -> list[str]:
        """Open a multi-file dialog; returns empty list if cancelled."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Открыть файл(ы) данных",
            "",
            "CSV / Excel (*.csv *.txt *.xls *.xlsx);;Все файлы (*)",
        )
        return paths

    def load_well_filter(self) -> None:
        """Open a file dialog and apply the selected well-name list as a filter."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить список скважин",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*)",
        )
        if not path:
            return
        names = _read_well_list(path)
        self.apply_well_filter(names)

    def apply_well_filter(self, names: list[str]) -> tuple[list[str], list[str]]:
        """Select only wells whose names appear in *names* (case-insensitive).

        Returns ``(found, missing)`` where *found* are data wells that were
        matched and selected, and *missing* are names from the filter file
        that could not be matched.
        """
        filter_lower = {n.lower(): n for n in names}  # lower → original
        found: list[str] = []
        matched_lower: set[str] = set()

        # Select matching items; block per-item signals to emit one combined event
        self.well_list.blockSignals(True)
        self.well_list.clearSelection()
        for i in range(self.well_list.count()):
            item = self.well_list.item(i)
            if item is None:
                continue
            item_lower = item.text().lower()
            if item_lower in filter_lower:
                item.setSelected(True)
                found.append(item.text())
                matched_lower.add(item_lower)
        self.well_list.blockSignals(False)

        missing = [filter_lower[k] for k in filter_lower if k not in matched_lower]

        # Update filter label
        if names:
            self.lbl_filter.setText(
                f"Фильтр: {len(found)}/{len(names)} скважин"
            )
        else:
            self.lbl_filter.setText("")

        # Emit signals once
        self.wells_changed.emit(found)
        self.filter_applied.emit(found, missing)
        return found, missing

    def populate_scenarios(self, names: list[str], active_idx: int = 0) -> None:
        """Replace the scenario combo items and select *active_idx*."""
        self.cmb_scenario.blockSignals(True)
        self.cmb_scenario.clear()
        for name in names:
            self.cmb_scenario.addItem(name)
        if names:
            self.cmb_scenario.setCurrentIndex(max(0, min(active_idx, len(names) - 1)))
        self.cmb_scenario.blockSignals(False)

    # ── Slots ────────────────────────────────────────────────

    def _select_all(self) -> None:
        self.well_list.selectAll()

    def _deselect_all(self) -> None:
        self.well_list.clearSelection()

    def _on_load_filter(self) -> None:
        self.load_well_filter()

    def _on_selection(self) -> None:
        self.wells_changed.emit(self.get_selected_wells())

    def _on_scenario_selected(self, idx: int) -> None:
        if idx >= 0:
            self.scenario_changed.emit(idx)
