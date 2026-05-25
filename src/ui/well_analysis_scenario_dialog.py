"""Well Analysis Scenario Manager — create, rename, duplicate, delete and activate
well-analysis scenarios for the adjusted production dialog."""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.data.models import WellAnalysisScenario


class WellAnalysisScenarioDialog(QDialog):
    """Modal dialog for managing well-analysis scenarios.

    Emits ``scenario_activated(int)`` when the user activates a scenario.
    The dialog works on the live list passed in; changes are visible
    immediately via ``result_scenarios()``.
    """

    scenario_activated = Signal(int)

    def __init__(
        self,
        scenarios: list[WellAnalysisScenario],
        active_idx: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Сценарии приведённой добычи")
        self.resize(800, 460)
        self.setModal(True)

        self._scenarios: list[WellAnalysisScenario] = list(scenarios)
        self._active_idx: int = active_idx

        self._build_ui()
        self._refresh_list()
        if self._scenarios:
            self._list.setCurrentRow(max(0, active_idx))

    # ── Public API ──────────────────────────────────────────────────────────

    def result_scenarios(self) -> list[WellAnalysisScenario]:
        return self._scenarios

    def result_active_idx(self) -> int:
        return self._active_idx

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        lbl = QLabel("<b>Сценарии приведённой добычи</b>")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(lbl)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left — list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget()
        self._list.setMinimumWidth(220)
        self._list.setSpacing(2)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(self._on_activate)
        ll.addWidget(self._list)
        splitter.addWidget(left)

        # Right — detail
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setMinimumWidth(340)
        splitter.addWidget(self._detail)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_new       = QPushButton("Создать")
        self._btn_rename    = QPushButton("Переименовать")
        self._btn_duplicate = QPushButton("Дублировать")
        self._btn_delete    = QPushButton("Удалить")
        self._btn_activate  = QPushButton("Активировать")
        btn_close           = QPushButton("Закрыть")
        self._btn_activate.setDefault(True)
        for btn in (self._btn_new, self._btn_rename, self._btn_duplicate,
                    self._btn_delete, self._btn_activate):
            btn_row.addWidget(btn)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        # Connections
        self._btn_new.clicked.connect(self._on_new)
        self._btn_rename.clicked.connect(self._on_rename)
        self._btn_duplicate.clicked.connect(self._on_duplicate)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_activate.clicked.connect(self._on_activate)
        btn_close.clicked.connect(self.accept)

    # ── List management ─────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        cur = self._list.currentRow()
        self._list.blockSignals(True)
        self._list.clear()
        bold = QFont()
        bold.setBold(True)
        for i, sc in enumerate(self._scenarios):
            n_wells = len(sc.wells)
            n_excl  = len(sc.excluded)
            has_pct = bool(sc.pct_months)
            phase_str = "газ" if sc.phase == "gas" else "нефть"
            sub = f"  {n_wells} скв. · {phase_str}"
            if n_excl:
                sub += f" · {n_excl} искл."
            if has_pct:
                sub += " · P90/P50/P10 ✓"
            if i == self._active_idx:
                sub += "  ★ активный"
            item = QListWidgetItem(f"{sc.name}\n{sub}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            if i == self._active_idx:
                item.setFont(bold)
            self._list.addItem(item)
        target = max(0, min(cur, self._list.count() - 1))
        self._list.blockSignals(False)
        self._list.setCurrentRow(target)

    def _selected_idx(self) -> int | None:
        item = self._list.currentItem()
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def _update_buttons(self) -> None:
        has = self._selected_idx() is not None
        self._btn_rename.setEnabled(has)
        self._btn_duplicate.setEnabled(has)
        self._btn_delete.setEnabled(has and len(self._scenarios) > 1)
        self._btn_activate.setEnabled(has)

    # ── Detail pane ─────────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            self._detail.clear()
        else:
            self._detail.setPlainText(self._build_detail(self._scenarios[idx]))
        self._update_buttons()

    def _build_detail(self, sc: WellAnalysisScenario) -> str:
        lines = [
            f"Сценарий: {sc.name}",
            f"Флюид: {'Газ' if sc.phase == 'gas' else 'Нефть'}",
            f"Скважины ({len(sc.wells)}): " + (", ".join(sc.wells) if sc.wells else "—"),
            f"Исключённых точек: {len(sc.excluded)}",
            "",
        ]
        if sc.pct_months:
            n = len(sc.pct_months)
            lines.append(f"P90/P50/P10: {n} точек")
            for pct in ("10", "50", "90"):
                tr = sc.pct_trends.get(pct)
                if tr and len(tr) == 2:
                    qi, Di = tr
                    lines.append(f"  P{pct} тренд: qi={qi:,.1f}, Di={Di:.4f}")
        else:
            lines.append("P90/P50/P10: не рассчитаны")
        return "\n".join(lines)

    # ── Button handlers ─────────────────────────────────────────────────────

    def _on_new(self) -> None:
        n = len(self._scenarios) + 1
        name, ok = QInputDialog.getText(
            self, "Новый сценарий", "Название:", text=f"Анализ {n}"
        )
        if not ok or not name.strip():
            return
        self._scenarios.append(WellAnalysisScenario(name=name.strip()))
        self._refresh_list()
        self._list.setCurrentRow(len(self._scenarios) - 1)

    def _on_rename(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            return
        sc = self._scenarios[idx]
        name, ok = QInputDialog.getText(
            self, "Переименовать", "Новое название:", text=sc.name
        )
        if not ok or not name.strip():
            return
        sc.name = name.strip()
        self._refresh_list()

    def _on_duplicate(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            return
        src = self._scenarios[idx]
        dup = WellAnalysisScenario(
            name       = f"{src.name} (копия)",
            wells      = list(src.wells),
            phase      = src.phase,
            excluded   = copy.deepcopy(src.excluded),
            pct_months = list(src.pct_months),
            pct_data   = copy.deepcopy(src.pct_data),
            pct_trends = copy.deepcopy(src.pct_trends),
        )
        self._scenarios.append(dup)
        self._refresh_list()
        self._list.setCurrentRow(len(self._scenarios) - 1)

    def _on_delete(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            return
        if len(self._scenarios) == 1:
            QMessageBox.information(
                self, "Удаление невозможно",
                "Нельзя удалить единственный сценарий."
            )
            return
        sc = self._scenarios[idx]
        reply = QMessageBox.question(
            self, "Удалить сценарий",
            f"Удалить «{sc.name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._scenarios.pop(idx)
        if self._active_idx >= len(self._scenarios):
            self._active_idx = len(self._scenarios) - 1
        elif self._active_idx > idx:
            self._active_idx -= 1
        self._refresh_list()

    def _on_activate(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            return
        self._active_idx = idx
        self._refresh_list()
        self.scenario_activated.emit(idx)
