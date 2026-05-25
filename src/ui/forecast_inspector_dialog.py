"""Forecast Inspector — manage multiple named forecast scenarios in a project."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.data.models import ForecastScenario


class ForecastInspectorDialog(QDialog):
    """Modal dialog for managing forecast scenarios.

    Emits ``scenario_activated(int)`` when the user activates a scenario
    (double-click or button).  The main window connects this to
    ``_activate_scenario(idx)``.

    The dialog operates on a *copy* of the scenarios list that it modifies
    in-place.  Call ``result_scenarios()`` after exec() to retrieve the
    (possibly edited) list.
    """

    scenario_activated = Signal(int)

    def __init__(
        self,
        scenarios: list[ForecastScenario],
        active_idx: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Инспектор прогнозов")
        self.resize(860, 480)
        self.setModal(False)   # non-blocking: main window remains interactive

        # Work on a shallow copy so edits are committed when the dialog is closed
        self._scenarios: list[ForecastScenario] = list(scenarios)
        self._active_idx: int = active_idx

        self._build_ui()
        self._refresh_list()
        self._list.setCurrentRow(active_idx)

    # ── Public API ──────────────────────────────────────────────────────────

    def result_scenarios(self) -> list[ForecastScenario]:
        """Return the (possibly edited) scenarios list."""
        return self._scenarios

    def result_active_idx(self) -> int:
        return self._active_idx

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ── Title label ────────────────────────────────────────────────────
        lbl = QLabel("<b>Сценарии прогнозов</b>")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(lbl)

        # ── Splitter: list | detail ─────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left — scenario list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.setMinimumWidth(240)
        self._list.setSpacing(2)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(self._on_activate)
        left_layout.addWidget(self._list)

        splitter.addWidget(left)

        # Right — phase selector + detail text
        right_panel = QWidget()
        right_panel.setMinimumWidth(350)
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        phase_row = QHBoxLayout()
        phase_row.addWidget(QLabel("Фаза прогноза:"))
        self._cmb_phase = QComboBox()
        self._cmb_phase.addItem("Нефть", "oil")
        self._cmb_phase.addItem("Газ", "gas")
        self._cmb_phase.setEnabled(False)  # enabled when a scenario is selected
        self._cmb_phase.setToolTip(
            "Нефть: все методы доступны.\n"
            "Газ: только DCA-методы."
        )
        phase_row.addWidget(self._cmb_phase)
        phase_row.addStretch()
        right_lay.addLayout(phase_row)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        right_lay.addWidget(self._detail)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

        # ── Button row ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._btn_new       = QPushButton("Создать")
        self._btn_rename    = QPushButton("Переименовать")
        self._btn_duplicate = QPushButton("Дублировать")
        self._btn_delete    = QPushButton("Удалить")
        self._btn_activate  = QPushButton("Активировать")
        btn_close           = QPushButton("Закрыть")

        self._btn_activate.setDefault(True)

        for btn in (
            self._btn_new, self._btn_rename, self._btn_duplicate,
            self._btn_delete, self._btn_activate,
        ):
            btn_row.addWidget(btn)

        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        # ── Connections ────────────────────────────────────────────────────────────────
        self._btn_new.clicked.connect(self._on_new)
        self._btn_rename.clicked.connect(self._on_rename)
        self._btn_duplicate.clicked.connect(self._on_duplicate)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_activate.clicked.connect(self._on_activate)
        btn_close.clicked.connect(self.accept)
        self._cmb_phase.currentIndexChanged.connect(self._on_phase_changed)

    # ── List management ─────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        """Rebuild the QListWidget from _scenarios."""
        current_row = self._list.currentRow()
        self._list.blockSignals(True)
        self._list.clear()

        bold = QFont()
        bold.setBold(True)

        for i, sc in enumerate(self._scenarios):
            n_wells   = len(sc.wells)
            n_methods = len(sc.results)
            reserves  = sum(
                r.monthly.remain_reserves
                for r in sc.results.values()
                if r.monthly and r.monthly.duration > 0
            )

            label = sc.name
            sub = f"  {n_wells} скв. · {n_methods} методов"
            if reserves > 0:
                sub += f" · {reserves:,.0f} т ост."
            if i == self._active_idx:
                sub += "  ★ активный"

            item = QListWidgetItem(f"{label}\n{sub}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            if i == self._active_idx:
                item.setFont(bold)
            self._list.addItem(item)

        # Restore selection
        target = max(0, min(current_row, self._list.count() - 1))
        self._list.blockSignals(False)
        self._list.setCurrentRow(target)

    def _selected_idx(self) -> int | None:
        item = self._list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    # ── Detail panel ────────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            self._detail.clear()
            self._cmb_phase.setEnabled(False)
            return
        sc = self._scenarios[idx]
        # Populate phase combo without triggering _on_phase_changed
        self._cmb_phase.blockSignals(True)
        phase = getattr(sc, "phase", "oil")
        ci = self._cmb_phase.findData(phase)
        self._cmb_phase.setCurrentIndex(max(0, ci))
        self._cmb_phase.blockSignals(False)
        self._cmb_phase.setEnabled(True)
        self._detail.setPlainText(self._build_detail(sc))
        self._update_button_states()

    def _on_phase_changed(self, _index: int) -> None:
        """Apply the selected phase to the currently displayed scenario."""
        idx = self._selected_idx()
        if idx is None:
            return
        sc = self._scenarios[idx]
        sc.phase = self._cmb_phase.currentData()

    def _build_detail(self, sc: ForecastScenario) -> str:
        lines: list[str] = [f"Сценарий: {sc.name}"]
        lines.append(f"Скважины ({len(sc.wells)}): " + (", ".join(sc.wells) if sc.wells else "—"))
        lines.append("")

        if not sc.results:
            lines.append("Нет рассчитанных прогнозов.")
            return "\n".join(lines)

        lines.append(f"{'Метод':<35} {'Горизонт':>10} {'Стоп':>10} {'Ост. запасы, т':>16} {'НТИК, т':>14}")
        lines.append("─" * 90)
        for key, r in sc.results.items():
            m = r.monthly
            if m and m.duration > 0:
                dur   = f"{m.duration} мес."
                stop  = m.stop_reason or "горизонт"
                rem   = f"{m.remain_reserves:,.0f}"
                uur   = f"{r.qo_hist_last + m.remain_reserves:,.0f}" if r.qo_hist_last > 0 else "—"
            else:
                dur = stop = rem = uur = "—"
            lines.append(f"{r.method_name:<35} {dur:>10} {stop:>10} {rem:>16} {uur:>14}")

        total = sum(
            r.monthly.remain_reserves
            for r in sc.results.values()
            if r.monthly and r.monthly.duration > 0
        )
        if total > 0:
            lines.append("─" * 90)
            lines.append(f"{'Итого ост. запасы':>57} {total:>16,.0f}")

        return "\n".join(lines)

    # ── Button actions ──────────────────────────────────────────────────────

    def _on_new(self) -> None:
        n = len(self._scenarios) + 1
        name, ok = QInputDialog.getText(
            self, "Новый сценарий", "Название сценария:",
            text=f"Сценарий {n}",
        )
        if not ok or not name.strip():
            return
        new_sc = ForecastScenario(name=name.strip())
        self._scenarios.append(new_sc)
        self._refresh_list()
        self._list.setCurrentRow(len(self._scenarios) - 1)

    def _on_rename(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            return
        sc = self._scenarios[idx]
        name, ok = QInputDialog.getText(
            self, "Переименовать сценарий", "Новое название:",
            text=sc.name,
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
        import copy
        dup = ForecastScenario(
            name=f"{src.name} (копия)",
            wells=list(src.wells),
            results=copy.deepcopy(src.results),
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
            f"Удалить сценарий «{sc.name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._scenarios.pop(idx)
        # Adjust active index
        if self._active_idx >= len(self._scenarios):
            self._active_idx = len(self._scenarios) - 1
        elif self._active_idx > idx:
            self._active_idx -= 1
        self._refresh_list()

    def _on_activate(self) -> None:
        """Emit scenario_activated without closing the dialog."""
        idx = self._selected_idx()
        if idx is None:
            return
        self._active_idx = idx
        self._refresh_list()
        self.scenario_activated.emit(idx)
        # dialog stays open — user can activate more scenarios or close manually

    def refresh_active(self, active_idx: int) -> None:
        """Update the active-indicator after the main window switches scenarios."""
        self._active_idx = active_idx
        self._refresh_list()

    def _update_button_states(self) -> None:
        has_sel = self._selected_idx() is not None
        self._btn_rename.setEnabled(has_sel)
        self._btn_duplicate.setEnabled(has_sel)
        self._btn_delete.setEnabled(has_sel and len(self._scenarios) > 1)
        self._btn_activate.setEnabled(has_sel)
