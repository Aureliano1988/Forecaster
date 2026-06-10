"""Forecast Inspector — manage multiple named forecast scenarios in a project.

Scenarios can be organised into named groups (collapsible tree nodes).
A multi-well scenario can be "ungrouped" into per-well scenarios that
inherit the parent's fitted trends.
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import numpy as np
import pandas as pd

from src.data.models import (
    COL_DATE, COL_GAS, COL_HOURS_WORK, COL_OIL, COL_WATER,
    COL_WELL, COL_WORK_TYPE,
    ForecastScenario, SavedMethodResult, WORK_TYPE_OIL,
)


# ── Constants for item types stored in UserRole ────────────────────────────────
_ROLE_TYPE  = Qt.ItemDataRole.UserRole      # "group" | "scenario"
_ROLE_INDEX = Qt.ItemDataRole.UserRole + 1  # int — flat-list index (scenarios only)
_ROLE_GROUP = Qt.ItemDataRole.UserRole + 2  # str — group name (groups only)


class ForecastInspectorDialog(QDialog):
    """Non-modal dialog for managing forecast scenarios with grouping.

    Emits ``scenario_activated(int)`` when the user activates a scenario.
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
        self.resize(900, 520)
        self.setMinimumWidth(340)
        self.setModal(False)

        self._scenarios: list[ForecastScenario] = list(scenarios)
        self._active_idx: int = active_idx

        self._build_ui()
        self._refresh_tree()

    # ── Public API ──────────────────────────────────────────────────────────

    def result_scenarios(self) -> list[ForecastScenario]:
        return self._scenarios

    def result_active_idx(self) -> int:
        return self._active_idx

    def refresh_active(self, active_idx: int) -> None:
        self._active_idx = active_idx
        self._refresh_tree()

    # ── Compat shim (called by MainWindow._on_forecast_inspector) ──────────
    def _refresh_list(self) -> None:
        self._refresh_tree()

    def sync_from_main(self, scenarios: list[ForecastScenario], active_idx: int) -> None:
        """Live-update the dialog when the main window state changes."""
        if not self.isVisible():
            return
        self._scenarios = list(scenarios)
        self._active_idx = active_idx
        self._refresh_tree()

    def _toggle_detail(self) -> None:
        """Show/hide the right detail panel and resize the dialog."""
        visible = self._right_panel.isVisible()
        self._right_panel.setVisible(not visible)
        self._btn_toggle_detail.setText(
            "▶ Показать детали" if visible else "◀ Скрыть детали"
        )
        if visible:
            # Shrink the dialog to the tree width
            self.resize(380, self.height())
        else:
            self.resize(900, self.height())

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        lbl = QLabel("<b>Сценарии прогнозов</b>")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(lbl)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: tree ──────────────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMinimumWidth(280)
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_activate)
        left_lay.addWidget(self._tree)
        self._splitter.addWidget(left)

        # ── Right: detail (collapsible) ──────────────────────────────
        self._right_panel = QWidget()
        right_lay = QVBoxLayout(self._right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        phase_row = QHBoxLayout()
        phase_row.addWidget(QLabel("Фаза прогноза:"))
        self._cmb_phase = QComboBox()
        self._cmb_phase.addItem("Нефть", "oil")
        self._cmb_phase.addItem("Газ", "gas")
        self._cmb_phase.setEnabled(False)
        phase_row.addWidget(self._cmb_phase)
        phase_row.addStretch()
        right_lay.addLayout(phase_row)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        right_lay.addWidget(self._detail)
        self._splitter.addWidget(self._right_panel)

        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 2)

        # Toggle button for detail panel
        toggle_row = QHBoxLayout()
        self._btn_toggle_detail = QPushButton("◀ Скрыть детали")
        self._btn_toggle_detail.setFixedHeight(22)
        self._btn_toggle_detail.clicked.connect(self._toggle_detail)
        toggle_row.addStretch()
        toggle_row.addWidget(self._btn_toggle_detail)
        root.addLayout(toggle_row)
        root.addWidget(self._splitter, stretch=1)

        # ── Button rows ─────────────────────────────────────────────────────
        row1 = QHBoxLayout()
        self._btn_new       = QPushButton("Создать")
        self._btn_rename    = QPushButton("Переименовать")
        self._btn_duplicate = QPushButton("Дублировать")
        self._btn_delete    = QPushButton("Удалить")
        self._btn_activate  = QPushButton("Активировать")
        for b in (self._btn_new, self._btn_rename, self._btn_duplicate,
                  self._btn_delete, self._btn_activate):
            row1.addWidget(b)
        row1.addStretch()
        root.addLayout(row1)

        row2 = QHBoxLayout()
        self._btn_group         = QPushButton("Группировать")
        self._btn_ungroup_group = QPushButton("Разгруппировать")
        self._btn_ungroup_wells = QPushButton("По скважинам")
        btn_close               = QPushButton("Закрыть")
        row2.addWidget(self._btn_group)
        row2.addWidget(self._btn_ungroup_group)
        row2.addWidget(self._btn_ungroup_wells)
        row2.addStretch()
        row2.addWidget(btn_close)
        root.addLayout(row2)

        # ── Connections ─────────────────────────────────────────────────────
        self._btn_new.clicked.connect(self._on_new)
        self._btn_rename.clicked.connect(self._on_rename)
        self._btn_duplicate.clicked.connect(self._on_duplicate)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_activate.clicked.connect(self._on_activate)
        self._btn_group.clicked.connect(self._on_group)
        self._btn_ungroup_group.clicked.connect(self._on_ungroup_group)
        self._btn_ungroup_wells.clicked.connect(self._on_ungroup_wells)
        btn_close.clicked.connect(self.accept)
        self._cmb_phase.currentIndexChanged.connect(self._on_phase_changed)

    # ── Tree management ─────────────────────────────────────────────────────

    def _refresh_tree(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()

        bold = QFont()
        bold.setBold(True)

        group_items: dict[str, QTreeWidgetItem] = {}

        for i, sc in enumerate(self._scenarios):
            g = getattr(sc, "group", "")

            # Create group header on first encounter
            if g and g not in group_items:
                gi = QTreeWidgetItem(self._tree, [f"\U0001f4c1 {g}"])
                gi.setData(0, _ROLE_TYPE, "group")
                gi.setData(0, _ROLE_GROUP, g)
                gi.setFlags(
                    Qt.ItemFlag.ItemIsEnabled |
                    Qt.ItemFlag.ItemIsSelectable
                )
                gi.setExpanded(True)
                group_items[g] = gi

            # Build scenario label
            n_wells = len(sc.wells)
            n_methods = len(sc.results)
            reserves = sum(
                r.monthly.remain_reserves
                for r in sc.results.values()
                if r.monthly and r.monthly.duration > 0
            )
            sub = f"{n_wells} скв. \u00b7 {n_methods} мет."
            if reserves > 0:
                sub += f" \u00b7 {reserves:,.0f} т"
            if i == self._active_idx:
                sub += "  \u2605"

            label = f"{sc.name}\n  {sub}"
            parent_node = group_items[g] if g else self._tree
            item = QTreeWidgetItem(parent_node, [label])
            item.setData(0, _ROLE_TYPE, "scenario")
            item.setData(0, _ROLE_INDEX, i)
            if i == self._active_idx:
                item.setFont(0, bold)

        self._tree.blockSignals(False)
        self._on_selection_changed()

    def _selected_scenario_indices(self) -> list[int]:
        """Return flat-list indices of all selected scenarios (incl. group children)."""
        indices: list[int] = []
        for item in self._tree.selectedItems():
            if item.data(0, _ROLE_TYPE) == "scenario":
                idx = item.data(0, _ROLE_INDEX)
                if idx not in indices:
                    indices.append(idx)
            elif item.data(0, _ROLE_TYPE) == "group":
                for ci in range(item.childCount()):
                    child = item.child(ci)
                    if child and child.data(0, _ROLE_TYPE) == "scenario":
                        idx = child.data(0, _ROLE_INDEX)
                        if idx not in indices:
                            indices.append(idx)
        return indices

    def _single_selected_idx(self) -> int | None:
        items = self._tree.selectedItems()
        if len(items) != 1:
            return None
        item = items[0]
        if item.data(0, _ROLE_TYPE) != "scenario":
            return None
        return item.data(0, _ROLE_INDEX)

    def _selected_group_name(self) -> str | None:
        items = self._tree.selectedItems()
        if len(items) != 1:
            return None
        item = items[0]
        if item.data(0, _ROLE_TYPE) != "group":
            return None
        return item.data(0, _ROLE_GROUP)

    # ── Detail / phase ──────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        idx = self._single_selected_idx()
        if idx is not None:
            sc = self._scenarios[idx]
            self._cmb_phase.blockSignals(True)
            ci = self._cmb_phase.findData(getattr(sc, "phase", "oil"))
            self._cmb_phase.setCurrentIndex(max(0, ci))
            self._cmb_phase.blockSignals(False)
            self._cmb_phase.setEnabled(True)
            self._detail.setPlainText(self._build_detail(sc))
        else:
            self._detail.clear()
            self._cmb_phase.setEnabled(False)
        self._update_button_states()

    def _on_phase_changed(self, _index: int) -> None:
        idx = self._single_selected_idx()
        if idx is None:
            return
        self._scenarios[idx].phase = self._cmb_phase.currentData()

    def _build_detail(self, sc: ForecastScenario) -> str:
        lines: list[str] = [f"Сценарий: {sc.name}"]
        if getattr(sc, "group", ""):
            lines.append(f"Группа: {sc.group}")
        lines.append(f"Скважины ({len(sc.wells)}): " + (", ".join(sc.wells) if sc.wells else "\u2014"))
        lines.append("")

        if not sc.results:
            lines.append("Нет рассчитанных прогнозов.")
            return "\n".join(lines)

        lines.append(f"{'Метод':<35} {'Горизонт':>10} {'Стоп':>10} {'Ост. запасы, т':>16} {'НТИК, т':>14}")
        lines.append("\u2500" * 90)
        for key, r in sc.results.items():
            m = r.monthly
            if m and m.duration > 0:
                dur  = f"{m.duration} мес."
                stop = m.stop_reason or "горизонт"
                rem  = f"{m.remain_reserves:,.0f}"
                uur  = f"{r.qo_hist_last + m.remain_reserves:,.0f}" if r.qo_hist_last > 0 else "\u2014"
            else:
                dur = stop = rem = uur = "\u2014"
            lines.append(f"{r.method_name:<35} {dur:>10} {stop:>10} {rem:>16} {uur:>14}")

        total = sum(
            r.monthly.remain_reserves
            for r in sc.results.values()
            if r.monthly and r.monthly.duration > 0
        )
        if total > 0:
            lines.append("\u2500" * 90)
            lines.append(f"{'Итого ост. запасы':>57} {total:>16,.0f}")

        return "\n".join(lines)

    def _update_button_states(self) -> None:
        sel_sc = self._selected_scenario_indices()
        single = self._single_selected_idx()
        grp = self._selected_group_name()

        has_single_sc = single is not None

        self._btn_rename.setEnabled(has_single_sc or grp is not None)
        self._btn_duplicate.setEnabled(has_single_sc)
        self._btn_delete.setEnabled(
            (len(sel_sc) > 0 or grp is not None) and len(self._scenarios) > 1
        )
        self._btn_activate.setEnabled(has_single_sc)
        self._btn_group.setEnabled(len(sel_sc) >= 1)
        self._btn_ungroup_group.setEnabled(grp is not None)
        self._btn_ungroup_wells.setEnabled(
            has_single_sc and single is not None and len(self._scenarios[single].wells) > 1
        )

    # ── Scenario actions ────────────────────────────────────────────────────

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
        self._refresh_tree()

    def _on_rename(self) -> None:
        grp = self._selected_group_name()
        if grp is not None:
            new_name, ok = QInputDialog.getText(
                self, "Переименовать группу", "Новое название:", text=grp,
            )
            if not ok or not new_name.strip():
                return
            new_name = new_name.strip()
            for sc in self._scenarios:
                if getattr(sc, "group", "") == grp:
                    sc.group = new_name
            self._refresh_tree()
            return

        idx = self._single_selected_idx()
        if idx is None:
            return
        sc = self._scenarios[idx]
        name, ok = QInputDialog.getText(
            self, "Переименовать сценарий", "Новое название:", text=sc.name,
        )
        if not ok or not name.strip():
            return
        sc.name = name.strip()
        self._refresh_tree()

    def _on_duplicate(self) -> None:
        idx = self._single_selected_idx()
        if idx is None:
            return
        src = self._scenarios[idx]
        dup = ForecastScenario(
            name=f"{src.name} (копия)",
            wells=list(src.wells),
            results=copy.deepcopy(src.results),
            stoiip=src.stoiip,
            hcpv=src.hcpv,
            phase=getattr(src, "phase", "oil"),
            dca_mode=getattr(src, "dca_mode", "production"),
            group=getattr(src, "group", ""),
        )
        self._scenarios.append(dup)
        self._refresh_tree()

    def _on_delete(self) -> None:
        grp = self._selected_group_name()
        if grp is not None:
            reply = QMessageBox.question(
                self, "Удалить группу",
                f"Снять группировку \u00ab{grp}\u00bb?\n"
                "Сценарии останутся, но будут вынесены из группы.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            for sc in self._scenarios:
                if getattr(sc, "group", "") == grp:
                    sc.group = ""
            self._refresh_tree()
            return

        indices = sorted(self._selected_scenario_indices(), reverse=True)
        if not indices:
            return
        if len(self._scenarios) - len(indices) < 1:
            QMessageBox.information(
                self, "Удаление невозможно",
                "Нельзя удалить все сценарии \u2014 должен остаться хотя бы один.",
            )
            return

        names = ", ".join(self._scenarios[i].name for i in indices[:5])
        if len(indices) > 5:
            names += f" (+{len(indices) - 5})"
        reply = QMessageBox.question(
            self, "Удалить сценарии",
            f"Удалить {len(indices)} сценарий(ев): {names}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        for i in indices:
            self._scenarios.pop(i)
        if self._active_idx >= len(self._scenarios):
            self._active_idx = len(self._scenarios) - 1
        self._refresh_tree()

    def _on_activate(self) -> None:
        idx = self._single_selected_idx()
        if idx is None:
            item = self._tree.currentItem()
            if item and item.data(0, _ROLE_TYPE) == "scenario":
                idx = item.data(0, _ROLE_INDEX)
        if idx is None:
            return
        self._active_idx = idx
        self._refresh_tree()
        self.scenario_activated.emit(idx)

    # ── Grouping ────────────────────────────────────────────────────────────

    def _on_group(self) -> None:
        indices = self._selected_scenario_indices()
        if not indices:
            return
        default = getattr(self._scenarios[indices[0]], "group", "") or "Новая группа"
        name, ok = QInputDialog.getText(
            self, "Группировать сценарии", "Название группы:", text=default,
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        for i in indices:
            self._scenarios[i].group = name
        self._refresh_tree()

    # ── Group ungroup ───────────────────────────────────────────────────────

    def _on_ungroup_group(self) -> None:
        """Remove the group tag from all scenarios in the selected group."""
        grp = self._selected_group_name()
        if grp is None:
            return
        for sc in self._scenarios:
            if getattr(sc, "group", "") == grp:
                sc.group = ""
        self._refresh_tree()

    # ── Per-well ungroup ──────────────────────────────────────────────────────

    def _on_ungroup_wells(self) -> None:
        idx = self._single_selected_idx()
        if idx is None:
            return
        src = self._scenarios[idx]
        if len(src.wells) <= 1:
            return

        # Access parent MainWindow for data + settings
        mw = self.parent()
        df = getattr(mw, "df", None) if mw else None
        if df is None:
            QMessageBox.warning(
                self, "Нет данных",
                "Данные не загружены — невозможно рассчитать прогноз по скважинам.",
            )
            return
        mp = getattr(mw, "method_panel", None)
        horizon   = mp.get_horizon()   if mp else 1200
        wor_limit = mp.get_wor_limit() if mp else 99.0
        min_oil   = mp.get_min_oil()   if mp else 30.0
        n_avg     = mp.get_n_avg()     if mp else 1
        dca_mode  = getattr(src, "dca_mode", "production")

        existing_names = {sc.name for sc in self._scenarios}

        target_group = getattr(src, "group", "")
        if not target_group:
            target_group = src.name
            src.group = target_group

        new_scenarios: list[ForecastScenario] = []
        for well in src.wells:
            candidate = well
            counter = 1
            while candidate in existing_names:
                candidate = f"{well} (Copy {counter})"
                counter += 1
            existing_names.add(candidate)

            well_results = self._build_well_results(
                df, well, src, horizon, wor_limit, min_oil, n_avg, dca_mode,
            )

            new_sc = ForecastScenario(
                name=candidate,
                wells=[well],
                results=well_results,
                stoiip=src.stoiip,
                hcpv=src.hcpv,
                phase=getattr(src, "phase", "oil"),
                dca_mode=dca_mode,
                group=target_group,
            )
            new_scenarios.append(new_sc)

        self._scenarios.extend(new_scenarios)
        self._refresh_tree()

        QMessageBox.information(
            self, "По скважинам",
            f"Создано {len(new_scenarios)} сценариев из \u00ab{src.name}\u00bb\n"
            f"в группе \u00ab{target_group}\u00bb.",
        )

    # ── Per-well forecast computation ────────────────────────────────────

    @staticmethod
    def _build_well_results(
        df: pd.DataFrame,
        well: str,
        parent_sc: ForecastScenario,
        horizon: int,
        wor_limit: float,
        min_oil: float,
        n_avg: int,
        dca_mode: str,
    ) -> dict[str, SavedMethodResult]:
        """Rebuild forecasts for *well* using parent’s trend parameters.

        For each method in the parent scenario:
          1. Reconstruct the method object from saved parameters.
          2. Compute physical last values from the well's own data.
          3. Build the monthly forecast anchored to the well's data.
          4. If the well’s last oil rate is below *min_oil*, skip.
        """
        from src.forecasting.displacement import DISPLACEMENT_METHODS, LinearDisplacement
        from src.forecasting.dca import DCA_METHODS
        from src.forecasting.fractional import FRACTIONAL_METHODS
        from src.forecasting.monthly import (
            build_dca_forecast, build_displacement_forecast,
            build_fractional_forecast, dca_time_shift, fractional_qo_anchor,
        )

        # ── Well sub-frame ──────────────────────────────────────────────
        sub = df[df[COL_WELL] == well].copy()
        if COL_WORK_TYPE in sub.columns:
            sub = sub[sub[COL_WORK_TYPE] == WORK_TYPE_OIL]
        if COL_DATE not in sub.columns or COL_OIL not in sub.columns or sub.empty:
            return {}

        # Aggregate by date
        agg_cols = {COL_OIL: "sum"}
        if COL_WATER in sub.columns:
            agg_cols[COL_WATER] = "sum"
        agg = sub.groupby(COL_DATE).agg(agg_cols).sort_index()
        qo_arr = agg[COL_OIL].values.astype(float)
        qw_arr = agg[COL_WATER].values.astype(float) if COL_WATER in agg.columns else np.zeros_like(qo_arr)
        ql_arr = qo_arr + qw_arr
        Qo_arr = np.cumsum(qo_arr)
        Ql_arr = np.cumsum(ql_arr)
        Qw_arr = np.cumsum(qw_arr)

        if len(qo_arr) == 0:
            return {}

        def _avg_last(arr, n):
            tail = arr[-n:]
            pos = tail[tail > 0]
            return float(pos.mean()) if len(pos) > 0 else 0.0

        Qo_last = float(Qo_arr[-1])
        Ql_last = float(Ql_arr[-1])
        Qw_last = float(Qw_arr[-1])
        qo_last = _avg_last(qo_arr, n_avg)
        qw_last = _avg_last(qw_arr, n_avg)
        ql_last = max(_avg_last(ql_arr, n_avg), 1.0)

        # Check cutoff: if well is already below min_oil, skip
        if min_oil > 0 and qo_last < min_oil:
            return {}

        # DCA-specific: x_last, rate-to-monthly
        ts = agg[COL_OIL]
        ts = ts[ts > 0]
        x_last_dca = float(len(ts) - 1) if len(ts) > 0 else 0.0
        q_dca = _avg_last(ts.values, n_avg) if len(ts) > 0 else 0.0
        rate_to_monthly = 1.0
        if dca_mode == "rate" and COL_HOURS_WORK in sub.columns:
            h_agg = sub.groupby(COL_DATE)[COL_HOURS_WORK].sum().sort_index()
            n_wells_agg = sub.groupby(COL_DATE)[COL_WELL].nunique().sort_index()
            tail_h = h_agg.iloc[-n_avg:]
            tail_nw = n_wells_agg.reindex(tail_h.index, fill_value=1)
            total_ph = float(tail_h.sum())
            total_ch = sum(
                pd.Timestamp(d).days_in_month * 24.0 * int(tail_nw.loc[d])
                for d in tail_h.index
            )
            ke = min(1.0, total_ph / total_ch) if total_ch > 0 else 1.0
            rate_to_monthly = 30.4375 * ke
            if rate_to_monthly > 0:
                q_dca = qo_last / rate_to_monthly

        fw_last = (ql_last - qo_last) / ql_last if ql_last > 0 else 0.0

        # ── Method class lookup ────────────────────────────────────────
        all_methods = {
            "Характеристики вытеснения": DISPLACEMENT_METHODS,
            "Кривые падения добычи (DCA)": DCA_METHODS,
            "Фракционный поток": FRACTIONAL_METHODS,
        }

        # Helper: get well's x-coordinates in each method's space
        def _well_xy(family_name, m_cls):
            """Return (x, y) arrays for this well in the method's coordinate space."""
            if family_name == "Характеристики вытеснения":
                try:
                    return m_cls.prepare_xy(Qo_arr, Ql_arr, Qw_arr, qo_arr, ql_arr, qw_arr)
                except Exception:
                    return np.array([]), np.array([])
            elif family_name == "Кривые падения добычи (DCA)":
                ts_pos = agg[COL_OIL]
                ts_pos = ts_pos[ts_pos > 0]
                return np.arange(len(ts_pos), dtype=float), ts_pos.values.astype(float)
            elif family_name == "Фракционный поток":
                from src.data.models import COL_CUM_OIL, COL_WATER_CUT
                if COL_CUM_OIL in agg.columns:
                    return agg[COL_CUM_OIL].values.astype(float), np.array([])
                return Qo_arr, np.array([])
            return np.array([]), np.array([])

        results: dict[str, SavedMethodResult] = {}
        for key, parent_result in parent_sc.results.items():
            family = key.split("|", 1)[0]
            method_classes = all_methods.get(family, [])

            # Reconstruct method from parameters
            method_cls = None
            for cls in method_classes:
                if cls().get_name() == parent_result.method_name:
                    method_cls = cls
                    break
            if method_cls is None:
                continue

            method = method_cls()
            for attr_key, val in parent_result.parameters.items():
                attr_lower = (attr_key[0].lower() + attr_key[1:]) if attr_key else attr_key
                if hasattr(method, attr_lower):
                    setattr(method, attr_lower, float(val))
                elif hasattr(method, attr_key):
                    setattr(method, attr_key, float(val))

            # Build forecast + visual forecast line
            monthly = None
            x_fc_list: list[float] = []
            y_fc_list: list[float] = []
            try:
                if family == "Характеристики вытеснения" and isinstance(method, LinearDisplacement):
                    from src.forecasting.monthly import anchor_displacement_method
                    monthly = build_displacement_forecast(
                        method, Qo_last, Ql_last, Qw_last,
                        qo_last, qw_last, ql_last,
                        horizon, wor_limit, min_oil,
                    )
                    # Visual: anchored predict in displacement-xy space
                    if monthly and monthly.duration > 0:
                        x_xy, _ = method_cls.prepare_xy(
                            Qo_arr, Ql_arr, Qw_arr, qo_arr, ql_arr, qw_arr,
                        )
                        if len(x_xy) > 1:
                            x_last_d = float(x_xy[-1])
                            dx = (x_xy[-1] - x_xy[0]) / max(len(x_xy) - 1, 1)
                            method_vis = anchor_displacement_method(
                                method, Qo_last, Ql_last, Qw_last,
                                qo_last, ql_last, qw_last,
                            )
                            xf = np.linspace(x_last_d, x_last_d + dx * monthly.duration, monthly.duration)
                            yf = method_vis.predict(xf)
                            x_fc_list = xf.tolist()
                            y_fc_list = yf.tolist()

                elif family == "Кривые падения добычи (DCA)":
                    t_shift = dca_time_shift(method, q_dca)
                    monthly = build_dca_forecast(
                        method, x_last_dca, q_dca, ql_last,
                        horizon, wor_limit, min_oil,
                        rate_to_monthly=rate_to_monthly,
                    )
                    # Visual: decline curve in month-index space
                    if monthly and monthly.duration > 0:
                        n_fc = monthly.duration
                        xf = np.arange(x_last_dca + 1, x_last_dca + 1 + n_fc, dtype=float)
                        yf = method.predict(t_shift + np.arange(1, n_fc + 1, dtype=float))
                        x_fc_list = xf.tolist()
                        y_fc_list = yf.tolist()

                elif family == "Фракционный поток":
                    Qo_eff = fractional_qo_anchor(method, fw_last, Qo_last)
                    monthly = build_fractional_forecast(
                        method, Qo_eff, fw_last, ql_last,
                        horizon, wor_limit, min_oil,
                    )
                    if monthly and monthly.duration > 0:
                        n_fc = monthly.duration
                        xf = np.linspace(Qo_eff, Qo_eff + 1.0 * n_fc, n_fc)
                        yf = method.predict(xf)
                        x_fc_list = xf.tolist()
                        y_fc_list = yf.tolist()
            except Exception:
                monthly = None

            # Build result text
            params = method.get_parameters()
            lines = [f"Метод: {method.get_name()}"]
            for k, v in params.items():
                lines.append(f"  {k} = {float(v):.6g}")
            if monthly and monthly.duration > 0:
                stopped_by = monthly.stop_reason or "горизонт"
                uur = Qo_last + monthly.remain_reserves
                lines += [
                    "\u2500" * 22,
                    f"Прогноз (стоп: {stopped_by}):",
                    f"  Горизонт: {monthly.duration} мес.",
                    f"  Нак. нефть (факт): {Qo_last:,.0f} т",
                    f"  Ост. запасы: {monthly.remain_reserves:,.0f} т",
                    f"  НТИК: {uur:,.0f} т",
                    f"  ВНФ (посл.): {monthly.wor_last:.2f}",
                ]

            # Compute trend line over well's last N data points,
            # anchored so it passes through the well's last (x, y) point.
            x_tr_list: list[float] = []
            y_tr_list: list[float] = []
            try:
                parent_n_pts = len(parent_result.x_trend)
                well_x, well_y = _well_xy(family, method_cls)
                if len(well_x) >= 2 and parent_n_pts >= 2:
                    n_pts = min(parent_n_pts, len(well_x))
                    x_start = float(well_x[-n_pts])
                    x_end   = float(well_x[-1])
                    xt = np.linspace(x_start, x_end, 300)

                    if family == "Кривые падения добычи (DCA)":
                        # Shift DCA curve so predict(t_shift + offset) = well's rate
                        t_shift_tr = dca_time_shift(method, float(well_y[-1])) if len(well_y) > 0 else 0.0
                        offsets = xt - x_end   # negative for points before last
                        yt = method.predict(t_shift_tr + offsets)
                    elif family == "Характеристики вытеснения" and isinstance(method, LinearDisplacement):
                        from src.forecasting.monthly import anchor_displacement_method
                        method_anch = anchor_displacement_method(
                            method, Qo_last, Ql_last, Qw_last,
                            qo_last, ql_last, qw_last,
                        )
                        yt = method_anch.predict(xt)
                    else:
                        yt = method.predict(xt)

                    x_tr_list = xt.tolist()
                    y_tr_list = yt.tolist()
            except Exception:
                pass

            results[key] = SavedMethodResult(
                method_name=method.get_name(),
                params_text="\n".join(lines),
                parameters={k: float(v) for k, v in params.items()},
                x_trend=x_tr_list,
                y_trend=y_tr_list,
                x_forecast=x_fc_list,
                y_forecast=y_fc_list,
                monthly=monthly,
                qo_hist_last=Qo_last,
            )

        return results
