"""Export Data dialog — build a table of forecast metrics across scenarios."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.data.models import ForecastScenario


# ── Available parameters (label, extractor function) ──────────────────────────
# Each extractor takes (sc: ForecastScenario, method_key: str, result) → str

def _method_name(sc, key, r, ctx=None):
    return r.method_name

def _family(sc, key, r, ctx=None):
    return key.split("|", 1)[0]

def _wells(sc, key, r, ctx=None):
    return ", ".join(sc.wells) if sc.wells else "—"

def _n_wells(sc, key, r, ctx=None):
    return str(len(sc.wells))

def _duration(sc, key, r, ctx=None):
    m = r.monthly
    return str(m.duration) if m and m.duration > 0 else "—"

def _stop(sc, key, r, ctx=None):
    m = r.monthly
    return (m.stop_reason or "горизонт") if m and m.duration > 0 else "—"

def _qo_hist(sc, key, r, ctx=None):
    return f"{r.qo_hist_last:,.0f}" if r.qo_hist_last > 0 else "—"

def _remain(sc, key, r, ctx=None):
    m = r.monthly
    return f"{m.remain_reserves:,.0f}" if m and m.duration > 0 else "—"

def _total_oil(sc, key, r, ctx=None):
    m = r.monthly
    if m and m.duration > 0:
        return f"{r.qo_hist_last + m.remain_reserves:,.0f}" if r.qo_hist_last > 0 else "—"
    return "—"

def _total_water(sc, key, r, ctx=None):
    m = r.monthly
    if m and m.duration > 0 and m.Qw:
        return f"{m.Qw[-1]:,.0f}"
    return "—"

def _total_liquid(sc, key, r, ctx=None):
    m = r.monthly
    if m and m.duration > 0 and m.Ql:
        return f"{m.Ql[-1]:,.0f}"
    return "—"

def _total_water_inj(sc, key, r, ctx=None):
    """Total forecasted water injection (constant-rate extrapolation)."""
    if not ctx:
        return "—"
    info = ctx.get(sc.name)
    if not info or info["qi_const"] <= 0:
        return "—"
    m = r.monthly
    if m and m.duration > 0:
        return f"{info['qi_const'] * m.duration:,.0f}"
    return "—"

def _wor_last(sc, key, r, ctx=None):
    m = r.monthly
    return f"{m.wor_last:.2f}" if m and m.duration > 0 else "—"

def _qo_last_month(sc, key, r, ctx=None):
    m = r.monthly
    if m and m.duration > 0 and m.qo:
        return f"{m.qo[-1]:,.1f}"
    return "—"

def _rf(sc, key, r, ctx=None):
    m = r.monthly
    stoiip = sc.stoiip
    if stoiip > 0 and m and m.duration > 0 and r.qo_hist_last > 0:
        total = r.qo_hist_last + m.remain_reserves
        return f"{total / stoiip:.4f}"
    return "—"

def _hcpvi(sc, key, r, ctx=None):
    """HCPVI at the end of the forecast horizon."""
    if not ctx:
        return "—"
    info = ctx.get(sc.name)
    if not info or info["eff_hcpv"] <= 0:
        return "—"
    m = r.monthly
    if m and m.duration > 0:
        total_inj = info["qi_hist_last"] + info["qi_const"] * m.duration
        return f"{total_inj / info['eff_hcpv']:.4f}"
    return "—"

def _group(sc, key, r, ctx=None):
    return getattr(sc, "group", "") or "—"


_PARAMS: list[tuple[str, object]] = [
    ("Группа",              _group),
    ("Скважины",            _wells),
    ("Кол-во скважин",      _n_wells),
    ("Семейство",           _family),
    ("Метод",               _method_name),
    ("Горизонт, мес.",      _duration),
    ("Стоп",                _stop),
    ("Нак. нефть факт, т",  _qo_hist),
    ("Ост. запасы, т",      _remain),
    ("НТИК, т",             _total_oil),
    ("Нак. вода прогн., т", _total_water),
    ("Нак. жидк. прогн., т",_total_liquid),
    ("Нак. закачка прогн., т", _total_water_inj),
    ("КИН (RF)",            _rf),
    ("HCPVI (посл.)",       _hcpvi),
    ("ВНФ (посл.)",         _wor_last),
    ("Посл. qo, т/мес",    _qo_last_month),
]

# Default-on parameters
_DEFAULT_ON = {
    "Метод", "Горизонт, мес.", "Ост. запасы, т", "НТИК, т",
    "ВНФ (посл.)", "Посл. qo, т/мес",
}


class ExportDataDialog(QDialog):
    """Dialog to select scenarios + parameters and copy a summary table."""

    def __init__(
        self,
        scenarios: list[ForecastScenario],
        project_name: str = "",
        inj_context: dict | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        title = "Экспорт данных прогнозов"
        if project_name:
            title += f" — {project_name}"
        self.setWindowTitle(title)
        self.resize(1100, 600)

        self._scenarios = scenarios
        self._inj_context: dict = inj_context or {}
        self._build_ui()
        self._refresh_preview()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: scenarios ─────────────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.addWidget(QLabel("<b>Сценарии</b>"))

        self._lst_sc = QListWidget()
        for sc in self._scenarios:
            has_results = any(
                r.monthly and r.monthly.duration > 0
                for r in sc.results.values()
            )
            if not has_results:
                continue
            item = QListWidgetItem(sc.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, sc)
            self._lst_sc.addItem(item)

        left_lay.addWidget(self._lst_sc)

        sc_btns = QHBoxLayout()
        btn_sc_all  = QPushButton("Все")
        btn_sc_none = QPushButton("Снять")
        btn_sc_all.clicked.connect(lambda: self._set_all_checks(self._lst_sc, True))
        btn_sc_none.clicked.connect(lambda: self._set_all_checks(self._lst_sc, False))
        sc_btns.addWidget(btn_sc_all)
        sc_btns.addWidget(btn_sc_none)
        sc_btns.addStretch()
        left_lay.addLayout(sc_btns)
        splitter.addWidget(left)

        # ── Middle: parameters ──────────────────────────────────────────
        mid = QWidget()
        mid_lay = QVBoxLayout(mid)
        mid_lay.setContentsMargins(4, 4, 4, 4)
        mid_lay.addWidget(QLabel("<b>Параметры</b>"))

        self._lst_par = QListWidget()
        for label, _ in _PARAMS:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if label in _DEFAULT_ON
                else Qt.CheckState.Unchecked
            )
            self._lst_par.addItem(item)

        mid_lay.addWidget(self._lst_par)

        par_btns = QHBoxLayout()
        btn_par_all  = QPushButton("Все")
        btn_par_none = QPushButton("Снять")
        btn_par_all.clicked.connect(lambda: self._set_all_checks(self._lst_par, True))
        btn_par_none.clicked.connect(lambda: self._set_all_checks(self._lst_par, False))
        par_btns.addWidget(btn_par_all)
        par_btns.addWidget(btn_par_none)
        par_btns.addStretch()
        mid_lay.addLayout(par_btns)
        splitter.addWidget(mid)

        # ── Right: preview ──────────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)
        right_lay.addWidget(QLabel("<b>Предпросмотр</b>"))

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        right_lay.addWidget(self._table)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 3)
        root.addWidget(splitter, stretch=1)

        # ── Buttons ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Обновить")
        btn_copy    = QPushButton("Копировать в буфер")
        btn_close   = QPushButton("Закрыть")
        btn_refresh.clicked.connect(self._refresh_preview)
        btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        # Auto-refresh on check changes
        self._lst_sc.itemChanged.connect(self._refresh_preview)
        self._lst_par.itemChanged.connect(self._refresh_preview)

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _set_all_checks(lst: QListWidget, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        lst.blockSignals(True)
        for i in range(lst.count()):
            item = lst.item(i)
            if item:
                item.setCheckState(state)
        lst.blockSignals(False)
        lst.itemChanged.emit(lst.item(0) if lst.count() else QListWidgetItem())

    def _checked_scenarios(self) -> list[ForecastScenario]:
        result = []
        for i in range(self._lst_sc.count()):
            item = self._lst_sc.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                result.append(item.data(Qt.ItemDataRole.UserRole))
        return result

    def _checked_params(self) -> list[tuple[str, object]]:
        result = []
        for i in range(self._lst_par.count()):
            item = self._lst_par.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                result.append(_PARAMS[i])
        return result

    # ── Build table ────────────────────────────────────────────────────

    def _build_rows(self) -> tuple[list[str], list[list[str]]]:
        """Return (headers, rows) for the current selection."""
        scenarios = self._checked_scenarios()
        params = self._checked_params()

        headers = ["Сценарий"] + [lbl for lbl, _ in params]
        rows: list[list[str]] = []

        for sc in scenarios:
            for key, r in sc.results.items():
                m = r.monthly
                if not (m and m.duration > 0):
                    continue
                row = [sc.name]
                for lbl, extractor in params:
                    try:
                        row.append(extractor(sc, key, r, self._inj_context))
                    except Exception:
                        row.append("—")
                rows.append(row)

        return headers, rows

    def _refresh_preview(self) -> None:
        headers, rows = self._build_rows()

        self._table.clear()
        self._table.setColumnCount(len(headers))
        self._table.setRowCount(len(rows))
        self._table.setHorizontalHeaderLabels(headers)

        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(r, c, item)

        self._table.resizeColumnsToContents()

    def _copy_to_clipboard(self) -> None:
        from PySide6.QtWidgets import QApplication

        headers, rows = self._build_rows()
        lines = ["\t".join(headers)]
        for row in rows:
            lines.append("\t".join(row))
        QApplication.clipboard().setText("\n".join(lines))
