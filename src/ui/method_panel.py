"""Right-side panel: choose forecasting method family, variant, and controls."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.forecasting.dca import DCA_METHODS
from src.forecasting.displacement import DISPLACEMENT_METHODS
from src.forecasting.fractional import FRACTIONAL_METHODS

# Family name → list of method classes
_FAMILIES: dict[str, list] = {
    "Характеристики вытеснения": DISPLACEMENT_METHODS,
    "Кривые падения добычи (DCA)": DCA_METHODS,
    "Фракционный поток": FRACTIONAL_METHODS,
}

# For gas phase: only DCA is applicable
_GAS_FAMILIES: dict[str, list] = {
    "Кривые падения добычи (DCA)": DCA_METHODS,
}


class MethodPanel(QWidget):
    """Panel to select and run a forecasting method."""

    build_requested = Signal()
    discard_requested = Signal()
    eraser_toggled = Signal(bool)   # True = eraser on
    edit_toggled   = Signal(bool)   # True = enter trend-edit mode
    autofit_requested = Signal()    # single-method autofit
    autofit_all_requested = Signal()  # autofit all methods

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)

        self._phase: str = "oil"  # current phase: "oil" | "gas"

        layout = QVBoxLayout(self)

        # ── Family selector ──────────────────────────────────────────────────
        grp = QGroupBox("Метод прогноза")
        grp_layout = QVBoxLayout(grp)

        grp_layout.addWidget(QLabel("Семейство:"))
        self.cmb_family = QComboBox()
        self.cmb_family.addItems(list(_FAMILIES.keys()))
        grp_layout.addWidget(self.cmb_family)

        grp_layout.addWidget(QLabel("Метод:"))
        self.cmb_method = QComboBox()
        grp_layout.addWidget(self.cmb_method)

        layout.addWidget(grp)

        # ── Forecast horizon ─────────────────────────────────────────────────
        grp2 = QGroupBox("Параметры прогноза")
        grp2_layout = QVBoxLayout(grp2)
        grp2_layout.addWidget(QLabel("Горизонт прогноза (мес.):"))
        self.spn_horizon = QSpinBox()
        self.spn_horizon.setRange(1, 1200)
        self.spn_horizon.setValue(1200)
        grp2_layout.addWidget(self.spn_horizon)

        self._lbl_wor = QLabel("Предел ВНФ (WOR):")
        grp2_layout.addWidget(self._lbl_wor)
        self.spn_wor_limit = QSpinBox()
        self.spn_wor_limit.setRange(1, 999)
        self.spn_wor_limit.setValue(99)
        grp2_layout.addWidget(self.spn_wor_limit)

        self._lbl_min_fluid = QLabel("Мин. добыча нефти, т/мес:")
        grp2_layout.addWidget(self._lbl_min_fluid)
        self.spn_min_oil = QSpinBox()
        self.spn_min_oil.setRange(0, 99999)
        self.spn_min_oil.setValue(30)
        grp2_layout.addWidget(self.spn_min_oil)

        grp2_layout.addWidget(QLabel("Осреднение на прогноз (мес.):"))
        self.spn_n_avg = QSpinBox()
        self.spn_n_avg.setRange(1, 120)
        self.spn_n_avg.setValue(1)
        self.spn_n_avg.setToolTip(
            "1 — используется последний месяц (default).\n"
            "N > 1 — прогноз стартует со средней добычи за последние N месяцев."
        )
        grp2_layout.addWidget(self.spn_n_avg)
        layout.addWidget(grp2)

        # ── Action buttons ───────────────────────────────────────────────────
        self.btn_build = QPushButton("Построить прогноз")
        self.btn_edit  = QPushButton("Редактировать тренд")
        self.btn_edit.setCheckable(True)
        self.btn_edit.setEnabled(False)   # enabled after a trend is built
        self.btn_autofit = QPushButton("Автоподбор")
        self.btn_autofit_all = QPushButton("Автоподбор всех")
        self.btn_discard = QPushButton("Сбросить")
        self.btn_eraser = QPushButton("Ластик (исключить данные)")
        self.btn_eraser.setCheckable(True)
        layout.addWidget(self.btn_build)
        layout.addWidget(self.btn_edit)
        layout.addWidget(self.btn_autofit)
        layout.addWidget(self.btn_autofit_all)
        layout.addWidget(self.btn_discard)
        layout.addWidget(self.btn_eraser)

        # ── Result display ───────────────────────────────────────────────────
        grp3 = QGroupBox("Результаты")
        grp3_layout = QVBoxLayout(grp3)
        self.txt_result = QTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setMinimumHeight(160)
        grp3_layout.addWidget(self.txt_result)
        layout.addWidget(grp3)

        layout.addStretch()

        # ── Connections ──────────────────────────────────────────────────────
        self.cmb_family.currentIndexChanged.connect(self._update_methods)
        self.btn_build.clicked.connect(self.build_requested.emit)
        self.btn_edit.toggled.connect(self.edit_toggled.emit)
        self.btn_autofit.clicked.connect(self.autofit_requested.emit)
        self.btn_autofit_all.clicked.connect(self.autofit_all_requested.emit)
        self.btn_discard.clicked.connect(self.discard_requested.emit)
        self.btn_eraser.toggled.connect(self.eraser_toggled.emit)

        # Initial population
        self._update_methods()

    # ── Public ───────────────────────────────────────────────────────────────

    def get_family_name(self) -> str:
        return self.cmb_family.currentText()

    def get_method_class(self):
        """Return the selected ForecastMethod *class*."""
        families = _GAS_FAMILIES if self._phase == "gas" else _FAMILIES
        family = families.get(self.get_family_name(), [])
        idx = self.cmb_method.currentIndex()
        if 0 <= idx < len(family):
            return family[idx]
        return None

    def get_horizon(self) -> int:
        return self.spn_horizon.value()

    def get_wor_limit(self) -> float:
        return float(self.spn_wor_limit.value())

    def get_min_oil(self) -> float:
        return float(self.spn_min_oil.value())

    def get_n_avg(self) -> int:
        """Number of last months to average for forecast starting conditions.

        Always returns a value in [1, 120]; the spinbox enforces this range
        but this method clamps defensively in case of any unexpected state.
        """
        return max(1, min(120, int(self.spn_n_avg.value())))

    def show_result(self, text: str) -> None:
        self.txt_result.setPlainText(text)

    def set_edit_enabled(self, enabled: bool) -> None:
        """Enable or disable the trend-edit button; uncheck it when disabling."""
        self.btn_edit.setEnabled(enabled)
        if not enabled:
            self.btn_edit.blockSignals(True)
            self.btn_edit.setChecked(False)
            self.btn_edit.blockSignals(False)

    def set_phase(self, phase: str) -> None:
        """Switch visible method families based on the forecasting phase.

        "gas" → only DCA families shown; other families hidden.
        "oil" → all families shown.
        """
        if phase == self._phase:
            return
        self._phase = phase
        families = _GAS_FAMILIES if phase == "gas" else _FAMILIES
        current_family = self.cmb_family.currentText()
        self.cmb_family.blockSignals(True)
        self.cmb_family.clear()
        self.cmb_family.addItems(list(families.keys()))
        # Restore previous selection when still valid
        idx = self.cmb_family.findText(current_family)
        self.cmb_family.setCurrentIndex(max(0, idx))
        self.cmb_family.blockSignals(False)
        self._update_methods()
        # Update labels
        if phase == "gas":
            self._lbl_min_fluid.setText("Мин. добыча газа, м\u00b3/мес:")
            self._lbl_wor.setText("Предел ВНФ (WOR):")
            self._lbl_wor.setVisible(False)
            self.spn_wor_limit.setVisible(False)
        else:
            self._lbl_min_fluid.setText("Мин. добыча нефти, т/мес:")
            self._lbl_wor.setText("Предел ВНФ (WOR):")
            self._lbl_wor.setVisible(True)
            self.spn_wor_limit.setVisible(True)

    def get_phase(self) -> str:
        """Return the currently active phase."""
        return self._phase

    # ── Slots ────────────────────────────────────────────────────────────────

    def _update_methods(self) -> None:
        self.cmb_method.clear()
        families = _GAS_FAMILIES if self._phase == "gas" else _FAMILIES
        family = families.get(self.get_family_name(), [])
        for cls in family:
            self.cmb_method.addItem(cls().get_name())
