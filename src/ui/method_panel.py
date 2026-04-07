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


class MethodPanel(QWidget):
    """Panel to select and run a forecasting method."""

    fit_requested = Signal()
    forecast_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)

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
        self.spn_horizon.setRange(1, 600)
        self.spn_horizon.setValue(60)
        grp2_layout.addWidget(self.spn_horizon)
        layout.addWidget(grp2)

        # ── Action buttons ───────────────────────────────────────────────────
        self.btn_fit = QPushButton("Построить тренд")
        self.btn_forecast = QPushButton("Рассчитать прогноз")
        layout.addWidget(self.btn_fit)
        layout.addWidget(self.btn_forecast)

        # ── Result display ───────────────────────────────────────────────────
        grp3 = QGroupBox("Результаты")
        grp3_layout = QVBoxLayout(grp3)
        self.txt_result = QTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setMaximumHeight(180)
        grp3_layout.addWidget(self.txt_result)
        layout.addWidget(grp3)

        layout.addStretch()

        # ── Connections ──────────────────────────────────────────────────────
        self.cmb_family.currentIndexChanged.connect(self._update_methods)
        self.btn_fit.clicked.connect(self.fit_requested.emit)
        self.btn_forecast.clicked.connect(self.forecast_requested.emit)

        # Initial population
        self._update_methods()

    # ── Public ───────────────────────────────────────────────────────────────

    def get_family_name(self) -> str:
        return self.cmb_family.currentText()

    def get_method_class(self):
        """Return the selected ForecastMethod *class*."""
        family = _FAMILIES.get(self.get_family_name(), [])
        idx = self.cmb_method.currentIndex()
        if 0 <= idx < len(family):
            return family[idx]
        return None

    def get_horizon(self) -> int:
        return self.spn_horizon.value()

    def show_result(self, text: str) -> None:
        self.txt_result.setPlainText(text)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _update_methods(self) -> None:
        self.cmb_method.clear()
        family = _FAMILIES.get(self.get_family_name(), [])
        for cls in family:
            self.cmb_method.addItem(cls().get_name())
