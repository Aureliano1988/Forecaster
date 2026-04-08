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

    build_requested = Signal()
    discard_requested = Signal()
    eraser_toggled = Signal(bool)   # True = eraser on
    save_requested = Signal()       # user clicked "Save project"
    autofit_requested = Signal()    # single-method autofit
    autofit_all_requested = Signal()  # autofit all methods

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
        self.spn_horizon.setRange(1, 1200)
        self.spn_horizon.setValue(60)
        grp2_layout.addWidget(self.spn_horizon)

        grp2_layout.addWidget(QLabel("Предел ВНО (WOR):" ))
        self.spn_wor_limit = QSpinBox()
        self.spn_wor_limit.setRange(1, 999)
        self.spn_wor_limit.setValue(99)
        grp2_layout.addWidget(self.spn_wor_limit)
        layout.addWidget(grp2)

        # ── Action buttons ───────────────────────────────────────────────────
        self.btn_build = QPushButton("Построить прогноз")
        self.btn_autofit = QPushButton("Автоподбор")
        self.btn_autofit_all = QPushButton("Автоподбор всех")
        self.btn_discard = QPushButton("Сбросить")
        self.btn_eraser = QPushButton("Ластик (исключить данные)")
        self.btn_eraser.setCheckable(True)
        self.btn_save = QPushButton("Сохранить проект…")
        layout.addWidget(self.btn_build)
        layout.addWidget(self.btn_autofit)
        layout.addWidget(self.btn_autofit_all)
        layout.addWidget(self.btn_discard)
        layout.addWidget(self.btn_eraser)
        layout.addWidget(self.btn_save)

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
        self.btn_autofit.clicked.connect(self.autofit_requested.emit)
        self.btn_autofit_all.clicked.connect(self.autofit_all_requested.emit)
        self.btn_discard.clicked.connect(self.discard_requested.emit)
        self.btn_eraser.toggled.connect(self.eraser_toggled.emit)
        self.btn_save.clicked.connect(self.save_requested.emit)

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

    def get_wor_limit(self) -> float:
        return float(self.spn_wor_limit.value())

    def show_result(self, text: str) -> None:
        self.txt_result.setPlainText(text)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _update_methods(self) -> None:
        self.cmb_method.clear()
        family = _FAMILIES.get(self.get_family_name(), [])
        for cls in family:
            self.cmb_method.addItem(cls().get_name())
