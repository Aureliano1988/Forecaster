"""Dialog for entering reservoir parameters (STOIIP and HCPV)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


class ReservoirDataDialog(QDialog):
    """Modal dialog for STOIIP and HCPV input."""

    def __init__(
        self,
        stoiip: float = 0.0,
        hcpv: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Данные пласта")
        self.setMinimumWidth(360)

        lay = QVBoxLayout(self)

        note = QLabel(
            "Введите начальные запасы нефти (STOIIP) и гидродинамически\n"
            "связанный поровый объём (HCPV). Нулевые значения отключают\n"
            "расчёт КИН (RF) и НГНПИ (HCPVI) на графиках прогнозов."
        )
        note.setWordWrap(True)
        lay.addWidget(note)

        form = QFormLayout()

        self._spn_stoiip = QDoubleSpinBox()
        self._spn_stoiip.setRange(0.0, 1e12)
        self._spn_stoiip.setDecimals(0)
        self._spn_stoiip.setSuffix(" т")
        self._spn_stoiip.setValue(stoiip)
        self._spn_stoiip.setSingleStep(1000.0)
        self._spn_stoiip.setGroupSeparatorShown(True)
        form.addRow("STOIIP (нач. запасы нефти):", self._spn_stoiip)

        self._spn_hcpv = QDoubleSpinBox()
        self._spn_hcpv.setRange(0.0, 1e12)
        self._spn_hcpv.setDecimals(0)
        self._spn_hcpv.setSuffix(" м³")
        self._spn_hcpv.setValue(hcpv)
        self._spn_hcpv.setSingleStep(1000.0)
        self._spn_hcpv.setGroupSeparatorShown(True)
        form.addRow("HCPV (поровый объём пласта):", self._spn_hcpv)

        lay.addLayout(form)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def get_stoiip(self) -> float:
        return self._spn_stoiip.value()

    def get_hcpv(self) -> float:
        return self._spn_hcpv.value()
