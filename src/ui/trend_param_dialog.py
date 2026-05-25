"""Floating dialog for editing trend parameters as numbers."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class TrendParamDialog(QDialog):
    """Non-modal tool window that shows editable numeric fields for the trend.

    Stays on top of the main window without blocking it.  When a field
    loses focus or the user presses Enter, ``params_changed`` is emitted
    with the complete ``{param_name: float}`` dict.

    The dialog is created once and reused: ``set_method`` rebuilds the form
    for the new method, ``update_params`` silently refreshes the displayed
    values (e.g. after a drag operation).
    """

    params_changed = Signal(dict)   # {param_name: float}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Параметры тренда")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setModal(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._method_label = QLabel()
        self._method_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._method_label)

        # Container rebuilt by set_method()
        self._form_container = QWidget()
        self._form = QFormLayout(self._form_container)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setSpacing(4)
        root.addWidget(self._form_container)

        self._fields: dict[str, QLineEdit] = {}

    # ── Public ──────────────────────────────────────────────────────────────

    def set_method(self, method_name: str, params: dict) -> None:
        """Rebuild the form for a new method and show it."""
        self._method_label.setText(f"<b>{method_name}</b>")
        self._rebuild_form(params)
        self.adjustSize()

    def update_params(self, params: dict) -> None:
        """Silently refresh displayed values (e.g. after a drag)."""
        for name, val in params.items():
            if name in self._fields:
                edit = self._fields[name]
                edit.blockSignals(True)
                edit.setText(f"{float(val):.6g}")
                edit.blockSignals(False)

    # ── Qt overrides ────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Hide instead of destroy when the user presses X."""
        event.ignore()
        self.hide()

    # ── Private ─────────────────────────────────────────────────────────────

    def _rebuild_form(self, params: dict) -> None:
        # Clear existing rows
        while self._form.rowCount():
            self._form.removeRow(0)
        self._fields.clear()

        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)

        for name, val in params.items():
            edit = QLineEdit(f"{float(val):.6g}")
            edit.setValidator(validator)
            edit.setMinimumWidth(140)
            edit.editingFinished.connect(self._on_any_changed)
            self._form.addRow(f"{name}:", edit)
            self._fields[name] = edit

    def _on_any_changed(self) -> None:
        collected: dict[str, float] = {}
        for name, edit in self._fields.items():
            try:
                collected[name] = float(edit.text())
            except ValueError:
                return  # invalid input — don't emit
        self.params_changed.emit(collected)
