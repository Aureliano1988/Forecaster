"""Matplotlib canvas widget with interactive span selection."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PlotWidget(QWidget):
    """Embeds a Matplotlib figure inside a Qt widget.

    Provides:
    - ``ax`` — main axes for plotting.
    - ``enable_span_selector(callback)`` — interactive X-range selection.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self._span: SpanSelector | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    # ── Public API ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        self.ax.clear()
        self.canvas.draw_idle()

    def redraw(self) -> None:
        self.ax.legend(fontsize=8)
        self.canvas.draw_idle()

    def enable_span_selector(self, callback) -> None:
        """Activate horizontal span selector.

        *callback(xmin, xmax)* is called when the user selects a range.
        """
        self._span = SpanSelector(
            self.ax,
            callback,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True,
        )

    def disable_span_selector(self) -> None:
        if self._span is not None:
            self._span.set_active(False)
            self._span = None
