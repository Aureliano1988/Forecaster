"""Matplotlib canvas widget with interactive lasso selection."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PlotWidget(QWidget):
    """Embeds a Matplotlib figure inside a Qt widget.

    Provides:
    - ``ax`` — main axes for plotting.
    - ``enable_lasso_selector(callback)`` — free-draw polygon point selection.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self._lasso: LassoSelector | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    # ── Public API ───────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self.ax.clear()
        self.canvas.draw_idle()

    def redraw(self) -> None:
        self.ax.legend(fontsize=8)
        self.canvas.draw_idle()

    def enable_lasso_selector(self, callback) -> None:
        """Activate freehand lasso selector.

        *callback(vertices)* is called when the user finishes drawing;
        *vertices* is a list of ``(x, y)`` coordinate pairs forming the polygon.
        """
        self._lasso = LassoSelector(
            self.ax,
            callback,
            useblit=True,
            props=dict(color="tab:blue", linewidth=1.5, linestyle="--"),
        )

    def disable_lasso_selector(self) -> None:
        if self._lasso is not None:
            self._lasso.set_active(False)
            self._lasso = None
