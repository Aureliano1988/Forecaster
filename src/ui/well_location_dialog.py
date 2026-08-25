"""Well location dialog — plot well X/Y coordinates with name labels.

Takes an already-resolved ``{well_name: (x, y)}`` mapping (from a text
file, a SQLite database, or a merge of both — see
``MainWindow._resolve_all_well_coords()``) and is purely a presentation /
selection component over it. Wells with (0, 0) coordinates are expected to
already be excluded by whoever resolved the mapping.
"""

from __future__ import annotations

import io

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class WellLocationDialog(QDialog):
    """Shows well locations (X, Y) with well-name labels.

    Usage (browse-only, non-modal)::

        dlg = WellLocationDialog(coords, parent=self)
        dlg.show()

    Usage (selection mode, modal)::

        dlg = WellLocationDialog(
            coords, parent=self,
            selection_mode=True, initial_selection=current_wells,
        )
        if dlg.exec() == WellLocationDialog.DialogCode.Accepted:
            wells = dlg.result_selected_wells()

    In selection mode a lasso tool lets the user draw one or more contours
    on the plot; wells enclosed by any drawn contour are added to the
    selection (union across contours).
    """

    def __init__(
        self,
        coords: dict[str, tuple[float, float]],
        parent=None,
        selection_mode: bool = False,
        initial_selection: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Расположение скважин")
        self.resize(900, 700)

        self._coords = coords or {}

        self._selection_mode = selection_mode
        self._selected: set[str] = set(initial_selection or []) if selection_mode else set()
        self._names: list[str] = []
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._lasso: LassoSelector | None = None

        self._build_ui()
        self._load_and_draw()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        self._fig = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._nav = NavigationToolbar2QT(self._canvas, self)
        root.addWidget(self._nav)
        root.addWidget(self._canvas, stretch=1)

        self._lbl_info = QLabel("")
        self._lbl_info.setStyleSheet("color: #555;")
        root.addWidget(self._lbl_info)

        if self._selection_mode:
            sel_row = QHBoxLayout()
            self._btn_lasso = QPushButton("Выделить контуром")
            self._btn_lasso.setCheckable(True)
            self._btn_lasso.setToolTip(
                "Нарисуйте замкнутый контур вокруг скважин для выбора.\n"
                "Можно нарисовать несколько контуров подряд."
            )
            self._btn_lasso.toggled.connect(self._toggle_lasso)
            sel_row.addWidget(self._btn_lasso)
            btn_clear_sel = QPushButton("Очистить выбор")
            btn_clear_sel.clicked.connect(self._on_clear_selection)
            sel_row.addWidget(btn_clear_sel)
            self._lbl_selected = QLabel("")
            self._lbl_selected.setStyleSheet("color: #555;")
            sel_row.addWidget(self._lbl_selected)
            sel_row.addStretch()
            root.addLayout(sel_row)

        btns = QHBoxLayout()
        btn_clip = QPushButton("Копировать график")
        btn_save = QPushButton("Сохранить картинку…")
        btn_clip.clicked.connect(self._to_clipboard)
        btn_save.clicked.connect(self._save_image)
        btns.addWidget(btn_clip)
        btns.addWidget(btn_save)
        btns.addStretch()
        root.addLayout(btns)

        if self._selection_mode:
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok |
                QDialogButtonBox.StandardButton.Cancel
            )
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Выбрать")
            buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            root.addWidget(buttons)

    # ── Data ───────────────────────────────────────────────────

    def _load_and_draw(self) -> None:
        names = sorted(self._coords.keys())
        xs = [self._coords[n][0] for n in names]
        ys = [self._coords[n][1] for n in names]

        self._names, self._xs, self._ys = names, xs, ys
        self._draw()

        self._lbl_info.setText(f"Показано скважин: {len(names)}")
        if self._selection_mode:
            self._update_selection_label()

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw(self) -> None:
        """Redraw the scatter plot from ``self._names/_xs/_ys``.

        Re-creates the axes each time, so an active lasso must be
        re-attached by the caller afterwards if it should remain active.
        """
        names, xs, ys = self._names, self._xs, self._ys
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        if not names:
            ax.set_title("Расположение скважин — нет данных")
            self._canvas.draw_idle()
            return

        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        if self._selection_mode:
            colors = ["tab:red" if n in self._selected else "C0" for n in names]
        else:
            colors = "C0"
        ax.scatter(xs_arr, ys_arr, s=28, c=colors, zorder=3)
        for name, x, y in zip(names, xs, ys):
            selected = self._selection_mode and name in self._selected
            ax.annotate(
                name, (x, y),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7,
                color="tab:red" if selected else "black",
                fontweight="bold" if selected else "normal",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Расположение скважин ({len(names)} шт.)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()

    # ── Selection (lasso) ─────────────────────────────────────────────

    def _toggle_lasso(self, checked: bool) -> None:
        if checked:
            # Deactivate any active toolbar tool (pan / zoom) so the lasso
            # drag gesture is not intercepted by it.
            try:
                if self._nav.mode:
                    self._nav.mode = ""
            except Exception:
                pass
            self._attach_lasso()
        else:
            self._detach_lasso()

    def _attach_lasso(self) -> None:
        self._detach_lasso()
        if not self._fig.axes:
            return
        ax = self._fig.axes[0]
        self._lasso = LassoSelector(
            ax, self._on_lasso_finished, useblit=True,
            props=dict(color="tab:red", linewidth=1.5, linestyle="--"),
        )

    def _detach_lasso(self) -> None:
        if self._lasso is not None:
            try:
                self._lasso.disconnect_events()
            except Exception:
                pass
            self._lasso = None

    def _on_lasso_finished(self, vertices) -> None:
        """Add wells enclosed by the drawn contour to the selection."""
        if len(vertices) < 3 or not self._names:
            return
        path = Path(vertices)
        points = np.column_stack([self._xs, self._ys])
        inside = path.contains_points(points)
        for name, is_in in zip(self._names, inside):
            if is_in:
                self._selected.add(name)
        self._update_selection_label()
        self._draw()   # rebuilds the axes; re-attach the lasso if still active
        if self._btn_lasso.isChecked():
            self._attach_lasso()

    def _on_clear_selection(self) -> None:
        self._selected.clear()
        self._update_selection_label()
        self._draw()
        if self._btn_lasso.isChecked():
            self._attach_lasso()

    def _update_selection_label(self) -> None:
        self._lbl_selected.setText(f"Выбрано: {len(self._selected)}")

    # ── Public API (selection mode) ──────────────────────────────────

    def result_selected_wells(self) -> list[str]:
        """Return the selected well names (only meaningful in selection mode)."""
        return sorted(self._selected)

    # ── Export ───────────────────────────────────────────────────────────

    def _to_clipboard(self) -> None:
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        QApplication.instance().clipboard().setImage(
            QImage.fromData(buf.getvalue())
        )

    def _save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить картинку", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
