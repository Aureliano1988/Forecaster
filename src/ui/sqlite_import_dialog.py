"""SQLite import dialog — browse a production DB and pick fields/objects.

Lets the user open a ``.sqldb``/``.db``/``.sqlite`` database, multi-select
one or more oilfields, then multi-select one or more objects belonging to
those oilfields, and load the combined production data for all of them in
one go.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from src.data.sqlite_loader import (
    SQLiteLoaderError,
    build_dataframe_for_objects,
    list_objects,
    list_oilfields,
    open_connection,
)


class SQLiteImportDialog(QDialog):
    """Dialog to pick oilfields/objects from a SQLite DB and load their data.

    Usage::

        dlg = SQLiteImportDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            df = dlg.result_dataframe()
            db_path = dlg.result_db_path()
            specs = dlg.result_object_specs()   # for project persistence
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Загрузка данных из БД (SQLite)")
        self.resize(760, 520)

        self._db_path: str = ""
        self._conn = None   # sqlite3.Connection, reused for this dialog's lifetime
        self._objects_cache: list[dict] = []   # last list_objects() result
        self._object_specs: list[dict] = []    # accepted selection
        self._result_df = None
        self._result_well_coords: dict[str, tuple[float, float]] = {}

        self._build_ui()
        self._update_ok_enabled()
        self.finished.connect(self._on_finished)

    # ── Public API ───────────────────────────────────────────────────────

    def result_dataframe(self):
        return self._result_df

    def result_db_path(self) -> str:
        return self._db_path

    def result_object_specs(self) -> list[dict]:
        return self._object_specs

    def result_well_coords(self) -> dict[str, tuple[float, float]]:
        """Well coordinates loaded from DW_PR_COORDS alongside production data."""
        return self._result_well_coords

    # ── Construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ── File picker row ─────────────────────────────────────────────
        file_row = QHBoxLayout()
        btn_open = QPushButton("Открыть базу данных…")
        btn_open.clicked.connect(self._on_open_db)
        file_row.addWidget(btn_open)
        self._lbl_db = QLabel("База данных не выбрана")
        self._lbl_db.setStyleSheet("color: #555;")
        file_row.addWidget(self._lbl_db, stretch=1)
        root.addLayout(file_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ── Oilfields panel ─────────────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        grp_fields = QGroupBox("Месторождения")
        gf_lay = QVBoxLayout(grp_fields)
        self._lst_fields = QListWidget()
        self._lst_fields.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        gf_lay.addWidget(self._lst_fields)
        row_f = QHBoxLayout()
        btn_f_all = QPushButton("Все")
        btn_f_none = QPushButton("Снять")
        btn_f_all.clicked.connect(self._lst_fields.selectAll)
        btn_f_none.clicked.connect(self._lst_fields.clearSelection)
        row_f.addWidget(btn_f_all)
        row_f.addWidget(btn_f_none)
        row_f.addStretch()
        gf_lay.addLayout(row_f)
        left_lay.addWidget(grp_fields)
        splitter.addWidget(left)

        # ── Objects panel ───────────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        grp_objs = QGroupBox("Объекты")
        go_lay = QVBoxLayout(grp_objs)
        self._lst_objects = QListWidget()
        self._lst_objects.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        go_lay.addWidget(self._lst_objects)
        row_o = QHBoxLayout()
        btn_o_all = QPushButton("Все")
        btn_o_none = QPushButton("Снять")
        btn_o_all.clicked.connect(self._lst_objects.selectAll)
        btn_o_none.clicked.connect(self._lst_objects.clearSelection)
        row_o.addWidget(btn_o_all)
        row_o.addWidget(btn_o_none)
        row_o.addStretch()
        go_lay.addLayout(row_o)
        right_lay.addWidget(grp_objs)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self._lbl_info = QLabel("")
        self._lbl_info.setStyleSheet("color: #555;")
        root.addWidget(self._lbl_info)

        # ── Buttons ─────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons = buttons
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Загрузить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._lst_fields.itemSelectionChanged.connect(self._on_fields_changed)
        self._lst_objects.itemSelectionChanged.connect(self._update_ok_enabled)

    # ── File handling ────────────────────────────────────────────────────

    def _on_open_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть базу данных", "",
            "SQLite базы (*.sqldb *.db *.sqlite);;Все файлы (*)",
        )
        if not path:
            return
        try:
            new_conn = open_connection(path)
            fields = list_oilfields(new_conn)
        except SQLiteLoaderError as exc:
            QMessageBox.critical(self, "Ошибка чтения БД", str(exc))
            return
        if not fields:
            QMessageBox.warning(
                self, "Нет данных",
                "В базе данных не найдено ни одного месторождения.",
            )
            new_conn.close()
            return

        # Replace any previously-open connection (e.g. user picked a different DB)
        if self._conn is not None:
            self._conn.close()
        self._conn = new_conn

        self._db_path = path
        self._lbl_db.setText(path)
        self._lst_objects.clear()
        self._objects_cache = []

        self._lst_fields.blockSignals(True)
        self._lst_fields.clear()
        for f in fields:
            item = QListWidgetItem(f["oilfield_name"])
            item.setData(Qt.ItemDataRole.UserRole, f)
            self._lst_fields.addItem(item)
        self._lst_fields.blockSignals(False)

        self._update_ok_enabled()

    # ── Field → object cascading ─────────────────────────────────────────

    def _on_fields_changed(self) -> None:
        self._lst_objects.clear()
        self._objects_cache = []

        field_items = self._lst_fields.selectedItems()
        if not field_items or self._conn is None:
            self._update_ok_enabled()
            return

        oilfield_ids = [
            item.data(Qt.ItemDataRole.UserRole)["oilfield_id"] for item in field_items
        ]
        try:
            objects = list_objects(self._conn, oilfield_ids)
        except SQLiteLoaderError as exc:
            QMessageBox.critical(self, "Ошибка чтения БД", str(exc))
            self._update_ok_enabled()
            return

        self._objects_cache = objects
        multi_field = len(field_items) > 1
        for obj in objects:
            label = f"{obj['object_name']}"
            if multi_field:
                label += f" — {obj['oilfield_name']}"
            label += f" [{obj['record_count']} зап.]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, obj)
            self._lst_objects.addItem(item)

        self._update_ok_enabled()

    # ── OK enablement ────────────────────────────────────────────────────

    def _update_ok_enabled(self) -> None:
        enabled = bool(self._lst_objects.selectedItems())
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enabled)

    # ── Accept / load ────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        selected_specs = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._lst_objects.selectedItems()
        ]
        if not selected_specs:
            return

        try:
            df, conflicts, well_coords = build_dataframe_for_objects(
                self._conn, selected_specs, include_coords=True
            )
        except SQLiteLoaderError as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            return

        if conflicts:
            lines = [
                f"  {c['well']} → {c['new_name']}" for c in conflicts
            ]
            QMessageBox.information(
                self, "Обнаружены повторяющиеся имена скважин",
                "Следующие скважины имеют одинаковое имя в разных "
                "объектах/месторождениях и были автоматически переименованы "
                "во избежание потери данных:\n\n" + "\n".join(lines),
            )

        # Persist only the identifying fields (drop the ephemeral record count)
        self._object_specs = [
            {
                "oilfield_id": s["oilfield_id"],
                "oilfield_name": s["oilfield_name"],
                "object_id": s["object_id"],
                "object_name": s["object_name"],
            }
            for s in selected_specs
        ]
        self._result_df = df
        self._result_well_coords = well_coords
        self.accept()

    # ── Cleanup ────────────────────────────────────────────────

    def _on_finished(self, _result: int) -> None:
        """Close the shared connection once the dialog is done (accept or cancel)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
