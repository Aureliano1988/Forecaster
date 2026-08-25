"""Load production data directly from a KIN_BN_BUR-style SQLite database.

Mirrors the mapping used by ``Extract-Production-Script-SEQUENTIAL.ps1``
(``DW_OILFIELD`` / ``DW_OBJECT`` / ``DW_MTH_OP_RAP`` / ``DW_WELLS``), but:

- reads with Python's built-in ``sqlite3`` module (no external ``sqlite3.exe``
  dependency);
- resolves the water production/injection column duality: for producing
  months (``OIL_T > 0``) the tonnes column ``WAT_LIQ_INJ_T`` is populated,
  but for non-producing/injection months it is always zero while the real
  volume lives in ``WAT_LIQ_INJ_M3`` — so the water value is read from
  whichever column is meaningful for that row (water density is assumed
  to be approximately 1.0 t/m3 when falling back to the m3 reading);
- feeds the resolved value in as :data:`COL_WATER_DUAL` so the existing
  production/injection auto-split in ``src/data/loader.py`` applies;
- detects well-name collisions (the same name used by more than one
  ``WELL_ID`` across the objects/fields being loaded together) and
  disambiguates them automatically.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.data.loader import apply_manual_mapping
from src.data.models import (
    COL_CONDENSATE,
    COL_DATE,
    COL_GAS,
    COL_HOURS_WORK,
    COL_OIL,
    COL_WATER_DUAL,
    COL_WELL,
)

# Raw column names produced by the extraction query below, mapped to the
# internal column constants used throughout the rest of the app.
_COL_MAPPING: dict[str, str] = {
    "Date": COL_DATE,
    "Well": COL_WELL,
    "Oil_T": COL_OIL,
    "Water_T": COL_WATER_DUAL,
    "Gas_M3": COL_GAS,
    "Condensate_T": COL_CONDENSATE,
    "Hours_Work": COL_HOURS_WORK,
}

_PRODUCTION_QUERY = """
    SELECT substr(r.DT, 1, 10) AS Date,
           r.WELL_ID AS WellId,
           COALESCE(w.WELL_NAME, 'Well_' || r.WELL_ID) AS Well,
           ROUND(COALESCE(r.OIL_T, 0.0), 3) AS Oil_T,
           ROUND(
               CASE WHEN COALESCE(r.OIL_T, 0.0) > 0
                    THEN COALESCE(r.WAT_LIQ_INJ_T, 0.0)
                    ELSE COALESCE(r.WAT_LIQ_INJ_M3, 0.0)
               END, 3
           ) AS Water_T,
           ROUND(COALESCE(r.GAS_NAT_M3, 0.0), 3) AS Gas_M3,
           ROUND(COALESCE(r.GAS_COND_T, 0.0), 3) AS Condensate_T,
           ROUND(COALESCE(r.WORKTIME, 0.0), 2) AS Hours_Work
    FROM DW_MTH_OP_RAP r
    LEFT JOIN DW_WELLS w ON r.WELL_ID = w.WELL_ID
    WHERE r.OBJECT_ID = ?
    ORDER BY r.DT, w.WELL_NAME
"""

# Well coordinates are object-scoped in DW_PR_COORDS (the same well has one
# row per productive layer/object it is completed in, with a slightly
# different X/Y reflecting the deviated wellbore's position at that layer's
# depth). Duplicate (OBJECT_ID, WELL_ID) rows are genuine survey revisions
# over time — keep only the most recent one (latest UPSERTED_ON, tie-break
# highest NO). Rows with (X=0 AND Y=0) mean "no coordinate" and are excluded.
_COORDS_QUERY_TEMPLATE = """
    SELECT ObjectId, WellId, Well, X, Y FROM (
        SELECT c.OBJECT_ID AS ObjectId,
               c.WELL_ID AS WellId,
               COALESCE(w.WELL_NAME, 'Well_' || c.WELL_ID) AS Well,
               c.X AS X,
               c.Y AS Y,
               ROW_NUMBER() OVER (
                   PARTITION BY c.OBJECT_ID, c.WELL_ID
                   ORDER BY c.UPSERTED_ON DESC, c.NO DESC
               ) AS rn
        FROM DW_PR_COORDS c
        LEFT JOIN DW_WELLS w ON w.WELL_ID = c.WELL_ID
        WHERE c.OBJECT_ID IN ({placeholders})
          AND NOT (c.X = 0 AND c.Y = 0)
    ) t
    WHERE rn = 1
"""


class SQLiteLoaderError(Exception):
    """Raised when the database cannot be read or has an unexpected schema."""


def list_oilfields(db_path: str) -> list[dict]:
    """Return every oilfield defined in the database.

    Each entry: ``{"oilfield_id": int, "oilfield_name": str}``.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                "SELECT OILFIELD_ID, OILFIELD_NAME FROM DW_OILFIELD "
                "ORDER BY OILFIELD_NAME"
            )
            return [
                {"oilfield_id": row[0], "oilfield_name": row[1]}
                for row in cur.fetchall()
            ]
    except sqlite3.Error as exc:
        raise SQLiteLoaderError(f"Не удалось прочитать месторождения: {exc}") from exc


def list_objects(db_path: str, oilfield_ids: list[int]) -> list[dict]:
    """Return objects with production data for any of *oilfield_ids*.

    Each entry: ``{"object_id", "object_name", "oilfield_id",
    "oilfield_name", "record_count"}``.
    """
    if not oilfield_ids:
        return []
    placeholders = ",".join("?" for _ in oilfield_ids)
    query = f"""
        SELECT DISTINCT o.OBJECT_ID, o.OBJECT_NAME, o.OILFIELD_ID, f.OILFIELD_NAME,
               (SELECT COUNT(*) FROM DW_MTH_OP_RAP r WHERE r.OBJECT_ID = o.OBJECT_ID)
               AS RecordCount
        FROM DW_OBJECT o
        JOIN DW_OILFIELD f ON f.OILFIELD_ID = o.OILFIELD_ID
        WHERE o.OILFIELD_ID IN ({placeholders})
          AND EXISTS (SELECT 1 FROM DW_MTH_OP_RAP r WHERE r.OBJECT_ID = o.OBJECT_ID)
        ORDER BY f.OILFIELD_NAME, o.OBJECT_NAME
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(query, oilfield_ids)
            return [
                {
                    "object_id": row[0],
                    "object_name": row[1],
                    "oilfield_id": row[2],
                    "oilfield_name": row[3],
                    "record_count": row[4],
                }
                for row in cur.fetchall()
            ]
    except sqlite3.Error as exc:
        raise SQLiteLoaderError(f"Не удалось прочитать объекты: {exc}") from exc


def _fetch_object_production_conn(conn: sqlite3.Connection, object_id: int) -> pd.DataFrame:
    cur = conn.execute(_PRODUCTION_QUERY, (object_id,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def fetch_object_production(db_path: str, object_id: int) -> pd.DataFrame:
    """Return raw monthly production rows for a single object.

    Columns: ``Date, WellId, Well, Oil_T, Water_T, Gas_M3, Condensate_T,
    Hours_Work`` (before internal column renaming). ``Water_T`` is already
    resolved between the tonnes/m3 columns as described in the module
    docstring.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            return _fetch_object_production_conn(conn, object_id)
    except sqlite3.Error as exc:
        raise SQLiteLoaderError(f"Не удалось прочитать данные добычи: {exc}") from exc


def _fetch_object_coords_conn(
    conn: sqlite3.Connection, object_ids: list[int]
) -> pd.DataFrame:
    """Return one coordinate row per (ObjectId, WellId) for *object_ids*.

    Columns: ``ObjectId, WellId, Well, X, Y``. Zero-coordinate rows are
    excluded; duplicate revisions are collapsed to the most recent one.
    """
    if not object_ids:
        return pd.DataFrame(columns=["ObjectId", "WellId", "Well", "X", "Y"])
    placeholders = ",".join("?" for _ in object_ids)
    query = _COORDS_QUERY_TEMPLATE.format(placeholders=placeholders)
    cur = conn.execute(query, object_ids)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def fetch_object_coords(db_path: str, object_ids: list[int]) -> pd.DataFrame:
    """Return resolved well coordinates for *object_ids* (see module docstring)."""
    try:
        with sqlite3.connect(db_path) as conn:
            return _fetch_object_coords_conn(conn, object_ids)
    except sqlite3.Error as exc:
        raise SQLiteLoaderError(f"Не удалось прочитать координаты скважин: {exc}") from exc


def build_dataframe_for_objects(
    db_path: str, object_specs: list[dict], include_coords: bool = True
) -> tuple[pd.DataFrame, list[dict], dict[str, tuple[float, float]]]:
    """Load and combine production data (and optionally coordinates) for
    several oilfield/object picks.

    *object_specs* — list of ``{"oilfield_id", "oilfield_name", "object_id",
    "object_name"}`` dicts (as returned by :func:`list_objects`).

    Returns ``(df, conflicts, well_coords)`` where:

    - *df* is a normalised, ready-to-use production DataFrame (same shape as
      ``load_file()``/``apply_manual_mapping()`` output);
    - *conflicts* is a list of ``{"well", "well_id", "object_name",
      "new_name"}`` dicts describing every well name that was automatically
      disambiguated because it was shared by more than one distinct well
      across the loaded objects/fields;
    - *well_coords* is ``{final_well_name: (X, Y)}`` from ``DW_PR_COORDS``,
      using the *same* disambiguated well names as *df* (empty dict when
      *include_coords* is ``False`` or no coordinates were found). When a
      well has coordinates under more than one of the selected objects, the
      object listed *earliest* in *object_specs* wins.
    """
    if not object_specs:
        raise SQLiteLoaderError("Не выбрано ни одного объекта для загрузки.")

    try:
        with sqlite3.connect(db_path) as conn:
            chunks: list[pd.DataFrame] = []
            for spec in object_specs:
                chunk = _fetch_object_production_conn(conn, spec["object_id"])
                if chunk.empty:
                    continue
                chunk["_object_name"] = spec["object_name"]
                chunk["_oilfield_name"] = spec["oilfield_name"]
                chunks.append(chunk)

            coords_raw = pd.DataFrame()
            if include_coords:
                coords_raw = _fetch_object_coords_conn(
                    conn, [s["object_id"] for s in object_specs]
                )
    except sqlite3.Error as exc:
        raise SQLiteLoaderError(f"Не удалось прочитать данные добычи: {exc}") from exc

    if not chunks:
        raise SQLiteLoaderError("Выбранные объекты не содержат данных добычи.")

    raw = pd.concat(chunks, ignore_index=True)
    raw["_orig_order"] = range(len(raw))

    conflicts, rename_map = _resolve_well_name_conflicts(raw)

    df = apply_manual_mapping(raw, _COL_MAPPING)

    well_coords: dict[str, tuple[float, float]] = {}
    if include_coords and not coords_raw.empty:
        object_order = {s["object_id"]: i for i, s in enumerate(object_specs)}
        coords_raw = coords_raw.assign(
            _obj_order=coords_raw["ObjectId"].map(object_order)
        ).sort_values("_obj_order")
        for row in coords_raw.itertuples(index=False):
            final_name = rename_map.get((row.Well, row.WellId), row.Well)
            if final_name not in well_coords:
                well_coords[final_name] = (float(row.X), float(row.Y))

    return df, conflicts, well_coords


def _resolve_well_name_conflicts(
    raw: pd.DataFrame,
) -> tuple[list[dict], dict[tuple[str, int], str]]:
    """Disambiguate well names shared by more than one WELL_ID, in place.

    For every well name used by more than one distinct ``WellId``, the
    first-encountered ``WellId`` keeps the original name; every other
    ``WellId`` sharing that name is renamed to ``"{name} [{object_name}]"``.

    Returns ``(conflicts, rename_map)`` where *rename_map* maps
    ``(original_well_name, well_id) -> final_well_name`` for every renamed
    well, so callers needing the same disambiguation elsewhere (e.g. for
    coordinates) can apply it consistently.
    """
    conflicts: list[dict] = []
    rename_map: dict[tuple[str, int], str] = {}

    for well_name, grp in raw.groupby("Well"):
        distinct_ids = grp["WellId"].unique()
        if len(distinct_ids) <= 1:
            continue
        first_seen = grp.groupby("WellId")["_orig_order"].min().sort_values()
        ordered_ids = first_seen.index.tolist()
        # Keep the first-seen WellId under the original name; disambiguate the rest.
        for well_id in ordered_ids[1:]:
            sub = grp[grp["WellId"] == well_id]
            object_name = sub["_object_name"].iloc[0]
            new_name = f"{well_name} [{object_name}]"
            rename_map[(well_name, well_id)] = new_name
            conflicts.append({
                "well": well_name,
                "well_id": int(well_id),
                "object_name": object_name,
                "new_name": new_name,
            })

    if rename_map:
        raw["Well"] = [
            rename_map.get((well, well_id), well)
            for well, well_id in zip(raw["Well"], raw["WellId"])
        ]

    return conflicts, rename_map
