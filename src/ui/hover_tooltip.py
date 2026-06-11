"""Reusable hover tooltip for matplotlib canvases.

Shows the label of the nearest plotted series (Line2D, PathCollection,
PolyCollection, etc.) when the user hovers over it.  Survives
``figure.clear()`` / axes recreation cycles.

Performance: labeled artists are cached per-axes and rebuilt lazily only
when the set of axes objects changes (e.g. after ``figure.clear()``).
"""

from __future__ import annotations


def install_hover_tooltip(canvas, figure) -> None:
    """Attach a hover tooltip to a matplotlib *canvas*.

    When the cursor hovers over an artist whose label does not start with
    ``_``, an annotation box with the label text appears near the cursor.
    """
    _state: dict = {
        "annot": None,
        "last_label": "",
        # Artist cache: list of (Axes, artist, label) tuples.
        # Rebuilt lazily when the set of Axes objects changes.
        "artist_cache": [],
        "cached_axes": [],
    }

    def _ensure_annot(ax):
        """Return a valid annotation on *ax*, creating one if needed."""
        a = _state["annot"]
        try:
            if a is not None and a.axes is ax:
                return a
        except Exception:
            pass
        # Old annotation was on a cleared/different axes — create a new one
        _state["annot"] = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.9),
            fontsize=8,
            zorder=100,
            annotation_clip=False,
        )
        _state["annot"].set_visible(False)
        _state["last_label"] = ""
        return _state["annot"]

    def _ensure_cache():
        """Return cached labeled artists, rebuilding when axes change."""
        axes = figure.axes
        if axes == _state["cached_axes"]:
            return _state["artist_cache"]
        cache = []
        for ax in axes:
            for artist in ax.get_children():
                try:
                    lbl = str(artist.get_label())
                except Exception:
                    continue
                if lbl and not lbl.startswith("_"):
                    cache.append((ax, artist, lbl))
        _state["artist_cache"] = cache
        _state["cached_axes"] = list(axes)
        return cache

    def _on_move(event):
        if event.inaxes is None:
            a = _state["annot"]
            if a is not None:
                try:
                    if a.get_visible():
                        a.set_visible(False)
                        _state["last_label"] = ""
                        canvas.draw_idle()
                except Exception:
                    _state["annot"] = None
            return

        if event.xdata is None or event.ydata is None:
            return

        ax = event.inaxes
        found_label = None

        # Use cached labeled artists instead of iterating all children
        cache = _ensure_cache()
        ax_bb = ax.bbox.bounds
        for cached_ax, artist, label in cache:
            # Match axes sharing the same position (e.g. twinx pairs)
            if cached_ax.bbox.bounds != ax_bb:
                continue
            try:
                contained, _ = artist.contains(event)
            except Exception:
                continue
            if contained:
                found_label = label
                break

        annot = _ensure_annot(ax)
        if found_label:
            if found_label != _state["last_label"] or not annot.get_visible():
                annot.set_text(found_label)
                annot.set_visible(True)
                annot.xy = (event.xdata, event.ydata)
                _state["last_label"] = found_label
                # Flip tooltip left/right so it stays inside the axes
                try:
                    renderer = canvas.get_renderer()
                    bb = annot.get_window_extent(renderer)
                    ax_bb_ext = ax.get_window_extent(renderer)
                    if bb.x1 > ax_bb_ext.x1:          # overflows right
                        annot.set_anncoords("offset points")
                        annot.xyann = (-15 - bb.width, 15)
                    elif bb.x0 < ax_bb_ext.x0:         # overflows left
                        annot.set_anncoords("offset points")
                        annot.xyann = (15, 15)
                    else:
                        annot.xyann = (15, 15)
                except Exception:
                    pass
                canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                _state["last_label"] = ""
                canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", _on_move)
