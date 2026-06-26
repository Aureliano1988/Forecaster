"""Shared legend helper — creates compact, draggable legends that fit the plot."""

from __future__ import annotations


def fit_legend(
    ax,
    handles=None,
    labels=None,
    *,
    extra_ax=None,
    loc="best",
    max_entries: int = 40,
):
    """Create a legend that adapts its layout to fit within the plot area.

    - Collects handles/labels from *ax* (and optionally *extra_ax* for twinx).
    - Deduplicates entries with the same label.
    - Truncates when there are more than *max_entries*.
    - Picks font size and column count so the legend stays compact.
    - Makes the legend draggable.

    If *handles*/*labels* are supplied they are used directly (no collection
    from the axes); *extra_ax* is still merged when given.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles, labels = list(handles), list(labels)

    if extra_ax is not None:
        h2, l2 = extra_ax.get_legend_handles_labels()
        handles += h2
        labels += l2

    if not handles:
        return None

    # Deduplicate: keep first occurrence of each label
    seen: set[str] = set()
    deduped_h, deduped_l = [], []
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen.add(lbl)
            deduped_h.append(h)
            deduped_l.append(lbl)
    handles, labels = deduped_h, deduped_l

    n = len(labels)
    if n == 0:
        return None

    # Truncate very large legends
    if n > max_entries:
        handles = handles[: max_entries - 1]
        labels = labels[: max_entries - 1] + [f"\u2026 ещё {n - max_entries + 1}"]
        n = max_entries

    # Adaptive font size and column count
    if n <= 5:
        fontsize, ncol = 8, 1
    elif n <= 10:
        fontsize, ncol = 7, 2
    elif n <= 20:
        fontsize, ncol = 6, max(2, (n + 7) // 8)
    elif n <= 30:
        fontsize, ncol = 5, max(3, (n + 7) // 8)
    else:
        fontsize, ncol = 5, max(4, (n + 7) // 8)

    # Place legend on the topmost axes layer so it stays draggable
    # when a twinx overlay is present.
    target = extra_ax if extra_ax is not None else ax
    leg = target.legend(handles, labels, fontsize=fontsize, loc=loc, ncol=ncol)
    if leg is not None:
        leg.set_draggable(True)
    return leg
