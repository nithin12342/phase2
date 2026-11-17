"""Class-balanced subset selection for pilot runs.

Purpose: a pilot must show both depression and normal cases. The D-Vlog CSV is
front-loaded with one class, so a plain head slice gives no label contrast.
This picks a deterministic, label-balanced subset spread across the full order.
Pure stdlib; no randomness, so a resume selects the same videos.
"""


def stratified_subset(records, limit):
    """Return up to limit records, balanced across labels.

    limit=None (or >= len) returns all records unchanged. Each class gets an
    even quota; picks are evenly spaced within the class so the subset spans
    the whole dataset rather than its head.
    """
    if limit is None or limit >= len(records):
        return list(records)
    by_label = {}
    for i, r in enumerate(records):
        by_label.setdefault(r["label"], []).append(i)
    labels = sorted(by_label)
    base, rem = divmod(limit, len(labels))
    chosen = []
    for k, label in enumerate(labels):
        idxs = by_label[label]
        quota = min(base + (1 if k < rem else 0), len(idxs))
        for i in range(quota):
            chosen.append(idxs[(i * len(idxs)) // quota])
    chosen = sorted(set(chosen))[:limit]
    return [records[i] for i in chosen]