"""Train-only feature selection by quotas or an explicit R2 threshold."""

import numpy as np

from .features.selection import select_r2


def select_domains(scores, catalog, method, options):
    scores = np.asarray(scores)
    if len(scores) != len(catalog) or not np.isfinite(scores).all():
        raise ValueError("Invalid score/catalog alignment")
    if method == "r2" and options.get("mode", "threshold") == "threshold":
        selected, _ = select_r2(scores, float(options["threshold"]))
        return selected, [
            {
                "group": "all_above_threshold",
                "requested": len(selected),
                "available": len(selected),
                "selected": len(selected),
            }
        ]
    selected, receipts = [], []

    def take(mask, number, group):
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"{group} count must be a nonnegative integer")
        indices = np.flatnonzero(mask)
        indices = sorted(indices, key=lambda i: (-scores[i], i))
        receipts.append(
            {
                "group": group,
                "requested": number,
                "available": len(indices),
                "selected": min(number, len(indices)),
            }
        )
        if len(indices) < number:
            raise ValueError(f"{group}: requested {number}, available {len(indices)}")
        selected.extend(indices[:number])

    has_acc = catalog.sensor.eq("acc").any()
    has_mic = catalog.sensor.eq("mic").any()
    acc = options["acc_axis_quota"]
    if has_acc and "acc_axis_quota" not in options:
        raise ValueError("ACC configuration requires explicit counts")
    if set(acc) != {"x", "y", "z"}:
        raise ValueError("ACC configuration requires x, y, z counts")
    for axis, count in acc.items():
        take(catalog.sensor.eq("acc") & catalog.axis.eq(axis), count, "acc_" + axis)
    acc_total = len(selected)
    if method == "j" or (method == "r2" and options.get("mode") == "top"):
        quotas = options["mic_domain_quota"]
        if has_mic and "mic_domain_quota" not in options:
            raise ValueError("MIC configuration requires explicit domain quotas")
        if set(quotas) not in ({"mic_frequency", "mic_time"}, {"mic_joint"}):
            raise ValueError("MIC configuration requires mic_frequency and mic_time")
        for domain, count in quotas.items():
            mask = catalog.sensor.eq("mic") if domain == "mic_joint" else catalog.domain.eq(domain)
            take(mask, count, domain)
        if (
            has_acc
            and has_mic
            and options["require_equal_sensor_totals"]
            and len(selected) - acc_total != acc_total
        ):
            raise ValueError(f"{method.upper()} requires equal ACC and MIC total counts")
    else:
        raise ValueError("Selection method must be j or r2")
    if not selected:
        raise ValueError("No features selected")
    return np.array(selected, dtype=int), receipts
