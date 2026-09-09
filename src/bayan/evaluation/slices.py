"""Lab 6: sliced classification evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


DEFAULT_SLICES = (
    "lang",
    "dialect_region",
    "class",
    "length_bucket",
)


def _macro_f1(frame):
    return float(
        f1_score(
            frame["y_true"],
            frame["y_pred"],
            average="macro",
            zero_division=0,
        )
    )


def _bootstrap_macro_f1(frame, *, n_boot, seed, alpha):
    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(n_boot):
        positions = rng.integers(0, len(frame), size=len(frame))
        sample = frame.iloc[positions]
        scores.append(_macro_f1(sample))

    lower, upper = np.quantile(
        scores,
        [alpha / 2, 1 - alpha / 2],
    )

    return float(lower), float(upper)


def sliced_report(
    data,
    *,
    slice_columns=DEFAULT_SLICES,
    min_size=30,
    n_boot=500,
    seed=42,
    alpha=0.05,
):
    """Return macro-F1 and bootstrap CIs for useful evaluation slices."""
    frame = pd.DataFrame(data).copy()

    required = {"y_true", "y_pred"}
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError("Evaluation data must not be empty")

    rows = []

    def add_row(slice_name, slice_value, selected, row_seed):
        lower, upper = _bootstrap_macro_f1(
            selected,
            n_boot=n_boot,
            seed=row_seed,
            alpha=alpha,
        )

        rows.append(
            {
                "slice": slice_name,
                "value": str(slice_value),
                "count": int(len(selected)),
                "macro_f1": _macro_f1(selected),
                "ci_lower": lower,
                "ci_upper": upper,
                "small_slice": bool(len(selected) < min_size),
            }
        )

    add_row("overall", "all", frame, seed)

    row_seed = seed + 1

    for column in slice_columns:
        source_column = "y_true" if column == "class" else column

        if source_column not in frame.columns:
            continue

        values = frame[source_column].fillna("unknown")

        for value in sorted(values.unique(), key=str):
            selected = frame.loc[values == value]

            add_row(
                column,
                value,
                selected,
                row_seed,
            )

            row_seed += 1

    return pd.DataFrame(rows)