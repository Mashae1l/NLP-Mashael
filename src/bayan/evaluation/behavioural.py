"""Lab 6: behavioural test generators and runners."""

from collections import defaultdict

import pandas as pd


NEGATION_MARKERS = (
    "لا",
    "ليس",
    "لم",
    "لن",
    "not",
    "never",
    "no ",
)


def _normalise_spaces(text):
    return " ".join(str(text).split())


def _contains_negation(text):
    lowered = text.casefold()
    return any(marker in lowered for marker in NEGATION_MARKERS)


def run_behavioural_suite(
    tests,
    *,
    topic_predictor=None,
    sentiment_predictor=None,
):
    """Run invariance, directional and minimum-functionality checks."""
    frame = pd.DataFrame(tests).copy()

    required = {
        "test_id",
        "test_type",
        "template",
        "term",
        "expected_relation",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    results = []

    for row in frame.to_dict(orient="records"):
        template = str(row["template"])
        term = str(row["term"])
        rendered = _normalise_spaces(
            template.format(term=term)
        )
        baseline = _normalise_spaces(
            template.format(term="")
        )

        test_type = str(row["test_type"]).casefold()
        passed = False
        details = ""

        if test_type == "invariance":
            if topic_predictor is None:
                passed = (
                    term.casefold() in rendered.casefold()
                    and rendered.replace(term, "").strip()
                    == baseline
                )
                details = "Template invariance contract"
            else:
                baseline_prediction = topic_predictor(baseline)
                rendered_prediction = topic_predictor(rendered)
                passed = baseline_prediction == rendered_prediction
                details = (
                    f"{baseline_prediction} -> "
                    f"{rendered_prediction}"
                )

        elif test_type == "directional":
            if sentiment_predictor is None:
                passed = _contains_negation(rendered)
                details = "Negation direction contract"
            else:
                baseline_score = float(
                    sentiment_predictor(baseline)
                )
                rendered_score = float(
                    sentiment_predictor(rendered)
                )
                passed = rendered_score <= baseline_score
                details = (
                    f"{baseline_score:.4f} -> "
                    f"{rendered_score:.4f}"
                )

        else:
            details = f"Unsupported test type: {test_type}"

        mft_passed = (
            bool(rendered)
            and "{term}" not in rendered
            and term.casefold() in rendered.casefold()
        )

        results.append(
            {
                "test_id": row["test_id"],
                "test_type": test_type,
                "lang": row.get("lang", "unknown"),
                "rendered_text": rendered,
                "passed": bool(passed),
                "mft_passed": bool(mft_passed),
                "details": details,
            }
        )

    result_frame = pd.DataFrame(results)
    summary = defaultdict(dict)

    for test_type, group in result_frame.groupby("test_type"):
        summary[test_type] = {
            "passed": int(group["passed"].sum()),
            "total": int(len(group)),
            "pass_rate": float(group["passed"].mean()),
        }

    summary["mft"] = {
        "passed": int(result_frame["mft_passed"].sum()),
        "total": int(len(result_frame)),
        "pass_rate": float(result_frame["mft_passed"].mean()),
    }

    return {
        "summary": dict(summary),
        "results": result_frame,
    }