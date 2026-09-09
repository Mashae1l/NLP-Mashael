"""Audit the dialect distribution in Bayan's Arabic feedback data."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def main() -> None:
    dataframe = pd.read_csv(DATA_PATH)

    required_columns = {"lang", "dialect_region", "split"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    arabic_data = dataframe[dataframe["lang"] == "ar"].copy()

    counts = (
        arabic_data["dialect_region"]
        .fillna("Unknown")
        .value_counts()
    )

    percentages = (
        arabic_data["dialect_region"]
        .fillna("Unknown")
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("Bayan Arabic dialect audit")
    print("=" * 30)
    print(f"All feedback rows: {len(dataframe)}")
    print(f"Arabic rows: {len(arabic_data)}")
    print()

    print("Dialect distribution:")
    for region, count in counts.items():
        percentage = percentages[region]
        print(f"- {region}: {count} rows ({percentage:.2f}%)")

    print()
    print("Distribution by split:")
    split_table = pd.crosstab(
        arabic_data["dialect_region"],
        arabic_data["split"],
    )
    print(split_table.to_string())

    validation_regions = set(
        arabic_data.loc[
            arabic_data["split"] == "validation",
            "dialect_region",
        ].dropna()
    )

    print()
    if validation_regions == {"MSA"}:
        print(
            "Implication: The Arabic validation split contains only MSA, "
            "so MSA-only evaluation does not measure performance on the "
            "majority Gulf-dialect population."
        )
    else:
        print(
            "Implication: Report results separately for every dialect "
            "represented in the validation split."
        )


if __name__ == "__main__":
    main()