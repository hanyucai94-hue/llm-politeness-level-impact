import sys
from pathlib import Path
import pandas as pd


def find_politeness_col(df: pd.DataFrame) -> str:
    candidates = [
        "Politeness Level",
        "Politeness_LEVEL",
        "Politeness_Level",
        "politeness_level",
        "politeness Level",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find politeness column. Available columns: {list(df.columns)}"
    )


def main() -> None:
    # Default path; can be overridden by CLI arg
    default_path = (
        Path(__file__).parent
        / "sanitized_test_data_wiz_politeness_level"
        / "dataset+test_random_500_sanitized_humanities.csv"
    )
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Ensure QID exists
    if "QID" not in df.columns:
        print("Missing required column: QID")
        sys.exit(2)

    # Find politeness column
    politeness_col = find_politeness_col(df)

    # Check duplicates of (QID, Politeness)
    dup_mask = df.duplicated(subset=["QID", politeness_col], keep=False)
    if dup_mask.any():
        dups = (
            df.loc[dup_mask, ["QID", politeness_col]]
            .value_counts()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print("Found duplicate QID+Politeness pairs:\n")
        print(dups.to_string(index=False))
        print(f"\nTotal duplicate rows: {int(dup_mask.sum())}")
        sys.exit(3)
    else:
        print(
            f"OK: All QID + {politeness_col} pairs are unique. Total rows: {len(df)}"
        )


if __name__ == "__main__":
    main()


