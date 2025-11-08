import sys
from pathlib import Path
import pandas as pd


INPUT_DIR = Path(__file__).parent / "sanitized_test_data_humanities"
OUTPUT_SAMPLE = Path(__file__).parent / "random_1000_sanitized_humanities.csv"
RANDOM_SEED = 42  # set for reproducibility
SAMPLE_SIZE = 1000


def main() -> None:
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"Input folder not found: {INPUT_DIR}")
        sys.exit(1)

    csv_paths = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in: {INPUT_DIR}")
        sys.exit(0)

    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    if not frames:
        print("No non-empty CSVs to combine.")
        sys.exit(0)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Assign unique incremental QID across the combined dataset (avoid duplicate column)
    if "QID" in combined.columns:
        combined = combined.drop(columns=["QID"])  # drop pre-existing QID
    combined.insert(0, "QID", range(1, len(combined) + 1))

    n = min(SAMPLE_SIZE, len(combined))
    sample = combined.sample(n=n, random_state=RANDOM_SEED).sort_values("QID")
    # Ensure QID is the first column when writing out
    ordered_cols = ["QID"] + [c for c in sample.columns if c != "QID"]
    sample = sample[ordered_cols]
    sample.to_csv(OUTPUT_SAMPLE, index=False)
    print(f"Wrote random sample: {OUTPUT_SAMPLE} (rows={n})")


if __name__ == "__main__":
    main()


