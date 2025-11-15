#!/usr/bin/env python3
"""
Random question picker from a single CSV file.
Similar to build_random_question_pick.py but takes an input CSV file instead of a directory.
"""
import sys
from pathlib import Path
import pandas as pd


# === CONFIGURATION ===
INPUT_CSV = Path(__file__).parent / "sanitized_test_data_humanities/base_question_professional_law_test.csv"
OUTPUT_SAMPLE = Path(__file__).parent / "random_500_sanitized_professional_law.csv"
RANDOM_SEED = 42  # set for reproducibility
SAMPLE_SIZE = 500


def main(input_csv=None, output_csv=None, sample_size=None, random_seed=None) -> None:
    """
    Sample random questions from an input CSV file.
    
    Args:
        input_csv: Path to input CSV file (overrides INPUT_CSV constant)
        output_csv: Path to output CSV file (overrides OUTPUT_SAMPLE constant)
        sample_size: Number of questions to sample (overrides SAMPLE_SIZE constant)
        random_seed: Random seed for reproducibility (overrides RANDOM_SEED constant)
    """
    # Use provided arguments or fall back to constants
    input_path = Path(input_csv) if input_csv else INPUT_CSV
    output_path = Path(output_csv) if output_csv else OUTPUT_SAMPLE
    n_samples = sample_size if sample_size is not None else SAMPLE_SIZE
    seed = random_seed if random_seed is not None else RANDOM_SEED
    
    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Input CSV file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.is_file():
        print(f"Error: Input path is not a file: {input_path}")
        sys.exit(1)
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV file {input_path}: {e}")
        sys.exit(1)
    
    if df.empty:
        print(f"Error: Input CSV is empty: {input_path}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows from: {input_path}")
    
    # Reassign unique incremental QID (drop existing if present)
    if "QID" in df.columns:
        df = df.drop(columns=["QID"])
        print("Dropped existing QID column")
    
    df.insert(0, "QID", range(1, len(df) + 1))
    print(f"Assigned new QIDs from 1 to {len(df)}")
    
    # Sample random rows
    n = min(n_samples, len(df))
    sample = df.sample(n=n, random_state=seed).sort_values("QID")
    
    # Ensure QID is the first column when writing out
    ordered_cols = ["QID"] + [c for c in sample.columns if c != "QID"]
    sample = sample[ordered_cols]
    
    # Write output
    sample.to_csv(output_path, index=False)
    print(f"✅ Wrote random sample: {output_path} (sampled {n} rows from {len(df)} total)")


if __name__ == "__main__":
    # Simple command line interface
    if len(sys.argv) > 1:
        # Usage: python script.py input.csv [output.csv] [sample_size] [random_seed]
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        size = int(sys.argv[3]) if len(sys.argv) > 3 else None
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
        
        main(input_csv=input_file, output_csv=output_file, sample_size=size, random_seed=seed)
    else:
        # Use default configuration
        print("Using default configuration from constants...")
        main()

