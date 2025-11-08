import sys
from pathlib import Path
import pandas as pd
import re


INPUT_DIR = Path(__file__).parent / "test"
OUTPUT_DIR = Path(__file__).parent / "sanitized_test_data"

BASE_COLUMNS = [
    "Question",
    "Choice_A",
    "Choice_B",
    "Choice_C",
    "Choice_D",
    "Correct_Answer",
]


def load_politeness_catalog(path: Path) -> pd.DataFrame:
    # Deprecated: no politeness prefixes used in this build.
    return pd.DataFrame(columns=['Politeness_Level', 'Prefix'])


def sanitize_prefix(prefix: str) -> str:
    # Deprecated: no politeness prefixes used in this build.
    return ""


def derive_domain_from_path(p: Path) -> str:
    stem = p.name.rsplit('.', 1)[0]
    stem = stem.replace('_test', '').replace('_', ' ').strip()
    return stem.title() if stem else ""


def format_prompt(row: pd.Series) -> str:
    q = str(row['Question']).strip()
    a = str(row['Choice_A']).strip()
    b = str(row['Choice_B']).strip()
    c = str(row['Choice_C']).strip()
    d = str(row['Choice_D']).strip()
    return f"""{q}

A) {a}
B) {b}
C) {c}
D) {d}

Please select the correct answer."""


def main() -> None:
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"Input folder not found: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in: {INPUT_DIR}")
        sys.exit(0)

    for src in csv_paths:
        # read base question file
        try:
            df = pd.read_csv(src, header=None, names=BASE_COLUMNS)
        except Exception:
            df = pd.read_csv(src)
            missing = [c for c in BASE_COLUMNS if c not in df.columns]
            if missing:
                print(f"Skipping {src.name}: missing columns {missing}")
                continue
            df = df[BASE_COLUMNS]

        df['QID'] = range(1, len(df) + 1)
        domain = derive_domain_from_path(src)

        rows = []
        for _, qrow in df.iterrows():
            qid = int(qrow['QID'])
            answer = str(qrow['Correct_Answer']).strip().upper()
            prompt = format_prompt(qrow)
            rows.append({
                'QID': qid,
                'Domain': domain,
                'Prompt': prompt,
                'Answer': answer,
            })

        out_df = pd.DataFrame(rows, columns=['QID', 'Domain', 'Prompt', 'Answer'])
        out_name = f"base_question_{src.name}"
        out_path = OUTPUT_DIR / out_name
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()


