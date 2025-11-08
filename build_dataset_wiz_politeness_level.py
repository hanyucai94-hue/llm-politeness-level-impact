# Build a sanitized dataset from a base CSV and politeness catalog
import os
import re
import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
#input_csvs = ["random_500_sanitized_humanities.csv","random_500_sanitized_stem.csv"]  
input_csvs = ["random_1000_sanitized_humanities.csv"]  
POLITENESS_FILE = "Politeness_Level_Prefix_Catalog - V2.csv"
OUTPUT_DIR = "sanitized_test_data_wiz_politeness_level_v2"

# === HELPERS ===
def load_politeness_catalog(path: str) -> pd.DataFrame:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        _ = f.readline()  # header
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "," in line:
                level, prefix = line.split(",", 1)
            else:
                level, prefix = line, ""
            rows.append({
                "Politeness_Level": level.strip(),
                "Prefix": prefix.strip(),
            })
    return pd.DataFrame(rows, columns=["Politeness_Level", "Prefix"])


def sanitize_prefix(prefix: str) -> str:
    if prefix is None:
        return ""
    s = str(prefix).strip().strip("'\"")
    s = re.sub(r"\s+", " ", s)
    if s == "" or s.lower() in {"neutral", "no prefix", "noprefix", "none", "n/a"}:
        return ""
    return s


def derive_domain_from_path(p: str) -> str:
    name = Path(p).name  # e.g., abstract_algebra_test.csv
    stem = name.rsplit(".", 1)[0]
    stem = stem.replace("_test", "")
    stem = stem.replace("_", " ").strip()
    return stem.title() if stem else ""


def format_prompt(base_prompt: str, prefix: str) -> str:
    base = (base_prompt or "").strip()
    return f"{prefix} {base}" if prefix else base


def main():
    import sys
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_df = load_politeness_catalog(POLITENESS_FILE)
    input_paths = [Path(p) for p in input_csvs] if input_csvs else [Path(p) for p in sys.argv[1:]]
    if not input_paths:
        print("No input CSVs provided. Set input_csvs in the script or pass file paths as CLI args.")
        return

    for csv_path in input_paths:
        base_df = pd.read_csv(csv_path)
        required_cols = {"QID", "Domain", "Prompt", "Answer"}
        if not required_cols.issubset(set(base_df.columns)):
            print(f"Skipping {csv_path.name}: missing columns {required_cols - set(base_df.columns)}")
            continue

        rows = []
        for _, qrow in base_df.iterrows():
            qid = int(qrow["QID"])
            domain = str(qrow["Domain"]).strip()
            base_question = str(qrow["Prompt"]).strip()
            answer = str(qrow["Answer"]).strip()
            for _, prow in catalog_df.iterrows():
                level = prow["Politeness_Level"]
                prefix = sanitize_prefix(prow["Prefix"])
                prompt = format_prompt(base_question, prefix)
                rows.append({
                    "QID": qid,
                    "Domain": domain,
                 # "Base Question": base_question,
                    "Politeness Level": level,
                    "Prompt": prompt,
                    "Answer": answer,
                })

        out_df = pd.DataFrame(rows, columns=[
            "QID", "Domain", #"Base Question",
             "Politeness Level", "Prompt", "Answer"
        ])
        output_name = f"dataset+test_{csv_path.name}"
        output_path = out_dir / output_name
        out_df.to_csv(output_path, index=False)
        print(f"Wrote {len(out_df)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()


