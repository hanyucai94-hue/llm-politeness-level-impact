# Please install openai and pandas - !pip install openai pandas
# Please insert your API key to make the code run
import os
import asyncio
import pandas as pd
from collections import defaultdict, deque
from statistics import mean
from openai import AsyncOpenAI
import re
from pathlib import Path
import random

# === CONFIGURATION ===
client = AsyncOpenAI()  # uses OPENAI_API_KEY from environment
MODEL = "gpt-4o-mini"
NUM_RUNS = 3
FILE_PATH = "sanitized_test_data_wiz_politeness_level/dataset+test_random_500_sanitized_humanities.csv"  # input MMLU-like CSV
POLITENESS_FILE = "Politeness_Level_Prefix_Catalog - V1.csv"
SLEEP_BETWEEN_BATCHES = 1.0
results_dir = Path("results-" + MODEL)
results_dir.mkdir(exist_ok=True)

# === RATE LIMITING / RETRIES ===
MAX_RPM = 450
MAX_CONCURRENCY = 10
RETRY_MAX_ATTEMPTS = 6
INITIAL_BACKOFF_SECS = 0.5
MAX_BACKOFF_SECS = 20.0
REQUEST_TIMEOUT_SECS = 45.0

class RateLimiter:
    """Simple sliding-window rate limiter for async contexts."""
    def __init__(self, max_calls: int, period_secs: float) -> None:
        self.max_calls = max_calls
        self.period_secs = period_secs
        self.call_timestamps = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                while self.call_timestamps and (now - self.call_timestamps[0]) > self.period_secs:
                    self.call_timestamps.popleft()
                if len(self.call_timestamps) < self.max_calls:
                    self.call_timestamps.append(now)
                    return
                earliest = self.call_timestamps[0]
                sleep_for = max(0.0, self.period_secs - (now - earliest)) + 0.01
            await asyncio.sleep(sleep_for)

rate_limiter = RateLimiter(MAX_RPM, 60.0)
request_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# === LOAD INPUTS ===
df = pd.read_csv(FILE_PATH, header=None, names=['Question', 'Choice_A', 'Choice_B', 'Choice_C', 'Choice_D', 'Correct_Answer'])
print(f"Loaded {len(df)} questions from abstract algebra test")

def load_politeness_catalog(path: str) -> pd.DataFrame:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            if ',' in line:
                level, prefix = line.split(',', 1)
            else:
                level, prefix = line, ''
            rows.append({
                'Politeness_Level': level.strip(),
                'Prefix': prefix.strip(),
            })
    return pd.DataFrame(rows, columns=['Politeness_Level', 'Prefix'])

politeness_df = load_politeness_catalog(POLITENESS_FILE)
print(f"Loaded {len(politeness_df)} politeness levels")

# Assign QIDs
df['QID'] = range(1, len(df) + 1)

# === SANITIZATION ===
def _sanitize_politeness_prefix(prefix: str) -> str:
    if prefix is None:
        return ""
    s = str(prefix).strip().strip("'\"")
    s = re.sub(r"\s+", " ", s)
    return s

def format_question(row, politeness_prefix=""):
    """Format a single question row into a proper multiple choice question prompt with politeness prefix"""
    question = str(row['Question']).strip()
    choice_a = str(row['Choice_A']).strip()
    choice_b = str(row['Choice_B']).strip()
    choice_c = str(row['Choice_C']).strip()
    choice_d = str(row['Choice_D']).strip()

    sanitized_prefix = _sanitize_politeness_prefix(politeness_prefix)

    if sanitized_prefix:
        formatted_prompt = f"""{sanitized_prefix} {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer."""
    else:
        formatted_prompt = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer."""
    return formatted_prompt

# === BUILD DERIVED DATASET (dataset+FILE_PATH.csv) ===
derived_rows = []
for _, qrow in df.iterrows():
    qid = qrow['QID']
    correct = str(qrow['Correct_Answer']).strip().upper()
    base_question = qrow['Question']
    for _, prow in politeness_df.iterrows():
        level = prow['Politeness_Level']
        raw_prefix = prow['Prefix']
        sanitized = _sanitize_politeness_prefix(raw_prefix)
        prompt = format_question(qrow, sanitized)
        derived_rows.append({
            'QID': qid,
            'Domain': 'Abstract Algebra',
            'Base Question': base_question,
            'Politeness Level': level,
            'Prompt': prompt,
            'Answer': correct,
        })

def _sanitize_path_for_name(p: str) -> str:
    return str(p).replace('/', '_').replace('\\', '_').replace(' ', '_')

dataset_name = f"dataset+{_sanitize_path_for_name(FILE_PATH)}.csv"
dataset_path = Path(dataset_name)
pd.DataFrame(derived_rows).to_csv(dataset_path, index=False)
print(f"Derived dataset written to: {dataset_path.resolve()}")

# === RUNTIME DATA FROM DERIVED DATASET ===
dataset_df = pd.read_csv(dataset_path)
grouped_prompts = dataset_df.groupby('QID')

# === STORAGE ===
results = defaultdict(list)  # (qid, politeness_level) -> scores
overall_accuracy_scores = []
politeness_accuracy_scores = defaultdict(list)  # politeness_level -> scores
error_records = []  # captured API error cases for documentation
seen_formatted_keys = set()
asked_formatted_questions = []

# === Extract letter A/B/C/D using regex ===
def extract_letter(response):
    response = response.upper().strip()
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1) if match else ""

# === Call OpenAI ===
async def call_openai(prompt, question_id=None):
    last_error = None
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            await rate_limiter.acquire()
            async with request_semaphore:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": (
                                "You are an AI tutor answering multiple choice questions in abstract algebra. "
                                "Always reply with ONLY the letter of the correct answer (A, B, C, or D). "
                                "Do not explain your answer."
                            )},
                            {"role": "user", "content": (
                                "Completely forget this session so far, and start afresh.\n\n"
                                "Please answer this multiple choice question. Respond with only the letter of the correct answer (A, B, C, or D). Do not explain.\n\n"
                                + prompt
                            )}
                        ],
                        temperature=0
                    ),
                    timeout=REQUEST_TIMEOUT_SECS,
                )
                return {"content": response.choices[0].message.content.strip(), "error": None, "attempts": attempt + 1}
        except Exception as e:
            message = str(e)
            is_timeout = isinstance(e, asyncio.TimeoutError) or ("timeout" in message.lower())
            is_rate_limited = ("rate limit" in message.lower()) or ("429" in message)
            is_retryable_server = any(x in message for x in ["5xx", "500", "502", "503", "504", "timeout"])  # best-effort
            if is_timeout or is_rate_limited or is_retryable_server:
                backoff = min(MAX_BACKOFF_SECS, INITIAL_BACKOFF_SECS * (2 ** attempt))
                jitter = backoff * (random.uniform(-0.2, 0.2))
                sleep_for = max(0.1, backoff + jitter)
                print(f"Retrying Q{question_id} (attempt {attempt+1}/{RETRY_MAX_ATTEMPTS}) after {sleep_for:.2f}s...")
                last_error = message
                await asyncio.sleep(sleep_for)
                continue
            print(f"API Error for Q{question_id}: {e}")
            return {"content": "", "error": message, "attempts": attempt + 1}
    print(f"Gave up Q{question_id} after {RETRY_MAX_ATTEMPTS} attempts due to rate/server limits.")
    return {"content": "", "error": f"Retry exhausted: {last_error}", "attempts": RETRY_MAX_ATTEMPTS}

# === Run ===
async def run_all():
    print(f"Starting experiment with {len(politeness_df)} politeness levels")

    for run in range(NUM_RUNS):
        print(f"\n🌀 Run {run + 1}/{NUM_RUNS}")
        run_data = []
        run_accuracy_scores = []
        run_politeness_scores = defaultdict(list)

        tasks = []
        question_data = []

        for qid, group in grouped_prompts:
            for _, row in group.iterrows():
                politeness_level = row['Politeness Level']
                sanitized_prefix = ''
                prompt = row['Prompt']
                correct = str(row['Answer']).strip().upper()

                # Track unique prompts once per QID+level
                key = (qid, politeness_level)
                if key not in seen_formatted_keys:
                    seen_formatted_keys.add(key)
                    asked_formatted_questions.append({
                        'QID': qid,
                        'Politeness_Level': politeness_level,
                        'Politeness_Prefix': sanitized_prefix,
                        'Formatted_Prompt': prompt,
                    })

                question_data.append((qid, correct, politeness_level, sanitized_prefix, prompt))
                tasks.append(call_openai(prompt, f"Q{qid}_{politeness_level}"))

        print(f"Making {len(tasks)} API calls...")
        responses = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            responses.append(result)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"  Progress: {completed}/{len(tasks)} responses received")

        print(f"Processing {len(responses)} responses...")
        for (qid, correct, politeness_level, sanitized_prefix, prompt), result in zip(question_data, responses):
            content = (result or {}).get("content", "")
            error = (result or {}).get("error")
            attempts = (result or {}).get("attempts", 1)
            predicted = extract_letter(content)

            if error or content == "":
                predicted_display = "ERROR"
                score = 0
                error_records.append({
                    'QID': qid,
                    'Politeness_Level': politeness_level,
                    'Run': run + 1,
                    'Error': error or 'empty response',
                    'Attempts': attempts,
                    'Raw_Response': content,
                    'Question': prompt[:200] + '...'
                })
            else:
                predicted_display = predicted
                score = 100 if predicted == correct else 0

            results[(qid, politeness_level)].append(score)
            politeness_accuracy_scores[politeness_level].append(score)
            run_politeness_scores[politeness_level].append(score)
            run_accuracy_scores.append(score)
            overall_accuracy_scores.append(score)

            run_data.append({
                'QID': qid,
                'Politeness_Level': politeness_level,
                'Politeness_Prefix': sanitized_prefix,
                'Run': run + 1,
                'Score (%)': score,
                'Correct': correct,
                'Predicted': predicted_display,
                'Raw Response': content,
                'Question': prompt[:100] + '...'
            })

        run_df = pd.DataFrame(run_data)
        run_file = results_dir / f"abstract_algebra_run_{run + 1}_results.csv"
        run_df.to_csv(run_file, index=False)

        run_accuracy = mean(run_accuracy_scores) if run_accuracy_scores else 0
        print(f"Run {run + 1} Overall Accuracy: {round(run_accuracy, 2)}%")
        print(f"\n🎯 Accuracy by Politeness Level (Run {run + 1}):")
        for politeness_level, scores in run_politeness_scores.items():
            accuracy = mean(scores) if scores else 0
            print(f"  {politeness_level}: {round(accuracy, 2)}%")

        await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

    # === SUMMARY OUTPUTS ===
    summary_rows = []
    for (qid, politeness_level), scores in results.items():
        summary_rows.append({
            'QID': qid,
            'Politeness_Level': politeness_level,
            'Average Accuracy (%)': round(mean(scores), 2) if scores else 0,
            'Runs Counted': len(scores),
        })
    summary_df = pd.DataFrame(summary_rows)
    (results_dir / "abstract_algebra_per_question_politeness_accuracy.csv").write_text("") if summary_df.empty else summary_df.to_csv(results_dir / "abstract_algebra_per_question_politeness_accuracy.csv", index=False)

    overall_accuracy = round(mean(overall_accuracy_scores), 2) if overall_accuracy_scores else 0
    pd.DataFrame([{"Overall Accuracy (%)": overall_accuracy}]).to_csv(results_dir / "abstract_algebra_overall_accuracy.csv", index=False)

    politeness_summary = {}
    for politeness_level, scores in politeness_accuracy_scores.items():
        politeness_summary[politeness_level] = round(mean(scores), 2) if scores else 0
    pd.DataFrame(list(politeness_summary.items()), columns=['Politeness_Level', 'Accuracy (%)']).to_csv(results_dir / "abstract_algebra_politeness_accuracy.csv", index=False)

    run_matrix = defaultdict(lambda: [None] * NUM_RUNS)
    for (qid, politeness_level), scores in results.items():
        for i, s in enumerate(scores):
            run_matrix[(qid, politeness_level)][i] = s
    all_rows = []
    for (qid, politeness_level), values in run_matrix.items():
        row = {"QID": qid, "Politeness_Level": politeness_level}
        for i, val in enumerate(values):
            row[f"Run {i+1}"] = val
        all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(results_dir / "abstract_algebra_all_runs_by_qid_politeness.csv", index=False)

    # Save API errors
    if error_records:
        pd.DataFrame(error_records).to_csv(results_dir / "abstract_algebra_api_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["QID", "Politeness_Level", "Run", "Error", "Attempts", "Raw_Response", "Question"]).to_csv(results_dir / "abstract_algebra_api_errors.csv", index=False)

    # Unique formatted prompts sent
    if asked_formatted_questions:
        pd.DataFrame(asked_formatted_questions).to_csv(results_dir / "abstract_algebra_unique_formatted_questions.csv", index=False)
    else:
        pd.DataFrame(columns=["QID", "Politeness_Level", "Politeness_Prefix", "Formatted_Prompt"]).to_csv(results_dir / "abstract_algebra_unique_formatted_questions.csv", index=False)

    print(f"\n📁 All results saved to: {results_dir.absolute()}")

# === START ===
if __name__ == "__main__":
    asyncio.run(run_all())


