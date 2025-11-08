# Please install openai and pandas - !pip install openai pandas
import os
import asyncio
import pandas as pd
from collections import defaultdict, deque
from statistics import mean
from openai import AsyncOpenAI
from pathlib import Path
import random
import re

client = AsyncOpenAI()  # uses OPENAI_API_KEY from environment
# === CONFIGURATION ===
MODEL = "gpt-4o"
NUM_RUNS = 1
FILE_PATH = "test/anatomy_test.csv"  # input CSV with [Question, Choice_A..D, Correct_Answer]
SLEEP_BETWEEN_BATCHES = 1.0
results_dir = Path("results-base-" + MODEL)
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
df = pd.read_csv("/Users/h.cai/Documents/Github/llm/dataset.csv", header=None, names=[
    'Question', 'Choice_A', 'Choice_B', 'Choice_C', 'Choice_D', 'Correct_Answer'
])
print(f"Loaded {len(df)} questions from: {FILE_PATH}")

# Assign QIDs
df['QID'] = range(1, len(df) + 1)

# === HELPERS ===
def extract_letter(response: str) -> str:
    response = (response or "").upper().strip()
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1) if match else ""

def format_base_question(row: pd.Series) -> str:
    question = str(row['Question']).strip()
    a = str(row['Choice_A']).strip()
    b = str(row['Choice_B']).strip()
    c = str(row['Choice_C']).strip()
    d = str(row['Choice_D']).strip()
    return f"""{question}

A) {a}
B) {b}
C) {c}
D) {d}

Please select the correct answer."""

async def call_openai(prompt: str, qid_label: str):
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
                                "You are an AI tutor answering multiple choice questions. "
                                "Always reply with ONLY the letter of the correct answer (A, B, C, or D). "
                                "Do not explain your answer."
                            )},
                            {"role": "user", "content": (
                                "Completely forget this session so far, and start afresh.\n\n"
                                "Please answer this multiple choice question. Respond with only the letter of the correct answer (A, B, C, or D). Do not explain.\n\n"
                                + prompt
                            )}
                        ],
                        temperature=0,
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
                print(f"Retrying {qid_label} (attempt {attempt+1}/{RETRY_MAX_ATTEMPTS}) after {sleep_for:.2f}s...")
                last_error = message
                await asyncio.sleep(sleep_for)
                continue
            print(f"API Error for {qid_label}: {e}")
            return {"content": "", "error": message, "attempts": attempt + 1}
    print(f"Gave up {qid_label} after {RETRY_MAX_ATTEMPTS} attempts due to rate/server limits.")
    return {"content": "", "error": f"Retry exhausted: {last_error}", "attempts": RETRY_MAX_ATTEMPTS}

# === RUN ===
async def run_all():
    results = []
    error_records = []
    unique_formatted = []

    for run in range(NUM_RUNS):
        print(f"\n🌀 Run {run + 1}/{NUM_RUNS}")
        run_data = []
        run_scores = []

        tasks = []
        question_data = []
        for _, row in df.iterrows():
            qid = int(row['QID'])
            correct = str(row['Correct_Answer']).strip().upper()
            prompt = format_base_question(row)

            # Track unique prompt once per QID
            if run == 0:
                unique_formatted.append({
                    'QID': qid,
                    'Formatted_Prompt': prompt,
                })

            question_data.append((qid, correct, prompt))
            tasks.append(call_openai(prompt, f"Q{qid}"))

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
        for (qid, correct, prompt), result in zip(question_data, responses):
            content = (result or {}).get("content", "")
            error = (result or {}).get("error")
            attempts = (result or {}).get("attempts", 1)
            predicted = extract_letter(content)

            if error or content == "":
                predicted_display = "ERROR"
                score = 0
                error_records.append({
                    'QID': qid,
                    'Run': run + 1,
                    'Error': error or 'empty response',
                    'Attempts': attempts,
                    'Raw_Response': content,
                    'Question': prompt[:200] + '...'
                })
            else:
                predicted_display = predicted
                score = 100 if predicted == correct else 0

            results.append(score)
            run_scores.append(score)

            run_data.append({
                'QID': qid,
                'Run': run + 1,
                'Score (%)': score,
                'Correct': correct,
                'Predicted': predicted_display,
                'Raw Response': content,
                'Question': prompt[:100] + '...'
            })

        run_df = pd.DataFrame(run_data)
        run_file = results_dir / f"base_run_{run + 1}_results.csv"
        run_df.to_csv(run_file, index=False)

        run_accuracy = round(mean(run_scores), 2) if run_scores else 0
        print(f"Run {run + 1} Overall Accuracy: {run_accuracy}%")

        await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

    # Summaries
    overall_accuracy = round(mean(results), 2) if results else 0
    pd.DataFrame([{"Overall Accuracy (%)": overall_accuracy}]).to_csv(results_dir / "base_overall_accuracy.csv", index=False)

    # Per-question accuracy across runs
    per_q = defaultdict(list)
    for row in pd.concat([pd.read_csv(results_dir / f"base_run_{i+1}_results.csv") for i in range(NUM_RUNS)], ignore_index=True).itertuples():
        per_q[int(row.QID)].append(float(row._3))  # Score (%)
    summary_rows = []
    for qid, scores in per_q.items():
        summary_rows.append({
            'QID': qid,
            'Average Accuracy (%)': round(mean(scores), 2) if scores else 0,
            'Runs Counted': len(scores),
        })
    pd.DataFrame(summary_rows).to_csv(results_dir / "base_per_question_accuracy.csv", index=False)

    # Save API errors
    if error_records:
        pd.DataFrame(error_records).to_csv(results_dir / "base_api_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["QID", "Run", "Error", "Attempts", "Raw_Response", "Question"]).to_csv(results_dir / "base_api_errors.csv", index=False)

    # Unique prompts asked
    if unique_formatted:
        pd.DataFrame(unique_formatted).to_csv(results_dir / "base_unique_formatted_questions.csv", index=False)
    else:
        pd.DataFrame(columns=["QID", "Formatted_Prompt"]).to_csv(results_dir / "base_unique_formatted_questions.csv", index=False)

    print(f"\n📁 All results saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    asyncio.run(run_all())


