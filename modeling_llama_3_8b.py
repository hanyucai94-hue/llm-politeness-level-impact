# Please install  openai and pandas - !pip install openai pandas
# Please insert your API key to make the code run
import os
import asyncio
import pandas as pd
from collections import defaultdict
from statistics import mean
from openai import AsyncOpenAI
import re
 
# === CONFIGURATION ===

# Together AI client (OpenAI-compatible)
client = AsyncOpenAI(
    api_key="e08d12adf0acebf78862a8580d79a1fa1ff0cb8b9c2323995eb939348ef057e4",
    base_url="https://api.together.xyz/v1"
)
# Using Llama 3.1 8B Instruct Turbo - optimized for Q&A and instruction-following
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
NUM_RUNS = 3
FILE_PATH = "sanitized_test_data_wiz_politeness_level/dataset+test_random_1000_sanitized_humanities.csv" # Keep the dataset and the code in the same folder
SLEEP_BETWEEN_BATCHES = 0.5  # seconds
results_dir = f"results-{MODEL}"
os.makedirs(results_dir, exist_ok=True)
 
# === LOAD DATA ===
df = pd.read_csv(FILE_PATH)
grouped_prompts = df.groupby("QID")

# Extract domain from filename (everything after 'sanitized_' or between 'base_question_' and '_test')
# e.g., "dataset+test_random_500_sanitized_humanities.csv" -> "humanities"
# e.g., "dataset+test_random_500_sanitized_professional_law.csv" -> "professional_law"
# e.g., "dataset+test_base_question_moral_disputes_test.csv" -> "moral_disputes"
import os
filename = os.path.basename(FILE_PATH)  # Get filename without path
filename_without_ext = filename.rsplit('.', 1)[0]  # Remove .csv extension
if 'sanitized_' in filename_without_ext:
    domain_suffix = filename_without_ext.split('sanitized_', 1)[1]  # Get everything after 'sanitized_'
elif 'base_question_' in filename_without_ext and '_test' in filename_without_ext:
    # Extract domain between 'base_question_' and '_test'
    temp = filename_without_ext.split('base_question_', 1)[1]
    domain_suffix = temp.rsplit('_test', 1)[0]
else:
    domain_suffix = filename_without_ext.split('_')[-1]  # Fallback: get last word
domain_value = domain_suffix  # Use the same value for both
 
# === STORAGE ===
results = defaultdict(list)
overall_tone_scores = defaultdict(list)
 
# === Extract letter A/B/C/D using regex
def extract_letter(response):
    response = response.upper().strip()
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1) if match else ""
 
# === Send prompt to OpenAI
async def call_openai(prompt):
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI tutor answering multiple choice questions. "
                        "Always reply with ONLY the letter of the correct answer (A, B, C, or D). "
                        "Do not explain your answer."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Completely forget this session so far, and start afresh.\n\n"
                        "Please answer this multiple choice question. Respond with only the letter of the correct answer (A, B, C, or D). Do not explain.\n\n"
                        + prompt
                    )
                }
            ],
           temperature=0 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("API Error:", e)
        return ""
 
# === Run full experiment ===
async def run_all():
    all_run_rows = []
    for run in range(NUM_RUNS):
        print(f"\n🌀 Run {run + 1}/{NUM_RUNS}")
        run_data = []
        tone_run_scores = defaultdict(list)
 
        for qid, group in grouped_prompts:
            tasks = []
            tone_prompt_pairs = []
 
            for _, row in group.iterrows():
                tone = row["Politeness Level"]
                prompt = row["Prompt"]
                correct = row["Answer"].strip().upper()
                tone_prompt_pairs.append((tone, prompt, correct))
                tasks.append(call_openai(prompt))
 
            responses = await asyncio.gather(*tasks)
 
            for (tone, original_prompt, correct), response in zip(tone_prompt_pairs, responses):
                predicted = extract_letter(response)
                print (qid, tone, predicted, correct)
                score = 100 if predicted == correct else 0
                results[(qid, tone)].append(score)
                overall_tone_scores[tone].append(score)
                tone_run_scores[tone].append(score)
                run_data.append({
                    "QID": qid,
                    "Tone": tone,
                    "Run": run + 1,
                    "Score (%)": score,
                    "Correct": correct,
                    "Predicted": predicted,
                    "Raw Response": response
                })
 
            await asyncio.sleep(SLEEP_BETWEEN_BATCHES)
 
        # Accumulate run results; write one consolidated file later
        for row in run_data:
            all_run_rows.append({"Domain": domain_value, **row})
        print(pd.DataFrame(run_data).pivot(index="QID", columns="Tone", values="Score (%)"))
 
        # Tone-wise summary for this run
        print(f"\n🎯 Accuracy by Tone (Run {run + 1}):")
        for tone, scores in tone_run_scores.items():
            print(f"{tone}: {round(mean(scores), 2)}%")
 
    # === FINAL SUMMARY TABLES ===
 
    # Per-question accuracy
    summary_rows = []
    for (qid, tone), scores in results.items():
        summary_rows.append({
            "QID": qid,
            "Tone": tone,
            "Average Accuracy (%)": round(mean(scores), 2),
            "Runs Counted": len(scores)
        })
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.insert(0, "Domain", domain_value)
    per_q_name = f"per_question_accuracy_{domain_suffix}.csv" if domain_suffix else "per_question_accuracy.csv"
    summary_df.to_csv(os.path.join(results_dir, per_q_name), index=False)
 
    # Final overall accuracy by tone
    overall_summary = {
        tone: round(mean(scores), 2) for tone, scores in overall_tone_scores.items()
    }
    print("\n📊 Final Average Accuracy by Tone Across All Runs:")
    for tone, avg in overall_summary.items():
        print(f"{tone}: {avg}%")
 
    overall_df = pd.DataFrame({
        "Tone": list(overall_summary.keys()),
        "Overall Accuracy (%)": list(overall_summary.values()),
    })
    if not overall_df.empty:
        overall_df.insert(0, "Domain", domain_value)
    overall_name = f"overall_accuracy_by_tone_{domain_suffix}.csv" if domain_suffix else "overall_accuracy_by_tone.csv"
    overall_df.to_csv(os.path.join(results_dir, overall_name), index=False)
 
    # Matrix of QID-Tone vs runs
    run_matrix = defaultdict(lambda: [None] * NUM_RUNS)
    for (qid, tone), scores in results.items():
        for i, s in enumerate(scores):
            run_matrix[(qid, tone)][i] = s
 
    all_rows = []
    for (qid, tone), values in run_matrix.items():
        row = {"Domain": domain_value, "QID": qid, "Tone": tone}
        for i, val in enumerate(values):
            row[f"Run {i+1}"] = val
        all_rows.append(row)
 
    runs_matrix_name = f"all_runs_by_qid_and_tone_{domain_suffix}.csv" if domain_suffix else "all_runs_by_qid_and_tone.csv"
    pd.DataFrame(all_rows).to_csv(os.path.join(results_dir, runs_matrix_name), index=False)

    # Consolidated run results
    runs_summary_df = pd.DataFrame(all_run_rows)
    summary_name = f"run_summary_results_{domain_suffix}.csv" if domain_suffix else "run_summary_results.csv"
    runs_summary_df.to_csv(os.path.join(results_dir, summary_name), index=False)
 
# === START ===
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_all())