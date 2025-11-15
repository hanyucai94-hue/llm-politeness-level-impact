# Please install google-generativeai and pandas - !pip install google-generativeai pandas
# Please insert your API key to make the code run
import os
import asyncio
import pandas as pd
from collections import defaultdict
from statistics import mean
import google.generativeai as genai
import re
 
# === CONFIGURATION ===

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDo1NCMLFUEKKc7rytbCj0dpr6seBKpZmk"  # Get from https://makersuite.google.com/app/apikey
genai.configure(api_key=GEMINI_API_KEY)

# Available models:
# - "models/gemini-1.5-flash" (fastest, cheapest)
# - "models/gemini-1.5-pro" (most capable)
# - "models/gemini-1.0-pro" (legacy)
MODEL = "models/gemini-2.0-flash"
MODEL_NAME = "gemini-2.0-flash"  # For directory naming
NUM_RUNS = 3
FILE_PATH = "sanitized_test_data_wiz_politeness_level/dataset+test_random_500_sanitized_stem.csv"
# Rate limit for Gemini 2.0 Flash (Paid Tier):
# - RPM: 2,000 (Requests Per Minute)
# - TPM: 4,000,000 (Tokens Per Minute)
# - RPD: Unlimited
# 
# With 2,000 RPM, we can process ~33 requests per second
# Each QID sends 5 requests in parallel
# Conservative: 400 QIDs per minute = 2,000 requests per minute
# Sleep time: 60 seconds / 400 QIDs = 0.15 seconds per QID
# 
# Using 0.2 seconds to be safe and avoid bursting
SLEEP_BETWEEN_BATCHES = 0.2  # seconds (for 2,000 RPM paid tier)
results_dir = f"results-{MODEL_NAME}"
os.makedirs(results_dir, exist_ok=True)
 
# === LOAD DATA ===
df = pd.read_csv(FILE_PATH)
grouped_prompts = df.groupby("QID")

# Extract domain from filename (everything after 'sanitized_' or between 'base_question_' and '_test')
# e.g., "dataset+test_random_500_sanitized_humanities.csv" -> "humanities"
# e.g., "dataset+test_random_500_sanitized_professional_law.csv" -> "professional_law"
# e.g., "dataset+test_base_question_moral_disputes_test.csv" -> "moral_disputes"
filename = os.path.basename(FILE_PATH)
filename_without_ext = filename.rsplit('.', 1)[0]
if 'sanitized_' in filename_without_ext:
    domain_suffix = filename_without_ext.split('sanitized_', 1)[1]  # Get everything after 'sanitized_'
elif 'base_question_' in filename_without_ext and '_test' in filename_without_ext:
    # Extract domain between 'base_question_' and '_test'
    temp = filename_without_ext.split('base_question_', 1)[1]
    domain_suffix = temp.rsplit('_test', 1)[0]
else:
    domain_suffix = filename_without_ext.split('_')[-1]  # Fallback: get last word
domain_value = domain_suffix
 
# === STORAGE ===
results = defaultdict(list)
overall_tone_scores = defaultdict(list)
 
# === Extract letter A/B/C/D using regex
def extract_letter(response):
    response = response.upper().strip()
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1) if match else ""
 
# === Send prompt to Gemini
async def call_gemini(prompt):
    try:
        # Create model instance
        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config={
                "temperature": 0,  # Deterministic output
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 100,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        )
        
        # Build the full prompt
        full_prompt = (
            "You are an AI tutor answering multiple choice questions. "
            "Always reply with ONLY the letter of the correct answer (A, B, C, or D). "
            "Do not explain your answer.\n\n"
            "Please answer this multiple choice question. Respond with only the letter of the correct answer (A, B, C, or D). Do not explain.\n\n"
            + prompt
        )
        
        # Generate response (async)
        response = await asyncio.to_thread(
            model.generate_content,
            full_prompt
        )
        
        # Check if response was blocked by safety filters
        try:
            text = response.text.strip()
            return text
        except ValueError as e:
            # This happens when finish_reason is SAFETY (blocked by filters)
            print(f"\n⚠️  API Error: Response blocked - {e}")
            
            if response.candidates:
                candidate = response.candidates[0]
                
                # Print finish reason
                finish_reason_map = {
                    0: "FINISH_REASON_UNSPECIFIED",
                    1: "STOP (normal completion)",
                    2: "SAFETY (blocked by safety filters)",
                    3: "RECITATION (blocked due to recitation)",
                    4: "OTHER"
                }
                finish_reason = candidate.finish_reason
                print(f"   Finish Reason: {finish_reason_map.get(finish_reason, finish_reason)}")
                
                # Examine safety ratings to see which categories were flagged
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    print(f"   Safety Ratings:")
                    for rating in candidate.safety_ratings:
                        # rating has: category, probability, blocked
                        category_name = str(rating.category).replace('HarmCategory.', '')
                        probability = str(rating.probability).replace('HarmProbability.', '')
                        blocked = getattr(rating, 'blocked', False)
                        
                        # Print ALL ratings when blocked to see what's happening
                        print(f"      - {category_name}: {probability} (blocked: {blocked})")
                else:
                    print(f"   No safety_ratings available")
            else:
                print(f"   No candidates in response")
            
            return ""
            
    except Exception as e:
        print(f"API Error: {e}")
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
                tasks.append(call_gemini(prompt))
 
            responses = await asyncio.gather(*tasks)
 
            for (tone, original_prompt, correct), response in zip(tone_prompt_pairs, responses):
                predicted = extract_letter(response)
                print(qid, tone, predicted, correct)
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
 
        # Accumulate run results
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

