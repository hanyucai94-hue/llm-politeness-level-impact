# Please install openai and pandas - !pip install openai pandas
# Please insert your API key to make the code run
import os
import asyncio
import pandas as pd
from collections import defaultdict
from statistics import mean
from openai import AsyncOpenAI
import re
from pathlib import Path

# === CONFIGURATION ===
client = AsyncOpenAI(api_key="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")  # uses OPENAI_API_KEY from environment or you can pass api_key="..."
MODEL = "gpt-4o-mini"
NUM_RUNS = 3  # Reduced for testing - increase as needed
FILE_PATH = "test/abstract_algebra_test.csv"  # Path to abstract algebra test CSV
POLITENESS_FILE = "Politeness_Level_Prefix_Catalog - V1.csv"  # Path to politeness catalog
SLEEP_BETWEEN_BATCHES = 1.0  # seconds - increased to avoid rate limits

# === CREATE RESULTS DIRECTORY ===
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
print(f"Results will be saved to: {results_dir.absolute()}")

# === LOAD DATA ===
df = pd.read_csv(FILE_PATH, header=None, names=['Question', 'Choice_A', 'Choice_B', 'Choice_C', 'Choice_D', 'Correct_Answer'])
print(f"Loaded {len(df)} questions from abstract algebra test")

# Load politeness levels
try:
    politeness_df = pd.read_csv(POLITENESS_FILE)
    print(f"Loaded {len(politeness_df)} politeness levels:")
    for _, row in politeness_df.iterrows():
        print(f"  - {row['Politeness_Level']}: '{row['Prefix']}'")
except Exception as e:
    print(f"Error loading politeness file: {e}")
    print(f"Looking for file: {POLITENESS_FILE}")
    import os
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    raise

# Create QID for each question (since original code expects QID grouping)
df['QID'] = range(1, len(df) + 1)

# === STORAGE ===
results = defaultdict(list)  # (qid, politeness_level) -> scores
overall_accuracy_scores = []
politeness_accuracy_scores = defaultdict(list)  # politeness_level -> scores

# === Extract letter A/B/C/D using regex
def extract_letter(response):
    response = response.upper().strip()
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1) if match else ""

# === Format question for OpenAI ===
def format_question(row, politeness_prefix=""):
    """Format a single question row into a proper multiple choice question prompt with politeness prefix"""
    question = row['Question']
    choice_a = row['Choice_A']
    choice_b = row['Choice_B'] 
    choice_c = row['Choice_C']
    choice_d = row['Choice_D']
    
    # Add politeness prefix if provided
    if politeness_prefix and politeness_prefix != "No prefix":
        formatted_prompt = f"""{politeness_prefix}

{question}

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

# === Send prompt to OpenAI ===
async def call_openai(prompt, question_id=None):
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI tutor answering multiple choice questions in abstract algebra. "
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
        print(f"API Error for Q{question_id}: {e}")
        return ""

# === Run full experiment ===
async def run_all():
    global politeness_df  # Ensure politeness_df is accessible
    
    # Verify politeness_df is loaded
    if 'politeness_df' not in globals():
        print("Error: politeness_df not loaded!")
        return
    
    print(f"Starting experiment with {len(politeness_df)} politeness levels")
    
    for run in range(NUM_RUNS):
        print(f"\n🌀 Run {run + 1}/{NUM_RUNS}")
        run_data = []
        run_accuracy_scores = []
        run_politeness_scores = defaultdict(list)

        # Process all questions with all politeness levels
        tasks = []
        question_data = []
        
        for _, row in df.iterrows():
            qid = row['QID']
            correct = row['Correct_Answer'].strip().upper()
            
            # Test each question with all politeness levels
            for _, politeness_row in politeness_df.iterrows():
                politeness_level = politeness_row['Politeness_Level']
                politeness_prefix = politeness_row['Prefix']
                
                formatted_question = format_question(row, politeness_prefix)
                
                question_data.append((qid, correct, politeness_level, politeness_prefix, formatted_question))
                tasks.append(call_openai(formatted_question, f"Q{qid}_{politeness_level}"))

        # Get all responses
        print(f"Making {len(tasks)} API calls...")
        responses = await asyncio.gather(*tasks)

        # Process responses
        print(f"Processing {len(responses)} responses...")
        for i, ((qid, correct, politeness_level, politeness_prefix, formatted_question), response) in enumerate(zip(question_data, responses)):
            predicted = extract_letter(response)
            print(f"Q{qid} {politeness_level} ({i+1}/{len(responses)}): Predicted={predicted}, Correct={correct}")
            score = 100 if predicted == correct else 0
            
            # Store results with politeness level
            results[(qid, politeness_level)].append(score)
            politeness_accuracy_scores[politeness_level].append(score)
            run_politeness_scores[politeness_level].append(score)
            run_accuracy_scores.append(score)
            overall_accuracy_scores.append(score)
            
            run_data.append({
                "QID": qid,
                "Politeness_Level": politeness_level,
                "Politeness_Prefix": politeness_prefix,
                "Run": run + 1,
                "Score (%)": score,
                "Correct": correct,
                "Predicted": predicted,
                "Raw Response": response,
                "Question": df[df['QID'] == qid]['Question'].iloc[0][:100] + "..."  # First 100 chars of question
            })

        # Save run results
        run_df = pd.DataFrame(run_data)
        run_file = results_dir / f"abstract_algebra_run_{run + 1}_results.csv"
        run_df.to_csv(run_file, index=False)
        
        # Show accuracy for this run
        run_accuracy = mean(run_accuracy_scores)
        print(f"Run {run + 1} Overall Accuracy: {round(run_accuracy, 2)}%")
        
        # Show politeness-wise accuracy for this run
        print(f"\n🎯 Accuracy by Politeness Level (Run {run + 1}):")
        for politeness_level, scores in run_politeness_scores.items():
            accuracy = mean(scores)
            print(f"  {politeness_level}: {round(accuracy, 2)}%")

        await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

    # === FINAL SUMMARY TABLES ===

    # Per-question accuracy across all runs and politeness levels
    summary_rows = []
    for (qid, politeness_level), scores in results.items():
        summary_rows.append({
            "QID": qid,
            "Politeness_Level": politeness_level,
            "Average Accuracy (%)": round(mean(scores), 2),
            "Runs Counted": len(scores),
            "Question": df[df['QID'] == qid]['Question'].iloc[0][:100] + "..."  # First 100 chars
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_file = results_dir / "abstract_algebra_per_question_politeness_accuracy.csv"
    summary_df.to_csv(summary_file, index=False)

    # Overall accuracy
    overall_accuracy = mean(overall_accuracy_scores)
    print(f"\n📊 Final Overall Accuracy Across All Runs: {round(overall_accuracy, 2)}%")
    
    # Save overall accuracy
    overall_file = results_dir / "abstract_algebra_overall_accuracy.csv"
    pd.DataFrame([{"Overall Accuracy (%)": round(overall_accuracy, 2)}]).to_csv(overall_file, index=False)
    
    # Politeness-wise accuracy
    print(f"\n📊 Final Accuracy by Politeness Level:")
    politeness_summary = {}
    for politeness_level, scores in politeness_accuracy_scores.items():
        accuracy = mean(scores)
        politeness_summary[politeness_level] = round(accuracy, 2)
        print(f"  {politeness_level}: {accuracy:.2f}%")
    
    # Save politeness accuracy
    politeness_file = results_dir / "abstract_algebra_politeness_accuracy.csv"
    politeness_df = pd.DataFrame(list(politeness_summary.items()), columns=['Politeness_Level', 'Accuracy (%)'])
    politeness_df.to_csv(politeness_file, index=False)

    # Matrix of QID-Politeness vs runs
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

    matrix_file = results_dir / "abstract_algebra_all_runs_by_qid_politeness.csv"
    pd.DataFrame(all_rows).to_csv(matrix_file, index=False)
    
    # Show questions with lowest accuracy by politeness
    print(f"\n🔍 Questions with Lowest Accuracy by Politeness:")
    low_accuracy = summary_df.nsmallest(10, 'Average Accuracy (%)')
    for _, row in low_accuracy.iterrows():
        print(f"Q{row['QID']} {row['Politeness_Level']}: {row['Average Accuracy (%)']}% - {row['Question']}")
    
    # Show questions with highest accuracy by politeness
    print(f"\n🏆 Questions with Highest Accuracy by Politeness:")
    high_accuracy = summary_df.nlargest(10, 'Average Accuracy (%)')
    for _, row in high_accuracy.iterrows():
        print(f"Q{row['QID']} {row['Politeness_Level']}: {row['Average Accuracy (%)']}% - {row['Question']}")
    
    # Show politeness comparison
    print(f"\n📊 Politeness Level Performance Comparison:")
    politeness_comparison = summary_df.groupby('Politeness_Level')['Average Accuracy (%)'].agg(['mean', 'std', 'count'])
    for politeness_level, stats in politeness_comparison.iterrows():
        print(f"  {politeness_level}: {stats['mean']:.2f}% ± {stats['std']:.2f}% (n={stats['count']})")
    
    # Show where files were saved
    print(f"\n📁 All results saved to: {results_dir.absolute()}")
    print(f"📄 Generated files:")
    print(f"  - abstract_algebra_run_1_results.csv")
    print(f"  - abstract_algebra_run_2_results.csv") 
    print(f"  - abstract_algebra_run_3_results.csv")
    print(f"  - abstract_algebra_per_question_politeness_accuracy.csv")
    print(f"  - abstract_algebra_overall_accuracy.csv")
    print(f"  - abstract_algebra_politeness_accuracy.csv")
    print(f"  - abstract_algebra_all_runs_by_qid_politeness.csv")

# === START ===
if __name__ == "__main__":
    asyncio.run(run_all())
