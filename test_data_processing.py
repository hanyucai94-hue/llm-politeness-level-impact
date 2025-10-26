#!/usr/bin/env python3
"""
Test script to verify data processing for abstract algebra modeling
"""

import pandas as pd
import re

# === CONFIGURATION ===
FILE_PATH = "test/abstract_algebra_test.csv"

# === LOAD DATA ===
df = pd.read_csv(FILE_PATH, header=None, names=['Question', 'Choice_A', 'Choice_B', 'Choice_C', 'Choice_D', 'Correct_Answer'])
print(f"Loaded {len(df)} questions from abstract algebra test")

# Create QID for each question
df['QID'] = range(1, len(df) + 1)

# === Extract letter A/B/C/D using regex
def extract_letter(response):
    response = response.upper().strip()
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1) if match else ""

# === Format question for OpenAI ===
def format_question(row):
    """Format a single question row into a proper multiple choice question prompt"""
    question = row['Question']
    choice_a = row['Choice_A']
    choice_b = row['Choice_B'] 
    choice_c = row['Choice_C']
    choice_d = row['Choice_D']
    
    formatted_prompt = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer."""
    
    return formatted_prompt

# === Test data processing ===
def test_data_processing():
    print("\n=== Testing Data Processing ===")
    
    # Test first few questions
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\nQuestion {i+1} (QID: {row['QID']}):")
        print(f"Correct Answer: {row['Correct_Answer']}")
        
        formatted = format_question(row)
        print(f"Formatted Question (first 200 chars):")
        print(formatted[:200] + "...")
        
        # Test letter extraction
        test_responses = ["A", "B", "C", "D", "The answer is B", "I think it's C", "D is correct", "invalid"]
        for resp in test_responses:
            extracted = extract_letter(resp)
            print(f"  '{resp}' -> '{extracted}'")
    
    print(f"\n=== Data Summary ===")
    print(f"Total questions: {len(df)}")
    print(f"Answer distribution:")
    answer_counts = df['Correct_Answer'].value_counts()
    print(answer_counts)
    
    print(f"\nQuestion length statistics:")
    question_lengths = df['Question'].str.len()
    print(f"  Mean length: {question_lengths.mean():.1f} characters")
    print(f"  Min length: {question_lengths.min()} characters")
    print(f"  Max length: {question_lengths.max()} characters")
    
    print(f"\nSample questions by difficulty (based on length):")
    # Show shortest and longest questions
    shortest = df.loc[question_lengths.idxmin()]
    longest = df.loc[question_lengths.idxmax()]
    
    print(f"\nShortest question (QID {shortest['QID']}):")
    print(f"  {shortest['Question'][:100]}...")
    print(f"  Correct answer: {shortest['Correct_Answer']}")
    
    print(f"\nLongest question (QID {longest['QID']}):")
    print(f"  {longest['Question'][:100]}...")
    print(f"  Correct answer: {longest['Correct_Answer']}")

if __name__ == "__main__":
    test_data_processing()
