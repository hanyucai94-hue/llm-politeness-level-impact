#!/usr/bin/env python3
"""
Script to add 'Correct_Answer_Count' column to all overall_accuracy_by_tone_*.csv files.
This column shows the actual number of correct answers for each tone.
"""

import pandas as pd
import os
import glob

def main():
    # Find all overall_accuracy_by_tone files
    files = glob.glob('results-*/**/overall_accuracy_by_tone_*.csv', recursive=True)
    
    print(f'Found {len(files)} files to process\n')
    
    processed = 0
    skipped = 0
    
    for file_path in files:
        print(f'Processing: {file_path}')
        
        # Read the overall accuracy file
        df = pd.read_csv(file_path)
        
        # Check if column already exists
        if 'Correct_Answer_Count' in df.columns:
            print(f'  ⚠️  Column already exists, skipping...')
            skipped += 1
            print()
            continue
        
        # Extract directory and domain suffix from filename
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        domain_suffix = filename.replace('overall_accuracy_by_tone_', '').replace('.csv', '')
        
        # Find corresponding run_summary_results file
        run_summary_file = os.path.join(directory, f'run_summary_results_{domain_suffix}.csv')
        
        if os.path.exists(run_summary_file):
            # Read run summary to count total attempts per tone
            run_df = pd.read_csv(run_summary_file)
            
            # Count total attempts per tone
            total_attempts = run_df.groupby('Tone').size()
            
            # Calculate correct answer count
            # Formula: (Overall Accuracy / 100) * Total Attempts
            df['Correct_Answer_Count'] = df.apply(
                lambda row: int(round(row['Overall Accuracy (%)'] * total_attempts.get(row['Tone'], 0) / 100)),
                axis=1
            )
            
            # Save back to the same file
            df.to_csv(file_path, index=False)
            print(f'  ✅ Added Correct_Answer_Count column')
            print(f'    Total attempts per tone: {dict(total_attempts)}')
            processed += 1
        else:
            print(f'  ❌ Warning: Could not find {run_summary_file}')
            skipped += 1
        
        print()
    
    print(f'\n📊 Summary:')
    print(f'  Processed: {processed}')
    print(f'  Skipped: {skipped}')
    print(f'  Total: {len(files)}')
    print('\nDone! ✨')

if __name__ == '__main__':
    main()

