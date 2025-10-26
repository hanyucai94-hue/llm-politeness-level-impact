#!/usr/bin/env python3
"""
Simple script to read all CSV files from the 'test' folder and return a list of DataFrames
with an added 'task_category' column containing the CSV filename.
"""

import pandas as pd
import os
from pathlib import Path

def read_csv_files_to_dataframes(test_folder_path):
    """
    Read all CSV files from the specified folder and return a list of DataFrames.
    Each DataFrame will have an additional 'task_category' column with the filename.
    
    Args:
        test_folder_path (str): Path to the folder containing CSV files
        
    Returns:
        list: List of pandas DataFrames, one for each CSV file
    """
    dataframes = []
    test_path = Path(test_folder_path)
    
    if not test_path.exists():
        return dataframes
    
    # Get all CSV files in the directory
    csv_files = list(test_path.glob("*.csv"))
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Add the task_category column with the filename (without extension)
            task_category = csv_file.stem  # Gets filename without extension
            df['task_category'] = task_category
            
            # Add the DataFrame to our list
            dataframes.append(df)
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue
    
    return dataframes

# Example usage
if __name__ == "__main__":
    # Path to the test folder
    test_folder = "/Users/hanyucai/Downloads/niw/llm-1/test"
    
    # Read all CSV files and create DataFrames
    dataframes = read_csv_files_to_dataframes(test_folder)
    
    print(f"Created {len(dataframes)} DataFrames")
    print(f"Total rows across all DataFrames: {sum(len(df) for df in dataframes)}")
    
    # Example: Show info about each DataFrame
    for i, df in enumerate(dataframes[:5]):  # Show first 5
        print(f"DataFrame {i+1}: {df['task_category'].iloc[0]} - {len(df)} rows")
