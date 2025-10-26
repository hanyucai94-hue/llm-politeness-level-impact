#!/usr/bin/env python3
"""
Script to read all CSV files from the 'test' folder and create separate DataFrames
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
    
    # Get the absolute path
    test_path = Path(test_folder_path)
    
    if not test_path.exists():
        print(f"Error: The folder '{test_folder_path}' does not exist.")
        return dataframes
    
    # Get all CSV files in the directory
    csv_files = list(test_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{test_folder_path}'.")
        return dataframes
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Add the task_category column with the filename (without extension)
            task_category = csv_file.stem  # Gets filename without extension
            df['task_category'] = task_category
            
            # Add the DataFrame to our list
            dataframes.append(df)
            
            print(f"[OK] Successfully processed: {csv_file.name} ({len(df)} rows)")
            
        except Exception as e:
            print(f"[ERROR] Error processing {csv_file.name}: {str(e)}")
            continue
    
    print(f"\nSuccessfully processed {len(dataframes)} out of {len(csv_files)} CSV files.")
    return dataframes

def main():
    """Main function to demonstrate usage."""
    # Path to the test folder
    test_folder = "/Users/hanyucai/Downloads/niw/llm-1/test"
    
    # Read all CSV files and create DataFrames
    dataframes = read_csv_files_to_dataframes(test_folder)
    
    if dataframes:
        print(f"\nCreated {len(dataframes)} DataFrames:")
        for i, df in enumerate(dataframes):
            print(f"  {i+1}. {df['task_category'].iloc[0]} - {len(df)} rows, {len(df.columns)} columns")
        
        # Example: Access the first DataFrame
        if dataframes:
            print(f"\nExample - First DataFrame info:")
            print(f"  Task Category: {dataframes[0]['task_category'].iloc[0]}")
            print(f"  Shape: {dataframes[0].shape}")
            print(f"  Columns: {list(dataframes[0].columns)}")
            print(f"  First few rows:")
            print(dataframes[0].head())
    
    return dataframes

if __name__ == "__main__":
    # Run the main function
    dataframes = main()
