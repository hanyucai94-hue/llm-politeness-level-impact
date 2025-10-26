#!/usr/bin/env python3
"""
Test script to verify politeness data loading works correctly
"""

import pandas as pd
import os

# Test politeness file loading
POLITENESS_FILE = "Politeness_Level_Prefix_Catalog - V1.csv"

print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

try:
    politeness_df = pd.read_csv(POLITENESS_FILE)
    print(f"✅ Successfully loaded {len(politeness_df)} politeness levels:")
    for _, row in politeness_df.iterrows():
        print(f"  - {row['Politeness_Level']}: '{row['Prefix']}'")
except Exception as e:
    print(f"❌ Error loading politeness file: {e}")
    print(f"Looking for file: {POLITENESS_FILE}")
    if os.path.exists(POLITENESS_FILE):
        print("File exists but couldn't be read")
    else:
        print("File does not exist")
