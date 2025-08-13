# phase1_data_preprocessing.py
import os
import pandas as pd
import numpy as np

# --- Configuration ---
# Directory where the original CIC-IDS-2017 CSV files are stored.
# Based on your screenshot, they are in a subfolder.
INPUT_CSV_DIRECTORY = 'CIC-IDS-2017/GeneratedLabelledFlows/' 

# The final, cleaned output file.
OUTPUT_CLEANED_CSV = 'final_labeled_flows_cicids.csv'


def load_and_combine_csvs(directory_path):
    """
    Loads all CSV files from the specified directory, combines them,
    and returns a single pandas DataFrame.
    """
    all_files = []
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"Reading file: {filename}...")
            # Read the CSV and append it to our list
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
            all_files.append(df)
    
    # Concatenate all DataFrames into one
    if not all_files:
        print("Error: No CSV files found in the directory. Please check the INPUT_CSV_DIRECTORY path.")
        return None
        
    print("Combining all DataFrames...")
    combined_df = pd.concat(all_files, ignore_index=True)
    return combined_df

def clean_data(df):
    """
    Cleans the combined DataFrame by fixing column names, removing duplicates,
    and handling invalid numerical values.
    """
    print("Cleaning the combined data...")
    
    # 1. Standardize Column Names
    # Removes leading/trailing spaces and replaces spaces with underscores
    original_columns = df.columns
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    print(" - Standardized column names.")
    
    # 2. Drop Duplicate Rows
    # Removes any rows that are exact copies of each other
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f" - Removed {initial_rows - len(df)} duplicate rows.")

    # 3. Handle Missing and Infinite Values
    # Replaces 'Infinity' and '-Infinity' with NaN (Not a Number)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drops rows that have any NaN values
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f" - Removed {initial_rows - len(df)} rows with missing or infinite values.")

    # 4. Final Data Check
    print(f"\nCleaning complete. Final dataset has {len(df)} rows and {len(df.columns)} columns.")
    
    return df


if __name__ == "__main__":
    # Step 1: Load and combine all the raw CSV data
    raw_df = load_and_combine_csvs(INPUT_CSV_DIRECTORY)
    
    if raw_df is not None:
        # Step 2: Clean the combined DataFrame
        cleaned_df = clean_data(raw_df)
        
        # Step 3: Save the final, clean DataFrame to a new CSV file
        print(f"\nSaving cleaned data to '{OUTPUT_CLEANED_CSV}'...")
        cleaned_df.to_csv(OUTPUT_CLEANED_CSV, index=False)
        print("Done!")