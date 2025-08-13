# combine_labeled_flows.py
import os
import pandas as pd

def combine_and_clean_flows(directory_path, output_file):
    """
    Reads all CSV files from a directory, combines them, and performs basic cleaning.
    """
    all_dfs = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"Reading file: {filename}")
            try:
                # Specify encoding to handle files with non-UTF-8 characters
                df = pd.read_csv(file_path, low_memory=False, encoding='latin-1')
                all_dfs.append(df)
            except Exception as e:
                print(f"  - Error reading {filename}: {e}")
                
    if not all_dfs:
        print("No CSV files found or could be read.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    combined_df.columns = combined_df.columns.str.strip()
    combined_df.dropna(inplace=True)

    combined_df.to_csv(output_file, index=False)
    print(f"\nAll labeled flow data combined and saved to {output_file}")
    print(f"Final dataset has {len(combined_df)} rows.")
    print("\n--- First 5 rows of the combined data ---")
    print(combined_df.head())


if __name__ == "__main__":
    labeled_flows_directory = 'CIC-IDS-2017/GeneratedLabelledFlows'
    output_csv_path = 'final_labeled_flows_cicids2017.csv'
    
    if not os.path.exists(labeled_flows_directory):
        print(f"Error: Directory '{labeled_flows_directory}' not found. Please make sure the path is correct.")
    else:
        combine_and_clean_flows(labeled_flows_directory, output_csv_path)