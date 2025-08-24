# list_columns.py
import pandas as pd

# --- Configuration ---
INPUT_CSV_PATH = 'final_labeled_flows_cicids.csv'
OUTPUT_TXT_PATH = 'column_names.txt'

def save_column_names():
    """
    Reads a CSV file, gets its column names, and saves them to a text file.
    """
    try:
        print(f"Reading column headers from '{INPUT_CSV_PATH}'...")
        # We only need to read the first row to get the headers, which is very fast.
        df = pd.read_csv(INPUT_CSV_PATH, nrows=0)
        column_names = df.columns.tolist()
        
        print(f"Saving {len(column_names)} column names to '{OUTPUT_TXT_PATH}'...")
        with open(OUTPUT_TXT_PATH, 'w') as f:
            f.write("Column Names from final_labeled_flows_cicids.csv\n")
            f.write("="*50 + "\n")
            for name in column_names:
                f.write(f"{name}\n")
        
        print(" - Done! File created successfully.")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_PATH}' was not found.")
        print("Please make sure you have run 'phase1_data_preprocessing.py' first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    save_column_names()