# phase2_feature_exploration.py
import pandas as pd

# --- Configuration ---
CLEANED_DATA_PATH = 'final_labeled_flows_cicids.csv'

def explore_data(file_path):
    """
    Loads the cleaned dataset and performs an initial exploration to understand
    its structure and content.
    """
    print(f"Loading cleaned data from '{file_path}'...")
    try:
        # Set low_memory=False to handle the DtypeWarning from the previous step
        df = pd.read_csv(file_path, low_memory=False)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the Phase 1 script ran successfully.")
        return

    # --- Data Exploration ---
    print("\n--- 1. Basic DataFrame Info ---")
    # Displays total rows, columns, and memory usage
    df.info()

    print("\n\n--- 2. Column Names ---")
    # Prints all the column names
    print("The dataset contains the following columns:")
    for col in df.columns:
        print(f" - {col}")
        
    print("\n\n--- 3. Attack Label Distribution ---")
    # Shows how many flows belong to each attack category (and benign)
    print("Distribution of traffic labels:")
    print(df['Label'].value_counts())


if __name__ == "__main__":
    explore_data(CLEANED_DATA_PATH)