# phase2_feature_engineering.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
CLEANED_DATA_PATH = 'final_labeled_flows_cicids.csv'
ENGINEERED_DATA_PATH = 'model_ready_data.csv'

def engineer_features(df):
    """
    Engineers new features, cleans the data, and prepares it for modeling.
    """
    print("Starting feature engineering...")

    # --- Step 1: Create New Semantic Features ---
    # Your project plan emphasizes adding meaning to the data.
    # Let's map the protocol numbers to their names.
    # Protocol numbers: 6 = TCP, 17 = UDP.
    protocol_map = {6.0: 'TCP', 17.0: 'UDP', 0.0: 'Other'}
    df['Protocol_Name'] = df['Protocol'].map(protocol_map).fillna('Other')
    print(" - Created 'Protocol_Name' feature.")
    
    # Let's also infer the service based on the destination port.
    # This adds another layer of semantic meaning.
    def map_service(port):
        if port == 80: return 'HTTP'
        if port == 443: return 'HTTPS'
        if port == 21: return 'FTP'
        if port == 22: return 'SSH'
        if port == 53: return 'DNS'
        if port > 1024: return 'Ephemeral' # Ports used for temporary connections
        return 'Other'
        
    df['Service'] = df['Destination_Port'].apply(map_service)
    print(" - Created 'Service' feature.")

    # --- Step 2: Clean and Drop Columns ---
    # Drop columns that are not useful for general attack detection.
    # IP addresses and Flow IDs are too specific for a general model.
    # Timestamps can cause the model to learn time-specific patterns.
    # Fwd_Header_Length.1 is a duplicate column from the original dataset.
    columns_to_drop = [
        'Flow_ID', 'Source_IP', 'Destination_IP', 'Timestamp',
        'Fwd_Header_Length.1'
    ]
    df.drop(columns=columns_to_drop, inplace=True)
    print(f" - Dropped {len(columns_to_drop)} unnecessary columns.")

    # Clean the 'Label' column by removing spaces
    df['Label'] = df['Label'].str.replace(' ', '_')
    print(" - Cleaned up spaces in the 'Label' column.")

    return df

def balance_data_with_undersampling(df):
    """
    Balances the dataset by undersampling the majority class (BENIGN).
    """
    print("\nBalancing the dataset with undersampling...")
    
    # Separate the majority class (BENIGN) from the minority classes (attacks)
    df_minority = df[df['Label'] != 'BENIGN']
    df_majority = df[df['Label'] == 'BENIGN']
    
    # We will downsample the BENIGN class to be 1.5 times the size of the
    # largest attack class ('DoS_Hulk' with ~230k samples).
    # This keeps a healthy amount of normal traffic without overwhelming the attacks.
    sample_size = int(df_minority['Label'].value_counts().max() * 1.5)
    
    print(f" - Largest attack class has {sample_size // 1.5} samples.")
    print(f" - Undersampling 'BENIGN' class from {len(df_majority)} to {sample_size} samples.")
    
    df_majority_downsampled = df_majority.sample(n=sample_size, random_state=42)
    
    # Combine the downsampled majority class with the minority classes
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    
    # Shuffle the dataset to mix the rows
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalancing complete. New dataset has {len(df_balanced)} rows.")
    print("Final label distribution:")
    print(df_balanced['Label'].value_counts())
    
    return df_balanced

def encode_categorical_features(df):
    """
    Converts all text-based columns into numerical format.
    """
    print("\nEncoding categorical features into numerical format...")
    # Find all columns that are of 'object' type (i.e., text)
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        # We use one-hot encoding for features and label encoding for the target.
        if col == 'Label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            print(f" - LabelEncoded '{col}' column.")
        else:
            # One-hot encoding creates new columns for each category
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
            print(f" - One-Hot Encoded '{col}' column.")
            
    return df


if __name__ == "__main__":
    # Load the clean data from Phase 1
    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    
    # Apply feature engineering
    df_engineered = engineer_features(df)
    
    # Balance the data
    df_balanced = balance_data_with_undersampling(df_engineered)
    
    # Encode all text columns to be numeric
    df_final = encode_categorical_features(df_balanced)

    # Save the final, model-ready dataset
    print(f"\nSaving final model-ready data to '{ENGINEERED_DATA_PATH}'...")
    df_final.to_csv(ENGINEERED_DATA_PATH, index=False)
    print("Done! Phase 2 is complete.")