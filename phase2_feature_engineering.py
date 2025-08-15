# phase2_feature_engineering.py (Updated)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

# --- Configuration ---
CLEANED_DATA_PATH = 'final_labeled_flows_cicids.csv'
ENGINEERED_DATA_PATH = 'model_ready_data.csv'

def engineer_features(df):
    """
    Engineers new features, cleans the data, and prepares it for modeling.
    """
    print("Starting feature engineering...")

    # --- Step 1: Create New Semantic Features ---
    # Map protocol numbers to their names for better readability and potential model use.
    protocol_map = {6.0: 'TCP', 17.0: 'UDP', 0.0: 'Other'}
    df['Protocol_Name'] = df['Protocol'].map(protocol_map).fillna('Other')
    print(" - Created 'Protocol_Name' feature.")
    
    # Infer the service based on well-known destination ports.
    def map_service(port):
        if port == 80: return 'HTTP'
        if port == 443: return 'HTTPS'
        if port == 21: return 'FTP'
        if port == 22: return 'SSH'
        if port == 53: return 'DNS'
        if port > 1024: return 'Ephemeral'
        return 'Other'
        
    df['Service'] = df['Destination_Port'].apply(map_service)
    print(" - Created 'Service' feature.")

    # --- Step 2: Clean and Drop Columns ---
    # Drop columns that are too specific, redundant, or not useful for a general model.
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
    
    df_minority = df[df['Label'] != 'BENIGN']
    df_majority = df[df['Label'] == 'BENIGN']
    
    # Downsample the BENIGN class to be 1.5 times the size of the largest attack class.
    # This maintains a healthy amount of normal traffic without overwhelming the attacks.
    sample_size = int(df_minority['Label'].value_counts().max() * 1.5)
    
    print(f" - Undersampling 'BENIGN' class from {len(df_majority)} to {sample_size} samples.")
    df_majority_downsampled = df_majority.sample(n=sample_size, random_state=42)
    
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    
    # Shuffle the dataset to ensure random distribution of rows
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalancing complete. New dataset has {len(df_balanced)} rows.")
    print("Final label distribution:")
    print(df_balanced['Label'].value_counts())
    
    return df_balanced

def encode_categorical_features(df):
    """
    Converts all text-based columns into a numerical format suitable for the model.
    Returns the final DataFrame and the fitted LabelEncoder for the target variable.
    """
    print("\nEncoding categorical features into numerical format...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    le = None  # Initialize LabelEncoder variable
    
    for col in categorical_cols:
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
            
    return df, le


if __name__ == "__main__":
    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    
    df_engineered = engineer_features(df)
    
    df_balanced = balance_data_with_undersampling(df_engineered)
    
    # We now get the label encoder back from the function
    df_final, label_encoder = encode_categorical_features(df_balanced)

    print(f"\nSaving final model-ready data to '{ENGINEERED_DATA_PATH}'...")
    df_final.to_csv(ENGINEERED_DATA_PATH, index=False)
    print(" - Model-ready data saved.")

    # --- NEW: Save the Label Encoder Mapping ---
    if label_encoder:
        # Create a mapping from the encoded number back to the original attack name
        label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        
        print("\nSaving label mapping to 'label_mapping.json'...")
        with open('label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=4)
        print(" - Label mapping saved.")
    
    print("\nPhase 2 is complete.")