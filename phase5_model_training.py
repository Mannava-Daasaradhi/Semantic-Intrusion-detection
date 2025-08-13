# phase5_model_training.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
MODEL_READY_DATA_PATH = 'model_ready_data.csv'

def train_and_evaluate_model(df):
    """
    Trains an XGBoost classifier and evaluates its performance.
    """
    print("Starting model training and evaluation...")

    # --- 1. Prepare Data ---
    # Separate features (X) from the target variable (y)
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Split the data into training and testing sets
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f" - Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- 2. Train the XGBoost Model ---
    print("\nTraining the XGBoost model (this may take a few minutes)...")
    # Initialize the XGBoost classifier
    # 'use_label_encoder=False' and 'eval_metric='mlogloss'' are modern defaults
    model = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss')
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    print(" - Model training complete.")

    # --- 3. Evaluate the Model ---
    print("\nEvaluating model performance on the test set...")
    # Make predictions on the unseen test data
    y_pred = model.predict(X_test)
    
    # To make the report readable, we need the original label names.
    # We'll create a simple mapping. A more robust solution would save/load the encoder from Phase 2.
    # NOTE: This mapping must be updated if the label encoding changes.
    labels_df = pd.read_csv('final_labeled_flows_cicids.csv', usecols=['Label'])
    labels_df['Label'] = labels_df['Label'].str.replace(' ', '_')
    unique_labels = sorted(labels_df['Label'].unique())
    target_names = {i: label for i, label in enumerate(unique_labels)}

    # Print the classification report
    print("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=[target_names.get(i, 'Unknown') for i in range(len(target_names))])
    print(report)
    
    # --- 4. Visualize the Confusion Matrix ---
    print("\nGenerating confusion matrix visualization...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[target_names.get(i, 'Unknown') for i in range(len(target_names))], 
                yticklabels=[target_names.get(i, 'Unknown') for i in range(len(target_names))])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print(" - Confusion matrix saved to 'confusion_matrix.png'")


if __name__ == "__main__":
    try:
        df = pd.read_csv(MODEL_READY_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{MODEL_READY_DATA_PATH}' was not found. Please run the Phase 2 script first.")
    else:
        train_and_evaluate_model(df)