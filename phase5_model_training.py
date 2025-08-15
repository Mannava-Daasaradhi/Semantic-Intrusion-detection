# phase5_model_training.py (Improved Version)
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# --- Configuration ---
MODEL_READY_DATA_PATH = 'model_ready_data.csv'
LABEL_MAPPING_PATH = 'label_mapping.json'
MODEL_OUTPUT_PATH = 'xgboost_cicids_model.json'

def load_label_mapping(path):
    """Loads the label mapping from a JSON file."""
    print(f"Loading label mapping from '{path}'...")
    with open(path, 'r') as f:
        mapping = json.load(f)
        # JSON saves keys as strings, so convert them back to integers
        return {int(k): v for k, v in mapping.items()}

def plot_feature_importance(model, feature_names):
    """
    Creates and saves a plot of the top 15 most important features.
    """
    print("\nGenerating feature importance plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=15, height=0.5, ax=ax, title='XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print(" - Feature importance plot saved to 'feature_importance.png'")
    plt.close() # Close the plot to free up memory

def train_and_evaluate_model(df, label_map):
    """
    Trains an XGBoost classifier, evaluates its performance, and saves visualizations.
    """
    print("\nStarting model training and evaluation...")

    # --- 1. Prepare Data ---
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f" - Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- 2. Train the XGBoost Model ---
    print("\nTraining the XGBoost model (this may take a few minutes)...")
    model = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss')
    
    model.fit(X_train, y_train)
    print(" - Model training complete.")

    # --- 3. Evaluate the Model ---
    print("\nEvaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    
    # Get the human-readable names for the report
    target_names = [label_map.get(i, 'Unknown') for i in sorted(label_map.keys())]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # --- 4. Visualize the Enhanced Confusion Matrix ---
    print("\nGenerating enhanced confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize for percentages

    # Create annotations that show both count and percentage
    annot = np.array([f"{count}\n({pct:.1%})" for count, pct in zip(cm.flatten(), cm_normalized.flatten())]).reshape(cm.shape)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix (Count and Recall %)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print(" - Enhanced confusion matrix saved to 'confusion_matrix.png'")
    plt.close()

    # --- 5. Plot Feature Importance ---
    plot_feature_importance(model, X.columns)

    # --- 6. Save the Trained Model ---
    print(f"\nSaving the trained model to '{MODEL_OUTPUT_PATH}'...")
    model.save_model(MODEL_OUTPUT_PATH)
    print(f" - Model saved successfully.")


if __name__ == "__main__":
    try:
        model_ready_df = pd.read_csv(MODEL_READY_DATA_PATH)
        label_mapping = load_label_mapping(LABEL_MAPPING_PATH)
        train_and_evaluate_model(model_ready_df, label_mapping)
    except FileNotFoundError as e:
        print(f"\nError: A required file was not found. Please ensure you have run the previous scripts.")
        print(f"Missing file: {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")