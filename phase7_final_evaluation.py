# phase7_final_evaluation.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# --- Configuration ---
MODEL_READY_DATA_PATH = 'model_ready_data.csv'

def final_evaluation(df):
    """
    Performs a final, comprehensive evaluation of the model, including ROC-AUC score.
    """
    print("Starting final model evaluation...")

    # --- 1. Prepare Data ---
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # We need the original label names for the reports
    labels_df = pd.read_csv('final_labeled_flows_cicids.csv', usecols=['Label'])
    labels_df['Label'] = labels_df['Label'].str.replace(' ', '_')
    unique_labels = sorted(labels_df['Label'].unique())
    target_names = {i: label for i, label in enumerate(unique_labels)}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f" - Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    # --- 2. Train the XGBoost Model ---
    print("\nTraining the final XGBoost model...")
    model = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    print(" - Model training complete.")

    # --- 3. Evaluate the Model ---
    print("\n--- Final Performance Evaluation ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Print the Classification Report (as before)
    print("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=[target_names.get(i, 'Unknown') for i in range(len(target_names))])
    print(report)

    # Calculate and print the ROC-AUC Score
    # The 'One-vs-Rest' (ovr) strategy is used for multi-class problems.
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        print("\n--- ROC-AUC Score ---")
        print(f"The weighted average ROC-AUC score is: {roc_auc:.4f}")
    except Exception as e:
        print(f"\nCould not compute ROC-AUC score: {e}")

if __name__ == "__main__":
    try:
        df = pd.read_csv(MODEL_READY_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{MODEL_READY_DATA_PATH}' was not found. Please run the Phase 2 script first.")
    else:
        final_evaluation(df)