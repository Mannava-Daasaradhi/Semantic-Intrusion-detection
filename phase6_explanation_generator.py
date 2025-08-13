# phase6_explanation_generator.py
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

# --- Configuration ---
MODEL_READY_DATA_PATH = 'model_ready_data.csv'
MODEL_PATH = 'xgb_model.json' # We'll save our trained model here

# Note: We need to re-run the training from Phase 5 to save the model.
# This is a modified version of the Phase 5 script.

def train_and_save_model(df):
    """Trains and saves the XGBoost model."""
    print("Training and saving the model...")
    X = df.drop('Label', axis=1)
    y = df['Label']
    model = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y) # Train on all data for the final model
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def generate_explanation(model, sample_data):
    """
    Simulates generating an LLM explanation for a given data sample.
    """
    print("\n--- Generating Explanation for a Sample Attack ---")
    
    # We need the original label names for the explanation.
    labels_df = pd.read_csv('final_labeled_flows_cicids.csv', usecols=['Label'])
    labels_df['Label'] = labels_df['Label'].str.replace(' ', '_')
    unique_labels = sorted(labels_df['Label'].unique())
    target_names = {i: label for i, label in enumerate(unique_labels)}
    
    # Use the model to predict the attack type of the sample
    prediction_index = model.predict(sample_data)[0]
    predicted_label = target_names.get(prediction_index, "Unknown Attack")
    
    # Prepare the features for the prompt
    sample_features = sample_data.iloc[0].to_dict()
    
    # ---- This is where you would format the prompt for an LLM ----
    prompt = f"""
    Explain this network attack in simple terms for a security analyst.
    
    Attack Detected: {predicted_label}
    
    Key Traffic Features:
    - Flow Duration: {sample_features.get('Flow_Duration', 'N/A'):.2f} microseconds
    - Protocol: {sample_features.get('Protocol_Name_TCP', 0) == 1 and 'TCP' or 'UDP/Other'}
    - Service: {sample_features.get('Service_HTTP', 0) == 1 and 'HTTP' or 'Other'}
    - Total Forward Packets: {sample_features.get('Total_Fwd_Packets', 'N/A')}
    - Total Backward Packets: {sample_features.get('Total_Backward_Packets', 'N/A')}
    - Flow Bytes/s: {sample_features.get('Flow_Bytes/s', 'N/A'):.2f}
    """
    
    print("\n--- LLM Prompt ---")
    print(prompt)
    
    # ---- Simulated LLM Response ----
    # In a real system, you would send the prompt to an LLM API (like Gemini).
    # Here, we will just generate a template response.
    
    explanation = f"Alert: A '{predicted_label}' attack was detected. The connection was identified as suspicious due to its behavioral characteristics, including an unusual flow duration and a high rate of data transfer, consistent with known patterns for this type of attack."
    
    print("\n--- Simulated LLM Explanation ---")
    print(explanation)


if __name__ == "__main__":
    df = pd.read_csv(MODEL_READY_DATA_PATH)
    
    # First, we need a saved model file.
    # Let's train and save it.
    model = train_and_save_model(df)
    
    # Let's find an interesting sample to explain.
    # We'll find the first 'DoS_Hulk' attack in the dataset.
    attack_sample = df[df['Label'] == 4].head(1) # 4 corresponds to DoS_Hulk
    
    if not attack_sample.empty:
        # Separate features from the label for the sample
        attack_features = attack_sample.drop('Label', axis=1)
        generate_explanation(model, attack_features)
    else:
        print("Could not find a sample for the specified attack type.")