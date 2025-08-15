# phase6_llm_explanation.py (Final Version with gpt2)
import pandas as pd
import xgboost as xgb
from transformers import pipeline, set_seed
import warnings
import torch

# --- Configuration ---
MODEL_READY_DATA_PATH = 'model_ready_data.csv'
MODEL_PATH = 'xgboost_cicids_model.json'

def generate_llm_explanation(model, sample_data, label_map):
    """
    Uses the gpt2 model from Hugging Face to generate a coherent explanation.
    """
    print("\n--- Generating Explanation for a Sample Attack ---")

    # --- 1. Load a More Powerful LLM (gpt2) ---
    print("Loading language model (gpt2)...")
    device = 0 if torch.cuda.is_available() else -1
    # We use the standard 'gpt2' model for better quality generation
    explainer = pipeline("text-generation", model="gpt2", device=device)
    print(f" - Model loaded successfully on device: {'cuda' if device == 0 else 'cpu'}.")
    set_seed(42)

    # --- 2. Make a Prediction ---
    prediction_index = model.predict(sample_data.drop('Label', axis=1))[0]
    predicted_label = label_map.get(prediction_index, "Unknown Attack")
    print(f" - XGBoost model detected: '{predicted_label}'")

    # --- 3. Use the "Completion" Style Prompt ---
    sample_features = sample_data.iloc[0].to_dict()
    prompt = f"""
Analysis of a network event flagged as a '{predicted_label}' attack.

Key Indicators:
- Extremely high packets per second: {sample_features.get('Flow_Packets/s', 'N/A'):.2f}
- Negligible data transfer (Bytes/s): {sample_features.get('Flow_Bytes/s', 'N/A'):.2f}
- One-way traffic with no return packets: {int(sample_features.get('Total_Backward_Packets', 0))} backward packets.

Explanation for Security Analyst: This traffic is highly indicative of a "{predicted_label}" attack because"""

    print("\n--- Sending Final Prompt to LLM ---")
    print(prompt)

    # --- 4. Generate the Explanation ---
    llm_output = explainer(
        prompt, 
        max_new_tokens=75,
        num_return_sequences=1,
        pad_token_id=explainer.tokenizer.eos_token_id
    )
    explanation = llm_output[0]['generated_text']

    print("\n--- AI-Generated Explanation ---")
    print(explanation)


if __name__ == "__main__":
    try:
        df = pd.read_csv(MODEL_READY_DATA_PATH)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(MODEL_PATH)

        labels_df = pd.read_csv('final_labeled_flows_cicids.csv', usecols=['Label'])
        labels_df['Label'] = labels_df['Label'].str.replace(' ', '_')
        unique_labels = sorted(labels_df['Label'].unique())
        target_names = {i: label for i, label in enumerate(unique_labels)}

        attack_sample = df[df['Label'] == 4].head(1) 
        
        if not attack_sample.empty:
            generate_llm_explanation(xgb_model, attack_sample, target_names)
        else:
            print("Could not find a sample for the specified attack type (Label == 4).")

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")