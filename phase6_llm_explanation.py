# phase6_llm_explanation.py (with Plot Generation)
import pandas as pd
import xgboost as xgb
from transformers import pipeline, set_seed
import warnings
import torch
import json
import matplotlib.pyplot as plt
import os # --- ADDED LINE ---

# --- Configuration ---
MODEL_READY_DATA_PATH = 'model_ready_data.csv'
MODEL_PATH = 'xgboost_cicids_model.json'
LABEL_MAPPING_PATH = 'label_mapping.json'
PLOT_OUTPUT_FOLDER = 'explanation_plots' # --- ADDED LINE ---

def generate_feature_plot(features, attack_name):
    """
    Creates and saves a bar chart of key traffic features.
    """
    features_to_plot = {
        'Packets/s': features.get('Flow_Packets/s', 0),
        'Bytes/s': features.get('Flow_Bytes/s', 0),
        'Total Fwd Pkts': features.get('Total_Fwd_Packets', 0),
        'Total Bwd Pkts': features.get('Total_Backward_Packets', 0)
    }
    
    plot_values = [v + 1 for v in features_to_plot.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(features_to_plot.keys(), plot_values, color=['#ff6666', '#ffcc66', '#66b3ff', '#99ff99'])
    plt.yscale('log')
    plt.title(f'Key Indicators for "{attack_name}" Attack', fontsize=16)
    plt.ylabel('Value (Log Scale)', fontsize=12)
    plt.xticks(rotation=10, fontsize=10)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval-1:.2f}', va='bottom', ha='center')

    # --- MODIFIED SECTION ---
    # Create the output folder if it doesn't exist
    os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
    
    # Save the plot with a unique name inside the new folder
    filename = f"explanation_plot_{attack_name.replace(' ', '_')}.png"
    save_path = os.path.join(PLOT_OUTPUT_FOLDER, filename)
    plt.savefig(save_path)
    print(f"--- Saved feature plot to '{save_path}' ---")
    plt.close()

def generate_llm_explanation(explainer, model, sample_data, label_map):
    """
    Generates a text explanation and a feature plot for an attack sample.
    """
    print("\n" + "="*80)
    
    label_id = sample_data['Label'].iloc[0]
    predicted_label = label_map.get(str(label_id), "Unknown Attack")
    print(f"--- Generating Explanation for a Sample '{predicted_label}' Attack ---")

    sample_features = sample_data.iloc[0].to_dict()
    
    generate_feature_plot(sample_features, predicted_label)

    prompt = f"""
Analysis of a network event flagged as a '{predicted_label}' attack.

Key Indicators:
- Packets per second: {sample_features.get('Flow_Packets/s', 'N/A'):.2f}
- Bytes per second: {sample_features.get('Flow_Bytes/s', 'N/A'):.2f}
- Total Forward Packets: {int(sample_features.get('Total_Fwd_Packets', 0))}
- Total Backward Packets: {int(sample_features.get('Total_Backward_Packets', 0))}

Explanation for Security Analyst: This traffic is highly indicative of a "{predicted_label}" attack because"""

    print("\n--- Sending Prompt to LLM ---")
    print(prompt)

    llm_output = explainer(
        prompt, 
        max_new_tokens=150,
        num_return_sequences=1,
        pad_token_id=explainer.tokenizer.eos_token_id
    )
    explanation = llm_output[0]['generated_text']

    print("\n--- AI-Generated Explanation ---")
    print(explanation)
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        df = pd.read_csv(MODEL_READY_DATA_PATH)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(MODEL_PATH)
        
        with open(LABEL_MAPPING_PATH, 'r') as f:
            label_mapping = json.load(f)

        print("Loading language model (gpt2)... This only happens once.")
        device = 0 if torch.cuda.is_available() else -1
        explainer_pipeline = pipeline("text-generation", model="gpt2", device=device)
        set_seed(42)
        print(f" - Model loaded successfully on device: {'cuda' if device == 0 else 'cpu'}.")

        unique_labels = df['Label'].unique()
        
        for label_id in unique_labels:
            label_name = label_mapping.get(str(label_id))
            if label_name == 'BENIGN':
                continue
            
            attack_sample = df[df['Label'] == label_id].head(1)
            
            if not attack_sample.empty:
                generate_llm_explanation(explainer_pipeline, xgb_model, attack_sample, label_mapping)
            else:
                print(f"Could not find a sample for attack type: {label_name}")

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")