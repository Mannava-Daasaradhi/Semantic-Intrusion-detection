# phase2_advanced_semantics.py
import pandas as pd
from scapy.all import rdpcap, TCP, UDP, Raw
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings

# Suppress Scapy's verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scapy")

# --- Configuration ---
# You must have the original .pcap file that corresponds to the dataset.
PCAP_FILE_PATH = r'C:\Users\daasa\OneDrive\Desktop\XAICN\CIC-IDS-2017\Friday-WorkingHours.pcap' # Or whatever the full name is
OUTPUT_CSV_PATH = 'semantic_features_dataset.csv'
# Using a lightweight but powerful model for generating embeddings.
MODEL_NAME = 'all-MiniLM-L6-v2'

def extract_semantic_features():
    """
    Reads a .pcap file, extracts packet payloads, and generates semantic embeddings.
    """
    print(f"Loading Sentence Transformer model: '{MODEL_NAME}'...")
    # This will download the model the first time you run it.
    model = SentenceTransformer(MODEL_NAME)
    print(" - Model loaded successfully.")

    try:
        print(f"Reading packets from '{PCAP_FILE_PATH}'... (This may take a while)")
        packets = rdpcap(PCAP_FILE_PATH)
        print(f" - Found {len(packets)} packets.")
    except FileNotFoundError:
        print(f"Error: The pcap file '{PCAP_FILE_PATH}' was not found.")
        print("Please download the original .pcap files for the CIC-IDS-2017 dataset.")
        return

    results = []
    
    # Use tqdm for a progress bar as this can be a slow process
    print("Processing packets and generating embeddings...")
    for packet in tqdm(packets, desc="Analyzing Packets"):
        payload_text = ""
        # We are interested in the payload inside TCP or UDP packets
        if packet.haslayer(TCP) and packet.haslayer(Raw):
            try:
                # Decode the raw payload into a string, ignoring non-UTF-8 characters
                payload_text = packet[Raw].load.decode('utf-8', errors='ignore')
            except Exception:
                continue # Skip if payload cannot be decoded
        
        elif packet.haslayer(UDP) and packet.haslayer(Raw):
            try:
                payload_text = packet[Raw].load.decode('utf-8', errors='ignore')
            except Exception:
                continue

        # Only process packets that had a text payload
        if payload_text:
            # Generate the semantic embedding for the payload text
            embedding = model.encode(payload_text)
            
            # Store the embedding along with some basic info
            # In a full system, you would merge this with the full flow stats
            results.append({
                'timestamp': float(packet.time),
                'payload_text': payload_text,
                'embedding': embedding
            })

    if not results:
        print("\nWarning: No valid TCP/UDP payloads with text data found in the pcap file.")
        print("This can happen if the traffic is mostly encrypted or doesn't contain Raw data.")
        return

    # --- Save the results to a DataFrame ---
    print(f"\nSuccessfully processed {len(results)} packets with payloads.")
    df = pd.DataFrame(results)
    
    # The 'embedding' column is a numpy array. We'll split it into separate columns.
    embedding_df = pd.DataFrame(df['embedding'].to_list(), index=df.index)
    embedding_df = embedding_df.add_prefix('embed_')

    # Combine the original data with the new embedding columns
    final_df = pd.concat([df.drop('embedding', axis=1), embedding_df], axis=1)

    print(f"Saving new dataset with semantic features to '{OUTPUT_CSV_PATH}'...")
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(" - Done!")
    print(f"\nDataset preview:\n{final_df.head()}")


if __name__ == "__main__":
    extract_semantic_features()