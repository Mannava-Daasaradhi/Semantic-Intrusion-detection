# phase2_advanced_semantics.py (Corrected with Countdown)
import pandas as pd
from scapy.all import PcapReader, TCP, UDP, Raw
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings

# Suppress Scapy's verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scapy")

# --- Configuration ---
PCAP_FILE_PATH = r'C:\Users\daasa\OneDrive\Desktop\XAICN\CIC-IDS-2017\Friday-WorkingHours.pcap'
OUTPUT_CSV_PATH = 'semantic_features_dataset.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 256

def get_packet_count(pcap_path):
    """Reads through the pcap file once just to get a total count."""
    print("Pre-counting packets in the file for progress bar...")
    count = 0
    with PcapReader(pcap_path) as pcap_reader:
        for _ in pcap_reader:
            count += 1
    return count

def extract_semantic_features():
    """
    Reads a .pcap file in a stream, extracts payloads, and generates embeddings.
    """
    print(f"Loading Sentence Transformer model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print(" - Model loaded successfully.")

    # --- FIX: Pre-count the packets to enable the countdown ---
    try:
        total_packets = get_packet_count(PCAP_FILE_PATH)
        print(f" - Found {total_packets} total packets.")
    except FileNotFoundError:
        print(f"Error: The pcap file '{PCAP_FILE_PATH}' was not found.")
        return

    packet_data = []
    print(f"\nStreaming packets from '{PCAP_FILE_PATH}' and extracting payloads...")
    
    with PcapReader(PCAP_FILE_PATH) as pcap_reader:
        # --- FIX: Configure tqdm to show remaining count ---
        progress_bar = tqdm(
            pcap_reader, 
            total=total_packets, 
            desc="Reading Packets",
            # This format shows description, percentage, bar, count, and remaining
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining} left]'
        )

        for packet in progress_bar:
            payload_text = ""
            if packet.haslayer(TCP) and packet.haslayer(Raw):
                try:
                    payload_text = packet[Raw].load.decode('utf-8', errors='ignore')
                except Exception: continue
            elif packet.haslayer(UDP) and packet.haslayer(Raw):
                try:
                    payload_text = packet[Raw].load.decode('utf-8', errors='ignore')
                except Exception: continue
            
            if payload_text:
                packet_data.append({
                    'timestamp': float(packet.time),
                    'payload_text': payload_text
                })

    if not packet_data:
        print("\nWarning: No valid TCP/UDP payloads with text data found in the pcap file.")
        return

    print(f"\nFound {len(packet_data)} packets with text payloads.")

    print("Generating embeddings for all payloads...")
    
    all_payloads = [data['payload_text'] for data in packet_data]
    
    embeddings = model.encode(
        all_payloads, 
        show_progress_bar=True,
        batch_size=BATCH_SIZE
    )

    print("\nCombining results...")
    for i, data in enumerate(packet_data):
        data['embedding'] = embeddings[i]

    df = pd.DataFrame(packet_data)
    
    embedding_df = pd.DataFrame(df['embedding'].to_list(), index=df.index).add_prefix('embed_')
    final_df = pd.concat([df.drop('embedding', axis=1), embedding_df], axis=1)

    print(f"Saving new dataset with semantic features to '{OUTPUT_CSV_PATH}'...")
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(" - Done!")
    print(f"\nDataset preview:\n{final_df.head()}")


if __name__ == "__main__":
    extract_semantic_features()