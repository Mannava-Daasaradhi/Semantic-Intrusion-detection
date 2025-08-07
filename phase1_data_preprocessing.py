# phase1_data_preprocessing.py
import os
from scapy.all import rdpcap
import pandas as pd

def process_pcap_files(directory_path):
    """
    Reads all pcap files in a directory and extracts detailed information,
    including initial semantic protocol tags.

    Args:
        directory_path (str): The path to the directory containing pcap files.

    Returns:
        pandas.DataFrame: A DataFrame with combined packet information.
    """
    all_packet_data = []

    # Iterate through all files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pcap"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {filename}")

            try:
                packets = rdpcap(file_path)
                print(f"  - Successfully read {len(packets)} packets.")
            except Exception as e:
                print(f"  - Error reading {filename}: {e}")
                continue

            # Process each packet in the file
            for packet in packets:
                packet_info = {}
                
                # Check for IP layer
                if 'IP' in packet:
                    packet_info['Source_IP'] = packet['IP'].src
                    packet_info['Destination_IP'] = packet['IP'].dst
                    
                    # --- Initial Semantic Tagging: Protocol ---
                    # Check for TCP/UDP layer to assign a protocol tag
                    if 'TCP' in packet:
                        packet_info['Protocol'] = 'TCP'
                        packet_info['Source_Port'] = packet['TCP'].sport
                        packet_info['Destination_Port'] = packet['TCP'].dport
                    elif 'UDP' in packet:
                        packet_info['Protocol'] = 'UDP'
                        packet_info['Source_Port'] = packet['UDP'].sport
                        packet_info['Destination_Port'] = packet['UDP'].dport
                    elif packet['IP'].proto == 1: # ICMP protocol number
                        packet_info['Protocol'] = 'ICMP'
                        packet_info['Source_Port'] = None
                        packet_info['Destination_Port'] = None
                    else:
                        packet_info['Protocol'] = 'Other'
                        packet_info['Source_Port'] = None
                        packet_info['Destination_Port'] = None
                
                # REVISED LOGIC: Check for HTTP/DNS layers safely
                # First, check if the Raw layer exists
                if 'Raw' in packet:
                    payload = str(packet['Raw'].load)
                    if 'GET' in payload or 'POST' in payload:
                        packet_info['Protocol'] = 'HTTP'
                
                # Check for DNS layer safely
                if 'DNS' in packet:
                    packet_info['Protocol'] = 'DNS'

                packet_info['Packet_Size'] = len(packet)
                all_packet_data.append(packet_info)

    df = pd.DataFrame(all_packet_data)
    return df

if __name__ == "__main__":
    # Path updated to match your screenshot
    pcap_directory = 'CIC-IDS-2017/PCAPs/'
    
    # Check if the directory exists
    if not os.path.exists(pcap_directory):
        print(f"Error: Directory '{pcap_directory}' not found. Please make sure the folder name is correct.")
    else:
        # Process all files and get the final DataFrame
        combined_df = process_pcap_files(pcap_directory)
        
        if combined_df is not None and not combined_df.empty:
            print("\n--- Combined Packet Data (First 5 rows) ---")
            print(combined_df.head())
            
            print(f"\nTotal packets processed from all files: {len(combined_df)}")
            
            # Save the processed data to a CSV for later use
            output_csv_path = 'processed_cicids2017_packets.csv'
            combined_df.to_csv(output_csv_path, index=False)
            print(f"Processed data saved to {output_csv_path}")