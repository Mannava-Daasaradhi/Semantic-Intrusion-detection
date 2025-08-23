# live_engine.py (Corrected Version)
import xgboost as xgb
from scapy.all import sniff, IP, TCP, UDP
import numpy as np
import json
import time
import threading
from collections import defaultdict

# --- Configuration ---
MODEL_PATH = 'xgboost_cicids_model.json'
LABEL_MAPPING_PATH = 'label_mapping.json'
FLOW_TIMEOUT = 60  # Seconds to wait before considering a flow inactive

# --- Flow Class to Store Flow Statistics ---
class Flow:
    def __init__(self, start_time, packet):
        self.start_time = start_time
        self.last_time = start_time
        self.packets = []
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.src_ip = packet[IP].src
        self.dst_ip = packet[IP].dst
        # Add the first packet
        self.add_packet(packet, 'fwd')

    def add_packet(self, packet, direction):
        self.packets.append(packet)
        self.last_time = time.time()
        
        packet_len = len(packet[IP].payload)
        
        if direction == 'fwd':
            self.fwd_packets += 1
            self.fwd_bytes += packet_len
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_len

    def get_features(self):
        """Calculates and returns the final feature vector for the model."""
        flow_duration = self.last_time - self.start_time
        if flow_duration == 0:
            flow_duration = 1e-6

        all_feature_names = model.get_booster().feature_names
        features = {fname: 0 for fname in all_feature_names}

        features['Flow_Duration'] = flow_duration
        features['Total_Fwd_Packets'] = self.fwd_packets
        features['Total_Backward_Packets'] = self.bwd_packets
        features['Total_Length_of_Fwd_Packets'] = self.fwd_bytes
        features['Total_Length_of_Bwd_Packets'] = self.bwd_bytes
        features['Flow_Bytes/s'] = (self.fwd_bytes + self.bwd_bytes) / flow_duration
        features['Flow_Packets/s'] = (self.fwd_packets + self.bwd_packets) / flow_duration
        
        # In a full implementation, you would calculate all ~80 features here.
        
        return np.array([features[fname] for fname in all_feature_names]).reshape(1, -1)


# --- Main Traffic Analyzer Class ---
class TrafficAnalyzer:
    def __init__(self):
        self.active_flows = {}
        self.model = self.load_model()
        self.label_map = self.load_labels()
        self.stop_sniffing = threading.Event()

    def load_model(self):
        print("Loading XGBoost model...")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        print(" - Model loaded successfully.")
        return model

    def load_labels(self):
        print("Loading label mapping...")
        with open(LABEL_MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
        print(" - Label mapping loaded.")
        return {int(k): v for k, v in mapping.items()}

    def process_packet(self, packet):
        """Callback function for Scapy's sniff()."""
        # --- THIS IS THE FIX ---
        # We only process packets that have a TCP or UDP layer.
        if not (packet.haslayer(TCP) or packet.haslayer(UDP)):
            return
        # --- END OF FIX ---

        current_time = time.time()
        # Now it is safe to access .sport and .dport
        flow_key = (packet[IP].src, packet.sport, packet[IP].dst, packet.dport)
        rev_flow_key = (packet[IP].dst, packet.dport, packet[IP].src, packet.sport)

        if flow_key in self.active_flows:
            self.active_flows[flow_key].add_packet(packet, 'fwd')
        elif rev_flow_key in self.active_flows:
            self.active_flows[rev_flow_key].add_packet(packet, 'bwd')
        else:
            self.active_flows[flow_key] = Flow(current_time, packet)

    def check_flow_timeouts(self):
        """Periodically checks for and processes inactive flows."""
        while not self.stop_sniffing.is_set():
            time.sleep(10)
            current_time = time.time()
            timed_out_flows = []
            
            for flow_key, flow in list(self.active_flows.items()):
                if current_time - flow.last_time > FLOW_TIMEOUT:
                    timed_out_flows.append(flow_key)
            
            if timed_out_flows:
                print(f"\n--- Found {len(timed_out_flows)} timed-out flow(s) ---")
            
            for flow_key in timed_out_flows:
                if flow_key in self.active_flows:
                    flow = self.active_flows.pop(flow_key)
                    feature_vector = flow.get_features()
                    
                    prediction_encoded = self.model.predict(feature_vector)[0]
                    prediction_label = self.label_map.get(prediction_encoded, "Unknown")
                    
                    print(f"Flow {flow.src_ip}:{flow_key[1]} -> {flow.dst_ip}:{flow_key[3]}: Prediction = {prediction_label}")
                    if prediction_label != "BENIGN":
                        print(f"  [!] ALERT: Potential malicious activity detected!")
    
    def start(self):
        """Starts the traffic analysis."""
        print("Starting live traffic analysis engine...")
        
        timeout_thread = threading.Thread(target=self.check_flow_timeouts, daemon=True)
        timeout_thread.start()
        
        print("Packet sniffing started. Press Ctrl+C to stop.")
        try:
            sniff(prn=self.process_packet, store=0, stop_filter=lambda p: self.stop_sniffing.is_set())
        except (KeyboardInterrupt, OSError): # OSError can happen on Windows when stopping
            print("\nStopping analysis...")
            self.stop_sniffing.set()
            timeout_thread.join(timeout=1)
            print("Engine stopped.")

if __name__ == "__main__":
    # Load the global model once
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    analyzer = TrafficAnalyzer()
    analyzer.start()