# live_engine_final.py
import xgboost as xgb
from scapy.all import sniff, IP, TCP, UDP
import numpy as np
import json
import time
import threading

class Flow:
    def __init__(self, start_time, packet):
        self.start_time = start_time
        self.last_time = start_time
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.src_ip = packet[IP].src
        self.dst_ip = packet[IP].dst
        self.add_packet(packet, 'fwd')

    def add_packet(self, packet, direction):
        self.last_time = time.time()
        packet_len = len(packet[IP].payload)
        if direction == 'fwd':
            self.fwd_packets += 1
            self.fwd_bytes += packet_len
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_len

    def get_features(self, model):
        flow_duration = self.last_time - self.start_time
        if flow_duration == 0: flow_duration = 1e-6
        all_feature_names = model.get_booster().feature_names
        features = {fname: 0 for fname in all_feature_names}
        features['Flow_Duration'] = flow_duration
        features['Total_Fwd_Packets'] = self.fwd_packets
        features['Total_Backward_Packets'] = self.bwd_packets
        features['Total_Length_of_Fwd_Packets'] = self.fwd_bytes
        features['Total_Length_of_Bwd_Packets'] = self.bwd_bytes
        features['Flow_Bytes/s'] = (self.fwd_bytes + self.bwd_bytes) / flow_duration
        features['Flow_Packets/s'] = (self.fwd_packets + self.bwd_packets) / flow_duration
        return np.array([features[fname] for fname in all_feature_names]).reshape(1, -1)

class TrafficAnalyzer:
    def __init__(self, ui_queue):
        self.ui_queue = ui_queue
        self.active_flows = {}
        self.model = xgb.XGBClassifier()
        self.model.load_model('xgboost_cicids_model.json')
        with open('label_mapping.json', 'r') as f:
            mapping = json.load(f)
            self.label_map = {int(k): v for k, v in mapping.items()}
        self.stop_sniffing = threading.Event()

    def process_packet(self, packet):
        if not (packet.haslayer(TCP) or packet.haslayer(UDP)): return
        current_time = time.time()
        flow_key = (packet[IP].src, packet.sport, packet[IP].dst, packet.dport)
        rev_flow_key = (packet[IP].dst, packet.dport, packet[IP].src, packet.sport)
        if flow_key in self.active_flows:
            self.active_flows[flow_key].add_packet(packet, 'fwd')
        elif rev_flow_key in self.active_flows:
            self.active_flows[rev_flow_key].add_packet(packet, 'bwd')
        else:
            self.active_flows[flow_key] = Flow(current_time, packet)

    def check_flow_timeouts(self):
        while not self.stop_sniffing.is_set():
            time.sleep(5)
            current_time = time.time()
            timed_out_flows = [k for k, v in self.active_flows.items() if current_time - v.last_time > 30]
            for flow_key in timed_out_flows:
                if flow_key in self.active_flows:
                    flow = self.active_flows.pop(flow_key)
                    feature_vector = flow.get_features(self.model)
                    prediction_encoded = self.model.predict(feature_vector)[0]
                    prediction_label = self.label_map.get(prediction_encoded, "Unknown")
                    
                    log_message = f"Flow {flow.src_ip}:{flow_key[1]} -> {flow.dst_ip}:{flow_key[3]}: Prediction = {prediction_label}"
                    self.ui_queue.put(log_message)
                    if prediction_label != "BENIGN":
                        alert_message = f"[!] ALERT: Potential '{prediction_label}' activity detected!"
                        self.ui_queue.put(alert_message)
    
    def start(self):
        self.ui_queue.put("Packet sniffing thread started.")
        timeout_thread = threading.Thread(target=self.check_flow_timeouts, daemon=True)
        timeout_thread.start()
        sniff(prn=self.process_packet, store=0, stop_filter=lambda p: self.stop_sniffing.is_set())
        self.ui_queue.put("Packet sniffing thread stopped.")

    def stop(self):
        self.stop_sniffing.set()