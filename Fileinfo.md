Project: S-XG-NID - A Semantic-Enhanced Dual-Modality Intrusion Detection System
Date: 23-08-2025
--------------------------------------------------------------------------------

This document provides a summary of each Python script in the project, detailing its purpose, inputs, outputs, and connection to the project plan.

================================================================================
File: phase1_data_preprocessing.py
================================================================================
- Purpose: Combines multiple raw CSV files from the CIC-IDS-2017 dataset into a single, clean master file.
- Process: Reads all CSVs, merges them, standardizes column names, and removes duplicate or invalid rows.
- Input: Raw CSV files from the 'GeneratedLabelledFlows' directory.
- Output: A single clean dataset named 'final_labeled_flows_cicids.csv'.
- [cite_start]Project Phase: 1 (Data Collection & Preprocessing) [cite: 31]

================================================================================
File: phase2_feature_engineering.py
================================================================================
- Purpose: Enriches the clean data with basic semantic features and converts it into a numerical format for machine learning.
- Process: Adds 'Protocol_Name' and 'Service' columns, balances the dataset via undersampling, and encodes text features into numbers.
- Input: 'final_labeled_flows_cicids.csv'.
- Outputs: 'model_ready_data.csv' and 'label_mapping.json'.
- [cite_start]Project Phase: 1 (Semantic Enrichment) [cite: 87]

================================================================================
File: phase2_advanced_semantics.py
================================================================================
- Purpose: Implements the advanced semantic analysis by extracting meaning from raw packet content.
- Process: Reads a .pcap file, extracts TCP/UDP payloads, and uses a Sentence-Transformers model to generate semantic embedding vectors.
- Input: A raw .pcap file (e.g., 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.pcap').
- Output: A new dataset named 'semantic_features_dataset.csv' containing the meaning vectors.
- [cite_start]Project Phase: 2 (Semantic Feature Extraction) [cite: 92]

================================================================================
File: phase3_knowledge_graph.py
================================================================================
- Purpose: To create a visual representation of the network behavior and relationships.
- Process: Builds a graph where nodes are protocols and services, and edges represent the type of attack, then saves a visualization.
- Input: 'model_ready_data.csv'.
- Output: An image file named 'knowledge_graph.png'.
- [cite_start]Project Phase: 3 (Knowledge Graph Construction for visualization) [cite: 99]

================================================================================
File: phase4_5_hgnn_implementation.py
================================================================================
- Purpose: Builds and trains the advanced, graph-based Heterogeneous Graph Neural Network (HGNN) model.
- Process: Constructs a heterogeneous graph with IP and Flow nodes and trains a GNN model using PyTorch Geometric to learn from the network's structure.
- Input: 'final_labeled_flows_cicids.csv'.
- Output: A trained HGNN model and its accuracy score printed to the console.
- [cite_start]Project Phase: 4 (Heterogeneous Graph Construction) [cite: 107] [cite_start]& 5 (Dual-Modality Learning) [cite: 112]

================================================================================
File: phase5_model_training.py
================================================================================
- Purpose: Builds and trains the high-accuracy statistical XGBoost model.
- Process: Trains an XGBoost classifier on the statistical features and evaluates its performance with detailed reports and visualizations.
- Input: 'model_ready_data.csv' and 'label_mapping.json'.
- Outputs: 'xgboost_cicids_model.json' (trained model), 'confusion_matrix.png', 'feature_importance.png'.
- [cite_start]Project Phase: 5 (Dual-Modality Learning) [cite: 112] [cite_start]& 7 (Evaluation) [cite: 126]

================================================================================
File: phase6_llm_explanation.py
================================================================================
- Purpose: Generates human-readable explanations for why a detected activity is considered malicious.
- Process: Loads the trained XGBoost model, creates a contextual prompt for a sample attack, and uses a pre-trained LLM (GPT-2) to generate a text explanation.
- Input: 'xgboost_cicids_model.json', 'model_ready_data.csv'.
- Output: A text explanation printed to the console.
- [cite_start]Project Phase: 6 (Semantic-Aware Explanation Layer) [cite: 119]

================================================================================
Files: live_engine_final.py & app_gui.py
================================================================================
- Purpose: To create a functional, real-time desktop application for intrusion detection.
- Process: The 'live_engine_final.py' script uses Scapy to capture and analyze live traffic. The 'app_gui.py' script creates the user interface with CustomTkinter, runs the engine in a background thread, and displays logs and alerts.
- Input: Live network traffic.
- Output: A running desktop application that provides real-time alerts.
- [cite_start]Project Phase: Implements the "Final Output" goal of the project. [cite: 81]

================================================================================
File: run_training_pipeline.py
================================================================================
- Purpose: To automate the entire model training process.
- Process: Executes the data processing and model training scripts (Phases 1, 2, 5, and 4) sequentially.
- Input: None.
- Output: All the model files and datasets created by the individual scripts.
- Project Phase: An automation utility for the entire pipeline.