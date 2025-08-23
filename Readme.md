-XG-NID: A Semantic-Enhanced Dual-Modality Intrusion Detection System
This project is a comprehensive implementation of the 

S-XG-NID architecture, a sophisticated intrusion detection system that not only detects cyberattacks but also understands and explains them. It leverages a dual-modality approach, combining high-accuracy statistical analysis with context-aware graph-based machine learning to identify known and novel threats. 



✨ Key Features
Dual-Modality Detection: Utilizes two powerful models for robust detection:


XGBoost Classifier: A high-performance model trained on over 80 statistical features of network flows for near-perfect accuracy on known attack patterns. 


Heterogeneous Graph Neural Network (HGNN): An advanced graph-based model that learns from the structural relationships between network entities (IPs, flows), giving it the potential to generalize to new, unseen threats. 

Live Traffic Analysis: A real-time engine built with Scapy that captures live network traffic, tracks network flows, and performs intrusion detection on the fly.


Explainable AI (XAI): Integrates a GPT-2 Large Language Model to provide clear, human-readable explanations for why a particular network activity was flagged as malicious. 



Graphical User Interface (GUI): A user-friendly desktop application built with CustomTkinter that provides a command center for starting/stopping the analysis and viewing real-time logs and alerts.

🛠️ Technology Stack
Backend & Machine Learning: Python, Pandas, NumPy, Scikit-learn

Statistical Model: XGBoost

Graph Model: PyTorch, PyTorch Geometric

Live Packet Capture: Scapy

Explanation Generation: Hugging Face Transformers (GPT-2)

GUI: CustomTkinter

Data Visualization: Matplotlib, Seaborn

🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

1. Prerequisites
Python 3.9+

The CIC-IDS-2017 dataset (both CSV and .pcap files) placed in the project directory.

2. Setup and Installation
First, clone the repository and create a Python virtual environment.

Bash

# Clone the repository
# cd S-XG-NID

# Create and activate a virtual environment
python -m venv venv
# On Windows:
# .\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
Next, install all the required dependencies from the requirements.txt file.

Bash

pip install -r requirements.txt
(You will need to create a requirements.txt file with the following content:)

Plaintext

# requirements.txt
pandas
numpy
scikit-learn
xgboost
torch
torchvision
torch-geometric
scapy
sentence-transformers
transformers
customtkinter
matplotlib
seaborn
tqdm
3. Usage
The project has two main operational modes: training the models and running the live detection application.

Training the Models
To process the data and train both the XGBoost and HGNN models from scratch, run the automated pipeline script:

Bash

python run_training_pipeline.py
This will execute all the necessary steps and save the trained model files.

Running the Live Detection Application
To launch the GUI for real-time network monitoring, run the following command with administrator/sudo privileges:

Bash

# On Windows (in an Administrator terminal):
python app_gui.py

# On macOS/Linux:
sudo python app_gui.py
Click "Start Capture" in the application to begin monitoring.

📁 File Descriptions
Here is a summary of each key script in the project:

phase1_data_preprocessing.py: Combines and cleans the raw CIC-IDS-2017 CSV files into a single master dataset (final_labeled_flows_cicids.csv).

phase2_feature_engineering.py: Prepares the clean data for ML by adding features, balancing classes, and converting data to a numerical format. Outputs model_ready_data.csv.

phase4_5_hgnn_implementation.py: Builds a heterogeneous graph of the network and trains the advanced HGNN model to learn from its structure.

phase5_model_training.py: Trains the statistical XGBoost model and generates evaluation artifacts like the confusion matrix and feature importance plot.

phase6_llm_explanation.py: Uses the trained XGBoost model and a GPT-2 LLM to generate human-readable explanations of detected attacks.

live_engine_final.py: The core backend engine that uses Scapy to capture live traffic, track flows, and make predictions.

app_gui.py: The main GUI application that provides a user-friendly interface for the live detection engine.

run_training_pipeline.py: An automation script that runs the entire data processing and model training pipeline in the correct sequence.

🔮 Future Work and Improvements
This project serves as a powerful foundation. The next steps to fully realize the original research vision include:

Full Live Feature Extractor: Expand the live_engine_final.py to calculate all 80+ features from the training dataset for maximum accuracy in live detection.


Deep Semantic Analysis: Implement phase2_advanced_semantics.py to analyze packet payloads with Sentence Transformers and integrate these "meaning vectors" into the ML models. 



Formal Ontology: Replace the NetworkX visualization with a formal reasoning engine using RDFLib to infer suspicious patterns based on a cyber threat ontology. 


Standalone Executable: Package the final application into a single executable file using PyInstaller for easy distribution.