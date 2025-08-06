import pandas as pd
import requests
import os

# Define the URL for one of the UNSW-NB15 dataset files
# This is a publicly available source for a subset of the data.
dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/2D2g7nBv68yBqgq/download?path=/UNSW-NB15%20v2&files=UNSW_NB15_1.csv"
dataset_filename = "UNSW_NB15_1.csv"

# Step 1: Download the dataset file
if not os.path.exists(dataset_filename):
    print(f"Downloading {dataset_filename}...")
    try:
        response = requests.get(dataset_url)
        response.raise_for_status()  # This will raise an exception for bad responses (4xx or 5xx)
        with open(dataset_filename, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        exit()
else:
    print(f"{dataset_filename} already exists. Skipping download.")

# Step 2: Load the dataset
print("\nLoading the UNSW-NB15 dataset...")
# The dataset has no header, so we provide one based on the official documentation.
column_names = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
    'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload',
    'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_srv_dst', 'is_sm_ips_ports',
    'attack_cat', 'label'
]
df = pd.read_csv(dataset_filename, header=None, names=column_names, encoding='latin1')
print("Dataset loaded successfully.")
print("\nOriginal Data Info:")
df.info()

# Step 3: Implement basic preprocessing
print("\nStarting data preprocessing...")
# Drop columns that are less useful for semantic tagging in this stage, like `ct_ftp_cmd` which has a lot of NaNs.
df = df.drop(columns=['ct_ftp_cmd'])
# Fill NaN values with a placeholder string 'unknown'
df['service'] = df['service'].fillna('unknown')
# Convert the 'label' column to human-readable strings
df['label'] = df['label'].replace({0: 'Normal', 1: 'Attack'})

# Step 4: Implement a semantic tagging function based on the UNSW-NB15 features
def semantic_tagging(row):
    """
    A more realistic function to add semantic tags based on the dataset's features.
    """
    device_type = 'unknown'
    function = 'general_traffic'

    # Rule-based tagging based on 'service' and 'proto'
    if row['service'] == 'dns':
        function = 'domain_resolution'
        device_type = 'server'
    elif row['service'] == 'http':
        function = 'web_Browse'
        device_type = 'user_pc'
    elif row['proto'] == 'tcp' and row['dsport'] == 22: # SSH
        function = 'remote_access'
        device_type = 'server'
    elif row['proto'] == 'udp' and row['dsport'] == 53: # DNS over UDP
        function = 'domain_resolution'
        device_type = 'server'
    elif 'ftp' in row['service']:
        function = 'file_transfer'
        device_type = 'server'
    else:
        # Default for other services or protocols
        device_type = 'user_pc'
        function = 'general_traffic'

    # Use a more detailed tag for attacks
    if row['label'] == 'Attack':
        function = row['attack_cat']
        device_type = 'attacker_pc' # A simple assumption for now

    return pd.Series([device_type, function], index=['device_type', 'function'])

# Apply the tagging function to the DataFrame
print("\nApplying semantic tagging...")
df[['device_type', 'function']] = df.apply(semantic_tagging, axis=1)

print("Semantic tagging complete.")
print("\nPreprocessed Data with Semantic Tags:")
print(df[['srcip', 'dstip', 'proto', 'service', 'attack_cat', 'device_type', 'function']].head())

# Save the preprocessed data for the next step
output_filename = 'preprocessed_UNSW_NB15.csv'
df.to_csv(output_filename, index=False)
print(f"\nPreprocessed data saved to '{output_filename}'.")