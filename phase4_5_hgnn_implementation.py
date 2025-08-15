# phase4_5_hgnn_implementation.py (Final Corrected Version)
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# --- Configuration ---
ENGINEERED_DATA_PATH = 'model_ready_data.csv'
SAMPLE_SIZE = 20000

# --- 1. Data Loading and Preprocessing ---
def load_and_prepare_data(path, sample_size):
    print("Loading and preparing data...")
    df_orig = pd.read_csv('final_labeled_flows_cicids.csv', low_memory=False)
    df_orig.columns = df_orig.columns.str.strip().str.replace(' ', '_')
    df_subset = df_orig[['Source_IP', 'Destination_IP', 'Label']].copy()
    df_subset['Label'] = df_subset['Label'].str.replace(' ', '_').astype('category')
    df_sample = df_subset.sample(n=sample_size, random_state=42)
    
    all_ips = pd.concat([df_sample['Source_IP'], df_sample['Destination_IP']]).astype('category')
    ip_encoder = {ip: i for i, ip in enumerate(all_ips.cat.categories)}
    
    df_sample['src_ip_idx'] = df_sample['Source_IP'].apply(lambda x: ip_encoder[x])
    df_sample['dst_ip_idx'] = df_sample['Destination_IP'].apply(lambda x: ip_encoder[x])
    
    label_encoder = {label: i for i, label in enumerate(df_sample['Label'].cat.categories)}
    df_sample['Label_encoded'] = df_sample['Label'].cat.codes
    
    print(f" - Data sample created with {len(df_sample)} flows.")
    print(f" - Found {len(ip_encoder)} unique IPs.")
    return df_sample, ip_encoder, label_encoder

# --- 2. Heterogeneous Graph Construction (Phase 4) ---
def build_heterogeneous_graph(df, ip_encoder):
    print("\nBuilding heterogeneous graph...")
    data = HeteroData()
    
    num_ips = len(ip_encoder)
    data['ip'].x = torch.eye(num_ips)
    
    data['flow'].y = torch.tensor(df['Label_encoded'].values, dtype=torch.long)
    
    # --- THIS IS THE FIX ---
    # We explicitly tell the graph how many 'flow' nodes there are.
    data['flow'].num_nodes = len(df)
    # --- END OF FIX ---
    
    src_ip_to_flow_edge = torch.tensor([df['src_ip_idx'].values, np.arange(len(df))], dtype=torch.long)
    flow_to_dest_ip_edge = torch.tensor([np.arange(len(df)), df['dst_ip_idx'].values], dtype=torch.long)
    
    data['ip', 'sends', 'flow'].edge_index = src_ip_to_flow_edge
    data['flow', 'is_sent_to', 'ip'].edge_index = flow_to_dest_ip_edge
    
    print(" - Graph construction complete.")
    print(data)
    return data

# --- 3. A SIMPLE GNN Model Definition (Phase 5) ---
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv((in_channels, in_channels), hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv((hidden_channels, hidden_channels), hidden_channels))
        
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.lin(x)

# --- 4. Training and Evaluation ---
def train_and_evaluate(model, data, df, label_encoder):
    print("\nStarting model training...")
    num_flow_nodes = len(df)
    train_mask = torch.zeros(num_flow_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_flow_nodes, dtype=torch.bool)
    
    indices = np.arange(num_flow_nodes)
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42, stratify=df['Label_encoded'])
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    data['flow'].train_mask = train_mask
    data['flow'].test_mask = test_mask

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    data = data.to(device)

    for epoch in tqdm(range(51), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        out_dict = model(data.x_dict, data.edge_index_dict)
        out = out_dict['flow']
        
        mask = data['flow'].train_mask
        loss = F.cross_entropy(out[mask], data['flow'].y[mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            tqdm.write(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

    model.eval()
    pred_dict = model(data.x_dict, data.edge_index_dict)
    pred = pred_dict['flow'].argmax(dim=-1)
    
    mask = data['flow'].test_mask
    correct = (pred[mask] == data['flow'].y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    print(f'\n--- Evaluation Complete ---')
    print(f'Test Accuracy: {acc:.4f}')

if __name__ == "__main__":
    df_prepared, ip_encoder, label_encoder = load_and_prepare_data(ENGINEERED_DATA_PATH, SAMPLE_SIZE)
    graph_data = build_heterogeneous_graph(df_prepared, ip_encoder)
    
    graph_data['flow'].x = torch.zeros(graph_data['flow'].num_nodes, graph_data['ip'].num_features)

    num_classes = len(label_encoder)
    simple_gnn = GNN(in_channels=graph_data['ip'].num_features, hidden_channels=64, out_channels=num_classes)

    hgnn_model = to_hetero(simple_gnn, graph_data.metadata(), aggr='sum')

    train_and_evaluate(hgnn_model, graph_data, df_prepared, label_encoder)