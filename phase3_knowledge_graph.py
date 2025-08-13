# phase3_knowledge_graph.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- Configuration ---
ENGINEERED_DATA_PATH = 'model_ready_data.csv'
# Let's work with a smaller sample for speed during development
SAMPLE_SIZE = 10000 

def build_knowledge_graph(df):
    """
    Builds a NetworkX graph from the provided DataFrame.
    """
    print(f"Building knowledge graph from {len(df)} samples...")
    
    # Create a new directed graph
    G = nx.DiGraph()
    
    # To make the labels readable again, we need to reverse the LabelEncoding from Phase 2.
    # We will assume a simple mapping for now. A more robust solution would save the encoder.
    # Based on the previous output: 0=BENIGN, 1=Bot, 2=DDoS, 3=DoS_GoldenEye, etc.
    # Let's create a placeholder mapping for major categories.
    label_map = {0: 'BENIGN', 2: 'DDoS', 10: 'PortScan', 4: 'DoS_Hulk'}
    
    # Iterate over each row in the DataFrame to build the graph
    for _, row in df.iterrows():
        # Get source and destination IPs from the one-hot encoded columns
        src_ip_col = [col for col in df.columns if col.startswith('Protocol_Name_') and row[col]==1]
        service_col = [col for col in df.columns if col.startswith('Service_') and row[col]==1]
        
        if not src_ip_col or not service_col:
            continue
            
        # Extract the actual names from the column headers
        protocol = src_ip_col[0].replace('Protocol_Name_', '')
        service = service_col[0].replace('Service_', '')
        
        # Determine the attack type for this connection
        attack_label = label_map.get(row['Label'], 'ATTACK') # Default to 'ATTACK' if not in our map

        # Add nodes to the graph. Nodes will be protocols and services.
        # We add attributes to the nodes to describe what they are.
        G.add_node(protocol, type='Protocol')
        G.add_node(service, type='Service')
        
        # Add an edge between the protocol and the service
        # The edge will be labeled with the type of attack
        G.add_edge(protocol, service, label=attack_label, weight=row['Flow_Duration'])
        
    print("Graph construction complete.")
    print(f" - Nodes: {G.number_of_nodes()}")
    print(f" - Edges: {G.number_of_edges()}")
    
    return G

def visualize_graph(G):
    """
    Creates and saves a visualization of the graph.
    """
    print("\nVisualizing graph (this might take a moment)...")
    plt.figure(figsize=(12, 12))
    
    # Use a layout algorithm to position nodes
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title("Network Behavior Knowledge Graph")
    plt.savefig("knowledge_graph.png")
    print("Graph visualization saved to 'knowledge_graph.png'")


if __name__ == "__main__":
    try:
        df = pd.read_csv(ENGINEERED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{ENGINEERED_DATA_PATH}' was not found. Please run the Phase 2 script first.")
    else:
        # Create a smaller sample of the data for quick processing
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
        
        # Build the knowledge graph
        knowledge_graph = build_knowledge_graph(df_sample)
        
        # Visualize the graph
        visualize_graph(knowledge_graph)