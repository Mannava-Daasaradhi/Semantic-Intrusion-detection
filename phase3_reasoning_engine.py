# phase3_reasoning_engine.py
import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
import warnings

# Suppress a specific pandas warning
warnings.filterwarnings("ignore", "This pattern is interpreted as a regular expression, which is different from 'str.contains'.")

# --- Configuration ---
SEMANTIC_DATA_PATH = 'semantic_features_dataset.csv'
# For demonstration, we'll process a smaller sample of the data.
SAMPLE_SIZE = 5000 

# --- 1. Define our simple Ontology (Vocabulary) ---
# Create a namespace for our project's custom terms
NIDS = Namespace("http://www.example.org/nids#")

def build_knowledge_graph(df):
    """
    Converts a DataFrame of packet data into an RDF knowledge graph.
    """
    print("Building knowledge graph from semantic data...")
    g = Graph()
    # Bind our custom namespace to a prefix for cleaner output
    g.bind("nids", NIDS)

    # --- 2. Convert Data to RDF Triples ---
    for idx, row in df.iterrows():
        # Create a unique identifier for each packet event
        packet_event = URIRef(f"http://www.example.org/packet/{idx}")
        
        # Add the basic type for this event
        g.add((packet_event, RDF.type, NIDS.PacketEvent))
        
        # Add properties (Predicate + Object) to the packet event (Subject)
        g.add((packet_event, NIDS.hasTimestamp, Literal(row['timestamp'])))
        
        # For simplicity, we create generic subjects for payload and IP
        payload_subject = URIRef(f"http://www.example.org/payload/{idx}")
        
        g.add((packet_event, NIDS.hasPayload, payload_subject))
        g.add((payload_subject, NIDS.containsText, Literal(row['payload_text'])))

    print(f" - Knowledge graph built with {len(g)} triples.")
    return g

def run_reasoning_engine(g):
    """
    Defines and executes a SPARQL query to find suspicious patterns.
    """
    print("\nRunning reasoning engine to find suspicious patterns...")
    
    # --- 3. Define a Reasoning Rule with SPARQL ---
    # This rule looks for any packet event where the payload contains the word "password".
    # This is a simple but powerful example of semantic reasoning.
    query = """
    PREFIX nids: <http://www.example.org/nids#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?packet ?payloadText
    WHERE {
        ?packet rdf:type nids:PacketEvent .
        ?packet nids:hasPayload ?payload .
        ?payload nids:containsText ?payloadText .
        
        FILTER(CONTAINS(LCASE(?payloadText), "password"))
    }
    """
    
    # Execute the query on our graph
    results = g.query(query)
    
    # --- 4. Report Findings ---
    if not results:
        print(" - No suspicious patterns found based on the current rules.")
    else:
        print(f"  [!] ALERT: Found {len(results)} suspicious packet(s) containing 'password'.")
        for row in results:
            print(f"  - Suspicious Packet Event: {row.packet}")
            print(f"    Payload Text: '{str(row.payloadText).strip()[:100]}...'") # Print first 100 chars

if __name__ == "__main__":
    try:
        df_full = pd.read_csv(SEMANTIC_DATA_PATH)
        # Create a smaller sample for faster processing
        df_sample = df_full.sample(n=min(SAMPLE_SIZE, len(df_full)), random_state=42)
    except FileNotFoundError:
        print(f"Error: The semantic data file '{SEMANTIC_DATA_PATH}' was not found.")
        print("Please run 'phase2_advanced_semantics.py' first.")
    else:
        knowledge_graph = build_knowledge_graph(df_sample)
        run_reasoning_engine(knowledge_graph)