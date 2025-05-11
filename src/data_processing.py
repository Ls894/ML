import networkx as nx
import pandas as pd
import os
import gzip
import shutil
import urllib.request

def download_facebook_data(output_dir="../data"):
    # Download the Facebook network dataset
    os.makedirs(output_dir, exist_ok=True)
    
    url = 'https://snap.stanford.edu/data/facebook_combined.txt.gz'
    output_path = os.path.join(output_dir, 'facebook_combined.txt.gz')
    
    urllib.request.urlretrieve(url, output_path)
    print(f"Dataset saved to {output_path}")
    
    # Unzip the file
    extract_path = os.path.join(output_dir, 'facebook_combined.txt')
    with gzip.open(output_path, 'rb') as f_in:
        with open(extract_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"Dataset extracted to {extract_path}")
    return extract_path

def load_network_data(file_path):
    # Load network data from file and build the graph
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    edges = pd.read_csv(file_path, sep=' ', header=None, names=['source', 'target'])
    print(f"Dataset contains {len(edges)} edges")
    
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    print(f"Created a network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def get_network_stats(G):
    # Get basic network statistics
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'diameter': nx.diameter(G) if nx.is_connected(G) else "Graph is not connected",
        'connected_components': nx.number_connected_components(G)
    }
    return stats

def load_or_download_data(data_path="../data/facebook_combined.txt"):
    # Load data from file, or download it if it doesn't exist
    if not os.path.exists(data_path):
        data_path = download_facebook_data()
    
    return load_network_data(data_path)
