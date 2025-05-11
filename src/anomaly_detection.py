import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from feature_extraction import extract_node_features
import os
import time

def load_network_data(file_path):
    # Load network data and build the graph
    print(f"Loading data from {file_path}...")
    edges = pd.read_csv(file_path, sep=' ', header=None, names=['source', 'target'])
    print(f"The dataset contains {len(edges)} edges")
    
    # Build an undirected graph from the edge list
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    print(f"Created a network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def detect_anomalies(feature_df, contamination=0.05):
    # Detect anomalies using Isolation Forest
    print(f"Detecting anomalies using Isolation Forest (contamination={contamination})...")
    
    # Select feature columns (exclude 'node')
    feature_columns = [col for col in feature_df.columns if col != 'node']
    features = feature_df[feature_columns].values
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform anomaly detection
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = clf.fit_predict(scaled_features)
    
    # Add anomaly labels to DataFrame
    result_df = feature_df.copy()
    result_df['anomaly'] = [1 if x == -1 else 0 for x in labels]  # 1 = anomaly, 0 = normal
    
    # Show anomaly statistics
    anomaly_count = result_df['anomaly'].sum()
    total_count = len(result_df)
    print(f"Detected {anomaly_count} anomalies, which is {anomaly_count/total_count:.2%} of all nodes")
    
    return result_df

def visualize_anomalies(anomaly_results, G, output_dir):
    # Visualize anomaly detection results
    print("Visualizing anomaly results...")
    
    # Boxplot comparison of features between anomaly and normal nodes
    plt.figure(figsize=(15, 10))
    features_to_plot = ['degree_centrality', 'clustering', 'pagerank', 'betweenness_centrality']

    for i, feature in enumerate(features_to_plot):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='anomaly', y=feature, data=anomaly_results)
        plt.title(f'{feature} by Anomaly Status')
        plt.xlabel('Is Anomaly')
        plt.ylabel(feature)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_feature_comparison.png'), dpi=300)
    
    # If the network is large, visualize only a sample
    if G.number_of_nodes() > 100:
        print("  Network too large, visualizing a sample...")
        # Extract subgraph with anomaly nodes and their neighbors
        anomaly_nodes = set(anomaly_results[anomaly_results['anomaly'] == 1]['node'])
        neighborhood = set()
        for node in anomaly_nodes:
            neighborhood.update(G.neighbors(node))
        sample_nodes = list(anomaly_nodes.union(neighborhood))[:100]
        subgraph = G.subgraph(sample_nodes)
    else:
        subgraph = G
        sample_nodes = list(G.nodes())
    
    # Get anomaly labels for sampled nodes
    sub_node_anomalies = anomaly_results[anomaly_results['node'].isin(sample_nodes)]
    node_color_map = {row['node']: 'red' if row['anomaly'] == 1 else 'skyblue' 
                      for _, row in sub_node_anomalies.iterrows()}
    node_colors = [node_color_map.get(node, 'skyblue') for node in subgraph.nodes()]
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, with_labels=True, node_color=node_colors, 
            node_size=300, edge_color='gray', font_size=8)
    plt.title("Network with Anomalies Highlighted")
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', 
                             markersize=10, label='Normal'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                             markersize=10, label='Anomaly')]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_network_visualization.png'), dpi=300)

def main():
    # Main function to execute the full anomaly detection pipeline
    start_time = time.time()
    
    # Create output directory
    output_dir = "../results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load network data
    data_path = "../data/facebook_combined.txt"
    G = load_network_data(data_path)
    
    # Extract node features
    node_features = extract_node_features(G)
    
    # Save node features to CSV
    node_features.to_csv(os.path.join(output_dir, "node_features.csv"), index=False)
    
    # Visualize features
    visualize_features(node_features, output_dir)
    
    # Detect anomalies
    anomaly_results = detect_anomalies(node_features, contamination=0.05)
    
    # Save anomaly results to CSV
    anomaly_results.to_csv(os.path.join(output_dir, "anomaly_detection_results.csv"), index=False)
    
    # Save list of anomaly nodes to TXT
    anomaly_nodes = anomaly_results[anomaly_results['anomaly'] == 1]['node'].tolist()
    with open(os.path.join(output_dir, "anomaly_nodes.txt"), 'w') as f:
        f.write('\n'.join(map(str, anomaly_nodes)))
    print(f"Anomaly node list saved to {os.path.join(output_dir, 'anomaly_nodes.txt')}")
    
    # Visualize anomalies on network
    visualize_anomalies(anomaly_results, G, output_dir)
    
    elapsed_time = time.time() - start_time
    print(f"Analysis complete! Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
