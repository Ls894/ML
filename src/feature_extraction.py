import networkx as nx
import pandas as pd
import numpy as np
import time

def extract_node_features(G):
    # Extract node features from the network"
    start_time = time.time()
    
    # Basic centrality measures
    degree_centrality = nx.degree_centrality(G)
    
    clustering = nx.clustering(G)
    
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # For large networks
    if G.number_of_nodes() < 1000:
        betweenness_centrality = nx.betweenness_centrality(G)
    else:
        # Use sampling to approximate betweenness centrality and reduce computation
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
    
    # Compute local heterogeneity
    local_heterogeneity = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            neighbor_degrees = [G.degree(n) for n in neighbors]
            local_heterogeneity[node] = np.std(neighbor_degrees) / np.mean(neighbor_degrees) if np.mean(neighbor_degrees) > 0 else 0
        else:
            local_heterogeneity[node] = 0
    
    # Combine features into a DataFrame
    nodes = list(G.nodes())
    feature_df = pd.DataFrame({
        'node': nodes,
        'degree_centrality': [degree_centrality[n] for n in nodes],
        'clustering': [clustering[n] for n in nodes],
        'pagerank': [pagerank[n] for n in nodes],
        'betweenness_centrality': [betweenness_centrality[n] for n in nodes],
        'local_heterogeneity': [local_heterogeneity[n] for n in nodes]
    })
    
    elapsed_time = time.time() - start_time
    print(f"Feature extraction complete. Time elapsed: {elapsed_time:.2f} seconds")
    
    return feature_df

def feature_analysis(feature_df):
    print("Analyzing features...")
    
    # Descriptive statistics
    stats = feature_df.describe().transpose()
    
    # Feature correlation matrix
    feature_columns = [col for col in feature_df.columns if col != 'node']
    correlation = feature_df[feature_columns].corr()
    
    return stats, correlation

def normalize_features(feature_df):
    # Normalize features using StandardScaler
    from sklearn.preprocessing import StandardScaler
    
    # Select feature columns (exclude 'node')
    feature_columns = [col for col in feature_df.columns if col != 'node']
    
    # Create new DataFrame to preserve node IDs
    normalized_df = feature_df[['node']].copy()
    
    # Apply standard normalization
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_df[feature_columns])
    
    # Add normalized features back to the DataFrame
    for i, col in enumerate(feature_columns):
        normalized_df[col] = normalized_features[:, i]
    
    return normalized_df
