import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

def evaluate_anomalies(anomaly_results, G, output_dir="../results"):

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Anomaly statistics
    total_nodes = len(anomaly_results)
    anomaly_nodes = anomaly_results[anomaly_results['anomaly'] == 1]
    normal_nodes = anomaly_results[anomaly_results['anomaly'] == 0]
    
    anomaly_count = len(anomaly_nodes)
    anomaly_percentage = anomaly_count / total_nodes * 100
    
    print(f"Total nodes: {total_nodes}")
    print(f"Anomalous nodes: {anomaly_count} ({anomaly_percentage:.2f}%)")
    
    # Feature comparison
    feature_columns = [col for col in anomaly_results.columns 
                      if col not in ['node', 'anomaly']]
    
    comparison = pd.DataFrame()
    for feature in feature_columns:
        anomaly_mean = anomaly_nodes[feature].mean()
        normal_mean = normal_nodes[feature].mean()
        anomaly_std = anomaly_nodes[feature].std()
        normal_std = normal_nodes[feature].std()
        
        comparison = pd.concat([comparison, pd.DataFrame({
            'feature': [feature],
            'anomaly_mean': [anomaly_mean],
            'normal_mean': [normal_mean],
            'ratio': [anomaly_mean / normal_mean if normal_mean != 0 else np.nan],
            'anomaly_std': [anomaly_std],
            'normal_std': [normal_std]
        })])
    
    # Save feature comparison
    comparison_path = os.path.join(output_dir, "feature_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"Feature comparison saved to {comparison_path}")
    
    # Network structure analysis
    anomaly_node_ids = set(anomaly_nodes['node'])
    anomaly_edges = 0
    
    for u, v in G.edges():
        if u in anomaly_node_ids and v in anomaly_node_ids:
            anomaly_edges += 1
    
    total_possible_anomaly_edges = anomaly_count * (anomaly_count - 1) / 2
    anomaly_density = anomaly_edges / total_possible_anomaly_edges if total_possible_anomaly_edges > 0 else 0
    network_density = nx.density(G)
    
    print(f"Anomaly node density: {anomaly_density:.4f}")
    print(f"Network density: {network_density:.4f}")
    print(f"Density ratio: {anomaly_density / network_density if network_density > 0 else 'N/A'}")
    
    plt.figure(figsize=(12, 6))

    # Use log10
    comparison['log_ratio'] = comparison['ratio'].apply(
        lambda x: np.log10(x) if x > 0 else -np.inf        # ratio>0 才能取对数
    )

    sorted_comparison = comparison.sort_values('log_ratio', ascending=False)

    sns.barplot(x='feature', y='log_ratio', data=sorted_comparison,
                palette='Blues_r')

    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)   # log10(1)=0 参考线
    plt.title('Feature Importance (log10 scale)')
    plt.xlabel('Feature')
    plt.ylabel('log10(Anomaly / Normal)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    
    return {
        'total_nodes': total_nodes,
        'anomaly_count': anomaly_count,
        'anomaly_percentage': anomaly_percentage,
        'feature_comparison': comparison,
        'anomaly_density': anomaly_density,
        'network_density': network_density
    }

def generate_evaluation_report(evaluation_results, output_dir="../results"):
    report = f"""
# Network Anomaly Detection Evaluation Report

## Basic Statistics

- Total nodes: {evaluation_results['total_nodes']}
- Anomalous nodes: {evaluation_results['anomaly_count']} ({evaluation_results['anomaly_percentage']:.2f}%)

## Feature Analysis

Comparison of features between anomalous and normal nodes:

{evaluation_results['feature_comparison'][['feature', 'anomaly_mean', 'normal_mean', 'ratio']].to_string(index=False)}

## Network Structure Analysis

- Density among anomaly nodes: {evaluation_results['anomaly_density']:.6f}
- Overall network density: {evaluation_results['network_density']:.6f}
- Density ratio: {evaluation_results['anomaly_density'] / evaluation_results['network_density'] if evaluation_results['network_density'] > 0 else 'N/A'}

## Conclusion

{generate_conclusion(evaluation_results)}
"""
    
    # Save report
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Evaluation report saved to {report_path}")
    return report

def generate_conclusion(evaluation_results):
    """Generate conclusion based on evaluation results"""
    comparison = evaluation_results['feature_comparison']
    
    # Identify the most discriminative features
    comparison['abs_ratio'] = abs(comparison['ratio'] - 1)
    top_features = comparison.sort_values('abs_ratio', ascending=False).head(2)
    
    conclusion = "Based on the analysis, we found:\n\n"
    
    for _, row in top_features.iterrows():
        feature = row['feature']
        ratio = row['ratio']
        if ratio > 1:
            conclusion += f"- The {feature} of anomalous nodes is significantly **higher** than normal nodes ({ratio:.2f}×)\n"
        else:
            conclusion += f"- The {feature} of anomalous nodes is significantly **lower** than normal nodes ({1/ratio:.2f}×)\n"
    
    if evaluation_results['anomaly_density'] > evaluation_results['network_density']:
        conclusion += "- Anomalous nodes tend to cluster together, suggesting potential formation of specialized communities.\n"
    else:
        conclusion += "- Anomalous nodes are more dispersed, which may indicate special or bridging roles in the network.\n"
    
    return conclusion
