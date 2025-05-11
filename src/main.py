"""
Main program for Network Anomaly Detection Project
"""
import os
import time
from data_processing import load_or_download_data, get_network_stats
from feature_extraction import extract_node_features, feature_analysis, normalize_features
from anomaly_detection import detect_anomalies, visualize_anomalies
from evaluation import evaluate_anomalies, generate_evaluation_report

def main():
    # Execute the full pipeline for network anomaly detection
    start_time = time.time()
    output_dir = "../results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Data loading
    print("\n== Load Data ==")
    G = load_or_download_data("../data/facebook_combined.txt")
    
    # Get basic network statistics
    network_stats = get_network_stats(G)
    for key, value in network_stats.items():
        print(f"- {key}: {value}")
    
    # Feature extraction
    print("\n== Feature Extraction ==")
    node_features = extract_node_features(G)
    
    # Analyze features
    stats, correlation = feature_analysis(node_features)
    print("\nFeature Summary Statistics:")
    print(stats)
    
    # Save feature data
    features_path = os.path.join(output_dir, "node_features.csv")
    node_features.to_csv(features_path, index=False)
    
    # Normalize features
    normalized_features = normalize_features(node_features)
    
    # Anomaly detection
    print("\n== Anomaly Detection ==")
    # Use normalized features for anomaly detection
    anomaly_results = detect_anomalies(normalized_features, contamination=0.05)
    
    # Save anomaly detection results
    results_path = os.path.join(output_dir, "anomaly_detection_results.csv")
    anomaly_results.to_csv(results_path, index=False)
    print(f"Anomaly detection results saved to {results_path}")
    
    # Visualize anomalies
    visualize_anomalies(anomaly_results, G, output_dir)
    
    # Evaluation
    print("\n== Evaluation ==")
    anomaly_results_full = node_features.copy()
    anomaly_results_full['anomaly'] = anomaly_results['anomaly'].values

    evaluation_results = evaluate_anomalies(anomaly_results_full, G, output_dir)
    
    # Generate evaluation report
    report = generate_evaluation_report(evaluation_results, output_dir)
    
    # Done
    elapsed_time = time.time() - start_time
    print(f"\n=== Analysis Complete! Total time: {elapsed_time:.2f} seconds ===")
    print(f"All results saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
