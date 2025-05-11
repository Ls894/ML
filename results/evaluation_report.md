
# Network Anomaly Detection Evaluation Report

## Basic Statistics

- Total nodes: 4039
- Anomalous nodes: 202 (5.00%)

## Feature Analysis

Comparison of features between anomalous and normal nodes:

               feature  anomaly_mean  normal_mean      ratio
     degree_centrality      0.021480     0.010259   2.093857
            clustering      0.224296     0.625618   0.358520
              pagerank      0.000515     0.000234   2.205760
betweenness_centrality      0.012325     0.000031 397.620947
   local_heterogeneity      0.796349     1.496063   0.532296

## Network Structure Analysis

- Density among anomaly nodes: 0.044825
- Overall network density: 0.010820
- Density ratio: 4.1428400424775615

## Conclusion

Based on the analysis, we found:

- The betweenness_centrality of anomalous nodes is significantly **higher** than normal nodes (397.62¡Á)
- The pagerank of anomalous nodes is significantly **higher** than normal nodes (2.21¡Á)
- Anomalous nodes tend to cluster together, suggesting potential formation of specialized communities.

