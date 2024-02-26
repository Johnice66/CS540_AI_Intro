import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram



def load_data(filepath):
    data = []
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data



def calc_features(row):
    
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])
    
    features = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)
    
    return features



def hac(features):
    n = len(features)
    Z = np.zeros((n-1, 4))
    
    # Compute the distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    
    # Initialize clusters
    clusters = {i: [i] for i in range(n)}
    
    # Main loop
    for i in range(n-1):
        # Find the two closest clusters
        min_dist = np.inf
        min_pair = (0, 0)
        for j in range(len(clusters)):
            for k in range(j+1, len(clusters)):
                c1, c2 = list(clusters.keys())[j], list(clusters.keys())[k]
                dist = max(distance_matrix[a, b] for a in clusters[c1] for b in clusters[c2])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (c1, c2)
        
        # Merge the two closest clusters
        new_cluster_idx = n + i
        clusters[new_cluster_idx] = clusters[min_pair[0]] + clusters[min_pair[1]]
        
        # Update Z
        Z[i, 0] = min_pair[0]
        Z[i, 1] = min_pair[1]
        Z[i, 2] = min_dist
        Z[i, 3] = len(clusters[new_cluster_idx])
        
        # Remove the merged clusters from the dictionary
        del clusters[min_pair[0]], clusters[min_pair[1]]
    
    return Z


def fig_hac(Z, names):

    fig = plt.figure()
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    plt.show()
    return fig



def normalize_features(features):

    features_matrix = np.array(features)
    means = np.mean(features_matrix, axis=0)
    stds = np.std(features_matrix, axis=0)
    normalized_features_matrix = (features_matrix - means) / stds
    normalized_features = [normalized_features_matrix[i, :] for i in range(normalized_features_matrix.shape[0])]
    
    return normalized_features



    