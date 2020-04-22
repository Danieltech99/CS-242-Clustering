# 
# Non federated algorithms
#
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

class DataSetCollection:
    data = None
    labeled_data = None
    noice_levels = ["vhigh", "high", "med", "low", "vlow"]
    data_sets_names = ["circles", "moons", "blobs", "longblobs"]

    def __init__(self):
        self.data = np.load("Data/sklearn_data.npz") # update for path
        self.labeled_data = np.load("Data/sklearn_labels.npz") # update for path
    
    def construct_key(self, name, noice_level):
        assert(name in self.data_sets_names)
        assert(noice_level in self.noice_levels)
        return name + "_" + noice_level + "_" + "noise"
    
    def get_set(self, name, noice_level):
        return self.get_data_set(name, noice_level), self.get_label_set(name, noice_level)

    def get_data_set(self, name, noice_level):
        return self.data[self.construct_key(name,noice_level)]
    
    def get_label_set(self, name, noice_level):
        return self.labeled_data[self.construct_key(name,noice_level)]

# Evaluation metrics
from sklearn import metrics

class Metrics:
    # Params: label_true, label_pred
    def __init__(self, params):
        self._true = params['true']
        self._pred = params['pred']
    
    # Adjusted Rand Index
    # Pros: random labels have ARI close to 0, bounded [-1, 1], no assumption on cluster structure
    # Cons: requires ground truth
    def ari(self):
        return metrics.adjusted_rand_score(self._true, self._pred)
    
    # Mutual information based scores
    # Pros: random labels have MIs close to 0, [0, 1]
    # Cons: requires ground truth
    def mi(self):
        return metrics.adjusted_mutual_info_score(self._true, self._pred)

    # Homogeneity (each cluster contains only members of a given class)
    # Completeness (all members of a given class are assigned to the same cluster)
    # V-measure (the harmonic mean of homogeneity and completeness)
    # Pros: [0, 1] with 1 as perfect, intuitive qualitative interp, no assumption on cluster structure
    # Cons: not normalized to random labeling, if samples < 1000 or clusters > 10, use ARI isntead
    def homogenity(self):
        return metrics.homogeneity_score(self._true, self._pred)
    
    def completeness(self):
        return metrics.completeness_score(self._true, self._pred)
    
    def v_measure(self):
        return metrics.v_measure_score(self._true, self._pred)


class NonFedAlgs:
    # returns the labels after clustering
    def __init__(self, params):
        self._num_clusters = params['num_clusters']
        self._data = params['data']
        self._seed = 0

        if 'seed' in params:
            self._seed = params['seed']
        
        # linkage for agglomerative clustering
        if 'linkage' in params:
            self._linkage = params['linkage']
    
    # Spectral clustering, a non-linear alternative to kmeans
    def spectral_clustering(self):
        clustering = SpectralClustering(n_clusters=self._num_clusters,
            random_state=self._seed).fit(self._data)
        return clustering.labels_

    # KMeans
    def kmeans(self):
        clustering = KMeans(n_clusters=self._num_clusters,
            random_state=self._seed).fit(self._data)
        return clustering.labels_
    
    # Hierarchical/Agglomerative Clustering
    


# Params
num_clusters = 2
num_data_points = 1000

# Load data
collection = DataSetCollection()
data, labels = collection.get_set("blobs", "vlow")
print(data)

clustering = KMeans(n_clusters=num_clusters,
       random_state=0).fit(data[:num_data_points])

metrics_params = {
    'true': labels[:num_data_points],
    'pred': clustering.labels_
}
m = Metrics(metrics_params)
print(m.ari())

