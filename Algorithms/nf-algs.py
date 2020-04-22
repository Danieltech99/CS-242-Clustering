# 
# Non federated algorithms
#
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom  

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
        self._input_len = len(self._data[0])
        self._labels = params['labels']

        if 'seed' in params:
            self._seed = params['seed']
        
        # linkage for agglomerative clustering
        if 'linkage' in params:
            self._linkage = params['linkage']
        
        # variables for SOM
        if 'SOM_dim' in params:
            self._som_dim = params['SOM_dim']
        if 'sigma' in params:
            self._sigma = params['sigma']
        if 'lr' in params:
            self._lr = params['lr']
        if 'som_iters' in params:
            self._som_iters = params['som_iters']
    
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
    def hierarchical(self):
        clustering = AgglomerativeClustering(n_clusters = self._num_clusters, linkage = self._linkage).fit(self._data)
        return clustering.labels_
    
    # Self Organizing Maps
    def SOM(self):
        som = MiniSom(self._som_dim, self._som_dim, self._input_len, sigma=self._sigma, learning_rate=self._lr) # initialization of 6x6 SOM
        som.train_random(self._data, self._som_iters) # trains the SOM with 100 iterations
        class_assignments = som.labels_map(self._data, self._labels)
        classification = self._classify(som, self._data, class_assignments)
        return classification
    
    def _classify(self, som, data, class_assignments):
        """Classifies each sample in data in one of the classes definited
        using the method labels_map.
        Returns a list of the same length of data where the i-th element
        is the class assigned to data[i].
        """
        winmap = class_assignments
        default_class = np.sum(list(winmap.values())).most_common()[0][0]
        result = []
        for d in data:
            win_position = som.winner(d)
            if win_position in winmap:
                result.append(winmap[win_position].most_common()[0][0])
            else:
                result.append(default_class)
        return result

# # Params
# num_clusters = 3
# num_data_points = 1000
# seed = 0
# linkage = 'ward'
# SOM_dim = 3
# sigma = 0.3
# lr = 0.5
# som_iters = 100

# # Load data
# collection = DataSetCollection()
# data, labels = collection.get_set("blobs", "vlow")

# params = {
#     'num_clusters': num_clusters,
#     'data': data,
#     'seed': seed,
#     'labels': labels,
#     'linkage': linkage,
#     'SOM_dim': SOM_dim,
#     'sigma': sigma,
#     'lr': lr,
#     'som_iters': som_iters
# }

# algs = NonFedAlgs(params)
# clustering_labels = algs.SOM()

# metrics_params = {
#     'true': labels,
#     'pred': clustering_labels
# }
# m = Metrics(metrics_params)
# print(m.ari())

