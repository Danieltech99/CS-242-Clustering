import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin, pairwise_distances_argmin_min
from sklearn.cluster import KMeans


params = {"N_CLUSTERS": 10,
          "MAX_ITERS": 100,
          "N_INITS": 10,
          "N_DEVICES": 10,
          "CACHE_SIZE": 4,
          "GLOBAL_INIT": None,
          "METRIC": "euclidean",
          "TOLERANCE": None}


class gossip_K_Means_Device:
    def __init__(self, data, params, id_num=None):
        """Initialize k-means member variables"""
        self._n_clusters = params["N_CLUSTERS"]
        self._max_iters = params["MAX_ITERS"]
        self._n_inits = params["N_INITS"]
        self._metric = params["METRIC"]
        self._n_devices = params["N_DEVICES"]
        self._cache_size = params["CACHE_SIZE"]
        self._global_init = params["GLOBAL_INIT"]
        self._id_num = id_num
        
        self._data = data
        self._n_dims = data.shape[1]

        if params["TOLERANCE"] is None:
            self._tolerance = np.min(np.std(data, axis=0))/100
        else:
            self._tolerance = params["TOLERANCE"]

        self.state = "ACTIVE"

        if id_num == 0:
            self.cache = [1, self._n_devices]
        elif id_num == self._n_devices - 1:
            self.cache = [0, self._n_devices]
        else:
            self.cache = [self._id_num - 1, self._id_num + 1]

        self.error = float("Inf")
        self.local_error = None

        if self._global_init is None:
            self.centers = self._init_cluster_centers(self._data)
        else:
            self.centers = self._global_init
        
        self.local_centers = None

        self.labels = None
        self.local_labels = None
        
        self.local_counts = None

        self._large_number = 999
        
        
    def run_on_device(self):
        """Trains the k-means"""
        # Sets data indices for training data input

        data = self._data
        centers = self.centers
        n_centers = self._n_clusters

        labels, local_error = self._compute_labels_inertia(data, centers)
        local_centers, local_counts = self._compute_centers_counts(data, labels, n_centers)

        self.labels = labels
        self.local_error = local_error
        self.local_centers = local_centers
        self.local_counts = local_counts

        
    def _init_cluster_centers(self, data, metric=None):
        n_clusters = self._n_clusters
        if metric == None:
            metric = "euclidean"

        centers = np.zeros((n_clusters, self._n_dims))
        centers[0] = data[np.random.choice(data.shape[0])]
        for i in range(1, n_clusters):
            all_centers = centers[0:i, :]
            distance_pairs = pairwise_distances(data, all_centers)
            sum_distances = np.sum(distance_pairs, axis=1)
            datapoint_probs = sum_distances/np.sum(sum_distances)
            
            next_center_idx = np.random.choice(data.shape[0], p=datapoint_probs)
            centers[i] = data[next_center_idx]

        return centers


    # def _fit(self, data, initial_centers):
    #     centers = initial_centers
    #     for i in range(self._max_iters):
    #         n_centers = centers.shape[0]
    #         centers_old = centers.copy()

    #         labels, inertia = self._compute_labels_inertia(data, centers_old)
    #         centers = self._compute_cluster_centers(data, labels, n_centers)

    #         if self._best_inertia == None or inertia < self._best_inertia:
    #             self._best_inertia = inertia.copy()
    #             self._best_labels = labels.copy()
    #             self._best_centers = centers_old.copy()
                
    #         if centers_old.shape == centers.shape and np.allclose(centers_old, centers, atol=self._tolerance):
    #             break


    def _compute_labels_inertia(self, data, centers, metric=None):
        if metric == None:
            metric = "euclidean"

        # print("centers", centers)
        labels, distances = pairwise_distances_argmin_min(data, centers, metric)
        inertia = np.sum(distances**2)

        return labels, inertia

    
    def _compute_centers_counts(self, data, labels, n_centers):
        data_dims = data.shape[1]
        centers = np.zeros((n_centers, data_dims))
        counts = np.zeros(n_centers)

        for i in range(n_centers):
            data_i = data[labels == i]
            counts[i] = data_i.shape[0]
            if data_i.shape[0] != 0:
                center = np.array(np.sum(data_i, axis=0)/data_i.shape[0])
                centers[i] = center
            else:
                centers[i] = self._large_number
        
        return centers, counts


    # def get_report_for_server(self):
    #     updates_for_server = self._best_centers
    #     return updates_for_server


    # def update_device(self, reports_from_server):
    #     self._server_centers = reports_from_server