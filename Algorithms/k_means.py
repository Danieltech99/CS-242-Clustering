import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin, pairwise_distances_argmin_min
from pyclustering.cluster.cure import cure
from pyclustering.cluster import cluster_visualizer


params = {"N_CLUSTERS": 10,
          "MAX_ITERS": 100,
          "N_INITS": 10,
          "METRIC": "euclidean",
          "TOLERANCE": None}

cure_params = {"N_CLUSTERS": 2,
               "N_REP_POINTS": 8,
               "COMPRESSION": 0.05}

class K_Means_Device:
# In chronological order of calling
    def __init__(self, data, params, id_num=None):
        """Initialize k-means member variables"""
        self._n_clusters = params["N_CLUSTERS"]
        self._max_iters = params["MAX_ITERS"]
        self._n_inits = params["N_INITS"]
        self._metric = params["METRIC"]
        
        self._data = data
        self._n_dims = data.shape[1]

        if params["TOLERANCE"] is None:
            self._tolerance = np.min(np.std(data, axis=0))/100
        else:
            self._tolerance = params["TOLERANCE"]

        self._server_centers = None

        self._best_inertia = None
        self._best_labels = None
        self._best_centers = None

        
    def run_on_device(self):
        """Trains the k-means"""
        # Sets data indices for training data input

        data = self._data
        for i in range(self._n_inits):
            initial_centers = self._init_cluster_centers(data)
            self._fit(data, initial_centers)

        
    def _init_cluster_centers(self, data, metric=None):
        server_centers = self._server_centers
        n_clusters = self._n_clusters
        if metric == None:
            metric = "euclidean"

        if server_centers is None:
            centers = np.zeros((n_clusters, self._n_dims))
            centers[0] = data[np.random.choice(data.shape[0])]
            for i in range(1, n_clusters):
                all_centers = centers[0:i, :]
                distance_pairs = pairwise_distances(data, all_centers)
                sum_distances = np.sum(distance_pairs, axis=1)
                datapoint_probs = sum_distances/np.sum(sum_distances)
                
                next_center_idx = np.random.choice(data.shape[0], p=datapoint_probs)
                centers[i] = data[next_center_idx]

            all_centers = centers


        else:
            centers = np.zeros((n_clusters, self._n_dims))
            server_centers = np.array(server_centers)
            for i in range(n_clusters):
                all_centers = np.vstack((server_centers, centers[0:i, :]))
                distance_pairs = pairwise_distances(data, all_centers)
                sum_distances = np.sum(distance_pairs, axis=1)
                datapoint_probs = sum_distances/np.sum(sum_distances)
                
                next_center_idx = np.random.choice(data.shape[0], p=datapoint_probs)
                centers[i] = data[next_center_idx]

            all_centers = np.vstack((server_centers, centers))
        return all_centers


    def _fit(self, data, initial_centers):
        centers = initial_centers
        for i in range(self._max_iters):
            n_centers = centers.shape[0]
            centers_old = centers.copy()

            labels, inertia = self._compute_labels_inertia(data, centers_old)
            centers = self._compute_cluster_centers(data, labels, n_centers)

            if self._best_inertia == None or inertia < self._best_inertia:
                self._best_inertia = inertia.copy()
                self._best_labels = labels.copy()
                self._best_centers = centers_old.copy()
                
            if centers_old.shape == centers.shape and np.allclose(centers_old, centers, atol=self._tolerance):
                break


    def _compute_labels_inertia(self, data, centers, metric=None):
        if metric == None:
            metric = "euclidean"

        # print("centers", centers)
        labels, distances = pairwise_distances_argmin_min(data, centers, metric)
        inertia = np.sum(distances**2)

        return labels, inertia

    
    def _compute_cluster_centers(self, data, labels, n_centers):
        # n_datapoints = data.shape[0]
        data_dims = data.shape[1]
        centers = np.zeros((n_centers, data_dims))

        j = 0
        for i in range(n_centers):
            data_i = data[labels == i]
            if data_i.shape[0] != 0:
                center = np.array(np.sum(data_i, axis=0)/data_i.shape[0])
                centers[i] = center
                j += 1

        return centers[0:j]


    def get_report_for_server(self):
        updates_for_server = self._best_centers
        return updates_for_server


    def update_device(self, reports_from_server):
        self._server_centers = reports_from_server


class CURE_Server:
    # In chronological order of calling
    def __init__(self, cure_params):
        self._n_clusters = cure_params["N_CLUSTERS"]
        self._n_rep_points = cure_params["N_REP_POINTS"]
        self._compression = cure_params["COMPRESSION"]

        self.clusters_from_devices = None
        self.representors = None
        self.server_clusters = None

    def update_server(self, reports_from_devices):
        self.clusters_from_devices = np.vstack(reports_from_devices)

    def run_on_server(self, clusters_from_devices = None):
        if clusters_from_devices is None: clusters_from_devices = self.clusters_from_devices
        cure_instance = cure(clusters_from_devices, self._n_clusters, self._n_rep_points, self._compression)
        cure_instance.process()
        representors = cure_instance.get_representors()
        self.representors = representors
        self.server_clusters = np.vstack(representors)
        clusters = cure_instance.get_clusters()
        visualizer = cluster_visualizer()
        visualizer.append_clusters(clusters, clusters_from_devices)
        visualizer.show()

    def get_reports_for_devices(self):
        return self.server_clusters
    
    def classify(self,data):
        representor_to_clustering = {}
        i = 0
        for j, cluster in enumerate(self.representors):
            for representor in cluster:
                representor_to_clustering[i] = j
                i += 1

        labels = pairwise_distances_argmin(data, self.server_clusters).tolist()
        labels = np.array([representor_to_clustering[label] for label in labels])
        return labels

class CURE_Server_Carry(CURE_Server):
    prev_round_info = None
    def update_server(self, reports_from_devices):
        # Save current state before update
        # print("previous varify clusters", self.clusters_from_devices)
        self.prev_round_info = self.clusters_from_devices
        # Now update
        super().update_server(reports_from_devices)
        # print("previous varify clusters", self.prev_round_info)
    
    def run_on_server(self):
        # Combine last and current state
        # print("device clusters", self.clusters_from_devices)
        if self.prev_round_info is None:
            self.prev_round_info = self.clusters_from_devices
        # print("previous clusters", self.prev_round_info)
        clusters = np.concatenate((self.clusters_from_devices, self.prev_round_info), axis=0)
        # print("concat", clusters)
        # clusters = None
        # Okay now run as normal
        super().run_on_server(clusters)

class CURE_Server_Keep(CURE_Server):
    prev_round_info = None
    def update_server(self, reports_from_devices):
        # Save current state before update
        self.prev_round_info = self.clusters_from_devices
        # Now update
        super().update_server(reports_from_devices)
        if self.prev_round_info is None:
            self.prev_round_info = self.clusters_from_devices
        self.clusters_from_devices = np.concatenate((self.clusters_from_devices, self.prev_round_info), axis=0)