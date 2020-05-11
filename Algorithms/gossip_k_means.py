import random
import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin, pairwise_distances_argmin_min
from sklearn.cluster import KMeans


device_params = {
    "N_CLUSTERS": 10,
    "N_INITS": 10,
    "N_DEVICES": 10,
    "CACHE_SIZE": 4,
    "GLOBAL_INIT": False,
    "METRIC": "euclidean",
    "MAX_ITERS": 100,
    "TOLERANCE": None
}

server_params = {
    "N_CLUSTERS": 4,
    "N_DEVICES": 10,
    "N_AGGREGATE_ROUNDS": 5
}


class gossip_KMeans_Device:
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
        self.cache = set(self.cache)

        # self.error = float("Inf")
        # self.local_error = None

        if self._global_init is None:
            if id_num == 0:
                self.centers = self._init_cluster_centers(self._data)
            else:
                self.centers = None
        else:
            self.centers = self._global_init
        
        self.local_centers = None

        self.labels = None
        self.local_labels = None
        
        self.local_counts = None

        self._large_number = 9999
        
        
    def run_on_device(self):
        """Trains the k-means"""
        # Sets data indices for training data input

        data = self._data
        centers = self.centers
        n_centers = self._n_clusters

        if centers is None:
            return

        labels, local_error = self._compute_labels_inertia(data, centers)
        local_centers, local_counts = self._compute_centers_counts(data, labels, n_centers)

        self.labels = labels
        # self.local_error = local_error
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


    def get_report_for_server(self):
        
        updates_for_server = {
                              "device_id": self._id_num,
                              "cache": self.cache,
                              "cache_size": self.cache_size,
                              "local_centers": self.local_centers,
                              "local_counts": self.local_counts
                              }
                
        return updates_for_server


    def update_device(self, reports_from_server):
        device_update = reports_from_server[self._id_num]

        self.cache = device_update["cache"]
        self.centers = device_update["centers"]
        # self.counts = device_update["counts"]
        # self.error = device_update["error"]



class gossip_KMeans_server:
    def __init__(self, params):
        self._n_clusters = params["N_CLUSTERS"]
        self._n_devices = params["N_DEVICES"]
        self._n_aggregate_rounds = params["N_AGGREGATE_ROUNDS"]

        self.updates_for_devices = {i:{} for i in range(self._n_devices)}


    def update_server(self, reports_from_devices):
        all_reports = {report["device_id"]:report for report in reports_from_devices}
        if reports_from_devices[1]["local_centers"] is None:
            first_round_sync(all_reports)
        else:
            gossip_sync(all_reports)

    def first_round_sync(self, all_reports):
        master_centers = all_reports[0]["local_centers"]

        updates_for_devices = {}
        for device_id in range(self._n_devices):
            update = {
                "cache": all_reports[device_id]["cache"],
                "centers": master_centers
            }
            updates_for_device[device_id] = update

        self.updates_for_devices = updates_for_devices
                

    def gossip_sync(self, all_reports):
        tmp_center_sums = {device_id:report["local_centers"] * report["local_counts"][:, None]
                            for device_id, report in all_reports.items()}
        tmp_weights = {device_id:report["local_counts"] 
                        for device_id, report in all_reports.items()}
        tmp_caches = {device_id:report["cache"]
                       for device_id, report in all_reports.items()}

        # Calculate new cluster center from neighbors
        for agg_round in range(self._n_aggregate_rounds):
            for device_id in range(self._n_devices):
                device_cache = tmp_caches[device_id]
                target_id = random.choice(device_cache)
                
                # update sums and weights
                tmp_center_sums[device_id] = tmp_center_sums[device_id] + \
                    1/2 * tmp_center_sums[target_id] * tmp_weights[target_id][:, None]
                tmp_weights[device_id] = tmp_weights[device_id] + 1/2*tmp_weights[target_id]
                
                # sync caches
                merged_cache = tmp_caches[device_id].union(tmp_caches[target_id])
                tmp_caches[device_id] = merged_cache
                tmp_caches[target_id] = merged_cache

                tmp_caches[device_id].discard(device_id)
                tmp_caches[target_id].discard(target_id)

                while len(tmp_caches[device_id]) > all_reports[device_id]["cache_size"]:
                    tmp_caches[device_id].pop()
                while len(tmp_caches[target_id]) > all_reports[target_id]["cache_size"]:
                    tmp_caches[target_id].pop() 


        updates_for_devices = {}
        for device_id in range(self._n_devices):
            update = {
                        "cache": tmp_caches[device_id],
                        "centers": tmp_center_sums[device_id] / tmp_weights[device_id][:, None]
                     }      
            updates_for_devices[device_id] = update

        self.updates_for_devices = updates_for_devices
                

    def get_reports_for_devices(self):
        return self.updates_for_devices

    def classify(self, data):
        rand_device_id = np.random.randint(self._n_devices)
        rand_device_centers = self.updates_for_devices[rand_device_id]["centers"]
        return pairwise_distances_argmin_min(data, rand_device_centers)[0]
 