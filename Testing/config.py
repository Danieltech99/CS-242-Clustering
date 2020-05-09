from Testing.data import DataSetCollection, DataSet, DataSampler

input_len = 2
def asymptotic_decay(learning_rate, t, max_iter):
    return learning_rate / (1+t/(max_iter/2))

layer_map = dict(
    som_params = {
        " ": { 
            "X": 2, 
            "Y": input_len, # must be the same as the input length to classify properly
            "INPUT_LEN": input_len, 
            "SIGMA": 1.0, 
            "LR": 0.5, 
            "SEED": 1,
            "NEIGH_FUNC": "gaussian",
            "ACTIVATION": 'euclidean',
            "MAX_ITERS": 5,
            "DECAY": asymptotic_decay
        }
    },
    k_means_device_params = {
        " ": {"N_CLUSTERS": 10, # device mid clusters
            "MAX_ITERS": 100,
            "N_INITS": 10,
            "METRIC": "euclidean",
            "TOLERANCE": None},
    },
    k_means_server_params = {
        " ": {"N_CLUSTERS": 3,
            "N_REP_POINTS": 1,
            "COMPRESSION": 0.05},
    }
)

# Layers
# --------
# Data set
# Suite 
# Noise
# Tests (Alg, Alg Param)
from functools import reduce
collection = DataSetCollection()
from Algorithms.k_means import CURE_Server,K_Means_Device,CURE_Server_Carry,CURE_Server_Keep, KMeans_Server, KMeans_Server_Carry, KMeans_Server_Keep
from Algorithms.som import SOM_Device, SOM_server
layers = {
    "noice": (lambda dataset: collection.noice_levels[:-1] if dataset == "circles-grouped" else collection.noice_levels),
    "suites": [
        {
            "name": "Basic",
            "datasets": DataSetCollection.data_sets_names,
            "rounds": 5,
            "devices": 1000, # number of devices per group
            "pct_data_per_device": 0.01,
            "transition": [False],
            "groups": lambda dsclass,size: {"uniform": (lambda: dsclass.rand(size))},
            "timeline": {
                0: {
                    "uniform": 10 # number of devices per round
                },
            }
        },
        {
            "name": "Crowd Discover",
            "datasets": DataSetCollection.data_sets_names[0:1],
            "rounds": lambda d,size: 3 * d.count,
            "devices": 1000,
            "pct_data_per_device": 0.01,
            "transition": [True, False],
            # Create groups with names counting up from 0 to count
            # group n has datapoints from cluster n and is concated with group n-1
            "groups": lambda d,s: {g: (lambda: d.rand_g_c(s,g)) for g in range(d.count)},
            # At each 3n rounds, switch to group n
            "timeline": lambda d,size: reduce((lambda o, s: o.update({3*s: {s: 10}}) or o), range(d.count), {}),
        },
        {
            "name": "Specialized Devices",
            "datasets": DataSetCollection.data_sets_names[0:1],
            "rounds": 5,
            "devices": 1000,
            "pct_data_per_device": 0.01,
            "transition": [False],
            # One group per cluster
            "groups": lambda d,s: {g: (lambda: d.rand_g(s,g)) for g in range(d.count)},
            # All rounds (round 0), sample evenly from all groups
            "timeline": lambda d,size: {0: reduce((lambda o, s: o.update({s: round(10/d.count)}) or o), range(d.count), {})},
        },
        {
            "name": "Crowd Replacement",
            "datasets": DataSetCollection.data_sets_names[0:1],
            "rounds": lambda d,size: 3 * d.count,
            "devices": 1000,
            "pct_data_per_device": 0.01,
            "transition": [True, False],
            # One group per cluster
            "groups": lambda d,s: {g: (lambda: d.rand_g(s,g)) for g in range(d.count)},
            # At each 3n rounds, switch to group n
            "timeline": lambda d,size: reduce((lambda o, s: o.update({3*s: {s: 10}}) or o), range(d.count), {}),
        },
        {
            # Start with all devices having cluster 1
            # After 3 rounds, let 20 devices discover another cluster
            # After another 3, let the same 20 discover another
            "name": "Subset Discover",
            "datasets": DataSetCollection.data_sets_names[0:1],
            "rounds": lambda d,size: 3 * d.count,
            "devices": 1000,
            "pct_data_per_device": 0.01,
            "transition": [True, False],
            # Create groups with names counting up from 0 to count
            # group n has datapoints from cluster n and is concated with group n-1
            "groups": lambda d,s: {g: (lambda: d.rand_g_c(s,g)) for g in range(d.count)},
            # Start all devices at group 1,
            # Move 20 devices up a group every 3 rounds
            "timeline": lambda d,size: reduce((lambda o, s: o.update({3*s: {s: 2, 0: (10 if s == 0 else 8)}}) or o), range(d.count), {}),
        },
        {
            # Start with all devices having cluster 1
            # After 3 rounds, let 20 devices discover another cluster
            # After another 3, let the same 20 discover another
            "name": "One Discover",
            "datasets": DataSetCollection.data_sets_names[0:1],
            "rounds": lambda d,size: 3 * d.count,
            "devices": 1000,
            "pct_data_per_device": 0.01,
            "transition": [True, False],
            # Create groups with names counting up from 0 to count
            # group n has datapoints from cluster n and is concated with group n-1
            "groups": lambda d,s: {g: (lambda: d.rand_g_c(s,g)) for g in range(d.count)},
            # Start all devices at group 1,
            # Move 20 devices up a group every 3 rounds
            "timeline": lambda d,size: reduce((lambda o, s: o.update({3*s: {s: 1, 0: (10 if s == 0 else 9)}}) or o), range(d.count), {}),
        },
    ],
    "algs": [
        {
            "name": "SOM (Fed)",
            "server": {"class": SOM_server, "kwargs": dict(params=layer_map["som_params"])}, 
            "device": {"class": SOM_Device, "kwargs": dict(params=layer_map["som_params"])},
        },
        {
            "name": "KMeans Server",
            "server": {"class": KMeans_Server, "kwargs": dict(cure_params=layer_map["k_means_server_params"])},
            "device": {"class": K_Means_Device, "kwargs": dict(params=layer_map["k_means_device_params"])},
        },
        {
            "name": "KMeans Server Carry",
            "server": {"class": KMeans_Server_Carry, "kwargs": dict(cure_params=layer_map["k_means_server_params"])},
            "device": {"class": K_Means_Device, "kwargs": dict(params=layer_map["k_means_device_params"])},
        },{
            "name": "KMeans Server Keep",
            "server": {"class": KMeans_Server_Keep, "kwargs": dict(cure_params=layer_map["k_means_server_params"])},
            "device": {"class": K_Means_Device, "kwargs": dict(params=layer_map["k_means_device_params"])},
        }
    ]
}