from Testing.data import DataSetCollection, DataSet, DataSampler
from functools import partial

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
        " ": {"N_CLUSTERS": 20, # device mid clusters
            "MAX_ITERS": 100,
            "N_INITS": 5,
            "METRIC": "euclidean",
            "TOLERANCE": None,
            "KEEP_EMPTY_CLUSTERS": True},
    },
    k_means_server_params = {
        " ": {"N_CLUSTERS": 3,
            "N_REP_POINTS": 1,
            "COMPRESSION": 0.05},
    },
    gossip_device_params = {
        " ": {
            "N_DEVICES": 120,
            "N_CLUSTERS": 6,
            "N_INITS": 10,
            "CACHE_SIZE": 4,
            "GLOBAL_INIT": None,
            "METRIC": "euclidean",
            "MAX_ITERS": 100,
            "TOLERANCE": None
        }
    },
    gossip_server_params = {
        " ": {
            "N_DEVICES": 120,
            "N_CLUSTERS": 6,
            "N_AGGREGATE_ROUNDS": 3
        }
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
from Algorithms.k_means import CURE_Server,K_Means_Device,CURE_Server_Carry,CURE_Server_Keep, KMeans_Server, KMeans_Server_Carry, KMeans_Server_Keep, KMeans_Server_Central, KMeans_Device_Central
from Algorithms.som import SOM_Device, SOM_server
from Algorithms.gossip_k_means import gossip_KMeans_server,gossip_KMeans_Device
layers = {
    "noice": (lambda dataset: collection.noice_levels[:-1] if dataset == "circles-grouped" else collection.noice_levels),
    "suites": [
        # {
        #     "name": "Static and Uniform",
        #     "non_fed": True,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": 5,
        #     "devices": 120, # number of devices per group
        #     "devices_per_round": 12,
        #     "pct_data_per_device": 0.01,
        #     "transition": [False],
        #     "groups": lambda dsclass,size: {"uniform": partial(dsclass.rand, size=size)},
        #     "timeline": {
        #         0: {
        #             "uniform": 12 # number of devices per round
        #         },
        #     }
        # },
        # {
        #     "name": "25% IID",
        #     "non_fed": True,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": 5,
        #     "devices": 120, # number of devices per group
        #     "devices_per_round": 12,
        #     "pct_data_per_device": 0.01,
        #     "transition": [False],
        #     "groups": lambda d,s: {g: partial(d.rand_iid, size=s,group=g,perc=0.25) for g in range(d.t_count)},
        #     "timeline": lambda d,size: {0: reduce((lambda o, s: o.update({s: round(12/d.t_count)}) or o), range(d.t_count), {})},
        # },
        # {
        #     "name": "50% IID",
        #     "non_fed": True,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": 5,
        #     "devices": 120, # number of devices per group
        #     "devices_per_round": 12,
        #     "pct_data_per_device": 0.01,
        #     "transition": [False],
        #     "groups": lambda d,s: {g: partial(d.rand_iid, size=s,group=g,perc=0.5) for g in range(d.t_count)},
        #     "timeline": lambda d,size: {0: reduce((lambda o, s: o.update({s: round(12/d.t_count)}) or o), range(d.t_count), {})},
        # },
        # {
        #     "name": "75% IID",
        #     "non_fed": True,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": 5,
        #     "devices": 120, # number of devices per group
        #     "devices_per_round": 12,
        #     "pct_data_per_device": 0.01,
        #     "transition": [False],
        #     "groups": lambda d,s: {g: partial(d.rand_iid, size=s,group=g,perc=0.75) for g in range(d.t_count)},
        #     "timeline": lambda d,size: {0: reduce((lambda o, s: o.update({s: round(12/d.t_count)}) or o), range(d.t_count), {})},
        # },
        # {
        #     "name": "100% IID",
        #     "non_fed": True,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": 5,
        #     "devices": 120,
        #     "devices_per_round": 12,
        #     "pct_data_per_device": 0.01,
        #     "transition": [False],
        #     # One group per cluster
        #     "groups": lambda d,s: {g: partial(d.rand_g, size=s,group=g) for g in range(d.t_count)},
        #     # All rounds (round 0), sample evenly from all groups
        #     "timeline": lambda d,size: {0: reduce((lambda o, s: o.update({s: round(12/d.t_count)}) or o), range(d.t_count), {})},
        # },
        # {
        #     "name": "Cross-Device Detection",
        #     "non_fed": True,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": 5,
        #     "devices": 120,
        #     "devices_per_round": 12,
        #     "pct_data_per_device": 0.01,
        #     "transition": [False],
        #     # One group per cluster
        #     "groups": lambda d,s: {g: partial(d.rand_g_cross, size=s,group=g) for g in range(d.count)},
        #     # All rounds (round 0), sample evenly from all groups
        #     "timeline": lambda d,size: {0: reduce((lambda o, s: o.update({s: round(12/d.count)}) or o), range(d.count), {})},
        # },
        # {
        #     "name": "Crowd Discover",
        #     "non_fed": False,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": lambda d,size: 3 * d.t_count,
        #     "devices": 120,
        #     "devices_per_round": 12,
        #     "pct_data_per_device": 0.01,
        #     "transition": [True],
        #     # Create groups with names counting up from 0 to count
        #     # group n has datapoints from cluster n and is concated with group n-1
        #     "groups": lambda d,s: {g: partial(d.rand_g_c, size=s,group=g) for g in range(d.t_count)},
        #     # At each 3n rounds, switch to group n
        #     "timeline": lambda d,size: reduce((lambda o, s: o.update({3*s: {s: 12}}) or o), range(d.t_count), {}),
        # },
        # {
        #     "name": "Crowd Replacement",
        #     "non_fed": False,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": lambda d,size: 3 * d.t_count,
        #     "devices": 120,
        #     "devices_per_round": 12,
        #     "pct_data_per_device": 0.01,
        #     "transition": [True],
        #     # One group per cluster
        #     "groups": lambda d,s: {g: partial(d.rand_g, size=s,group=g) for g in range(d.t_count)},
        #     # At each 3n rounds, switch to group n
        #     "timeline": lambda d,size: reduce((lambda o, s: o.update({3*s: {s: 12}}) or o), range(d.t_count), {}),
        # },
        {
            # Start with all devices having cluster 1
            # After 3 rounds, let 20 devices discover another cluster
            # After another 3, let the same 20 discover another
            "name": "Subset Discover",
            "non_fed": False,
            "datasets": DataSetCollection.data_sets_names,
            "rounds": lambda d,size: 6 * d.t_count,
            "devices": 120,
            "devices_per_round": 12,
            "pct_data_per_device": 0.01,
            "transition": [True],
            # Create groups with names counting up from 0 to count
            # group n has datapoints from cluster n and is concated with group n-1
            "groups": lambda d,s: {g: partial(d.rand_g_c, size=s,group=g) for g in range(d.t_count)},
            # Start all devices at group 1,
            # Move 20 devices up a group every 3 rounds
            "timeline": lambda d,size: reduce((lambda o, s: o.update({3*s: {s: 6, 0: (12 if s == 0 else 6)}}) or o), range(d.t_count), {}),
        },
        # #{
        # #    # Start with all devices having cluster 1
        # #    # After 3 rounds, let 20 devices discover another cluster
        # #    # After another 3, let the same 20 discover another
        # #    "name": "Subset Discover Long",
        # #    "non_fed": False,
        # #    "datasets": DataSetCollection.data_sets_names,
        # #    "rounds": lambda d,size: 6* d.t_count,
        # #    "devices": 120,
        # #    "devices_per_round": 12,
        # #    "pct_data_per_device": 0.01,
        # #    "transition": [True],
        # #    # Create groups with names counting up from 0 to count
        # #    # group n has datapoints from cluster n and is concated with group n-1
        # #    "groups": lambda d,s: {g: partial(d.rand_g_c, size=s,group=g) for g in range(d.t_count)},
        # #    # Start all devices at group 1,
        # #    # Move 20 devices up a group every 3 rounds
        # #    "timeline": lambda d,size: reduce((lambda o, s: o.update({6*s: {s: 6, 0: (12 if s == 0 else 6)}}) or o), range(d.t_count), {}),
        # #},
        # {
        #     # Start with all devices having cluster 1
        #     # After 3 rounds, let 20 devices discover another cluster
        #     # After another 3, let the same 20 discover another
        #     "name": "One Discover",
        #     "devices_per_round": 12,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": lambda d,size: 3 * d.t_count,
        #     "devices": 20,
        #     "pct_data_per_device": 0.05,
        #     "transition": [True, False],
        #     # Create groups with names counting up from 0 to count
        #     # group n has datapoints from cluster n and is concated with group n-1
        #     "groups": lambda d,s: {g: partial(d.rand_g_c, size=s,group=g) for g in range(d.t_count)},
        #     # Start all devices at group 1,
        #     # Move 20 devices up a group every 3 rounds
        #     "timeline": lambda d,size: reduce((lambda o, s: o.update({3*s: {s: 1, 0: (12 if s == 0 else 9)}}) or o), range(d.t_count), {}),
        # },
        # {
        #     # Start with all devices having cluster 1
        #     # After 3 rounds, let 20 devices discover another cluster
        #     # After another 3, let the same 20 discover another
        #     "name": "One Discover Long",
        #     "devices_per_round": 12,
        #     "datasets": DataSetCollection.data_sets_names,
        #     "rounds": lambda d,size: 5* d.t_count,
        #     "devices": 15,
        #     "pct_data_per_device": 0.05,
        #     "transition": [True, False],
        #     # Create groups with names counting up from 0 to count
        #     # group n has datapoints from cluster n and is concated with group n-1
        #     "groups": lambda d,s: {g: partial(d.rand_g_c, size=s,group=g) for g in range(d.t_count)},
        #     # Start all devices at group 1,
        #     # Move 20 devices up a group every 3 rounds
        #     "timeline": lambda d,size: reduce((lambda o, s: o.update({5*s: {s: 1, 0: (12 if s == 0 else 9)}}) or o), range(d.t_count), {}),
        # },
    ],
    "algs": [
        {
            "name": "KMeans Distributed Decentralized (Gossip)",
            "device_multi": 10,
            "server": {"class": gossip_KMeans_server, "kwargs": dict(params=layer_map["gossip_server_params"])},
            "device": {"class": gossip_KMeans_Device, "kwargs": dict(params=layer_map["gossip_device_params"])},
        },
        {
            "name": "KMeans Distributed Centralized",
            "device_multi": 10,
            "server": {"class": KMeans_Server_Central, "kwargs": dict(cure_params=layer_map["k_means_server_params"])},
            "device": {"class": KMeans_Device_Central, "kwargs": dict(params=layer_map["k_means_device_params"])},
        },
        # {
        #     "name": "SOM (Fed)",
        #     "server": {"class": SOM_server, "kwargs": dict(params=layer_map["som_params"])}, 
        #     "device": {"class": SOM_Device, "kwargs": dict(params=layer_map["som_params"])},
        # },
        {
            "name": "KMeans Server",
            "server": {"class": KMeans_Server, "kwargs": dict(cure_params=layer_map["k_means_server_params"])},
            "device": {"class": K_Means_Device, "kwargs": dict(params=layer_map["k_means_device_params"])},
        },
        # {
        #     "name": "KMeans Server Carry",
        #     "server": {"class": KMeans_Server_Carry, "kwargs": dict(cure_params=layer_map["k_means_server_params"])},
        #     "device": {"class": K_Means_Device, "kwargs": dict(params=layer_map["k_means_device_params"])},
        # },
        {
            "name": "KMeans Server Keep",
            "server": {"class": KMeans_Server_Keep, "kwargs": dict(cure_params=layer_map["k_means_server_params"])},
            "device": {"class": K_Means_Device, "kwargs": dict(params=layer_map["k_means_device_params"])},
        },
    ]
}