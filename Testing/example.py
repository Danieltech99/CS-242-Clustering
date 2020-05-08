#!/usr/bin/python3
from abc import ABC,abstractmethod 
import numpy as np
import math
from collections import OrderedDict 
import threading

# Data Sets (x4)
# 5 levels of noice per data set
# 10,000 data points
from functools import partial
import matplotlib.pyplot as plt

import random
random.seed(242)
np.random.rand(242)

ENABLE_PLOTS = False
ENABLE_PRINTS = False
ENABLE_PROGRESS = True
ENABLE_ROUND_PROGRESS_PLOT = False

CURRENT_FILE_NAME = None
from Testing.data import DataSetCollection, DataSet, DataSampler
from Testing.devices import Device, Server

from Testing.testing import DeviceSuite,asymptotic_decay

collection = DataSetCollection()



def basic(server_alg_class, device_alg_class):
    data, labels = collection.get_set("blobs", "vhigh")
    dataset = DataSet(data, labels)    
    
    num_of_devices = 100
    pct_data_per_device = np.array([0.1] * num_of_devices) 
    perc_iid_per_device = np.array([0.9] * num_of_devices) 
    group_per_device    = np.round(np.linspace(0,2,num_of_devices))

    suite = DeviceSuite(server_alg_class, device_alg_class, dataset, num_of_devices, pct_data_per_device, perc_iid_per_device, group_per_device)

    number_of_rounds = 10
    num_devices_per_group_per_round = [{0: 10, 1:10, 2:10}] * number_of_rounds

    suite.run_rounds_with_accuracy(num_devices_per_group_per_round,data, labels)

    if ENABLE_PRINTS: print("Done Fed")
    if ENABLE_PRINTS: print("Accuracy: ", suite.accuracy(data, labels))


from Algorithms.som import SOM_server,SOM_Device
def som():
    input_len = 2 # this is the length of each data point
    params = { 
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
    # set seed
    random.seed(params['SEED'])
    basic(partial(SOM_server, params=params), partial(SOM_Device, params=params))

from Algorithms.k_means import CURE_Server,K_Means_Device,CURE_Server_Carry,CURE_Server_Keep, KMeans_Server, KMeans_Server_Carry, KMeans_Server_Keep
def cure():
    params = {"N_CLUSTERS": 10,
          "MAX_ITERS": 100,
          "N_INITS": 10,
          "METRIC": "euclidean",
          "TOLERANCE": None}

    cure_params = {"N_CLUSTERS": 3,
               "N_REP_POINTS": 1,
               "COMPRESSION": 0.05}

    # set seed
    basic(partial(CURE_Server, cure_params=cure_params), partial(K_Means_Device, params=params))



































