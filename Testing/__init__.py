#!/usr/bin/python3
from abc import ABC,abstractmethod 
import numpy as np
import math

# Data Sets (x4)
# 5 levels of noice per data set
# 10,000 data points
from functools import partial


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
        return self.data[self.construct_key(name,noice_level)][::4]
    
    def get_label_set(self, name, noice_level):
        return self.labeled_data[self.construct_key(name,noice_level)][::4]

class DataSet:
    data = None
    labeled_data = None

    def __init__(self, data, labels):
        self.data = data
        self.labeled_data = labels
    
    def add_label_to_data(self):
        return np.column_stack((self.data,self.labeled_data))
    
    def get_indices(self):
        return np.arange(0, self.data.shape[0])
    
    def get_indices_for_label(self, label_num):
        return self.get_indices()[self.labeled_data == label_num]
    

# SAMPLERS

class DataSampler:
    def sample(self, dataset_class_instance, num_devices, pct_data_per_device, perc_iid_per_device, group_per_device): 
        assert(pct_data_per_device.size == num_devices)
        assert(perc_iid_per_device.size == num_devices)
        assert(group_per_device.size == num_devices)
        c = np.dstack((pct_data_per_device,perc_iid_per_device,group_per_device))[0]

        # device_indices = []
        groups = {}
        for [data_pct, perc_iid, group] in c:
            assert(data_pct >= 0 and data_pct <= 1)
            assert(perc_iid >= 0 and perc_iid <= 1)
            iid_indices_population = dataset_class_instance.get_indices()
            num_iid_items = math.floor(iid_indices_population.size * data_pct * perc_iid)
            iid_indices_sample = np.random.choice(iid_indices_population, size=num_iid_items, replace=False)
            
            num_noniid_items = math.floor(iid_indices_population.size * data_pct * (1-perc_iid))
            non_iid_indices_population = dataset_class_instance.get_indices_for_label(group)
            non_iid_indices_sample = np.random.choice(non_iid_indices_population, size=num_noniid_items, replace=False)

            if group not in groups:
                groups[group] = []
            final_indicies = np.unique(np.concatenate((iid_indices_sample,non_iid_indices_sample),0))
            data_subset = [dataset_class_instance.data[i] for i in final_indicies]
            groups[group].append(data_subset)
        return groups



# Algorithm Interfaces 

class DeviceAlg:
    # In chronological order of calling
    def __init__(self, indicies_for_data_subset):
        pass
    def run_on_device(self):
        pass
    def get_report_for_server(self):
        updates_for_server = []
        return updates_for_server
    def update_device(self, reports_from_server):
        pass

class ServerAlg:
    # In chronological order of calling
    def __init__(self):
        pass
    def update_server(self, reports_from_devices):
        pass
    def run_on_server(self):
        pass
    def get_reports_for_devices(self):
        updates_for_devices = []
        return updates_for_devices

# DEVICES

class Device:
    # data = np.array([], ndmin=2)
    indicies = np.array([])

    def __init__(self, alg_device_class, indicies, id_num = None):
        # self.data = data
        # Passes starting data to alg through constructor
        self.alg = alg_device_class(np.array(indicies), id_num=id_num)
        self.indicies = np.array(indicies)
    
    def run(self):
        self.alg.run_on_device()

    def report_back_to_server(self):
        return self.alg.get_report_for_server()

    def update(self, update_data):
        self.alg.update_device(update_data)

import random
class Server:
    device_groups = []

    def classify(self, data):
        return self.alg.classify(data)

    def __init__(self, alg_server_class, device_groups):
        self.alg = alg_server_class()
        self.device_groups = device_groups
    
    def run_round(self, num_devices_per_group_dict):
        devices = []
        for group,num_of_devices in num_devices_per_group_dict.items():
            devices += random.sample(self.device_groups[group],num_of_devices)
        reports = self.run_devices(devices)
        self.update(reports)
        self.run()
        update_for_devices = self.get_reports_for_devices()
        self.send_updates_to_device(devices, update_for_devices)


    def run_devices(self, devices):
        reports = []
        for device in devices:
            device.run()
            report = device.report_back_to_server()
            reports.append(report)
        return reports

    def run(self):
        self.alg.run_on_server()
    
    def update(self, reports):
        # aggregates all the device feedback
        self.alg.update_server(reports)
    
    def get_reports_for_devices(self):
        return self.alg.get_reports_for_devices()

    def send_updates_to_device(self, devices, update_for_devices): # report
        for device in devices:
            device.update(update_for_devices)

from sklearn import metrics
class DeviceSuite:
    server = None
    devices = []
    groups = {}

    def __init__(self, server_alg_class, device_alg_class, dataset_class_instance, num_devices, pct_data_per_device, perc_iid_per_device, group_per_device):
        sampler_instance = DataSampler()
        data = sampler_instance.sample(dataset_class_instance, num_devices, pct_data_per_device, perc_iid_per_device, group_per_device)
        # assert(len(data) == num_devices)
        for group, devices in data.items():
            self.groups[group] = [Device(device_alg_class, indicies, id_num=(group*10000+i)) for i,indicies in enumerate(devices)]
        
        self.server = Server(server_alg_class, self.groups)
    
    def run_rounds(self, num_devices_per_group_per_round):
        # list of dictionaries of integers
        for num_devices_per_group in num_devices_per_group_per_round:
            self.server.run_round(num_devices_per_group)

    def run_rounds_with_accuracy(self, num_devices_per_group_per_round, data, labels):
        for num_devices_per_group in num_devices_per_group_per_round:
            self.server.run_round(num_devices_per_group)
            print("Accuracy: ", self.accuracy(data, labels))

    def accuracy(self, data, labels):
        pred_labels = self.server.classify(data)
        return metrics.adjusted_rand_score(labels, pred_labels)


# algorithm
# dataset
# number of devices
# iid-ness (perc_iid_per_device)
    # groupings
# pct_data_per_device
# number of devices per group

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

    print("Done Fed")
    print("Accuracy: ", suite.accuracy(data, labels))


def asymptotic_decay(learning_rate, t, max_iter):
    return learning_rate / (1+t/(max_iter/2))

from Algorithms.som import SOM_server,SOM_Device
def som():
    params = { 
        "X": 2, 
        "Y": 2, 
        "INPUT_LEN": 10000, # TODO: Fix
        "SIGMA": 1.0, 
        "LR": 0.5, 
        "SEED": 1,
        "NEIGH_FUNC": "gaussian",
        "ACTIVATION": 'euclidean',
        "MAX_ITERS": 10,
        "DECAY": asymptotic_decay
    }
    # set seed
    random.seed(params['SEED'])
    basic(partial(SOM_server, params=params), partial(SOM_Device, params=params))

from Algorithms.k_means import CURE_Server,K_Means_Device
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

if __name__ == "__main__":
    # som()
    cure()