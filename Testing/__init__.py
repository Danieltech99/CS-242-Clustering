#!/usr/bin/python3
from abc import ABC,abstractmethod 
import numpy as np


# Data Sets (x4)
# 5 levels of noice per data set
# 10,000 data points



class DataSetCollection:
    data = None
    labeled_data = None
    noice_levels = ["vhigh", "high", "med", "low", "vlow"]
    data_sets_names = ["circles", "moons", "blobs", "longblobs"]

    def __init__(self):
        self.data = np.load("sklearn_data.npz") # update for path
        self.labeled_data = np.load("sklearn_labels.npz") # update for path
    
    def construct_key(self, name, noice_level):
        assert(name in self.data_sets_names)
        assert(noice_level in self.noice_levels)
        return name + "_" + noice_level + "_" + "noise"
    
    def get_set(self, name, noice_level):
        return self.get_data_set, self.get_label_set

    def get_data_set(self, name, noice_level):
        return self.data[self.construct_key(name,noice_level)]
    
    def get_label_set(self, name, noice_level):
        return self.labeled_data[self.construct_key(name,noice_level)]

class DataSet:
    data = None
    labeled_data = None

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
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
            num_iid_items = iid_indices_population.size * data_pct * perc_iid
            iid_indices_sample = np.random.choice(iid_indices_population, size=num_iid_items, replace=False)
            
            num_noniid_items = iid_indices_population.size * data_pct * (1-perc_iid)
            non_iid_indices_population = dataset_class_instance.get_indices_for_label(group)
            non_iid_indices_sample = np.random.choice(non_iid_indices_population, size=num_noniid_items, replace=False)

            if group not in groups:
                groups[group] = []
            groups[group].append(np.unique(np.concatenate((iid_indices_sample,non_iid_indices_sample),0)))
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
    def run_on_server(self):
        pass
    def update_server(self, reports_from_devices):
        pass
    def get_reports_for_devices(self):
        updates_for_devices = []
        return updates_for_devices

# DEVICES

class Device:
    # data = np.array([], ndmin=2)
    indicies = np.array([])

    def __init__(self, alg_device_class, indicies):
        # self.data = data
        # Passes starting data to alg through constructor
        self.alg = alg_device_class(indicies)
        self.indicies = indicies
    
    def run(self):
        self.alg.run_on_device()

    def report_back_to_server(self, input):
        return self.alg.get_report_for_server()

    def update(self, update_data):
        self.alg.update_device(update_data)

import random
class Server:
    device_groups = []

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
            device.run_on_server()
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


class DeviceSuite:
    server = None
    devices = []
    groups = {}

    def __init__(self, server_alg_class, device_alg_class, dataset_class_instance, num_devices, pct_data_per_device, perc_iid_per_device, group_per_device):
        sampler_instance = DataSampler()
        data = sampler_instance.sample(dataset_class_instance, num_devices, pct_data_per_device, perc_iid_per_device, group_per_device)
        assert(len(data) == num_devices)
        for group, devices in data.items():
            self.groups[group] = [Device(device_alg_class, indicies) for indicies in range(devices)]
        
        self.server = Server(server_alg_class, self.groups)
    
    def run_rounds(self, num_devices_per_group_per_round):
        # list of dictionaries of integers
        for num_devices_per_group in num_devices_per_group_per_round:
            self.server.run_round(num_devices_per_group)



# algorithm
# dataset
# number of devices
# iid-ness (perc_iid_per_device)
    # groupings
# pct_data_per_device
# number of devices per group

collection = DataSetCollection()

def basic(server_alg_class, device_alg_class):
    data, labels = collection.get_set("blobs", "vlow")
    dataset = DataSet(data, labels)    
    
    num_of_devices = 100
    pct_data_per_device = np.array([0.2] * num_of_devices) 
    perc_iid_per_device = np.array([0.2] * num_of_devices) 
    group_per_device    = np.round(np.linspace(0,1,num_of_devices))

    suite = DeviceSuite(server_alg_class, device_alg_class, dataset, num_of_devices, pct_data_per_device, perc_iid_per_device, group_per_device)

    number_of_rounds = 50
    num_devices_per_group_per_round = [{0: 10, 1:10}] * number_of_rounds
    
    suite.run_rounds(num_devices_per_group_per_round)