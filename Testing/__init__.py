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
    data_sets = ["circles", "moons", "blobs", "longblobs"]

    def __init__(self):
        self.data = np.load("sklearn_data.npz") # update for path
        self.labeled_data = np.load("sklearn_labels.npz") # update for path
    
    def get_data_set(self, name, noice_level):
        return self.data[name + "_" + noice_level + "_" + "noise"]

class DataSet:
    data = None
    label_data = None

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def add_label_to_data(self):
        return np.column_stack((self.data,self.labeled_data))
    
    def get_indices(self):
        self.data.shape[0]
    
    def get_indices_for_label(self, label_num):
        return np.arange(0, self.get_indices(self))[self.label_data == label_num]
    

# SAMPLERS

class DataSampler:
    def sample(self, dataset_class_instance, num_devices, pct_data_per_device, perc_iid_per_device, group_per_device): 
        assert(pct_data_per_device.size == num_devices)
        assert(perc_iid_per_device.size == num_devices)
        assert(group_per_device.size == num_devices)
        c = np.dstack((pct_data_per_device,perc_iid_per_device,group_per_device))[0]

        device_indices = []
        for [data_pct, perc_iid, group] in c:
            assert(data_pct >= 0 and data_pct <= 1)
            assert(perc_iid >= 0 and perc_iid <= 1)
            iid_indices_population = dataset_class_instance.get_indices()
            num_iid_items = iid_indices_population.size * data_pct * perc_iid
            iid_indices_sample = np.random.choice(iid_indices_population, size=num_iid_items, replace=False)
            
            num_noniid_items = iid_indices_population.size * data_pct * (1-perc_iid)
            non_iid_indices_population = dataset_class_instance.get_indices_for_label(group)
            non_iid_indices_sample = np.random.choice(non_iid_indices_population, size=num_noniid_items, replace=False)

            device_indices.append(np.unique(np.concatenate((iid_indices_sample,non_iid_indices_sample),0)))
        assert(len(device_indices) == num_devices)
        return device_indices



# Algorithm Interfaces 

class DeviceAlg:
    # In chronological order of calling
    def __init__(self, indicies_for_data_subset):
        pass
    def run_on_device():
        pass
    def get_report_for_server(self):
        updates_for_devices = []
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

    def report_back_to_server(input):
        return self.alg.get_report_for_server()

    def update(update_data):
        self.alg.update_device(update_data)

class Server:
    indicies = np.array([])
    device_groups = []

    def __init__(self, alg_server_class, indicies, device_groups):
        self.alg = alg_class()
        self.indicies = indicies
        self.device_groups = device_groups
    
    def round(self):
        devices = []
        reports = self.run_devices(devices)
        self.update(reports)
        self.run()
        update_for_devices = self.get_update_for_devices()
        self.send_updates_to_device(devices, update_for_devices)


    def run_devices(self, devices):
        reports = []
        for device in devices:
            device.run_on_server()
            report = device.report_back_to_server()
            reports.append(reports)
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
    devices = []

    def __init__(self, dataset_class_instance, num_devices, pct_data_per_device, perc_iid_per_device, group_per_device):
        sampler_instance = DataSampler()
        data = sampler_instance.sample(dataset_class_instance, num_devices, pct_data_per_device, perc_iid_per_device, group_per_device)
        assert(len(data) == num_devices)
        self.devices = [Device(alg_class, indicies) for indicies in range(data)]

    