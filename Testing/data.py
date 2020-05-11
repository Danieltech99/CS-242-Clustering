import numpy as np
import math
import random

class DataSetCollection:
    data = None
    labeled_data = None
    noice_levels = [
        # "vhigh", 
        "high", 
        # "med", 
        # "low", 
        # "vlow"
        ]
    # data_sets_names = ["moons", "circles", "longblobs", "blobs", "blobs2", "circle-grouped", "blobs-grouped"]
    data_sets_names = [
        # "blobs",  
        # "blobs2",
        # "circle-grouped",
        "blobs-grouped",
    ]
    data_sets_names_validate = ["blobs", "blobs2", "circle", "blobs"]
    count_map = {
        "circles" : 2, 
        "moons": 2, 
        "blobs": 3,  
        "longblobs": 3,
        "blobs2": 3,
        "circle-grouped": 4,
        "blobs-grouped": 9,
    }
    file_map = {
        "moons" : ("data", "circles"),
        "circles": ("data", "circles"),
        "longblobs": ("data", "longblobs"),
        "blobs": ("data", "blobs"),
        "blobs2": ("data", "blobs2"),
        "circle-grouped": ("data_grouped", "circle"),
        "blobs-grouped": ("data_grouped", "blobs"),
    }

    def __init__(self):
        self.data = np.load("Data/sklearn_data.npz") # update for path
        self.labeled_data = np.load("Data/sklearn_labels.npz") # update for path
        self.data_grouped = np.load("Data/multigroup_data.npz") # update for path
        self.labeled_data_grouped = np.load("Data/multigroup_labels.npz") # update for path
    
    def construct_key(self, name, noice_level):
        if name not in self.data_sets_names_validate:
            print("INVALID NAME: ", name)
        assert(name in self.data_sets_names_validate)
        assert(noice_level in self.noice_levels)
        return name + "_" + noice_level + "_" + "noise"
    
    def get_set(self, name, noice_level):
        return (self.get_data_set(name, noice_level), self.get_label_set(name, noice_level))
    
    def get_set_true(self, name, noice_level):
        return self.get_set(name, noice_level) + (self.get_true_label_set(name, noice_level),)

    def get_data_set(self, name, noice_level):
        location,key = self.file_map[name]
        data_source = getattr(self, location)
        res = data_source[self.construct_key(key,noice_level)][::4]
        return res
    
    def get_label_set(self, name, noice_level):
        location,key = self.file_map[name]
        data_source = getattr(self, "labeled_"+location)
        res = data_source[self.construct_key(key,noice_level)][::4]
        return res
    
    def get_true_label_set(self, name, noice_level):
        location,key = self.file_map[name]
        data_source = getattr(self, "labeled_"+location)
        res = data_source[self.construct_key(key,noice_level)+"_true"][::4]
        return res

class DataSet:
    data = None
    labeled_data = None
    true_labels = None
    count = None

    def __init__(self, data, labels, true_labels = None):
        self.data = data
        self.labeled_data = labels
        if true_labels is not None:
            self.true_labels = true_labels
        else: 
            self.true_labels = self.labeled_data
        self.count = np.max(labels) + 1
    
    def add_label_to_data(self):
        return np.column_stack((self.data,self.labeled_data))
    
    def get_indices(self):
        return np.arange(0, self.data.shape[0])
    
    def get_indices_for_label(self, label_num):
        return self.get_indices()[self.labeled_data == label_num]
    
    def get_indices_for_label_c(self, label_num):
        return self.get_indices()[self.labeled_data <= label_num]
    
    # def get_data(self):
    #     indicies = self.get_indices()
    #     return [self.data[i] for i in indicies]
    
    # def get_data_for_label(self, label_num):
    #     indicies = self.get_indices_for_label(label_num)
    #     return [self.data[i] for i in indicies]
    
    # def get_data_for_label_c(self, label_num):
    #     indicies = self.get_indices_for_label_c(label_num)
    #     return [self.data[i] for i in indicies]
    

    def rand(self, *, size):
        res = {}
        res["sample"] = lambda: [self.data[i] for i in np.random.choice(self.get_indices(), size=size, replace=False)]
        res["population"] = lambda: self.get_indices()
        return res
    
    def rand_g(self, *, size, group):
        res = {}
        res["sample"] = lambda: [self.data[i] for i in np.random.choice(self.get_indices_for_label(group), size=size, replace=False)]
        res["population"] = lambda: self.get_indices_for_label(group)
        return res
    
    def rand_g_c(self, *, size, group):
        res = {}
        res["sample"] = lambda: [self.data[i] for i in np.random.choice(self.get_indices_for_label_c(group), size=size, replace=False)]
        res["population"] = lambda: self.get_indices_for_label_c(group)
        return res

    def rand_iid(self, *, size, group, perc):
        res = {}
        res["sample"] = lambda: [self.data[i] for i in np.unique(np.concatenate((self.rand(size=size*(1-perc_iid)),self.rand_g(size=size*(perc_iid))),0))]
        res["population"] = lambda: self.get_indices()
        return res
    
    def get_data_for_indicies(self, indicies):
        return np.array([self.data[i] for i in indicies])
    
    def concat(self, first_group, second_group):
        return np.unique(np.concatenate((first_group,second_group),0))

    def merge(self, first_group, first_group_pct, second_group, second_group_pct):
        return np.unique(np.concatenate((first_group,second_group),0))
    

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

