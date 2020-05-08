#!/usr/bin/python3
from abc import ABC,abstractmethod 
import numpy as np
import math
from collections import OrderedDict 

# Data Sets (x4)
# 5 levels of noice per data set
# 10,000 data points
from functools import partial
import matplotlib.pyplot as plt

import random
random.seed(242)
np.random.rand(242)

ENABLE_PLOTS = True
ENABLE_PRINTS = False
ENABLE_PROGRESS = True
ENABLE_ROUND_PROGRESS_PLOT = False

CURRENT_FILE_NAME = None



class DataSetCollection:
    data = None
    labeled_data = None
    noice_levels = ["vhigh", "high", "med", "low", "vlow"]
    data_sets_names = ["moons", "circles", "longblobs", "blobs"]
    count_map = {
        "circles" : 2, 
        "moons": 2, 
        "blobs": 3, 
        "longblobs": 3
    }

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
        round_accs = []
        run_num = 1
        for num_devices_per_group in num_devices_per_group_per_round:
            self.server.run_round(num_devices_per_group)
            if ENABLE_ROUND_PROGRESS_PLOT or ENABLE_PRINTS: 
                acc = self.accuracy(data, labels, run_num)
                round_accs.append(acc)
            if ENABLE_PRINTS: print("Accuracy: ", acc)
            run_num += 1
        return round_accs

    def accuracy(self, data, labels, run_num = 1):
        pred_labels = self.server.classify(data)
        if ENABLE_PLOTS:
            plt.scatter(data[:, 0], data[:, 1], c=pred_labels, s=1)
            # plt.show()
            plt.savefig('Plots/' + CURRENT_FILE_NAME + "-round-" + str(run_num) + '.png')
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

    if ENABLE_PRINTS: print("Done Fed")
    if ENABLE_PRINTS: print("Accuracy: ", suite.accuracy(data, labels))


def asymptotic_decay(learning_rate, t, max_iter):
    return learning_rate / (1+t/(max_iter/2))

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

from Algorithms.k_means import CURE_Server,K_Means_Device,CURE_Server_Carry,CURE_Server_Keep
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



def custom(
        server_alg_class, device_alg_class,
        data_set = collection.get_set("blobs", "vhigh"),
        num_of_devices = 100,
        pct_data_per_device = np.array([0.1] * 100),
        perc_iid_per_device = np.array([0.9] * 100),
        group_per_device    = np.round(np.linspace(0,2,100)),
        number_of_rounds    = 2,
        num_devices_per_group_per_round = [{0: 10, 1:10, 2:10}] * 10
    ):
    data, labels = data_set
    # data, labels = collection.get_set("blobs", "vhigh")
    dataset = DataSet(data, labels)    
    
    # num_of_devices = 100
    # pct_data_per_device = np.array([0.1] * num_of_devices) 
    # perc_iid_per_device = np.array([0.9] * num_of_devices) 
    # group_per_device    = np.round(np.linspace(0,2,num_of_devices))

    suite = DeviceSuite(server_alg_class, device_alg_class, dataset, num_of_devices, pct_data_per_device, perc_iid_per_device, group_per_device)

    # number_of_rounds = 10
    # num_devices_per_group_per_round = [{0: 10, 1:10, 2:10}] * number_of_rounds

    round_accs = suite.run_rounds_with_accuracy(num_devices_per_group_per_round,data, labels)

    if ENABLE_PRINTS: print("Done Fed")
    acc = suite.accuracy(data, labels)
    if ENABLE_PRINTS: print("Accuracy: ", acc)
    return (acc,round_accs)


def run_test_and_save(
        result_dict,
        first_key,
        second_key,
        progress_lock,
        number_of_tests_finished,
        number_of_tests,
        data_set_name,
        level,
        test,
        data_set,
        num_of_devices,
        pct_data_per_device,
        perc_iid_per_device,
        group_per_device,
        number_of_rounds,
        num_devices_per_group_per_round,
    ): 
    global CURRENT_FILE_NAME
    if ENABLE_PRINTS: print(test["name"] + ":", data_set_name + "-" + level)
    CURRENT_FILE_NAME = test["name"] + ":" + data_set_name + "-" + level
    (result_dict["end"].value,round_accs) = custom(
            test["server"], test["device"],
            data_set = data_set,
            num_of_devices = num_of_devices,
            pct_data_per_device = pct_data_per_device,
            perc_iid_per_device = perc_iid_per_device,
            group_per_device    = group_per_device,   
            number_of_rounds    = number_of_rounds,   
            num_devices_per_group_per_round = num_devices_per_group_per_round
        )
    if ENABLE_ROUND_PROGRESS_PLOT: 
        result_dict["rounds"].extend(round_accs)
    if ENABLE_PROGRESS: 
        with progress_lock:
            number_of_tests_finished.value += 1
            print('Progress: {}/{} Complete \t {} \t {}'.format(number_of_tests_finished.value, number_of_tests, result_dict["end"].value, test["name"] + ":" + data_set_name + "-" + level) )


import multiprocessing 
def run_tests(tests, data_sets = collection.data_sets_names, levels = collection.noice_levels, number_of_rounds    = 4):
    results_dict = OrderedDict()
    manager = multiprocessing.Manager()
    
    progress_lock = multiprocessing.Lock()
    number_of_tests = 0
    number_of_tests_finished = multiprocessing.Value('i', 0)

    for data_set_name in data_sets:
        for level in levels:
            key_tests = OrderedDict()
            for test in tests:
                key_tests[test["name"]] = {
                    "end": multiprocessing.Value("d", 0.0, lock=False),
                    "rounds": manager.list()
                }
                number_of_tests += 1
            key = (data_set_name,level)
            results_dict[key] = key_tests

    if ENABLE_PROGRESS: 
        print("Number of Tests: {}".format(number_of_tests))

    num_of_devices = 100
    pct_data_per_device = np.array([0.1] * num_of_devices)
    perc_iid_per_device = np.array([0.5] * num_of_devices)
    

    processes = []

    for data_set_name in data_sets:

        count = collection.count_map[data_set_name]

        group_per_device    = np.round(np.linspace(0,count-1,num_of_devices))    
        num_devices_per_group_per_round = [({i:10 for i in range(count-1)})] * number_of_rounds

        for level in levels:
            count = collection.count_map[data_set_name]
            key = (data_set_name,level)
            
            for test in tests:
                p = multiprocessing.Process(target=run_test_and_save, args=(results_dict[key][test["name"]],
                    key,
                    test["name"],
                    progress_lock,
                    number_of_tests_finished,
                    number_of_tests,
                    data_set_name,
                    level,
                    test,
                    collection.get_set(data_set_name, level),
                    num_of_devices,
                    pct_data_per_device,
                    perc_iid_per_device,
                    group_per_device,
                    number_of_rounds,
                    num_devices_per_group_per_round,))
                processes.append(p)
                p.start()
                
                # results_dict[key][test["name"]] = acc
            
    for process in processes:
        process.join()
    
    return results_dict


def create_tests():
    input_len = 2 # this is the length of each data point
    som_params = { 
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
    random.seed(som_params['SEED'])
    
    tests = [
        # {
        #     "name": "SOM (Fed)",
        #     "server": partial(SOM_server, params=som_params), 
        #     "device": partial(SOM_Device, params=som_params),
        # },
    ]

    k_means_servers = [{
        "name": "",
        "server": CURE_Server,
    },
    {
        "name": " Carry",
        "server": CURE_Server_Carry
    },
    {
        "name": " Keep",
        "server": CURE_Server_Keep
    }
    ]
    for k in k_means_servers:
        # Cure
        params = {
            # "device min clusters": {"N_CLUSTERS": 3,
            #     "MAX_ITERS": 100,
            #     "N_INITS": 10,
            #     "METRIC": "euclidean",
            #     "TOLERANCE": None},
            "device mid clusters": {"N_CLUSTERS": 10,
                "MAX_ITERS": 100,
                "N_INITS": 10,
                "METRIC": "euclidean",
                "TOLERANCE": None},
            # "device max clusters": {"N_CLUSTERS": 20,
            #     "MAX_ITERS": 100,
            #     "N_INITS": 10,
            #     "METRIC": "euclidean",
            #     "TOLERANCE": None}
        }
        cure_params = {
            "server min rep points": {"N_CLUSTERS": 3,
                "N_REP_POINTS": 1,
                "COMPRESSION": 0.05},
            # "server mid rep points": {"N_CLUSTERS": 3,
            #     "N_REP_POINTS": 3,
            #     "COMPRESSION": 0.05},
            # "server max rep points": {"N_CLUSTERS": 3,
            #     "N_REP_POINTS": 10,
            #     "COMPRESSION": 0.05}
        }
        for device_params_name, device_params in params.items():
            for server_params_name, server_params in cure_params.items():
                tests.append({
                    "name": "KMeans" + k["name"] + " - " + device_params_name + " - " + server_params_name + " (Fed)",
                    "server": partial(k["server"], cure_params=server_params), 
                    "device": partial(K_Means_Device, params=device_params),
                })

    return tests


import csv
def save_test_results(results):
    with open('results.csv', 'w', newline='') as csvfile:
        r_file = csv.writer(csvfile, delimiter=',')
        r_file.writerow(["Data Set (ARI)", "Type"] + list(list(results.values())[0].keys()))
        for data_pair,pair_results in results.items():
            r_file.writerow(list(data_pair) + [i["end"].value for i in list(pair_results.values())])

def plot_rounds(results):
    fig, axs = plt.subplots(2)
    i = 0
    for data_pair,pair_results in results.items():
        file_name = ' '.join(data_pair)
        axs[i].set_title(file_name)
        for name,data in list(pair_results.items()):
            x = list(range(1, len(data["rounds"])+1))
            y = data["rounds"]
            # print("plot", name)
            # print("x:", x)
            # print("y:", y)
            axs[i].plot(x,y,label=name)
        i+=1
        plt.legend(loc=0,bbox_to_anchor=(1,0.5))
        # plt.savefig('{}.png'.format(file_name), bbox_inches='tight')
    plt.savefig('rounds-algs.png', bbox_inches='tight')


def main():
    cure()

def run_all_tests():
    tests = create_tests()
    results = run_tests(tests,
        data_sets = collection.data_sets_names[-2:], levels = collection.noice_levels[-2:])
    save_test_results(results)

def evaluate_accuracy_evolution():
    global ENABLE_ROUND_PROGRESS_PLOT
    ENABLE_ROUND_PROGRESS_PLOT = True
    tests = create_tests()

    results = run_tests(tests,
        data_sets = collection.data_sets_names[-1:], 
        levels = collection.noice_levels[-3:-1],
        number_of_rounds = 8)
    
    # print("round acc", results)

    if ENABLE_ROUND_PROGRESS_PLOT: 
        plot_rounds(results)


if  __name__ == "__main__":
    import time
    starttime = time.time()
    
    # run_all_tests()
    evaluate_accuracy_evolution()

    delta = time.time() - starttime
    print('That took {} seconds / {} minutes'.format(delta, delta/60))