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



def custom(
        server_alg_class, device_alg_class,
        data_set = collection.get_set("blobs", "vhigh"),
        num_of_devices = 1000,
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

    num_of_devices = 1000
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
                # p.join()
                
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

    k_means_servers = [
    #     {
    #     "name": "",
    #     "server": CURE_Server,
    # },
    # {
    #     "name": " Carry",
    #     "server": CURE_Server_Carry
    # },
    # {
    #     "name": " Keep",
    #     "server": CURE_Server_Keep
    # },
    {
        "name": "KMeans Server",
        "server": KMeans_Server
    },
    {
        "name": "KMeans Server Carry",
        "server": KMeans_Server_Carry
    },
    {
        "name": "KMeans Server Keep",
        "server": KMeans_Server_Keep
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
    fig, axs = plt.subplots(len(results.keys()))
    i = 0
    print("results keys", results.keys())
    print("results item keys", results[list(results.keys())[0]].keys())
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
        data_sets = collection.data_sets_names, levels = collection.noice_levels)
    save_test_results(results)

def evaluate_accuracy_evolution():
    global ENABLE_ROUND_PROGRESS_PLOT
    ENABLE_ROUND_PROGRESS_PLOT = True
    tests = create_tests()

    results = run_tests(tests,
        data_sets = collection.data_sets_names[0:2], 
        levels = collection.noice_levels[:-1],
        number_of_rounds = 8)
    
    # print("round acc", results)
    save_test_results(results)

    if ENABLE_ROUND_PROGRESS_PLOT: 
        plot_rounds(results)


if  __name__ == "__main__":
    import time
    starttime = time.time()
    
    # run_all_tests()
    evaluate_accuracy_evolution()

    delta = time.time() - starttime
    print('That took {} seconds / {} minutes'.format(delta, delta/60))