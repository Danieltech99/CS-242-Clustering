#!/usr/bin/python3
from abc import ABC,abstractmethod 
from collections import OrderedDict 
import threading
import multiprocessing 
import random


# Data Sets (x4)
# 5 levels of noice per data set
# 10,000 data points
import numpy as np
import math
from functools import partial
import matplotlib.pyplot as plt
from sklearn import metrics

random.seed(242)
np.random.rand(242)

ENABLE_PLOTS = False
ENABLE_PRINTS = False
ENABLE_PROGRESS = True
ENABLE_ROUND_PROGRESS_PLOT = False
MULTIPROCESSED = True

CURRENT_FILE_NAME = None

from Testing.data import DataSetCollection, DataSet, DataSampler
from Testing.devices import Device, Server
import Testing.analysis as analysis
from Algorithms.k_means import CURE_Server,K_Means_Device,CURE_Server_Carry,CURE_Server_Keep, KMeans_Server, KMeans_Server_Carry, KMeans_Server_Keep

collection = DataSetCollection()

def asymptotic_decay(learning_rate, t, max_iter):
    return learning_rate / (1+t/(max_iter/2))



class DeviceSuite:
    server = None
    devices = []
    groups = {}

    def __init__(
            self, 
            server_alg_class, 
            device_alg_class, 
            dataset_class_instance, 
            num_devices, 
            pct_data_per_device, 
            perc_iid_per_device, 
            group_per_device
        ):
        self.dataset_class_instance = dataset_class_instance
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

    def run_rounds_with_accuracy(self, num_devices_per_group_per_round):
        round_accs = []
        run_num = 1
        for num_devices_per_group in num_devices_per_group_per_round:
            self.server.run_round(num_devices_per_group)
            if ENABLE_ROUND_PROGRESS_PLOT or ENABLE_PRINTS: 
                acc = self.accuracy(run_num)
                round_accs.append(acc)
            if ENABLE_PRINTS: print("Accuracy: ", acc)
            run_num += 1
        return round_accs

    def accuracy(self, run_num = 1):
        data = self.dataset_class_instance.data
        labels = self.dataset_class_instance.labeled_data
        pred_labels = self.server.classify(data)
        if ENABLE_PLOTS:
            plt.scatter(data[:, 0], data[:, 1], c=pred_labels, s=1)
            # plt.show()
            plt.savefig('Plots/' + CURRENT_FILE_NAME + "-round-" + str(run_num) + '.png')
        return metrics.adjusted_rand_score(labels, pred_labels)








def custom(
        suite,
        server_alg_class, device_alg_class,
        num_devices_per_group_per_round = [{0: 10, 1:10, 2:10}] * 10
    ):
    round_accs = suite.run_rounds_with_accuracy(num_devices_per_group_per_round)

    acc = suite.accuracy()
    return (acc,round_accs)


def run_test_and_save(
        *,
        result_dict,
        progress_lock,
        number_of_tests_finished,
        number_of_tests,
        data_set_name,
        level,
        test,
        **kwargs
    ): 
    # Adds prints and progress to testing for multiprocessing
    global CURRENT_FILE_NAME
    if ENABLE_PRINTS: print(test["name"] + ":", data_set_name + "-" + level)
    CURRENT_FILE_NAME = test["name"] + ":" + data_set_name + "-" + level
    (result_dict["end"].value,round_accs) = custom(
            server_alg_class = test["server"], 
            device_alg_class = test["device"],
            **kwargs
        )
    if ENABLE_ROUND_PROGRESS_PLOT: 
        result_dict["rounds"].extend(round_accs)
    if ENABLE_PROGRESS: 
        with progress_lock:
            number_of_tests_finished.value += 1
            print('Progress: {}/{} Complete \t {} \t {}'.format(number_of_tests_finished.value, number_of_tests, result_dict["end"].value, test["name"] + ":" + data_set_name + "-" + level) )


class MultiProcessing:
    def __init__(self, MULTIPROCESSED = True):
        self.MULTIPROCESSED = MULTIPROCESSED
        self.manager = multiprocessing.Manager()
    
    def run(self, construction, target = run_test_and_save, **kwargs):
        # Take a list of process specs and run in parallel
        (number_of_tests, specs, results_dict) = construction
        processes = []
        number_of_tests_finished = multiprocessing.Value('i', 0)

        for spec in specs:
            kwargs = spec
            kwargs["progress_lock"] = multiprocessing.Lock()
            kwargs["number_of_tests"] = number_of_tests
            kwargs["number_of_tests_finished"] = number_of_tests_finished

            p = multiprocessing.Process(target=target, kwargs=kwargs)
            processes.append(p)
            p.start()
            if not self.MULTIPROCESSED: p.join()

        if self.MULTIPROCESSED:
            for process in processes:
                process.join()

        return results_dict

    def createResultObjItem(self):
        return {
            "end": multiprocessing.Value("d", 0.0, lock=False),
            "rounds": self.manager.list()
        }

    def constructAndRun(self, *args, **kwargs):
        res = self.constructProcessTests(*args, **kwargs)
        return self.run(res, **kwargs)

    def constructProcessTests(self, tests, data_sets = collection.data_sets_names, levels = collection.noice_levels, number_of_rounds    = 4,**kwargs):
        # Create a list of process specs
        number_of_tests = 0
        specs = []

        results_dict = OrderedDict()
        for data_set_name in data_sets:

            count = collection.count_map[data_set_name]

            for level in levels:
                key = (data_set_name,level)
                results_dict[key] = OrderedDict()
                
                for test in tests:
                    number_of_tests += 1
                    results_dict[key][test["name"]] = self.createResultObjItem()
                    specs.append(dict(
                            result_dict = results_dict[key][test["name"]],
                            data_set = collection.get_set(data_set_name, level),
                            # progress_lock = progress_lock,
                            # number_of_tests_finished = number_of_tests_finished,
                            # number_of_tests = number_of_tests,
                            data_set_name = data_set_name,
                            level = level,
                            test = test,
                            # suite = suite
                        ))
        return number_of_tests, specs, results_dict

from config import layers
def smooth(timeline):
    # add missing keys
    timeline = dict(timeline)
    keys = set()
    for time,state in timeline.items():
        keys.update(state.keys())
    defaults = dict((k,0) for k in keys)
    for time,state in timeline.items():
        temp = dict(defaults)
        temp.update(timeline[time])
        timeline[time] = temp
    # smooth states
    last_time = None
    t_keys = list(timeline.keys())
    t_keys.sort()
    for time in t_keys:
        if last_time is None:
            last_time = time
            continue
        r = range(last_time + 1, time)
        print("range", last_time, time)
        l = len(r)
        for t in r:
            timeline[t] = {}
            for key in timeline[time].keys():
                delta_val = (timeline[time][key] - timeline[last_time][key]) / (l+1)
                timeline[t][key] = timeline[t-1][key] + round(delta_val)
        last_time = time
    return timeline
def apply_down(obj,*args,**kwargs):
    for key,value in obj:
        if callable(value):
            obj[key] = value(*args, **kwargs)
        if type(obj[key]) is dict:
            obj[key] = apply_down(obj[key],*args,**kwargs)
    return obj
def create_suites(layers):
    suites = []
    data_sets = levels = layers["datasets"]
    for data_set_name in data_sets:
        levels = layers["noice"](data_set_name)
        for level in levels:
            (data,labels) = collection.get_set(data_set_name, level)
            dataset = DataSet(data, labels)    
            for s in layers["suites"]:
                for transition in s["transition"]:
                    suite = apply_down(s, dataset, s["pct_data_per_device"] * s["devices"])
                    if transition: 
                        suite["name"] += " Transitioned"
                        suite["timeline"] = smooth(s["timeline"])
                    suite["name"] += " - {} ({})".format(data_set_name, level)
                    suites.append(suite)
    return suites
def create_tests(layers):
    tests = []

    for alg in layers["algs"]:
        for server_params_key, server_params_dict in alg["server"]["kwargs"].items():
            for server_param_name, server_params in server_params_dict.items():
                for device_params_key, device_params_dict in alg["device"]["kwargs"].items():
                    for device_param_name, device_params in device_params_dict.items():
                        tests.append({
                            "name": alg["name"] + server_param_name + device_param_name,
                            "server": partial(alg["server"]["class"], **{server_params_key: server_params}),
                            "device": partial(alg["device"]["class"], **{device_params_key: device_params})
                        })
    return tests



class DeviceSuite:
    server = None
    devices = []
    groups = {}

    def __init__(
            self, 
            server_alg_class, 
            device_alg_class,
        ):
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

def run_test_suites(suites,tests):
    for suite in suites:
        for test in tests:



def evaluate_accuracy_evolution():
    global ENABLE_ROUND_PROGRESS_PLOT
    ENABLE_ROUND_PROGRESS_PLOT = True
    tests = create_tests()

    m = MultiProcessing(MULTIPROCESSED)
    results = m.constructAndRun(tests,
        data_sets = collection.data_sets_names[0:1], 
        levels = collection.noice_levels[:-1],
        number_of_rounds = 8,
        target=run_test_and_save)
    
    # print("round acc", results)
    analysis.save_test_results(results)

    if ENABLE_ROUND_PROGRESS_PLOT: 
        analysis.plot_rounds(results)


if  __name__ == "__main__":
    analysis.calculate_time(evaluate_accuracy_evolution())