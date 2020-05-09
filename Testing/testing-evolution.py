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

from Testing.data import DataSetCollection, DataSet, DataSampler
from Testing.devices import Device, Server
import Testing.analysis as analysis

from Testing.config import layers,collection


def asymptotic_decay(learning_rate, t, max_iter):
    return learning_rate / (1+t/(max_iter/2))


def custom(
        suite,
        server_alg_class, device_alg_class,
        num_devices_per_group_per_round = [{0: 10, 1:10, 2:10}] * 10
    ):
    round_accs = suite.run_rounds_with_accuracy(num_devices_per_group_per_round)

    acc = suite.accuracy()
    return (acc,round_accs)


class MultiProcessing:
    def __init__(self, MULTIPROCESSED = True):
        self.MULTIPROCESSED = MULTIPROCESSED and False
        self.manager = multiprocessing.Manager()

    def run_single(self, *, 
            result_dict, suite,test, 
            progress_lock,number_of_tests,number_of_tests_finished
        ):
        name = suite["name"] + ": " + test["name"]
        
        d = DeviceSuite(suite, test)
        result_dict["rounds"].extend(d.run_rounds_with_accuracy())
        result_dict["end"].value = d.accuracy()
        
        with progress_lock:
            number_of_tests_finished.value += 1
            print('Progress: {}/{} Complete \t {} \t {}'.format(number_of_tests_finished.value, number_of_tests, result_dict["end"].value, name) )
    
    def run(self, construction, **kwargs):
        # Take a list of process specs and run in parallel
        (number_of_tests, specs, results_dict) = construction
        processes = []
        number_of_tests_finished = multiprocessing.Value('i', 0)

        for spec in specs:
            kwargs = spec
            kwargs["progress_lock"] = multiprocessing.Lock()
            kwargs["number_of_tests"] = number_of_tests
            kwargs["number_of_tests_finished"] = number_of_tests_finished

            p = multiprocessing.Process(target=self.run_single, kwargs=kwargs)
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

    def constructProcessTests(self, suites, tests,**kwargs):
        # Create a list of process specs
        number_of_tests = 0
        specs = []

        results_dict = OrderedDict()
        for suite in suites:
            key = suite["name"]
            results_dict[key] = OrderedDict()
            
            for test in tests:
                number_of_tests += 1
                results_dict[key][test["name"]] = self.createResultObjItem()
                specs.append(dict(
                        result_dict = results_dict[key][test["name"]],
                        suite = suite,
                        test = test
                    ))
        return number_of_tests, specs, results_dict


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
        l = len(r)
        for t in r:
            timeline[t] = {}
            for key in timeline[time].keys():
                delta_val = (timeline[time][key] - timeline[last_time][key]) / (l+1)
                timeline[t][key] = timeline[t-1][key] + round(delta_val)
        last_time = time
    return timeline
def apply_down(obj,*args,**kwargs):
    for key,value in obj.items():
        if callable(value):
            obj[key] = value(*args, **kwargs)
        # if type(obj[key]) is dict:
        #     obj[key] = apply_down(obj[key],*args,**kwargs)
    return obj
def create_suites(layers):
    suites = []
    for s in layers["suites"]:
        for data_set_name in s["datasets"]:
            levels = layers["noice"](data_set_name)
            for level in levels:
                (data,labels) = collection.get_set(data_set_name, level)
                dataset = DataSet(data, labels)    
                for transition in s["transition"]:
                    suite = apply_down(dict(s), dataset, round(s["pct_data_per_device"] * s["devices"]))
                    suite["dataset"] = dataset
                    if transition: 
                        suite["name"] += " Transitioned"
                        suite["timeline"] = smooth(suite["timeline"])
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
            suite,
            test
        ):
        self.suite = suite
        self.test = test
        counter = 0
        for group, f in self.suite["groups"].items():
            self.groups[group] = [Device(self.test["device"], f(), id_num=(counter*100000+i)) for i in range(self.suite["devices"])]
            counter += 1
        
        self.server = Server(self.test["server"], self.groups)
    
    def run_rounds(self):
        for _round,groups in self.suite["timeline"].items():
            self.server.run_round(groups)

    def run_rounds_with_accuracy(self):
        round_accs = []
        for _round,groups in self.suite["timeline"].items():
            self.server.run_round(groups)
            round_accs.append(self.accuracy())
        return round_accs

    def accuracy(self):
        data = self.suite["dataset"].data
        labels = self.suite["dataset"].labeled_data
        pred_labels = self.server.classify(data)
        return metrics.adjusted_rand_score(labels, pred_labels)



def evaluate_accuracy_evolution():
    suites = create_suites(layers)
    tests = create_tests(layers)

    # print("suites", suites)
    # print("testsc", tests)

    m = MultiProcessing(MULTIPROCESSED)
    results = m.constructAndRun(suites,tests)
    
    analysis.save_test_results(results)

    if ENABLE_ROUND_PROGRESS_PLOT: 
        analysis.plot_rounds(results)


if  __name__ == "__main__":
    analysis.calculate_time(evaluate_accuracy_evolution())