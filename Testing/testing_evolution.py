#!/usr/bin/python3
import threading
import multiprocessing
import random
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from Testing.testing_evolution_create import create_suites, create_tests
from Testing.config import layers
import Testing.analysis as analysis
from Testing.devices import Device, Server
import time
import math
from Algorithms.k_means import KMeans_Server
import copy
import json

random.seed(242) 
np.random.rand(242)
NON_FED_KEY = "Traditional K-Means"

ENABLE_ROUND_PROGRESS_PLOT = True
PLOT = False
RUN_NON_FED = True
MULTIPROCESSED = True
MAX_PROC = 40


class MultiProcessing:
    def __init__(self, MULTIPROCESSED=True):
        self.MULTIPROCESSED = MULTIPROCESSED
        self.manager = multiprocessing.Manager()

    def convert(self, res):
        o = {}
        for k,v in res.items():
            o[k] = {}
            for k2,v2 in v.items():
                o[k][k2] = {
                    "end": v2["end"].value,
                    "rounds": list(v2["rounds"])
                }
        return o

    def run_single(self, *,
                   result_dict, suite, test,
                   progress_lock, number_of_tests, number_of_tests_finished,res
                   ):
        name = suite["name"] + ": " + test["name"]

        d = DeviceSuite(suite, test, name = name)
        if ENABLE_ROUND_PROGRESS_PLOT:
            result_dict["rounds"].extend(d.run_rounds_with_accuracy())
        else:
            d.run_rounds_with_accuracy()
        result_dict["end"].value = d.accuracy()
        d.complete()

        with progress_lock:
            number_of_tests_finished.value += 1
            o = self.convert(res)
        with open('results-gossip-specialized-inter2.json', 'w') as outfile:
            json.dump(o, outfile)
            
        print('\tProgress: {}/{} Complete \t {} \t {}'.format(number_of_tests_finished.value, number_of_tests, result_dict["end"].value, name))
        print('\tProgress: {}/{} Complete \t {} \t {}'.format(number_of_tests_finished.value, number_of_tests, result_dict["end"].value, name))
        print('\tProgress: {}/{} Complete \t {} \t {}'.format(number_of_tests_finished.value, number_of_tests, result_dict["end"].value, name))

    def run(self, construction, **kwargs):
        # Take a list of process specs and run in parallel
        (number_of_tests, specs, results_dict) = construction
        processes = []
        number_of_tests_finished = multiprocessing.Value('i', 0)
        print("\tRunning {} Tests".format(number_of_tests))

        for spec in specs:
            kwargs = spec
            kwargs["progress_lock"] = multiprocessing.Lock()
            kwargs["number_of_tests"] = number_of_tests
            kwargs["number_of_tests_finished"] = number_of_tests_finished
            kwargs["res"] = results_dict

            if not self.MULTIPROCESSED:
                self.run_single(**kwargs)
            else:
                p = multiprocessing.Process(target=self.run_single, kwargs=kwargs)
                processes.append(p)
                p.start()

        # self.run_non_fed(results_dict)

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
        res = analysis.calculate_time(self.constructProcessTests)(*args, **kwargs)
        return self.run(res, **kwargs)

    def non_fed_k_means(self, suite):
        _, kmeans = KMeans_Server.find_optimal_k_silhouette(suite["dataset"].data)
        pred_labels = kmeans.predict(suite["dataset"].data)

        labels = suite["dataset"].true_labels
        return metrics.adjusted_rand_score(labels, pred_labels)

    def run_non_fed(self,results_dict, suites = None, cond = False, create = False):
        if suites is None: suites = self.suites
        for i,suite in enumerate(suites):
            key = suite["name"]
            if create: results_dict[key] = {}
            if suite["non_fed"] and (RUN_NON_FED or cond):
                results_dict[key][NON_FED_KEY] = self.createResultObjItem()
                a = self.non_fed_k_means(suite)
                results_dict[key][NON_FED_KEY]["end"].value = a
                results_dict[key][NON_FED_KEY]["rounds"].extend([a] * suite["rounds"])
                print("Finished Traditional K Means on Suite {}".format(i))
        return results_dict

    def constructProcessTests(self, suites, tests, current = None, **kwargs):
        # Create a list of process specs
        number_of_tests = 0
        specs = []

        results_dict = OrderedDict()
        for suite in suites:
            key = suite["name"]
            results_dict[key] = OrderedDict()

            added = 0
            for test in tests:
                if current is None or key not in current or len(current[key][test["name"]]["rounds"]) == 0:
                    if current is not None and key not in current:
                        results_dict[key] = {}
                    added += 1
                    results_dict[key][test["name"]] = self.createResultObjItem()
                    specs.append(dict(
                        result_dict=results_dict[key][test["name"]],
                        suite=suite,
                        test=test
                    ))
            if added == 0:
                del results_dict[key]
            number_of_tests += added

        self.suites = suites
        return number_of_tests, specs, results_dict



class DeviceSuite:
    server = None
    devices = []
    groups = {}

    def __init__(
        self,
        suite,
        test,
        name = None
    ):
        self.suite = copy.deepcopy(suite)
        self.test = test
        self.name = name
        counter = 0

        for group, f in self.suite["groups"].items():
            self.groups[group] = [Device(self.test["device"], f()["sample"](), id_num=(i)) for i in range(self.suite["devices"])]

        self.server = Server(self.test["server"], self.groups)

        rounds = len(self.suite["timeline"])
        max_devices = 0
    
        if "device_multi" in self.test:
            for t,groups in self.suite["timeline"].items():
                for group, num_devices in groups.items():
                    self.suite["timeline"][t][group] = num_devices * self.test["device_multi"]
        
        for x in self.suite["timeline"].values():
            max_devices = max(sum(x.values()),max_devices)

        # print("timeline", name, max_devices, self.suite["timeline"])
        
        if PLOT:
            self.server.define_plotter(rounds = rounds, devices=max_devices)

    def run_rounds(self):
        for _round, groups in sorted(self.suite["timeline"].items()):
            self.server.run_round(groups)
            print("\t\tCompleted round {}/{}".format(_round+1, len(self.suite["timeline"])))

    def get_population_of_round(self, target_round):
        indicies = None
        group_set = set()
        for _round, groups in sorted(self.suite["timeline"].items()):
            if _round <= target_round:
                for g,val in groups.items():
                    if val > 0 and g not in group_set:
                        group_set.add(g)
                        f = self.suite["groups"][g]
                        _indicies = f()["population"]()
                        if indicies is None:
                            indicies = _indicies
                        else:
                            indicies = np.concatenate((indicies, _indicies),0)
        indicies = np.unique(indicies)
        data,labels = self.suite["dataset"].get_data_for_indicies(indicies), self.suite["dataset"].get_labels_for_indicies(indicies)
        return data,labels

    def plot_round(self, target_round):
        data,labels = self.get_population_of_round(target_round)
        acc = self.sub_accuracy(data,labels)
        if self.server.PLOT:
            self.server.plotter.plot_a(int(target_round), data, self.last_pred)
        return acc

    def run_rounds_with_accuracy(self, return_rounds = True):
        # data = self.suite["dataset"].data
        round_accs = []
        for _round, groups in sorted(self.suite["timeline"].items()):
            self.server.run_round(groups, int(_round))
            if return_rounds:
                round_accs.append(self.plot_round(_round))
            print("\t\tCompleted round {}/{}".format(_round+1, len(self.suite["timeline"])))
        return round_accs
    
    def complete(self):
        if self.server.PLOT:
            self.server.plotter.save(self.name)

    last_pred = None
    def accuracy(self):
        data = self.suite["dataset"].data
        labels = self.suite["dataset"].true_labels
        pred_labels = self.server.classify(data)
        self.last_pred = pred_labels
        return metrics.adjusted_rand_score(labels, pred_labels)
    def sub_accuracy(self, data,labels):
        pred_labels = self.server.classify(data)
        self.last_pred = pred_labels
        return metrics.adjusted_rand_score(labels, pred_labels)


def u(obj1,obj2):
    o = copy.deepcopy(obj1)
    for k,v in obj2.items():
        if k not in o: o[k] = {}
        for k2,v2 in v.items():
            o[k][k2] = v2
            # for k3,v3 in v2.items():
    return o

def evaluate_accuracy_evolution():
    suites = analysis.calculate_time(create_suites)(layers)
    tests = analysis.calculate_time(create_tests)(layers)

    l = len(suites) * len(tests)
    max_proc = MAX_PROC
    split_n = math.floor(l/(l/max_proc))
    split_n = math.ceil(split_n / len(tests))
    partitions = [suites[i:i + split_n] for i in range(0, len(suites), split_n)]
    sets = len(partitions)
    print("Running {} Sets of Tests".format(sets))

    current = None
    # with open('results-updated.json') as f:
    #     current = json.load(f)

    m = MultiProcessing(MULTIPROCESSED)
    results = {}
    for i, part in enumerate(partitions):
        res = analysis.calculate_time(m.constructAndRun)(part, tests, current = current)
        results.update(res)
        print("Progress: {} of {} Complete".format(i+1, sets))

    o = m.convert(results)
    with open('results-gossip-specialized-new.json', 'w') as outfile:
        json.dump(o, outfile)
    with open('results-gossip-specialized-updated.json', 'w') as outfile:
        if current is None: current = {}
        json.dump(u(current,o), outfile)

    # analysis.save_test_results(results)

    # if ENABLE_ROUND_PROGRESS_PLOT:
    #     analysis.calculate_time(analysis.plot_rounds)(results)


def run_non_fed_and_save():
    suites = analysis.calculate_time(create_suites)(layers)
    tests = analysis.calculate_time(create_tests)(layers)

    current = None
    with open('results.json') as f:
        current = json.load(f)

    m = MultiProcessing(MULTIPROCESSED)
    results = {}
    results = m.run_non_fed(results, suites, True, True)

    o = m.convert(results)
    with open('results-traditional-new.json', 'w') as outfile:
        json.dump(o, outfile)
    with open('results-traditional-updated.json', 'w') as outfile:
        json.dump(u(current,o), outfile)


if __name__ == "__main__":
    analysis.calculate_time(evaluate_accuracy_evolution)()

    # with open('results-updated.json') as f:
    #     current = json.load(f)
    
    # # with open('_json_pieces/results-gossip.json') as f:
    # with open('results-gossip-specialized-new.json') as f:
    #     o = json.load(f)

    # with open('results-updated.json', 'w') as outfile:
    #     json.dump(u(current, o), outfile)
    
    # with open('results-onlineness-2-new.json') as f:
    #     o2 = json.load(f)

    # with open('results-updated.json', 'w') as outfile:
    #     json.dump(u(u(current,o2), o), outfile)
