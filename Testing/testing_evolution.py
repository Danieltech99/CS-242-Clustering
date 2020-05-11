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

random.seed(242) 
np.random.rand(242)

ENABLE_ROUND_PROGRESS_PLOT = True
MULTIPROCESSED = True


class MultiProcessing:
    def __init__(self, MULTIPROCESSED=True):
        self.MULTIPROCESSED = MULTIPROCESSED
        self.manager = multiprocessing.Manager()

    def run_single(self, *,
                   result_dict, suite, test,
                   progress_lock, number_of_tests, number_of_tests_finished
                   ):
        name = suite["name"] + ": " + test["name"]

        d = DeviceSuite(suite, test, name = name)
        if ENABLE_ROUND_PROGRESS_PLOT:
            result_dict["rounds"].extend(d.run_rounds_with_accuracy())
        result_dict["end"].value = d.accuracy()
        d.complete()

        with progress_lock:
            number_of_tests_finished.value += 1
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

            p = multiprocessing.Process(target=self.run_single, kwargs=kwargs)
            processes.append(p)
            p.start()
            if not self.MULTIPROCESSED:
                p.join()

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

    def constructProcessTests(self, suites, tests, **kwargs):
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
                    result_dict=results_dict[key][test["name"]],
                    suite=suite,
                    test=test
                ))
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
        self.suite = suite
        self.test = test
        self.name = name
        counter = 0

        for group, f in self.suite["groups"].items():
            self.groups[group] = [Device(self.test["device"], f()["sample"](), id_num=(
                counter*100000+i)) for i in range(self.suite["devices"])]

        self.server = Server(self.test["server"], self.groups)

        rounds = len(self.suite["timeline"])
        max_devices = 0
        for x in self.suite["timeline"].values():
            max_devices = max(sum(x.values()),max_devices)
        self.server.define_plotter(rounds = rounds, devices=max_devices)

    def run_rounds(self):
        for _round, groups in self.suite["timeline"].items():
            self.server.run_round(groups)

    def get_population_of_round(self, target_round):
        indicies = None
        group_set = set()
        for _round, groups in self.suite["timeline"].items():
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
        data = self.suite["dataset"].get_data_for_indicies(indicies)
        return data

    def plot_round(self, target_round):
        data = self.get_population_of_round(target_round)
        self.sub_accuracy(data)
        if self.server.PLOT:
            self.server.plotter.plot_a(int(target_round), data, self.last_pred)

    def run_rounds_with_accuracy(self, return_rounds = True):
        # data = self.suite["dataset"].data
        round_accs = []
        for _round, groups in self.suite["timeline"].items():
            self.server.run_round(groups, int(_round))
            if return_rounds:
                round_accs.append(self.accuracy())
            self.plot_round(_round)
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
    def sub_accuracy(self, data):
        pred_labels = self.server.classify(data)
        self.last_pred = pred_labels


def evaluate_accuracy_evolution():
    suites = create_suites(layers)
    tests = create_tests(layers)

    l = len(suites) * len(tests)
    max_proc = 2
    split_n = math.floor(l/(l/max_proc))
    split_n = math.ceil(split_n / len(tests))
    partitions = [suites[i:i + split_n] for i in range(0, len(suites), split_n)]
    sets = len(partitions)
    print("Running {} Sets of Tests".format(sets))

    m = MultiProcessing(MULTIPROCESSED)
    results = {}
    for i, part in enumerate(partitions):
        res = analysis.calculate_time(m.constructAndRun)(part, tests)
        # results.update(res)
        print("Progress: {} of {} Complete".format(i+1, sets))

    # analysis.save_test_results(results)

    # if ENABLE_ROUND_PROGRESS_PLOT:
    #     analysis.plot_rounds(results)


if __name__ == "__main__":
    analysis.calculate_time(evaluate_accuracy_evolution)()
