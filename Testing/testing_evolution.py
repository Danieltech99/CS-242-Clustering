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

random.seed(242) 
np.random.rand(242)

ENABLE_ROUND_PROGRESS_PLOT = False
MULTIPROCESSED = True


class MultiProcessing:
    def __init__(self, MULTIPROCESSED=True):
        self.MULTIPROCESSED = MULTIPROCESSED and False
        self.manager = multiprocessing.Manager()

    def run_single(self, *,
                   result_dict, suite, test,
                   progress_lock, number_of_tests, number_of_tests_finished
                   ):
        name = suite["name"] + ": " + test["name"]

        d = DeviceSuite(suite, test)
        result_dict["rounds"].extend(d.run_rounds_with_accuracy())
        result_dict["end"].value = d.accuracy()

        with progress_lock:
            number_of_tests_finished.value += 1
            print('Progress: {}/{} Complete \t {} \t {}'.format(number_of_tests_finished.value, number_of_tests, result_dict["end"].value, name))

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
        test
    ):
        self.suite = suite
        self.test = test
        counter = 0
        for group, f in self.suite["groups"].items():
            self.groups[group] = [Device(self.test["device"], f(), id_num=(
                counter*100000+i)) for i in range(self.suite["devices"])]
            counter += 1

        self.server = Server(self.test["server"], self.groups)

    def run_rounds(self):
        for _round, groups in self.suite["timeline"].items():
            self.server.run_round(groups)

    def run_rounds_with_accuracy(self):
        round_accs = []
        for _round, groups in self.suite["timeline"].items():
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
    
    m = MultiProcessing(MULTIPROCESSED)
    results = m.constructAndRun(suites, tests)

    analysis.save_test_results(results)

    if ENABLE_ROUND_PROGRESS_PLOT:
        analysis.plot_rounds(results)


if __name__ == "__main__":
    analysis.calculate_time(evaluate_accuracy_evolution())