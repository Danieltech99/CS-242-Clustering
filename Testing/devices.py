import numpy as np
import random
random.seed(242)
np.random.rand(242)
# Algorithm Interfaces 

from Testing.analysis import MajorPlotter

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
    
    def set_id(self, id_num):
        # if alg has update_id method, call it
        set_id_number = getattr(self.alg, "set_id_number", None)
        if callable(set_id_number):
            set_id_number(id_num)
    
    def run(self):
        self.alg.run_on_device()

    def report_back_to_server(self):
        return self.alg.get_report_for_server()

    def update(self, update_data):
        self.alg.update_device(update_data)


class Server:
    device_groups = []
    PLOT = False

    def classify(self, data):
        return self.alg.classify(data)

    def __init__(self, alg_server_class, device_groups):
        self.alg = alg_server_class()
        self.device_groups = device_groups
    
    def define_plotter(self, rounds, devices):
        self.PLOT = True
        self.plotter = MajorPlotter(rounds, devices)
    
    def run_round(self, num_devices_per_group_dict, round_num = None):
        if round_num is None:
            self.PLOT = False
        devices = []
        for group,num_of_devices in num_devices_per_group_dict.items():
            # print("picking", num_of_devices, "from ", len(self.device_groups[group]))
            devices += random.sample(self.device_groups[group],int(num_of_devices))
        # print("picked", len(devices))
        if len(devices) > 120:
            print("PICKED MORE", len(devices))
            print("PICKED MORE", len(devices))
            print("PICKED MORE", len(devices))
            print("PICKED MORE", len(devices))
            print("PICKED MORE", len(devices))
        if len(devices) == 0:
            print("NO DEVICES IN ROUND", num_devices_per_group_dict)
            return 
        reports = self.run_devices(devices, round_num)
        self.update(reports)
        self.run()
        update_for_devices = self.get_reports_for_devices()
        if self.PLOT:
            self.plotter.plot_s(round_num, self.alg.points_read_only, update_for_devices)
        self.send_updates_to_device(devices, update_for_devices)

    def run_devices(self, devices, round_num = 0):
        reports = []
        # print("here ",len(devices))
        for device_num,device in enumerate(devices):
            device.set_id(device_num)
            device.run()
            report = device.report_back_to_server()
            reports.append(report)
            if self.PLOT:
                self.plotter.plot_d(round_num, device_num, device.indicies, report)
        return reports

    def run(self):
        self.alg.run_on_server()
    
    def update(self, reports):
        # aggregates all the device feedback
        self.alg.update_server(reports)
    
    def get_reports_for_devices(self):
        return self.alg.get_reports_for_devices()

    def send_updates_to_device(self, devices, update_for_devices): # report
        for group_name, devices in self.device_groups.items():
            for device in devices:
                device.update(update_for_devices)
