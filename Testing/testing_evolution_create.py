from functools import partial
from Testing.data import DataSet
from Testing.config import collection
import time

def smooth(timeline,rounds,transition,devices_per_round):
    # add missing keys
    timeline = dict(timeline)

    # add last round
    t_keys = list(timeline.keys())
    t_max = max(t_keys)
    timeline[rounds - 1] = dict(timeline[t_max])

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
            for key,val in timeline[time].items():
                delta_val = (timeline[time][key] - timeline[last_time][key]) / (l+1)
                amount = timeline[t-1][key] + round(delta_val) if transition else timeline[t-1][key]
                timeline[t][key] = round(amount)

        last_time = time
    for t,v in timeline.items():
        missing = devices_per_round - sum(timeline[t].values())
        if missing >= 1:
            for key in timeline[t]:
                if timeline[t][key] >= 1:
                    timeline[t][key] += missing
                    break
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
                (data,labels,true_labels) = collection.get_set_true(data_set_name, level)
                dataset = DataSet(data, labels,true_labels)    
                pop_size = dataset.data.shape[0]
                for transition in s["transition"]:
                    suite = apply_down(dict(s), dataset, round(s["pct_data_per_device"] * pop_size))
                    suite["dataset"] = dataset
                    if transition: 
                        suite["name"] += " Transitioned"
                    suite["timeline"] = smooth(suite["timeline"],suite["rounds"],transition, suite["devices_per_round"])
                    # suite["name"] += " - {} ({})".format(data_set_name, level)
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
                            "device_multi": alg.get("device_multi",1),
                            "server": partial(alg["server"]["class"], **{server_params_key: server_params}),
                            "device": partial(alg["device"]["class"], **{device_params_key: device_params})
                        })
    return tests
