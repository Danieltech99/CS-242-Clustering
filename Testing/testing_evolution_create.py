from functools import partial
from Testing.data import DataSet
from Testing.config import collection

def smooth(timeline,rounds,transition):
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
            for key in timeline[time].keys():
                delta_val = (timeline[time][key] - timeline[last_time][key]) / (l+1)
                timeline[t][key] = timeline[t-1][key] + round(delta_val) if transition else timeline[t-1][key]
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
                pop_size = dataset.get_indices().size
                for transition in s["transition"]:
                    suite = apply_down(dict(s), dataset, round(s["pct_data_per_device"] * pop_size))
                    suite["dataset"] = dataset
                    if transition: 
                        suite["name"] += " Transitioned"
                    suite["timeline"] = smooth(suite["timeline"],suite["rounds"],transition)
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
