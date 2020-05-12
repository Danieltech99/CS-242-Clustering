# CS-242-Clustering

## To Run a Test Suite
Edit `Testing/config.py` to include the benchmarks and algorithms to run. Then run:

```
python3 -m Testing.testing
```

## Running an Algorithm 

For all algorithms, first create an instance of a "device algorithm" with data and a "server algorithm". For example:
```
from Algorithms import gossip_k_means
alg_instance = gossip_k_means.gossip_KMeans_Device(data, params)
server_instance = gossip_k_means.gossip_KMeans_server()
```

Then a round can be run on the device as follows:
```
alg_instance.run_on_device()
```

The round output can be communicated with the central server and acted upon:
```
report_for_server = alg_instance.get_report_for_server()
server_instance.update_server()
server_instance.run_on_server()
```

Then the server report can be transmitted to next round devices as follows:
```
report_from_server = server_instance.get_report_for_devices()
alg_instance.update_device(report_from_server)
```

Rinse and repeat!



## Folder: Data
Includes files regarding data.

## Folder: Algorithms 
Includes files for each algorithm.

## Folder: Testing
Includes files for test suites.
Includes data partitioning and device suites.
