# CS-242-Clustering

## To Run a Test Suite
Edit `Testing/config.py` to include the benchmarks and algorithms to run. Then run:

```
python3 -m Testing.testing
```


#### Flags

DO NOT RUN WITHOUT READING!

Default state is set to >50 tests.

This will run the tests in parallel. To custimize settings like multiprocessing (MULTIPROCESSED), estimated max simultaneous processes (MAX_PROC), plot, and more, change the flags in 'Testing/testing_evolution.py'.

ENABLE_ROUND_PROGRESS_PLOT  - Plot scatter of each device per round.
RUN_NON_FED                 - Run Traditional K Means for each Scenario from config.py
MULTIPROCESSED              - Run with one process per scenario test
MAX_PROC                    - Estimated max simultaneous processes

#### Specifying Scenarios/Tests

To specify which tests to run, go to 'Testing/config.py' and comment out any elements in the layers["suites"] or layers["algs"] list (make sure to maintain proper syntax). The items left in the lists will be used to run combinations of a suite with an algorithm agains the testing framework.

The algs list specifies the different algorithms being compared.

The suites are different test scenarios like baseline, Bias of Devices, Non-Homogeneous Cluster Partitioning, and Dynamic Distributions of Data.

To run tests use the command `python(3) -m Testing.testing_evolution`



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

## Folder: Figures
Includes line plots for analysis.

## Folder: Data
Includes scatter plots for scenarios.

## Folder: Algorithms 
Includes files for each algorithm.

gossip_k_means.py   - Gossip Distributed Decentralized k-means Algorithm
nf-algs.py          - Traditional Clustering Algorithms
k_means.py          - Federated k-means Server (Server and Device Classes)
som.py          - Federated Self Organizing Maps Server (Server and Device Classes)

## Folder: Testing
Includes files for test suites.
Includes data partitioning and device suites.


