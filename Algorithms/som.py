import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply)
from numpy.linalg import norm
from sklearn.neighbors import NearestCentroid

# Helper functions-- Training
def build_iteration_indexes(data_len, num_iterations,
                            random_generator=None):
    """Selects which data elements will be input. May randomly shuffle"""
    iterations = arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)

    return iterations

def asymptotic_decay(learning_rate, t, max_iter):
    return learning_rate / (1+t/(max_iter/2))

class SOM_Device:
# In chronological order of calling
    def __init__(self, data, params, id_num = None):
        """Initialize SOM member variables"""
        self.id = params['ID'] if id_num is None else id_num
        self._data = data
        self._x = params['X']
        self._y = params['Y']
        self._learning_rate = params['LR']
        self._sigma = params['SIGMA']
        self._max_iters = params['MAX_ITERS']
        self._input_len = params['INPUT_LEN']
        self._random_seed = params['SEED']
        self._random_generator = random.RandomState(self._random_seed)
        self._decay_function = params['DECAY']

        # Initialize weights
        self._weights = self._random_generator.rand(self._x, self._y, self._input_len) * 2 - 1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        # Activation map
        self._activation_map = np.zeros((self._x, self._y))
        self._neigx = arange(self._x)
        self._neigy = arange(self._y)

        # Similarity and distance functions
        neighborhood_function = params['NEIGH_FUNC']
        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        self.neighborhood = neig_functions[neighborhood_function]

        activation_distance = params['ACTIVATION']
        distance_functions = {'euclidean': self._euclidean_distance,
                        'cosine': self._cosine_distance,
                        'manhattan': self._manhattan_distance}
                        
        if activation_distance not in distance_functions:
            msg = '%s not supported. Distances available: %s'
            raise ValueError(msg % (activation_distance,
                                    ', '.join(distance_functions.keys())))

        self._activation_distance = distance_functions[activation_distance]

    def run_on_device(self):
        """Trains the SOM"""
        # print("Running Device ")
        # Sets data indices for training data input
        num_iteration = self._max_iters
        data = self._data
        iterations = build_iteration_indexes(len(data), num_iteration, self._random_generator)

        # Training
        for t, iteration in enumerate(iterations):
            self.update(data[iteration], self.winner(data[iteration]),
                        t, num_iteration)

    def get_report_for_server(self):
        updates_for_server = self._weights
        return updates_for_server

    def update_device(self, reports_from_server):
        self._weights = reports_from_server

    # Helper functions-- Training
    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        self._activation_map = self._activation_distance(x, self._weights)

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += einsum('ij, ijk->ijk', g, x-self._weights)

    # Helper functions-- Neighborhood
    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*pi*sigma*sigma
        ax = exp(-power(self._neigx-c[0], 2)/d)
        ay = exp(-power(self._neigy-c[1], 2)/d)
        return outer(ax, ay)  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        xx, yy = meshgrid(self._neigx, self._neigy)
        p = power(xx-c[0], 2) + power(yy-c[1], 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        return outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)

    # Helper functions-- Distance
    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum+1e-8)

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def _manhattan_distance(self, x, w):
        return linalg.norm(subtract(x, w), ord=1, axis=-1)

class SOM_server:
    # In chronological order of calling
    def __init__(self, params):
        # Initialize dimensions of map
        self._x = params['X']
        self._y = params['Y']
        self._input_len = params['INPUT_LEN']
        self._random_seed = params['SEED']
        self._random_generator = random.RandomState(self._random_seed)

        # Initialize weights
        self._weights = self._random_generator.rand(self._x, self._y, self._input_len) * 2 - 1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        activation_distance = params['ACTIVATION']
        distance_functions = {'euclidean': self._euclidean_distance,
                        'cosine': self._cosine_distance,
                        'manhattan': self._manhattan_distance}
                        
        if activation_distance not in distance_functions:
            msg = '%s not supported. Distances available: %s'
            raise ValueError(msg % (activation_distance,
                                    ', '.join(distance_functions.keys())))

        self._activation_distance = distance_functions[activation_distance]

    def run_on_server(self):
        # print("Running Server ")
        return self._weights
    
    def classify(self, data):
        centers = self._weights[:,:,0]
        y = np.arange(centers.shape[0])
        clf = NearestCentroid(centers,y)
        clf.fit(centers, y)
        return clf.predict(data)

    def update_server(self, reports_from_devices):
        # find lowest euclidean distance correspondence between cluster assignments among devices
        # row_ind, col_ind = linear_sum_assignment(reports_from_devices)
        num_reports = len(reports_from_devices)
        random_perm = random.permutation(num_reports)
        for i, _ in enumerate(random_perm):
            report = reports_from_devices[i]
            cost = self._activation_distance(self._weights, report)
            row_ind, col_ind = linear_sum_assignment(cost)
            self._update_weights(row_ind, col_ind, report)

    def get_reports_for_devices(self):
        updates_for_devices = self._weights
        return updates_for_devices

    # Helper functions-- Update weights
    def _update_weights(self, row_ind, col_ind, report_from_devices):
        for r, c in zip(row_ind, col_ind):
            self._weights[r, c] = (self._weights[r, c] + report_from_devices[r, c]) / 2.0

    # Helper functions-- Distance
    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum+1e-8)

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def _manhattan_distance(self, x, w):
        return linalg.norm(subtract(x, w), ord=1, axis=-1)

def create_devices(data, device_class, num_devices, params):
    num_data_points = len(data)
    indices = range(num_data_points)
    devices = []
    sample_size = int(num_data_points / num_devices)
    for ID in range(num_devices):
        sub_indices = random.choice(indices, size=sample_size, replace=True)
        params["ID"] = ID
        data_subset = [data[i] for i in sub_indices]
        devices.append(device_class(data_subset, params))
    return devices

def run_devices(devices):
    results_to_server = []
    for device in devices:
        # print("Running Device " + str(device.id))
        device.run_on_device()
        results_to_server.append(device.get_report_for_server())
    return results_to_server

def update_devices(devices, server_to_devices):
    for device in devices:
        device.update_device(server_to_devices)

if __name__ == "__main__":

    ## Run Federated Algorithm
    # Test Parameters
    data = [[10, 10], [9, 10], [8, 10], [9, 9], [1, 1], [0, 1], [0, 0], [1, 0], [10, 10], [9, 10], [8, 10], [9, 9], [1, 1], [0, 1], [0, 0], [1, 0], [10, 10], [9, 10], [8, 10], [9, 9], [1, 1], [0, 1], [0, 0], [1, 0], [10, 10], [9, 10], [8, 10], [9, 9], [1, 1], [0, 1], [0, 0], [1, 0]] # usually some method to get_data()
    num_devices = 4
    num_selected_devices = 2
    num_iterations = 100

    params = { 
        "X": 2, 
        "Y": 2, 
        "INPUT_LEN": len(data[0]),
        "SIGMA": 1.0, 
        "LR": 0.5, 
        "SEED": 1,
        "NEIGH_FUNC": "gaussian",
        "ACTIVATION": 'euclidean',
        "MAX_ITERS": 10,
        "DECAY": asymptotic_decay
    }
    # set seed
    random.seed(params['SEED'])

    # Run federated algorithm
    server = SOM_server(params)
    devices = create_devices(data, SOM_Device, num_devices, params)
    for it in range(num_iterations):
        # select device subset
        indices = range(num_devices)
        device_subset_indices = random.choice(indices, num_selected_devices, replace=False) 
        device_subset = [devices[i] for i in device_subset_indices]
        # run training on devices and send results to the server
        results_to_server = run_devices(device_subset)
        server.update_server(results_to_server)
        server_to_devices = server.get_reports_for_devices()
        # update devices
        update_devices(devices, server_to_devices)

    res = server.run_on_server()
    print(res.shape)
    print(res[:,:,0])
    print(res)