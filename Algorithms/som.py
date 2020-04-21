import json
import numpy as np
from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply)
from numpy.linalg import norm

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

# Global Consts for SOM Devices
PARAMS = { 
    "X": 5, 
    "Y": 5, 
    "INPUT_LEN": 2,
    "SIGMA": 1.0, 
    "LR": 0.5, 
    "SEED": None,
    "NEIGH_FUNC": "gaussian",
    "ACTIVATION": 'euclidean',
    "MAX_ITERS": 100,
    "DECAY": asymptotic_decay
}

class SOM_Device:
# In chronological order of calling
    def __init__(self, indicies_for_data_subset, params):
        """Initialize SOM member variables"""
        self.data = [[10, 10], [9, 10], [8, 10], [9, 9], [1, 1], [0, 1], [0, 0], [1, 0]]
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
        # Sets data indices for training data input
        num_iteration = self._max_iters
        data = self.data
        iterations = build_iteration_indexes(len(data), num_iteration, self._random_generator)

        # Training
        for t, iteration in enumerate(iterations):
            self.update(data[iteration], self.winner(data[iteration]),
                        t, num_iteration)

    def get_report_for_server(self):
        updates_for_server = []
        return updates_for_server

    def update_device(self, reports_from_server):
        pass

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

class SOM_Server:
    # In chronological order of calling
    def __init__(self):
        pass
    def run_on_server(self):
        pass
    def update_server(self, reports_from_devices):
        pass
    def get_reports_for_devices(self):
        updates_for_devices = []
        return updates_for_devices

s = SOM_Device(1, PARAMS)
s.run_on_device()