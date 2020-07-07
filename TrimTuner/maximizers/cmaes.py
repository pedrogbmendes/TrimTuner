import sys
import logging
import numpy as np

try:
    import cma
except ImportError:
    raise ImportError("""
        In order to use this module, CMA need to be installed. Try running
        pip install cma
    """)

from robo.maximizers.base_maximizer import BaseMaximizer
from robo.initial_design.init_random_uniform import init_random_uniform_Tensorflow




def transform(s, s_min, s_max):
    s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
    return s_transform



def retransform(s_transform, s_min, s_max):
    s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
    return int(s)


class CMAES(BaseMaximizer):

    def __init__(self, objective_function, seed, lower, upper,
                 verbose=True, restarts=0, n_func_evals=1000, rng=None):
        """
        Interface for the  Covariance Matrix Adaptation Evolution Strategy
        python package

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_func_evals: int
            The maximum number of function evaluations
        verbose: bool
            If set to False the CMAES output is disabled
        restarts: int
            Number of restarts for CMAES
        rng: numpy.random.RandomState
            Random number generator
        """
        if lower.shape[0] == 1:
            raise RuntimeError("CMAES does not works in a one \
                dimensional function space")

        super(CMAES, self).__init__(objective_function, lower, upper, rng)
        self.seed = seed
        self.restarts = restarts
        self.verbose = verbose
        self.n_func_evals = n_func_evals

    def maximize(self, numberConfigs_unTest):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with the highest acquisition value.
        """

        verbose_level = -9
        if self.verbose:
            verbose_level = 0

        list_points = init_random_uniform_Tensorflow(self.lower, self.upper, 500, self.seed, self.rng)
        np.random.seed(self.seed)
        col = np.random.randint(low=0, high=np.shape(list_points)[0], size=1)[0]

        start_point = list_points[col,:]
        start_point[-1] = transform(start_point[-1], 1000, 60000)
       # print(start_point)
       

        def obj_func(x):
            a = self.objective_func(x[None, :])
            return -a[0]
        
        std_v = self.lower.shape[0] * [1]
        std_v[1] = 10e-6
        std_v[2] = 10
        std_v[3] = 10e-1
        std_v[4] = 10e-1
        std_v[5] = 10e-1
        
        res = cma.fmin(obj_func, x0=start_point, sigma0=0.6,
                       restarts=self.restarts,
                       options={"bounds": [self.lower, self.upper],
                                "verbose": verbose_level,
                                "verb_log": sys.maxsize,
                                "maxfevals": numberConfigs_unTest,
                                "CMA_stds": std_v})
        if res[0] is None:
            logging.error("CMA-ES did not find anything. \
                Return random configuration instead.")
            return start_point

        x_ = res[0]
        #x[0] = nr of workers
        #x[1] = learning rate
        #x[2] =  batch size
        #x[3] = synchronism
        #x[4] = flavor
        #x[5] = size

        #learning rate
        if x_[1] <= 5.5e-5:
            x_[1] = 1e-5
        elif x_[1] >= 5.5e-4:
            x_[1] = 1e-3
        else:
            x_[1] = 1e-4

        #batch size
        if x_[2] < 136:
            x_[2] = 16
        else:
            x_[2] = 256

        x_[3] = np.rint(x_[3]) #synchronism
        x_[4] = np.rint(x_[4]) #flavor

        #number of workers
        if x_[4] == 0: 
            #small
            if x_[0] <= 12:
                x_[0] = 8
            elif x_[0] > 12 and x_[0] <= 24:
                x_[0] = 16
            elif x_[0] > 24 and x_[0] <= 40:
                x_[0] = 32
            elif x_[0] > 40 and x_[0] <= 56:
                x_[0] = 48
            elif x_[0] > 56 and x_[0] <= 72:
                x_[0] = 64
            else:
                x_[0] = 80

        elif x_[4] == 1:
            #medium
            if x_[0] <= 6:
                x_[0] = 4
            elif x_[0] > 6 and x_[0] <= 12:
                x_[0] = 8
            elif x_[0] > 12 and x_[0] <= 20:
                x_[0] = 16
            elif x_[0] > 20 and x_[0] <= 28:
                x_[0] = 24
            elif x_[0] > 28 and x_[0] <= 36:
                x_[0] = 32
            else:
                x_[0] = 40

        elif x_[4] == 2:
            #xlarge
            if x_[0] <= 3:
                x_[0] = 2
            elif x_[0] > 3 and x_[0] <= 6:
                x_[0] = 4
            elif x_[0] > 6 and x_[0] <= 10:
                x_[0] = 8
            elif x_[0] > 10 and x_[0] <= 14:
                x_[0] = 12
            elif x_[0] > 14 and x_[0] <= 18:
                x_[0] = 16
            else:
                x_[0] = 20

        else:
            #2xlarge
            if x_[0] <= 1.5:
                x_[0] = 1
            elif x_[0] > 1.5 and x_[0] <= 3:
                x_[0] = 2
            elif x_[0] > 3 and x_[0] <= 5:
                x_[0] = 4
            elif x_[0] > 5 and x_[0] <= 7:
                x_[0] = 6
            elif x_[0] > 7 and x_[0] <= 9:
                x_[0] = 8
            else:
                x_[0] = 10

        #size
        s = retransform(x_[5], 1000, 60000) #real value
        if s <= 3500:
            x_[5] = transform(1000, 1000, 60000)
        elif s > 3500 and s <= 10500:
            x_[5] = transform(6000, 1000, 60000)
        elif s > 10500 and s <= 22500:
            x_[5] = transform(15000, 1000, 60000)
        elif s > 22500 and s <= 4500:
            x_[5] = transform(30000, 1000, 60000) 
        else:
            x_[5] = transform(60000, 1000, 60000) 


        return x_

    def maximize_fab(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with the highest acquisition value.
        """

        verbose_level = -9
        if self.verbose:
            verbose_level = 0

        start_point = init_random_uniform(self.lower, self.upper, 1, self.rng)

        def obj_func(x):
            a = self.objective_func(x[None, :])
            return -a[0]

        res = cma.fmin(obj_func, x0=start_point[0], sigma0=0.6,
                       restarts=self.restarts,
                       options={"bounds": [self.lower, self.upper],
                                "verbose": verbose_level,
                                "verb_log": sys.maxsize,
                                "maxfevals": self.n_func_evals})
        if res[0] is None:
            logging.error("CMA-ES did not find anything. \
                Return random configuration instead.")
            return start_point

        return res[0]       
