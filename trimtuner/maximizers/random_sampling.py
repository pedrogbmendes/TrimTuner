import numpy as np
from scipy.stats import norm
from robo.maximizers.base_maximizer import BaseMaximizer
import random


def transform(s, s_min, s_max):
    s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
    return s_transform

def retransform(s_transform, s_min, s_max):
    s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
    return int(s)


class RandomSampling(BaseMaximizer):

    def __init__(self, objective_function, lower, upper, seed, per):
        self.seed = seed
        self.per = per
        super(RandomSampling, self).__init__(objective_function, lower, upper, rng=None)


    ############################################
    #  maximize the acquisition function
    #  
    #  evaluates per% of the unexplored configs
    #  at random
    ############################################
    def maximize(self, X, y, c, unexplored_set):

        numberConfigs_unTest = int(np.shape(unexplored_set)[0] * self.per)
        random.seed(self.seed)

        configs_toTest = random.sample(unexplored_set,k=numberConfigs_unTest)
        unexplored_set_v = np.array(configs_toTest)

        for i in range(0, numberConfigs_unTest):
            unexplored_set_v[i,5] = transform(unexplored_set_v[i,5], 1000, 60000)

        # maximize the acquisition function of trimtuner
        y = self.objective_func(unexplored_set_v)
        x_star = unexplored_set_v[np.argmax(y),:] 

        return x_star
