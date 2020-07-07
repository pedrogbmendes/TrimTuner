
import numpy as np
import random

def initial_sampling(seed, listConfig, n_points):

    random.seed(seed)
    random.shuffle(listConfig)

    init_configs = random.sample(listConfig, k=n_points)

    return init_configs
