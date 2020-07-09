
import numpy as np
import random

def initial_sampling_trimtuner(seed, listConfig, n_points, s_max):

    list_full_data = []
    for conf in listConfig:
        if conf[-1] == s_max:
            #full dataset
            list_full_data.append(conf[:-1])

    random.seed(seed)
    random.shuffle(list_full_data)

    init_configs = random.sample(list_full_data, k=n_points)
    array_config = np.array(init_configs)

    return array_config



def initial_sampling(seed, listConfig, n_points):

    random.seed(seed)
    random.shuffle(listConfig)

    init_configs = random.sample(listConfig, k=n_points)
    array_config = np.array(init_configs)
    return array_config
