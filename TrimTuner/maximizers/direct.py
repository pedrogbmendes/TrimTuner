import os
import sys

try:
    import DIRECT
except ImportError:
    raise ImportError("""
        In order to use this module, DIRECT need to be installed. Try running
        pip install direct
    """)

import numpy as np
from robo.maximizers.base_maximizer import BaseMaximizer


def transform(s, s_min, s_max):
    s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
    return s_transform

def retransform(s_transform, s_min, s_max):
    s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
    return int(s)


class Direct(BaseMaximizer):

    def __init__(self, objective_function, lower, upper):

        super(Direct, self).__init__(objective_function, lower, upper)

    def _direct_acquisition_fkt_wrapper(self, acq_f):
        def _l(x, user_data):
            return -acq_f(np.array([x])), 0

        return _l

    def maximize(self, numberConfigs_unTest):

        n_iters = int(numberConfigs_unTest * 2)

        x, _, _ = DIRECT.solve(self._direct_acquisition_fkt_wrapper(self.objective_func),
                                l=[self.lower],
                                u=[self.upper],
                                maxT=n_iters,
                                maxf=numberConfigs_unTest)



        #x[0] = nr of workers
        #x[1] = learning rate
        #x[2] =  batch size
        #x[3] = synchronism
        #x[4] = flavor
        #x[5] = size

        #learning rate
        if x[1] <= 5.5e-5:
            x[1] = 1e-5
        elif x[1] >= 5.5e-4:
            x[1] = 1e-3
        else:
            x[1] = 1e-4

        #batch size
        if x[2] < 136:
            x[2] = 16
        else:
            x[2] = 256

        x[3] = np.rint(x[3]) #synchronism
        x[4] = np.rint(x[4]) #flavor

        #number of workers
        if x[4] == 0: 
            #small
            if x[0] <= 12:
                x[0] = 8
            elif x[0] > 12 and x[0] <= 24:
                x[0] = 16
            elif x[0] > 24 and x[0] <= 40:
                x[0] = 32
            elif x[0] > 40 and x[0] <= 56:
                x[0] = 48
            elif x[0] > 56 and x[0] <= 72:
                x[0] = 64
            else:
                x[0] = 80

        elif x[4] == 1:
            #medium
            if x[0] <= 6:
                x[0] = 4
            elif x[0] > 6 and x[0] <= 12:
                x[0] = 8
            elif x[0] > 12 and x[0] <= 20:
                x[0] = 16
            elif x[0] > 20 and x[0] <= 28:
                x[0] = 24
            elif x[0] > 28 and x[0] <= 36:
                x[0] = 32
            else:
                x[0] = 40

        elif x[4] == 2:
            #xlarge
            if x[0] <= 3:
                x[0] = 2
            elif x[0] > 3 and x[0] <= 6:
                x[0] = 4
            elif x[0] > 6 and x[0] <= 10:
                x[0] = 8
            elif x[0] > 10 and x[0] <= 14:
                x[0] = 12
            elif x[0] > 14 and x[0] <= 18:
                x[0] = 16
            else:
                x[0] = 20

        else:
            #2xlarge
            if x[0] <= 1.5:
                x[0] = 1
            elif x[0] > 1.5 and x[0] <= 3:
                x[0] = 2
            elif x[0] > 3 and x[0] <= 5:
                x[0] = 4
            elif x[0] > 5 and x[0] <= 7:
                x[0] = 6
            elif x[0] > 7 and x[0] <= 9:
                x[0] = 8
            else:
                x[0] = 10

        #size
        s = retransform(x[5], 1000, 60000) #real value
        if s <= 3500:
            x[5] = transform(1000, 1000, 60000)
        elif s > 3500 and s <= 10500:
            x[5] = transform(6000, 1000, 60000)
        elif s > 10500 and s <= 22500:
            x[5] = transform(15000, 1000, 60000)
        elif s > 22500 and s <= 4500:
            x[5] = transform(30000, 1000, 60000) 
        else:
            x[5] = transform(60000, 1000, 60000) 

        return x

