import numpy as np
from scipy.stats import norm
from robo.maximizers.base_maximizer import BaseMaximizer

def transform(s, s_min, s_max):
    s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
    return s_transform

def retransform(s_transform, s_min, s_max):
    s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
    return int(s)


class CEA(BaseMaximizer):

    def __init__(self, objective_function, lower, upper, per, constraints):
        self.per = per
        self.constraint = constraints
        super(CEA, self).__init__(objective_function, lower, upper)

   
    ############################################
    #  maximize the acquisition function
    #  
    #  evaluates per% of the unexplored configs
    #  with the higher cea
    ############################################
    def maximize(self, X, y, c, unexplored_set):

        numberConfigs_unTest = int(np.shape(unexplored_set)[0] * self.per)
        v_conf = np.zeros((np.shape(unexplored_set)[0], 7))
        aux_counter = 0

        for config in unexplored_set:
            conf = np.zeros((1,6))
            conf[0,0] = config[0]
            conf[0,1] = config[1]
            conf[0,2] = config[2]
            conf[0,3] = config[3]
            conf[0,4] = config[4]
            conf[0,5] = transform(config[5], 1000, 60000)

            #prediction 
            loss1, _ = self.objective_func.model.predict(conf) # loss model
            cost1, cost_sigma1 = self.objective_func.cost_model.predict(conf)  #cost model

            #real accuracy and cost predictions
            real_loss1 = np.exp(loss1)
            acc1 = 1 - real_loss1

            real_cost1 = np.exp(cost1)
            real_sigma_cost1 = real_cost1 * cost_sigma1

            normal_cost = norm(real_cost1, real_sigma_cost1)
            P_constrainCost = normal_cost.cdf(self.constraint)

            v_conf[aux_counter,0] = config[0]
            v_conf[aux_counter,1] = config[1]
            v_conf[aux_counter,2] = config[2]
            v_conf[aux_counter,3] = config[3]
            v_conf[aux_counter,4] = config[4]
            v_conf[aux_counter,5] = transform(config[5], 1000, 60000)

            #CEA - constrained expected accuracy
            v_conf[aux_counter, 6] = acc1 * P_constrainCost
            
            aux_counter += 1
            
        #select the best configs with the higher cea
        topCEA = v_conf[v_conf[:,6].argsort()]
        unexplored_set_v = topCEA[-numberConfigs_unTest: , 0:6]

        # maximize the acquisition function of trimtuner 
        y = self.objective_func(unexplored_set_v)
        x_star = unexplored_set_v[np.argmax(y),:] 

        return x_star
