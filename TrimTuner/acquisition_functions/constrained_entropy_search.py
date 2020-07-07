import numpy as np
import copy
from scipy.stats import norm

from robo.acquisition_functions.information_gain_per_unit_cost import InformationGainPerUnitCost
from trimtuner.trimtuner.incumbent_estimation import incumbent_estimation_cea, incumbent_estimation

###############################################################
#  TrimTuner acquisition function
###############################################################

class Constrained_EntropySearch(InformationGainPerUnitCost):

    def __init__(self,model,cost_model,constraint,lower,upper,is_env_variable,sampling_acquisition=None,n_representer=50):

        self._X = None
        self._y = None
        self._c = None
        self.constraint = constraint

        super(Constrained_EntropySearch, self).__init__(model,cost_model,lower,upper,is_env_variable,sampling_acquisition=sampling_acquisition,n_representer=n_representer)

    def update(self, model, cost_model, X,y,c):
        self._X = X
        self._y = y
        self._c = c
        super(Constrained_EntropySearch, self).update(model,cost_model)


    def compute(self, X):
        
        ret = np.zeros(np.shape(X)[0])

        #clone the model
        clone_model_loss = copy.deepcopy(self.model)
        clone_model_cost = copy.deepcopy(self.cost_model)
      
        for i in range(0, np.shape(X)[0]):
            x = np.ones((1, 6))
            x[0,0] = X[i,0]
            x[0,1] = X[i,1]
            x[0,2] = X[i,2]
            x[0,3] = X[i,3]
            x[0,4] = X[i,4]
            x[0,5] = X[i,5]
        
            #information_gain_per_unit_cost FABOLAS acquisition function
            acquisition_value = super(Constrained_EntropySearch, self).compute(x,derivative=False)

            #prediction for cost and loss - results in logarithm
            cost_act, _ = clone_model_cost.predict(x)
            loss_act, _ = clone_model_loss.predict(x)

            # Add predicted observation to the data
            X_aux = np.concatenate((self._X, x), axis=0)
            y_aux = np.concatenate((self._y, loss_act), axis=0)  # Model the target function on a logarithmic scale
            c_aux = np.concatenate((self._c, cost_act), axis=0)  # Model the cost function on a logarithmic scale

            #train  the model with predicted info
            clone_model_loss.train(X_aux, y_aux,  do_optimize=True)
            clone_model_cost.train(X_aux, c_aux,  do_optimize=True)

            #predict the new incumbent
            incumbent, _, _ = incumbent_estimation_cea(clone_model_loss, clone_model_cost, X_aux[:, :-1], self.constraint)
            #incumbent, _, _ = incumbent_estimation_(clone_model_loss, clone_model_cost, X_aux[:, :-1], self.constraint)

            inc = np.ones((1, 6))
            inc[0,0] = incumbent[0]
            inc[0,1] = incumbent[1]
            inc[0,2] = incumbent[2]
            inc[0,3] = incumbent[3]
            inc[0,4] = incumbent[4]
            inc[0,5] = incumbent[5]

            #prediction for s=1
            cost1, cost_sigma1 = clone_model_cost.predict(inc)  #cost model
            real_cost1 = np.exp(cost1)
            real_sigma_cost1 = real_cost1 * cost_sigma1

            normal_cost = norm(real_cost1, real_sigma_cost1)
            P_constrainCost = normal_cost.cdf(self.constraint)

            ret[i] = acquisition_value * P_constrainCost

        return ret

