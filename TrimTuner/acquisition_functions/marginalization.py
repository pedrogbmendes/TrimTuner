import numpy as np
from copy import deepcopy
from robo.acquisition_functions.base_acquisition import BaseAcquisitionFunction


###############################################################
#     Marginalization over the hyper-parameters of GPs
# (for the TrimTuner's acquisition function)
###############################################################
class MarginalizationGPMCMC(BaseAcquisitionFunction):
    def __init__(self, acquisition_func):

        self.acquisition_func = acquisition_func
        self.model = acquisition_func.model

        self.cost_model = acquisition_func.cost_model

        self.estimators = []
        for i in range(len(self.model.models)):
            estimator = deepcopy(acquisition_func)

            if len(self.model.models) == 0:
                estimator.model = None
            else:
                estimator.model = self.model.models[i]

            if len(self.cost_model.models) == 0:
                estimator.cost_model = None
            else:
                estimator.cost_model = self.cost_model.models[i]
            
            self.estimators.append(estimator)


    def update(self, model, cost_model, X, y, c, **kwargs):

        if len(self.estimators) == 0:
            for i in range(len(self.model.models)):
                estimator = deepcopy(self.acquisition_func)
                if len(self.model.models) == 0:
                    estimator.model = None
                else:
                    estimator.model = self.model.models[i]

                if len(self.cost_model.models) == 0:
                    estimator.cost_model = None
                else:
                    estimator.cost_model = self.cost_model.models[i]
        
                self.estimators.append(estimator)

        self.model = model
        self.cost_model = cost_model

        for i in range(len(self.model.models)):
            self.estimators[i].update(self.model.models[i],self.cost_model.models[i], X, y, c,**kwargs)



    def compute(self, X_test, derivative=False):

        acquisition_values = np.zeros([len(self.model.models), X_test.shape[0]])

        for i in range(len(self.model.models)):
            acquisition_values[i] = self.estimators[i].compute(X_test, derivative=derivative)

        return acquisition_values.mean(axis=0)


###############################################################
#    Bagging ensemble of decision trees
# (for the TrimTuner's acquisition function with DTs)
###############################################################

class MarginalizationDT(BaseAcquisitionFunction):
    def __init__(self, acquisition_func):

        self.acquisition_func = acquisition_func
        self.model = acquisition_func.model
        self.cost_model = acquisition_func.cost_model


    def update(self, model, cost_model, X, y, c, **kwargs):

        self.model = model
        self.cost_model = cost_model
        self.acquisition_func.update(model, cost_model, X, y, c)


    def compute(self, X_test, derivative=False):

        acquisition_values = np.zeros([1, X_test.shape[0]])
        acquisition_values = self.acquisition_func.compute(X_test, derivative=derivative)
        return acquisition_values.mean(axis=0)
