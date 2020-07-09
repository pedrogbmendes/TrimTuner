import numpy as np 
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from robo.models.base_model import BaseModel


###############################################################
#   Decision tree
###############################################################
class DecisionTreeRegressor_(DecisionTreeRegressor):
    def __init__(self):
        self.original_X = None
        self.models = None
        super(DecisionTreeRegressor_, self).__init__()
        self.bagging = None

    def UpdateBagging(self, bagging):
        self.bagging = bagging


    def train(self, X, y, do_optimize=True):
        #train is done in the ensemble using function fit
        self.original_X = X
        return


    def predict(self, X_test, full_cov=False, **kwargs):
        #predictiond done by the entire ensemble
        m, v = self.bagging.predict(X_test, full_cov)
        return m, v


    def predict_mean(self, X_test, **kwargs):
        #prection of each tree
        m =  super(DecisionTreeRegressor_, self).predict(X_test)
        return m


    def get_incumbent(self):   
        projection = np.ones([self.original_X.shape[0], 1]) 
        X_projected = np.concatenate((self.original_X[:, :-1], projection), axis=1)
        m, _ = self.predict(X_projected)

        best = np.argmin(m)
        incumbent = X_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value


    def get_noise(self):
        return 1e-3

    def nll(self, theta):
        return 0 
    
    def grad_nll(self, theta):
        return 0

    def optimize(self):
        return 0
        
    def sample_functions(self, X_test, n_funcs=1):
        return 0        

    def predict_variance(self, x1, X2):
        x_ = np.concatenate((x1, X2))
        _, var = self.bagging.predict(x_, full_cov=True)

        var = var[-1, :-1, np.newaxis]
        return var



###############################################################
#   Extra Decision tree
###############################################################
class ExtraTreeRegressor_(ExtraTreeRegressor):
    def __init__(self):
        self.original_X = None
        self.models = None
        super(ExtraTreeRegressor_, self).__init__()
        self.bagging = None

    def UpdateBagging(self, bagging):
        self.bagging = bagging


    def predict_variance(self, x1, X2):
        x_ = np.concatenate((x1, X2))
        _, var = self.bagging.predict(x_, full_cov=True)

        var = var[-1, :-1, np.newaxis]
        return var


    def train(self, X, y, do_optimize=True):
        self.original_X = X
        return


    def predict(self, X_test, full_cov=False, **kwargs):
        m, v = self.bagging.predict(X_test, full_cov)
        return m, v


    def predict_mean(self, X_test, **kwargs):
        m =  super(ExtraTreeRegressor_, self).predict(X_test)
        return m


    def get_incumbent(self):   
        projection = np.ones([self.original_X.shape[0], 1]) * 1

        X_projected = np.concatenate((self.original_X[:, :-1], projection), axis=1)
        m, _ = self.predict(X_projected)

        best = np.argmin(m)
        incumbent = X_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value


    def get_noise(self):
        return 1e-3

    def nll(self, theta):
        return 0 
    
    def grad_nll(self, theta):
        return 0

    def optimize(self):
        return 0

    def sample_functions(self, X_test, n_funcs=1):
        return 0



###############################################################
#   Bagging ensemble of decision trees
###############################################################

class BaggingRegressor_(BaggingRegressor):
    def __init__(self, base_estimator, n_estimators, random_state):
        super(BaggingRegressor_,self).__init__(base_estimator=base_estimator, 
                                                n_estimators=n_estimators, 
                                                random_state=random_state)
        self.models = None 
        self.original_X = None
        self.n_estimators = n_estimators


    def train(self, X, y, do_optimize=True):
        self.original_X = X
        self.models = self.fit(X, y)
        
        for tree in self.estimators_:
            tree.train(X,y)       
        return self.models
        

    def predict(self, X_test, full_cov=False, **kwargs):
        
        mu = np.zeros([self.n_estimators, X_test.shape[0]])
        counter = 0
    
        # predicted mean value of each tree
        for tree in self.models.estimators_:
            mu[counter,:] = tree.predict_mean(X_test)
            counter +=1

        #mean and standar deviation in the ensemble
        m = np.mean(mu, axis=0)
        v_ = np.std(mu, axis=0)
        
        for i in range(len(v_)):
            if not np.isfinite(v_[i]) or v_[i] < 0:
                v_[i] = 1e-1

        if full_cov:
            v = np.identity(v_.shape[0]) * v_
        else:
            v = v_

        return m, v     
      
      
    def get_incumbent(self):   
        projection = np.ones([self.original_X.shape[0], 1]) * 1

        X_projected = np.concatenate((self.original_X[:, :-1], projection), axis=1)
        m, _ = self.predict(X_projected)

        best = np.argmin(m)
        incumbent = X_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value
       
      

###############################################################
#   Bagging ensemble of decision trees
###############################################################

class EnsembleDTs(BaseModel):

    def __init__(self, number_trees, seed):
        self.no_ensenble = number_trees
        self.X = None
        self.y = None
        self.seed = seed
        self.is_trained = False
    
        # select the tree -> trimtuner uses extra trees
        #self.tree = DecisionTreeRegressor_()
        self.tree = ExtraTreeRegressor_()

        self.forest = BaggingRegressor_(base_estimator=self.tree, n_estimators=number_trees, random_state=self.seed)

        a = np.zeros(1)
        self.models = self.forest.train(a.reshape(-1,1),a)
        
        
    def train(self, X, y, **kwargs):
        self.models = self.forest.train(X, y)

        for tree in self.forest.estimators_:
            tree.UpdateBagging(self.forest)        

        self.X = X
        self.y = y
        self.is_trained = True


    def predict(self, X_test, **kwargs):
        m, v = self.forest.predict(X_test)   
        return m, v


    def get_incumbent(self):
        inc, inc_value = self.forest.get_incumbent()   
        return inc, inc_value