import time
import sys
import math
import random
import george
import numpy as np
import os

#acq function
from trimtuner.acquisition_functions.constrained_entropy_search import Constrained_EntropySearch
from trimtuner.acquisition_functions.marginalization import MarginalizationGPMCMC, MarginalizationDT
from robo.acquisition_functions.ei import *

#heuristics to filter
from trimtuner.maximizers.random_sampling import RandomSampling
from trimtuner.maximizers.cea import CEA
#from trimtuner.maximizers.direct import Direct
#from trimtuner.maximizers.cmaes import CMAES


#models
from trimtuner.models.trimtuner_dt import EnsembleDTs
from trimtuner.models.trimtuner_gp import EnsembleGPs
from robo.priors.env_priors import EnvPrior


#bootstrapping
from trimtuner.trimtuner.initial_sampling import initial_sampling_trimtuner
#incumbent estimation
from trimtuner.trimtuner.incumbent_estimation import incumbent_estimation_cea, incumbent_estimation




def transform(s, s_min, s_max):
    s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
    return s_transform



def retransform(s_transform, s_min, s_max):
    s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
    return int(s)



class Logs():
    #class to print log files
    def __init__(self, seed, initSamples, model, heuristic):
        dir = os.path.abspath(os.getcwd())
        path = dir + "/runLogs"

        self.initSamples = initSamples
        self.seed = seed

        if not os.path.isdir(path):
            try:
                os.mkdir(path) #create runLogs folder
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)

        filename_orig = path + "/trimtuner_logs_seed" + str(seed) + "_initSamples" + str(initSamples) + "_model_" + model + "_heuristic_" + heuristic 
 
        filename = filename_orig + ".txt"
        counter = 1
        while os.path.isfile(filename):

            filename = filename_orig + "_" + str(counter) + ".txt"
            counter += 1
            if counter >= 10000:
                print("ERROR createing the log files!!! Check folder " + path)
                sys.stdout.flush() 
                sys.exit(0)

        #filename += ".txt" 

        self.file_logs = open(filename, "w")
        self.file_logs.write("runID;initSamples;explorationNumber;incumbent;incTime;incAcc;incCost;configTested;Time;Acc;Cost;Overhead;CumulativeCost;\n")


    def printLogs(self, it, inc, incTime, incAcc, incCost, conf, confTime, confAcc, confCost, overhead, CumulativeCost):
    
        strWrite = str(self.seed) + ";" + str(self.initSamples) + ";" + str(it) + ";" + str(inc) + ";" + str(incTime) + ";" + str(incAcc) + ";" + str(incCost) + ";" + str(conf) + ";" + str(confTime) + ";" + str(confAcc) + ";" + str(confCost) + ";" + str(overhead) + ";" + str(CumulativeCost) + "\n"
        self.file_logs.write(strWrite)
 

    def close(self):
         self.file_logs.close()




##################################################################################
# TrimTuner: 
# Efficient Optimization of Machine Learning Jobs in the Cloud via Sub-Sampling
#
##################################################################################

def trimtuner(objective_function, all_configs, constraints, seed, filterHeuristic, model,
            lower, upper, s_min, s_max, n_init=30, num_iterations=100, subsets=[60, 10, 4, 2]):

    # internal paramaters
    burnin=100
    chain_length=100
    n_hypers=12

    #percentage of unexplored configs to test in the acquisition function
    per = 0.1 

    np.random.seed(seed)
    rng = np.random.RandomState(np.random.randint(0, 10000))

    #assert n_init * len(
    assert n_init <= num_iterations, "Number of initial points (n_init) has to be smaller than the  number of iterations" 
    assert lower.shape[0] == upper.shape[0], "Dimension miss match between upper and lower bound"
    assert model == "gp" or model == "dt", "ERROR: wrong model techniques. Chose 'gp' for Gaussian Processes or 'dt' for an ensemble decision tress"
    assert filterHeuristic == "cea" or filterHeuristic == "random"  or filterHeuristic == "nofilter", "ERROR: wrong filtering heuristic. Chose 'cea', 'random', or 'nofilter'!"

    costCumulative = 0

    n_dims = lower.shape[0]

    # Bookkeeping logs
    logs = Logs(seed, n_init, model, filterHeuristic)

    unexplored_Set = all_configs  # list with all possible configurations
    training_Set = []  # traning set

    X = []
    y = []
    c = []

    if model == "dt":
        #ensemble of descision trees
        number_trees = 10
        model_objective = EnsembleDTs(number_trees, seed)
        model_cost = EnsembleDTs(number_trees, seed)

    elif model == "gp":
        #Gaussian Processes

        #kernels functions based on FABOLAS

        # Define model for the objective function
        cov_amp =           1  # Covariance amplitude
        kernel = cov_amp

        for d in range(n_dims):
            kernel *= george.kernels.Matern52Kernel(np.ones([1])*0.01, ndim=n_dims+1, axes=d)

        # Kernel for the environmental variable
        # We use (1-s)**2 as basis function for the Bayesian linear kernel
        env_kernel = george.kernels.BayesianLinearRegressionKernel(log_a=0.1,log_b=0.1,ndim=n_dims + 1,axes=n_dims)
        kernel *= env_kernel

        # Take 3 times more samples than we have hyperparameters
        if n_hypers < 2 * len(kernel):
            n_hypers = 3 * len(kernel)
            if n_hypers % 2 == 1:
                n_hypers += 1


        prior = EnvPrior(len(kernel)+1, n_ls=n_dims, n_lr=2, rng=rng)

        quadratic_bf = lambda x: (1 - x) ** 2
        linear_bf = lambda x: x

        #model for accuracy
        model_objective = EnsembleGPs(kernel,
                                        prior=prior,
                                        burnin_steps=burnin,
                                        chain_length=chain_length,
                                        n_hypers=n_hypers,
                                        normalize_output=False,
                                        basis_func=quadratic_bf,
                                        lower=lower,
                                        upper=upper,
                                        rng=rng)

        # Define model for the cost function
        cost_cov_amp = 1
        cost_kernel = cost_cov_amp

        for d in range(n_dims):
            cost_kernel *= george.kernels.Matern52Kernel(np.ones([1])*0.01, ndim=n_dims+1, axes=d)

        cost_env_kernel = george.kernels.BayesianLinearRegressionKernel(log_a=0.1,log_b=0.1,ndim=n_dims+1,axes=n_dims)
        cost_kernel *= cost_env_kernel

        cost_prior = EnvPrior(len(cost_kernel)+1, n_ls=n_dims, n_lr=2, rng=rng)

        #model for cost
        model_cost = EnsembleGPs(cost_kernel,
                                prior=cost_prior,
                                burnin_steps=burnin,
                                chain_length=chain_length,
                                n_hypers=n_hypers,
                                basis_func=linear_bf,
                                normalize_output=False,
                                lower=lower,
                                upper=upper,
                                rng=rng)


    # Extend input space by task variable
    extend_lower = np.append(lower, 0)
    extend_upper = np.append(upper, 1)
    is_env = np.zeros(extend_lower.shape[0])
    is_env[-1] = 1


    acq_func = Constrained_EntropySearch(model_objective,
                                    model_cost,
                                    constraints,
                                    extend_lower,
                                    extend_upper,
                                    sampling_acquisition=EI,
                                    is_env_variable=is_env,
                                    n_representer=50)

    #if model == 'gp':
    #gps marginalization
    acquisition_func = MarginalizationGPMCMC(acq_func)
    #else:
    #    acquisition_func = MarginalizationDT(acq_func)


    if filterHeuristic == 'random':
        maximizer = RandomSampling(acquisition_func, extend_lower, extend_upper, seed, per)

    if filterHeuristic == 'nofilter':
        maximizer = RandomSampling(acquisition_func, extend_lower, extend_upper, seed, 1)
    
    elif filterHeuristic == 'cea':
        maximizer = CEA(acquisition_func, extend_lower, extend_upper, per, constraints)

    # elif filterHeuristic == 'direct':
    #     #CMAES
    #     maximizer = Direct(acquisition_func, extend_lower, extend_upper, n_func_evals=144, n_iters=300)

    # elif filterHeuristic == 'cmaes':
    #     #CMAES
    #     maximizer = CMAES(acquisition_func, seed, extend_lower, extend_upper, n_func_evals=144)     
       

    # Initial Design
    print("Initial Design")
    sys.stdout.flush()
    counter_it = 1

    real_n_init = int(n_init / len(subsets))   
    x_init = initial_sampling_trimtuner(seed, unexplored_Set, real_n_init, s_max)

    for it in range(real_n_init):

        for subset in subsets:
            start_time_overhead = time.time()
            s = int(s_max / float(subset)) ##real_size

            x = x_init[it]
            print("Evaluate %s on subset size %d" % (x, s))
            sys.stdout.flush()

            #time to select a config to test
            overhead_init = time.time() - start_time_overhead

            func_val, cost, runTime = objective_function(x, s)
            costCumulative += cost

            print("Configuration has an accuracy of %f with cost %f and took %f seconds" % (1-func_val,cost,runTime))
            sys.stdout.flush()

            start_time_overhead = time.time()

            #add config tested to the training set and remove from the untested configs
            tested_config = np.copy(x)
            tested_config[-1] = s
            training_Set.append(tested_config)
            count = 0
            while count != len(unexplored_Set):
                if np.array_equal(unexplored_Set[count], tested_config):
                    unexplored_Set.pop(count)
                    break
                count += 1

            # Bookkeeping
            config = np.append(x, transform(s, s_min, s_max))
            X.append(config)
            y.append(np.log(func_val))  # Model the target function on a logarithmic scale
            c.append(np.log(cost))      # Model the cost on a logarithmic scale

            #time to update the training and the unexplored set
            overhead_updateSet = time.time() - start_time_overhead

            overhead_time = overhead_updateSet + overhead_init

            #write logs in the files
            logs.printLogs(counter_it, x, runTime, 1-func_val, cost, x, runTime, 1-func_val, cost, overhead_time, costCumulative)

            counter_it +=1

    #end initial sampling

    X = np.array(X)
    y = np.array(y)
    c = np.array(c)

    # Train models
    model_objective.train(X, y, do_optimize=True) #model of accuracy
    model_cost.train(X, c, do_optimize=True) #model of cost


    #start optimization
    for it in range(X.shape[0]+1, num_iterations+1):
        print("Start iteration %d ... " % (it))
        sys.stdout.flush()

        start_time = time.time()

        acquisition_func.update(model_objective, model_cost, X, y, c)
        new_x = maximizer.maximize(X, y, c, unexplored_Set) #maximize the acquisition function

        s = retransform(new_x[-1], s_min, s_max)  # Map s from log space to original linear space

        #time to compute the acquisition function
        overhead_time_acqFunc = time.time() - start_time 

        # Evaluate the chosen configuration
        print("Evaluate candidate " + str(new_x[:-1]) + " on subset size " + str(int(s)))
        sys.stdout.flush()

        new_y, new_c, new_t = objective_function(new_x[:-1], int(s))

        costCumulative += new_c    

        #add config tested to the training set and remove from the untested configs
        tested_config = np.copy(new_x)
        tested_config[-1] = s
        training_Set.append(tested_config)
        count = 0
        while count != len(unexplored_Set):
            if np.array_equal(unexplored_Set[count], tested_config):
                unexplored_Set.pop(count)
                break
            count += 1

        print("Configuration has an accuracy of %.3f with cost %.3f and took %.3f seconds" % (1-new_y,new_c,new_t))
        sys.stdout.flush()

        start_time = time.time() #overhead

        # Add new observation to the data
        X = np.concatenate((X, new_x[None, :]), axis=0)
        y = np.concatenate((y, np.log(np.array([new_y]))), axis=0)  # Model the target function on a logarithmic scale
        c = np.concatenate((c, np.log(np.array([new_c]))), axis=0)  # Model the cost function on a logarithmic scale

        # Train models
        model_objective.train(X, y, do_optimize=True) #model of accuracy
        model_cost.train(X, c, do_optimize=True) #model of cost

        # determine the incumbent
        inc, inc_acc, inc_cost = incumbent_estimation_cea(model_objective, model_cost, X[:, :-1], constraints)
        inc[-1] = retransform(inc[-1], s_min, s_max)

        print("Current incumbent " + str(inc) + " with estimated accuracy of " + str(inc_acc) + "%")

        #time to train the models
        overhead_time_trainModels = time.time() - start_time

        #overhead - training models and compute the acq. func.
        total_overhead = overhead_time_trainModels + overhead_time_acqFunc

        print("Optimization overhead was %.3f seconds" % (total_overhead))
        sys.stdout.flush()

        #write logs in the files
        logs.printLogs(it, inc, 0, inc_acc, inc_cost, tested_config, new_t, 1-new_y, new_c, total_overhead, costCumulative)


    logs.close()

    results = "\n The optimal configuration is " + str(inc) + " with estimated accuracy of " + str(inc_acc) + "\0025 and a cost of " + str(inc_cost) + "\n"
    print(results)

    return inc