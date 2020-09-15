# TrimTuner: Efficient Optimization of MachineLearning Jobs in the Cloud via Sub-Sampling

TrimTuner jointly optimizes the resource allocation and specific application parameters of machine learning (ML) jobs in the cloud in order to maximize the accuracy of the model subject to user-defined Quality of Service (QoS) constraints on the full dataset. Trimtuner use Bayesian Optimization (BO) techniques to solve the optimization problem. It evaluates only configurations using sub-sampled datasets, which are cheap to train, and through transfer learning techniques leverages the knowledge gained by these cheap evaluation of the objective function to predict and recommend optimal configuration that maximize the accuracy on the full dataset and complies with the constraints. .


Paper: (https://github.com/pedrogbmendes/TrimTuner/blob/master/trimtuner.pdf)



### Instalation:
* download the TrimTuner: 
```git clone https://github.com/pedrogbmendes/TrimTuner.git```
* enter in the downloaded directory: 
```cd TrimTuner```
* run the comand to install the required software and python packages: 
```sudo /bin/sh requirements.sh```
* run the following command to install TrimTuner: 
```sudo python3 setup.py install```



### Requirements

TrimTuner runs in Python3 (in particular, our experiments were performed using Python 3.6.9).
It requires a frotran compiler (gfortran) (for Direct algorithm), and libeigen3-dev and swig softwares.
TrimTuner uses RoBO (https://github.com/automl/RoBO) and ir requires the following python libraries:
- george; - emcee; - pyrfr; - pybnn; - cython; - scipy; - numpy; - sklearn; - torch; - torchvision; - nose; - pyyaml; - jinja2; - pybind11; - pybnn; - george; - direct; - cma; - theano; - matplotlib; - lasagne; - sgmcmc; - hpolib2; - robo;



### Run TrimTuner

Calling TrimTuner:
```
def trimtuner(objective_function, all_configs, constraints, seed, filterHeuristic, model, lower, upper, s_min, s_max,n_init, num_iterations, subsets=[60, 10, 4, 2]):
```

Input Parameters:
* objective_function -> ML model to optimize
* all_configs -> set containing all the possible configurations of the search space
* constraints -> user-defined QoS constraints
* seed -> run ID
* filterHeuristic -> filtering heuristic to select the configurations to test in the acquisition function (cea, random, nofilter, direct, cmaes)
* model -> modelling techniques (gp (Gaussian Processes) or dt (ensemble of decision trees))
* lower/ upper -> lower and upper bound of the search space
* s_min -> minimum dataset size
* s_max -> full dataset
* n_init -> number of initial samples to bootstrap the models
* num_iterations -> stop condition
* subsets -> possible sub-sampled dataset sizes to use

