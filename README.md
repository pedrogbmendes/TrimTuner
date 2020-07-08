# TrimTuner: Efficient Optimization of MachineLearning Jobs in the Cloud via Sub-Sampling

TrimTuner jointly optimizes the resource allocation and specific application parameters of machine learning (ML) jobs in the cloud in order to maximize the accuracy of the model subject to user-defined Quality of Service (QoS) constraints on the full dataset. Trimtuner use Bayesian Optimization (BO) techniques to solve the optimization problem. It evaluates only configurations using sub-sampled datasets, which are cheap to train, and through transfer learning techniques leverages the knowledge gained by these cheap evaluation of the objective function to predict and recommend optimal configuration that maximize the accuracy on the full dataset and complies with the constraints. .

Paper: (link)


Developed in python3



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




    
  
### To run:

The job to deployed must be implemented in the file run_fabolas in the function called objective_function.

```python3 run_fabolas.py job acq_func```
  
or if you want to run  several networks or with different parameters run:
```python3 run_fabolas_all.py acq_func```
  
  
## Rules

This code can run with 4 different acquisition functions of:

* fabulinus (ES_c):  acq_func=entropy
* fabolas (ES): acq_func=entropy_c
* BO with EI:   acq_func=ei
* BO with EI_c:  acq_func= ei_c


You can define the:
* number of initial samples using the variable n_init.
* the subset sizes using the subsets.
* the minimum and maximum dataset size using s_min and s_max.
* the lower and upper bounds of the dimensions of the search scape using lower and upper.
* the number of iteration through num_iterations.
* the cost constraint using the varible CostConstrain.
 
###
 
The fabulinus is implemented in fabolas.py (robo/fmin/fabolas.py).
 
To the initial sampling is used LHS (initial_design/init_latin_hypercube_sampling.py)

A gaussian process is implemented in FabolasGPMCMC.py (in folder models/fabolas_gp.py).

To calculate the incumbent is used all the tested configs x and projected on s=1 (implemented in util/incumbent_estimation.py).

The acquisition function Entropy Search (ES/c) used in fabolas is implemented in information_gain_per_unit_cost.py. 

The maximizers to maximize the acquisition fucntion are implemented in maximizers/random_sampling.py. 
There are different functions implemented. The latest version of Fabulinus the acquisition function is a(x,s) = ES(x,s)/c * P(c(x_inx, s=1)<Cmax).
The function to maximize is called maximize_Tensorflow_Sim_filter that filter at random 10% of configurations, simulates the models for each (x,s),  and uses threads to speed up the simulation. If you want to change the number of configs to be filter change the variable numberConfigs_unTest = int(np.shape(unexplored_set)[0] * 0.1) (line109)

