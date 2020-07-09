#!/usr/bin/env python
import argparse
import sys
import numpy as np
import collections
import csv
import math

#from robo.fmin import fabolas
from trimtuner.trimtuner import trimtuner


csv.field_size_limit(sys.maxsize)

listConfig = []


####################################################################
#            Configureation Tensorflow NNs
# cloud parameters: - no. of parameter servers
#                   - no. of workers
#                   - VM type
#
# application parameters:   - batch size
#                           - synchronism
#                           - learning rate
#
# dataset size parameter
####################################################################
class TensorflowConfig:

    def __init__(self, nrps, nrworkers, lr, batch, sync, flavor, size, runtime, price, acc ):
        self.nr_ps = nrps	# 1
        self.nr_workers = nrworkers
        self.learning_rate = lr
        self.batch_size = batch     # 16; 256
        self.synchronism = sync     # 0 = async ; 1 = sync
        self.vm_flavor = flavor     # 0 = t2.small ; 1 = t2.medium ; 2 = t2.xlarge ; 3 = t2.2xlarge
        self.s = size               # dataset size  

        self.time = runtime #in seconds
        self.cost = price
        self.accuracy = acc

    def __str__(self):
        # Override to print a readable string presentation of your object
        return "[nr_ps=" + str(self.nr_ps) + ", nr_workers=" + str(self.nr_workers) + ", learning_rate=" + str(self.learning_rate) + ", batch_size=" + str(self.batch_size) + ", synchronism=" + str(self.synchronism) + ", vm_flavor=" + str(self.vm_flavor) + ", dataset_size=" + str(self.s) + "];cost=" + str(self.cost) + ";accuracy=" + str(self.accuracy) + ";time=" + str(self.time)


    def __eq__(self, c):
        if self is c:
            return True
        if c is None:
            return False
        if self.nr_ps != c.nr_ps:
            return False
        if self.nr_workers != c.nr_workers:
            return False
        if self.learning_rate != c.learning_rate:
            return False
        if self.batch_size != c.batch_size:
            return False
        if self.synchronism != c.synchronism:
            return False
        if self.vm_flavor != c.vm_flavor:
            return False
        if self.s != c.s:
            return False
        return True


    def getValue(self, listConfig):
        for x in listConfig:
            if self == x:
                return x.accuracy, x.cost, x.time
        return


    def getSize(self, listConfig):
        for x in listConfig:
            if self == x:
                return x.s
        return

    def getAcc(self, listConfig):
        for x in listConfig:
            if self == x:
                return x.accuracy
        return

    def getCost(self, listConfig):
        for x in listConfig:
            if self == x:
                return x.cost
        return

    def getTime(self, listConfig):
        for x in listConfig:
            if self == x:
                return x.time
        return

    def setValues(self, listConfig, acc, cost, time):
        for x in listConfig:
            if self == x:
                listConfig.remove(x)
                self.accuracy = acc
                self.cost = cost
                self.time = time
                listConfig.append(self)
        return



#unexplored set for tensorflow experiments
def testSet():
    all_configs = []

    # Configurations to test in the acquisition function
    for flavor in [0,1,2,3]:
        for batch in [16, 256]:
            for lr in [0.001, 0.0001, 0.00001]:
                for sync in [0, 1]:
                    for nr_cores in [8, 16, 32, 48, 64, 80]:
                        for s in [1000, 6000, 15000, 30000, 60000]:
                            if flavor == 0:
                                nr_worker = nr_cores
                            elif flavor == 1:
                                nr_worker = nr_cores/2
                            elif flavor == 2:
                                nr_worker = nr_cores/4
                            else:
                                nr_worker = nr_cores/8

                            config = np.zeros((1,6))

                            config[0,0] = nr_worker
                            config[0,1] = lr
                            config[0,2] = batch
                            config[0,3] = sync
                            config[0,4] = flavor
                            config[0,5] = s 
                            
              
                            all_configs.append(config)    
    return all_configs



#Read the dataset
def load_dataset(CSV):

    print("Reading the dataset....")
    sys.stdout.flush()

    with open(CSV) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader, None)

        for row in csv_reader:
            nr_workers = int(row[1])
            learning_rate = float(row[3])
            batch_size = int(row[4])
            synchronism = 0 if row[6] == "async" else  1
            performance = float(row[7])    # training time in seconds
            nr_ps = int(row[10])
            accuracy = float(row[11])
            flavor = row[13]
            size = int(row[15])
            nr_iterations = int(row[17])

            intermediateValues = list(map(float, row[9].split()))
            cost = 0

            if flavor == "t2.small":
                vm_flavor = 0
                cost = (((nr_workers+nr_ps) * 0.023/60.0) + (0.3712/60.0)) * (performance/60.0)
            elif flavor == "t2.medium":
                vm_flavor = 1
                cost = (((nr_workers+nr_ps) * 0.0464/60.0) + (0.3712/60.0)) * (performance/60.0)
            elif flavor == "t2.xlarge":
                vm_flavor = 2
                cost = (((nr_workers+nr_ps) * 0.1856/60.0) + (0.3712/60.0)) * (performance/60.0)
            elif flavor == "t2.2xlarge":
                vm_flavor = 3
                cost = (((nr_workers+nr_ps) * 0.3712/60.0) + (0.3712/60.0)) * (performance/60.0)
            else:
                print("Tensorflow configuration - Unknown flavor" + flavor)
                sys.exit(0)

            listConfig.append(TensorflowConfig(nr_ps, nr_workers, learning_rate, batch_size, synchronism, vm_flavor, size, performance, cost, accuracy))

        csv_file.close()

    print("Dataset read")
    sys.stdout.flush()



# ML job to optimize
def objective_function(x, s):

    nr_worker = int(x[0])   # no of workers
    lr = float(x[1])        # learning rate
    batch = int(x[2])       # batch size
    sync = int(x[3])        # synchronism
    flavor = int(x[4])      # flavor
    nr_ps = 1               # no of parameter servers
    size = int(s)           # dataset size

    # Train the ML model in the configuration x on the dataset s
    C = TensorflowConfig(nr_ps, nr_worker, lr, batch, sync, flavor, size, 0, 0, 0)
    #sprint(len(listConfig))
    c_acc, c_cost, c_time = C.getValue(listConfig)

    loss_func = 1 - c_acc #loss functions 

    return loss_func, c_cost, c_time




def main(NetworkType, seed, n_init, num_iterations, filterHeuristic, model):

    #define the constraints for the different NNs
    if NetworkType == "cnn":
        constraints = 0.1
        CSV = "files/cnn.csv"

    elif NetworkType == "rnn":
        constraints = 0.02
        CSV = "files/rnn.csv"

    elif NetworkType == "multilayer" :
        constraints = 0.06
        CSV = "files/multilayer.csv"
                
    load_dataset(CSV)
    all_configs = testSet()

    # We optimize s on a log scale, as we expect that the performance varies
    # logarithmically across s
    s_min = 1000
    s_max = 60000

    subsets = [60, 10, 4, 2]

    #[nr_worker, lr, batch, sync, flavor]
    lower = np.array([1, 0.00001, 16, 0, 0])
    upper = np.array([80, 0.001, 256, 1, 3])


    # run trimtuner
    res = trimtuner(objective_function, all_configs, constraints, seed, filterHeuristic, model,
                    lower=lower,upper=upper,s_min=s_min, s_max=s_max, n_init=n_init, 
                    num_iterations=num_iterations, subsets=subsets)


    print("End optimization!!!")
    sys.stdout.flush()
    print(res)
    sys.stdout.flush()



if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--network",
        type=str,
        default="rnn",
        help="network"
    )

    parser.add_argument(
        "--initSamples",
        type=int,
        default=4,
        help="number initial samples"
    )
    
    parser.add_argument(
        "--filter",
        type=str,
        default="cea",
        help="heuristic to filter"
    )   

    parser.add_argument(
        "--iterations",
        type=int,
        default=44,
        help="number of iterations"
    )


    parser.add_argument(
        "--model",
        type=str,
        default="gp",
        help="modelling technique"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed"
    )

    FLAGS, unparsed = parser.parse_known_args()


    #NN type -> FLAGS.network
    Networks = []
    if FLAGS.network == 'cnn':
        Networks.append('cnn')

    elif FLAGS.network == 'rnn':
        Networks.append('rnn')

    elif FLAGS.network == 'multilayer':
        Networks.append('multilayer')

    elif FLAGS.network == 'all':
        Networks.append('cnn')
        Networks.append('rnn')
        Networks.append('multilayer')
    else: 
        print("ERROR: Wrong Neural network " + FLAGS.network)
        sys.stdout.flush()
        sys.exit(0)


    #initial Samples -> FLAGS.initSamples
    # default value used in trimtuner paper was 4 initial samples
    if FLAGS.initSamples < 0:
        print("ERROR: Initial Samples cannot be negative")
        sys.stdout.flush()
        sys.exit(0)

    elif FLAGS.initSamples > 1400:
        print("ERROR: Initial Samples value is to large")
        sys.stdout.flush()
        sys.exit(0)

    else:
        initialSample = int(FLAGS.initSamples)


    #Number of iterations (stop condition) -> FLAGS.iterations
    if FLAGS.iterations < 0: 
        print("ERROR: Number of iterations cannot be negative")
        sys.stdout.flush()
        sys.exit(0)

    num_iterations = FLAGS.iterations
    
    if initialSample >= num_iterations:
        print("ERROR: Initial samples higher than the total number of iterations")
        sys.stdout.flush()
        sys.exit(0)
             

    # filtering heuristic -> FLAGS.filter
    # cea, random, no filter
    if not (FLAGS.filter == "cea" or FLAGS.filter == "nofilter" or FLAGS.filter == "random"):
        print("ERROR: Wrong filtering heuristic. Chose cea, nofilter or random")
        sys.stdout.flush()
        sys.exit(0)

    if FLAGS.model != "gp" and FLAGS.model != "dt":
        print("ERROR: Wrong modelling technique. Chose gp or dt for Gaussian Processes or Ensemble of decision trees, respectively")
        sys.stdout.flush()
        sys.exit(0)      


    RunSeed = []
    if FLAGS.seed == 0:
        for j in range(1,11):
            RunSeed.append(j)
            
    elif FLAGS.seed > 0 and FLAGS.seed <= 1000000:
        RunSeed.append(int(FLAGS.seed))

    else:
        print("Wrong Seed. Please choose an integer between 1 and 1000000, or 0 to run 10 seed [1,10]")
        sys.stdout.flush()
        sys.exit(0)


    for NetworkType in Networks: 
    
        for seed_ in RunSeed: #seeds
            print("Running Network %s using Trimtuner (%s) with seed %d, %d initial Samples, %d iterations, and filter Heuristic %s" % (NetworkType, FLAGS.model, seed_, initialSample, num_iterations, FLAGS.filter))
            sys.stdout.flush()

            main(NetworkType, seed_, initialSample, num_iterations, FLAGS.filter, FLAGS.model)
            listConfig.clear()
