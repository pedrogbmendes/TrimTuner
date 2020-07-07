
def toStringConfig(c,s):
    return "[n_workers=" + str(c[0]) + ", learning_rate=" + str(c[1]) + ", batch_size=" + str(c[2]) + ", synchronism=" + str(c[3]) + ", vm_flavor=" + str(c[4]) + ", size=" + str(s) + "]"


def toStringIncumbent(c):
    return "[n_workers=" + str(c[0]) + ", learning_rate=" + str(c[1]) + ", batch_size=" + str(c[2]) + ", synchronism=" + str(c[3]) + ", vm_flavor=" + str(c[4]) + ", size=60000]"


def testSet():
    total_configs = 1440
    v_configs = np.zeros((total_configs, 6))
    it = 0

    list_configs = []

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

                            v_configs[it,0] = nr_worker
                            v_configs[it,1] = lr
                            v_configs[it,2] = batch
                            v_configs[it,3] = sync
                            v_configs[it,4] = flavor
                            v_configs[it,5] = transform(s, 1000, 60000)
                            aux = np.copy(v_configs[it,:])
                            aux[-1] = s
                            list_configs.append(aux)
                            it += 1
    
    return v_configs, list_configs


def testSet_without_Subsampling():
    total_configs = 288
    v_configs = np.zeros((total_configs, 6))
    it = 0

    list_configs = []

    # Configurations to test in the acquisition function
    for flavor in [0,1,2,3]:
        for batch in [16, 256]:
            for lr in [0.001, 0.0001, 0.00001]:
                for sync in [0, 1]:
                    for nr_cores in [8, 16, 32, 48, 64, 80]:
                            if flavor == 0:
                                nr_worker = nr_cores
                            elif flavor == 1:
                                nr_worker = nr_cores/2
                            elif flavor == 2:
                                nr_worker = nr_cores/4
                            else:
                                nr_worker = nr_cores/8

                            v_configs[it,0] = nr_worker
                            v_configs[it,1] = lr
                            v_configs[it,2] = batch
                            v_configs[it,3] = sync
                            v_configs[it,4] = flavor
                            v_configs[it,5] = 1 #size
                            aux = np.copy(v_configs[it,:])
                            aux[-1] = 60000
                            list_configs.append(aux)
                            it += 1
    
    return v_configs, list_configs