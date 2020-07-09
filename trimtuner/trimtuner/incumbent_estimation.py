import numpy as np
from scipy.stats import norm

###############################################################
#  Incumbent estimation on s=1 (full dataset) using the max cea
###############################################################

def incumbent_estimation_cea(model_objective, model_cost, X, Constraint):

    #projection on the full dataset s=1
    projection = np.ones([X.shape[0], 1])
    X_projected = np.concatenate((X, projection), axis=1)

    cost, sigma_cost = model_cost.predict(X_projected) #predict the cost
    loss, sigma_loss = model_objective.predict(X_projected) #predict the loss

    #the prediction are in logaritmic scale, so we have to change of ariable
    #mean_loss->    loss_real = exp(loss)
    #std_loss ->    std_loss_real = loss_real * std_loss

    #mean_acc ->    acc_real = 1 - loss_real
    #std_acc  ->    std_acc_real = std_loss_real
    #mean_cost->    cost_real = exp(cost)
    #std_cost ->    std_cost_real = cost_real * std_cost

    #mean_time->    time_real = cost_real / costPerConfigPerSecond
    #std_time ->    std_time_real = std_cost_real / costPerConfigPerSecond

    real_loss = np.exp(loss)
    acc = 1 - real_loss
    #sigma_acc = real_loss * sigma_loss

    real_cost = np.exp(cost)
    real_sigma_cost = real_cost * sigma_cost

    normal_cost = norm(real_cost, real_sigma_cost)
    P_constrainCost = normal_cost.cdf(Constraint)

    CEA = acc * P_constrainCost
    best = np.argmax(CEA) # maximizes the cea
    
    incumbent = X_projected[best]        # configuration that minimizes the cost
    incumbent_acc = acc[best]            # predicted accuracy  of that configuration
    incumbent_cost = real_cost[best]     #predicted cost of that configuration

    return incumbent, incumbent_acc, incumbent_cost



###############################################################
#  Incumbent estimation on s=1 (full dataset) using the
# max accuracy that meets the constraint
###############################################################

def incumbent_estimation(model_objective, model_cost, X, Constraint):

    #projection on the full dataset s=1
    projection = np.ones([X.shape[0], 1])
    X_projected = np.concatenate((X, projection), axis=1)

    cost, _ = model_cost.predict(X_projected) #predict the cost
    loss, _ = model_objective.predict(X_projected) #predict the loss

    real_loss = np.exp(loss)
    acc = 1 - real_loss   
    real_cost = np.exp(cost)


    # Estimate incumbent as the best observed value so far
    acc_aux = acc.copy()
    cost_aux = real_cost.copy()
    x_aux = X_projected.copy()

    best = np.argmax(acc_aux) # config that minimizes loss

    incumbent = x_aux[best]             #configuration that minimizes the cost
    incumbent_acc = acc_aux[best]       #predicted accuracy  of that configuration
    incumbent_cost = cost_aux[best]     #predicted cost of that configuration


    while incumbent_cost > Constraint:
    #delete from the list the config that minimizes the loss but do not respect the constraint
        acc_aux = np.delete(acc_aux, best, 0)
        cost_aux = np.delete(cost_aux, best, 0)
        x_aux = np.delete(x_aux, best, 0)       

        if x_aux.shape[0] == 0:
            break          
        
        best = np.argmax(acc_aux)

        incumbent = x_aux[best]            
        incumbent_acc = acc_aux[best]      
        incumbent_cost = cost_aux[best]    

    if x_aux.shape[0] == 0:
        # there isn't an incumbent that respect the cost constraint
        # predict the incumbent through the cea
        incumbent, incumbent_acc, incumbent_cost =  incumbent_estimation_cea(model_objective, model_cost, X, Constraint)

    return incumbent, incumbent_acc, incumbent_cost


###############################################################
#  Incumbent estimation on s=1 without constraints
###############################################################

def incumbent_estimation_noconstraints(model_objective, model_cost, X):

    #projection on the full dataset s=1
    projection = np.ones([X.shape[0], 1])
    X_projected = np.concatenate((X, projection), axis=1)

    cost, _ = model_cost.predict(X_projected) #predict the cost
    loss, _ = model_objective.predict(X_projected) #predict the loss

    real_loss = np.exp(loss)
    acc = 1 - real_loss
    real_cost = np.exp(cost)

    best = np.argmax(acc) # config that minimizes loss

    incumbent = X_projected[best]        #configuration that minimizes the cost
    incumbent_acc = acc[best]            #predicted accuracy  of that configuration
    incumbent_cost = real_cost[best]     #predicted cost of that configuration

    return incumbent, incumbent_acc, incumbent_cost

