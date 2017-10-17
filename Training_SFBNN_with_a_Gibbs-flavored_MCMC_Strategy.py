# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:03:06 2017

@author: victor biaggi
Training SFBNN with a Gibbs-flavored MCMC Strategy


Test :
    
Implementation of a 1 input layer - 1 hidden layer - 1 output layer with stochastic binary units.
Uses Gibbs Flavor Method (nodes h, weights W) for training

the training data will be loaded into the variables X and y
"""

#SFNN Model

import numpy as np
import scipy
from scipy.stats import bernoulli,norm

 #The Network
 
L =1 #Number of hidden layers

def create_network(lengthsn) :
   """
   --lengthsn is a list with the number of nodes for each layer
   --lengthsn[0] is the number of inputs
   --let n = len(lengthsn) ; lengthsn[n-1] is the number of outputs
   """
   n = len(lengthsn)
   global h #list of nodes
   h = [np.array([0]*(lengthsn[i])) for i in range(0,n)] 
   global W #list of weights
   W = [np.array([np.array([0]*lengthsn[i])]*(lengthsn[i+1])) for i in range(0,n-1)] 
   
lengthsn = [400,25,10]
create_network(lengthsn)


# Data loading  
global X
global y
                                                                                                                        
X = np.load("X_train.npy")
y = np.load("y_train.npy")


#Sigmoid function
def s(x) :
   return(1/(1+np.exp(-x)))


#Metroplis_SFBNN for L hidden layers

"""
Please refer to the paper "Training_SFBNN_with_a_Gibbs-flavored_MCMC_Strategy.pdf"
Here we consider that y is a vector for a multiclass calssification problem
"""


def B(a,theta):
    return(a*theta+(1-a)*(1-theta))


def produit_bernoulli(v1,v2):
    """
    Product of B functions as in (E) (see our paper)
    """
    l = len(v1)
    prod = 1
    for i in range(0,l) :
        prod*=B(v1[i],v2[2])
    return(prod)


def produit_normal(y_i,v,sigma) :
    """ 
    Product of N functions as in (E') (see our paper)
    """
    l = len(v)
    prod = 1
    for i in range(0,l) :
        prod*= norm(v[i],sigma).pdf(y[j])
    return(prod)
   

sigma = 1


def Metropolis_decide(init,proposal,r) : 
    next_state = init
    ratio = min(1,r)
    if ratio == 1 :
        next_state = proposal
    elif bernoulli.rvs(ratio) == 1 :
        next_state = proposal
    return(next_state)


def Metropolis_SFBNN(k,i,L):
    
    """ 
    input parameters :
        k : hidden layer we are updating
        i : node we are updating
        L : number of hidden layers
    The proposal is a Bern as in our model
    
    For more information on distributions in python, see
    https://www.johndcook.com/blog/distributions_scipy/
    
    As shown in our model, the ratio for A-R is nothing else but products and division of "bernoullis" and "normals" proba.
    """
    
    # The next quantity
    
    forward = W[k+1]*h[k]

    # The Metropolis ratio
    
    if k==1 :
        lx = len(X)
        Sx = 0
        for j in range(0,lx):
            parameter = s((W[1]*X[j])[i])
            proposal = bernoulli.rvs(parameter) 
            # The new vector - quantity
            h_new = h[k]
            h_new[i] = proposal
            forward_new = W[k+1]*h_new
            if k==L :
                ly = len(y)
                Sy = 0
                for j in range(0,ly) :
                   p = B(parameter,proposal)*norm(forward_new,sigma).pdf(y[j]) # the numerator
                   q = B(parameter,h[k][i])*norm(forward,sigma).pdf(y[j])    # the denominator
                   Sy+= Metropolis_decide(h[k][i],proposal,p/q)
                Sx+= Metropolis_decide(h[k][i],proposal,Sy/ly)
            else :
                p = B(parameter,proposal)*produit_bernoulli(s(forward_new),h[k+1]) 
                q = B(parameter,h[k][i])*produit_bernoulli(s(forward),h[k+1]) 
                Sx+= Metropolis_decide(h[k][i],proposal,p/q)
        return(Metropolis_decide(h[k][i],proposal,Sx/lx))
        
    else :
        parameter = s((W[k]*h[k-1])[i])    #case where k>1
        proposal = bernoulli.rvs(parameter) 
        # The new vector - quantity
        h_new = h[k]
        h_new[i] = proposal
        forward_new = W[k+1]*h_new
        if k == L :
            ly = len(y)
            Sy = 0
            for j in range(0,ly) :
                p = B(parameter,proposal)*norm(forward_new,sigma).pdf(y[j])  # the numerator
                q = B(parameter,h[k][i])*norm(forward,sigma).pdf(y[j])       # the denominator
                Sy+= Metropolis_decide(h[k][i],proposal,p/q)
            return(Metropolis_decide(h[k][i],proposal,Sy/ly))
        else :
            p = B(parameter,proposal)*produit_bernoulli(s(forward_new),h[k+1]) 
            q = B(parameter,h[k][i])*produit_bernoulli(s(forward),h[k+1]) 
            return(Metropolis_decide(h[k][i],proposal,p/q))



#Components updates




#def h_update():
    
    
    

    
#def W_update():
    
    






