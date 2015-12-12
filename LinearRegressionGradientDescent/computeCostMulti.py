# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:02:25 2015

@author: johnzupan
"""

import numpy as np

def computeCostMulti(X, y, theta, alpha, num_iters):
    '''
    Computes the cost for linear regression with multiple variables.
    Uses theta as the parameter for linear regression to fit the data points X and y
    '''
    
    # set m equal to the number of training examples     
    m = len(y)
    #J_history will store the cost after every iteration
    J_history = []
    
    for i in range(num_iters):
        #compute the cost
        #predictedOutput = np.dot(np.transpose(theta), X)
        predictedOutput = np.dot(X, theta)
        delta = predictedOutput - y
        squaredDelta = (delta) ** 2
        sumSquaredDiff = sum(sum(squaredDelta))
        J = (1 / (2 *m)) * sumSquaredDiff
    
        #update theta
        theta -= ((alpha / m) * (np.dot(np.transpose(X), delta)))
    
        #record the cost during this iteration
        J_history.append(J)
    
    return J, theta, J_history