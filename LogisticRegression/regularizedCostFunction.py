# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:34:51 2015

@author: johnzupan
"""

import sigmoidFunction as sf
import numpy as np

reload(sf)

def regularizedCostFunction(theta, X, y, lambda_):
    '''
    Compute the cost and gradient for logistic regression with regularization.
    '''
    
    m = len(y)
    yMatrix = np.hstack([(y), (1-y)])
   
    #cost    
    prediction = sf.sigmoidFunction(np.dot(X, theta))
    
    #does there need to be a -1 here?
    predictionCost = np.hstack([np.log(prediction), np.log(1-prediction)])
    
    yCost = -1 * (predictionCost * yMatrix)
    
    J = (sum(sum(yCost))) / m
    
    regTerm = (lambda_ / (2 * m)) * (sum(sum(theta ** 2)))
    
    J += regTerm
    
###########################################################################
    #gradient
    delta = prediction - y
    
    #lambdaTermReg = ((lambda_ / m) * theta[1:])
    
    #lambdaTerm = np.append(np.zeros((1,1)), ((lambda_ / m) * theta[1:])), axis = 0)
    lambdaTerm = np.append(np.zeros((1,1)), ((lambda_ / m) * theta[1:]), axis =0)    
    print np.shape(lambdaTerm)
    gradient = ((1 / m) * np.dot(np.transpose(X), delta)) + lambdaTerm
    
    return J, gradient