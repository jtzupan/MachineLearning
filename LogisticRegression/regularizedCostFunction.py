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
    
    yCost = predictionCost * yMatrix
    
    J = sum(sum(yCost))
    
    regTerm = (lambda_ / (2 * m)) * (sum(sum(theta ** 2)))
    
    J += regTerm
    
###########################################################################
    #gradient
    delta = prediction - y
    