# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:06:52 2015

@author: johnzupan
"""

import numpy as np

def mapFeatures(X1, X2, degree):
    '''
    feature mapping to polynomial features
    '''
    
    output = np.ones((np.shape(X1)[0],1))
    for i in range(1, degree + 1):
        for j in range(0,i + 1):
            intermediateVar = ((X1 ** (i-j)) * (X2 ** j))
#            np.append(output, ((X1 ** (i-j)) * (X2 ** j))])
            output = np.append(output, intermediateVar, axis = 1)
    
    return output