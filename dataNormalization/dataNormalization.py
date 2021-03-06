# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 08:01:24 2015

@author: tzupan
"""

import numpy as np

def meanNormalization(inputArray, normTypeChoice):
    '''
    Normalizes the input array.  use normTypeChoice to specify norm type.
        '1' uses the equation (value - mean) / std
        '2' uses the equation (value - mean) / range
    '''
    columnMeans = inputArray.mean(axis = 0)
    columnStd = inputArray.std(axis = 0)
    columnRange = np.ptp(inputArray, axis = 0)
    
    normType = {1: columnStd, 2: columnRange}

    normalizedArray = (inputArray - columnMeans) / normType[normTypeChoice]
    
    return normalizedArray, columnMeans, columnStd
