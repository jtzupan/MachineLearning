# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:57:40 2015

@author: johnzupan
"""

import numpy as np

def sigmoidFunction(z):
    '''
    '''
    sigmoid = (1 / (1 + np.exp(1) **(-z)))
    return sigmoid