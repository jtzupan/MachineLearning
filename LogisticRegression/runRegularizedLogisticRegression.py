# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:45:07 2015

@author: johnzupan
"""

import numpy as np
import regularizedCostFunction as cf

reload(cf)

dataIn = np.loadtxt('ex2data2.txt', delimiter = ',')

m = np.shape(dataIn)[0]
n = np.shape(dataIn)[1]

X = dataIn[:,0:n-1]
y = dataIn[:,n-1:]

theta = np.zeros((n,1))

lambda_ = 0.01

print 'Normalizing features...'
#X, mu, sigma = dn.meanNormalization(X, 1)

#add column of ones for intercept
X = np.append(np.ones((m,1)), X, axis = 1)

cf.regularizedCostFunction(theta, X, y, lambda_)