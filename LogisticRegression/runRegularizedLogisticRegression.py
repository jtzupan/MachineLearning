# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:45:07 2015

@author: johnzupan
"""

import numpy as np
import regularizedCostFunction as cf
import dataNormalization as dn
import mapFeatures as mp

reload(cf)
reload(mp)


#load the data
print 'Loading data...'
dataIn = np.loadtxt('ex2data2.txt', delimiter = ',')

#set training and target fields
m = np.shape(dataIn)[0]
n = np.shape(dataIn)[1]
X = dataIn[:,0:n-1]
y = dataIn[:,n-1:]

#set X to expanded X with polynomial features
#degrees variable sets higher order polynomial to expand to
degree = 6
X = mp.mapFeatures(X[:,0:1], X[:,1:], degree)

#set parameters to be used in the optimization
n = np.shape(X)[1]

theta = np.zeros((n,1))
#lambda_ is the regularization term
lambda_ = 0.01


#do not need to normalize for logistic regression
#print 'Normalizing features...'
#X, mu, sigma = dn.meanNormalization(X, 1)

#do not have to add intercept, aleady added during feature mapping
#add column of ones for intercept
#X = np.append(np.ones((m,1)), X, axis = 1)

cost, gradient = cf.regularizedCostFunction(theta, X, y, lambda_)
print cost, gradient