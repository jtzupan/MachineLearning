# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:29:32 2015

@author: johnzupan
"""

import numpy as np
import dataNormalization as dn
import computeCostMulti as cm

print 'Loading data...'

#read in data
fileIn = np.loadtxt('ex1data2.txt', delimiter = ',')
m = np.shape(fileIn)[0]
n = np.shape(fileIn)[1]
X = fileIn[:, :n-1]
y = fileIn[:, n-1:]

print 'Normalizing features...'
X, mu, sigma = dn.meanNormalization(X, 1)

#add column of ones for intercept
X = np.append(np.ones((m,1)), X, axis = 1)

print 'Running gradient descent...'

#set learning rate and number of iterations
alpha = .01
num_iters = 10

#initialize theta with zeros
theta = np.zeros((n,1))

J, theta, J_history = cm.computeCostMulti(X, y, theta, alpha, num_iters)

print 'Final J {}'.format(J)
print 'Final theta: {}'.format(theta)
#print 'J history: {}'.format(J_history)