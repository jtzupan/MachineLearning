# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:29:32 2015

@author: johnzupan
"""

import numpy as np
import matplotlib.pyplot as plt
import dataNormalization as dn
import computeCostMulti as cm

#will need to reload modules if changing
#dataNormalization or computeCostMulti functions
reload(dn)
reload(cm)

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
num_iters = 400

#initialize theta with zeros
theta = np.zeros((n,1))

J, theta, J_history = cm.computeCostMulti(X, y, theta, alpha, num_iters)

#plot J at each iteration
#should be decreasing every time
x_axis = [q for q in range(num_iters)]
plt.plot(x_axis, J_history)
plt.title('Cost vs. Number of iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.show()

print 'Final J {}'.format(J)
print 'Final theta: {}'.format(theta)
#print 'J history: {}'.format(J_history)