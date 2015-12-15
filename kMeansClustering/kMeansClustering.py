# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 07:29:02 2015

@author: tzupan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dataNormalization as dn

from sklearn.cluster import KMeans
#from sklearn import datasets

np.random.seed(5)

#read data into a np.ndarray
rawData = np.loadtxt('practiceClusteringData.csv', delimiter = ',')

#split X and y from the raw data
X = rawData[:,:3]
print X
X = dn.meanNormalization(X, 1)
print X
y = rawData[:,3]

#centers = [[1, 1], [-1,-1], [1, -1]]
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

fignum = 1
fig = plt.figure(fignum, figsize = (4, 3))
plt.clf()
ax = Axes3D(fig, rect = [0, 1, .95, 1], elev = 20, azim = 134)


plt.cla()
model = KMeans(n_clusters = 2)
model.fit(X)
labels = model.labels_

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('FICO')
ax.set_ylabel('UPEM Score')
ax.set_zlabel('LTV')
fignum = fignum + 1
#estimators = {
#                'k_means_iris_3': KMeans(n_clusters = 3),
#                'k_means_iris_8': KMeans(n_clusters = 8),
#                'k_means_iris_badInit': KMeans(n_clusters = 3,
#                                        n_init = 1, init = 'random')}
#                                        
#fignum = 1
#for name, est in estimators.items():
#    fig = plt.figure(fignum, figsize = (4, 3))
#    plt.clf()
#    ax = Axes3D(fig, rect = [0, 0, .95, 1], elev = 48, azim = 134)
#    
#    plt.cla()
#    est.fit(X)
#    labels = est.labels_
#    
#    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
#    
#    ax.w_xaxis.set_ticklabels([])
#    ax.w_yaxis.set_ticklabels([])
#    ax.w_zaxis.set_ticklabels([])
#    ax.set_xlabel('Petal width')
#    ax.set_ylabel('Sepal length')
#    ax.set_zlabel('Petal length')
#    fignum = fignum + 1
    
#plot the ground truth
#fig = plt.figure(fignum, figsize = (4,3))    
#plt.clf()
#ax = Axes3D(fig, rect = [0, 0, .95, 1], elev = 48, azim = 134)
#plt.cla()
#
#for name, label in [('Setosa', 0),
#                    ('Versicolour', 1),
#                    ('Virginica', 2)]:
#    ax.text3D(X[y == label, 3].mean(),
#              X[y == label, 0].mean() + 1.5,
#              X[y == label, 2].mean(), name,
#              horizontalalignment = 'center',
#              bbox = dict(alpha = .5, edgecolor = 'w', facecolor = 'w')
#              )
#
##reorder the labels to have colors matching the cluster results
#y = np.choose(y, [1,2,0]).astype(np.float)
#ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)
#
#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])
#ax.set_xlabel('Petal width')
#ax.set_ylabel('Sepal length')
#ax.set_zlabel('Petal length')
plt.show()

    