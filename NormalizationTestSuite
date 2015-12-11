# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 13:36:32 2015

@author: tzupan
"""

import unittest
import numpy as np
import dataNormalization as dm

np.random.seed(3)

class testNormalization(unittest.TestCase):
    
    def setUp(self):
        self.correctOutput = np.array([[ 0.65596265,  1.064836  ],
       [-0.21482622,  0.02364791],
       [ 1.8023552 ,  2.05761116],
       [-0.76873967, -1.57825993],
       [-1.01707718, -0.34581074],
       [-1.0894192 , -0.2612611 ],
       [ 0.98547778, -1.20232873],
       [ 1.07631445,  0.44596504],
       [-1.10916853,  0.27706645],
       [-0.32087927, -0.48146605]], dtype = float)    
       

    def test_normalCase(self):                                   
        output = dm.meanNormalization(np.random.rand(10,2), 1)
        self.assertAlmostEqual(float(sum(sum(output))), sum(sum(self.correctOutput)))
    
    def test_length(self):
        
        
        
        
if __name__ == '__main__':
    unittest.main()
