# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:01:40 2020

@author: zaissmz
"""
import numpy as np


#%% ############################################################################
# reverse 

AA=np.array([[11,12,13,14,15],[21,22,23,24,25]])
print(AA)
AA[::2,:]=AA[::2,::-1]
print(AA)