#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:40:12 2018

@author: Joschua
"""

import numpy as np
import solvers

def read_matrix(fname):
    ''' This function imports a matrix out of a given fname.txt file'''
    file = open(fname, 'r')
    out = []
    lines = file.readlines()
    for line in lines:
        out.append(line.split())
    file.close()

    nn = int(out[0][0])
    aa = np.asarray(out[1:nn+1], dtype=float)
    bb = np.asarray(out[nn+1], dtype=float)
    return aa, bb


aa, bb = read_matrix("linsolve.in")
xx_gauss = solvers.gaussian_eliminate(aa, bb)
print(xx_gauss)
