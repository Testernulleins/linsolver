"""Routines for solving a linear system of equations."""
import numpy as np


def gaussian_eliminate(aa, bb):
    """Solves a linear system of equations (Ax = b) by Gauss-elimination

    Args:
        aa: Matrix with the coefficients. Shape: (n, n).
        bb: Right hand side of the equation. Shape: (n,)

    Returns:
        Vector xx with the solution of the linear equation or None
        if the equations are linearly dependent.
    """
    nn = aa.shape[0]
    xx = np.zeros((nn,), dtype=float)
    
    
    # first loop
    fac1 = aa[1,0]/aa[0,0]
    fac2 = aa[2,0]/aa[0,0]
    aa[1] = aa[1] - fac1*aa[0]
    aa[2] = aa[2] - fac2*aa[0]
    bb[1] = bb[1] - fac1*bb[0]
    bb[2] = bb[2] - fac2*bb[0]
    
    # before moving on, swap lines if neccessary
    '''Different approaches
    len1 = aa[1].nonzero()[0].shape[0]
    len2 = aa[2].nonzero()[0].shape[0]
    sum1 = sum(np.abs(aa[1]))
    sum2 = sum(np.abs(aa[2]))'''
    abs1 = np.abs(aa[1,1])
    abs2 = np.abs(aa[2,1])
    
    if abs1 < abs2:
        aa[[1,2]]=aa[[2,1]]
        bb[[1,2]]=bb[[2,1]]
    
    # second loop
    fac3 = aa[2,1]/aa[1,1]
    aa[2] = aa[2] - fac3*aa[1]
    bb[2] = bb[2] - fac3*bb[1]
    
    # calculate xx
    if aa[2,2] == 0:
        xx = None
    else:
        xx[2] = bb[2]/aa[2,2]
        xx[1] = (bb[1] - aa[1,2]*xx[2]) / aa[1,1]
        xx[0] = (bb[0] - aa[0,2]*xx[2] - aa[0,1]*xx[1]) / aa[0,0]
    
    return xx