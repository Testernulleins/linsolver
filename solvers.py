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

    for jj in range(0, nn):
        # before moving on, swap lines if neccessary
        max_idx = np.abs(aa[jj:, jj]).argmax() + jj  # we skipped jj rows

        if jj != max_idx:
            aa[[jj, max_idx]] = np.array(aa[[max_idx, jj]])
            bb[[jj, max_idx]] = np.array(bb[[max_idx, jj]])

        # if the current value is close to zero, the matrix is linear dependent
        if np.abs(aa[jj, jj]) < 1e-13:
            return None

        # do gauss for one column
        factors = np.zeros(nn)
        for ii in range(jj+1, nn):
            factors[ii] = aa[ii, jj]/aa[jj, jj]
            aa[ii] = aa[ii] - factors[ii]*aa[jj]
            bb[ii] = bb[ii] - factors[ii]*bb[jj]

    # calculate xx
    for jj in range(nn-1, -1, -1):
        for ii in range(jj+1, nn):
            bb[jj] = bb[jj] - aa[jj, ii] * xx[ii]
        xx[jj] = bb[jj]/aa[jj, jj]

    return xx
