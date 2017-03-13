# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:55:30 2017

@author: Derek Y
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import scipy.linalg as spla

#Requires driver script
#Broken -- has very high RMSE values
#Compute softImpute ALS based on https://github.com/cran/softImpute/blob/master/R/Ssimpute.als.R
# R: Incomplete rating matrix (mxn), a sparse matrix in CSR format ('0' means missing entries)
# U: Randomly chosen m-by-k dense matrix with orthonormal columns
    #U = np.zeros((num_users,k), order='F'))
    #U[:] = np.random.normal(num_users*k)
# V: Zero n-by-k dense matrix
   #V = np.zeros((num_items,k), order='F')
# D: np.eye(k)
# k: lower rank to approximate
def Ssimpute_ALS(R, U, V, Dsq, k, thresh = 1e-05, Lambda = 0, maxit = 100):
    m, n = R.shape
    Rt = R.transpose()
    row, col = R.nonzero()
    Xt_star = np.zeros((n,m))
    X_star = np.zeros((m,n))
    #nonzero = np.zeros((n,m))
    #nonzero[row,col] = 1
    A = np.dot(U, Dsq)
    B = np.dot(V, Dsq)
    U_old = np.zeros((m, k))
    V_old = np.zeros((n, k))
    Dsq_old = np.eye(k)
    ratio = 1
    itr = 0
    while((thresh < ratio) and (itr < maxit)):
        print "Starting Iteration " + str(itr)
        itr = itr + 1
        U_old[:, :] = U
        V_old[:, :] = V
        Dsq_old[:, :] = Dsq

        #B step
        #b = B.transpose()
        pattern = (R != 0)
        ABt = A.dot(B.T)
        proj_ABt = pattern.multiply(ABt)
        X_star[:,:] = R.multiply(pattern) - proj_ABt + ABt

        left_side = Dsq**2 + (Lambda * np.eye(k))
        right_side = np.dot(Dsq, np.dot(U.T, X_star))
        B_tilde = np.linalg.solve(left_side, right_side)
        V[:, :], D, _ = np.linalg.svd(np.dot(B_tilde.T, Dsq), False)
        Dsq[:, :] = np.diag(np.sqrt(D))
        B[:, :] = np.dot(V, Dsq)

        #A step
        pattern = (R.transpose() != 0)
        BAt = B.dot(A.T)
        proj_BAt = pattern.multiply(BAt)
        Xt_star[:,:] = Rt.multiply(pattern) - proj_BAt + BAt
        
        left_side = Dsq**2 + (Lambda * np.eye(k))
        right_side = np.dot(Dsq, np.dot(V.T, Xt_star))
        A_tilde = np.linalg.solve(left_side, right_side)
        #could be error here with postion of U
        U[:,:], d, _ = np.linalg.svd(np.dot(A_tilde.T, Dsq), False)
        Dsq[:, :] = np.diag(np.sqrt(d))
        A[:, :] = np.dot(U,Dsq)
        ratio = frob(U_old, Dsq_old, V_old, U, Dsq, V)
        cost = compute_cost(R, A, B, Lambda)
        print "Cost => " + str(cost)
        print "Ratio => " + str(ratio)

    #Final Step: Output Solution
    M = X_star.dot(V)
    U[:,:], d, v = np.linalg.svd(M, False)
    V[:,:] = np.dot(V, v)
    Dsq[:,:] = np.diag(d - Lambda)
    Dsq[:,:] = np.fmax(Dsq, 0)
    return

def compute_cost(R, A, B, Lambda):
    pattern = R != 0
    proj_ABt = pattern.multiply((A).dot(B.T))
    cost = np.linalg.norm(proj_ABt - R, 'fro') ** 2
    norm_A = Lambda * (np.linalg.norm(A, 'fro')) ** 2
    norm_B = Lambda * (np.linalg.norm(B, 'fro')) ** 2
    return cost + norm_A + norm_B

def compute_rmse_UV(R, U, V, Dsq):
    prediction = U.dot(Dsq.dot(V.T))
    pattern = (R != 0)
    s = np.linalg.norm(pattern.multiply(prediction) - R, 'fro') ** 2
    return np.sqrt(s / R.nnz)


def frob(U_old, Dsq_old, V_old, U, Dsq, V):
    denom = np.trace((Dsq_old ** 2))
    utu = np.multiply(Dsq, (U.transpose().dot(U_old)))
    vtv = np.multiply(Dsq_old, (V_old.transpose().dot(V)))
    uvprod = np.trace(utu.dot(vtv))
    num = denom + np.trace((Dsq ** 2)) - 2*uvprod
    return num / denom
