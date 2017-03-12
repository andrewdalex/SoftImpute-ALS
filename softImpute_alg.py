# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:55:30 2017

@author: Derek Y
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

#Requires driver script
#Broken -- has very high RMSE values
#Compute softImpute ALS based on https://github.com/cran/softImpute/blob/master/R/Ssimpute.als.R
# R: Incomplete rating matrix (n-by-m), a sparse matrix in CSR format ('0' means missing entries)
# U: Randomly chosen n-by-k dense matrix with orthonormal columns
    #U = np.zeros((num_users,k), order='F'))
    #U[:] = np.random.normal(num_users*k)
# V: Zero m-by-k dense matrix
   #V = np.zeros((num_items,k), order='F') 
# D: np.eye(k)
# k: lower rank to approximate
def Ssimpute_ALS(R, U, V, Dsq, k, thresh = 1e-05, Lambda = 0, maxit = 100):
    n,m = R.shape
    Rt = R.transpose()    
    row, col = R.nonzero()
    Xt_hat = np.zeros((m,n))
    X_hat = np.zeros((n,m))
    #nonzero = np.zeros((n,m))
    #nonzero[row,col] = 1
    A = np.dot(U, Dsq)
    B = np.dot(V, Dsq)
    ratio = 1
    iter = 0
    while((ratio>thresh) & (iter < maxit)):
        print(iter)
        iter = iter + 1
        U_old = U
        V_old = V
        Dsq_old = Dsq
        
        #B step
        b = B.transpose()
        for point in range(len(row)):
            i = row[point]
            j = col[point]
            X_hat[i,j] = R[i,j] - np.dot(A[i,:], b[:,j])
        X_star = X_hat + np.dot(A, B.transpose())
        B_tilde = np.linalg.solve(Dsq**2 + Lambda*np.eye(k), np.dot(Dsq, np.dot(U.transpose(), X_star)))
        V, D, _ = np.linalg.svd(np.dot(B_tilde.transpose(), Dsq), False)
        Dsq = np.diag(np.sqrt(D))
        B = np.dot(V, Dsq)
        
        #A step
        a = A.transpose()
        for point in range(len(row)):
            i = row[point]
            j = col[point]
            Xt_hat[j,i] = Rt[j,i] - np.dot(B[j,:], a[:,i])
        Xt_star = Xt_hat + np.dot(B, A.transpose())
        A_tilde = np.linalg.solve(Dsq**2 + Lambda*np.eye(k), np.dot(Dsq, np.dot(V.transpose(), Xt_star)))
        U, D, _ = np.linalg.svd(np.dot(A_tilde.transpose(), Dsq), False)
        Dsq = np.diag(np.sqrt(D))
        A = np.dot(U,Dsq)
        ratio = frob(U_old, Dsq_old, V_old, U, Dsq, V)
        print(ratio)
    M = X_star.dot(V)
    U, D_final, v = np.linalg.svd(M, False)
    V = np.dot(V, v)
    Dsq = D_final - Lambda
    
    return

def compute_rmse_UV(R, U, V):

    RMSE = np.sqrt((np.square(R[R.nonzero()] - np.dot(U,V.transpose())[R.nonzero()])).mean())
    return RMSE


def frob(U_old, Dsq_old, V_old, U, Dsq, V):
    denom = np.trace((Dsq_old ** 2))
    utu = np.multiply(Dsq, (U.transpose().dot(U_old)))
    vtv = np.multiply(Dsq_old, (V_old.transpose().dot(V)))
    uvprod = np.trace(utu.dot(vtv))
    num = denom + np.trace((Dsq ** 2)) - 2*uvprod
    return num/ denom