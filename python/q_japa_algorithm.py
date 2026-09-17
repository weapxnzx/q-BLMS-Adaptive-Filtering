# -*- coding: utf-8 -*-
"""
=============================================================================
Algorithm: q-JAPA (q-Jackson Affine Projection Algorithm)
Associated Paper: "A q-Jackson Affine Projection Algorithm for Robust 
                   Adaptive Filtering under Correlated Inputs"
Authors: Jusset Z. Soto-Islas et al.
=============================================================================
"""
import numpy as np

def q_japa_stream(u, d, M, K, eta, q_vec):
    """
    Generator for the q-JAPA adaptive filter.
    Processes samples simulating real-time streaming.
    
    Parameters:
    u     : Reference noise signal (array)
    d     : Primary signal (array)
    M     : Filter length
    K     : Projection order (Block size)
    eta   : Step size (Learning rate)
    q_vec : Deformation parameter vector (array of size M)
    """
    N = len(u)
    w = np.zeros(M)
    ones_vec = np.ones(M)
    
    for n in range(M + K - 1, N):
        X_n = np.zeros((M, K))
        
        for k in range(K):
            # Safe forward extraction and inversion
            x_n_minus_k = u[n - k - M + 1 : n - k + 1][::-1]
            X_n[:, k] = x_n_minus_k
            
        d_n = d[n : n - K : -1]
        
        y_n = X_n.T @ w         
        e_n = d_n - y_n         
        
        XXT_diag = np.sum(X_n**2, axis=1) 
        
        # Rigorous application of the Jackson q-derivative
        q_correction = (q_vec - ones_vec) * w * XXT_diag
        
        # Weight update recursion
        w = w + 2 * eta * (X_n @ e_n) - eta * q_correction
        
        yield n, e_n[0], w.copy()
