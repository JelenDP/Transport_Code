#!/usr/bin/python3
# coding: utf-8

"""
Functions used to calculate the transmission.

Created on Tue Sep 27 2023
@author: Dávid P. Jelenfi
"""

###############################################################################
# %% Imports
###############################################################################
import numpy as np
import scipy.linalg as linalg

# %% Transsmission
def calc_trans(energy_grid, GF_r, Gamma_L, Gamma_R):
    """
    Landauer Transmission
    """
    # t0 = timer()
    trans = np.zeros((len(energy_grid)))
    for en in range(len(energy_grid)):
        trans[en] = np.real(np.trace(np.matmul(np.matmul(GF_r[en],Gamma_L[en]),np.matmul(GF_r[en].T.conj(),Gamma_R[en]))))
    # t1 = timer()
    # print("  t(for) = ",timedelta(seconds=t1-t0))
    
    # t0 = timer()
    # trans2 = np.real(np.trace(np.matmul(np.matmul(GF_r,Gamma_L),np.matmul(GF_r.transpose(0,2,1).conj(),Gamma_R)),axis1=1,axis2=2))
    # t1 = timer()
    # print("  t(one) = ",timedelta(seconds=t1-t0))
    # print("             Same?",np.allclose(trans,trans2))
                    
    return trans

def calc_t_coeff(GF_r, Gamma_L, Gamma_R):
    """
    Landauer Transmission
    """
    
    trans = np.real(np.trace(Gamma_L @ GF_r @ Gamma_R @ GF_r.T.conj()))
                    
    return trans

def wbl(gf_0,gamma=1,dim_L=[],dim_R=[],S=[],pinv=False):
    """
    Wide band limit approximation
    
    """
    
    dim = gf_0.shape
    
    if not list(dim_L):
        dim_L = [0,dim[1]]
        
    if not list(dim_R):
        dim_R = [0,dim[1]]
    
    if not list(S):
        S = np.eye(dim[1])

    Gamma_L = np.zeros(dim)
    Gamma_R = np.zeros(dim) 
    
    Gamma_L[:,dim_L[0]:dim_L[1],dim_L[0]:dim_L[1]] = gamma*S[dim_L[0]:dim_L[1],dim_L[0]:dim_L[1]]#np.eye(abs(dim_L[1]-dim_L[0]))
    Gamma_R[:,dim_R[0]:dim_R[1],dim_R[0]:dim_R[1]] = gamma*S[dim_R[0]:dim_R[1],dim_R[0]:dim_R[1]]#np.eye(abs(dim_R[1]-dim_R[0]))

    # for i in range(len(en_range)):
    #     Gamma_L[i], Gamma_R[i] = gamma*S, gamma*S
    
    Lambda_L = np.zeros(dim)
    Lambda_R = np.zeros(dim)
    
    Sigma_L = Lambda_L-0.5j*Gamma_L
    Sigma_R = Lambda_R-0.5j*Gamma_R
       
    if pinv:
        GF_r = np.zeros(dim,dtype=complex)
        for en in range(dim[0]):
            H = linalg.pinv(gf_0[en])
            GF_r[en] = linalg.pinv(H - Sigma_L[en] - Sigma_R[en])
       
        return GF_r, Gamma_L, Gamma_R
        # GF_r = []
        # Sigma_L_red, Sigma_R_red = [], []
        # Gamma_L_red, Gamma_R_red = [], []
        # 
        # for en in range(dim[0]):
        #     u,s,vh = linalg.svd(gf_0[en])
        #     idx_ns = np.where(s > 10**-10)[0]
        #     idx_s = np.where(s < 10**-10)[0]
        #     u = np.delete(u,idx_s,0)
        #     vh = np.delete(vh,idx_s,1)
            
        #     Sigma_L_red.append( u @ Sigma_L[en] @ vh )
        #     Sigma_R_red.append( u @ Sigma_R[en] @ vh )
        #     Gamma_L_red.append( u @ Gamma_L[en] @ vh )
        #     Gamma_R_red.append( u @ Gamma_R[en] @ vh )
            
        #     u = np.delete(u,idx_s,1)
        #     vh = np.delete(vh,idx_s,0)
        #     H = u @ np.diag(1/s[idx_ns]) @ vh
            
        #     GF_r.append(linalg.inv(H - Sigma_L_red[en] - Sigma_R_red[en]))
            
        # Gamma_L_red = np.array(Gamma_L_red)
        # Gamma_R_red = np.array(Gamma_R_red)
        # GF_r = np.array(GF_r)
        
        # return GF_r, Gamma_L_red, Gamma_R_red
    else:
        GF_r = np.zeros(dim,dtype=complex)
        for en in range(dim[0]):
            H = linalg.inv(gf_0[en])
            GF_r[en] = linalg.inv(H - Sigma_L[en] - Sigma_R[en])
       
        return GF_r, Gamma_L, Gamma_R