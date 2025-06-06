#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This collects the functions used to preform or analyse BAND calculations.

Work with BAND version 2023.102. 

Created on 19.12.2023

@author: Dávid P. Jelenfi
"""

###############################################################################
# %% Imports
###############################################################################
import os
from sys import exit
import numpy as np

class BAND:
    
    # Common parameters 
    def __init__(self,path=os.getcwd(),PBC=False):
        """
        Read the input & output data from a BAND calculation.
        
        """
        # Instance parameters 
        self.path = path            # Path
        if not os.path.isdir(self.path):
            print("ERROR: {0:} folder does not exist!".format(self.path))
        
        # Dimensions
        self.nsao = -1              # Number of basis functions
        self.PBC = PBC
        self.methods = ["BAND"]
    
    def read_HS(self):
        if self.PBC:
            self.S, S_nsao = read_M(fn=self.path + "/S.dat", rank=3)
            self.H, H_nsao = read_M(fn=self.path + "/H.dat", rank=3)
            if S_nsao != H_nsao:
                print("ERROR: The dimenson of H and S are not the same.")
            else:
                self.nsao = H_nsao
        else:
            self.S, S_nsao = read_M(fn=self.path + "/S.dat", rank=2)
            self.H, H_nsao = read_M(fn=self.path + "/H.dat", rank=2)
            if S_nsao != H_nsao:
                print("ERROR: The dimenson of H and S are not the same.")
            else:
                self.nsao = H_nsao
                
def read_M(fn, rank):
    if not os.path.isfile(fn):
        print("ERROR: S.dat not found!\n")
        exit()        
    i = 0
    M_l= []
    with open(fn,"r") as f:
        for line in f:
            if i == 2:
                dim = int(line.split()[0])
            if i > 2:
                for item in line.split():
                    M_l.append(float(item))
            i += 1
            
    if rank == 2:
        n = int(np.sqrt(dim))
        M = np.zeros((n,n))  
        k = 0
        for i in range(n):
            for j in range(n):
                M[i,j] = M_l[k]
                k += 1
    elif rank == 3:
        n = int(np.sqrt(dim/2))
        M = np.zeros((2,n,n))  
        k = 0
        for l in range(2):
            for i in range(n):
                for j in range(n):
                    M[l,i,j] = M_l[k]
                    k += 1
    
            
    return M, n
        