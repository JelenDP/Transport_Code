#!/usr/bin/python3
# coding: utf-8

"""
General functions for math, IO and formating.

Created on Tue Sep 27 2023
@author: Dávid P. Jelenfi
"""

# %% IMPORTS
# System
import os
from sys import exit

# Matek
import numpy as np
import scipy.linalg as linalg

# Ábrázolás
import matplotlib.pylab as plt
from matplotlib.colors import Normalize, ListedColormap

###############################################################################
# %% General Functions
###############################################################################

###############################################################################
# %% Printing/Ploting out the results
###############################################################################
def sign(a):
    """
    Return with a sign of the input number.
    """
    if a >= 0:
        return "+"
    else:
        return "-"
    
def nm_form(n):
    """
    Write the normal form of the input number.
    
    Parameters
    --------- 
        n : float 
        
    Returns
    ---------
          : str 
            The normal form of n
    
    """
    a = '{:.18E}'.format(float(abs(n)))
    if n < 0:
        sign = "-"
    else:
        sign = " "
    e = a.find('E')
    return ' {}0.{}{}{}{:02d}'.format(sign,a[0],a[2:e],a[e:e+2],abs(int(a[e+1:])*1+1))

def TMformat(n):
    """
    Convert a number to TURBOMOLE format used in mos file.

    ...

    Parameters
    ----------
    n : float/int
        The number which you want to convert.

    Returns
    -------
    str
        The number in the TURBOMOLE format.

    """
    a = '{:.13E}'.format(float(abs(n)))
    e = a.find('E')
    if n < 0:
        return '-.{}{}D{}{:02d}'.format(a[0],a[2:e],a[e+1:e+2],abs(int(a[e+1:])*1+1))
    else:
        return '0.{}{}D{}{:02d}'.format(a[0],a[2:e],a[e+1:e+2],abs(int(a[e+1:])*1+1))

def print_M(M):
    """
    Print a matrix to the prompt
    """
    if len(M.shape) == 1:
        if M.dtype == 'float64': 
            print(' \n'.join([' {: .5e}'.format(item) for item in M]))
        elif M.dtype == 'complex128':
            print(' \n'.join([' {: .5e}'.format(item) for item in np.real(M)]))
            #print("")
            #print(' \n'.join([' {: 7.5f}'.format(item) for item in np.imag(M)]))
        print("")
        return
    
    if len(M.shape) > 2:
        M = M[0]
    if M.dtype == 'float64': 
        print(' \n'.join([' '.join([' {: .5e}'.format(item) for item in row]) 
          for row in M]))
    elif M.dtype == 'complex128':
        print(' \n'.join([' '.join([' {: .5e}'.format(item) for item in row]) 
          for row in np.real(M)]))
        #print("")
        #print(' \n'.join([' '.join([' {: .5e}i'.format(item) for item in row]) 
        #  for row in np.imag(M)]))
        
    print("")
    
def write_M(fn,M):
    """
    Write matrix (M) to file (fn)
    """
    with open(fn,"w") as f:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                f.write(f'{M[i,j]: .14e} ')
            f.write("\n")

def plot_M(M, thrd=1e-10, max_val=None, data=[]):
    """
    Plot a matrix with values below the threshold in white and the rest with a continuous colormap.
    Allows for setting a maximum value to make smaller elements more visible.
    
    Parameters:
    M (ndarray): The input matrix to plot.
    thrd (float): Threshold below which values are masked (plotted in white).
    max_val (float, optional): Maximum value for colormap normalization. If None, defaults to the maximum of M.
    data (list, optional): Contains x_lines and y_lines for grid-like visualization.
    """
    # Create a new figure for each matrix
    plt.figure()
    
    # Mask the matrix values below the threshold
    masked_matrix = np.ma.masked_where(np.abs(M) < thrd, M)
    
    # Determine the normalization limits
    vmin = np.min(masked_matrix)
    vmax = max_val if max_val is not None else np.max(masked_matrix)
    
    # Handle edge cases for uniform matrices
    if vmin == vmax:
        vmax = vmin + 1e-10
    
    # Set up the colormap
    cmap = plt.cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Add white color for values below the threshold
    cmap_with_white = cmap(np.arange(cmap.N))
    cmap_with_white = np.vstack((np.array([1, 1, 1, 1]), cmap_with_white))  # Add white to the colormap
    custom_cmap = ListedColormap(cmap_with_white)
    
    # Plot the matrix
    plt.imshow(np.zeros_like(M), cmap=ListedColormap(['white']))  # White background
    plt.imshow(masked_matrix, cmap=custom_cmap, norm=norm)  # Plot with the custom colormap
    
    # Plot optional grid lines
    if data:
        x_lines, y_lines = data
        for x in x_lines:
            plt.axvline(x=x - 0.5, color='gray', linestyle='-')  # Vertical lines
        for y in y_lines:
            plt.axhline(y=y - 0.5, color='gray', linestyle='-')  # Horizontal lines
    
    # Add a colorbar
    plt.colorbar()
    plt.show()


###############################################################################
# %% Reading QC code outputs
###############################################################################

def read_molden(fn_molden,ao_type="sao",read_coeff=True):        
    molden_atom = {"label" : [], "index" : [], "atom_number" : [], "coord" : []}  
    molden_mo = {"Sym" : [], "Energy" : [] , "Spin" : [], "Occup" : [], "Coeff": [] }
    molden_gto = []
    basis_order = {"s": [],"p": [],"d": [],"f": [],"g": []}
    
    if ao_type.lower() == "sao":
        ao_type_d = {"s": 1, "p": 3, "d": 5,"f": 7,"g": 9}
    elif ao_type.lower() == "cao":
        ao_type_d = {"s": 1, "p": 3, "d": 6,"f": 10,"g": 15}
    else:
        print("Wrong ao type! Choise SAO or CAO. ")
    
    ao_idx=0
    
    with open(fn_molden) as molden:
        at_copy = False
        gt_copy = False
        mo_copy = False
                
        for line in molden:
            if line[0] == "[":
                if line.strip().lower() == "[atoms] au":
                    at_copy = True
                    gt_copy = False
                    mo_copy = False
                    continue
                elif line.strip().lower() == "[gto]":
                    at_copy = False
                    gt_copy = True
                    mo_copy = False
                    continue
                elif line.strip().lower() == "[mo]":
                    if not read_coeff:
                        break
                    at_copy = False
                    gt_copy = False
                    mo_copy = True
                    coeff_arr = np.zeros((ao_idx,ao_idx))
                    i = 0
                    continue
                else:
                    at_copy = False
                    gt_copy = False
                    mo_copy = False
                    continue
            # elif line.strip().lower() == "":
            #     at_copy = False
            #     gt_copy = False
            #     mo_copy = False
            #     continue
            
            #Read [ATOMS] section
            if at_copy:
                molden_atom["label"].append(line.split()[0])
                molden_atom["index"].append(int(line.split()[1]))
                molden_atom["atom_number"].append(line.split()[2])
                molden_atom["coord"].append(np.array(list(np.float_(line.split()[3:6]))))
            #Read [GTO] section
            elif gt_copy:
                molden_gto.append(line)
                if  line.split() != []:
                    if line.split()[0] == "s":
                        basis_order["s"].append(ao_idx)
                        ao_idx += 1
                    elif line.split()[0] == "p":
                        for i in range(ao_type_d["p"]):
                            basis_order["p"].append(ao_idx)
                            ao_idx += 1
                    elif line.split()[0] == "d":
                        for i in range(ao_type_d["d"]): 
                            basis_order["d"].append(ao_idx)
                            ao_idx += 1
                    elif line.split()[0] == "f":
                        for i in range(ao_type_d["f"]):
                            basis_order["f"].append(ao_idx)
                            ao_idx += 1
                    elif line.split()[0] == "g":
                        for i in range(ao_type_d["g"]):
                            basis_order["g"].append(ao_idx)
                            ao_idx += 1
            #Read [MO] section  
            elif mo_copy:
                if line.split()[0].lower() == "sym=":
                    molden_mo["Sym"].append(line.split()[1])
                elif line.split()[0].lower() == "ene=":
                    molden_mo["Energy"].append(float(line.split()[1]))
                elif line.split()[0].lower() == "spin=":
                    molden_mo["Spin"].append(line.split()[1])        
                elif line.split()[0].lower() == "occup=":
                    molden_mo["Occup"].append(float(line.split()[1]))
                else:
                    j = int(line.split()[0])-1
                    coeff_arr[i,j] = float(line.split()[1])
                    if j+1 == ao_idx:
                        i += 1  

        if read_coeff:
            molden_mo["Coeff"].append(coeff_arr)
            
        return molden_atom, molden_mo, molden_gto, basis_order
            
    
def write_molden(fn, molden_atom, molden_gto, eigs, coeffs,ao_type="sao"):     
        
        with open(fn,'w') as f:
            f.write("[Molden Format]\n")
            
            #Write ATOM section
            f.write("[ATOMS] au \n")
            
            for atom_i in range(len(molden_atom['label'])):
                f.write('{0:>7}'.format(molden_atom["label"][atom_i]))
                f.write('{0:>6.0f}'.format(atom_i+1))
                f.write('{0:>3.0f}'.format(float(molden_atom["atom_number"][atom_i])))
                for coord_i in molden_atom["coord"][atom_i]:
                    f.write("      ")
                    f.write('{0:>14.8f}'.format(coord_i))
                f.write("\n")
            
            #write GTO section
            f.write("[GTO]\n")
            for line in molden_gto:
                line = line.replace("D","E")
                if (len(line.split()) == 2 and
                    float(line.split()[1]) == 0 and
                    line.split()[1].find("E")==-1 and 
                    line.split()[1].find("D")==-1 and
                    line.split()[1].find("e")==-1 and
                    line.split()[1].find("d")==-1):
                                                    f.write("{0:>3}{1:>2}\n".format(line.split()[0],line.split()[1]))
                else:
                    if len(line.split()) == 2:
                        exps = str(nm_form(float(line.split()[0]))) + str(nm_form(float(line.split()[1]))) + "\n"
                        f.write(exps)
                    if len(line.split()) == 3:
                        f.write("{0:>2}{1:>5.0f}{2:>5.2f}\n".format(
                                line.split()[0],float(line.split()[1]),float(line.split()[2])))
                    if len(line.split()) < 2:
                        f.write("\n")
            
            f.write("\n")
            if ao_type.lower() == "sao":
                f.write("[5D]\n[7F]\n[9G]\n")
            #elif ao_type.lower() == "cao": #default in MOLDEN
            #    f.write("[6D]\n[10F]\n[15G]\n")
            else:
                print("Wrong ao type! Choise SAO or CAO. ")
            #Write MO section
            f.write("[MO]\n")
            
            for st in range(len(eigs)):
                f.write(f'  Sym=  {st+1}a\n')
                f.write(f'  Ene=  {eigs[st]: 10.6f}\n')
                f.write( "  Spin= Alpha\n")
                f.write( "  Occup= 2.000\n")
                for coeff_i, coeff in enumerate(coeffs[st]):
                    f.write('{0: >4}{1:<2}{2: 12.10f}\n'.format(coeff_i+1,"",coeff))
    
def write_mos(eigval,mo_m,fn="mos2"):
    """
    Write TURBOMOLE mos file from a mos matrix.
    INPUT:
        eigval - 1D np.array - contain energy of molecular orbitals
        mo_m   - 2D np.array - contain molecular orbital coefficients (nmo,nao)
    """
    
    f = open(fn,"w")
    f.write("$scfmo    scfconv=7   format(4d20.14)\n")
    f.write("# SCF total energy is        0.0000000000 a.u.\n") #-113.7702238482 a.u.\n")
    f.write("#\n")
    nmo = mo_m.shape[0]
    nao = mo_m.shape[1]

    for i in range(nmo):
        if i != 0:
            f.write("\n")
        f.write("     {0:}  a      eigenvalue={1:}   nsaos={2:}\n".format(i+1,eigval[0],nao))
        for j in range(nao):
            val = TMformat(mo_m[i,j])
            f.write(val)
            if (j+1)%4==0:
                f.write("\n")
    f.write("\n$end\n")
    f.close()

###############################################################################
# %% Analysis tools
###############################################################################
def geo_mean(iterable):
    """
    Geometric mean of a series.

    Parameters
    --------- 
        iterable : list or np.array
                   Random list of numbers
    Returns
    ---------
          : float
            Geometric mean of the input
                   
        
    """
    a = np.array(iterable)
    #return a.prod()**(1.0/len(a))
    try:
        return a.prod()**(1.0/len(a))
    except RuntimeWarning:
        return 0

###############################################################################
# %% Matrix tools
###############################################################################
def read_M(fn):
    l = []

    with open(fn,"r") as f:
        for line in f:
            l.append(float(line))

    n = int((-1 + np.sqrt(1+8*len(l))) / 2)

    M = np.zeros((n,n))
    k = 0
    for i in range(n):
        for j in range(i+1):
            if i == j:
                M[i,j] = l[k]
            else:
                M[i,j] = l[k]
                M[j,i] = l[k]
            k += 1
    return M

def reorder_M(M,idx_list,axis=3):
    """
    Reorder a matrices based on the given idx_list
    
    Params:
    ----------
        M - 1/2D np.array
        
        idx_list - list
                    
        axis - int
            Possible values:
                - 1 : M is a 1D vector
                - 2 : M is a 2D matrix, but order only only for 2 axis
                - 3 : M is a 2D matrix, and both axis are ordered (default)
    Returns:
    -----------
        M_new - 1/2D np.array same size as the original M 
    """
    M_new = np.zeros(np.shape(M))
    if axis == 1:
        for i in range(np.shape(M)[0]):
            M_new[idx_list[i]] = M[i]
    elif axis == 2:
        for i in range(np.shape(M)[0]):
            M_new[:,idx_list[i]] = M[:,i]
    elif axis == 3:
        for i in range(np.shape(M)[0]):
            for j in range(np.shape(M)[1]):
                M_new[idx_list[i],idx_list[j]] = M[i,j]
    elif axis > 3:
        print("ERROR: The axis = %s is not valid value. Use 1,2 or 3!" % (str(axis)))
        exit()
    return M_new

def drop_line(M,idx):
    """
    Drop the idx-th element or row/column of a 1D, 2D matrices.
    
    """
    n = M.shape[0] - 1 
    
    if len(M.shape) == 1:
        M_cut = np.zeros((n))
        M_cut[:idx] = M[:idx]
        M_cut[idx:] = M[idx+1:]
        
    elif len(M.shape) == 2:
        M_cut = np.zeros((n,n))
        M_cut[:idx,:idx] = M[:idx,:idx]
        M_cut[idx:,idx:] = M[idx+1:,idx+1:]
        M_cut[:idx,idx:] = M[:idx,idx+1:]
        M_cut[idx:,:idx] = M[idx+1:,:idx]
    
    return M_cut

def is_pos_def(M):
    return np.all(linalg.eigvals(M) > -1e-10)

def is_hermitian(M):
    return np.all(np.abs(M - M.conj().T) < 1e-10)

from scipy.spatial.transform import Rotation
def align_geoms(P:np.ndarray, Q:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns two coordinates of the same system using the Kabsch algorithm.
    The order of the atoms have to be the same.
    
    Args:
        P (np.ndarray): (N,3) dimension, 'original' coordinates 
        Q (np.ndarray): (N,3) dimension, transformed coordinates

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            R (np.ndarray): rotational matrix (3,3)
            t (np.ndarray): translational vector (1,3)
    """
    # Compute optimal rotation
    cP = np.mean(P, axis=0)
    cQ = np.mean(Q, axis=0)
    
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)
    
    rot, rmsd = Rotation.align_vectors(P_centered, Q_centered)
    if rmsd > 1e-5:
        print(f'WARNING: RMSD of the alignment is large: {rmsd: .3e}.')
    # Apply rotation and get translation
    R = rot.as_matrix()
    t = cP - R @ cQ

    return R, t

###############################################################################
# %% File tools
###############################################################################
def check_file(file):
    """
    Check if a file exists or not. 
    If not: print error and make exit
    
    Parameters
    ---------
        fn : str
             Name of the file
    
    """
    if not os.path.isfile(file):
        print("ERROR: {} doesn't exists.".format(file))
        exit()

def check_file2(file):
    """
    Check if a file exists or not. 
    
    Parameters
    ---------
        fn : str
             Name of the file
    Return
    --------
        0 - if file exists
        1 - if file doesn't exist
    
    """
    if not os.path.isfile(file):
        print("ERROR: {} doesn't exists.".format(file))
        return 1

    return 0

def create_fn(fn):
    """
    Create a new file_name and check that this file
    exits or not. If it exits copy to <>_old.dat, if 
    <>_old.dat exits, too, then delete it. 
    """
    if ".dat" not in fn:
        fn_dat = fn + ".dat"
        if os.path.isfile(fn_dat):
            fn_old = fn_dat.split(".")[0] + "_old.dat"
            if os.path.isfile(fn_old):
                os.remove(fn_old)
                print(f'Remove {fn_old:}.')
            os.rename(fn_dat,fn_old)
        
    else:
        if os.path.isfile(fn):
            fn_old = fn.split(".")[0] + "_old.dat"
            if os.path.isfile(fn_old):
                os.remove(fn_old)
                print(f'Remove {fn_old:}.')
            os.rename(fn,fn_old)
    
    return fn

def get_full_path(path,cwd,full="False"):
    """
    Extend the relative path to full path.
    """
    if not cwd in path and not full:
        back = path.split("/").count("..")
        cwd2 = "/".join(cwd.split("/")[:-1*back])
        path2 = cwd2 + "/" + "/".join(path.split("/")[back:])
    else:
        path2 = path
    
    return path2


