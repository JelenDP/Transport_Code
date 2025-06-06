#!/usr/bin/python3
# coding: utf-8

"""
Functions used to calculate the Green's Functions.

Created on Tue Sep 27 2023
@author: Dávid P. Jelenfi
"""

###############################################################################
# %% Imports
###############################################################################
import numpy as np

###############################################################################
# %% Green Function
###############################################################################
def get_gf(en,H,S,Sigmas=[],typ="r",inv=True,eta=1e-5):
        """
        Create the retarded Green's function from the Hamiltonian and the Overlap matrices 
        in non orthogonal AO basis.
        If self-energies (Sigmas) presented, then the perturbed Green's function is created.
        
        ...

        Parameters
        ----------
        en : float
            Energy point where Green's function will be calculated.
        H : 2D np.array
            Hamiltonian (Fock or Kohn-Sham) matrix.
        S : 2D np.array
            Overlap matrix between atomic orbitals. 
        Sigma : list with 2D np.arrays, default: []
           List which contain the Self-energy matrices for Leads at en. 
        typ : str, optional, default: "a"
            Type of the Green's function:
                - a : advenced Green's function
                - r : retarded Green's function
        inv : bool, optinola, default: True
              if False it not makes the inversion 
                      
        Returns
        -------
        gf : 3D np.array (len(en_range),nao,nao)
            The calculated dressed Green's function.
        
        """
        
        if typ.lower() == "r":
           eta = abs(eta)
        elif typ.lower() == "a":
           eta = -1*abs(eta)
        
        z = en + 1j*eta
        Ginv = np.empty(H.shape, complex)
        Ginv = z * S - H
        if len(Sigmas) >= 1:
            for Sigma in Sigmas:
                Ginv -= Sigma
        
        if inv:
            return np.linalg.inv(Ginv)
        else:
            return Ginv

###############################################################################
def get_gf_lcao(en_range,C,E,S=np.array([]),typ="r",eta=1e-5):
    """
    Create the Green's function from LCAO coefficients (either MO or DO)
    with (general) or without (physicist) using the Overlap matrix.

    Args:
        en_range (float/list/np.array): Energy parameter
        C (2D np.array): LCAO coefficients (nao x nsts)
        E (1D np.array): Eigenvalues (nsts)
        S (2D np.array, optional): Overlap matrix. Defaults to np.array([]).
        typ (str, optional): Type of Green's function, r=retarded, a=advenced. Defaults to "r".
        eta (float, optional): Parameter for complex part. Defaults to 1e-5.

    Returns:
        (np.array): Green's function (e_num x nao x nao)
    """
    if not "array" in str(type(en_range)):
        if "float" in str(type(en_range)):
            en_range = np.array([en_range])
        elif type(en_range) == list:
            en_range = np.array(en_range)
        else:
            print("ERROR: %s is wrong type for en_range!" %(str(type(en_range))))
    en_range = en_range.astype(complex)
    if typ.lower()[0] == "a":
        en_range -= 1j*abs(eta)
    elif typ.lower()[0] == "r":
        en_range += 1j*abs(eta)
    else:
        print("Wrong type!")
    
    e_num = len(en_range)
    nao = C.shape[0]  # (nao x nsts)
    gf = np.zeros((e_num,nao,nao)).astype(complex)
    if S.size > 0: # use general definition 
        C = np.matmul(S,C)

    for ei in range(e_num):
        e_n = 1/(en_range[ei]-E)
        gf[ei] = np.einsum("in,jn,n->ij",C,C,e_n)
        
    return gf
###############################################################################

###############################################################################
def get_LeadSelfEnergy2(H_ii, H_ij, S_ii, S_ij, H_im, S_im, energy, eta=1e-4,bias=.0):
    """
    Return self-energy (sigma) evaluated at specified energy.
    ....
    Args:
        H_ii (2D np.array): onsite principal layer Hamiltonian
        H_ij (2D np.array): coupling between principal layers Hamiltonian
        S_ii (2D np.array): onsite principal layer Overlap
        S_ij (2D np.array): coupling between principal layers Overlap        
        H_im (2D np.array): coupling to the central region Hamiltonian
        S_im (2D np.array): coupling to the central region Overlap
        energy (float): energy point 
        eta (float, optional): imaginary part factor. Defaults to 1e-4.
        bias (float, optional): bias factor. Defaults to 0.0.

    Returns:
        sigma_mm (2D np.array): self-energy (sigma) evaluated at specified energy
    """    
       
    sfgf_inv = get_LeadSurfaceGFinv(H_ii, H_ij, S_ii, S_ij, energy=energy,eta=eta,bias=bias)
        
    z = energy - bias + eta * 1.j
    tau_im = z * S_im - H_im
    a_im = np.linalg.solve(sfgf_inv, tau_im)
    tau_mi = z * S_im.T.conj() - H_im.T.conj()
    sigma_mm = np.dot(tau_mi, a_im)
        
    return sigma_mm

###############################################################################
def get_LeadSelfEnergy(sfgf_inv, H_im, S_im, energy, eta=1e-4,bias=.0):
    """
    Return self-energy (sigma) evaluated at specified energy.
    ....
    Args:
        sfgf_inv (2D np.array): inverse of Lead's Surface-Green's function       
        H_im (2D np.array): coupling to the central region Hamiltonian
        S_im (2D np.array): coupling to the central region Overlap
        energy (float): energy point 
        eta (float, optional): imaginary part factor. Defaults to 1e-4.
        bias (float, optional): bias factor. Defaults to 0.0.

    Returns:
        sigma_mm (2D np.array): self-energy (sigma) evaluated at specified energy
    """    
               
    z = energy - bias + eta * 1.j
    
    #coup = z * S_im - H_im
    #sigma = coup @ np.linalg.inv(sfgf_inv) @ coup.conj().T
    sfgf = np.linalg.inv(sfgf_inv)
    
    tau_mi = z * S_im - H_im
    #tau_im = z * S_im.T - H_im.T
    sigma = tau_mi @ sfgf @ tau_mi.conj().T
        
    return sigma

###############################################################################
def get_LeadSurfaceGFinv(H_ii, H_ij, S_ii, S_ij, energy, eta=1e-4, bias=.0,thrd=1e-8):
    """
    The inverse of the retarded surface Green function
    
     Args:
        H_ii (2D np.array): onsite principal layer Hamiltonian
        H_ij (2D np.array): coupling between principal layers Hamiltonian
        S_ii (2D np.array): onsite principal layer Overlap
        S_ij (2D np.array): coupling between principal layers Overlap
        energy (float): energy point 
        eta (float, optional): imaginary part factor. Defaults to 1e-4.
        bias (float, optional): bias factor. Defaults to 0.0.
        thrd (float, optional): convergence treshold. Defaults to 1e-8.

    Returns:
        sigma_mm (2D np.array): self-energy (sigma) evaluated at specified energy
    """
    exp = abs(np.log10(eta))+2
    if exp > abs(np.log10(thrd)):
        thrd = 10**exp
    
    z = energy - bias + eta * 1.j

    v_00 = z * S_ii - H_ii #z * S_ii.T.conj() - H_ii.T.conj()
    v_11 = v_00.copy()
    v_01 = z * S_ij - H_ij
    v_10 = v_01.conj().T #z * S_ij.T - H_ij.T
    delta = thrd + 1
    while delta > thrd:
        a = np.linalg.solve(v_11, v_01)
        b = np.linalg.solve(v_11, v_10)
        v_01_dot_b = np.dot(v_01, b)
        v_00 -= v_01_dot_b
        v_11 -= np.dot(v_10, a)
        v_11 -= v_01_dot_b
        v_01 = -np.dot(v_01, a)
        v_10 = -np.dot(v_10, b)
        delta = abs(v_01).max()
    return v_00
    
    
###############################################################################

###############################################################################
def get_gf_hf_eig(mo_eigs,en_range,typ="r",eta=1e-5):
    """
    Create the Hartree Fock Green's function from MOs eigenvaule
    in MO basis.
    
    ...

    Parameters
    ----------
    mo_eigs : 1D np.array (nmo)
        The energies of molecular orbitals (usually in eV).
    en_range : 1D np.array
        The energy range where the Green's function will be calculated.
    typ : str, optional
        The type of the Green's function:
            - a : advenced Green's function
            - r : retarded Green's function
        The default is "a".

    Returns
    -------
    gf : 3D np.array (len(en_range),nmo,nao)
        The calculated Hartree Fock Green's function.

    """
        
    e_num = len(en_range)
    nmo = len(mo_eigs)

    gf = np.zeros((e_num,nmo,nmo)).astype(complex)
    
    en_range_c = en_range.astype(complex)

    if typ.lower() == "a":
        en_range_c -= 1j*eta
    elif typ.lower() == "r":
        en_range_c += 1j*eta
    else:
        print("Wrong type!")
        
    for ei in range(e_num):
        for i in range(nmo):
            gf[ei,i,i] += 1/(en_range_c[ei] - mo_eigs[i])
            
    return gf

###############################################################################
def get_gf_do_old(DO_EAs,EAs,DO_IPs,IPs,en_range,typ="r",eta=1e-5j):
    """
    Create the interacting (?) Green's function from Dyson orbitals within a given
    energy range.
    
    ...

    Parameters
    ----------
    DO_EAs : 2D np.array (n_EAs,nao)
        The Dyson orbitals belonging to the EA states.
    EAs : 1D np.array (n_EAs)
        The "excitation" energies of the EA states in eV.
    DO_IPs : 2D np.array)(n_IPs,nao)
        The Dyson orbitals belonging to the IP states.
    IPs : 1D np.array (n_IPs)
        The "excitation" energies of the IP states in eV.
    en_range : 1D np.array
        The energy range where the Green's function will be calculated.
    typ : str, optional
        The type of the Green's function:
            - a : advenced Green's function
            - r : retarded Green's function
        The default is "a".

    Returns
    -------
    gf : 3D np.array (len(en_range),nao,nao)
        The calculated Green's function.

    """
    
    en_range_c = en_range.astype(complex)
    
    e_num = len(en_range)
    
        
    if type(DO_EAs) == list:
        DO_EAs = np.array(DO_EAs)
    if type(EAs) == list:
        EAs = np.array(EAs)
    if type(DO_IPs) == list:
        DO_IPs = np.array(DO_IPs)
    if type(IPs) == list:
        IPs = np.array(IPs)
    
    nao = DO_EAs.shape[1]
    gf = np.zeros((e_num,nao,nao)).astype(complex)
    
    if typ.lower() == "a":
        en_range_c -= eta
    elif typ.lower() == "r":
        en_range_c += eta
    else:
        print("Wrong type!")
    
    for ei in range(e_num):
        #EAs
        ndo_ea = len(EAs)
        if ndo_ea != DO_EAs.shape[0]:
            print("The number of EA energies and DOs are not the same.")
            exit()
            
        for i in range(ndo_ea):
            gf[ei] += np.outer(DO_EAs[i],DO_EAs[i])/(en_range_c[ei] - EAs[i])
        
        #IPs
        ndo_ip = len(IPs)
        if ndo_ip != DO_IPs.shape[0]:
            print("The number of IP energies and DOs are not the same.")
            exit()
        
        if ndo_ip != 0:
            for i in range(ndo_ip):
                gf[ei] += np.outer(DO_IPs[i],DO_IPs[i])/(en_range_c[ei] + IPs[i])

    return gf

###############################################################################
def get_gf_do(DO_EAs,EAs,DO_IPs,IPs,en,typ="r",eta=1e-5j):
    """
    Create the interacting Green's function from Dyson orbitals within a given
    energy range.
    
    ...

    Parameters
    ----------
    DO_EAs : 2D np.array (n_EAs,nao)
        The Dyson orbitals belonging to the EA states.
    EAs : 1D np.array (n_EAs)
        The "excitation" energies of the EA states in eV.
    DO_IPs : 2D np.array)(n_IPs,nao)
        The Dyson orbitals belonging to the IP states.
    IPs : 1D np.array (n_IPs)
        The "excitation" energies of the IP states in eV.
    en : float
        The energy where the Green's function will be evaluated.
    typ : str, optional
        The type of the Green's function:
            - a : advenced Green's function
            - r : retarded Green's function
        The default is "a".

    Returns
    -------
    gf : 3D np.array (len(en_range),nao,nao)
        The calculated Green's function.

    """
        
    if type(DO_EAs) == list:
        DO_EAs = np.array(DO_EAs)
    if type(EAs) == list:
        EAs = np.array(EAs)
    if type(DO_IPs) == list:
        DO_IPs = np.array(DO_IPs)
    if type(IPs) == list:
        IPs = np.array(IPs)
    
    nao = DO_EAs.shape[1]
    gf = np.zeros((nao,nao)).astype(complex)
    
    if typ.lower() == "a":
        en -= eta
    elif typ.lower() == "r":
        en += eta
    else:
        print("Wrong type!")
    
    #EAs
    ndo_ea = len(EAs)
    if ndo_ea != DO_EAs.shape[0]:
        print("The number of EA energies and DOs are not the same.")
        exit()
        
    for i in range(ndo_ea):
        gf += np.outer(DO_EAs[i],DO_EAs[i])/(en - EAs[i])
    
    #IPs
    ndo_ip = len(IPs)
    if ndo_ip != DO_IPs.shape[0]:
        print("The number of IP energies and DOs are not the same.")
        exit()
    
    if ndo_ip != 0:
        for i in range(ndo_ip):
            gf += np.outer(DO_IPs[i],DO_IPs[i])/(en + IPs[i])

    return gf

###############################################################################

def self_energy(H,S,H_lm,S_lm,en_range,typ="r",eta=1e-5j):
        
    if typ.lower() == "a":
        eta *= -1
 
    nao = H_lm.shape[1]
    e_num = len(en_range)
    self_energy = np.zeros((e_num,nao,nao),dtype=complex)

    gf_sf = get_gf(S,H,en_range,typ=typ)
    
    for ei in range(e_num):
        z = en_range[ei] + eta * 1.j
        K_lm = z*S_lm-H_lm
        K_ml = z*S_lm.T.conj()-H_lm.T.conj()

        self_energy[ei] = np.matmul(K_ml,np.matmul(gf_sf[ei], K_lm))
        
    return self_energy

###############################################################################
def get_dos(GF,S):
    DOS = np.zeros(np.shape(GF)[0])
    for i in range(np.shape(GF)[0]):
        GS = GF[i] @ S
        DOS[i] = (-1/np.pi) * GS.trace()
    return DOS