#!/usr/bin/python3
# coding: utf-8
"""
Calculate the trassmission using NEGF formalism
and FDE or Huzinaga embedding scheme
 - TURBOMOLE FDE calculation
 - MRCC Huzinaga embedding calculation

Created on 08.02.2024
@author: Dávid P. Jelenfi
"""

###############################################################################
# %% Imports 
###############################################################################
# System ----------------------------------------------------------------------
import os, sys
import argparse
# Time ------------------------------------------------------------------------
from datetime import timedelta
from timeit import default_timer as timer
# Math ------------------------------------------------------------------------
import numpy as np
import scipy.constants as const
import scipy.linalg as linalg
# Own -------------------------------------------------------------------------
from turbomole import TM
from mrcc import MRCC, Huzinaga
from GreenFunction import get_gf, get_LeadSurfaceGFinv, get_LeadSelfEnergy, get_gf_do
###############################################################################
from general import is_pos_def, is_hermitian, create_fn, plot_M, write_M, read_M, get_full_path
import matplotlib.pylab as plt

###############################################################################
# %% MAIN
###############################################################################
def main():
    # TODO: change this argument mess to a single input file!
    ap = argparse.ArgumentParser()
    # energy range ---------------------------------------------------------------------------
    ap.add_argument("-e_min", "--e_min", type=float, required=False, default=0,
                    help="The minimum of the energy window in eV.")
    
    ap.add_argument("-e_max", "--e_max", type=float, required=False, default=0,
                    help="The maximum of the energy window in eV.")
    
    ap.add_argument("-e_num", "--e_num", type=int, required=False, default=1,
                    help="The number of the energy points.")

    ap.add_argument("-ef","--Fermi_level", action="store_true",
                help="Calculate only at Lead Fermi level.")
    
    # path -----------------------------------------------------------------------------------
    ap.add_argument("-p_full", "--full_path", action="store_true",
                    help="Use full path.")
    
    ap.add_argument("-p_lead", "--path_of_lead", type=str, required=False, default="Lead", 
                    help="Path of Lead calculation.")
    
    ap.add_argument("-p_em", "--path_of_em", type=str, required=False, default="EM",
                    help="Path of Extended Molecule molecule calculation.")  
    
    ap.add_argument("-p_m_ip", "--path_of_mol_ips", type=str, required=False, 
                    help="Relative path of embedded IP calculation for Molecule region of EM.")
    
    ap.add_argument("-p_m_ea", "--path_of_mol_eas", type=str, required=False,
                    help="Relative path of embedded EA calculation for Molecule region of EM.")
    
    # lead options -----------------------------------------------------------------------------
    ap.add_argument("-wbl", "--wide_band_limit", action="store_true",
                help="Use wide band limit approximation.")
           
    ap.add_argument("-pl", "--tm_lead_pl", type=int, required=False, default=2,
                    help="Numnber of PLs in Lead TM calculation.")

    # GF parameters --------------------------------------------------------------------------
    ap.add_argument("-eta", "--eta", type=float, required=False, default=1e-4,
                    help="The eta parameter of the GF stuff in eV.")
   
    # extended molecule options ---------------------------------------------------------------
    ap.add_argument("-low", "--low_level", action="store_true",
                    help="Use the low-level Fock matrix for the calculation.")

    ap.add_argument("-high", "--high_level", action="store_true",
                    help="Use the high-level Fock matrix for the calculation.")

    ap.add_argument("-sch", "--schema", type=int, required=False, default=2,
                    help="Embedding schema.")

    ap.add_argument("-sub", "--substitute", action="store_true",
                    help="Substitute L and R last PL parts of EM region with PL part from Lead.")
    
    ap.add_argument("-dcp", "--decouple", action="store_true",
                    help="Decouple L and R last PL part of EM region.")

    ap.add_argument("-mo","--molecular_orbitals", action="store_true",
                    help="Use Molecular Orbital basis")
    
    ap.add_argument("-do","--dyson_orbitals", action="store_true",
                    help="Use Dyson orbitals")
    
    ap.add_argument("-trc","--truncated", nargs=2, default=[None,None],
                    help="Use truncated Molecular or Dyson Orbital space. Two variable [e_max,e_min].")

    # other parameters ---------------------------------------------------------------
    ap.add_argument("-dos","--density_of_states", action="store_true",
                    help="If you want to calculate density of states, too.")
    
    # parallelisation ----------------------------------------------------------------
    ap.add_argument("-tid","--task_id", type=int, required=False,
                    help="Task id for slurm job array or other parallel runs.")
    
    ap.add_argument("-tct","--task_count", type=int, required=False,
                    help="Number of task for slurm job array or other parallel runs.")
    
    args = vars(ap.parse_args())
    # --------------------------------------------------------------------------------

    # header
    print("***************************************************", flush=True)
    print("*                                                 *", flush=True)
    print("*                TRANSMISSION CODE                *", flush=True)
    print("*                                                 *", flush=True)
    print("*  Developed by David P. Jelenfi                  *", flush=True)
    print("*  Last modification: 2025.01.15.                 *", flush=True)
    print("*                                                 *", flush=True)
    print("***************************************************", flush=True)

    # Physical constans
    eh2ev = const.physical_constants['hartree-electron volt relationship'][0]

    # energy grid
    e_min = float(args["e_min"])/eh2ev
    e_max = float(args["e_max"])/eh2ev
    e_num = int(args["e_num"])
    if e_num == 1:
        egrid = np.array([e_min])
    else:
        egrid = np.linspace(e_min,e_max,e_num)

    # paralellization
    if args["task_count"] is not None and args["task_id"] is not None:
        task_count = int(args["task_count"])
        task_id = int(args["task_id"])
        task_num = e_num//task_count
        if e_num%task_num != 0 and task_id == task_count:
            egrid = egrid[task_id*task_num:(task_id+1)*task_num+e_num%task_num]
        else:
            egrid = egrid[task_id*task_num:(task_id+1)*task_num]
        e_num = task_num
        e_min, e_max = np.min(egrid), np.max(egrid)
        print(f'TASK-ID: {task_id}, e_min: {e_min*eh2ev: 3.4f}, e_max: {e_max*eh2ev: 3.4f}, e_num: {e_num}')

    # set paths
    print("")
    print("Location of the electronic structure calculations:")
    # use full path
    cwd = os.getcwd()
    Lead_path = get_full_path(str(args["path_of_lead"]),cwd,args["full_path"])
    print("  Lead:               ", Lead_path)
    
    EM_path = get_full_path(str(args["path_of_em"]),cwd,args["full_path"])
    print("  Extended Molecule:  ", EM_path)

    if args["dyson_orbitals"]:
        M_IPs_path = get_full_path(str(args["path_of_mol_ips"]),cwd,args["full_path"])
        print("  IPs of Molecule:    ", M_IPs_path)

        M_EAs_path = get_full_path(str(args["path_of_mol_eas"]),cwd,args["full_path"])
        print("  EAs of Molecule:    ", M_EAs_path)
    
    ###########################################################################
    # Local variables
    ###########################################################################
    # for Leads
    TM_PL = int(args["tm_lead_pl"]) # Number of PLs in TURBOMOLE Lead calculation
    read_sfgf = False               # read SFGFs from file
    part_sfgf = False               # only a part of the SFGFs are precalculated
    
    # write SFGFs to file
    if args["task_id"] is not None:
        write_sfgf = False  # TODO: write from multiple task is not implemented! 
    else:
        write_sfgf = not read_sfgf
    
    #if args["task_id"] is not None:
    #    sfgf_fn_L = f'lead_sfgf_L_{task_id}.bin'   # Left SFGF of the Lead
    #    sfgf_fn_R = f'lead_sfgf_R_{task_id}.bin'   # Right SFGF of the Lead
    #    fn_egrid  = f'egrid_{task_id}.npy'         # corresponding energy grid 
    #else:
    sfgf_fn_L = "lead_sfgf_L.bin"   # Left SFGF of the Lead
    sfgf_fn_R = "lead_sfgf_R.bin"   # Right SFGF of the Lead
    fn_egrid = "egrid.npy"          # corresponding energy grid

    # for GF
    eta=float(args["eta"]) #1e-5/eh2ev #0.02 eV in Thygensen paper, should we try this instead??
    bias=.0
    
    # debug
    debug = True
    
    # Dyson orbital
    cut_cont = False # cut out continuum orbital (True) or not (False)
    do_norm = True   # normalize them (True) or not (False)

    ###########################################################################
    # Start code
    ###########################################################################    
    #--------------------------------------------------------------------------
    print("",flush=True)
    print("  1. Prepare Lead(s)", flush=True) 
    if debug: t0 = timer()
    # read the output of the (1D) periodic calculation made with TM riper for the Lead(s)
    Lead = TM(path=Lead_path)
    Lead.read_HS()
    if TM_PL == 1:
        print("Use at least 2 PL for Lead calculation!")
        exit()
    elif TM_PL > 1:
        # struct
        #   | H_pl    | H_pl_ij |
        #   | H_pl_ji | H_pl    |           
        H_unit, S_unit = Lead.H, Lead.S
        nsao_pl  = Lead.nsao // TM_PL
        H_pl    = H_unit[:nsao_pl,:nsao_pl]
        H_pl_ij = H_unit[:nsao_pl,nsao_pl:]
        H_pl_ji = H_unit[nsao_pl:,:nsao_pl]
        S_pl    = S_unit[:nsao_pl,:nsao_pl]
        S_pl_ij = S_unit[:nsao_pl,nsao_pl:]
        S_pl_ji = S_unit[nsao_pl:,:nsao_pl]
    else:
        print("Error: invalide number of PL", flush=True)
        return
    
    if args["Fermi_level"]:
        Lead.read_fermi()
        Lead.Fermi = Lead.Fermi
        print(f'     - Transmission will be calculated only at Lead Fermi level ({Lead.Fermi*eh2ev: 7.5f} eV).', flush=True)
        egrid = np.array([Lead.Fermi])

    if debug: print(f'     t = {timedelta(seconds=timer()-t0)}', flush=True)
    #--------------------------------------------------------------------------
    if debug: t0 = timer()
    print("",flush=True)
    print("  2. Prepare EM region", flush=True)

    # EM super system calculation: 1PL(Bulky) --- Electrode --- Molecule --- Electrode --- 1PL(Bulky)
    # Electrode: surface + at least 1PL (should be enough to make the last one "bulky")
    EM = Huzinaga(path=EM_path)
    S_em = EM.S.copy()                 # AO overlap of EM
    nsao_em = EM.nsao                  # number of SAO of the EM
    nsao_m = EM.nsao_a                 # number of SAO of the M
    nlmo_m = EM.nocc_a + EM.nvir_a     # number of localised MOs of the M
    nsao_els = EM.nsao_b               # number of SAO of the Els (both) containing tip, surface, etc.
    if nsao_els % 2 != 0:
        print("WARNING: electrode parts in EM are not symmetric.")
    else:
        nsao_el = int(nsao_els / 2)    # number of SAO of one El
    
    # check PL part of the EM 
    diff_L   = np.max(abs(S_em[:nsao_pl,:nsao_pl] - S_pl))
    diff_R   = np.max(abs(S_em[-1*nsao_pl:,-1*nsao_pl:] - S_pl))
    if diff_L > 1e-8 or diff_R > 1e-8:
        print(f'     - Structure of Lead part of EM is differ from sturcture of Lead :  L={diff_L: .2e}, R={diff_R: .2e} ', flush=True)
    del diff_L, diff_R    
    if debug: print(f'     t = {timedelta(seconds=timer()-t0)}', flush=True)
    
    #--------------------------------------------------------------------------
    # Prepar the effective Hamiltonian of the EM
    if debug: t0 = timer()
    # Prepare Dyson orbitals for M
    if args["dyson_orbitals"]:
        #sys.exit("ERROR: calculation with dyson orbitals are not yet implemented!")
        print("     - WFT-in-DFT embedding")
        print("     - Prepare Dyson orbitals of IPs and EAs of embedded M", flush=True)
        
        do_norm = False #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        M_IPs = MRCC(path=M_IPs_path,sub="em3")
        M_EAs = MRCC(path=M_EAs_path,sub="em3")
        
        if M_IPs.nsao != M_EAs.nsao:
            print("ERROR: IP calculation has different number of SAO then EA!", flush=True)
        if nsao_em != M_EAs.nsao:
            print(f'ERROR: mistmatch in the number of SAOs in the DFT ({nsao_m}) and WF ({M_EAs.nsao}) embedded molecule', flush=True)        
        
        # Dyson-orbitals for the molecule
        
        if M_IPs.ccprog == "cis":
        
            M_IPs.read_dos(norm=do_norm)
            M_IPs.do_eigs = -1*M_IPs.do_eigs # flip the sign of IP states
            #M_IPs.do_eigs = M_IPs.do_eigs[:-1] # don't flip!
            M_IPs.dos = M_IPs.dos
            idx_ip = np.argsort(M_IPs.do_eigs)
            nips = len(M_IPs.do_eigs)

            M_EAs.read_dos(norm=do_norm)
            idx_ea = np.argsort(M_EAs.do_eigs)
            neas = len(M_EAs.do_eigs)

            ndos = nips + neas

            DOs = np.zeros((ndos,EM.nsao))
            DOs[:nips] = M_IPs.dos[idx_ip][-1*nips:]
            DOs[nips:] = M_EAs.dos[idx_ea][:neas]

            DO_eigs = np.zeros((ndos))
            DO_eigs[:nips] = M_IPs.do_eigs[idx_ip][-1*nips:]
            DO_eigs[nips:] = M_EAs.do_eigs[idx_ea][:neas]

            # truncation to the given energy range
            nips_trc = nips
            if args["truncated"][0] != None:
                nips_trc = int(args["truncated"][0])
                #min_trc = float(args["truncated"][0])
                #print(min_trc)
                #ip_idxs_trc = np.where(DO_eigs[:nips]*eh2ev > min_trc)[0]
                #print(ip_idxs_trc)
                #nips_trc = len(ip_idxs_trc)

            neas_trc = neas     
            if args["truncated"][1] != None:
                neas_trc = int(args["truncated"][1])
                #max_trx = float(args["truncated"][1]) 
                #print(max_trx)
                #ea_idxs_trc = np.where(DO_eigs[nips:]*eh2ev < max_trx)[0]+nips_trc
                #print(ea_idxs_trc)
                #neas_trc = len(ea_idxs_trc)

            if nips_trc != nips or neas_trc != neas:
                ndos_trc = nips_trc + neas_trc

                DOs_trc = np.zeros((ndos_trc,EM.nsao))
                #DOs_trc[:nips_trc] = DOs[ip_idxs_trc]
                #DOs_trc[nips_trc:] = DOs[ea_idxs_trc]
                DOs_trc[:nips_trc] = M_IPs.dos[idx_ip][-1*nips_trc:]
                DOs_trc[nips_trc:] = M_EAs.dos[idx_ea][:neas_trc]

                DO_eigs_trc = np.zeros((ndos_trc))
                #DO_eigs_trc[:nips_trc] = DO_eigs[ip_idxs_trc]
                #DO_eigs_trc[nips_trc:] = DO_eigs[ea_idxs_trc]
                DO_eigs_trc[:nips_trc] = M_IPs.do_eigs[idx_ip][-1*nips_trc:]
                DO_eigs_trc[nips_trc:] = M_EAs.do_eigs[idx_ea][:neas_trc]

                DOs = DOs_trc
                DO_eigs = DO_eigs_trc

                nips = nips_trc
                neas = neas_trc
                ndos = nips_trc + neas_trc

            # transform the DOs to the Huzinaga basis
            # TODO: get directly these vectros from MRCC
            DOs_in_ho =  DOs @ EM.S @ HOs[:nhos].T

            # DOs overlap
            S_do = DOs_in_ho @ DOs_in_ho.T

            # project out the HF contribution 
            # it's necessary beacuse we don't have all DOs
            P = DOs_in_ho.T @ linalg.inv(S_do) @ DOs_in_ho

        elif M_IPs.ccprog == "mrcc":
            
            M_IPs.read_dos(norm=do_norm,typ="IP")
            #M_IPs.do_eigs = -1*M_IPs.do_eigs # flip the sign of IP states
            print(M_IPs.dos)
            print(M_IPs.do_eigs)
            idx_ip = np.argsort(M_IPs.do_eigs[0])
            nips = len(M_IPs.do_eigs[0])

            M_EAs.read_dos(norm=do_norm,typ="EA")
            idx_ea = np.argsort(M_EAs.do_eigs[0])
            neas = len(M_EAs.do_eigs[0])

            ndos = nips + neas

            DOs_R = np.zeros((ndos,EM.nsao))
            DOs_R[:nips] = M_IPs.dos[0][idx_ip][-1*nips:]
            DOs_R[nips:] = M_EAs.dos[0][idx_ea][:neas]

            DOs_L = np.zeros((ndos,EM.nsao))
            DOs_L[:nips] = M_IPs.dos[1][idx_ip][-1*nips:]
            DOs_L[nips:] = M_EAs.dos[1][idx_ea][:neas]

            DO_eigs_R = np.zeros((ndos))
            DO_eigs_R[:nips] = M_IPs.do_eigs[0][idx_ip][-1*nips:]
            DO_eigs_R[nips:] = M_EAs.do_eigs[0][idx_ea][:neas]
            
            DO_eigs_L = np.zeros((ndos))
            DO_eigs_L[:nips] = M_IPs.do_eigs[1][idx_ip][-1*nips:]
            DO_eigs_L[nips:] = M_EAs.do_eigs[1][idx_ea][:neas]
            
            # DOs overlap
            S_do = DOs_L @ DOs_R.T
            
            # project out the HF contribution 
            # it's necessary beacuse we don't have all DOs
            P = DOs_L.T @ linalg.inv(S_do) @ DOs_R

        # check of dimensions
        print(f'     - Number of AOs on M: {EM.nsao_a:}')
        nhos = EM.nocc_a + EM.nvir_a
        print(f'     - Number of HOs of M: {nhos:}')
        print(f'           - occupied HOs: {EM.nocc_a:}')
        print(f'           - virtual HOs:  {EM.nvir_a:}')
        print(f'     - Number of used DOs: {ndos:}')
        print("     - Used DO energies: ")
        for i in range(ndos):
            if M_IPs.ccprog == "cis":
                print(f'        {DO_eigs[i]*eh2ev : 3.4f}')
                if i == nips-1:
                    print("        ----------")
            elif M_IPs.ccprog == "mrcc":
                print(f'        {DO_eigs_R[i]*eh2ev : 3.10f}  {DO_eigs_L[i]*eh2ev : 3.10f}')
                if i == nips-1:
                    print("        -----------------------------")



        #H_do = EM.S @ DOs.T @ np.diag(DO_eigs) @ DOs @ EM.S
        #
        #H_em = EM.em1.H.copy()
        ## substitute the M part with the high-level
        #H_em[np.ix_(EM.iaos_a,EM.iaos_a)] = H_do[np.ix_(EM.iaos_a,EM.iaos_a)]

        # read the Huzinaga matrix, orbitals, and energies
        EM.read_Huzinaga()
                   
        # Huzinaga energies of the M 
        HO_eigs_m = EM.em3_huzi.mo_eigs[EM.nocc_b:EM.nocc+EM.nvir_a]
        
        HOs = np.zeros((EM.nsao,EM.nsao))
        
        HOs[:EM.nocc_a]                            = EM.em3_huzi.mos[EM.nocc_b:EM.nocc]          # M occ. HOs
        HOs[EM.nocc_a:EM.nocc_a+EM.nvir_a]         = EM.em3_huzi.mos[EM.nocc:EM.nocc+EM.nvir_a]  # M vir. HOs
        HOs[EM.nocc_a+EM.nvir_a:EM.nocc+EM.nvir_a] = EM.em3_huzi.mos[:EM.nocc_b]                 # El occ. HOs
        HOs[EM.nocc+EM.nvir_a:]                    = EM.em3_huzi.mos[EM.nocc+EM.nvir_a:]         # El vir. HOs
        
        #U = DOs_in_ho.T @ S_do @ DOs_in_ho
        
        # transform the low-level Fock into the Huzinaga basis
        F_in_ho = HOs @ EM.em1.H @ HOs.T
        
        # 1. substitute M part with HF HO energies 
        F_in_ho[:nhos,:nhos]  = np.diag(HO_eigs_m)
        
        # 2. project out the HF contribution
        F_in_ho[:nhos,:nhos] -= P @ np.diag(HO_eigs_m)
        F_in_ho[:nhos,:nhos] -=     np.diag(HO_eigs_m) @ P
        F_in_ho[:nhos,:nhos] += P @ np.diag(HO_eigs_m) @ P  
        
        #F_in_ho[:EM.nsao_a,:EM.nsao_a] -= U @ np.diag(HO_eigs_m) @ U
        
        # 3. add DO contributions
        if M_IPs.ccprog == "cis":
            F_in_ho[:nhos,:nhos] += DOs_in_ho.T @ np.diag(DO_eigs) @ DOs_in_ho
        elif M_IPs.ccprog == "mrcc":
            F_in_ho[:nhos,:nhos] += DOs_L.T @ np.diag(DO_eigs_R) @ DOs_R           
        
        # 4. transform back to AO
        H_em = EM.S @ HOs.T @ F_in_ho @ HOs @ EM.S            
        
    ## change M localised occupied orbitals to the high-level dyson orbitals
    #F_loc_em[EM.nocc_b:EM.nocc,EM.nocc_b:EM.nocc] = np.diag(IP_do_eigs)
    #mos_loc_em[EM.nocc_b:EM.nocc,:] = IP_dos 
    ## change M localised virtual orbitals to the high-level dyson orbitals
    #F_loc_em[EM.nocc_b:EM.nocc,EM.nocc_b:EM.nocc] = np.diag(EA_do_eigs)
    #mos_loc_em[EM.nocc_b:EM.nocc,:] = EA_dos 
    ## transform back to AO basis with the modified localised orbitals
    #H_em = S_em @ mos_loc_em.T @ F_loc_em @ mos_loc_em @ S_em 
          
    # DFT/HF-in-DFT embedding
    #  1. using localisation of the Fock matrix
    #elif args["molecular_orbitals"]:
    #    print("     - DFT/HF-in-DFT embedding based on LMOs")
    #    # Create the mixed localised orb. (Huzinaga operator eigenvectors) basis
    #    # -> low-level loc. orbs for El
    #    # -> high-level loc. orbs for M
    #    
    #    # "high-level" Huzinaga matrix
    #    EM.read_Hh()  
    #    
    #    # "low-level" Huzinaga matrix 
    #    # -> built from low-level Fock matirx
    #    # -> using loc. orbs. after 1. SCF for the projectors  
    #    EM.build_Hh() 
    #    # low-level loc. orbs for M
    #    lmos_m_l = np.zeros((EM.em3_huzi.norb_a,EM.nsao))             
    #    lmos_m_l[:EM.nocc_a] = EM.mos_loc[EM.nocc_b:EM.nocc]        # occ
    #    lmos_m_l[EM.nocc_a:] = EM.mos_loc[EM.nocc+EM.nvir_b:]       # virt
    #    # high-level loc. orbs for M
    #    lmos_m_h = np.zeros((EM.em3_huzi.norb_a,EM.nsao))               
    #    lmos_m_h[:EM.nocc_a] = EM.em3_huzi.mos[EM.nocc_b:EM.nocc]   # occ.
    #    lmos_m_h[EM.nocc_a:] = EM.em3_huzi.mos[EM.nocc+EM.nvir_b:]  # virt.
    #    # Overlap between high- and low-level loc. orbs of M
    #    ov_lmos_lh = lmos_m_h @ EM.S @ lmos_m_l.T
    #    # correct the relative phase -> is it necessary?
    #    phase = np.sign(np.sum(ov_lmos_lh,axis=0))
    #    lmos_m_h = lmos_m_h * phase[:,np.newaxis]
    #    ov_lmos_lh = lmos_m_h @ EM.S @ lmos_m_l.T # recalc with the "corrected phase"
    #    # all low-level loc. orbs
    #    lmos_tild = EM.mos_loc.copy() 
    #    lmos_tild[EM.nocc_b:EM.nocc] = lmos_m_h[:EM.nocc_a] # high-level occ. loc. orbs of M
    #    lmos_tild[EM.nocc+EM.nvir_b:] = lmos_m_h[EM.nocc_a:] # high-level occ. loc. orbs of M
    #    # build the mixed Fock matrix
    #    # -> low-level for El, 
    #    # -> low-level for coupling El-El and El-M coupling (is it true?)
    #    # -> high-level for M
    #    F_loc_tild = EM.F_loc.copy()
    #    # energies of M substituted with high-level energies
    #    F_loc_tild[EM.nocc_b:EM.nocc,EM.nocc_b:EM.nocc]   = np.diag(EM.em3_huzi.mo_eigs[EM.nocc_b:EM.nocc])    # high-level loc. occ. of M
    #    F_loc_tild[EM.nocc+EM.nvir_b:,EM.nocc+EM.nvir_b:] = np.diag(EM.em3_huzi.mo_eigs[EM.nocc+EM.nvir_b:])   # high-level loc. virt. of M
    #    # El-M coupling part is "transformed" with overlap between high- and low-level loc. orbs of M
    #    # -> original coupling describes the coupling between low-level loc. orbs. of El and M
    #    F_loc_tild[EM.nocc_b:EM.nocc,:EM.nocc_b] = ov_lmos_lh[:EM.nocc_a,:EM.nocc_a] @ F_loc_tild[EM.nocc_b:EM.nocc,:EM.nocc_b]
    #    F_loc_tild[:EM.nocc_b,EM.nocc_b:EM.nocc] = F_loc_tild[:EM.nocc_b,EM.nocc_b:EM.nocc] @ ov_lmos_lh[:EM.nocc_a,:EM.nocc_a].T
    #    F_loc_tild[EM.nocc+EM.nvir_b:,EM.nocc:EM.nocc+EM.nvir_b] = ov_lmos_lh[EM.nocc_a:,EM.nocc_a:] @ F_loc_tild[EM.nocc+EM.nvir_b:,EM.nocc:EM.nocc+EM.nvir_b] 
    #    F_loc_tild[EM.nocc:EM.nocc+EM.nvir_b,EM.nocc+EM.nvir_b:] = F_loc_tild[EM.nocc:EM.nocc+EM.nvir_b,EM.nocc+EM.nvir_b:] @ ov_lmos_lh[EM.nocc_a:,EM.nocc_a:].T
    #    F_loc_tild[EM.nocc:EM.nocc+EM.nvir_b,EM.nocc_b:EM.nocc] = F_loc_tild[EM.nocc:EM.nocc+EM.nvir_b,EM.nocc_b:EM.nocc] @ ov_lmos_lh[:EM.nocc_a,:EM.nocc_a].T
    #    F_loc_tild[EM.nocc_b:EM.nocc,EM.nocc:EM.nocc+EM.nvir_b] = ov_lmos_lh[:EM.nocc_a,:EM.nocc_a] @ F_loc_tild[EM.nocc_b:EM.nocc,EM.nocc:EM.nocc+EM.nvir_b]
    #    F_loc_tild[EM.nocc+EM.nvir_b:,:EM.nocc_b] = ov_lmos_lh[EM.nocc_a:,EM.nocc_a:] @ F_loc_tild[EM.nocc+EM.nvir_b:,:EM.nocc_b] 
    #    F_loc_tild[:EM.nocc_b,EM.nocc+EM.nvir_b:] = F_loc_tild[:EM.nocc_b,EM.nocc+EM.nvir_b:] @ ov_lmos_lh[EM.nocc_a:,EM.nocc_a:].T
    #    # transform back to AO basis with the modified localised orbitals
    #    H_em = EM.S @ lmos_tild.T @ F_loc_tild @ lmos_tild @ EM.S
    
    elif args["molecular_orbitals"]:
        print("     - DFT/HF-in-DFT embedding based on LMOs")
        EM.read_Huzinaga()  # read high-level "Huzinaga" matrix (there are PFP terms)
        
        # EM.build_Huzinaga() # build low-level Huzinaga matrix (there isn't any PFP terms)
        
        E_m_huz = EM.em3_huzi.mo_eigs[EM.nocc_b:EM.nocc+EM.nvir_a]
        
        nhos_o = EM.nocc_a
        nhos_v = EM.nvir_a
        if args["truncated"][0] != None:
            e_min = float(args["truncated"][0])
            
            nhos_o = len(np.where(E_m_huz[:EM.nocc_a]*eh2ev > e_min)[0])

            if nhos_o > EM.nocc_a:
                nhos_o = EM.nocc_a
                
        if args["truncated"][1] != None:
            e_max = float(args["truncated"][1])
            
            nhos_v = len(np.where(E_m_huz[EM.nocc_a:]*eh2ev < e_max)[0])
            
            if nhos_v > EM.nvir_a:
                nhos_v = EM.nvir_a
        
        M_nocc = nhos_o
        M_nvir = nhos_v
        nhos= nhos_o + nhos_v
        
        print("Used M HOs: ",E_m_huz[EM.nocc_a-nhos_o:EM.nocc_a+nhos_v],flush=True)
            
        # make the transformation matrix: Huzinaga orbitals (HOs)
        L_huz = np.zeros((EM.nsao,EM.nsao))

        if EM.nsao_a != EM.nocc_a+EM.nvir_a:
            print(EM.nsao_a - EM.nocc_a+EM.nvir_a)

        if EM.version == 2:
            L_huz[EM.nocc_a-nhos_o:EM.nocc_a]    = EM.em3_huzi.mos[EM.nocc-nhos_o:EM.nocc]  # M occ.
            L_huz[EM.nocc_a:EM.nocc_a+nhos_v]    = EM.em3_huzi.mos[EM.nocc:EM.nocc+nhos_v]  # M vir.

            L_huz[EM.nocc_a+EM.nvir_a:EM.nocc_a+EM.nvir_a+EM.nocc_b] = EM.em3_huzi.mos[:EM.nocc_b]              # El occ.
            L_huz[EM.nocc_a+EM.nvir_a+EM.nocc_b:]         = EM.em3_huzi.mos[EM.nocc+EM.nvir_a:]      # El vir.
        
        elif EM.version == 1:
            L_huz[EM.nocc_a-nhos_o:EM.nocc_a]    = EM.em3_huzi.mos[EM.nocc-nhos_o:EM.nocc]  # M occ.
            L_huz[EM.nocc_a:EM.nocc_a+nhos_v]    = EM.em3_huzi.mos[EM.nocc+EM.nvir_b:EM.nocc+EM.nvir_b+nhos_v]  # M vir.

            L_huz[EM.nocc_a+EM.nvir_a:EM.nocc_a+EM.nvir_a+EM.nocc_b] = EM.em3_huzi.mos[:EM.nocc_b]              # El occ.
            L_huz[EM.nocc_a+EM.nvir_a+EM.nocc_b:]          = EM.em3_huzi.mos[EM.nocc:EM.nocc+EM.nvir_b]      # El vir.            
        
        # transform the low-level Fock matrix to HO basis
        F_huz = L_huz @ EM.em1.H @ L_huz.T

        # substitute the M part with the "Huzinaga eneriges"
        F_huz[:nhos,:nhos] = np.diag(E_m_huz[EM.nocc_a-nhos_o:EM.nocc_a+nhos_v])

        # transform back to AO basis
        H_em = EM.S @ L_huz.T @ F_huz @ L_huz @ EM.S

    elif False:
        print("     - DFT/HF-in-DFT embedding based on LMOs")
        # Embedded molecule 
        EM.read_Hh()
        
        lmos_m = np.zeros((EM.em3_huzi.norb_a,EM.nsao))

        lmos_m[:EM.nocc_a,:] = EM.em3_huzi.mos[EM.nocc_b:EM.nocc,:]    # occ.  high-level loc. orbs. of M 
        lmos_m[EM.nocc_a:,:] = EM.em3_huzi.mos[EM.nocc+EM.nvir_b:,:]   # virt. high-level loc. orbs. of M

        mo_eigs_m = np.zeros((EM.em3_huzi.norb_a))

        mo_eigs_m[:EM.nocc_a] = EM.em3_huzi.mo_eigs[EM.nocc_b:EM.nocc]   # energies of occ.  high-level loc. orbs. of M 
        mo_eigs_m[EM.nocc_a:] = EM.em3_huzi.mo_eigs[EM.nocc+EM.nvir_b:]  # energies of virt. high-level loc. orbs. of M
       
        # truncate orb. space of the molecule 
        # mainly due to testing of convergence with number of orbs.
        if (args["truncated"][0] != None or args["truncated"][1] != None):
            print("     - Use truncated orbital space for the molecule")
            M_HOMO = mo_eigs_m[EM.nocc_a]
            M_LUMO = mo_eigs_m[EM.nocc_a+1]
            M_gap_mid = ( M_HOMO + M_LUMO ) / 2
            if args["truncated"][0] != None:
                M_occ_eig_min = float(args["truncated"][0])/eh2ev + M_gap_mid # minimum energy of occ. MOs
                idx = np.where(mo_eigs_m < M_occ_eig_min)
                mo_eigs_m[idx] = 0.0
                lmos_m[idx,:] = 0.0

            if args["truncated"][1] != None:
                M_vir_eig_max = float(args["truncated"][1])/eh2ev + M_gap_mid # maximum energy of vir. MOs
                idx = np.where(mo_eigs_m > M_vir_eig_max)
                mo_eigs_m[idx] = 0.0
                lmos_m[idx,:] = 0.0

            M_nocc = ( mo_eigs_m != 0.0 and mo_eigs_m <= M_HOMO).sum()
            M_nvir = ( mo_eigs_m != 0.0 and mo_eigs_m >= M_LUMO).sum()
            
            # some printing
            print(f'      \-> M HOMO energy: {M_HOMO*eh2ev: 7.5f} eV', flush=True)
            print(f'      \-> M LUMO energy: {M_LUMO*eh2ev: 7.5f} eV', flush=True)
            print(f'      \-> M HOMO-LUMO gap {(M_HOMO-M_LUMO)*eh2ev: 7.5f} eV', flush=True)
            print(f'      \-> Use truncated space with {M_nocc:3d} occupied and {M_nvir:3d} virtual orbital', flush=True)
            print(f'      \-> Last occupied energy: {np.min(mo_eigs_m)*eh2ev: 7.5f} eV', flush=True)
            print(f'      \-> Last virtual energy: {np.max(mo_eigs_m)*eh2ev: 7.5f} eV', flush=True)

        # create the AO basis Huzinaga matrix only from the high-level loc. orbs. and energies of M
        Hh_em = S_em @ lmos_m.T @ np.diag(mo_eigs_m) @ lmos_m @ S_em
        
        print("       F = F^tild[D^tild_A] for M from high-level localised orbitals", flush=True)
        print("         = F_II,0[D_AB]     for El, El-El and M-El", flush=True)
        # use schema2
        # use the low-level for El, El-El and El-Ml parts
        H_em = EM.em1.H.copy()
        
        # substitute the M part with the high-level
        H_em[np.ix_(EM.iaos_a,EM.iaos_a)] = Hh_em[np.ix_(EM.iaos_a,EM.iaos_a)]
    
    elif args["low_level"]:
        print("     - Use the low-level Fock matrix", flush=True)
        H_em = EM.em1.H.copy()

    elif args["high_level"]:
        print("     - Use the high-level Fock matrix", flush=True)
        H_em = EM.em3.H.copy()
    
    else:
        print("     - DFT/HF-in-DFT embedding", flush=True)

        if args["schema"] == 1:
            # 1. -------------------------------------------------------------------------
            print("       F = F^tild[D^tild_A] for M and M-El")
            print("         = F_II,0[D_AB]     for El and El-El")
            # substitute the El parts and the El-El parts with the low-level
            # High-level Fock matrix with embedding potential (F^tild[D^tild_A] in paper)
            H_em = EM.em3.H.copy()
            # Substitute the enviroment part with the low-level
            H_em[np.ix_(EM.iaos_b,EM.iaos_b)] = EM.em1.H[np.ix_(EM.iaos_b,EM.iaos_b)] 

        elif args["schema"] == 2:
            # 2. -------------------------------------------------------------------------
            print("       F = F^tild[D^tild_A] for M", flush=True)
            print("         = F_II,0[D_AB]     for El, El-El and M-El", flush=True)
            H_em = EM.em1.H.copy()
            # substitute the M part with the high-level
            H_em[np.ix_(EM.iaos_a,EM.iaos_a)] = EM.em3.H[np.ix_(EM.iaos_a,EM.iaos_a)]
    
    if debug: print(f'     t = {timedelta(seconds=timer()-t0)}', flush=True)
    if debug: t0 = timer()
    #--------------------------------------------------------------------------
    # Modifications of Hamiltonian and Overlap of EM
    # substitute first/last PL region in the EM with the PL from the Lead calculation
    if args["substitute"]:
        print("     - Approximate PL part of EM region", flush=True)
        H_em[:nsao_pl,:nsao_pl]                  = H_pl
        H_em[-1*nsao_pl:,-1*nsao_pl:]            = H_pl
    # calculate the difference between PL in EM and PL in Lead 
    else:
        diff_L_max   = np.max(abs(H_em[:nsao_pl,:nsao_pl] - H_pl))
        diff_L_mean  = np.mean(abs(H_em[:nsao_pl,:nsao_pl] - H_pl))
        diff_R   = np.max(abs(H_em[-1*nsao_pl:,-1*nsao_pl:] - H_pl))
        #if diff_L > 1e-8 or diff_R > 1e-8:
        print(f'     - Difference between H_PL part in EM and in Lead:  L={diff_L_max: .2e} ( {diff_L_mean: .2e} ), R={diff_R: .2e} ', flush=True)
        del diff_L_max, diff_L_mean, diff_R

    # and turn-off the coupling between the Leads
    if args["decouple"]:
        print("     - Decouple the Leads", flush=True)
        H_em[:nsao_pl,-1*nsao_pl:]               = 0  
        H_em[-1*nsao_pl:,:nsao_pl]               = 0
        # the below modification makes S no longer positive definite!!
        S_em[:nsao_pl,-1*nsao_pl:]               = 0  
        S_em[-1*nsao_pl:,:nsao_pl]               = 0
    else:
        coup_R_L_max = np.max(abs(H_em[:nsao_pl,-1*nsao_pl:]))
        coup_R_L_mean = np.mean(abs(H_em[:nsao_pl,-1*nsao_pl:]))
        #if coup_R_L_max > 1e-8:
        print(f'     - Coupling between Left and Right PL in EM: {coup_R_L_max: .2e} {coup_R_L_mean: .2e}', flush=True)
        del coup_R_L_max, coup_R_L_mean        
        
        # decouple M from Left Lead
        #print("     - Decouple the Molecule and the Lead", flush=True)
        #H_em[nsao_el:-1*nsao_el,:nsao_el] = 0
        #H_em[:nsao_el,nsao_el:-1*nsao_el] = 0
        #S_em[nsao_el:-1*nsao_el,:nsao_el] = 0
        #S_em[:nsao_el,nsao_el:-1*nsao_el] = 0
        ## decouple M from Right Lead
        #H_em[-1*nsao_el:,nsao_el:-1*nsao_el] = 0
        #H_em[nsao_el:-1*nsao_el,-1*nsao_el:] = 0
        #S_em[-1*nsao_el:,nsao_el:-1*nsao_el] = 0
        #S_em[nsao_el:-1*nsao_el,-1*nsao_el:] = 0
        
        #nsao_em_red = nsao_em - nsao_m
        #H_em_red = np.zeros((nsao_em_red,nsao_em_red))
        #H_em_red[:nsao_el,:nsao_el]       = H_em[:nsao_el,:nsao_el]
        #H_em_red[-1*nsao_el:,-1*nsao_el:] = H_em[-1*nsao_el:,-1*nsao_el:]
        #H_em_red[:nsao_el,-1*nsao_el:]    = H_em[:nsao_el,-1*nsao_el:]
        #H_em_red[-1*nsao_el:,:nsao_el]    = H_em[-1*nsao_el:,:nsao_el]
        #
        #S_em_red = np.zeros((nsao_em_red,nsao_em_red))
        #S_em_red[:nsao_el,:nsao_el]       = S_em[:nsao_el,:nsao_el]
        #S_em_red[-1*nsao_el:,-1*nsao_el:] = S_em[-1*nsao_el:,-1*nsao_el:]
        #S_em_red[:nsao_el,-1*nsao_el:]    = S_em[:nsao_el,-1*nsao_el:]
        #S_em_red[-1*nsao_el:,:nsao_el]    = S_em[-1*nsao_el:,:nsao_el]
        
        
        #nsao_em_red = 2*nsao_pl
        #H_em_red = np.zeros((2*nsao_pl,2*nsao_pl))
        #H_em_red[:2*nsao_pl,:2*nsao_pl]       = H_em[:2*nsao_pl,:2*nsao_pl]
        #H_em_red[-1*2*nsao_pl:,-1*2*nsao_pl:] = H_em[-1*2*nsao_pl:,-1*2*nsao_pl:]
        #H_em_red[:2*nsao_pl,-1*2*nsao_pl:]    = H_em[:2*nsao_pl,-1*2*nsao_pl:]
        #H_em_red[-1*2*nsao_pl:,:2*nsao_pl]    = H_em[-1*2*nsao_pl:,:2*nsao_pl]
        #
        #S_em_red = np.zeros((2*nsao_pl,2*nsao_pl))
        #S_em_red[:2*nsao_pl,:2*nsao_pl]       = S_em[:2*nsao_pl,:2*nsao_pl]
        #S_em_red[-1*2*nsao_pl:,-1*2*nsao_pl:] = S_em[-1*2*nsao_pl:,-1*2*nsao_pl:]
        #S_em_red[:2*nsao_pl,-1*2*nsao_pl:]    = S_em[:2*nsao_pl,-1*2*nsao_pl:]
        #S_em_red[-1*2*nsao_pl:,:2*nsao_pl]    = S_em[-1*2*nsao_pl:,:2*nsao_pl]
        #
        #nsao_em = nsao_em_red
        #H_em = H_em_red
        #S_em = S_em_red
        
        #print("     - Zero out the Molecule part", flush=True)
        #H_em[np.ix_(EM.iaos_a,EM.iaos_a)] = 0    

    #--------------------------------------------------------------------------  
    # Check the embededd Hamiltonian, Overlap and resulted states
    # check Hermicity
    if not is_hermitian(H_em):
        print("ERROR: The H_EM is not hermitian!", flush=True)
        return 
    if not is_hermitian(S_em):
        print("ERROR: The S_EM is not hermitian!", flush=True)
        return   
    # write out the eigenvaluse of the extended molecule Hamiltonian
    if debug: print(f'     t = {timedelta(seconds=timer()-t0)}', flush=True)
    
    #if debug:
    #    print('     - Check the embededd Hamiltonian: ')
    #    t0 = timer()
    #    emb_eigs, emb_mos = linalg.eigh(H_em, S_em)
    #    emb_mos = emb_mos.T
    #    
    #    #emb_eigs = np.sort(np.real(emb_eigs))
    #    zero = 0
    #    if args["truncated"] != [None,None] and not args["dyson_orbitals"]:   
    #        fn = f'eig_mo_emb_{M_nocc:03d}-{M_nvir:03d}.dat'
    #    else:
    #        fn = "eig_emb.dat"
    #    f = open(fn,'w')
    #    for i, e in enumerate(emb_eigs):
    #        f.write(f'{ e*eh2ev:.14e}\n')
    #        if abs(e) < 1e-10:
    #            zero += 1
    #            #if abs(emb_mos[i][nsao_el+M_EAs.cont_sao]) > 0.1:
    #            #    print(i, zero, emb_mos[i][nsao_el+M_EAs.cont_sao])
    #    f.close()
    #    
    #    print(f'       Number of states with zero eig: { zero: 4d}')
    #    if args["dyson_orbitals"]: 
    #        print(f'                 dyson orbitals:       { nip+nea: 4d}')
    #    print(f'                 LMO in M part:        { nlmo_m: 4d}')
    #    print(f'                 SAO in M part:        { nsao_m: 4d}')
    #
    #    f = open("emb_pop.dat","w")
    #    for i in range(nsao_em):
    #        norm = np.sum(emb_mos[i]**2)
    #        l_pop = np.sum(emb_mos[i,:nsao_el]**2)/norm
    #        m_pop = np.sum(emb_mos[i,nsao_el:nsao_el+nsao_m]**2)/norm
    #        r_pop = np.sum(emb_mos[i,-1*nsao_el:]**2)/norm
    #        
    #        f.write(f'{i:} {l_pop: 7.5f} {m_pop: 7.5f} {r_pop: 7.5f}\n')
    #    f.close()    
    #    print(f'     t(debug) = {timedelta(seconds=timer()-t0)}', flush=True)

    #--------------------------------------------------------------------------
    # Coupling between EM and Lead(s) 
    # -- approximated with coupling between PLs in the Lead
    H_L_em = np.zeros((nsao_em,nsao_pl))
    S_L_em = np.zeros((nsao_em,nsao_pl))
    H_L_em[:nsao_pl,:] = H_pl_ji
    S_L_em[:nsao_pl,:] = S_pl_ji
    H_R_em = np.zeros((nsao_em,nsao_pl))
    S_R_em = np.zeros((nsao_em,nsao_pl))
    H_R_em[-1*nsao_pl:,:] = H_pl_ij
    S_R_em[-1*nsao_pl:,:] = S_pl_ij
    # -- check the validity of this approximation
    coup_max = np.max(abs(H_em[:nsao_pl,nsao_pl:2*nsao_pl] - H_pl_ij))
    coup_mean = np.mean(abs(H_em[:nsao_pl,nsao_pl:2*nsao_pl] - H_pl_ij))
    #if coup > 1e-8:
    print(f'     - Difference between V_(PL,PL) vs. V_(PL,EM): {coup_max: .2e}  {coup_mean: .2e}', flush=True)
    del coup_max, coup_mean
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # set up data for transmission loop
    print("",flush=True)
    print("  3. Calculate Transmission", flush=True)
    if args["molecular_orbitals"]:
        t_fn = 't_emb_mo'      
        if args["truncated"] != [None,None]:   
            t_fn += f'_{M_nocc:03d}-{M_nvir:03d}'
    
    elif args["dyson_orbitals"]:
        t_fn = 't_emb_do'
        #print("  3. Calculate Transmission using DOs of M", flush=True)
        if args["truncated"] != [None,None] or (nips+neas) != nsao_m:
            print(f'     - Use truncated space with {nips:3d} IP and {neas:3d} EA states. ', flush=True)
            print(f'     - Energy of last IP: {DO_eigs[0]*eh2ev: 7.5f} eV', flush=True)
            print(f'     - Energy of last EA: {DO_eigs[-1]*eh2ev: 7.5f} eV', flush=True)
            #t_fn = create_fn(f't_do_{nip:03d}-{nea:03d}.dat')
            t_fn += f'_{nips:03d}-{neas:03d}'
            if  os.path.isfile(t_fn + ".dat"):
                print(f'{t_fn} is already calculated!')
                return
        
    else:
        if "riper" in EM.em1.methods:
            t_fn = 't_emb_riper'
        else:
            t_fn = f't_emb_ao_{EM.em1.methods[0]}'
    
    if args["task_id"] is not None:
        t_fn += f'_id{task_id:03d}.dat'
    else:
        t_fn += '.dat'
     
    t_fn = create_fn(t_fn)

    # Read SFGFs from file if avaible (if not save it)
    if ( os.path.isfile(sfgf_fn_L) and os.path.isfile(sfgf_fn_R) ):
        # check egrid
        if not os.path.isfile(fn_egrid):
            print("WARNING: egrid.npy file is missing!", flush=True)
        else:
            with open(fn_egrid,"rb") as f_egrid:
                sv_egrid = np.load(f_egrid)
                
            if len(egrid) == len(sv_egrid):
                if (egrid == sv_egrid).all():
                    print("     - Load Lead Surface Green's functions from file", flush=True)
                    read_sfgf = True
                    
            elif len(np.intersect1d(egrid,sv_egrid)) > 0:
                print("     - Load part of Lead Surface Green's functions from file", flush=True)
                read_sfgf = True
                part_sfgf = True
                    
            if not read_sfgf:
                print("     - Energy range mistmatch with the founded Lead Surface Green's function files", flush=True)
                os.rename(fn_egrid,  fn_egrid.split(".")[0] + "_old.npy")
                os.rename(sfgf_fn_L,sfgf_fn_L.split(".")[0] + "_old.bin")
                os.rename(sfgf_fn_R,sfgf_fn_R.split(".")[0] + "_old.bin")
    
    if not read_sfgf:
        print("     - Prepare Lead Surface Green's functions", flush=True)
        with open(fn_egrid,"wb") as f_egrid:
                np.save(f_egrid,egrid) 

    ###########################################################################
    # Transmission LOOP
    ###########################################################################
    debug = False
    t00 = timer()
    for i_en, en in enumerate(egrid): # loop for egrid
        #--------------------------------------------------------------------------
        # Lead Self-Energies and Coupling matrices
        if read_sfgf and en in sv_egrid:
            if part_sfgf:
                i_en = np.where(sv_egrid == en)[0][0]
            dim = nsao_pl * nsao_pl
            count = 2 * dim
            offset = i_en * 2 * (dim + 7 * dim)  # Experimental factor 7
            with open(sfgf_fn_L, "rb") as f_L, open(sfgf_fn_R, "rb") as f_R:
                f_L.seek(offset)
                f_R.seek(offset)
                sfgf_inv_L = np.frombuffer(f_L.read(count * 8), dtype=complex).reshape((nsao_pl, nsao_pl))
                sfgf_inv_R = np.frombuffer(f_R.read(count * 8), dtype=complex).reshape((nsao_pl, nsao_pl))
        else:
            if debug: t0 = timer()
            sfgf_inv_L = get_LeadSurfaceGFinv(H_ii=H_pl, H_ij=H_pl_ij, S_ii=S_pl, S_ij=S_pl_ij,
                                              bias=bias, eta=eta, energy=en, thrd=1e-8)
            sfgf_inv_R = get_LeadSurfaceGFinv(H_ii=H_pl, H_ij=H_pl_ji, S_ii=S_pl, S_ij=S_pl_ji,
                                              bias=bias, eta=eta, energy=en, thrd=1e-8)
            if debug: print(f'     - t(SFGF) = {timedelta(seconds=timer()-t0)}', flush=True)
            
            if write_sfgf:
                if debug: t0 = timer()
                # write SFGFs to files
                with open(sfgf_fn_L,"ab") as Sigma_L_fn_open:
                    sfgf_inv_L.tofile(Sigma_L_fn_open)
                with open(sfgf_fn_R,"ab") as Sigma_R_fn_open:
                    sfgf_inv_R.tofile(Sigma_R_fn_open)    
                if debug: print(f'     - t(SFGF) = {timedelta(seconds=timer()-t0)}', flush=True)
        
        if debug: t0 = timer()
        Sigma_L = get_LeadSelfEnergy(sfgf_inv=sfgf_inv_L,H_im=H_L_em, S_im=S_L_em, 
                                     bias=bias, eta=eta, energy=en)
        Sigma_R = get_LeadSelfEnergy(sfgf_inv=sfgf_inv_R,H_im=H_R_em, S_im=S_R_em, 
                                     bias=bias, eta=eta, energy=en)
        if debug: print(f'     - t(Self) = {timedelta(seconds=timer()-t0)}', flush=True)
            
        Gamma_L =  1j*(Sigma_L - Sigma_L.conj().T) # == -2*np.imag(Sigma_L)
        Gamma_R =  1j*(Sigma_R - Sigma_R.conj().T) # == -2*np.imag(Sigma_R) 
        
        #if debug: t0 = timer()
        # this check is realy slow !!!! (more than 60% of the runtime) !!!!
        #if not is_pos_def(Gamma_L):
        #    min = np.min(linalg.eigvals(Gamma_L))
        #    print(f'Gamma_L is not positive definite at E={en: 2.5f} (min: {min: .5e})', flush=True)
        #if not is_pos_def(Gamma_R):
        #    min = np.min(linalg.eigvals(Gamma_R))
        #    print(f'Gamma_R is not positive definite at E={en: 2.5f} (min: {min: .5e})', flush=True)
        #if debug: print(f'     - t(Gamm) = {timedelta(seconds=timer()-t0)}', flush=True)
        
        #--------------------------------------------------------------------------
        # calculate transmission in AO basis using H and S
        if args["truncated"] == [None,None] or args["dyson_orbitals"]:
            # make the dressed GF at the specified energy point
            if debug: t0 = timer()
            GF = get_gf(en,H_em,S_em,Sigmas=[Sigma_L,Sigma_R],eta=eta)
            if debug: print(f'     - t( GF ) = {timedelta(seconds=timer()-t0)}', flush=True)
            # calculate the transmission
            if debug: t0 = timer()
            T = np.real(np.trace(Gamma_R @ GF @ Gamma_L @ GF.T.conj()))
            if debug: print(f'     - t(Tran) = {timedelta(seconds=timer()-t0)}', flush=True)
        
        #elif args["dyson_orbitals"]:
        #    GF_m_do = get_gf_do(M_EAs.dos,M_EAs.do_eigs,M_IPs.dos,M_IPs.do_eigs,en) 
        #           
        #    GF_m_huz = get_gf(en,H_m_huz,EM.S,eta=eta)
        #         
        #    Sigma_M = linalg.inv(GF_m_huz) - linalg.inv(GF_m_do)
        #                
        #    GF_inv = get_gf(en,H_em,S_em,Sigmas=[Sigma_M,Sigma_L,Sigma_R],eta=eta,inv=False)
        #    GF = np.linalg.pinv(GF_inv)
        #
        #    T = np.real(np.trace(Gamma_R @ GF @ Gamma_L @ GF.T.conj()))
        #--------------------------------------------------------------------------
        # calculate transmission in truncated MO/DO space
        else:
            if debug: t0 = timer()
            GF_inv = get_gf(en,H_em,S_em,Sigmas=[Sigma_L,Sigma_R],eta=eta,inv=False)
            GF = np.linalg.pinv(GF_inv)
            if debug: print(f'     - t( GF ) = {timedelta(seconds=timer()-t0)}', flush=True)
            if debug: t0 = timer()
            T = np.real(np.trace(Gamma_R @ GF @ Gamma_L @ GF.T.conj()))
            if debug: print(f'     - t(Tran) = {timedelta(seconds=timer()-t0)}', flush=True)

        #--------------------------------------------------------------------------
        # calc. DOS       
        if args["density_of_states"]:
            DOS = (-1/np.pi) * np.trace(np.matmul(np.imag(GF),S_em))
            with open("dos.dat","a") as fn:
                fn.write(f' {en: .14e} {DOS: .14e}\n')
            
            gf_em = get_gf(en,H_em,S_em,eta=eta)
            #gf_em_mo = EM.mos @ S_em @ gf_em @ S_em @ EM.mos.T
            DOS_em = (-1/np.pi) * np.trace(np.matmul(np.imag(gf_em),S_em))
            #DOS_em = (-1/np.pi) * np.trace(np.imag(gf_em_mo))
            with open("dos_em.dat","a") as fn:
                fn.write(f' {en: .14e} {DOS_em: .14e}\n')
                
            #gf_pl = get_gf(H_pl,S_pl,en,eta=eta)[0]
            #DOS_pl = (-1/np.pi) * np.trace(np.matmul(np.imag(gf_pl),S_pl))
            #with open("dos_pl.dat","a") as fn:
            #    fn.write(f' {en: .14e} {DOS_pl: .14e}\n')
            #DOS_L = (-1/np.pi) * np.matmul(np.linalg.inv(sfgf_inv_L),S_pl).imag.trace()
            #with open("dos_L.dat","a") as fn:
            #    fn.write(f' {en: .14e} {DOS_L: .14e}\n')
            #
            #DOS_R = (-1/np.pi) * np.matmul(np.linalg.inv(sfgf_inv_R),S_pl).imag.trace()
            #with open("dos_R.dat","a") as fn:
            #    fn.write(f' {en: .14e} {DOS_R: .14e}\n')
        
        #--------------------------------------------------------------------------
        # check Transmission and write out
        if T < 0.0 and abs(T) > 1e-10:
            print(f'WARNING: Negative Transmission ({T: .3e}) at {en*eh2ev: 3.5f} eV!', flush=True)
                
        with open(t_fn,"a") as fn:
            fn.write(f' {en*eh2ev: .14e} {T: .14e}\n')  

        #--------------------------------------------------------------------------
        # time
        if e_num < 10:
            t1 = timer()
            print(f'         {i_en+1: 6d}, e = {en*eh2ev: 6.2f}, t = ',timedelta(seconds=t1-t00), flush=True)
        elif i_en % (e_num // 10) == 0 or i_en == e_num-1:
            t1 = timer()
            if i_en == 0:
                print(f'         {i_en+1: 6d}, e = {en*eh2ev: 6.2f}, t = ',timedelta(seconds=t1-t00),
                      " estimated time: ",timedelta(seconds=(t1-t00)*e_num) ,flush=True)
            else:
                print(f'         {i_en+1: 6d}, e = {en*eh2ev: 6.2f}, t = ',timedelta(seconds=t1-t00), flush=True)
    ###########################################################################
    # END of Transmission LOOP
    ###########################################################################
    
#----END
def end():
    print("THE END", flush=True)
    
if __name__ == "__main__":
    main()
    end()


# %%
