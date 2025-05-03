#!/usr/bin/python3
# coding: utf-8
"""
Calculate the trassmission using NEGF-DFT formalism
from TURBOMOLE DSCF or RIPER calculations

Created on 10.11.2023
@author: Dávid P. Jelenfi
"""

###############################################################################
# %% Imports 
###############################################################################
# System ----------------------------------------------------------------------
import os
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
from band import BAND
from mrcc import MRCC
from GreenFunction import get_gf, get_LeadSurfaceGFinv, get_LeadSelfEnergy, get_gf_lcao, get_dos
from transmission import calc_t_coeff
###############################################################################
from general import is_pos_def, is_hermitian, create_fn, print_M, read_M
from wbl_gf import get_gf_hf_eig

###############################################################################
# %% MAIN
###############################################################################
def main():     
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

    # GF parameters --------------------------------------------------------------------------
    ap.add_argument("-eta", "--eta", type=float, required=False, default=1e-4,
                    help="The eta parameter of the GF stuff in eV.")
    
    # path -----------------------------------------------------------------------------------
    ap.add_argument("-p_lead", "--path_of_lead", type=str, required=False, default="Lead", 
                    help="Path of the Lead calculation.")
    
    ap.add_argument("-p_em", "--path_of_em", type=str, required=False,default="EM",
                    help="Path of the EM calculation.")  
    
    ap.add_argument("-p_ip", "--path_of_ips", type=str, required=False, default="IPs", 
                    help="The path of the IP calculation.")
    
    ap.add_argument("-p_ea", "--path_of_eas", type=str, required=False,default="EAs",
                    help="The path of the EA calculation.")
    
    # lead options -----------------------------------------------------------------------------  
    ap.add_argument("-wbl", "--wide_band_limit", action="store_true",
                    help="Use wide band limit approximation.")
    
    ap.add_argument("-pl", "--tm_lead_pl", type=int, required=False, default=2,
                    help="Numnber of PLs in the Lead TM calculation.")
    
    # extended molecule options ---------------------------------------------------------------
    ap.add_argument("-c_em", "--code_em", type=str, required=False,default="MRCC",
                    help="Code (MRCC/TURBOMOLE/BAND) for the EM calculation.")     
    
    ap.add_argument("-sub", "--substitute", action="store_true",
                    help="Substitute the PL part of EM region with PL part from Lead.")

    ap.add_argument("-ext", "--extend", action="store_true",
                    help="Extend the EM with the PL part of the Lead.")
    
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
    print("*  Last modification: 2024.08.27.                 *", flush=True)
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
    Lead_path=str(args["path_of_lead"])
    EM_path=str(args["path_of_em"])
    IPs_path=str(args["path_of_ips"])
    EAs_path=str(args["path_of_eas"])
    # use full path
    cwd = os.getcwd()
    if not cwd in Lead_path:
        Lead_path = cwd + "/" + Lead_path
    if not cwd in EM_path:
        EM_path = cwd + "/" + EM_path
    if not cwd in IPs_path:
        IPs_path = cwd + "/" + IPs_path
    if not cwd in EAs_path:
        EAs_path = cwd + "/" + EAs_path
           
    ###########################################################################
    # Local variables
    ###########################################################################
    
    # for EM region
    EM_PBC = False
    # for Leads
    TM_PL = int(args["tm_lead_pl"]) # Number of PLs in TURBOMOLE Lead calculation
    sfgf_fn_L = "lead_sfgf_L.bin"
    sfgf_fn_R = "lead_sfgf_R.bin"
    fn_egrid = "egrid.npy"
    # for GF
    eta=float(args["eta"]) #1e-5/eh2ev
    bias=.0
    # for WBL
    gamma = 4/eh2ev # 4eV in Jos-Verzijl paper

    #which electron structure calc. code
    #elec_code = "BAND"
    #elec_code = "TM"
    elec_code_lead = "TM"
    tmp = args["code_em"].upper()
    if  tmp == "TURBOMOLE" or tmp == "TM": 
        elec_code_em = "TM"
    elif tmp == "MRCC":
        elec_code_em = "MRCC"
    elif tmp == "BAND":
        elec_code_em = "BAND"
        elec_code_lead = "BAND"
    

    # for MO aprx.
    #nocc = int(args["truncated"][0]) # number of occ. MOs
    #nvir = int(args["truncated"][1]) # number of vir. MOs
  
    # debug
    debug = False
    
    # Dyson orbital
    cut_cont = False # cut out continuum orbital (True) or not (False)
    do_norm = False # normalize them (True) or not (False)

    ###########################################################################
    # Start code
    ###########################################################################
    t00 = timer()
    
    #--------------------------------------------------------------------------
    print("  1. Prepare EM region", flush=True)
    # EM : 1PL --- Branch --- Mol --- Branch --- 1PL
    # use (non)-periodic calculation with TM riper for the EM region
    if elec_code_em == "TM":
        EM = TM(path=EM_path)
        EM.read_HS()
        if "riper" in EM.methods and not EM_PBC:
            print("     - RIPER calculation was found (periodic EM region).", flush=True)
            H_em = EM.H
            S_em = EM.S
        elif "riper" in EM.methods and EM_PBC: # If you need them between unit cells
            print("ERROR: EM_PBC = True is not implemented!", flush=True)
        else:
            print("     - DSCF calculation was found", flush=True)
            H_em = EM.H
            S_em = EM.S
            #EM.mo_eigs = EM.mo_eigs*eh2ev
    #        print("     - Use embedded Fock matrix", flush=True)
    #        H_em = read_M(EM_path + "/sfmat1.sao")
    #        
    #    emb_eigs, emb_mos = linalg.eigh(H_em,S_em)
    #    emb_mos = emb_mos.T
    #    
    #    with open("emb_eig.dat","w") as f:
    #        for e in emb_eigs:
    #            f.write(f'{ e*eh2ev:.14e}\n')
    #            
    #    EM.write_molden("emb_mos.molden",emb_eigs*eh2ev,emb_mos)
    #    #        
    #    #with open("supsys_eig.dat","w") as f:
    #    #    for e in EM.mo_eigs:
    #    #        f.write(f'{ e*eh2ev:.14e}\n')
            
    elif elec_code_em == "BAND":
        print("     - BAND calculation was found", flush=True)
        EM = BAND(path=EM_path)
        EM.read_HS()
        H_em = EM.H
        S_em = EM.S
    
    elif elec_code_em == "MRCC":
        print("     - MRCC calculation was found", flush=True)
        EM = MRCC(path=EM_path)
        EM.read_HS()
        H_em = EM.H
        S_em = EM.S
        
        #print(EM.mo_eigs)
        #print("f_mo:")
        #print(np.diag(EM.mos @ H_em @ EM.mos.T))
        #print(np.max(np.abs(EM.mo_eigs-np.diag(EM.mos @ H_em @ EM.mos.T))))
        #print("S:")
        #print(np.diag(EM.mos @ S_em @ EM.mos.T))
        
    nsao_em = EM.nsao
    
    f = open("eig.dat","w")
    for i, e in enumerate(EM.mo_eigs):
        f.write(f'{ e*eh2ev:.14e}\n')
    f.close()
    
    #egrid = list(egrid)
    #for ei in EM.mo_eigs:
    #    if ei > e_min and ei < e_max:
    #        egrid.append(ei)
    #egrid.sort()
    #egrid = np.array(egrid)
    
    #--------------------------------------------------------------------------
    if True: #not args["wide_band_limit"]:
        print("  2. Prepare Lead(s)", flush=True) 
        # read the output of the (1D) periodic calculation made with TM riper for the Lead(s)
        if elec_code_lead == "TM":
            Lead = TM(path=Lead_path)
            Lead.read_HS()
            if args["Fermi_level"]:
                Lead.read_fermi()
                Lead.Fermi = Lead.Fermi
                print(f'     - Transmission will be calculated only at Lead Fermi level ({Lead.Fermi*eh2ev: 7.5f} eV).', flush=True)
                egrid = np.array([Lead.Fermi])
            if TM_PL == 1:
                print("Use at least 2 PL for Lead calculation!")
                exit()
                #H_pl, H_c, S_pl, S_c = Lead.H[0]*eh2ev, Lead.H[1]*eh2ev, Lead.S[0], Lead.S[1]
                #H_pl_ij, H_pl_ji = np.tril(H_c,k=1), np.triu(H_c,k=-1)
                #S_pl_ij, S_pl_ji = np.tril(S_c,k=1), np.triu(S_c,k=-1)
                #nsao_pl = Lead.nsao
            elif TM_PL > 1:
                # struct
                #   | H_pl    | H_pl_ij |
                #   | H_pl_ji | H_pl    |           
                H_unit, S_unit = Lead.H, Lead.S
                nsao_pl = Lead.nsao // TM_PL
                H_pl    = H_unit[:nsao_pl,:nsao_pl]
                H_pl_ij = H_unit[:nsao_pl,nsao_pl:2*nsao_pl]
                H_pl_ji = H_unit[nsao_pl:2*nsao_pl,:nsao_pl]
                S_pl    = S_unit[:nsao_pl,:nsao_pl]
                S_pl_ij = S_unit[:nsao_pl,nsao_pl:2*nsao_pl]
                S_pl_ji = S_unit[nsao_pl:2*nsao_pl,:nsao_pl]
                
                # PL "on-site"
                #H_pl_L    = H_unit[:nsao_pl,:nsao_pl]
                #S_pl_L    = S_unit[:nsao_pl,:nsao_pl]
                #H_pl_R    = H_unit[-1*nsao_pl:,-1*nsao_pl:]
                #S_pl_R    = S_unit[-1*nsao_pl:,-1*nsao_pl:]
                ##PL "hopping"
                #H_pl_L_ij = H_unit[:nsao_pl,nsao_pl:2*nsao_pl]
                #S_pl_L_ij = S_unit[:nsao_pl,nsao_pl:2*nsao_pl]
                #H_pl_R_ij = H_unit[-2*nsao_pl:-1*nsao_pl,-1*nsao_pl:]                
                #S_pl_R_ij = S_unit[-2*nsao_pl:-1*nsao_pl,-1*nsao_pl:]      

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
               
            else:
                print("Error: invalide number of PL", flush=True)
                return
        elif elec_code_lead == "BAND":
            print("     - BAND calculation was found", flush=True)
            Lead = BAND(path=Lead_path,PBC=True)
            Lead.read_HS()
            H_pl    = Lead.H[0]
            H_pl_ij = Lead.H[1].T
            H_pl_ji = Lead.H[1]
            S_pl    = Lead.S[0]
            S_pl_ij = Lead.S[1].T
            S_pl_ji = Lead.S[1]

            nsao_pl = Lead.nsao

        #--------------------------------------------------------------------------
        # check Lead (electrode) part of the EM 
        diff_L   = np.max(abs(S_em[:nsao_pl,:nsao_pl] - S_pl))
        diff_R   = np.max(abs(S_em[-1*nsao_pl:,-1*nsao_pl:] - S_pl))
        if diff_L > 1e-8 or diff_R > 1e-8:
            print(f'     - Structure of Lead part of EM is differ from sturcture of Lead :  L={diff_L: .2e}, R={diff_R: .2e} ', flush=True)
        del diff_L, diff_R    
        
        #--------------------------------------------------------------------------
        # Modifications of the Hamiltonian and Overlap
        # substitute the first PL region in the EM with the PL from the Lead calculation
        # and turn-off the coupling between the Leads
        if args["substitute"]: 
            print("     - Substitute terminal Lead PL part of EM region with PL from Lead", flush=True)
            H_em[:nsao_pl,:nsao_pl]                  = H_pl
            H_em[-1*nsao_pl:,-1*nsao_pl:]            = H_pl
            H_em[:nsao_pl,-1*nsao_pl:]               = 0  
            H_em[-1*nsao_pl:,:nsao_pl]               = 0
            # the below modification makes S no longer positive definite!!
            S_em[:nsao_pl,-1*nsao_pl:]               = 0  
            S_em[-1*nsao_pl:,:nsao_pl]               = 0

            # BAND doesn't do this, due to S^Lead_pl == S^EM_pl (from periodic calc. only unit cell matrices should be used)
            #S_em[:nsao_pl,:nsao_pl]                  = S_pl 
            #S_em[-1*nsao_pl:,-1*nsao_pl:]            = S_pl

        # extend the EM Hamiltonian with the PL from the Lead calculation
        if args["extend"]:
            print("     - Extend the EM region with PL part from Lead", flush=True)
            nsao_em_ext = nsao_em + 2*nsao_pl
            H_em_ext = np.zeros((nsao_em_ext,nsao_em_ext))
            S_em_ext = np.zeros((nsao_em_ext,nsao_em_ext))

            H_em_ext[nsao_pl:-1*nsao_pl,nsao_pl:-1*nsao_pl] = H_em
            H_em_ext[:nsao_pl,:nsao_pl]                 = H_pl
            H_em_ext[-1*nsao_pl:,-1*nsao_pl:]           = H_pl
            H_em_ext[:nsao_pl,nsao_pl:2*nsao_pl]        = H_pl_ij
            H_em_ext[nsao_pl:2*nsao_pl,:nsao_pl]        = H_pl_ji
            H_em_ext[-2*nsao_pl:-1*nsao_pl,-1*nsao_pl:] = H_pl_ij
            H_em_ext[-1*nsao_pl:,-2*nsao_pl:-1*nsao_pl] = H_pl_ji

            S_em_ext[nsao_pl:-1*nsao_pl,nsao_pl:-1*nsao_pl] = S_em
            S_em_ext[:nsao_pl,:nsao_pl]                 = S_pl
            S_em_ext[-1*nsao_pl:,-1*nsao_pl:]           = S_pl
            S_em_ext[:nsao_pl,nsao_pl:2*nsao_pl]        = S_pl_ij
            S_em_ext[nsao_pl:2*nsao_pl,:nsao_pl]        = S_pl_ji
            S_em_ext[-2*nsao_pl:-1*nsao_pl,-1*nsao_pl:] = S_pl_ij
            S_em_ext[-1*nsao_pl:,-2*nsao_pl:-1*nsao_pl] = S_pl_ji

            H_em = H_em_ext
            S_em = S_em_ext
            nsao_em = nsao_em_ext

        # check Hermicity
        if not is_hermitian(H_em):
            print("ERROR: The H_EM is not hermitian!", flush=True)
            return 
        if not is_hermitian(S_em):
            print("ERROR: The S_EM is not hermitian!", flush=True)
            return  
 
        # calculate the difference between PL in EM and PL in Lead 
        if not args["substitute"] or not args["extend"]:
            diff_L   = np.max(abs(H_em[:nsao_pl,:nsao_pl] - H_pl))
            diff_R   = np.max(abs(H_em[-1*nsao_pl:,-1*nsao_pl:] - H_pl))
            coup_R_L = np.max(abs(H_em[:nsao_pl,-1*nsao_pl:]))
            if diff_L > 1e-8 or diff_R > 1e-8:
                print(f'     - Difference between H_PL part in EM and in Lead:  L={diff_L: .2e}, R={diff_R: .2e} ', flush=True)
            if coup_R_L > 1e-8:
                print(f'     - Coupling between Left and Right PL in EM: {coup_R_L: .2e}', flush=True)
            del diff_L, diff_R, coup_R_L

        # -- check the validation of this approximation
        coup = np.max(abs(H_em[:nsao_pl,nsao_pl:2*nsao_pl] - H_pl_ij))
        if coup > 1e-8:
            print(f'     - Difference between V_(PL,PL) vs. V_(PL,EM): {coup: .2e}', flush=True)
        del coup
    
    #--------------------------------------------------------------------------
    # MO LCAO coeffs and energies correspond to the unit cell in case of RIPER
    # and/or to the modified Hamiltonian in case of substitution
    if args["molecular_orbitals"] and (args["substitute"] or "riper" in EM.methods):
        print("     - Recalculate MO energies and coefficients of EM", flush=True)
        if not is_pos_def(S_em):
            print("WARNING: S_em is not positive definite!", flush=True)
            EM.mo_eigs, EM.mos = linalg.eig(H_em, S_em)
        else:
            EM.mo_eigs, EM.mos = linalg.eigh(H_em, S_em, eigvals_only=False)
        EM.mos = EM.mos.T

    #--------------------------------------------------------------------------
    # make the truncated MO space
    if args["molecular_orbitals"] and (args["truncated"][0] != None or args["truncated"][1] != None):
        if args["truncated"][0] != None:
            eocc = float(args["truncated"][0]) # maximum energy of occ. MOs
        else:
            eocc = np.min(EM.mo_eigs)
        if args["truncated"][1] != None:
            evir = float(args["truncated"][1]) # minimum energy of vir. MOs
        else:
            evir = np.max(EM.mo_eigs)
        gap_mid = (EM.mo_eigs[EM.HOMO] + EM.mo_eigs[EM.HOMO+1]) / 2
        nocc = ( EM.mo_eigs[:EM.HOMO+1] > (eocc+gap_mid) ).sum()
        nvir = ( EM.mo_eigs[EM.HOMO+1:] < (evir+gap_mid) ).sum()
        minocc = EM.HOMO+1 - nocc
        maxvir = EM.HOMO+1 + nvir
        mos_em_trc = []
        mo_eigs_em_trc = []
        for i in range(minocc,maxvir):
            mos_em_trc.append(EM.mos[i])
            mo_eigs_em_trc.append(EM.mo_eigs[i])
        mos_em_trc = np.array(mos_em_trc)
        mo_eigs_em_trc = np.array(mo_eigs_em_trc)

    #--------------------------------------------------------------------------
    # load Dyson orbitals
    if args["dyson_orbitals"]:
        print("     - Prepare Dyson orbitals", flush=True)
        EM_IPs = TM(path=IPs_path)
        EM_IPs.read_exc_ens()
        EM_IPs.create_dos(norm=do_norm,cut=cut_cont)
        EM_EAs = TM(path=EAs_path)
        EM_EAs.read_exc_ens()
        EM_EAs.create_dos(norm=do_norm,cut=cut_cont)    
        if EM_IPs.nsao != EM_EAs.nsao:
            print("ERROR: IP calculation has different number of SAO then EA!", flush=True)

        EM_IPs.exc_ens = -1*EM_IPs.exc_ens # flip the sign of IP states
        idx = np.argsort(EM_IPs.exc_ens)
        EM_IPs.exc_ens  = EM_IPs.exc_ens[idx]
        EM_IPs.dos = EM_IPs.dos[idx]
        
        idx = np.argsort(EM_EAs.exc_ens)
        EM_EAs.exc_ens  = EM_EAs.exc_ens[idx]
        EM_EAs.dos = EM_EAs.dos[idx]
        
        nip_max = EM_IPs.exc_sts['a']
        nea_max = EM_EAs.exc_sts['a']            
        
        if args["truncated"][0] != None:
            eip = float(args["truncated"][0]) # maximum energy of IPs
            nip = ( EM_IPs.exc_ens > eip ).sum()
        else:
            nip = nip_max

        if args["truncated"][1] != None:
            eea = float(args["truncated"][1]) # minimum energy of EAs
            nea = ( EM_EAs.exc_ens < eea ).sum()
        else:
            nea = nea_max
            
        ndos = nip + nea 
        if cut_cont:
            DOs = np.zeros((ndos,EM_EAs.nsao-1))
        else:
            DOs = np.zeros((ndos,EM_EAs.nsao))
        DO_eigs = np.zeros((ndos))
        
        # add IP-DOs (occ. part)
        for st in range(nip):
            DOs[st] = EM_IPs.dos[st]
            DO_eigs[st] = EM_IPs.exc_ens[st]
        # add EA-DOs (virt. part)
        for st in range(nea):
            DOs[nip+st] = EM_EAs.dos[st]
            DO_eigs[nip+st] = EM_EAs.exc_ens[st]
        
        print("DEBUG: DO energies: ", DO_eigs)
                
    #--------------------------------------------------------------------------
    # set up data for transmission loop 
    if args["molecular_orbitals"]:
        print("  3. Calculate Transmission using Molecular Orbitals of EM", flush=True)
        if args["truncated"] == [None,None]:
            t_fn = create_fn(f't_mo')
            XO = EM.mos
            XO_eig = EM.mo_eigs
        else:
            print(f'     - EM HOMO index:  {EM.HOMO+1} & energy: {EM.mo_eigs[EM.HOMO]*eh2ev: 7.5f} eV', flush=True)
            print(f'     - EM LUMO index:  {EM.HOMO+2} & energy: {EM.mo_eigs[EM.HOMO+1]*eh2ev: 7.5f} eV', flush=True)
            print(f'     - EM HOMO-LUMO gap {EM.mo_eigs[EM.HOMO+1]-EM.mo_eigs[EM.HOMO]: 7.5f} & middle: {gap_mid: 7.5f} eV', flush=True)
            print(f'     - Use truncated space with {nocc:3d} occupied and {nvir:3d} virtual MOs.', flush=True)
            print(f'     - Last Occ. MO index:  {minocc} & energy: {EM.mo_eigs[minocc]*eh2ev: 7.5f} eV', flush=True)
            print(f'     - Last Vir. MO index:  {maxvir} & energy: {EM.mo_eigs[maxvir]*eh2ev: 7.5f} eV', flush=True)

            #t_fn = create_fn(f't_mo_{nocc:03d}-{nvir:03d}.dat')
            t_fn = f't_mo_{nocc:03d}-{nvir:03d}'
            if  os.path.isfile(t_fn):
                print(f'{t_fn} is already calculated!')
                return
            XO = mos_em_trc
            XO_eig = mo_eigs_em_trc
    
    elif args["dyson_orbitals"]:
        print("  3. Calculate Transmission using Dyson Orbitals of EM", flush=True)
        if args["truncated"] == [None,None]:
            t_fn = create_fn("t_do")
        else:
            print(f'  - Use truncated space with {nip:3d} IP and {nea:3d} EA states. ', flush=True)
            print(f'  - Energy of last IP: {EM_IPs.exc_ens[nip-1]*eh2ev: 7.5f} eV', flush=True)
            print(f'  - Energy of last EA: {EM_EAs.exc_ens[nea-1]*eh2ev: 7.5f} eV', flush=True)
            #t_fn = create_fn(f't_do_{nip:03d}-{nea:03d}.dat')
            t_fn = f't_do_{nip:03d}-{nea:03d}'
            if  os.path.isfile(t_fn):
                print(f'{t_fn} is already calculated!')
                return
        XO = DOs
        XO_eig = DO_eigs
        
    else:
        print("  3. Calculate Transmission using Kohn-Sham Hamiltonina of EM", flush=True)
        if "riper" in EM.methods:
            t_fn = create_fn('t_riper')
        else:
            t_fn = create_fn(f't_{EM.methods[0]}')
    if args["task_id"] is not None:
        t_fn += f'{args["task_id"]}.dat'
    else:
        t_fn += '.dat'
    
    if args["wide_band_limit"]:
        print("     - Use WBL approximation", flush=True)
        Sigma_L = np.zeros((nsao_em,nsao_em),dtype=complex)
        Sigma_L[:nsao_pl,:nsao_pl] = -0.5j * gamma * S_em[:nsao_pl,:nsao_pl]
        Sigma_R = np.zeros((nsao_em,nsao_em),dtype=complex)
        Sigma_R[-1*nsao_pl:,-1*nsao_pl:] = -0.5j * gamma * S_em[-1*nsao_pl:,-1*nsao_pl:]
        
        L = linalg.fractional_matrix_power(S_em,-0.5) # Lowdin
        
        GF = H_em - Sigma_L - Sigma_R
        
        GF_orth = L @ GF @ L
        
        e_em, mos_eff = linalg.eig(GF_orth)
        
        mos_eff = mos_eff.T
        for i in range(len(e_em)):
            mos_eff[i] = mos_eff[i] / np.sqrt(np.sum(mos_eff[i]**2))
        
        mos_eff = L @ mos_eff
        
        np.max(abs(e_em - np.diag(mos_eff @ GF @ mos_eff.T)))
        
        Gamma_L = -2 * np.imag(Sigma_L)
        Gamma_R = -2 * np.imag(Sigma_R)

        A_L = mos_eff @ np.conj(Gamma_L @ mos_eff)
        A_R = mos_eff @ np.conj(Gamma_R @ mos_eff)
        Gamma_orth = np.conj(np.transpose(A_L)) @ A_R
        
    else:
        read_sfgf = False
        if ( os.path.isfile(sfgf_fn_L) and os.path.isfile(sfgf_fn_R) ) and (not debug):
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
                else:
                    print("     - Energy range mistmatch with the founded Lead Surface Green's function files", flush=True)
                    os.rename(fn_egrid,  fn_egrid.split(".")[0] + "_old.npy")
                    os.rename(sfgf_fn_L,sfgf_fn_L.split(".")[0] + "_old.bin")
                    os.rename(sfgf_fn_R,sfgf_fn_R.split(".")[0] + "_old.bin")
            
        if not read_sfgf:
            print("     - Prepare Lead Surface Green's function", flush=True)
            # //TODO: something is wrong with Lead SFGF writting
            if False:
                with open(fn_egrid,"wb") as f_egrid:
                    np.save(f_egrid,egrid) 
                    
    ###########################################################################
    # Transmission LOOP
    ###########################################################################
    for i_en, en in enumerate(egrid): # loop for egrid
        t0 = timer()
        #--------------------------------------------------------------------------
        # Self-Energies and Coupling matrices
        if not args["wide_band_limit"]:
            if read_sfgf:
                dim = nsao_pl * nsao_pl
                count  = 2 * dim # factor 2 due to it's a complex matrix
                offset = i_en * 2 * (dim + 7 * dim) # factor 7 is "experimental" !!!
                sfgf_inv_L  = np.fromfile(sfgf_fn_L,dtype=complex,count=count,offset=offset)
                sfgf_inv_L.resize((nsao_pl,nsao_pl), refcheck=False)
                sfgf_inv_R  = np.fromfile(sfgf_fn_R,dtype=complex,count=count,offset=offset)
                sfgf_inv_R.resize((nsao_pl,nsao_pl), refcheck=False)
            else:
                sfgf_inv_L = get_LeadSurfaceGFinv(H_ii=H_pl, H_ij=H_pl_ij, S_ii=S_pl, S_ij=S_pl_ij,
                                                  bias=bias, eta=eta, energy=en, thrd=1e-8)
                sfgf_inv_R = get_LeadSurfaceGFinv(H_ii=H_pl, H_ij=H_pl_ji, S_ii=S_pl, S_ij=S_pl_ji,
                                                  bias=bias, eta=eta, energy=en, thrd=1e-8)
                # //TODO: something is wrong with Lead SFGF writting
                if False:#not debug:
                    with open(sfgf_fn_L,"ab") as Sigma_L_fn_open:
                        sfgf_inv_L.tofile(Sigma_L_fn_open)
                    with open(sfgf_fn_R,"ab") as Sigma_R_fn_open:
                        sfgf_inv_R.tofile(Sigma_R_fn_open)    

            Sigma_L = get_LeadSelfEnergy(sfgf_inv=sfgf_inv_L,H_im=H_L_em, S_im=S_L_em, 
                                         bias=bias, eta=eta, energy=en)
            Sigma_R = get_LeadSelfEnergy(sfgf_inv=sfgf_inv_R,H_im=H_R_em, S_im=S_R_em, 
                                         bias=bias, eta=eta, energy=en)
            
        Gamma_L =  1j*(Sigma_L - Sigma_L.conj().T) # == -2*np.imag(Sigma_L)
        Gamma_R =  1j*(Sigma_R - Sigma_R.conj().T) # == -2*np.imag(Sigma_R) 
        
        if not is_pos_def(Gamma_L):
            min = np.min(linalg.eigvals(Gamma_L))
            print(f'Gamma_L is not positive definite at E={en*eh2ev: 2.5f} (min: {min: .5e})', flush=True)
        if not is_pos_def(Gamma_R):
            min = np.min(linalg.eigvals(Gamma_R))
            print(f'Gamma_R is not positive definite at E={en*eh2ev: 2.5f} (min: {min: .5e})', flush=True) 
        
        #--------------------------------------------------------------------------
        # calculate transmission in AO basis using H and S
        # make the dressed GF at the specified energy point
        if args["wide_band_limit"]:
            GF = get_gf(en,H_em,S_em,Sigmas=[Sigma_L,Sigma_R],eta=eta)    
            # calculate the transmission 
            T0 = np.real(np.trace(Gamma_R @ GF @ Gamma_L @ GF.T.conj()))
            
            e = en + 1j*eta
            gf_diag = e - e_em
            T = np.real(np.sum(Gamma_orth / (gf_diag[:, np.newaxis] * np.conj(gf_diag[np.newaxis, :]))))
            #T = np.real(np.trace(Gamma_R_orth @ GF @ Gamma_L_orth @ GF.T.conj()))
            
            print(T0,T)
            
        else:            
            if not args["molecular_orbitals"] and not args["dyson_orbitals"]:
                GF = get_gf(en,H_em,S_em,Sigmas=[Sigma_L,Sigma_R],eta=eta)    
                # calculate the transmission 
                T = np.real(np.trace(Gamma_R @ GF @ Gamma_L @ GF.T.conj()))

                # Transmission matrix
                #tm = linalg.fractional_matrix_power(Gamma_L,1/2) @ GF @ linalg.fractional_matrix_power(Gamma_R,1/2)
                ##tm_dag = linalg.fractional_matrix_power(Gamma_R,1/2) @ GF.T.conj() @ linalg.fractional_matrix_power(Gamma_L,1/2)
                ##print(np.max(np.abs(tm.T.conj() - tm_dag))) # TRUE
                ##print(is_hermitian(tm @ tm.T.conj())) # TRUE
                #t, tv = np.linalg.eigh(tm @ tm.T.conj())
                #print(f'T:     {T:.5e}' )
                #print(f't_sum: {np.sum(t):.5e}' )
                #for i,ti in enumerate(t):
                #    if abs(ti) > 1e-10:
                #        print(f't_{i}: {ti:.5e}')
                #        for j in range(nsao_em):
                #           ov = tv[:,i] @ S_em @ EM.mos[j]
                #           if abs(ov)**2 > 1e-7:
                #                print(f'{j} {abs(ov)**2:.5e}')
            #--------------------------------------------------------------------------
            # calculate transmission in AO basis using XO space
            if args["molecular_orbitals"] or args["dyson_orbitals"]:
                gf_em_ao = get_gf_lcao(en, XO.T, XO_eig, eta=eta)[0]
                H_em_ao_eff = np.linalg.inv(XO @ S_em @ gf_em_ao @ S_em @ XO.T)
                Sigma_L_xo = XO @ Sigma_L @ XO.T
                Sigma_R_xo = XO @ Sigma_R @ XO.T
                GF = np.linalg.inv(H_em_ao_eff - Sigma_L_xo - Sigma_R_xo) # create GF in XO basis
                #Gamma_L_xo = -2*np.imag(Sigma_L_xo)
                #Gamma_R_xo = -2*np.imag(Sigma_R_xo)
                #T = np.real(np.trace(Gamma_L_xo @ GF @ Gamma_R_xo @ GF.T.conj()))
                # back to AO basis
                GF_ao = XO.T @ GF @ XO
                T = np.real(np.trace(Gamma_R @ GF_ao @ Gamma_L @ GF_ao.T.conj()))

                # Transmission matrix
                #tm = linalg.fractional_matrix_power(Gamma_L,1/2) @ GF_ao @ linalg.fractional_matrix_power(Gamma_R,1/2)
                #tm_dag = linalg.fractional_matrix_power(Gamma_R,1/2) @ GF_ao.T.conj() @ linalg.fractional_matrix_power(Gamma_L,1/2)
                #print("Difference:", np.max(np.abs(tm.T.conj() - tm_dag))) # TRUE
                #print("Hermition: ",is_hermitian(tm @ tm.T.conj())) # TRUE
                #
                #t, tv = np.linalg.eigh(tm @ tm.T.conj())
                #print(f'T:     {T:.5e}' )
                #print(f't_sum: {np.sum(t):.5e}' )
                #for i,ti in enumerate(t):
                #    if abs(ti) > 1e-10:
                #        print(f't_{i}: {ti:.5e}')
                #        for j in range(nsao_em):
                #           ov = tv[:,i] @ S_em @ EM.mos[j]
                #           if abs(ov)**2 > 1e-7:
                #                print(f'{j} {abs(ov)**2:.5e}')

                # pinv version -> is not deal correctly with the coupling, due to it contains the coupling to all orbitals,
                # while the gf_em_ao has only those orbitals which is in the truncated space
                #GF = np.linalg.pinv(np.linalg.pinv(gf_em_ao) - Sigma_L - Sigma_R)
                #T = np.real(np.trace(Gamma_L @ GF @ Gamma_R @ GF.T.conj()))

            #--------------------------------------------------------------------------
            # calc. DOS       
            if args["density_of_states"]:
                DOS = (-1/np.pi) * np.trace(np.matmul(np.imag(GF),S_em))
                with open("dos.dat","a") as fn:
                    fn.write(f' {en: .14e} {DOS: .14e}\n')

                gf_em = get_gf(H_em,S_em,en,eta=eta)
                #gf_em_mo = EM.mos @ S_em @ gf_em @ S_em @ EM.mos.T
                DOS_em = (-1/np.pi) * np.trace(np.matmul(np.imag(gf_em),S_em))
                #DOS_em = (-1/np.pi) * np.trace(np.imag(gf_em_mo))
                with open("dos_em.dat","a") as fn:
                    fn.write(f' {en: .14e} {DOS_em: .14e}\n')

                #gf_pl = get_gf(H_pl,S_pl,en,eta=eta)
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
            print(f'         {i_en+1: 6d}, e = {en*eh2ev: 6.2f}, t = ',timedelta(seconds=t1-t0), flush=True)
        elif i_en % (e_num // 10) == 0 or i_en == e_num-1:
            t1 = timer()
            if i_en == 0:
                print(f'         {i_en+1: 6d}, e = {en*eh2ev: 6.2f}, t = ',timedelta(seconds=t1-t0)," estimated time: ",timedelta(seconds=(t1-t0)*e_num) ,flush=True)
            else:
                print(f'         {i_en+1: 6d}, e = {en*eh2ev: 6.2f}, t = ',timedelta(seconds=t1-t0), flush=True)
    ###########################################################################
    # END of Transmission LOOP
    ###########################################################################
    
#----END
def end():
    print("THE END", flush=True)
    
if __name__ == "__main__":
    
    #i = 6
    #j = 54
    #main(nocc=i,nvir=j)
    #for j in range(0,54):
    #    main(nocc=i,nvir=j)
    
    main()
    end()

# %%
