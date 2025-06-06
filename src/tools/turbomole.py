#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This collects the functions used to preform or analyse TURBOMOLE calculations.

Work with TURBOMOLE version 7.6. 
  (especially with our own developer version made from  the FDE branch)

Created on 27.09.2023

@author: Dávid P. Jelenfi
"""

###############################################################################
# %% Imports
###############################################################################
import os
from sys import exit
from general import check_file, nm_form

import numpy as np
import scipy.linalg as linalg

###############################################################################
# %% TURBOMOLE class
###############################################################################


class TM:
    
    # Common parameters 
    sup_methods = ["dft","riper"]                        # Supported method(s) for SCF
    sup_ricc2_methods = ["cis","ccs","cc2","mp2","adc(2)"]     # Supported method(s) for RICC2
    
    def __init__(self,path=os.getcwd(), fn_molden ="molden.input"):
        """
        Read the input & output data from a TURBOMOLE calculation.
        
        """
        # Instance parameters 
        self.path = path            # Path
        if not os.path.isdir(self.path):
            print("ERROR: {0:} folder does not exist!".format(self.path))
        
        # Dimensions
        self.natoms = -1            # Number of atoms
        self.nsao = -1              # Number of SAOs
        self.ncao = -1              # Number of CAOs
        #self.saocao = None         # CAO -> SAO transformation matrix (from tm2molden_sao)
        self.HOMO = -1              # Number of the HOMO
        
        # Symmetry
        self.sym = False            # Symmetry is used (True) or not (False)
        self.nirrep = 1             # Number of irreps
        self.nsao_p_irrep = []      # Number of SAOs per irrep
        
        # SCF data
        self.methods = []           # Used method(s) for SCF
        self.mos = None             # LCAO coefficients from mos file (nmo,nsao)
        self.mo_eigs = None         # MO eigenvalues from mos file
        self.H = None               # Hamiltonian matrix
        self.S = None               # Overlap matrix
        self.fn_molden=self.path + "/" + fn_molden    # Name of the MOLDEN file (created with tm2molden)
        
        # RICC2 data
        self.ricc2_methods = []     # Used method(s) for RICC2
        self.ricc2_calc = False     # RICC2 calculation was done or not? True/False
        self.exc_calc = False       # Excited state calculation was done or not? True/False
        self.exc_sts = {}           # Dictionary with irrep label (key) and number of exctied states (value) within the key irrep
        #self.exc_ens = None        # Excitation energies from exstates file
        self.fr_core = 0            # Number of frozen core orbitals
        
        # Dyson orbitals
        self.dos = []               # Dyson orbital coefficients from transition density file (ndo,nsao)
        
        # Periodic data
        self.n_lattice = -1        # Number of lattice cells
        self.Fermi = 0             # Fermi energy level
        
        # Initialize input parameters 
        dtgrps = ["symmetry","rundimensions","orbitals","ricc2","excitations","freeze"]
        self.read_control(dtgrps)
        
    ###########################################################################
    # Read data 
    ########################################################################### 
    def read_sym_dtgrp(self):
        fn_control = self.path + "/control"
        with open(fn_control,"r") as control:
            for line in control:
                split = line.split()
                if split[0] == "$symmetry":
                        self.sym = ( split[1].strip() != 'c1' )
                        break  

    def read_dim_dtgrp(self):
        read = False
        fn_control = self.path + "/control"
        with open(fn_control,"r") as control:
            for line in control:
                split = line.split()
                if split[0][0] == "$":
                    if split[0][1:] =="$rundimensions":
                        read = True
                        continue
                    else:
                        read = False
                elif read and split[0][:6] == "natoms":
                    self.natoms = int(split[0].split("=")[1])
    
    def read_orbs_dtgrp(self):
        """
        Read parameters from closed/open shells data group
        // TODO: implement open shells case
        """
        if self.sym:
            self.nirrep = 0
        read_closed = False
        read_open = False
        fn_control = self.path + "/control"
        with open(fn_control,"r") as control:
            for line in control:
                split = line.split()
                if split[0][0] == "$":
                    if split[0][1:] == "closed" and split[1] == "shells":
                        read_closed = True
                        continue
                    elif split[0][1:] == "open" and split[1] == "shells":
                        read_open = True
                        continue
                    else:
                        read_closed = False
                        read_open = False
                elif read_closed:
                    if self.sym:
                        self.nirrep += 1
                    else:
                        if len(line.split("-")) > 1:
                            self.HOMO += int(line.split("-")[1].split(" ")[0])
                        else:
                            self.HOMO += int(line.split()[1])
                elif read_open:
                    print("WARNING: Open shell case have not implemented yet!")

    def read_ricc2_dtgrp(self):
        read = False
        fn_control = self.path + "/control"
        with open(fn_control,"r") as control:
            for line in control:
                if read and line.split()[0] in TM.sup_ricc2_methods:
                    self.ricc2_methods.append(line.split()[0])
                elif line.split()[0][0] == "$":
                    if line.split()[0][1:] == "ricc2":
                        read = True
                        self.ricc2_calc = True
                    else:
                        read = False
                
    def read_cao_info(self):
        fn = self.path + "/ricc2.out"
        with open(fn,"r") as f:
            for line in f:
                if line.split()[:2] == ['frozen', 'occupied']:
                    self.fr_core_cao = int(line.split()[3])
                elif line.split()[:2] == ['active', 'occupied']:
                    self.occup_cao = int(line.split()[3])
                elif line.split()[:2] == ['active', 'virtual']:
                    self.virt_cao = int(line.split()[3])
                elif line.split()[:2] == ['frozen', 'virtual']:
                    self.fr_virt_cao = int(line.split()[3])
                elif line.split()[:2] == ['all', 'together']:
                    self.ncao = int(line.split()[3])
                    
    def read_exc_dtgrp(self):
        """
        Read parameters from excitations data group
        """
        read = False
        fn_control = self.path + "/control"
        with open(fn_control,"r") as control:
            for line in control:
                split = line.split()
                if split[0][0] == "$":
                    if split[0][1:] == "excitations":
                        read = True
                        self.exc_calc = True
                        continue
                    else:
                        read = False
                elif read:
                    if split[0].split("=")[0] == "irrep":
                        if split[0].split("=")[1] == '':
                            irrep = str(split[1])
                        else:
                            irrep = str(split[0].split("=")[1])
                        for i in range(1,len(split)):
                            if split[i].split("=")[0] == "nexc":
                                if split[i].split("=")[1] == '':
                                    nexc = int(split[i+1])
                                else:
                                    nexc = int(split[i].split("=")[1])
                        self.exc_sts.update({irrep : nexc})

    def read_frzcore_dtgrp(self):
        read = False
        fn_control = self.path + "/control"
        with open(fn_control,"r") as control:
            for line in control:
                split = line.split()
                if split[0][0] == "$":
                    if split[0][1:] =="freeze":
                        read = True
                        continue
                    else:
                        read = False
                elif read:
                    core_found = False
                    virt_found = False
                    for item in split:
                        if core_found:
                            self.fr_core = int(item)
                            core_found = False
                        elif virt_found:
                            self.fr_virt = int(item)
                            virt_found = False
                            
                        if "core=" in item:
                            item_sp = item.split("=")
                            if item_sp[1] != "":
                               self.fr_core = int(item_sp[1])
                            else:
                                core_found = True
                        elif "virt=" in item:
                            item_sp = item.split("=")
                            if item_sp[1] != "":
                                self.fr_core = int(item_sp[1])
                            else:
                                virt_found = True
    def read_tddft_dtgrp(self):
        fn_control = self.path + "/control"
        read = False
        with open(fn_control,"r") as control:
            for line in control:
                split = line.split()
                if split[0][0] == "$":
                    if split[0][1:] =="soes":
                        if len(split) > 1 and split[1] == "all":
                            exit("ERROR: Cannot read excitatied state information!")
                        else:
                            read = True
                            continue
                    else:
                        read = False
                if read:
                    irrep = split[0]
                    nexc = int(split[1])
                    self.exc_sts.update({irrep : nexc})
                    
    def read_control_dtgrp(self,key):
        """
        Read a specific datafroup from control file
        """
        if key == "symmetry":
            self.read_sym_dtgrp()
        elif key == "rundimensions":
            self.read_dim_dtgrp()
        elif key == "orbitals":
            self.read_orbs_dtgrp()
        elif key == "ricc2":
            self.read_ricc2_dtgrp()          
        elif key == "excitations":
            self.read_exc_dtgrp()
        elif key == "freeze":
            self.read_frzcore_dtgrp()
        elif key == "tddft":
            self.read_tddft_dtgrp()
        else:
            print(f'Unknown data group: {key}')
    
    def read_control(self,dtgrps):
        """
        Read data from control file
        """     
        fn_control = self.path + "/control"
        check_file(fn_control)
        with open(fn_control,"r") as control:
            for line in control:
                split = line.split()
                if split[0] == "$riper":
                    self.methods.append("riper")
                elif split[0] == "$dft":
                    self.methods.append("dft")
                elif split[0] == "$scfinstab":
                    self.methods.append("tddft")
                    dtgrps.append("tddft")
                    self.exc_calc = True
                    
        if self.methods == []:
            self.methods.append("hf")
        
        for key in dtgrps:
            self.read_control_dtgrp(key)
    
    ###########################################################################
    def read_mos(self):
        """
        Read the LCAO coefficients and the MO eigenvalues from the mos file
        """
        # Local variables
        fn_mos = self.path + "/mos"        
        mos_l = []
        mo_eigs_l = []
        
        check_file(fn_mos)
        if not self.sym:
            with open(fn_mos,"r") as f_mos:
                for line in f_mos:
                    if line[0] == "$" or line[0] == "#":
                        if "format" in line:
                            litem = int(line.split("(")[1].split("d")[1].split(".")[0])
                        elif "$restart" in line:
                            break
                        continue
                    
                    elif "eigenvalue" in line:
                        mo_eigs_l.append(float(line.split("=")[1].split()[0].replace("D","E")))
                        self.nsao = int(line.split("nsaos=")[1]) # only for non symmetric case
                        continue
                    
                    elif line[0] == "0" or line[0] == "-":
                        val = ""
                        k = 0
                        for char in line:
                            if k == litem:
                                mos_l.append(float(val.replace("D","E")))
                                val=char
                                k = 1
                            else:
                                val += char 
                                k += 1 
            self.mo_eigs = np.array(mo_eigs_l)
            self.mos = np.array(mos_l).reshape(self.nsao,self.nsao)
        
        elif self.sym:
            with open(fn_mos,"r") as f_mos:
                mos_l_irreps = []
                self.nsao_p_irrep = []
                irrep0 = ""
                for line in f_mos:
                    if line[0] == "$" or line[0] == "#":
                        if "format" in line:
                            litem = int(line.split("(")[1].split("d")[1].split(".")[0])
                        continue
                    
                    elif "eigenvalue" in line:
                        mo_eigs_l.append(float(line.split("=")[1].split()[0].replace("D","E")))
                        irrep = line.split()[1]
                        if irrep != irrep0:
                            irrep0 = irrep
                            self.nsao_p_irrep.append(int(line.split("nsaos=")[1])) # only for non symmetric case
                            print(irrep," : ",self.nsao_p_irrep[-1])
                            if len(self.nsao_p_irrep) > 1:
                                self.nirrep += 1
                                mos_l_irreps.append(mos_l)
                                mos_l = []
                        continue
                    
                    elif line[0] == "0" or line[0] == "-":
                        val = ""
                        k = 0
                        for char in line:
                            if k == litem:
                                mos_l.append(float(val.replace("D","E")))
                                val=char
                                k = 1
                            else:
                                val += char 
                                k += 1 
                mos_l_irreps.append(mos_l)
            
            self.nsao = sum(self.nsao_p_irrep)
            self.mos = np.zeros((self.nsao,self.nsao))
            dim1,dim2 = 0,0
            for i, block in enumerate(mos_l_irreps):
                block = np.array(block).reshape(self.nsao_p_irrep[i],self.nsao_p_irrep[i])
                dim2 += self.nsao_p_irrep[i]
                self.mos[dim1:dim2,dim1:dim2] = block
                dim1 = dim2
            
            self.mo_eigs = np.array(mo_eigs_l)
            # Order the orbitals according to increasing energy
            idxs = np.argsort(self.mo_eigs)
            self.mo_eigs = self.mo_eigs[idxs]
            self.mos = self.mos[idxs]
                    
    ###########################################################################   
    def read_HS(self):
        """
        Read Hamiltonian (H) and Overlap (S) matrices
        
          Notes:
              - use the $fock keyword in the control file
              - use the modified dscf and riper codes
            
            a) DSCF
                - with DSCF (HF or DFT)  - H and S writen in SAO basis
                - matrices are ordered ???
            
            b) RIPER
                - H and S writen in CAO basis --> transform to SAO (done in this function)
                - matrices are ordered first via orbital types (s,p,d,...), 
                  then via  atomic order in the coord file
                  --> convert to ordered first via atomic then orbital order
                  --> this is done via the CAO->SAO transformation
        """
        
        if self.sym:
            print("Symmetry are not yet implemeted!")
            exit()
        
        #if "riper" in self.methods:            
        #    # Need data from MOLDEN output         
        #    self.H_cao = self.read_riper_M(fn=self.path + "/ksfock.cao")
        #    self.S_cao = self.read_riper_M(fn=self.path + "/overlap.cao")
        #    # !!!! make the CAO -> SAO transformation
        #    if not "saocao" in self.__dict__.keys():
        #        self.read_saocao()
        #    self.H = np.zeros((self.n_lattice,self.nsao,self.nsao))
        #    self.S = np.zeros((self.n_lattice,self.nsao,self.nsao))
        #    for l in range(self.n_lattice):
        #        self.H[l] = self.caosao @ self.H_cao[l] @ self.caosao.T
        #        self.S[l] = self.caosao @ self.S_cao[l] @ self.caosao.T
        
        if "hf" or "dft" or "riper" in self.methods:
            
            if "riper" in self.methods:
                #if not "saocao" in self.__dict__.keys():
                #    self.read_saocao()
                #self.H_cao = self.read_riper_M(fn=self.path + "/ksfock.cao")
                #self.S_cao = self.read_riper_M(fn=self.path + "/overlap.cao")                
                
                print("Only matrices corresponding to the central unit cell are read.")
            
            if self.nsao == -1:
                self.read_mos()
                    
            ov_fn = self.path + "/overlap.sao"
            if not os.path.isfile(ov_fn):
                print("ERROR: overlap.sao not found!\n" +\
                      "       Use $fock keyword in control!")
                exit()
            S_l= []
            with open(ov_fn,"r") as f:
                for line in f:
                    S_l.append(float(line.replace('\n','')))
            
            self.S = np.zeros((self.nsao,self.nsao))  
            k = 0
            for i in range(self.nsao):
                for j in range(i+1):
                    if i == j:
                        self.S[i,j] = S_l[k]
                    else:
                        self.S[i,j] = S_l[k]
                        self.S[j,i] = S_l[k]
                    k += 1
            
            hm_fn = self.path + "/ksfock.sao"
            if os.path.isfile(hm_fn):
                H_l= []
                with open(hm_fn,"r") as f:
                    for line in f:
                        H_l.append(float(line.replace('\n','')))
                
                self.H = np.zeros((self.nsao,self.nsao))  
                k = 0
                for i in range(self.nsao):
                    for j in range(i+1):
                        if i == j:
                            self.H[i,j] = H_l[k]
                        else:
                            self.H[i,j] = H_l[k]
                            self.H[j,i] = H_l[k]
                        k += 1
            else:
                print("WARNING: ksfock.sao not found!\n" 
                      "         Hamiltonian created using MOs.")
                H_mo = np.diag(self.mo_eigs)
                U = self.mos @ self.S
                self.H = U.T @ H_mo @ U      
        else:
            print("Unkown method!")
            exit()
        
    ###########################################################################
    def read_riper_M(self,fn):
        """
        General scheme for reading Kohn-Sham Hamiltonian and Overlap matrix
        from RIPER output files.
        
        Params:
        -----------
            self ...
            fn - str
                 Name of the file.
        
        Returns:
        ------------
            M_m - 3D np.array((n_lattice,ncao,ncao))
                - All H/S to corresponding lattice cells  --> Would be better to write the lattice vector instead !!!!
        
        """
        if self.ncao == -1:
            self.read_molden()
        
        if not os.path.isfile(fn):
            print("ERROR: $riper keyword found, but no %s output file!"%(fn) )
        else:
            i_l, j_l, M_l = [], [], []
            with open(fn,"r") as file:
                for line in file:
                    if line.split()[0] == "Lattice":
                        if int(line.split()[1]) == 1:
                            i_l_tmp, j_l_tmp, M_l_tmp = [], [], []
                        else:
                            i_l.append(i_l_tmp)
                            j_l.append(j_l_tmp)
                            M_l.append(M_l_tmp)
                            i_l_tmp, j_l_tmp, M_l_tmp = [], [], []
                    else:
                        i_l_tmp.append(int(line.split()[0])-1)
                        j_l_tmp.append(int(line.split()[1])-1)
                        M_l_tmp.append(float(line.split()[2]))
                i_l.append(i_l_tmp)
                j_l.append(j_l_tmp)
                M_l.append(M_l_tmp)

            self.n_lattice = len(i_l)
            
            M_m = np.zeros((self.n_lattice,self.ncao,self.ncao))
            for l in range(self.n_lattice):
                for i, j, item in zip(i_l[l],j_l[l],M_l[l]):
                    M_m[l,i,j] = item
                    M_m[l,j,i] = item
            
            return M_m

    
    ###########################################################################
    def read_fermi(self):
        fn = self.path + "/riper.out"
        check_file(fn)
        with open(fn,"r") as f:
            for line in f:
                if line.split()[0:2] == ["Fermi","level"]:
                    self.Fermi = float(line.split()[3])
        
        if self.Fermi == 0:
            print("Warning: Fermi energy didn't found in output!")
        
    ###########################################################################
    def read_saocao(self):
        """
        Read the SAO->CAO transformation matrix made by modified dscf or tm2molden.
        This matrix also convert orbital order to atomic order. 
        
        To transform from CAO to SAO one must use the inverse of this matrix!
        """

        if self.ncao == -1:
            self.read_molden()
        if self.nsao == -1:
            self.read_mos()
            
        self.saocao = np.zeros((self.nsao,self.ncao))
        
        fn = self.path + "/saocao.dat"
        check_file(fn) 
        with open(fn,"r") as f:
            for line in f:
                split = line.split()
                i, j, item = int(split[0])-1, int(split[1])-1, float(split[2])
                self.saocao[i,j] = item
        
        self.caosao = linalg.pinv(self.saocao).T
                
    ###########################################################################
    def read_molden(self, ao_type="sao"):
        """
        Read molden file
        
        ...
        
        Parameters
        ---------              
            ao_type : str
                Type of the AO basis : sao or cao. default = sao
            
            mo_copy : bool
                Read MOs data (True) or not (False - default). 
        
        Returns
        ---------
            atom_df : dictionary {label , index, atom_number,coord}
                 The [ATOMS] section of the MOLDEN file.
                  Name        |    Description        |   Type
                 -----------------------------------------------------
                  label       |   label of atom       |   str
                  inex        |   index of atom       |   int
                  atom_number |   number of atom      |   int
                  coord       |   coordinates of atom |   np.array(3)
                  
            gto_l : list 
                The [GTO] section of the MOLDEN file.
                
            mos_df : dictionary {Sym, Energy, Spin, Occup, Coeff}
                The [MO] section of the MOLDEN file.
                 Name      |    Description        |   Type
                -----------------------------------------------------------
                Sym       |   symmetry of MO      |   str
                Energy    |   energy of MO        |   float
                Spin      |   alpha / beta        |   str
                Occup     |   Occupitation of MO  |   float
                Coeff     |   MO coefficients     |   np.array((nmo,nmo))
        """
        
        if ao_type.lower() != "sao" and ao_type.lower() != "cao":
            print("Wrong ao type! Choise SAO or CAO. ")
            return
            
        self.molden_atom = {"label" : [], "index" : [], "atom_number" : [], "coord" : []}  
        self.molden_mo = {"Sym" : [], "Energy" : [] , "Spin" : [], "Occup" : [], "Coeff": [] }
        self.molden_gto = []
        self.basis_order = {"s": [],"p": [],"d": [],"f": [],"g": []}
        
        if ao_type.lower() == "sao":
            ao_type_d = {"s": 1, "p": 3, "d": 5,"f": 7,"g": 9}
        elif ao_type.lower() == "cao":
            ao_type_d = {"s": 1, "p": 3, "d": 6,"f": 10,"g": 15}
        
        ao_idx=0
        
        check_file(self.fn_molden)
        with open(self.fn_molden) as molden:
            at_copy = False
            gt_copy = False
            mo_copy = False
            coeff_l = [] #np.zeros((nao,1))
            
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
                        at_copy = False
                        gt_copy = False
                        mo_copy = True
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
                    self.molden_atom["label"].append(line.split()[0])
                    self.molden_atom["index"].append(int(line.split()[1]))
                    self.molden_atom["atom_number"].append(line.split()[2])
                    self.molden_atom["coord"].append(np.array(list(np.float_(line.split()[3:6]))))

                elif gt_copy:
                    self.molden_gto.append(line)
                    if  line.split() != []:
                        if line.split()[0] == "s":
                            self.basis_order["s"].append(ao_idx)
                            ao_idx += 1
                        elif line.split()[0] == "p":
                            for i in range(ao_type_d["p"]):
                                self.basis_order["p"].append(ao_idx)
                                ao_idx += 1
                        elif line.split()[0] == "d":
                            for i in range(ao_type_d["d"]): 
                                self.basis_order["d"].append(ao_idx)
                                ao_idx += 1
                        elif line.split()[0] == "f":
                            for i in range(ao_type_d["f"]):
                                self.basis_order["f"].append(ao_idx)
                                ao_idx += 1
                        elif line.split()[0] == "g":
                            for i in range(ao_type_d["g"]):
                                self.basis_order["g"].append(ao_idx)
                                ao_idx += 1
                    
                #Read [MO] section  
                elif mo_copy:   
                    if line.split()[0].lower() == "sym=":
                        self.molden_mo["Sym"].append(line.split()[1])
                        i = 0
                    elif line.split()[0].lower() == "ene=":
                        self.molden_mo["Energy"].append(float(line.split()[1]))    
                    elif line.split()[0].lower() == "spin=":
                        self.molden_mo["Spin"].append(line.split()[1])        
                    elif line.split()[0].lower() == "occup=":
                        self.molden_mo["Occup"].append(float(line.split()[1]))
                    else:
                        i = int(line.split()[0])
                        coeff = float(line.split()[1])
                        coeff_l.append(coeff)
                        if i == ao_idx:
                            self.molden_mo["Coeff"].append(coeff_l)
                            coeff_l = []
            
            if ao_idx == 0:
                print("WARNING: [MO] section missing from %s!" %(self.fn_molden))
            else:
                if ao_type == "cao":
                    self.ncao = ao_idx
                elif ao_type == "sao":
                    if self.nsao == -1:
                        self.nsao = ao_idx
                    else:
                        if self.nsao != ao_idx:
                            print(f'Warning! {ao_idx: 4d}  MO found in the MOLDEN instead of {self.nsao: 4d}.')     

            #self.molden_mo["Coeff"] = np.array(self.molden_mo["Coeff"])
            
            self.orbital_order = []
            for ao_type in self.basis_order:
                self.orbital_order += self.basis_order[ao_type]
    ###########################################################################
    def read_exc_ens(self):
        """
        Read the exctiations energies from the TURBOMOLE exstates.
        Necessary file(s): exstates
        """
        # // TODO: implement symmetry
        # Implemented only for one irrep -> C1 symmetry required !!!!
        
        if len(self.ricc2_methods) > 0:
            self.exc_ens = []
            find = False 
            nstates = 0
            i = 0
            sline= "$excitation_energies_{0:}".format(self.ricc2_methods[0].upper()) 
            fn = self.path + "/exstates"
            check_file(fn)
            with open(fn,"r") as exstates:
                for line in exstates:
                    if find:
                        self.exc_ens[i] = float(line.split()[1])
                        i += 1
                        if i >= nstates:
                            find = False
                    if sline in line:
                        nstates = int(line.split("nstates=")[1])
                        if nstates != self.exc_sts['a']:
                            print("ERROR: the number of excitations from control and from exstates are not agree!")
                            exit()
                        self.exc_ens = np.zeros((nstates))
                        find = True
        elif "tddft" in self.methods:
            fn = self.path + "/exspectrum"
            nstates = self.exc_sts['a']
            self.exc_ens = np.zeros((nstates))
            check_file(fn)
            with open(fn,"r") as f:
                for line in f:
                    split = line.split()
                    if split[0] != "#":
                        st = int(split[0])-1
                        en = float(split[2])
                        self.exc_ens[st] = en
        else:
            exit("ERROR: Not found any excited state calculation!")
    
    ###########################################################################
    def read_tdens(self,trn=True):
        """
        Read the transition density file created by TURBOMOLE. This file consist of
        the transition density matrix for a given state in a Fortran format (sorfolytonos). 
        
        The original binary file has to be converted to readable format. 
          --> It can be done with trd_cao_b2l.x or trd_mo_b2l.x scripts.
        
        The transition density matrices:
            - are in CAO basis --> transformed to SAO
            - are in orbital order --> transformed to atomic order
            use saocao matrix from TM to do both.
        
        Params:
        ------------
        trn(bool) : transform to SAO basis or not. Default : True
        
        """   
        if len(self.exc_sts) > 1:
            print("ERROR: Only defined for C1 symmetry!")

        if set(self.ricc2_methods).isdisjoint(["ccs","adc(2)"]) and set(self.methods).isdisjoint(["tddft"]):
            print("ERROR: Transition density evaluation is avaible only for CCS and ADC(2)!")
            exit()
            
        # transform to SAO basis
        if trn:
            nbf = self.nsao
            if not "saocao" in self.__dict__.keys():
                self.read_saocao()
        else:
            if self.ncao == -1:
                self.read_molden() # find a better way to get ncao !!!!
            nbf = self.ncao
            
        # irrep loop - works only for c1
        for i_irrep in range(len(self.exc_sts)): # Implement for other symmetries !!!!
            irrep = list(self.exc_sts.keys())[i_irrep]
            #nirrep = len(list(self.exc_sts.keys()))
            nst = self.exc_sts[irrep]
            # Right vectors
            if not set(self.ricc2_methods).isdisjoint(["adc(2)"]):
                self.tdR = np.zeros((nst,nbf,nbf))
                for st in range(nst):
                    fn_trd_R = self.path + "/trd-R-1{0:}-{1:03d}.dat".format(irrep,st+1)
                    if not os.path.isfile(fn_trd_R):
                        print("ERROR: %s not found!\n" %(fn_trd_R) +\
                              "       Use 'spectrum ... export=cao' in control and generate the file with trd_cao_b21.x!")
                        exit()                
                    tdens_l = []    
                    with open(fn_trd_R) as file:
                        tdens_l.append([float(line) for line in file])
                    tmp_td = np.array(tdens_l)
                    tmp_td.resize(self.ncao,self.ncao)
                    if trn:
                        self.tdR[st] = self.caosao @ tmp_td @ self.caosao.T
                    else:
                        self.tdR[st] = tmp_td
                
            # Left vectors
            if not set(self.ricc2_methods).isdisjoint(["ccs","cc2"]):
                self.tdL = np.zeros((nst,nbf,nbf))
                for st in range(nst):
                    fn_trd_L = self.path + "/trd-L-1{0:}-{1:03d}.dat".format(irrep,st+1)
                    if not os.path.isfile(fn_trd_L):
                        print("ERROR: %s not found!\n" %(fn_trd_L) +\
                            "       Use 'spectrum ... export=cao' in control and generate the file with trd_cao_b21.x!")
                        exit()                
                        
                    tdens_l = []    
                    with open(fn_trd_L) as file:
                        tdens_l.append([float(line) for line in file])
                    tmp_td = np.array(tdens_l)
                    tmp_td.resize(self.ncao,self.ncao)
                    if trn:
                        self.tdL[st] = self.caosao @ tmp_td @ self.caosao.T
                    else:
                        self.tdL[st] = tmp_td
                    
            # TDDFT
            if not set(self.methods).isdisjoint(["tddft"]):
                self.tdL = np.zeros((nst,nbf,nbf))
                for st in range(nst):
                    fn_trd_L = self.path + "/trd-{0:03d}.dat".format(st+1)
                    if not os.path.isfile(fn_trd_L):
                        print("WARNING: %s not found!\n" %(fn_trd_L))
                        continue
                    else:
                        tdens_l = []    
                        with open(fn_trd_L) as file:
                            tdens_l.append([float(line) for line in file])
                        tdens_l = tdens_l[0]
                        tmp_td = np.zeros((self.ncao,self.ncao))
                        k = 0
                        for i in range(self.ncao):
                            for j in range(i+1):
                                if i == j:
                                    tmp_td[i,j] = tdens_l[k]
                                else:
                                    tmp_td[i,j] = tdens_l[k]
                                    tmp_td[j,i] = tdens_l[k]
                                k += 1
                        if trn:
                            self.tdL[st] = self.caosao @ tmp_td @ self.caosao.T
                        else:
                            self.tdL[st] = tmp_td

    ###########################################################################               
    def create_dos(self,norm=False,cut=False,ao_type="sao"):
        """
        Create dyson orbital from transition density calculated with continuum orbital strategy
        ...
        Params:
            norm - bool
                 - Normalize Dyson orbitals (True) or not (False: default)?
            cut  - bool
                 - Cut out coefficient correspont to continuum orbital (True) or not (False: default)?
        """
        
        # check calculation type
        if not self.exc_calc:
            print("ERROR: Excited state(s) was not calculated!")
            exit()
            
        #if self.mos == None:
        #    self.read_mos()
        
        if self.ncao == -1:
            self.read_molden()
        
        # read H and S
        self.read_HS()
        
        # read transition densities
        if ao_type == "sao":
            self.read_tdens()
        elif ao_type == "cao":
            self.read_tdens(trn=False)
        else:
            exit(f'ERROR in create_dos: {ao_type:} is not a valid ao_type!')
        
        # find continuum orbital
        tmp = np.where(abs(self.mo_eigs) < 1e-6)[0] # continuum MO
        if tmp is None:
            print("ERROR: Continuum orbital is not found! (There is no orbital with energy less then 1E-6.)")
        if len(tmp) == 1:
            self.cont_cao = tmp[0]
            self.cont_sao = np.where(self.mos[tmp[0]] == 1e+00)[0][0] # continuum AO
            print(f'  Continuum orbital is the {self.cont_cao+1: 3d}th MO\CAO.')
            print(f'  Continuum orbital is the {self.cont_sao+1: 3d}th SAO.')
            # ea and ip need only to define that the DO coorespont to a column or a row
            # BUT TURBOMOLE transisition density is symmetric somehow 
            # tested only for adc(2), ccs !!! 
            #print("Only tested for CCS, CC2 and ADC(2)!")
            # find occupation of continuum orbital
            #  - occupied   -> EA states
            #  - unoccupied -> IP states
            i = self.molden_mo["Sym"].index(str(self.cont_cao - self.fr_core + 1)+"a")
            if self.molden_mo["Occup"][i] == 0.:
                self.exc_typ = "ip"
            elif self.molden_mo["Occup"][i] == 2.:
                self.exc_typ = "ea"
            else:
                print("ERROR: Invalid occupation number for continuum orbital:", self.molden_mo["Occup"][tmp[0]])
        elif len(tmp) > 1:
            print("WARNING: More then one orbital found with energy below 1E-8!")
            exit()
        elif len(tmp) < 1:
            print("ERROR: Continuum orbital was not found!")
            exit()
        
        if len(self.exc_sts) > 1:
            print("ERROR: Only defined for C1 symmetry!")
        
        if cut:
            S = np.zeros((self.nsao-1,self.nsao-1))
            S[:self.cont_sao,:self.cont_sao] = self.S[:self.cont_sao,:self.cont_sao]
            S[self.cont_sao:,self.cont_sao:] = self.S[self.cont_sao+1:,self.cont_sao+1:]
            S[:self.cont_sao,self.cont_sao:] = self.S[:self.cont_sao,self.cont_sao+1:]
            S[self.cont_sao:,:self.cont_sao] = self.S[self.cont_sao+1:,:self.cont_sao]
            self.S = S #!!!!!!
        
        print(f'  Creating {self.exc_typ:} type Dyson orbitals.')
        # irrep loop - works only for c1
        for i_irrep in range(len(self.exc_sts)): # Implement for other symmetries !!!!
            irrep = list(self.exc_sts.keys())[i_irrep]
            nst = self.exc_sts[irrep]
                       
            if cut:
                self.dos = np.zeros((nst,self.nsao-1))
            else:
                self.dos = np.zeros((nst,self.nsao))
            
            for st in range(nst):
                if not set(self.ricc2_methods).isdisjoint(["adc(2)"]): # Right vectors
                    if self.exc_typ == "ip":
                        dos = self.tdR[st].T[self.cont_sao] 
                    elif self.exc_typ == "ea":
                        dos = self.tdR[st][self.cont_sao]
                elif not set(self.ricc2_methods).isdisjoint(["ccs","cc2"]) or not set(self.methods).isdisjoint(["tddft"]): # Left vectors
                    if self.exc_typ == "ip":
                        dos = self.tdL[st][self.cont_sao]
                    elif self.exc_typ == "ea":
                        dos = self.tdL[st].T[self.cont_sao]
                else:
                    print("ERROR: %s method have not implemented yet!" %(self.ricc2_methods))
                
                dos = dos / 2 # due to tdR contain alpha+beta

                if cut:
                    self.dos[st] = np.delete(dos,self.cont_sao)
                else:
                    self.dos[st] = dos
                if norm:
                    norm = 1.0 / np.sqrt(self.dos[st] @ self.S @ self.dos[st])
                    self.dos[st] = norm * self.dos[st] 
                
                
    ###########################################################################               
         
    def write_dos2molden(self):     
        
        fn = self.path + '/DOs.molden'  
        ao_type = "sao" #hard coded
        
        if self.ncao == -1:
            self.read_molden()
        
        with open(fn,'w') as f:
            f.write("[Molden Format]\n")
            
            #Write ATOM section
            f.write("[ATOMS] au \n")
            
            for atom_i in range(len(self.molden_atom['label'])):
                f.write('{0:>7}'.format(self.molden_atom["label"][atom_i]))
                f.write('{0:>6.0f}'.format(atom_i+1))
                f.write('{0:>3.0f}'.format(float(self.molden_atom["atom_number"][atom_i])))
                for coord_i in self.molden_atom["coord"][atom_i]:
                    f.write("      ")
                    f.write('{0:>14.8f}'.format(coord_i))
                f.write("\n")
            
            #write GTO section
            f.write("[GTO]\n")
            for line in self.molden_gto:
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
                f.write("[5D]\n")
            #elif ao_type.lower() == "cao": #default in MOLDEN
            #    f.write("[6D]\n[10F]\n[15G]\n")
            else:
                print("Wrong ao type! Choise SAO or CAO. ")
            #Write MO section
            f.write("[MO]\n")

            if len(list(self.exc_sts.keys())) > 1:
                print("WARNING: Implemented only for C1 symmetry!")
            
            for st in range(self.exc_sts['a']):
                f.write(f'  Sym=  {st+1}a\n')
                f.write(f'  Ene=  {self.exc_ens[st]: 10.6f}\n')
                f.write( "  Spin= Alpha\n")
                if self.exc_typ == "ip":
                    f.write("  Occup= 2.000\n")
                elif self.exc_typ == "ea":
                    f.write("  Occup= 0.000\n")
                else:
                    print("Implemented only for IP and EA states!")
                for coeff_i, coeff in enumerate(self.dos[st]):
                    f.write('{0: >4}{1:<2}{2: 10.8f}\n'.format(coeff_i+1,"",coeff))
    
    ###########################################################################               
         
    def write_molden(self,fn,eig,coeffs):     
        
        ao_type = "sao" #hard coded
        
        if self.ncao == -1:
            self.read_molden()
        
        with open(fn,'w') as f:
            f.write("[Molden Format]\n")
            
            #Write ATOM section
            f.write("[ATOMS] au \n")
            
            for atom_i in range(len(self.molden_atom['label'])):
                f.write('{0:>7}'.format(self.molden_atom["label"][atom_i]))
                f.write('{0:>6.0f}'.format(atom_i+1))
                f.write('{0:>3.0f}'.format(float(self.molden_atom["atom_number"][atom_i])))
                for coord_i in self.molden_atom["coord"][atom_i]:
                    f.write("      ")
                    f.write('{0:>14.8f}'.format(coord_i))
                f.write("\n")
            
            #write GTO section
            f.write("[GTO]\n")
            for line in self.molden_gto:
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
                f.write("[5D]\n")
            #elif ao_type.lower() == "cao": #default in MOLDEN
            #    f.write("[6D]\n[10F]\n[15G]\n")
            else:
                print("Wrong ao type! Choise SAO or CAO. ")
            #Write MO section
            f.write("[MO]\n")
            
            for st in range(len(eig)):
                f.write(f'  Sym=  {st+1}a\n')
                f.write(f'  Ene=  {eig[st]: 10.6f}\n')
                f.write( "  Spin= Alpha\n")
                f.write( "  Occup= 2.000\n")
                for coeff_i, coeff in enumerate(coeffs[st]):
                    f.write('{0: >4}{1:<2}{2: 10.8f}\n'.format(coeff_i+1,"",coeff))

    ###########################################################################               

    def tm2mrcc(self,ao_type="sao"):
        """
         Reorder TM AO matrices to MOLDEN order 
         and fix the relative phase of functions higher then d.
         
         Input args:
            M - numpy.array
            dim - dimension of M
            axis - how many axis will be reordered
         -------------------------------------------------------
            TURBOMOLE order and phase correction:
                p: same as MOLDEN order
                d: (d0,d+1,d-1,d+2,d-2) <- (d0,d+1,d-1,d-2,d+2)
                f: (f0,f+1,f-1,f+2,f-2,f+3,f-3) <- (f0,f+1,f-1,f-2,f+2,f+3,f-3)
                   definition of f-3 differ: 
                      MRCC ~ (3xxy-yyy) <-> TURBOMOLE = (yyy-3xxy)/sqrt(24)
                TODO: g,h,i
        """
        self.read_molden(ao_type=ao_type)
        
        # read basis infromation from MOLDEN for reordering to TM order
        if self.basis_order["g"] != []:
           print("WARNING: Reordering of g or higher functions is not yet implemented!!!")
        
        self.read_mos()
        
        c = self.mos.copy()
        #c = self.molden_mo["Coeff"][0]
        
        I=np.arange(c.shape[0])    
        
        idx = []
        f3b = []
        di,fi = 0,0
        for i in range(self.nsao):
            if i in self.basis_order["d"]:
                    if di == 3:
                        idx.append(i+1)
                        di += 1
                    elif di == 4:
                        idx.append(i-1)
                        di = 0
                    else:
                        idx.append(i)
                        di += 1
            elif i in self.basis_order["f"]:
                    if fi == 3:
                        idx.append(i+1)
                        fi += 1
                    elif fi == 4:
                        idx.append(i-1)
                        fi +=1
                    elif fi == 6:
                        idx.append(i)
                        f3b.append(i)
                        fi = 0
                    else:
                        idx.append(i)
                        fi += 1
            else:
                idx.append(i)
                
        #print(self.basis_order)
        #print(idx)
        
        c = c[np.ix_(I, idx)]
                
        # correct f-3 definition by swapping sign
        c[:,f3b] = -1*c[:,f3b]
 
        #self.write_molden(fn="MOLDEN_sao",eig=self.mo_eigs,coeffs=c)
    
        #def molden2mrcc(self):
        """
         Reorder MOLDEN AO matrices to MRCC order
         
         Input args:
            M - numpy.array
            dim - dimension of M
            axis - how many axis will be reordered 
         -------------------------------------------------------
            MOLDEN order:
                p: (pz,py,px) <- (px,py,pz)
                d: (d-2,d-1,d0,d+1,d+2) <- (d0,d+1,d-1,d+2,d-2)
                f: (f-3,f-2,f-1,f0,f+1,f+2,f+3) <- (f0,f+1,f-1,f+2,f-2,f+3,f-3)
                TODO: g,h,i
        """
        # read MOLDEN.perm for reordering to MRCC order
        idx = []
        fn_perm = self.path + "/MOLDEN.perm"
        check_file(fn_perm)		
        with open(fn_perm) as m:
            for line in m:
                for i in line.split():
                    idx.append(int(i)-1)
        idx_to_MLD = np.array(idx)
        idx_to_MRCC_idx = np.argsort(idx_to_MLD)
        
        #self.read_molden()
        #c = np.array(self.molden_mo["Coeff"])   
        #I=np.arange(c.shape[0])
        c = c[np.ix_(I, idx_to_MRCC_idx)]
        
        with open("mrcc_mos","w") as f:
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    f.write(f'{c[i,j]: .25e}\n')

    ###########################################################################               

    def molden2tm(self,ao_type='sao'):
        """
         Reorder MOLDEN AO matrices to TURBOMOLE order 
         and fix the relative phase of functions higher then d.
         
         Input args:
            M - numpy.array
            dim - dimension of M
            axis - how many axis will be reordered
         -------------------------------------------------------
            TURBOMOLE order and phase correction:
                p: same as MOLDEN order
                d: (d0,d+1,d-1,d+2,d-2) -> (d0,d+1,d-1,d-2,d+2)
                f: (f0,f+1,f-1,f+2,f-2,f+3,f-3) -> (f0,f+1,f-1,f-2,f+2,f+3,f-3)
                   definition of f-3 differ: 
                      MRCC ~ (3xxy-yyy) <-> TURBOMOLE = (yyy-3xxy)/sqrt(24)
                TODO: g,h,i
        """
        
        self.read_molden(ao_type=ao_type)
        
        c = self.molden_mo["Coeff"]
        
        # read basis infromation from MOLDEN for reordering to TM order
        if self.basis_order["g"] != []:
           print("WARNING: Reordering of g or higher functions is not yet implemented!!!")
        
        I=np.arange(self.nsao)    
        idx_to_TM = []
        f3b = []
        di,fi = 0,0
        for i in range(self.nsao):
            if i in self.basis_order["d"]:
                    if di == 3:
                        idx_to_TM.append(i+1)
                        di += 1
                    elif di == 4:
                        idx_to_TM.append(i-1)
                        di = 0
                    else:
                        idx_to_TM.append(i)
                        di += 1
            elif i in self.basis_order["f"]:
                    if fi == 3:
                        idx_to_TM.append(i+1)
                        fi += 1
                    elif fi == 4:
                        idx_to_TM.append(i-1)
                        fi +=1
                    elif fi == 5:
                        idx_to_TM.append(i)
                        fi += 1
                    elif fi == 6:
                        idx_to_TM.append(i)
                        f3b.append(i)
                        fi = 0
                    else:
                        idx_to_TM.append(i)
                        fi += 1
            else:
                idx_to_TM.append(i)

        c = c[np.ix_(I, idx_to_TM)]
        c[:,f3b] = -1*c[:,f3b]
        
        self.mos = c 
        self.mo_eigs = self.molden_mo["Energy"]

    ###########################################################################               
                    
    def write_mos(self,fn="mos2"):
        """
        Write TURBOMOLE mos file from a mos matrix.
        INPUT:
            mo_eigs - 1D np.array - contain energy of molecular orbitals
            mos   - 2D np.array - contain molecular orbital coefficients (nmo,nao)
        """

        f = open(fn,"w")
        f.write("$scfmo    scfconv=7   format(4d20.14)\n")
        f.write("# SCF total energy is        0.0000000000 a.u.\n") #-113.7702238482 a.u.\n")
        f.write("#\n")

        extra_line_break = False
        for i in range(self.nsao):
            if extra_line_break:
                f.write("\n")
            f.write("     {0:}  a      eigenvalue={1:}   nsaos={2:}\n".format(i+1,self.mo_eigs[i],self.nsao))
            for j in range(self.nsao):
                val = self.TMformat(self.mos[i,j])
                f.write(val)
                extra_line_break = True
                if (j+1)%4==0:
                    f.write("\n")
                    extra_line_break = False
        if extra_line_break:
            f.write("\n")           
        f.write("$end\n")
        f.close()     

    ###########################################################################               
        
    def TMformat(self,n):
        """
        Convert a number to TURBOMOLE format used in mos file.

        ...

        Parameters
        ----------
        n : float/int
            The number which you want convert.

        Returns
        -------
        str
            The number converted to the TURBOMOLE format.

        """
        a = '{:.13E}'.format(float(abs(n)))
        e = a.find('E')
        if n < 0:
            return '-.{}{}D{}{:02d}'.format(a[0],a[2:e],a[e+1:e+2],abs(int(a[e+1:])*1+1))
        else:
            return '0.{}{}D{}{:02d}'.format(a[0],a[2:e],a[e+1:e+2],abs(int(a[e+1:])*1+1))            

    ###########################################################################               
        
    def ao_rotation_matrices(self,R_cart):
        """
        Return rotation matrices for real spherical harmonic orbitals (s, p, d, f)
        using PySCF's Dmatrix implementation.

        Args:
            R_cart: 3x3 Cartesian rotation matrix

        Returns:
            Tuple of rotation matrices for (s, p, d, f) orbitals
        """
        
        from scipy.spatial.transform import Rotation
        
        try:
            from pyscf.symm.Dmatrix import Dmatrix
        except ImportError:
            print("pyscf is needed for AO matrix rotation.")
            return 0
        
        rot = Rotation.from_matrix(R_cart)
        alpha, beta, gamma = rot.as_euler('zyx', degrees=False)

        # Use PySCF Dmatrix to get real spherical harmonic rotation matrices
        R_s = np.eye(1)
        R_p = Dmatrix(1, alpha, beta, gamma, reorder_p=True)
        R_d = Dmatrix(2, alpha, beta, gamma, reorder_p=True)
        R_f = Dmatrix(3, alpha, beta, gamma, reorder_p=True)

        R_d = R_d[np.ix_([2,3,1,0,4],[2,3,1,0,4])]
        R_f[:,-1] = -1*R_f[:,-1]
        R_f[-1,:] = -1*R_f[-1,:]
        R_f = R_f[np.ix_([3,4,2,1,5,6,0],[3,4,2,1,5,6,0])]

        return R_s, R_p, R_d, R_f

    ###########################################################################               

    def rotate_ao_matrices(self,R_cart,rot_mos=False):
        """
        Rotate the Fock and Overlap matrices in AO basis. 
        
        Args:
            R_cart: 3x3 Cartesian rotation matrix
            rot_mos: Rotate the MOs too
        """
        
        if not hasattr(self, 'basis_order'):
            self.read_molden()
        
        R = self.ao_rotation_matrices(R_cart)
        if R == 0:
            return
        
        u = []
        # add rotational matrices
        idx = 0
        while not idx == self.nsao:
            if idx in self.basis_order['s']:
                u.append(R[0])
                idx += 1
            elif idx in self.basis_order['p']:
                u.append(R[1])
                idx += 3
            elif idx in self.basis_order['d']:
                u.append(R[2])
                idx += 5
            elif idx in self.basis_order['f']:
                u.append(R[3])
                idx += 7       
            elif idx in self.basis_order['g']:
                print("ERROR: higher then f functions are not implemented for AO basis rotation.")
                return
            else:
                print("ERROR: Basis info in the MOLDEN file is missing.")
                return
              
        U = linalg.block_diag(*u)
        
        self.H = U @ self.H @ U.T
        self.S = U @ self.S @ U.T
        
        if rot_mos:
            self.mo_eigs, self.mos = linalg.eigh(self.H, self.S)
        else:
            print("WARNING: The MOs cofficient matrix are not rotated!")