#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This collects the functions used to preform or analyse MRCC calculations.

Work with MRCC last modified developer version(s)

Created on 06.07.2024
Last Modified on 06.06.2025

@author: Dávid P. Jelenfi
"""

###############################################################################
# %% Imports
###############################################################################
import os, sys
import numpy as np
import scipy.linalg as linalg
import struct

import re
from general import check_file, check_file2, read_molden, nm_form

###############################################################################
# %% MRCC class
###############################################################################
class MRCC:
    
    # Common parameters 
    
    def __init__(self,path=os.getcwd(),sub="",read_mos=True):
        """
        Initialise the MRCC class variables.
        """
        # Instance parameters 
        self.path = path            # Path
        if not os.path.isdir(self.path):
            print("ERROR: {0:} folder does not exist!".format(self.path))
               
        # Input and output files
        self.sub = sub
        self.get_fns()       # file names
        
        # Correlatied calculations
        self.nstate = -1             # MRCC nstate includes the ground state as well, therefore self.nstate is initialized to -1
        
        # Embedding
        self.embed = False

        # Initialize input parameters 
        self.read_minp()
        self.getkeywd()
        
        # compare MINP to KEYWD
        if self.nstate != int(self.keywd[" nstate"])-1:
            print("WARNING: nstate mistmatch!")
            self.nstate = int(self.keywd[" nstate"])-1
        
        # Dimensions
        self.read_dims()
        
        self.ccprog=self.keywd[" ccprog"][:-1].lower()
        
        if read_mos:
            #if os.path.isfile(self.fn_molden):
            #    # Read MOLDEN
            #    self.read_molden(read_coeff=False)
            #else:
            #    print(f'WARNING: {self.fn_molden:} file is missing!')
            
            if os.path.isfile(self.fn_c) and os.path.isfile(self.fn_eigs):
                # Read MOs
                self.read_mos()
            else:
                print(f'WARNING: {self.fn_c:} or {self.fn_eigs:} file is missing!')                     

    def get_fns(self):
        """
        Define necessary input and output file names
        """
        self.fn_minp = self.path + "/MINP"       # Name of MINP input file
        
        self.fn_molden = self.path + "/MOLDEN"   # Name of the MOLDEN file

        self.fn_ov     = self.path + "/ov"       # AO Overlap matrix
        self.fn_h      = self.path + "/fock"     # AO Fock matrix
        self.fn_c      = self.path + "/mos"      # LCAO coefficients
        self.fn_eigs   = self.path + "/moeig"    # MO eigenvalues
        #self.fn_hc    = self.path + "/core"     # Core Hamiltonian (one-electron integrals + embedding pot. in case of Huzinaga)
        #self.fn_p     = self.path + "/dens"     # density matrix
        
        self.out       = self.path + "/dmrcc.out" # output file
        
        if self.sub != "":
            if "huzi" in self.sub:
                self.sub = self.sub.split("_")[0]
                self.fn_h = self.path + "/huzi"     # Huzinaga operator 
            self.fn_h   += "_" + self.sub          
            #self.fn_hc += "_" + self.sub          
            #self.fn_p  += "_" + self.sub          
            if self.sub == "em1":
                self.fn_molden += "_" + self.sub   # Name of the MOLDEN file

            if self.sub == "em1" or self.sub == "em3":
                self.fn_c += "_" + self.sub
                self.fn_eigs += "_" + self.sub
            elif self.sub == "em2":
                self.fn_c += "_" + self.sub
                self.fn_eigs += "_em3" # there is no MO eigenvalues file for locilised orbitals after 1. SCF
            elif self.sub == "tild":
                self.fn_c += "_em3"    # there is no canonical LCAO file for high-level Fock
                self.fn_eigs += "_em3" # there is no MO eigenvalues file for high-level Fock
        
        self.fn_ov     += ".dat"
        self.fn_h      += ".dat"
        self.fn_c      += ".dat"
        self.fn_eigs   += ".dat"
        #self.fn_hc    += ".dat"
        #self.fn_p     += ".dat"
    
    # ---------------------------------------------------------------------------
    # %% Functions for read input data
    def read_version(self):
        """
        Read the version of the MRCC from the output file.
        Works only if the output file name is dmrcc.out and it is one of our modified MRCC version!
        """
        self.version = 2
        if check_file2(self.out) == 0:
            with open(self.out,"r") as out:
                for line in out:
                    if "Last modification date" in line:
                        if "October" in line:
                            self.version = 1
                        elif "November" in line:
                            self.version = 2
                        else:
                            print(f'WARNING! Unkown version: {line:}')
                            self.version = 1
                        break
                    elif "Version" in line:
                        if line.split(":") == "25.1.0":
                            self.version = 2
                        break
                    
                    elif "Executing minp..." in line:
                        print(f'WARNING! Version infromation not found. Version set to 2.')
                        break
        else:
            print("WARNING! Used version 2.")
            
        print("MRCC version: ", self.version)
          
    def read_dims(self):
        """
        Read necessary dimensions from MRCC output files
        """
        self.nsao = self.getvar("nbasis")          # number of SAO of the full system 
        self.ndao = self.getvar("ndao")            # number of AOs in the linearly dependent subspace of the full AO basis
        if self.ndao != 0:
            print(f'{self.ndao: 2d} AOs in the linearly dependent subspace.')
        self.norb = self.nsao #- self.ndao         # number of MOs of the full system
        nal = self.getvar("nal")                   # number of alpha electrons
        nbe = self.getvar("nbe")                   # number of beta electrons
        if nal != nbe:
            print("WARNING: Open-shell system!")
        self.nocc = nal                            # number of occupied orb. of the full system (number of alpha electrons: nal - alpha, nbe - beta)
        self.ncore = self.getvar("ncore")          # number of core orb.     
        self.nvir = self.norb - self.nocc          # number of virtual orb. of the full system
        self.natoms = self.getvar("natoms")              # number of atoms
        self.nbset = self.getvar("nbset")                # number of basis sets (basic, auxilary, etc.)
        if self.embed:
            self.nocc_b = self.getvar("nfroz")         # number of occupied orb. of the enviroment (equal with the number of frozen occupied orb.)
            self.nocc_a = self.nocc - self.nocc_b      # number of occupied orb. of the active system
            self.nvir_b = self.getvar("nvfroz")        # number of virtual orb. of the enviroment (equal with the number of frozen virtual orb.)
            self.nvir_a = self.nvir - self.nvir_b      # number of virtual orb. of the active system
            self.norb_a = self.nocc_a + self.nvir_a    # number of orb. of the active system
            self.norb_b = self.nocc_b + self.nvir_b    # number of orb. of the enviroment
            
        natrange_tmp = self.getvar("natrange")           # from AO idx to AO idx per atom
        self.natrange = np.reshape(natrange_tmp, (self.nbset*self.natoms,2)).T        
        
    def getvar(self,vname,typ='int'):
        """
        Get varible 'vname' from file VARS
           be carefull with this function!!!
        """
        bvname = vname.ljust(10).encode("utf-8")
        
        fn = self.path + '/VARS'
        check_file(fn)
        with open(fn,'rb') as f:
            fbin = f.read()
        
        pos = fbin.find(bvname)
        if pos == -1:
            print(f'ERROR: {vname:} not found in file VARS')
            return 0
            #sys.exit(f'ERROR: {vname:} not found in file VARS')
        pos += 10
        n = int.from_bytes(fbin[pos:pos+8],"little")
        pos += 8
        if typ[:3].lower() == 'int':
            if n > 8:
                if n%8 == 0:
                    var = []
                    for i in range(int(n/8)):
                        p1, p2 = pos+i*8, pos+(i+1)*8
                        var.append(int.from_bytes(fbin[p1:p2],"little"))
                else:
                    print(f'ERROR: {vname:} is not an integer or integer list')
                    return
                    #sys.exit(f'ERROR: {vname:} is not an integer / integer list')
            else:
                var = int.from_bytes(fbin[pos:pos+n],"little")
        
        elif typ.lower() == "float" or typ.lower() == "real":
            if n > 8:
                if n%8 == 0:
                    var = []
                    for i in range(int(n/8)):
                        p1, p2 = pos+i*8, pos+(i+1)*8
                        var.append(struct.unpack('d', fbin[p1:p2])[0])
                else:
                    print(f'ERROR: {vname:} is not a float or float list')
                    return
                    #sys.exit(f'ERROR: {vname:} is not an integer / integer list')
            else:
                var = struct.unpack('d', fbin[pos:pos+n])[0]

        elif typ[:3].lower() == 'str':
            var = fbin[pos:pos+n].decode("utf-8")

        else:
            print("ERROR: unkonwn data type.")
            return
            #sys.exit("ERROR: unkonwn data type.")
      
        return var
    
    def getkeywd(self):
        """
        Read parameters from file KEYWD to diconary
        """
        
        fn = self.path + "/KEYWD"
        check_file(fn)
        self.keywd = {}
        with open(fn,"r") as f:
            for line in f:
                key, value = line.split("=")
                self.keywd.update({key : value})
     
    def read_minp(self):
        """
        Read data from MINP input file
        """        
        self.methods = []
        check_file(self.fn_minp)
        with open(self.fn_minp,"r") as minp:
            for line in minp:
                line = line.split("!")[0].strip()
                if len(line) > 1:
                    line = line.replace("="," ")
                    split = line.split()
                    if split[0] == "calc":
                        method = split[1].lower()
                        if "(" in method or ")" in method:
                            method = re.sub(r'\((.*?)\)', r'p\1', method)
                        self.methods.append(method)
                    
                    elif split[0] == "nstate":
                        self.nstate += int(split[1])
    
                    elif split[0] == "embed":
                        self.embed = True
                        
        if self.nstate == -1:
            self.nstate=0
        
    def read_coord(self):
        """
        Read the coordinates
        """
        found_geom = False
        atom = -2
        
        if not hasattr(self, 'natoms'):
            self.read_dims()
            
        self.coords = np.zeros((self.natoms,3))
        self.atom_list = []
        check_file(self.fn_minp)
        with open(self.fn_minp,"r") as minp:
            for line in minp:
                line = line.strip()
                if line == "":
                    continue
                if "!" in line:
                    if line[0] == "!":
                        continue
                    else:
                        line = line.split("!")[0].strip()  
                                          
                split = line.split("=")
                if split[0].strip() == "geom":
                    found_geom = True
                    continue
                
                if found_geom:
                    atom += 1
                
                if atom >= 0 and atom < self.natoms:
                    a,x,y,z = line.split()
                    self.atom_list.append(str(a))
                    self.coords[atom,0] = float(x)
                    self.coords[atom,1] = float(y)
                    self.coords[atom,2] = float(z)
    
    def mrcc2tm(self,M,dim=2,axis=2):
        """
        Reorder MRCC AO matrices to TURBOMOLE order.
        """
        
        M_molden = self.mrcc2molden(M=M,dim=dim,axis=axis)
        M_tm = self.molden2tm(M=M_molden,dim=dim,axis=axis)
        
        return M_tm
    
    def mrcc2molden(self,M,dim=2,axis=2):
        """
         Reorder MRCC AO matrices to MOLDEN order
         
         Input args:
            M - numpy.array
            dim - dimension of M
            axis - how many axis will be reordered 
         -------------------------------------------------------
            MOLDEN order:
                p: (pz,py,px) -> (px,py,pz)
                d: (d-2,d-1,d0,d+1,d+2) -> (d0,d+1,d-1,d+2,d-2)
                f: (f-3,f-2,f-1,f0,f+1,f+2,f+3) -> (f0,f+1,f-1,f+2,f-2,f+3,f-3)
                TODO: g,h,i
        """
    
        # read MOLDEN.perm for reordering to MOLDEN order
        I=np.arange(M.shape[0])
        idx = []
        fn_perm = self.path + "/MOLDEN.perm"
        check_file(fn_perm)		
        with open(fn_perm) as m:
            for line in m:
                for i in line.split():
                    idx.append(int(i)-1)
        idx_to_MLD = np.array(idx)
        #self.rev_idx = np.argsort( self.idx)
        
        if dim == 2:
            if axis == 2:
                M_reord = M[np.ix_(idx_to_MLD, idx_to_MLD)]
            elif axis == 1:
                M_reord = M[np.ix_(I, idx_to_MLD)]
        else:
            print("WARNING: only 2D reordering is implemented!")
    
        return M_reord
    
    def molden2tm(self,M,dim=2,axis=2):
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
        
        if not hasattr(self, 'basis_order'):
            self.read_molden(read_coeff=False)
        
        # read basis infromation from MOLDEN for reordering to TM order
        if self.basis_order["g"] != []:
           print("WARNING: Reordering of g or higher functions is not yet implemented!!!")
        
        I=np.arange(M.shape[0])    
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

        if dim == 2:
            if axis == 2:
                M_reord = M[np.ix_(idx_to_TM, idx_to_TM)]
            elif axis == 1:
                M_reord = M[np.ix_(I, idx_to_TM)]
        else:
            print("WARNING: only 2D reordering is implemented!")
        
        # correct f-3 definition by swapping sign
        if dim == 2:
            if axis == 2:
                M_reord[f3b,:] = -1*M_reord[f3b,:]
                M_reord[:,f3b] = -1*M_reord[:,f3b]
            elif axis == 1:
                M_reord[:,f3b] = -1*M_reord[:,f3b]
        else:
            print("WARNING: only 2D reordering is implemented!")
        
        return M_reord
    
    def tm2molden(self,M,dim=2,axis=2):
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
            if not hasattr(self, 'basis_order'):
                self.read_molden(read_coeff=False)
            
            # read basis infromation from MOLDEN for reordering to TM order
            if self.basis_order["g"] != []:
               print("WARNING: Reordering of g or higher functions is not yet implemented!!!")
            
            I=np.arange(M.shape[0])    
            idx_to_TM = []
            f3b = []
            di,fi = 0,0
            for i in range(self.nsao):
                if i in self.basis_order["d"]:
                        if di == 3:
                            idx_to_TM.append(i-1)
                            di += 1
                        elif di == 4:
                            idx_to_TM.append(i+1)
                            di = 0
                        else:
                            idx_to_TM.append(i)
                            di += 1
                elif i in self.basis_order["f"]:
                        if fi == 3:
                            idx_to_TM.append(i-1)
                            fi += 1
                        elif fi == 4:
                            idx_to_TM.append(i+1)
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
    
            if dim == 2:
                if axis == 2:
                    M_reord = M[np.ix_(idx_to_TM, idx_to_TM)]
                elif axis == 1:
                    M_reord = M[np.ix_(I, idx_to_TM)]
            else:
                print("WARNING: only 2D reordering is implemented!")
            
            # correct f-3 definition by swapping sign
            if dim == 2:
                if axis == 2:
                    M_reord[f3b,:] = -1*M_reord[f3b,:]
                    M_reord[:,f3b] = -1*M_reord[:,f3b]
                elif axis == 1:
                    M_reord[:,f3b] = -1*M_reord[:,f3b]
            else:
                print("WARNING: only 2D reordering is implemented!")
            
            return M_reord
    
    # TODO: make a MOLDEN class for mrcc and turbomole package!!!
    ###########################################################################
    def read_molden(self,read_coeff=True):
        """
        Read Molecular Orbitals from MOLDEN.
        """
        self.molden_atom, self.molden_mo, self.molden_gto, self.basis_order = read_molden(self.fn_molden,read_coeff=read_coeff)
        
        #self.mo_eigs = np.array(molden_mo["Energy"])
        #self.mos = np.array(molden_mo["Coeff"][0])
        #self.nsao = self.mos.shape[0]       

    ###########################################################################               
         
    def write_molden(self,fn,eig,coeffs):     
        """
        Write Molecular Orbitals to MOLDEN.
        """        
        ao_type = "sao" #hard coded
        
        if not hasattr(self, 'molden_gto'):
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

    def read_M(self,fn,n1,n2=0,sym=True):
        """
        Read a general matrix from file fn.
        """
        check_file(fn)
        data = np.loadtxt(fn)
        row = data[:, 0].astype(int)-1
        col = data[:, 1].astype(int)-1
        if sym:
            M = np.zeros((n1,n1))
            M[row,col] = data[:, 2]
            M[col,row] = data[:, 2]
        else:
            M = np.zeros((n1,n2))
            M[row,col] = data[:, 2]
        return M
    
    def read_mos(self):
        """
        Read LCAO coefficients from file mos_*.dat 
        and the correponding eigenvalues from file moeig_*.dat
        """
        warning = False        						
        mos0 = self.read_M(self.fn_c,self.nsao,self.norb,sym=False)
        self.mos0 = mos0.T #nsao x norb
        self.mos = self.mrcc2tm(self.mos0,axis=1)
        
        self.mo_eigs = np.zeros((self.norb))
        check_file(self.fn_eigs)
        self.mo_eigs = np.loadtxt(self.fn_eigs)						
    
    def read_H(self):
        self.H0 = self.read_M(self.fn_h,self.nsao)
        self.H = self.mrcc2tm(self.H0)
    
    def read_S(self):
        self.S0 = self.read_M(self.fn_ov,self.nsao)
        self.S = self.mrcc2tm(self.S0)
   
    def read_HS(self):
        """
        Read Fock and Overlap matrices
        """
        self.read_H()
        self.read_S()
    
    def read_dos(self,norm=False,typ="IP"):
        
        # TODO: egysegesiteni a ket eset kimenetet
        if self.ccprog == "cis":
            print(f'WARNING: CIS DOs are read in AO basis.')
            dos_R, do_eigs_R = self.read_dos_cis(norm=norm)
            self.dos = dos_R
            self.do_eigs = do_eigs_R
            #dos_L, do_eigs_L = [], []
        elif self.ccprog == "mrcc":
            print(f'WARNING: MRCC DOs are read in MO basis.')
            dos_R, dos_L = self.read_dos_mrcc(norm=norm,typ=typ)
            do_eigs_R, do_eigs_L = self.read_eig_mrcc(typ=typ)
            self.do_eigs = [do_eigs_R,do_eigs_L]
            self.dos = [dos_R,dos_L]
        else:
            print(f'ERROR: ccprog={self.ccprog:} is not known!')
    
    def read_dos_cis(self,norm=False):
        """
        Read dyson orbitals from file MOLDEN_DYSON_IP/EA.X
        """ 
        
        typ=self.keywd[' ip_ea'][:2].upper()
        
        if not hasattr(self, 'S'):
            self.read_S()
        
        l_dos = np.zeros((self.nstate,self.nsao))
        do_eigs = np.zeros((self.nstate))
        
        for st in range(self.nstate):
            fn_molden = self.path + "/MOLDEN_DYSON_" + typ + "." + str(st+1)
            check_file(fn_molden)
            atom, dos, gto, basis_order = read_molden(fn_molden)
            l_dos[st] = dos["Coeff"][0][0]
        #    self.do_eigs[st] = dos["Energy"][0] # it didn't write the energy to the MOLDEN file
        
        #TODO: Fix the part of getting the dyson orbitals
        print("WARNING: Fix the part of getting the dyson orbital energy!")    
        os.system(f'grep "excitation energy" {self.out} > {self.path}/energy.out')
        with open(self.path + "/energy.out","r") as f:
            st = 0
            for line in f:
                if st >= self.nstate:
                    print("WARNING! More states are listed in energy.out then required in MINP.")
                    print(f'         Only the first {self.nstate:2d}th are used.')
                    break
                else:
                    do_eigs[st] = float(line.split()[3])
                    st += 1

        l_dos = self.molden2tm(l_dos,dim=2,axis=1) # csak MOLDEN -> TM kell

        if norm:
            for st in range(self.nstate):
                norm = 1.0 / np.sqrt(l_dos[st] @ self.S @ l_dos[st])
                l_dos[st] = norm * l_dos[st]
            
        #self.dos = l_dos
        return l_dos, do_eigs
    
    def read_tden_mrcc(self,side="L",add_path=""):
        """
        Read CC transition densities from MRCC tden_X.dat file
        """
                
        # if the tdens are not in the main dir
        if add_path != "":
            path = self.path + "/" + str(add_path)
        else:
            path = self.path
        
        if side[0].upper() == "L":
            fn = path + "/tden_L.dat"
        elif side[0].upper() == "R":
            fn = path + "/tden_R.dat"
        else:
            print("ERROR: write valid side...")
        
        # in embed, tden is on the HO basis of M only!
        if self.embed:
            n = self.norb_a
            nc = self.ncore - self.nocc_b
            no = self.nocc_a
            nv = self.nvir_a
            
        else:
            n = self.nsao
            nc = self.ncore
            no = self.nocc
            nv = self.nvir
        
        shapes = [(nv,nv),
                  (no-nc,no-nc),
                  (no-nc,nv), 
                  (nv,no-nc)]
        
        orders = ['C','F','C','F']
        
        tden_a_bls = []
        tden_b_bls = [] # TODO: implement beta density
        
        # if MRCC (out-of-core algorithm) run instead of CC, 
        # then the blocks are further diveded into subblocks
        tden_a_sub_bls = []

        with open(fn, "r") as f:
            tmp_a = []
            ib = 0
            sub = 0
            j = 0
            k = 0
            for line in f:
                
                # if the number was larger than excepted by the format 
                # used in the density writer script in MRCC
                # only for alpha density !
                if "*" in line or len(line.split()) < 4:
                    if len(line.split()) == 2:
                        coeff = float(line.split()[1])        
                        i = int(line[:4])
                        #j = line[4:8]
                        #k = line[8:12]         
                    elif len(line.split()) == 3:
                        #k = line.split()[1]
                        coeff = float(line.split()[2])
                        i = int(line[:4])
                        #j = line[4:8]
                    else:
                        print(line.split())
                        exit()
                                
                    i = int(i)
                    j += 1
                    
                    if i == 1:
                        tmp_a.append(coeff)
                        k = -1
                        
                    elif i != 1:
                        if k == -1:
                            # check if the block size is match with the real block size
                            if j == shapes[ib][0]*shapes[ib][1]:
                                # version before 25.1.0
                                #if EM.embed and ib in [1,2,3,6,8]:
                                #    # these blocks contains the enviroment
                                #    tmp_a = []
                                #    ib += 1
                                #else:
                                tmp_a = np.array(tmp_a).reshape(shapes[ib],order=orders[ib])
                                tden_a_bls.append(tmp_a)
                                tmp_a = []       
                                ib += 1
                                j, k = 0, 0
                            else:
                                if ib == 0:
                                    if (nv%2) != 0:
                                        d1 = int((nv+nv%2)//2)
                                        d2 = int(nv//2)
                                    else:
                                        d1 = int(nv//2)
                                        d2 = int(nv//2)

                                    if sub == 0:
                                        dim = (d1,d1)
                                        sub += 1
                                    elif sub == 1:
                                        dim = (d2,d1)
                                        sub += 1
                                    elif sub == 2:
                                        dim = (d1,d2)
                                        sub += 1
                                    elif sub == 3:
                                        dim = (d2,d2)
                                        sub = 0

                                elif ib == 1:
                                    print("WARNING: too large occupied space!")

                                elif ib == 2:
                                    if sub == 0:
                                        d1 = int(len(tmp_a)/(no-nc))
                                        dim = (no-nc,d1)
                                        sub += 1
                                    else:
                                        d2 = int(len(tmp_a)/(no-nc))
                                        dim = (no-nc,d2)
                                        sub = 0

                                elif ib == 3:
                                    if sub == 0:
                                        d1 = int(len(tmp_a)/(no-nc))
                                        dim = (d1,no-nc)
                                        sub += 1
                                    else:
                                        d2 = int(len(tmp_a)/(no-nc))
                                        dim = (d2,no-nc)
                                        sub = 0

                                tmp_a = np.array(tmp_a).reshape(dim,order=orders[ib])
                                tden_a_sub_bls.append(tmp_a)

                                if ib == 0 and sub == 0:
                                    M = np.zeros(shapes[ib])
                                    M[:d1,:d1] = tden_a_sub_bls[0]
                                    M[d1:nv,:d1] = tden_a_sub_bls[1]
                                    M[:d1,d1:nv] = tden_a_sub_bls[2]
                                    M[d1:nv,d1:nv] = tden_a_sub_bls[3]
                                    tden_a_sub_bls = []
                                    tden_a_bls.append(M)

                                elif ib == 2 and sub == 0:
                                    M = np.zeros(shapes[ib])
                                    M[:no-nc,:d1] = tden_a_sub_bls[0]
                                    M[:no-nc,d1:nv] = tden_a_sub_bls[1]
                                    tden_a_sub_bls = []
                                    tden_a_bls.append(M)

                                elif ib == 3 and sub == 0:
                                    M = np.zeros(shapes[ib])
                                    M[:d1,:no-nc] = tden_a_sub_bls[0]
                                    M[d1:nv,:no-nc] = tden_a_sub_bls[1]
                                    tden_a_sub_bls = []
                                    tden_a_bls.append(M)

                                tmp_a = []
                                if sub == 0:
                                    ib += 1     
                                j, k = 0, 0
                        else:
                            j = 0                              
                else:
                    i, j, k, coeff = map(float, line.split())
                    i, j, k = map(int, (i, j, k))
                                
                    if i == 1:
                        tmp_a.append(coeff)
                        
                        if j == k:
                            # check if the block size is match with the real block size
                            if j == shapes[ib][0]*shapes[ib][1]:
                                # version before 25.1.0
                                #if EM.embed and ib in [1,2,3,6,8]:
                                #    # these blocks contains the enviroment
                                #    tmp_a = []
                                #    ib += 1
                                #else:
                                tmp_a = np.array(tmp_a).reshape(shapes[ib],order=orders[ib])
                                tden_a_bls.append(tmp_a)
                                tmp_a = []       
                                ib += 1
                                j, k = 0, 0
                            else:
                                if ib == 0:
                                    if (nv%2) != 0:
                                        d1 = int((nv+nv%2)//2)
                                        d2 = int(nv//2)
                                    else:
                                        d1 = int(nv//2)
                                        d2 = int(nv//2)

                                    if sub == 0:
                                        dim = (d1,d1)
                                        sub += 1
                                    elif sub == 1:
                                        dim = (d2,d1)
                                        sub += 1
                                    elif sub == 2:
                                        dim = (d1,d2)
                                        sub += 1
                                    elif sub == 3:
                                        dim = (d2,d2)
                                        sub = 0

                                elif ib == 1:
                                    print("WARNING: too large occupied space!")

                                elif ib == 2:
                                    if sub == 0:
                                        d1 = int(len(tmp_a)/(no-nc))
                                        dim = (no-nc,d1)
                                        sub += 1
                                    else:
                                        d2 = int(len(tmp_a)/(no-nc))
                                        dim = (no-nc,d2)
                                        sub = 0

                                elif ib == 3:
                                    if sub == 0:
                                        d1 = int(len(tmp_a)/(no-nc))
                                        dim = (d1,no-nc)
                                        sub += 1
                                    else:
                                        d2 = int(len(tmp_a)/(no-nc))
                                        dim = (d2,no-nc)
                                        sub = 0

                                tmp_a = np.array(tmp_a).reshape(dim,order=orders[ib])
                                tden_a_sub_bls.append(tmp_a)

                                if ib == 0 and sub == 0:
                                    M = np.zeros(shapes[ib])
                                    M[:d1,:d1] = tden_a_sub_bls[0]
                                    M[d1:nv,:d1] = tden_a_sub_bls[1]
                                    M[:d1,d1:nv] = tden_a_sub_bls[2]
                                    M[d1:nv,d1:nv] = tden_a_sub_bls[3]
                                    tden_a_sub_bls = []
                                    tden_a_bls.append(M)

                                elif ib == 2 and sub == 0:
                                    M = np.zeros(shapes[ib])
                                    M[:no-nc,:d1] = tden_a_sub_bls[0]
                                    M[:no-nc,d1:nv] = tden_a_sub_bls[1]
                                    tden_a_sub_bls = []
                                    tden_a_bls.append(M)

                                elif ib == 3 and sub == 0:
                                    M = np.zeros(shapes[ib])
                                    M[:d1,:no-nc] = tden_a_sub_bls[0]
                                    M[d1:nv,:no-nc] = tden_a_sub_bls[1]
                                    tden_a_sub_bls = []
                                    tden_a_bls.append(M)

                                tmp_a = []
                                if sub == 0:
                                    ib += 1     
                                j, k = 0, 0
                    elif i != 1:
                        tden_b_bls.append(coeff)
                        j, k = 0, 0
                        
        
        # Constructing T_a matrix
        tden_a = np.zeros((n,n))
        tden_a[nc:no,nc:no] = tden_a_bls[1]
        tden_a[no:no+nv,no:no+nv]     = tden_a_bls[0]
        tden_a[nc:no,no:no+nv]   = tden_a_bls[2]
        tden_a[no:no+nv,nc:no]   = tden_a_bls[3]

        return tden_a
    
    def read_eig_mrcc(self,typ="IP"):
        do_eigs_L = np.zeros((self.nstate))
        do_eigs_R = np.zeros((self.nstate))
        
        i = 0
        side = 0
        for st in range(self.nstate):
            fn = self.path + "/root" + str(st+1) + "/mrcc.out"
            with open(fn,"r") as f:
                for line in f:
                    if "Total CCSD energy" in line:
                        E0 = float(line.split()[4])
                        if i == 0:
                            E0_ref = E0
                        else:
                            if E0_ref - E0 > 1e-8:
                                print(f'ERROR: E0 of root{(st+1):} is differ from the root1.')
                                break
                        i += 1
                    elif "Total LR-CCSD energy" in line:
                        E2 = float(line.split()[4])
                        if typ.upper() == "IP":
                            DE = E0 - E2
                        elif typ.upper() == "EA":
                            DE = E2 - E0
                        else:
                            print(f'ERROR: {typ:} is invalide typ!')
                        if side == 0:
                            do_eigs_L[st] = DE
                            side = 1
                        elif side == 1:
                            do_eigs_R[st] = DE
                            side = 2
                        elif side == 2:
                            side = 0 
            
        return do_eigs_R, do_eigs_L 
        
    def read_dos_mrcc(self,norm=False,typ="IP"):#,mean=True):
        """
        Create Dyson Orbitals from MRCC transition densities
        """ 
        
        DOs_L = np.zeros((self.nstate,self.norb_a))
        DOs_R = np.zeros((self.nstate,self.norb_a))
        
        for st in range(self.nstate):
            trd_L = self.read_tden_mrcc(side="L",add_path=f'root{st+1:d}')
            trd_R = self.read_tden_mrcc(side="R",add_path=f'root{st+1:d}')
        
            if typ == "IP":
                if self.embed:
                    ncont = self.nocc_a
                else:
                    ncont = self.nocc        
            elif typ == "EA":
                if self.embed:
                    ncont = self.nocc_a-1 # TODO: why -1 is necessary??
                else:
                    ncont = self.nocc-1

            print(st," ",np.max(abs(trd_L))," ",np.max(abs(trd_R)))

            if typ == "IP":
                dos_L = trd_L[ncont,:]                
                dos_R = trd_R[:,ncont]
            elif typ == "EA":
                dos_L = trd_L[:,ncont]
                dos_R = trd_R[ncont,:]
            #if norm:
            #    norm = 1.0 / np.sqrt(dos_L @ self.S @ dos_L)
            #    dos_L = norm * dos_L
            #    norm = 1.0 / np.sqrt(dos_R @ self.S @ dos_R)
            #    dos_R = norm * dos_R
                
            if norm:
                #norm = 1.0 / np.sqrt(dos_L @ dos_R)
                norm = 1.0 / np.sqrt(dos_L @ dos_L)
                dos_L = norm * dos_L
                norm = 1.0 / np.sqrt(dos_R @ dos_R)
                dos_R = norm * dos_R
            
            DOs_L[st] = dos_L
            DOs_R[st] = dos_R
        
        #if mean:
        #    
        #    
        #    
        #    return 0
        #else:
        return DOs_R, DOs_L
        
        ## CFOUR notation : <Phi_0||\Phi_exc>
        #right_tden_fn = self.path + "/CCDENSITIES.1" 
        #right_tden = np.zeros((n,n))
        #
        #with open(right_tden_fn) as f:
        #    for line in f:
        #        coeff,i,j,k,l = line.split()
        #        i,j,k,l = int(i),int(j),int(k),int(l)
        #        coeff = float(coeff)
        #        if k == 0 and l == 0:
        #            right_tden[i,j] = coeff
        #
        ## CFOUR notation : <Phi_exc||\Phi_0>
        #left_tden_fn = self.path + "/CCDENSITIES.0" 
        #left_tden = np.zeros((n,n))        
        #
        #with open(left_tden_fn) as f:
        #    for line in f:
        #        coeff,i,j,k,l = line.split()
        #        i,j,k,l = int(i),int(j),int(k),int(l)
        #        coeff = float(coeff)
        #        if k == 0 and l == 0:
        #            left_tden[i,j] = coeff
        #
        #return right_tden, left_tden
        
    #def read_trds(self):
    #    """
    #    Read one-electron transition densities from file trd.dat.
    #    File trd.dat contain all of them in a row.
    #    """
    #    self.trds = np.zeros((self.nstate,self.nsao,self.nsao))
    #    with open("trd.dat","r") as tnf:
    #        k = 0
    #        for line in tnf:
    #            line_s = line.split()
    #            i,j,coeff = int(line_s[0])-1,int(line_s[1])-1,float(line_s[2])
    #            self.trds[k,i,j] = coeff
    #            if i == self.nsao-1 and j == self.nsao-1:
    #                k += 1
#
    #    self.mrcc2tm(self.trds)
    #
    #def create_dos(self,exc_typ='ip',norm=False):
    #    
    #    # search continuum orbital
    #    self.ncont_mo = np.where(abs(self.mo_eigs) < 1e-8)[0] # continuum MO
    #    if self.ncont_mo is None:
    #        print("ERROR: Continuum orbital is not found! (There is no orbital with energy less then 1E-6.)")
    #    elif len(self.ncont_mo) > 1:
    #        print("WARNING: More then one orbital found with energy below 1E-8!")
    #        exit()
    #    self.ncont_ao = np.where(self.mos[self.ncont_mo[0]] == 1e+00)[0][0]
    #    
    #    self.dos = np.zeros((self.nstate,self.nsao))
    #    for st in range(self.nstate):
    #        if exc_typ == 'ip':
    #            do = self.trds[st,:,self.ncont_ao]
    #        elif exc_typ == "ea":
    #            do = self.trds[st,self.ncont_ao,:]
    #        else:
    #            print(f'{exc_typ:} is wrong type!')
    #    
    #        if norm:
    #            norm = 1.0 / np.sqrt(do @ self.S @ do)     
    #            do = norm * do
    #        
    #        self.dos[st] = do

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
        alpha, beta, gamma = rot.as_euler('zyz', degrees=False)

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


###############################################################################
# %% Huzinaga class
###############################################################################
class Huzinaga:
    
    def __init__(self, path=os.getcwd()):
        """
        Initialise the MRCC class variables. 
            - 'a' always denotes the active system 
            - 'b' always denotes the enverioment 
        """
        # Instance parameters 
        self.path = path            # Path
        if not os.path.isdir(self.path):
            print("ERROR: {0:} folder does not exist!".format(self.path))        

        # read necessary dimensions from MRCC output files
        self.read_low_H() # version is also determined in this call
        self.read_high_H()

        self.get_emb_idxs()
        
    # ---------------------------------------------------------------------------
    # %% Functions for read input data           
    def read_low_H(self):
        """
        Read low-level Fock matrix + canonical and localised orbitals
        """
        # F (fock_em1.dat) -> 1. SCF utáni teljes rendszer, low-level Fock operátor
        # LCAO (mos_em1.dat) -> kanonikus MO-k
        self.em1 = MRCC(path=self.path,sub="em1")  
        self.em1.read_HS()         
        
        # copy dimensions and overlap
        self.S         = self.em1.S.copy()
        self.natrange  = self.em1.natrange.copy()
        self.nocc_b    = self.em1.nocc_b
        self.nocc_a    = self.em1.nocc_a
        self.nocc      = self.em1.nocc
        self.nvir_b    = self.em1.nvir_b
        self.nvir_a    = self.em1.nvir_a
        self.nsao      = self.em1.nsao
        self.norb      = self.em1.norb
        self.natoms    = self.em1.natoms
        
        # read version (necessary to determine the order of the Huzinaga orbitals!)
        self.em1.read_version()
        self.version = self.em1.version
        
        # F (fock_em2.dat) -> aktiv rendszer lokalizált pályákkal
        # LCAO (mos_em2.dat) -> SPADE lokalizált occ. tér + kanonikus virt. pályák "maradéka" + az aktiv részhez tartozó PAO-k
        # a virt. rész aljára kerülnek a PAO-k a többi rész változatlan marad -> így a virt. tér csak az aktiv részre lesz értelmes
        self.em2 = MRCC(path=self.path,sub="em2") 
        self.em2.read_HS() 
        
    def read_Huzinaga(self):
        """
        Read Huzianaga matrix from MRCC output
        """
        # F (huzi_em3.dat) -> Huzinaga operátor 
        # LCAO (mos_em3.dat) -> Huzinaga operátor saját vektorai (lokalizált occ. és virt. pályák az aktivrészre és a környezetre is)
        self.em3_huzi = MRCC(path=self.path,sub="em3_huzi")  
        self.em3_huzi.read_HS()         

    def read_high_H(self,create_mos=False):
        """
        Read high-level Fock matrix (with embedding potential)
        """
        # F (fock_em3.dat) -> 2. SCF F operátora, a magas szinten (teljes rendszerre!)
        # LCAO (mos_em3.dat) -> Nem a hozátartozó saját vektorok, hanem a Huzinaga operátor saját vektorai (lokalizált occ. és virt. pályák az aktivrészre és a környezetre is)
        self.em3 = MRCC(path=self.path,sub="em3") 
        self.em3.read_HS()
        
        if create_mos:
            mo_eigs,mos = linalg.eigh(self.em3.H,self.em3.S)
            self.em3.mo_eigs = mo_eigs
            self.em3.mos = mos.T
            
    def get_emb_idxs(self):
        """
        Create AO index list for active system and enviroment
        """
        # atom and AO index lists 
        self.get_emb_inf_from_minp()
        
        self.iaos_a = []        
        for iatom in self.iatoms_a:
            for iao in range(self.natrange[0,iatom],self.natrange[1,iatom]):
                self.iaos_a.append(iao)
        
        self.iaos_b = []        
        for iatom in self.iatoms_b:
            for iao in range(self.natrange[0,iatom],self.natrange[1,iatom]):
                self.iaos_b.append(iao)
        
        self.nsao_a = len(self.iaos_a)    # number of SAO of the enviroment
        self.nsao_b = len(self.iaos_b)    # number of SAO of the active system    
    
    def get_emb_inf_from_minp(self):
        """
        Read embedding infromation from input file MINP
        """

        fn = self.path + "/MINP"
        check_file(fn)
        with open(fn,"r") as minp:
            read = False
            a_idx = []
            b_idx = np.arange(self.natoms)
            for line in minp:
                line = line.split("!")[0].strip()
                if "embed" in line and "=" in line:
                    read = True
                elif read:
                    idx = ''
                    l = False
                    for c in line:
                        if c == ",":
                            if l:
                                eidx = int(idx)
                                for i in range(sidx,eidx+1):
                                    a_idx.append(i-1)
                                l = False
                            else:
                                a_idx.append(int(idx)-1)
                            idx = ''
                        elif c == "-":
                            sidx = int(idx)
                            idx = ''
                            l = True
                        else:
                            idx += c
                    if l:
                        eidx = int(idx)
                        for i in range(sidx,eidx+1):
                            a_idx.append(i-1)
                    else:
                        a_idx.append(int(idx)-1)
                    break
        b_idx = [item for item in b_idx if item not in a_idx]
        self.iatoms_a = a_idx
        self.iatoms_b = b_idx
        
    # ---------------------------------------------------------------------------
    # %% Functions for creating new stuff
    def build_proj(self):
        """
        Build the projectors for the enviroment
        
        Orbital order (in the version 2):
         - enviroment occupied / active system occupied / active system virtual / enviroment virtual
        """
        if not hasattr(self, 'nocc'):
            self.get_orb_dims()
        
        # projector of occupied subspace of the enviroment    
        self.pb_o = self.em2.mos[:self.nocc_b].T @ self.em2.mos[:self.nocc_b]

        # projector of occupied subspace of the active system    
        self.pa_o = self.em2.mos[self.nocc_b:self.nocc].T @ self.em2.mos[self.nocc_b:self.nocc]

        # projector of occupied space of the supersystem    
        self.p_o = self.em2.mos[:self.nocc].T @ self.em2.mos[:self.nocc]
        
        # the unit matrix of the space of the supersystem 
        I = self.em1.mos.T @ self.em1.mos
        
        # projector of virtual subspace of the active system
        if self.version == 1: # old order: enviroment occupied / active system occupied / enviroment virtual /  active system virtual
            self.pa_v = self.em2.mos[self.nocc+self.nvir_b:].T @ self.em2.mos[self.nocc+self.nvir_b:]        
        elif self.version == 2:
            self.pa_v = self.em2.mos[self.nocc:self.nocc+self.nvir_a].T @ self.em2.mos[self.nocc:self.nocc+self.nvir_a]

        # projector of virtual subspace of the enviroment           
        self.pb_v = I - self.p_o - self.pa_v
        
    def read_proj(self):
        """
        Read the projectors for the enviroment
        """
        if not hasattr(self, 'nocc'):
            self.get_orb_dims()
        
        if not hasattr(self, 'em3'):
            self.read_high_H()
        
        pb_o_s0 = np.zeros((self.nsao,self.nsao))
        check_file(self.path+"/rso_em3.dat")						
        with open(self.path+"/rso_em3.dat",'r') as fn:
            for line in fn:
                sp = line.split()
                i,j,coeff = int(sp[0])-1,int(sp[1])-1,float(sp[2])
                pb_o_s0[i,j] = coeff
        
        self.pb_o_s = self.em3.mrcc2tm(pb_o_s0)
        
        pb_v_s0 = np.zeros((self.nsao,self.nsao))
        check_file(self.path+"/rsv_em3.dat")						
        with open(self.path+"/rsv_em3.dat",'r') as fn:
            for line in fn:
                sp = line.split()
                i,j,coeff = int(sp[0])-1,int(sp[1])-1,float(sp[2])
                pb_v_s0[i,j] = coeff
        
        self.pb_v_s = self.em3.mrcc2tm(pb_v_s0)
        
    def build_Huzinaga(self,l="low",terms=["o","v"]):
        """
        Build Huzinaga matrix for active system
        
         - ordering in subsystem localised orbital basis: 
           enviroment occupied / active system occupied / active system virtual / enviroment virtual 
        
        Input:
            l - string  - level: which Fock matrix are used to build the Huzinaga matrix
                        - low: 1. SCF (em1) Fock matrix (default)
                        - high: 3. SCF (em3) Fock matix
            terms - list of terms
        Ouptut:
            Hh - Huzianga matrix
            mos - subsystem localised orbitals 
            eigs - corresponding eigenvaules
            F_loc - low-level Fock matrix in the subsystem localised orbital basis
        """        
        # read low-level Fock matrix + canonical and localised orbitals
        if not hasattr(self, 'em1'):
            self.read_low_H()
        
        # build the Huzinaga projectors for the enviroment
        if not hasattr(self, 'pb'):
            self.build_proj()

        # "low-level" Huzinaga operátor
        if l.lower() == "low":
            F = self.em1.H.copy()
        elif l.lower() == "high":
            if not hasattr(self, 'em3'):
                self.read_high_H()
            F = self.em3.H.copy()
        else:
            print(f'{l:} is not a valid level for building Huzinaga matrix. Choose "low" or "high".')
            sys.exit()
            
        self.Hu = F.copy()
        if "o" in terms:
            self.Hu = self.Hu - F @ self.pb_o @ self.S - self.S @ self.pb_o @ F
        if "v" in terms:
            self.Hu = self.Hu - F @ self.pb_v @ self.S - self.S @ self.pb_v @ F 
        if "oo" in terms:            
            self.Hu = self.Hu + (self.S @ self.pb_o @ F @ self.pb_o @ self.S)  # "fix" the energy of occ. orbs of env.
        if "vv" in terms:
            self.Hu = self.Hu + (self.S @ self.pb_v @ F @ self.pb_v @ self.S)  # "fix" the energy of virt. orbs of env.
        if "2oo" in terms:            
            self.Hu = self.Hu + 2*(self.S @ self.pb_o @ F @ self.pb_o @ self.S)  # "fix" the energy of occ. orbs of env.
        if "2vv" in terms:
            self.Hu = self.Hu + 2*(self.S @ self.pb_v @ F @ self.pb_v @ self.S)  # "fix" the energy of virt. orbs of env.
        if "ov" in terms or "vo" in terms:
            self.Hu = self.Hu + (self.S @ self.pb_o @ F @ self.pb_v @ self.S) + (self.S @ self.pb_v @ F @ self.pb_o @ self.S) # cross terms
        
        #self.Hh = self.em1.H - self.S @ self.pb @ self.em1.H - self.em1.H @ self.pb @ self.S 
        # correct signs of enviroment's localised orbital energies
        #self.Hh += 2*(self.S @ self.pb @ self.em1.H @ self.pb @ self.S)

        e,v = linalg.eigh(self.Hu,self.S)
        mos = v.T

        # order localised orbitals based on population
        # calculate the overlap with the Huzianaga projectors (for the enviroment)
        # overlap is 1 for enviroment orbitals and 0 for active system
        w = self.S @ self.pb_o @ self.S @ mos.T
        idx_o = []
        for i in range(self.norb):
            ov = -1*abs(w.T[i] @ mos[i])
            idx_o.append(np.round(ov,1))
        idx_o = np.argsort(idx_o,kind='stable')

        mos = mos[np.ix_(idx_o,np.arange(self.nsao))]
        e = e[np.ix_(idx_o)]

        w = self.S @ self.pa_o @ self.S @ mos.T
        idx_o = []
        for i in range(self.nocc_b,self.nsao):
            ov = -1*abs(w.T[i] @ mos[i])
            idx_o.append(np.round(ov,1))
        idx_ob = np.arange(self.nocc_b)
        idx_o = np.argsort(idx_o,kind='stable')+self.nocc_b
        idx = np.concatenate((idx_ob,idx_o),axis=0)

        mos = mos[np.ix_(idx,np.arange(self.nsao))]
        e = e[np.ix_(idx)]

        if self.version == 1: # old order: enviroment occupied / active system occupied / enviroment virtual /  active system virtual
            w = self.S @ self.pb_v @ self.S @ mos.T
        elif self.version == 2:
            w = self.S @ self.pa_v @ self.S @ mos.T
        idx_o = []
        for i in range(self.nocc,self.nsao):
            ov = -1*abs(w.T[i] @ mos[i])
            idx_o.append(np.round(ov,1))
        idx_ob = np.arange(self.nocc)
        idx_o = np.argsort(idx_o,kind='stable')+self.nocc
        idx = np.concatenate((idx_ob,idx_o),axis=0)

        mos = mos[np.ix_(idx,np.arange(self.nsao))]
        e = e[np.ix_(idx)]

        if self.version == 1: 
            n = [0,self.nocc_b,self.nocc,self.nocc+self.nvir_b]
            m = [self.nocc_b,self.nocc,self.nocc+self.nvir_b,self.nsao]
        elif self.version == 2:
            n = [0,self.nocc_b,self.nocc,self.nocc+self.nvir_a]
            m = [self.nocc_b,self.nocc,self.nocc+self.nvir_a,self.nsao]
            
        for i in range(4):
            if i == 0:
                idx = np.argsort(-1*e[n[i]:m[i]])                
            elif i == 2 and self.version == 1:
                idx = np.argsort(-1*e[n[i]:m[i]])
            elif i == 3 and self.version == 2:
                idx = np.argsort(-1*e[n[i]:m[i]])
            else:
                idx = np.argsort(e[n[i]:m[i]])
            mos[n[i]:m[i]] = mos[n[i]:m[i]][np.ix_(idx,np.arange(self.nsao))]
            e[n[i]:m[i]] = e[n[i]:m[i]][np.ix_(idx)]

        self.mos_loc = mos
        self.eigs_loc = e    
                  
        self.F_loc = self.mos_loc @ self.em1.H @ self.mos_loc.T
                
        # check the structer of the Fock matrix
        for i in range(4):
            s = abs(np.max(self.F_loc[n[i]:m[i],n[i]:m[i]] - np.diag(np.diag(self.F_loc[n[i]:m[i],n[i]:m[i]])))) #/self.nsao
            if s > 1e-10:
                print(f'WARNING: Fock matrix in localised orbital basis have non-zero off-diagonal elements! ({s: .5e})')
        
        #return Hu, mos, e
        
        
        #if not hasattr(self,'iaos_a'):
        #    self.get_emb_idxs()
        #    
        #ao, av = [], []
        #bo, bv = [], []
        #for i in range(self.nsao):
        #    ma = mos[i][np.ix_(self.iaos_a)] @ self.S[np.ix_(self.iaos_a,self.iaos_a)] @ mos[i][np.ix_(self.iaos_a)].T
        #    mb = mos[i][np.ix_(self.iaos_b)] @ self.S[np.ix_(self.iaos_b,self.iaos_b)] @ mos[i][np.ix_(self.iaos_b)].T
        #    print(f' {i:}  {ma: 3.5f}  {mb: 3.5f}')
        #    #if ma < 0.8 and mb < 0.8:
        #    #    sys.exit(f'ERROR: population on A or B is less then 0.9 ({i:}: {ma: 2.5f}, {mb: 2.5f})') 
        #    if i < self.nocc:
        #        if ma > mb:
        #            ao.append(i)
        #        else:
        #            bo.append(i)
        #    else:
        #        if ma > mb:
        #            av.append(i)
        #        else:
        #            bv.append(i)
        #idx = bo + ao + bv + av
    #

    #def build_hproj_3(self):
    #    """
    #    Build the Huzinaga projectors for the enviroment
    #    """
    #    if not hasattr(self, 'nocc'):
    #        self.get_orb_dims()
    #        
    #    # környezet betöltött terének projektora
    #    self.pb_o_3 = self.em3_huzi.mos[:self.nocc_b].T @ self.em3_huzi.mos[:self.nocc_b]
    #    # környezet virtuális tér projektora
    #    self.pb_v_3 = self.em3_huzi.mos[self.nocc:self.nocc+self.nvir_b].T @ self.em3_huzi.mos[self.nocc:self.nocc+self.nvir_b]
    #    # környezet teljes terének projektora
    #    self.pb_3 = self.pb_o_3 + self.pb_v_3
    #
    #def build_Hh_3(self):
    #    """
    #    Build 'high-level' Huzinaga operator for active system
    #    
    #     - ordering in subsystem localised orbital basis: 
    #       enviroment occupied / active system occupied / enviroment virtual / active system virtual
    #               
    #    Ouptut:
    #        Hh - 'high-level' Huzianga operator
    #        mos - subsystem localised orbitals 
    #        eigs - corresponding eigenvaules
    #        F_loc - low-level Fock matrix in the subsystem localised orbital basis
    #    """        
    #    # read high-level Fock matrix + canonical and localised orbitals
    #    if not hasattr(self, 'em3'):
    #        self.read_high_H()
    #    
    #    # build the Huzinaga projectors for the enviroment
    #    if not hasattr(self, 'pb_3'):
    #        self.build_hproj_3()
#
    #    # "low-level" Huzinaga operátor
    #    self.Hh_3 = self.em3.H - self.S @ self.pb_3 @ self.em3.H - self.em3.H @ self.pb_3 @ self.S 
    #    # correct signs of enviroment's localised orbital energies
    #    self.Hh_3 += 2*(self.S @ self.pb_3 @ self.em3.H @ self.pb_3 @ self.S)
#
    #    eigs,v = linalg.eigh(self.Hh_3,self.S)
    #    mos = v.T
#
    #    # order localised orbitals based on population
    #    # calculate the overlap with the Huzianaga projectors (for the enviroment)
    #    # overlap is 1 for enviroment orbitals and 0 for active system
    #    # order orbitals: enviroment occupied / active system occupied / enviroment virtual / active system virtual
    #    w = self.S @ self.pb_o_3 @ self.S @ mos.T
    #    idx_o = []
    #    for i in range(self.nocc):
    #        ov = -1*(w.T[i] @ mos[i])
    #        idx_o.append(np.round(ov,1))
    #    idx_o = np.argsort(idx_o,kind='stable')
    #    
    #    w = self.S @ self.pb_v_3 @ self.S @ mos.T
    #    idx_v = []
    #    for i in range(self.nocc,self.nsao):
    #        ov = -1*(w.T[i] @ mos[i])
    #        idx_v.append(np.round(ov,1))       
    #    idx_v = np.argsort(idx_v,kind='stable')+self.nocc
    #
    #    idx = np.concatenate((idx_o,idx_v),axis=0)
    #    
    #    self.mos_loc_3 = mos[np.ix_(idx,np.arange(self.nsao))]
    #    self.eigs_loc_3 = eigs[np.ix_(idx)]        
    #    self.F_loc_3 = self.mos_loc @ self.em3.H @ self.mos_loc.T
    #    
    #    # check the structer of the Fock matrix
    #    N = [0,self.nocc_b,self.nocc,self.nocc+self.nvir_b]
    #    M = [self.nocc_b,self.nocc,self.nocc+self.nvir_b,self.nsao]
    #    for i in range(4):
    #        n = N[i]
    #        m = M[i]
    #        s = abs(np.max(self.F_loc_3[n:m,n:m] - np.diag(np.diag(self.F_loc_3[n:m,n:m])))) #/self.nsao
    #        if s > 1e-10:
    #            print(f'WARNING: Fock matrix in localised orbital basis have non-zero off-diagonal elements! ({s: .5e})')
