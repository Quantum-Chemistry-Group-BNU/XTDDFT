from pyscf import gto, dft, scf, ao2mo, lib, tddft,symm
from pyscf.scf.uhf import spin_square
from pyscf.dft.numint import _dot_ao_ao_sparse,_scale_ao_sparse,_tau_dot_sparse
from pyscf.dft.gen_grid import NBINS
import numpy as np
import time
from pyscf.dft import numint
import scipy
from functools import reduce
from pyscf.symm import direct_prod
#import sys
#sys.path.append('/home/lenovo2/usrs/zhw/TDDFT')

#from line_profiler import profile

from .SF_TDA import SF_TDA, mf_info,gen_response_sf,gen_response_sf_mc
au2ev = 27.21138505 # 27.2113834 in whb's untils

def get_irrep_occupancy_directly(mol, mf):
    results = {}
    
    if len(mf.mo_occ)==1:  # RKS/ROKS
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        orb_symm = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
        
        #results['total'] = _calculate_irrep_occupancy(orb_symm, mo_occ)
        
    else:  # UKS
        mo_coeff_alpha, mo_coeff_beta = mf.mo_coeff
        mo_occ_alpha, mo_occ_beta = mf.mo_occ
        
        orb_symm_alpha = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff_alpha)
        orb_symm_beta = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff_beta)
        
        results['alpha'] = _calculate_irrep_occupancy(orb_symm_alpha, mo_occ_alpha)
        results['beta'] = _calculate_irrep_occupancy(orb_symm_beta, mo_occ_beta)
    res = {}
    alpha = results['alpha']
    beta = results['beta']
    for name in mol.irrep_name: # like mf.analyze()
        if name in alpha:
            tmp_a = alpha[name]
        else:
            tmp_a = 0
        if name in beta:
            tmp_b = beta[name]
        else:
            tmp_b = 0
        res[name] = (int(tmp_a),int(tmp_b))
    return res

def _calculate_irrep_occupancy(orb_symm, mo_occ):
    """计算指定轨道的不可约表示占据"""
    occupancy = {}
    for sym, occ in zip(orb_symm, mo_occ):
        if occ > 0:  # 只考虑占据轨道
            if sym not in occupancy:
                occupancy[sym] = 0.0
            occupancy[sym] += occ
    return occupancy

def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return np.einsum('z,zr->r', charges, coords)/charges.sum()
    

class SA_SF_TDA():
    def __init__(self,mf,SA=3,davidson=True,method=0):
        """SA=0: SF-TDA
           SA=1: only add diagonal block for dA
           SA=2: add all dA except for OO block
           SA=3: full dA
        """
        print('method=0 (default) ALDA0, method=1 multicollinear, method=2 collinear')
        if np.array(mf.mo_coeff).ndim==3:# UKS
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = mf.mo_occ
            self.type_u = True
            self.SA = 0
        else: # ROKS
            self.mo_energy = np.array([mf.mo_energy, mf.mo_energy])
            self.mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            self.mo_occ = np.zeros((2,len(mf.mo_coeff)))
            self.mo_occ[0][np.where(mf.mo_occ>=1)[0]]=1
            self.mo_occ[1][np.where(mf.mo_occ>=2)[0]]=1
            self.type_u = False
            self.SA = SA

        self.mol = mf.mol
        self.nao = self.mol.nao_nr()
        self.mf = mf
        self.davidson=davidson
        self.method = method
        _,dsp1 = mf.spin_square()
        self.ground_s = (dsp1-1)/2
        
        self.occidx_a = np.where(self.mo_occ[0]==1)[0]
        self.viridx_a = np.where(self.mo_occ[0]==0)[0]
        self.occidx_b = np.where(self.mo_occ[1]==1)[0]
        self.viridx_b = np.where(self.mo_occ[1]==0)[0]
        self.orbo_a = self.mo_coeff[0][:,self.occidx_a]
        self.orbv_a = self.mo_coeff[0][:,self.viridx_a]
        self.orbo_b = self.mo_coeff[1][:,self.occidx_b]
        self.orbv_b = self.mo_coeff[1][:,self.viridx_b]
        self.nocc_a = self.orbo_a.shape[1]
        self.nvir_a = self.orbv_a.shape[1]
        self.nocc_b = self.orbo_b.shape[1]
        self.nvir_b = self.orbv_b.shape[1]
        self.nmo_a = self.nocc_a + self.nvir_a
        self.nc = self.nocc_b
        self.nv = self.nvir_a
        self.no = self.nocc_a-self.nocc_b
        
        try: # dft
            xctype = self.mf.xc
            ni = self.mf._numint
            _, _, self.hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.mol.spin)
        except: # HF
            xctype = None
            self.hyb = 1.0
        
        
    def get_Amat(self,SA=None,foo=1.0,fglobal=1.0):
        """SA=0: SF-TDA
           SA=1: only add diagonal block for dA
           SA=2: add all dA except for OO block
           SA=3: full dA
        """
        mf = self.mf
        if SA is None:
            SA = self.SA
        #a_a2b = self.get_a2b()
        nc = self.nc
        nv = self.nv
        no = self.no
        nao = nc+nv+no
        dm = self.mf.make_rdm1()
        vhf = self.mf.get_veff(self.mf.mol, dm)
        h1e = self.mf.get_hcore()
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fockA = self.mo_coeff[0].T @ focka @ self.mo_coeff[0]
        fockB = self.mo_coeff[1].T @ fockb @ self.mo_coeff[1] 
        iden_C = np.identity(nc)
        iden_O = np.identity(no)
        iden_V = np.identity(nv)
        fockA_C = fockA[:nc,:nc]
        fockA_O = fockA[nc:nc+no,nc:nc+no] 
        fockA_V = fockA[nc+no:,nc+no:]
        fockB_C = fockB[:nc,:nc]
        fockB_O = fockB[nc:nc+no,nc:nc+no]
        fockB_V = fockB[nc+no:,nc+no:]
        dim = (nc+no)*(nv+no)
        Amat = np.zeros((dim,dim))
        sf_tda = SF_TDA(mf,method=self.method)
        sf_tda_A = sf_tda.get_Amat() 
        Amat = np.zeros_like(sf_tda_A)
        
        dim1 = nc*nv
        dim2 = dim1+nc*no
        dim3 = dim2+no*nv
        if SA==0:
            si = 1.e10
            print('Perform SF-TDA calculating...')
        else:
            si = no/2
        
        # \Delta A only for ROHF/ROKS
        if np.array(mf.mo_coeff).ndim==3: # UHF/UKS
            return sf_tda.A
        
        hf = scf.ROHF(mf.mol)
        dm = mf.make_rdm1()
        vhf = hf.get_veff(hf.mol, dm)
        h1e = hf.get_hcore()
        fockA_hf = h1e + vhf[0]
        fockB_hf = h1e + vhf[1]
        fockA_hf = mf.mo_coeff.T @ fockA_hf @ mf.mo_coeff
        fockB_hf = mf.mo_coeff.T @ fockB_hf @ mf.mo_coeff
        fockS = (fockB_hf-fockA_hf)/2
        fockS_C = fockS[:nc,:nc]
        fockS_O = fockS[nc:nc+no,nc:nc+no]
        fockS_V = fockS[nc+no:,nc+no:]
        fockS_CV = fockS[:nc,nc+no:]
        fockS_CO = fockS[:nc,nc:nc+no]
        fockS_OV = fockS[nc:nc+no,nc+no:]
        
        
        eri = ao2mo.general(mf.mol, [mf.mo_coeff,mf.mo_coeff,mf.mo_coeff,mf.mo_coeff], compact=False)
        eri = eri.reshape(nao,nao,nao,nao)
            
        #----- Diagonal blocks -----
        #CV-CV  a_a2b(nocc_a,nvir_b,nocc_a,nvir_b) (7,3,7,3) iajb
        Amat[:dim1,:dim1] += (np.einsum('ij,ab->iajb',iden_C,fockS_V).reshape(nv*nc,nv*nc) \
                            + np.einsum('ji,ab->iajb',fockS_C,iden_V).reshape(nv*nc,nv*nc))/si
        # CO-CO iujv
        Amat[dim1:dim2,dim1:dim2] += np.einsum('ji,uv->iujv',fockS_C,iden_O).reshape(no*nc,no*nc)*2/(2*si-1) \
                            - (np.einsum('uijv->iujv',eri[nc:nc+no,:nc,:nc,nc:nc+no])).reshape(no*nc,no*nc)/(2*si-1)
        # OV-OV vaub
        Amat[dim2:dim3,dim2:dim3] += np.einsum('uv,ab->uavb',iden_O,fockS_V).reshape(nv*no,nv*no)*2/(2*si-1) \
                            - (np.einsum('auvb->uavb',eri[nc+no:,nc:nc+no,nc:nc+no,nc+no:])).reshape(nv*no,nv*no)/(2*si-1)
        
        #----- Off-Diagonal blocks -----
        # CV-CO, CO-CV aivj
        if SA > 1:
            tmp_CV_CO = (np.sqrt(1+1/(2*si))-1)*(np.einsum('ij,av->iajv',iden_C,fockB_hf[nc+no:,nc:nc+no])-\
                                     np.einsum('avji-> iajv', eri[nc+no:,nc:nc+no,:nc,:nc])).reshape(nv*nc,no*nc)
            Amat[:dim1,dim1:dim2] += tmp_CV_CO
            Amat[dim1:dim2,:dim1] += tmp_CV_CO.T
        # CV-OV, OV-CV iaub
            tmp_CV_OV = (np.sqrt(1+1/(2*si))-1)*(- np.einsum('iv,ab->iavb',fockA_hf[:nc,nc:nc+no],iden_V) \
                                                -np.einsum('abvi->iavb',eri[nc+no:,nc+no:,nc:nc+no,:nc])).reshape(nv*nc,nv*no)
            Amat[:dim1,dim2:dim3] += tmp_CV_OV
            Amat[dim2:dim3,:dim1] += tmp_CV_OV.T
        # CO-OV, OV-CO iuvb
            tmp_CO_OV = (1/(2*si-1))*(np.einsum('uivb->iuvb',eri[nc:nc+no,:nc,nc:nc+no,nc+no:])-
                                     np.einsum('ubvi->iuvb',eri[nc:nc+no,nc+no:,nc:nc+no,:nc])).reshape(no*nc,nv*no)
            Amat[dim1:dim2,dim2:dim3] += tmp_CO_OV
            Amat[dim2:dim3,dim1:dim2] += tmp_CO_OV.T
        
        #--- blocks involving OO ---  
        # CV-OO, OO-CV iawv
        factor = np.sqrt((2*si+1)/(2*si-1))
        if SA > 2:
            tmp_CV_OO = -(factor-1)*np.einsum('avwi->iawv',eri[nc+no:,nc:nc+no,nc:nc+no,:nc]).reshape(nv*nc,no*no) + \
                         (1/si)*factor*np.einsum('ia,wv->iawv',fockS_CV,iden_O).reshape(nv*nc,no*no)
            Amat[:dim1,dim3:] += foo*tmp_CV_OO
            Amat[dim3:,:dim1] += foo*tmp_CV_OO.T
        
        # CO-OO, OO-CO  iuvw
            tmp_CO_OO = (np.sqrt(2*si/(2*si-1))-1)*(- np.einsum('wi,uv->iuwv',fockA_hf[nc:nc+no,:nc],iden_O).reshape(no*nc,no*no)\
                                                    -np.einsum('uvwi->iuwv',eri[nc:nc+no,nc:nc+no,nc:nc+no,:nc]).reshape(no*nc,no*no))\
                         +(1/np.sqrt(2*si*(2*si-1)))*np.einsum('iu,wv->iuwv',fockB_hf[:nc,nc:nc+no],iden_O).reshape(no*nc,no*no)
            Amat[dim1:dim2,dim3:] += foo*tmp_CO_OO
            Amat[dim3:,dim1:dim2] += foo*tmp_CO_OO.T
        # OV-OO, OO-OV uawv
            tmp_OV_OO = (np.sqrt(2*si/(2*si-1))-1)*(np.einsum('wu,av->uawv',iden_O,fockB_hf[nc+no:,nc:nc+no]).reshape(nv*no,no*no) \
                                                  - np.einsum('avwu->uawv',eri[nc+no:,nc:nc+no,nc:nc+no,nc:nc+no]).reshape(nv*no,no*no)) \
                         -(1/np.sqrt(2*si*(2*si-1)))*np.einsum('ua,wv->uawv',fockA_hf[nc:nc+no,nc+no:],iden_O).reshape(nv*no,no*no)
            Amat[dim2:dim3,dim3:] += foo*tmp_OV_OO
            Amat[dim3:,dim2:dim3] += foo*tmp_OV_OO.T

        # Add global scale
        Amat = sf_tda_A + fglobal*Amat
        
        return Amat
    
    def get_vect(self): # construct Vmat N*(N-1)
        tmp_v = np.zeros((self.no-1,self.no)) #(self.no-1,self.no)
        for i in range(1,self.no): # 1->v1 2->v2 3->v3 ...
            factor = 1/np.sqrt((self.no-i+1)*(self.no-i))
            tmp = [self.no-i] + [-1]*(self.no-i) #(N-i,-1,-1,-1 ...)
            tmp_v[i-1][i-1:] = np.array(tmp)*factor
        self.vect = tmp_v.T # N(N-1)
        #print('v ',v)
        vects = np.eye(self.no*self.no)
        vects = vects[:,:-1] # no*no*(no*no-1)
        index = [0]
        for i in range(1,self.no):
            index.append(i*(self.no+1))
        #print('index ',index)
        for i in range(self.vect.shape[1]):
            vects[0::self.no+1, index[i]] = self.vect[:,i]
        #print(vect)
        return vects
    
    def remove(self):
        # remove sf=si state
        dim3 = self.nc*self.nv + self.nc*self.no + self.no*self.nv
        dim = self.A.shape[0]
        self.vects = self.get_vect() # (no*no,no*no-1)
        A = np.zeros((dim-1,dim-1))
        A[:dim3,:dim3] = self.A[:dim3,:dim3]
        A[:dim3,dim3:] = self.A[:dim3,dim3:] @ self.vects
        A[dim3:,:dim3] = self.vects.T @ self.A[dim3:,:dim3]
        A[dim3:,dim3:] = self.vects.T @ self.A[dim3:,dim3:] @ self.vects
        
        return A
    
    def calculate_TDM(self):
        if np.array(self.mf.mo_coeff).ndim==3:
            self.calculate_TDM_U()
        else:
            self.calculate_TDM_R()
            
            
    def calculate_TDM_U(self):
        def _charge_center(mol):
            charges = mol.atom_charges()
            coords = mol.atom_coords()
            return np.einsum('z,zr->r', charges, coords)/charges.sum()
        with self.mol.with_common_orig(_charge_center(self.mol)):
            ints = self.mol.intor_symmetric('int1e_r', comp=3) # (3,nao,nao)
        ints_ab = np.einsum('xpq,pi,qj->xij',ints,self.mf.mo_coeff[0].conj(),self.mf.mo_coeff[1]) # a->b
        ints_aa = np.einsum('xpq,pi,qj->xij',ints,self.mf.mo_coeff[0].conj(),self.mf.mo_coeff[0])
        ints_bb = np.einsum('xpq,pi,qj->xij',ints,self.mf.mo_coeff[1].conj(),self.mf.mo_coeff[1])
        dim1 = self.nc*self.nv
        dim2 = dim1+self.nc*self.no
        dim3 = dim2+self.no*self.nv
            
        print("Excited state to Excited state transition dipole moments(Au)")
        print("State State    X     Y     Z     OSC.")
        
        si = self.mol.spin/2
        iden_C = np.identity(self.nc)
        iden_O = np.identity(self.no)
        iden_V = np.identity(self.nv)
        if self.SA == 0:
            factor1 = 1
            factor2 = 1
            factor3 = 0
        else:
            factor1 = np.sqrt((2*si+1)/(2*si)) 
            factor2 = np.sqrt((2*si)/(2*si-1))
            factor3 = 1/np.sqrt(2*si*(2*si-1))
            
        for i in range(len(self.e)):
            s0_cv1 = self.v[:,i][:dim1].reshape(self.nc,self.nv)
            s0_co1 = self.v[:,i][dim1:dim2].reshape(self.nc,self.no)
            s0_ov1 = self.v[:,i][dim2:dim3].reshape(self.no,self.nv)
            if self.re:
                s0_oo1 = (self.vects @ self.v[:,i][dim3:].reshape(-1,1)).reshape(self.no,self.no)
            else:
                s0_oo1 = self.v[:,i][dim3:].reshape(self.no,self.no)
            for j in range(len(self.e)):
                s1_cv1 = self.v[:,j][:dim1].reshape(self.nc,self.nv)
                s1_co1 = self.v[:,j][dim1:dim2].reshape(self.nc,self.no)
                s1_ov1 = self.v[:,j][dim2:dim3].reshape(self.no,self.nv)
                if self.re:
                    s1_oo1 = (self.vects @ self.v[:,j][dim3:].reshape(-1,1)).reshape(self.no,self.no)
                else:
                    s1_oo1 = self.v[:,j][dim3:].reshape(self.no,self.no)
                # CV1-CV1
                tdm = np.einsum('ia,xab,jb,ij->x',s0_cv1,ints_bb[:,self.nc+self.no:,self.nc+self.no:],s1_cv1,iden_C)-\
                      np.einsum('ia,xij,jb,ab->x',s0_cv1,ints_aa[:,:self.nc,:self.nc],s1_cv1,iden_V)
                # CV1-CO1
                tdm += factor1*np.einsum('ia,xav,jv,ij->x',s0_cv1,ints_bb[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_co1,iden_C)
                tdm += factor1*np.einsum('iu,xbu,jb,ij->x',s0_co1,ints_bb[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_cv1,iden_C)
                # CV1-OV1
                tdm += -factor1*np.einsum('ia,xiv,vb,ab->x',s0_cv1,ints_aa[:,:self.nc,self.nc:self.nc+self.no],s1_ov1,iden_V)
                tdm += -factor1*np.einsum('ua,xju,jb,ab->x',s0_ov1,ints_aa[:,:self.nc,self.nc:self.nc+self.no],s1_cv1,iden_V)
                # CO1-CO1
                tdm += np.einsum('iu,xuv,jv,ij->x',s0_co1,ints_bb[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_co1,iden_C)-\
                       np.einsum('iu,xij,jv,uv->x',s0_co1,ints_aa[:,:self.nc,:self.nc],s1_co1,iden_O)
                # CO1-OO1
                tdm += -factor2*np.einsum('iu,xiv,vw,uw->x',s0_co1,ints_aa[:,:self.nc,self.nc:self.nc+self.no],s1_oo1,iden_O)
                tdm +=  factor3*np.einsum('iu,xiu,vw,vw->x',s0_co1,ints_ab[:,:self.nc,self.nc:self.nc+self.no],s1_oo1,iden_O)
                tdm += -factor2*np.einsum('ut,xju,jv,tv->x',s0_oo1,ints_aa[:,:self.nc,self.nc:self.nc+self.no],s1_co1,iden_O)
                tdm +=  factor3*np.einsum('ut,xjv,jv,ut->x',s0_oo1,ints_ab[:,:self.nc,self.nc:self.nc+self.no],s1_co1,iden_O)
                # OV1-OV1
                tdm += np.einsum('ua,xab,vb,uv->x',s0_ov1,ints_bb[:,self.nc+self.no:,self.nc+self.no:],s1_ov1,iden_O)-\
                       np.einsum('ua,xuv,vb,ab->x',s0_ov1,ints_aa[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_ov1,iden_V)
                # OV1-OO1
                tdm +=  factor2*np.einsum('ua,xaw,vw,uv->x',s0_ov1,ints_bb[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_oo1,iden_O)
                tdm += -factor3*np.einsum('ua,xua,vw,vw->x',s0_ov1,ints_ab[:,self.nc:self.nc+self.no,self.nc+self.no:],s1_oo1,iden_O)
                tdm +=  factor2*np.einsum('ut,xbt,vb,uv->x',s0_oo1,ints_bb[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_ov1,iden_O)
                tdm += -factor3*np.einsum('ut,xvb,vb,ut->x',s0_oo1,ints_ab[:,self.nc:self.nc+self.no,self.nc+self.no:],s1_ov1,iden_O)
                # OO1-OO1
                tdm += np.einsum('ut,xtv,wv,uw->x',s0_oo1,ints_bb[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_oo1,iden_O)-\
                       np.einsum('ut,xuw,wv,tv->x',s0_oo1,ints_aa[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_oo1,iden_O)
                osc = (2/3)*(self.e[i]-self.e[j])*(tdm[0]**2 + tdm[1]**2 + tdm[2]**2)
                print(f'{i+1:2d} {j+1:2d} {tdm[0]:>8.4f} {tdm[1]:>8.4f} {tdm[2]:>8.4f}  {osc:>8.4f} ')
                     
    
    def calculate_TDM_R(self):
        def _charge_center(mol):
            charges = mol.atom_charges()
            coords = mol.atom_coords()
            return np.einsum('z,zr->r', charges, coords)/charges.sum()
        with self.mol.with_common_orig(_charge_center(self.mol)):
            ints = self.mol.intor_symmetric('int1e_r', comp=3) # (3,nao,nao)
        ints_mo = np.einsum('xpq,pi,qj->xij', ints, self.mf.mo_coeff, self.mf.mo_coeff)
        dim1 = self.nc*self.nv
        dim2 = dim1+self.nc*self.no
        dim3 = dim2+self.no*self.nv
            
        print("Excited state to Excited state transition dipole moments(Au)")
        print("State State    X     Y     Z     OSC.")
        
        si = self.mol.spin/2
        iden_C = np.identity(self.nc)
        iden_O = np.identity(self.no)
        iden_V = np.identity(self.nv)
        if self.SA == 0:
            factor1 = 1
            factor2 = 1
            factor3 = 0
        else:
            factor1 = np.sqrt((2*si+1)/(2*si)) 
            factor2 = np.sqrt((2*si)/(2*si-1))
            factor3 = 1/np.sqrt(2*si*(2*si-1))
        for i in range(len(self.e)):
            s0_cv1 = self.v[:,i][:dim1].reshape(self.nc,self.nv)
            s0_co1 = self.v[:,i][dim1:dim2].reshape(self.nc,self.no)
            s0_ov1 = self.v[:,i][dim2:dim3].reshape(self.no,self.nv)
            if self.re:
                s0_oo1 = (self.vects @ self.v[:,i][dim3:].reshape(-1,1)).reshape(self.no,self.no)
            else:
                s0_oo1 = self.v[:,i][dim3:].reshape(self.no,self.no)
            for j in range(len(self.e)):
                s1_cv1 = self.v[:,j][:dim1].reshape(self.nc,self.nv)
                s1_co1 = self.v[:,j][dim1:dim2].reshape(self.nc,self.no)
                s1_ov1 = self.v[:,j][dim2:dim3].reshape(self.no,self.nv)
                if self.re:
                    s1_oo1 = (self.vects @ self.v[:,j][dim3:].reshape(-1,1)).reshape(self.no,self.no)
                else:
                    s1_oo1 = self.v[:,j][dim3:].reshape(self.no,self.no)

                # CV1-CV1
                tdm = np.einsum('ia,xab,jb,ij->x',s0_cv1,ints_mo[:,self.nc+self.no:,self.nc+self.no:],s1_cv1,iden_C)-\
                      np.einsum('ia,xij,jb,ab->x',s0_cv1,ints_mo[:,:self.nc,:self.nc],s1_cv1,iden_V)
                # CV1-CO1
                tdm += factor1*np.einsum('ia,xav,jv,ij->x',s0_cv1,ints_mo[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_co1,iden_C)
                tdm += factor1*np.einsum('iu,xbu,jb,ij->x',s0_co1,ints_mo[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_cv1,iden_C)
                # CV1-OV1
                tdm += -factor1*np.einsum('ia,xiv,vb,ab->x',s0_cv1,ints_mo[:,:self.nc,self.nc:self.nc+self.no],s1_ov1,iden_V)
                tdm += -factor1*np.einsum('ua,xju,jb,ab->x',s0_ov1,ints_mo[:,:self.nc,self.nc:self.nc+self.no],s1_cv1,iden_V)
                # CO1-CO1
                tdm += np.einsum('iu,xuv,jv,ij->x',s0_co1,ints_mo[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_co1,iden_C)-\
                       np.einsum('iu,xij,jv,uv->x',s0_co1,ints_mo[:,:self.nc,:self.nc],s1_co1,iden_O)
                # CO1-OO1
                tdm += -factor2*np.einsum('iu,xiv,vw,uw->x',s0_co1,ints_mo[:,:self.nc,self.nc:self.nc+self.no],s1_oo1,iden_O)
                tdm +=  factor3*np.einsum('iu,xiu,vw,vw->x',s0_co1,ints_mo[:,:self.nc,self.nc:self.nc+self.no],s1_oo1,iden_O)
                tdm += -factor2*np.einsum('ut,xju,jv,tv->x',s0_oo1,ints_mo[:,:self.nc,self.nc:self.nc+self.no],s1_co1,iden_O)
                tdm +=  factor3*np.einsum('ut,xjv,jv,ut->x',s0_oo1,ints_mo[:,:self.nc,self.nc:self.nc+self.no],s1_co1,iden_O)
                # OV1-OV1
                tdm += np.einsum('ua,xab,vb,uv->x',s0_ov1,ints_mo[:,self.nc+self.no:,self.nc+self.no:],s1_ov1,iden_O)-\
                       np.einsum('ua,xuv,vb,ab->x',s0_ov1,ints_mo[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_ov1,iden_V)
                # OV1-OO1
                tdm +=  factor2*np.einsum('ua,xaw,vw,uv->x',s0_ov1,ints_mo[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_oo1,iden_O)
                tdm += -factor3*np.einsum('ua,xua,vw,vw->x',s0_ov1,ints_mo[:,self.nc:self.nc+self.no,self.nc+self.no:],s1_oo1,iden_O)
                tdm +=  factor2*np.einsum('ut,xbt,vb,uv->x',s0_oo1,ints_mo[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_ov1,iden_O)
                tdm += -factor3*np.einsum('ut,xvb,vb,ut->x',s0_oo1,ints_mo[:,self.nc:self.nc+self.no,self.nc+self.no:],s1_ov1,iden_O)
                # OO1-OO1
                tdm += np.einsum('ut,xtv,wv,uw->x',s0_oo1,ints_mo[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_oo1,iden_O)-\
                       np.einsum('ut,xuw,wv,tv->x',s0_oo1,ints_mo[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_oo1,iden_O)
                osc = (2/3)*(self.e[i]-self.e[j])*(tdm[0]**2 + tdm[1]**2 + tdm[2]**2)
                print(f'{i+1:2d} {j+1:2d} {tdm[0]:>8.4f} {tdm[1]:>8.4f} {tdm[2]:>8.4f}  {osc:>8.4f} ')
        
    
    def calculate_irrep(self,orb1,orb2):
        orb_sym = self.mf.get_orbsym(self.mf.mo_coeff)
        ground_sym = self.mf.get_wfnsym()
        print('ground_irrep ',ground_sym)
        #print('orb_sym',orb_sym)
        #print(orb1,orb2)
        if self.type_u:
            orb1_sym = np.array([orb_sym[0][orb1]])
            orb2_sym = np.array([orb_sym[1][orb2]])
        else:
            orb1_sym = np.array([orb_sym[orb1]])
            orb2_sym = np.array([orb_sym[orb2]])
        direct_s = direct_prod(orb1_sym,orb2_sym,self.mol.groupname)
        direct_s = direct_prod(direct_s[0],np.array(ground_sym),self.mol.groupname)
        if direct_s[0][0] >= len(self.mol.irrep_name):
            return 'A'
        else:
            return self.mol.irrep_name[direct_s[0][0]]
    
    def deltaS2_U(self,nstate):
        mo = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        occidxa = np.where(mo_occ[0]>0)[0]
        occidxb = np.where(mo_occ[1]>0)[0]
        viridxa = np.where(mo_occ[0]==0)[0]
        viridxb = np.where(mo_occ[1]==0)[0]
        mooa = mo[0][:,occidxa]
        moob = mo[1][:,occidxb]
        mova = mo[0][:,viridxa]
        movb = mo[1][:,viridxb]
        ovlp = self.mf.get_ovlp()
        #S = self.mol.intor('int1e_ovlp')
        #print('s ovlp',np.allclose(S,ovlp))
        # spin transfer matrix
        sab_oo = reduce(np.dot, (mooa.conj().T, ovlp, moob))
        #Sccba = np.einsum('pq,pi,qj->ij', ovlp, moob, mooa)
        sba_oo = sab_oo.conj().T
        #print('allclose ',np.allclose(sba_oo,Sccba))
        sab_vo = reduce(np.dot, (mova.conj().T, ovlp, moob))
        sba_vo = reduce(np.dot, (movb.conj().T, ovlp, mooa))
        #x_ba = self.v[:,nstate].reshape((self.nc+self.no,self.no+self.nv)).transpose(1,0)
        value = self.v[:,nstate]
        nc = self.nc
        no = self.no
        nv = self.nv
        x_cv_ab = value[:nc*nv].reshape(nc,nv)
        x_co_ab = value[nc*nv:nc*nv+nc*no].reshape(nc,no)
        x_ov_ab = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no,nv)
        x_oo_ab = value[nc*nv+nc*no+no*nv:].reshape(no,no)
        tmp1 = np.hstack([x_co_ab,x_cv_ab])
        tmp2 = np.hstack([x_oo_ab,x_ov_ab])
        x_ba = np.concatenate([tmp1,tmp2],axis=0).transpose(1,0)
        P_ab = lib.einsum('ai,aj,jk,ki',x_ba.conj(),x_ba,sba_oo.T.conj(),sba_oo)\
                  -lib.einsum('ai,bi,kb,ak',x_ba.conj(),x_ba,sba_vo.T.conj(),sba_vo)\
                  +lib.einsum('ai,bj,jb,ai',x_ba.conj(),x_ba,sba_vo.T.conj(),sba_vo)
        return P_ab
    
    def deltaS2_U0(self,nstate):
        nc = self.nc
        no = self.no
        nv = self.nv
        value = self.v[:,nstate]
        #print('norm: ',np.linalg.norm(value))
        value = value/np.linalg.norm(value)
        x_cv_ab = value[:nc*nv].reshape(nc,nv).transpose(1,0)
        x_co_ab = value[nc*nv:nc*nv+nc*no].reshape(nc,no).transpose(1,0)
        x_ov_ab = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no,nv).transpose(1,0)
        x_oo_ab = value[nc*nv+nc*no+no*nv:].reshape(no,no).transpose(1,0)
        mo_coeff = self.mf.mo_coeff
        orbo_a = mo_coeff[0][:, :self.nc+self.no]
        orbv_a = mo_coeff[0][:, self.nc+self.no:]
        orbo_b = mo_coeff[1][:, :self.nc]
        orbv_b = mo_coeff[1][:, self.nc:]
        S = self.mol.intor('int1e_ovlp')
        Sccba = np.einsum('pq,pi,qj->ij', S, orbo_b, orbo_a)
        Sccab = np.einsum('pq,pi,qj->ij', S, orbo_a, orbo_b)
        Scvab = np.einsum('pq,pi,qj->ij', S, orbo_a, orbv_b) 
        Svcba = np.einsum('pq,pi,qj->ij', S, orbv_b, orbo_a)
        #a = np.einsum('ai,aj,jk,ki->',x_cv_ab,x_ov_ab,Sccab[self.nc:,:],Sccba[:,:self.nc])
        #b = np.einsum('ai,aj,jk,ki->',x_ov_ab,x_cv_ab,Sccab[:self.nc,:],Sccba[:,self.nc:])
        #print('a-b', abs(a-b))
        P_ab = (np.einsum('ai,aj,jk,ki->',x_cv_ab,x_cv_ab,Sccab[:self.nc,:],Sccba[:,:self.nc])  # 1 cv-cv
              +np.einsum('ai,aj,jk,ki->',x_co_ab,x_co_ab,Sccab[:self.nc,:],Sccba[:,:self.nc])  # 1 co-co
              +np.einsum('ai,aj,jk,ki->',x_ov_ab,x_ov_ab,Sccab[self.nc:,:],Sccba[:,self.nc:]) # 1 ov-ov
              +np.einsum('ai,aj,jk,ki->',x_oo_ab,x_oo_ab,Sccab[self.nc:,:],Sccba[:,self.nc:]) # 1 oo-oo
              +np.einsum('ai,aj,jk,ki->',x_cv_ab,x_ov_ab,Sccab[self.nc:,:],Sccba[:,:self.nc]) # 1 cv-ov
              +np.einsum('ai,aj,jk,ki->',x_ov_ab,x_cv_ab,Sccab[:self.nc,:],Sccba[:,self.nc:]) # 1 ov-cv
              +np.einsum('ai,aj,jk,ki->',x_co_ab,x_oo_ab,Sccab[self.nc:,:],Sccba[:,:self.nc]) # 1 co-oo
              +np.einsum('ai,aj,jk,ki->',x_oo_ab,x_co_ab,Sccab[:self.nc,:],Sccba[:,self.nc:]) # 1 oo-co
              -np.einsum('ai,bi,kb,ak->',x_cv_ab,x_cv_ab,Scvab[:,self.no:],Svcba[self.no:,:]) # 2 cv-cv
              -np.einsum('ai,bi,kb,ak->',x_co_ab,x_co_ab,Scvab[:,:self.no],Svcba[:self.no,:]) # 2 co-co
              -np.einsum('ai,bi,kb,ak->',x_ov_ab,x_ov_ab,Scvab[:,self.no:],Svcba[self.no:,:]) # 2 ov-ov
              -np.einsum('ai,bi,kb,ak->',x_oo_ab,x_oo_ab,Scvab[:,:self.no],Svcba[:self.no,:]) # 2 oo-oo
              -np.einsum('ai,bi,kb,ak->',x_cv_ab,x_co_ab,Scvab[:,:self.no],Svcba[self.no:,:]) # 2 cv-co
              -np.einsum('ai,bi,kb,ak->',x_co_ab,x_cv_ab,Scvab[:,self.no:],Svcba[:self.no,:]) # 2 co-cv
              -np.einsum('ai,bi,kb,ak->',x_ov_ab,x_oo_ab,Scvab[:,:self.no],Svcba[self.no:,:]) # 2 ov-oo
              -np.einsum('ai,bi,kb,ak->',x_oo_ab,x_ov_ab,Scvab[:,self.no:],Svcba[:self.no,:]) # 2 oo-ov
              +np.einsum('ai,bj,jb,ai->',x_cv_ab,x_cv_ab,Scvab[:self.nc,self.no:],Svcba[self.no:,:self.nc]) # 3 cv-cv
              +np.einsum('ai,bj,jb,ai->',x_co_ab,x_co_ab,Scvab[:self.nc,:self.no],Svcba[:self.no,:self.nc]) # 3 co-co
              +np.einsum('ai,bj,jb,ai->',x_ov_ab,x_ov_ab,Scvab[self.nc:,self.no:],Svcba[self.no:,:self.no]) # 3 ov-ov
              +np.einsum('ai,bj,jb,ai->',x_oo_ab,x_oo_ab,Scvab[self.nc:,:self.no],Svcba[:self.no,self.nc:]) # 3 oo-oo
              +np.einsum('ai,bj,jb,ai->',x_cv_ab,x_co_ab,Scvab[:self.nc,:self.no],Svcba[self.no:,:self.nc]) # 3 cv-co
              +np.einsum('ai,bj,jb,ai->',x_co_ab,x_cv_ab,Scvab[:self.nc,self.no:],Svcba[:self.no,:self.nc])
              +np.einsum('ai,bj,jb,ai->',x_cv_ab,x_ov_ab,Scvab[self.nc:,self.no:],Svcba[self.no:,:self.nc]) # 3 cv-ov
              +np.einsum('ai,bj,jb,ai->',x_ov_ab,x_cv_ab,Scvab[:self.nc,self.no:],Svcba[self.no:,:self.no])
              +np.einsum('ai,bj,jb,ai->',x_cv_ab,x_oo_ab,Scvab[self.nc:,:self.no],Svcba[self.no:,:self.nc]) # 3 cv-oo
              +np.einsum('ai,bj,jb,ai->',x_oo_ab,x_cv_ab,Scvab[:self.nc,self.no:],Svcba[:self.no,self.nc:])
              +np.einsum('ai,bj,jb,ai->',x_co_ab,x_ov_ab,Scvab[self.nc:,self.no:],Svcba[:self.no,:self.nc]) # 3 co-ov
              +np.einsum('ai,bj,jb,ai->',x_ov_ab,x_co_ab,Scvab[:self.nc,:self.no],Svcba[self.no:,:self.no])
              +np.einsum('ai,bj,jb,ai->',x_co_ab,x_oo_ab,Scvab[self.nc:,:self.no],Svcba[:self.no,:self.nc]) # 3 co-oo
              +np.einsum('ai,bj,jb,ai->',x_oo_ab,x_co_ab,Scvab[:self.nc,:self.no],Svcba[:self.no,self.nc:])
              +np.einsum('ai,bj,jb,ai->',x_ov_ab,x_oo_ab,Scvab[self.nc:,:self.no],Svcba[self.no:,:self.no]) # 3 ov-oo
              +np.einsum('ai,bj,jb,ai->',x_oo_ab,x_ov_ab,Scvab[self.nc:,self.no:],Svcba[:self.no,self.nc:]))# 
        return P_ab


    def analyse(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        Ds = []
        self.syms = []

        for nstate in range(self.nstates):
            m_excited = 0.
            orb1 = None
            orb2 = None
            value = self.v[:,nstate]
            #print('self.vects.shape', self.vects.shape)
            #print(len(value))
            x_cv_ab = value[:nc*nv].reshape(nc,nv)
            x_co_ab = value[nc*nv:nc*nv+nc*no].reshape(nc,no)
            x_ov_ab = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no,nv)
            if self.re:
                x_oo_ab = (self.vects @ value[nc*nv+nc*no+no*nv:].reshape(-1,1)).reshape(no,no)
            else:
                x_oo_ab = value[nc*nv+nc*no+no*nv:].reshape(no,no)

            for o,v in zip(* np.where(abs(x_cv_ab)>0.1)):
                if abs(x_cv_ab[o,v]) > m_excited:
                    m_excited = abs(x_cv_ab[o,v])
                    orb1 = o
                    orb2 = v+self.nc+self.no
                print(f'{100*x_cv_ab[o,v]**2:3.0f}% CV(ab) {o+1}a -> {v+1+self.nc+self.no}b {x_cv_ab[o,v]:10.5f} ')
            for o,v in zip(* np.where(abs(x_co_ab)>0.1)):
                if abs(x_co_ab[o,v]) > m_excited:
                    m_excited = abs(x_co_ab[o,v])
                    orb1 = o
                    orb2 = v+self.nc
                print(f'{100*x_co_ab[o,v]**2:3.0f}% CO(ab) {o+1}a -> {v+1+self.nc}b {x_co_ab[o,v]:10.5f} ')
            for o,v in zip(* np.where(abs(x_ov_ab)>0.1)):
                if abs(x_ov_ab[o,v]) > m_excited:
                    m_excited = abs(x_ov_ab[o,v])
                    orb1 = o + self.nc
                    orb2 = v + self.nc+self.no
                print(f'{100*x_ov_ab[o,v]**2:3.0f}% OV(ab) {o+self.nc+1}a -> {v+1+self.nc+self.no}b {x_ov_ab[o,v]:10.5f} ')
            for o,v in zip(* np.where(abs(x_oo_ab)>0.1)):
                if abs(x_oo_ab[o,v]) > m_excited:
                    m_excited = abs(x_oo_ab[o,v])
                    orb1 = o + self.nc
                    orb2 = v + self.nc
                print(f'{100*x_oo_ab[o,v]**2:3.0f}% OO(ab) {o+nc+1}a -> {v+1+self.nc}b {x_oo_ab[o,v]:10.5f} ')
            if self.mol.groupname != 'C1':
                sym = self.calculate_irrep(orb1,orb2)
            else:
                sym = 'A'
            self.syms.append(sym)

            if self.SA == 0 and not self.type_u:
                Dp_ab = 0.
                Dp_ab += sum(sum(x_cv_ab*x_cv_ab)) -sum(sum(x_oo_ab*x_oo_ab))
                for i in range(no):
                    for j in range(no):
                        Dp_ab += x_oo_ab[i,i]*x_oo_ab[j,j]
                ds2 = -2*self.ground_s+1+Dp_ab
                print(f'Excited state {nstate+1} {self.e[nstate]*27.21138505:10.5f} eV {self.e[nstate]+self.mf.e_tot:11.8f} Hartree D<S^2>={ds2:3.2f} {sym}')
                Ds.append(ds2)
            elif self.type_u:
                P_ab = self.deltaS2_U(nstate)
                #ds2 = P_ab - 2*self.ground_s + 1
                ds2 = P_ab - self.no + 1
                Ds.append(ds2)
                print(f'Excited state {nstate+1} {self.e[nstate]*27.21138505:10.5f} eV {self.e[nstate]+self.mf.e_tot:11.8f} Hartree D<S^2>={ds2:3.2f} {sym}')
            else:
                print(f'Excited state {nstate+1} {self.e[nstate]*27.21138505:10.5f} eV {self.e[nstate]+self.mf.e_tot:11.8f} Hartree {sym}')
            print('')
        print('='*60)
        print(f'SF(down)-TDA |S-⟩ Energy: cost time {self.times:.2f}s')
        em = self.e * au2ev
        for i in range(0,self.nstates,1):
            print(f"No.{i:3d}  Esf={(em[i]):>10.5f} eV, En-E1={(em[i]-em[0]):>10.5f} eV,  symmetry={self.syms[i]}")
        return Ds,self.syms
    
    def recoder(self): # pec for HF molecule in 6-31G
        print("Return first three excited vectors(CO(3->5 4->5) OO(6->5) or OO(5->5 6->6) configurations).")
        dim1 = self.nc*self.nv
        dim2 = dim1 + self.nc*self.no
        dim3 = dim2 + self.no*self.nv
        if abs(self.e[0]-self.e[1]) < 1e-5:
            index = 3
        else:
            index = 1
        #vectors = np.zeros((4,5))
        print('index ',index)
        for nstate in range(index-1,index):
            cos = 0
            oos = 0
            value = self.v[:,nstate]
            #x_cv_ab = value[:dim1].reshape(nc,nv)
            x_co_ab = value[dim1:dim2].reshape(self.nc,self.no)
            #x_ov_ab = value[dim2:dim3].reshape(no,nv)
            if self.re:
                x_oo_ab = (self.vects @ value[dim3:].reshape(-1,1)).reshape(self.no,self.no)
            else:
                x_oo_ab = value[dim3:].reshape(self.no,self.no)
            for o,v in zip(* np.where(abs(x_co_ab)>1e-3)):
                if o == 2 and v == 0:  
                    cos += abs(x_co_ab[o,v])
            for o,v in zip(* np.where(abs(x_oo_ab)>1e-3)):
                if o == 1 and  v == 0:
                    oos += abs(x_oo_ab[o,v])
                #if o == v:
                #    if o+self.nc+1 == 5:
                #        vectors[nstate,-2] = x_oo_ab[o,v]
                #    else:
                #        vectors[nstate,-1] = x_oo_ab[o,v]
                #else:
                #    vectors[nstate,-3] = x_oo_ab[o,v]
        print('Recoder the abs of OO(6->5) and CO(3->5) excitation.')      
        print(oos,cos)      
        return np.array([oos,cos])

    def init_guess(self,mf, nstates): # only spin down
       
        mo_energy,mo_occ,mo_coeff = mf_info(mf)
        
        occidxa = np.where(mo_occ[0]>0)[0]
        occidxb = np.where(mo_occ[1]>0)[0]
        viridxa = np.where(mo_occ[0]==0)[0]
        viridxb = np.where(mo_occ[1]==0)[0]
        #e_ia_b2a = (mo_energy[0][viridxa,None] - mo_energy[1][occidxb]).T
        e_ia_a2b = (mo_energy[1][viridxb,None] - mo_energy[0][occidxa]).T
        #e_ia_a2b = np.array(list(cv.ravel()) + list(co.ravel())+list(ov.ravel())+list(oo.ravel()))
        no=self.no
        nc=self.nc
        nv=self.nv
        nvir = no+nv

        e_ia_a2b = e_ia_a2b.ravel()
        nov_a2b = e_ia_a2b.size

        nstates = min(nstates, nov_a2b)
        e_threshold = np.sort(e_ia_a2b)[nstates-1]
        e_threshold += 1e-5

        # spin-down
        idx = np.where(e_ia_a2b <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov_a2b))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        if self.re:
            x0 = x0[:,:-1]
        if False:
        #if self.re:
            oo = np.zeros((np.array(x0).shape[0],no*no))
            for i in range(no):
                oo[:,i*no:(i+1)*no] = x0[:,nc*nvir+i*nvir:nc*nvir+no+i*nvir]
            new_x0 = np.zeros((x0.shape[0],x0.shape[1]-1))
            new_oo = np.einsum('nx,xy->ny',oo,self.vects)
            new_x0[:,:nc*nvir] = x0[:,:nc*nvir]
            for i in range(no-1):
                new_x0[:,nc*nvir+i*nvir:nc*nvir+no+i*nvir] = new_oo[:,i*no:(i+1)*no]
                new_x0[:,nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir] = x0[:,nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir]
            new_x0[:,nc*nvir+(no-1)*nvir:nc*nvir+no-1+(no-1)*nvir] = new_oo[:,-(no-1):]
            new_x0[:,nc*nvir+(no-1)*nvir+no-1:]=x0[:,nc*nvir+(no-1)*nvir+no:]
            x0 = new_x0.copy()
            
            #x0 = np.einsum('nx,xy->ny',x0,proj)
        #print('x0.shape',x0.shape)
        return np.array(x0)
    
    def init_guess0(self,mf,nstates):
        mo_energy,mo_occ,mo_coeff = mf_info(mf)
        
        D = np.zeros((nstates,self.nc*self.nv+self.nc*self.no+self.no*self.nv+self.no*self.no))
        D = D.reshape((nstates,self.nc+self.no,self.no+self.nv))
        dm = mf.make_rdm1()
        vhf = mf.get_veff(mf.mol, dm)
        h1e = mf.get_hcore()
        fockA = h1e + vhf[0]
        fockB = h1e + vhf[1]
        fockA = mf.mo_coeff.T @ fockA @ mf.mo_coeff
        fockB = mf.mo_coeff.T @ fockB @ mf.mo_coeff
        cv1 = np.diag(fockB[self.nc+self.no:,self.nc+self.no:]).reshape(1,-1)-np.diag(fockA[:self.nc,:self.nc]).reshape(-1,1)
        co1 = np.diag(fockB[self.nc:self.nc+self.no,self.nc:self.nc+self.no]).reshape(1,-1)-np.diag(fockA[:self.nc,:self.nc]).reshape(-1,1)
        ov1 = np.diag(fockB[self.nc+self.no:,self.nc+self.no:]).reshape(1,-1)-np.diag(fockA[self.nc:self.nc+self.no,self.nc:self.nc+self.no]).reshape(-1,1)
        oo1 = np.diag(fockB[self.nc:self.nc+self.no,self.nc:self.nc+self.no]).reshape(1,-1)-np.diag(fockA[self.nc:self.nc+self.no,self.nc:self.nc+self.no]).reshape(-1,1)
        #for i in range(nstates):
            
        D[:,:self.nc,self.no:] += cv1
        D[:,:self.nc,:self.no] += co1
        D[:,self.nc:,self.no:] += ov1
        D[:,self.nc:,:self.no] += oo1
        
        return D.reshape((nstates,-1))
    
    def gen_response_sf_delta_A(self,hermi=0,max_memory=None): # only \Delta A
        mf = self.mf
        mo_energy,mo_occ,mo_coeff = mf_info(mf)
        mol = mf.mol
        #if hf_correction: # for \Delta A
        def vind(dm1):
            vj,vk = mf.get_jk(mol,dm1,hermi=hermi)
            return vj,vk
        return vind
    
    
    def gen_tda_operation_sf(self,foo,fglobal):
        mf = self.mf
        if np.array(mf.mo_coeff).ndim==3:# UKS
            mo_energy = mf.mo_energy
            mo_coeff = mf.mo_coeff
            mo_occ = mf.mo_occ
        else: # ROKS
            mo_energy = np.array([mf.mo_energy, mf.mo_energy])
            mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            mo_occ = np.zeros((2,len(mf.mo_coeff)))
            mo_occ[0][np.where(mf.mo_occ>=1)[0]]=1
            mo_occ[1][np.where(mf.mo_occ>=2)[0]]=1

        mol = mf.mol
        assert (mo_coeff[0].dtype == np.double)
        nao, nmo = mo_coeff[0].shape
        occidxa = np.where(mo_occ[0]==1)[0]
        occidxb = np.where(mo_occ[1]==1)[0]
        viridxa = np.where(mo_occ[0]==0)[0]
        viridxb = np.where(mo_occ[1]==0)[0]
        nocca = len(occidxa)
        noccb = len(occidxb)
        nvira = len(viridxa)
        nvirb = len(viridxb)
        orboa = mo_coeff[0][:,occidxa]
        orbob = mo_coeff[1][:,occidxb]
        orbva = mo_coeff[0][:,viridxa]
        orbvb = mo_coeff[1][:,viridxb]
        nc = noccb
        nv = nvira
        no = nocca - noccb
        nvir = no+nv
        si = no/2.0
        ndim = (nocca,nvirb)
        orbov = (orboa,orbvb)
        iden_C = np.eye(nc)
        iden_V = np.eye(nv)
        iden_O = np.eye(no)
        delta_ij = np.eye(nocca)
        delta_ab = np.eye(nvirb)

        dm = mf.make_rdm1()
        vhf = mf.get_veff(mf.mol, dm)
        h1e = mf.get_hcore()
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fockA = mo_coeff[0].T @ focka @ mo_coeff[0]
        fockB = mo_coeff[1].T @ fockb @ mo_coeff[1]

        e_ia = (mo_energy[1][viridxb,None] - mo_energy[0][occidxa]).T
        if self.re:
            tmp_hdiag = e_ia.ravel()
            oo = np.zeros((self.no*self.no)) # full oo
            for i in range(self.no):
                oo[i*no:(i+1)*no] = tmp_hdiag[nc*nvir+nvir*i:nc*nvir+no+nvir*i]
            new_oo = np.einsum('x,xy->y',oo,self.vects)
            new_hdiag = np.zeros(len(tmp_hdiag)-1)
            new_hdiag[:nc*nvir] = tmp_hdiag[:nc*nvir]
            for i in range(self.no-1):
                new_hdiag[nc*nvir+i*nvir:nc*nvir+no+i*nvir] = new_oo[i*no:(i+1)*no]
                new_hdiag[nc*nvir+no+i*nvir:nc*nvir+no+i*nvir+nv] = tmp_hdiag[nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir]
            new_hdiag[nc*nvir+(self.no-1)*nvir:nc*nvir+(self.no-1)*nvir+no-1] = new_oo[(self.no-1)*no:]
            new_hdiag[nc*nvir+(self.no-1)*nvir+no-1:] = tmp_hdiag[nc*nvir+(self.no-1)*nvir+no:]
            hdiag = new_hdiag
        else:
            hdiag = e_ia.ravel()
        if self.method == 0:
            vresp = gen_response_sf(self.mf,hermi=0)
        elif self.method == 1:
            vresp = gen_response_sf_mc(self.mf,hermi=0)

        if self.SA > 0:
            vresp_hf = self.gen_response_sf_delta_A(hermi=0)# to calculate \Delta A
            hf = scf.ROHF(mf.mol)
            dm = mf.make_rdm1()
            vhf = hf.get_veff(hf.mol, dm)
            h1e = hf.get_hcore()
            fockA_hf = mf.mo_coeff.T @ (h1e + vhf[0]) @ mf.mo_coeff
            fockB_hf = mf.mo_coeff.T @ (h1e + vhf[1]) @ mf.mo_coeff
            factor1 = np.sqrt((2*si+1)/(2*si))-1
            factor2 = np.sqrt((2*si+1)/(2*si-1))
            factor3 = np.sqrt((2*si)/(2*si-1))-1
            factor4 = 1/np.sqrt(2*si*(2*si-1))
            
        #@profile
        def vind(zs0): # vector-matrix product for indexed operations
            ndim0,ndim1 = ndim # ndom0:numuber of alpha orbitals, ndim1:number of beta orbitals
            orbo,orbv = orbov # mo_coeff for alpha and beta
            start_t = time.time()

            if self.re:
                oo = np.zeros((np.array(zs0).shape[0],no*no-1)) # get oo from zs0, which is no*no-1

                for i in range(no-1):
                    #print('nc*nvir+i*nvir:nc*nvir+no+i*nvir ',nc*nvir+i*nvir,nc*nvir+no+i*nvir)
                    oo[:,i*no:(i+1)*no] = np.array(zs0)[:,nc*nvir+i*nvir:nc*nvir+no+i*nvir]
                #print(oo[:,(no-1)*no:].shape, zs0[:,nc*nvir+(no-1)*nvir:nc*nvir+(no-1)*nvir+no-1].shape)
                oo[:,(no-1)*no:] = np.array(zs0)[:,nc*nvir+(no-1)*nvir:nc*nvir+(no-1)*nvir+no-1] # no*no-1
                new_oo = np.einsum('xy,ny->nx',self.vects,oo)# we want the whole matrix of oo, which is no*no
                new_zs0 = np.zeros((np.array(zs0).shape[0],np.array(zs0).shape[1]+1)) # full matrix
                new_zs0[:,:nc*nvir] = np.array(zs0)[:,:nc*nvir]
                for i in range(no-1):
                    new_zs0[:,nc*nvir+i*nvir:nc*nvir+no+i*nvir] = new_oo[:,i*no:(i+1)*no]
                    new_zs0[:,nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir] = np.array(zs0)[:,nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir]
                new_zs0[:,nc*nvir+(no-1)*nvir:nc*nvir+no+(no-1)*nvir] = new_oo[:,-no:]
                new_zs0[:,nc*nvir+(no-1)*nvir+no:]=np.array(zs0)[:,nc*nvir+(no-1)*nvir+no-1:]
            else:
                new_zs0 = zs0.copy()
            #print('new_zs.shape',new_zs0.shape)
            zs = np.asarray(new_zs0).reshape(-1,ndim0,ndim1)
            #print('zs.shape ',zs.shape)
            vs = np.zeros_like(zs)
            dmov = lib.einsum('xov,qv,po->xpq', zs,orbv.conj(), orbo,optimize=True) # (x,nmo,nmo)
            v1ao = vresp(np.asarray(dmov))   # with density and get response function
            vs += lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv,optimize=True) # (-1,nocca,nvirb)
            #np.save('v1_t.npy',v1[:,nc:,no:])
            #vs += np.einsum('ij,ab,xjb->xia',delta_ij,fockB[noccb:,noccb:],zs)-\
            #       np.einsum('ab,ij,xjb->xia',delta_ab,fockA[:nocca,:nocca],zs)
            vs += np.einsum('ab,xib->xia',fockB[noccb:,noccb:],zs)-\
                   np.einsum('ij,xja->xia',fockA[:nocca,:nocca],zs,optimize=True)
            vs_dA = np.zeros_like(vs)
            end_t = time.time()
            #print(f'USF times use {(end_t-start_t)/3600} hours')

            if self.SA > 0:
        
                cv1 = zs[:,:nc,no:]
                co1 = zs[:,:nc,:no]
                ov1 = zs[:,nc:,no:]
                oo1 = zs[:,nc:,:no]
                
                cv1_mo = np.einsum('xov,qv,po->xpq', cv1, orbvb[:,no:].conj(), orboa[:,:nc],optimize=True) # (-1,nmo,nmo)
                co1_mo = np.einsum('xov,qv,po->xpq', co1, orbvb[:,:no].conj(), orboa[:,:nc],optimize=True) # (-1,nmo,nmo)
                ov1_mo = np.einsum('xov,qv,po->xpq', ov1, orbvb[:,no:].conj(), orboa[:,nc:nc+no],optimize=True)
                oo1_mo = np.einsum('xov,qv,po->xpq', oo1, orbvb[:,:no].conj(), orboa[:,nc:nc+no],optimize=True)
                _,v1ao_cv1_k = vresp_hf(np.asarray(cv1_mo)) # (-1,nmo,nmo)
                v1ao_co1_j,v1ao_co1_k = vresp_hf(np.asarray(co1_mo))
                v1ao_ov1_j,v1ao_ov1_k = vresp_hf(np.asarray(ov1_mo))
                _,v1ao_oo1_k = vresp_hf(np.asarray(oo1_mo))
                #v1_cv1_j = np.einsum('xpq,po,qv->xov',v1ao_cv1_j,orbo.conj(), orbv) # (-1,nocca,nvirb)
                v1_co1_j = np.einsum('xpq,po,qv->xov',v1ao_co1_j,orbo.conj(), orbv,optimize=True)
                v1_ov1_j = np.einsum('xpq,po,qv->xov',v1ao_ov1_j,orbo.conj(), orbv,optimize=True)
                #v1_oo1_j = np.einsum('xpq,po,qv->xov',v1ao_oo1_j,orbo.conj(), orbv)
                v1_cv1_k = np.einsum('xpq,po,qv->xov',v1ao_cv1_k,orbo.conj(), orbv,optimize=True) # (-1,nocca,nvirb)
                v1_co1_k = np.einsum('xpq,po,qv->xov',v1ao_co1_k,orbo.conj(), orbv,optimize=True)
                v1_ov1_k = np.einsum('xpq,po,qv->xov',v1ao_ov1_k,orbo.conj(), orbv,optimize=True)
                v1_oo1_k = np.einsum('xpq,po,qv->xov',v1ao_oo1_k,orbo.conj(), orbv,optimize=True)

                # cv1 - cv1
                #vs[:,:nc,no:] += (np.einsum('ji,ab,xjb->xia',iden_C,fockB_hf[nc+no:,nc+no:],zs[:,:nc,no:])-\
                #                  np.einsum('ji,ab,xjb->xia',iden_C,fockA_hf[nc+no:,nc+no:],zs[:,:nc,no:])+\
                #                  np.einsum('ab,ji,xjb->xia',iden_V,fockB_hf[:nc,:nc],zs[:,:nc,no:])-\
                #                  np.einsum('ab,ji,xjb->xia',iden_V,fockA_hf[:nc,:nc],zs[:,:nc,no:]))/(2*si)
                vs_dA[:,:nc,no:] += (np.einsum('ab,xib->xia',fockB_hf[nc+no:,nc+no:],zs[:,:nc,no:])-\
                                  np.einsum('ab,xib->xia',fockA_hf[nc+no:,nc+no:],zs[:,:nc,no:])+\
                                  np.einsum('ji,xja->xia',fockB_hf[:nc,:nc],zs[:,:nc,no:])-\
                                  np.einsum('ji,xja->xia',fockA_hf[:nc,:nc],zs[:,:nc,no:]))/(2*si)
                # co1 - co1 (𝑢𝑖|𝑗𝑣)
                #vs[:,:nc,:no] += -v1_co1_j[:,:nc,:no]/(2*si-1)+\
                #                  (np.einsum('uv,ji,xjv->xiu',iden_O,fockB_hf[:nc,:nc],zs[:,:nc,:no])-\
                #                   np.einsum('uv,ji,xjv->xiu',iden_O,fockA_hf[:nc,:nc],zs[:,:nc,:no]))/(2*si-1)
                vs_dA[:,:nc,:no] += -v1_co1_j[:,:nc,:no]/(2*si-1)+\
                                  (np.einsum('ji,xju->xiu',fockB_hf[:nc,:nc],zs[:,:nc,:no])-\
                                   np.einsum('ji,xju->xiu',fockA_hf[:nc,:nc],zs[:,:nc,:no]))/(2*si-1)
                # ov1 - ov1 (𝑎𝑢|𝑣𝑏)
                #vs[:,nc:,no:] += -v1_ov1_j[:,nc:,no:]/(2*si-1)+\
                #                  (np.einsum('uv,ab,xvb->xua',iden_O,fockB_hf[nc+no:,nc+no:],zs[:,nc:,no:])-\
                #                   np.einsum('uv,ab,xvb->xua',iden_O,fockA_hf[nc+no:,nc+no:],zs[:,nc:,no:]))/(2*si-1)
                vs_dA[:,nc:,no:] += -v1_ov1_j[:,nc:,no:]/(2*si-1)+\
                                  (np.einsum('ab,xub->xua',fockB_hf[nc+no:,nc+no:],zs[:,nc:,no:])-\
                                   np.einsum('ab,xub->xua',fockA_hf[nc+no:,nc+no:],zs[:,nc:,no:]))/(2*si-1)

            if self.SA > 1:
                # cv1 - co1
                #vs[:,:nc,no:] += factor1*(-v1_co1_k[:,:nc,no:] + np.einsum('ij,av,xjv->xia',iden_C,fockB_hf[nc+no:,nc:nc+no],zs[:,:nc,:no]))
                vs_dA[:,:nc,no:] += factor1*(-v1_co1_k[:,:nc,no:] + np.einsum('av,xiv->xia',fockB_hf[nc+no:,nc:nc+no],zs[:,:nc,:no]))
                #vs[:,:nc,:no] += factor1*(-v1_cv1_k[:,:nc,:no] + np.einsum('ij,av,xia->xjv',iden_C,fockB_hf[nc+no:,nc:nc+no],zs[:,:nc,no:]))
                vs_dA[:,:nc,:no] += factor1*(-v1_cv1_k[:,:nc,:no] + np.einsum('av,xja->xjv',fockB_hf[nc+no:,nc:nc+no],zs[:,:nc,no:]))
                # cv1 - ov1
                #vs[:,:nc,no:] += factor1*(-v1_ov1_k[:,:nc,no:] - np.einsum('ab,vi,xvb->xia',iden_V,fockA_hf[nc:nc+no,:nc],zs[:,nc:,no:]))
                vs_dA[:,:nc,no:] += factor1*(-v1_ov1_k[:,:nc,no:] - np.einsum('vi,xva->xia',fockA_hf[nc:nc+no,:nc],zs[:,nc:,no:]))
                #vs[:,nc:,no:] += factor1*(-v1_cv1_k[:,nc:,no:] - np.einsum('ab,vi,xia->xvb',iden_V,fockA_hf[nc:nc+no,:nc],zs[:,:nc,no:]))
                vs_dA[:,nc:,no:] += factor1*(-v1_cv1_k[:,nc:,no:] - np.einsum('vi,xib->xvb',fockA_hf[nc:nc+no,:nc],zs[:,:nc,no:]))
                # co1 - ov1
                vs_dA[:,:nc,:no] += (v1_ov1_j[:,:nc,:no] - v1_ov1_k[:,:nc,:no])/(2*si-1)
                vs_dA[:,nc:,no:] += (v1_co1_j[:,nc:,no:] - v1_co1_k[:,nc:,no:])/(2*si-1)

            if self.SA > 2:
                # cv1 - oo1
                #vs[:,:nc,no:] += foo*(-(factor2-1)*(v1_oo1_k[:,:nc,no:]) + \
                #                  (factor2/(2*si))*(np.einsum('vw,ia,xvw->xia',iden_O,fockB_hf[:nc,nc+no:],zs[:,nc:,:no])-\
                #                                    np.einsum('vw,ia,xvw->xia',iden_O,fockA_hf[:nc,nc+no:],zs[:,nc:,:no])))
                vs_dA[:,:nc,no:] += foo*(-(factor2-1)*(v1_oo1_k[:,:nc,no:]) + \
                                  (factor2/(2*si))*(np.einsum('ia,xvv->xia',fockB_hf[:nc,nc+no:],zs[:,nc:,:no])-\
                                                    np.einsum('ia,xvv->xia',fockA_hf[:nc,nc+no:],zs[:,nc:,:no])))
                vs_dA[:,nc:,:no] += foo*(-(factor2-1)*(v1_cv1_k[:,nc:,:no]) + \
                                  (factor2/(2*si))*(np.einsum('vw,ia,xia->xvw',iden_O,fockB_hf[:nc,nc+no:],zs[:,:nc,no:])-\
                                                    np.einsum('vw,ia,xia->xvw',iden_O,fockA_hf[:nc,nc+no:],zs[:,:nc,no:])))

                # co1 - oo1
                #vs[:,:nc,:no] += foo*(factor3*(-v1_oo1_k[:,:nc,:no]-np.einsum('uv,iw,xwv->xiu',iden_O,fockA_hf[:nc,nc:nc+no],zs[:,nc:,:no]))+\
                #                 factor4*np.einsum('vw,iu,xvw->xiu',iden_O,fockB_hf[:nc,nc:nc+no],zs[:,nc:,:no]))
                vs_dA[:,:nc,:no] += foo*(factor3*(-v1_oo1_k[:,:nc,:no]-np.einsum('iw,xwu->xiu',fockA_hf[:nc,nc:nc+no],zs[:,nc:,:no]))+\
                                 factor4*np.einsum('vw,iu,xvw->xiu',iden_O,fockB_hf[:nc,nc:nc+no],zs[:,nc:,:no]))
                #vs[:,nc:,:no] += foo*(factor3*(-v1_co1_k[:,nc:,:no]-np.einsum('uv,iw,xiu->xwv',iden_O,fockA_hf[:nc,nc:nc+no],zs[:,:nc,:no]))+\
                #                 factor4*np.einsum('vw,iu,xiu->xvw',iden_O,fockB_hf[:nc,nc:nc+no],zs[:,:nc,:no]))
                vs_dA[:,nc:,:no] += foo*(factor3*(-v1_co1_k[:,nc:,:no]-np.einsum('iw,xiv->xwv',fockA_hf[:nc,nc:nc+no],zs[:,:nc,:no]))+\
                                 factor4*np.einsum('vw,iu,xiu->xvw',iden_O,fockB_hf[:nc,nc:nc+no],zs[:,:nc,:no]))
                # ov1 - oo1
                #vs[:,nc:,no:] += foo*(factor3*(-v1_oo1_k[:,nc:,no:]+np.einsum('wu,av,xwv->xua',iden_O,fockB_hf[nc+no:,nc:nc+no],zs[:,nc:,:no]))-\
                #                 factor4*(np.einsum('vw,au,xvw->xua',iden_O,fockA_hf[nc+no:,nc:nc+no],zs[:,nc:,:no])))
                vs_dA[:,nc:,no:] += foo*(factor3*(-v1_oo1_k[:,nc:,no:]+np.einsum('av,xuv->xua',fockB_hf[nc+no:,nc:nc+no],zs[:,nc:,:no]))-\
                                 factor4*(np.einsum('vw,au,xvw->xua',iden_O,fockA_hf[nc+no:,nc:nc+no],zs[:,nc:,:no])))
                #vs[:,nc:,:no] += foo*(factor3*(-v1_ov1_k[:,nc:,:no]+np.einsum('wu,av,xua->xwv',iden_O,fockB_hf[nc+no:,nc:nc+no],zs[:,nc:,no:]))-\
                #                 factor4*(np.einsum('vw,au,xua->xwv',iden_O,fockA_hf[nc+no:,nc:nc+no],zs[:,nc:,no:])))
                vs_dA[:,nc:,:no] += foo*(factor3*(-v1_ov1_k[:,nc:,:no]+np.einsum('av,xwa->xwv',fockB_hf[nc+no:,nc:nc+no],zs[:,nc:,no:]))-\
                                 factor4*(np.einsum('vw,au,xua->xwv',iden_O,fockA_hf[nc+no:,nc:nc+no],zs[:,nc:,no:])))
            vs = vs + fglobal * vs_dA
            nz = zs.shape[0]
            hx = vs.reshape(nz,-1)
            end_da = time.time()
            #print(f'Delta A times use {(end_da-end_t)/3600} hours')

            if self.re:
                new_hx = np.zeros_like(zs0)
                new_hx[:,:nc*nvir] += hx[:,:nc*nvir]
                oo = np.zeros((np.array(zs0).shape[0],no*no))
                for i in range(no):
                    oo[:,i*no:(i+1)*no] = hx[:,nc*nvir+i*nvir:nc*nvir+no+i*nvir]
                new_oo = np.einsum('xy,nx->ny',self.vects,oo)# no*no-1
                for i in range(no-1):
                    new_hx[:,nc*nvir+i*nvir:nc*nvir+i*nvir+no] = new_oo[:,i*no:(i+1)*no]
                    new_hx[:,nc*nvir+i*nvir+no:nc*nvir+i*nvir+no+nv] = hx[:,nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir]
                new_hx[:,nc*nvir+(no-1)*nvir:nc*nvir+(no-1)*nvir+no-1] = new_oo[:,(no-1)*no:]
                new_hx[:,nc*nvir+(no-1)*nvir+no-1:] = hx[:,nc*nvir+(no-1)*nvir+no:]
                hx = new_hx.copy()
                #zs0 = zs0.copy()
                #print('zs0.shape ',zs0.shape)
                #print('hx.shape ',hx.shape)
                
            else:
                new_hx = hx.copy()
            #print('hx ',hx.shape)
            debug = False
            if debug:
                self.hx = hx
                if self.re:
                    self.debug_hx_dav(hx)
                else:
                    self.debug_hx(hx)
            return new_hx
        return vind, hdiag
    
    def debug_hx(self,hx): # without self.re
        no = self.no
        nc = self.nc
        nv = self.nv
        nvir = no+nv
        ndim = hx.shape[1]
        passed = nc*nvir
        cv = np.zeros((nc*nv,ndim))
        co = np.zeros((nc*no,ndim-nc*nv))
        ov = np.zeros((no*nv,ndim-nc*nv-nc*no))
        oo = np.zeros((no*no,no*no))
        dim1 = nc*nv
        dim2 = dim1+nc*no
        dim3 = dim2+no*nv
        for index in range(nc): # nc-> no|nv
            for i in range(no):
                for j in range(nc):
                    co[index*no+i,j*no:j*no+no] = hx[index*nvir+i,j*nvir:nvir*j+no] # co-co
                for k in range(no):
                    co[index*no+i,nc*no+k*nv:nc*no+(k+1)*nv] = hx[index*nvir+i,passed+nvir*k+no:passed+nvir*k+no+nv] # co-ov
                    co[index*no+i,nc*no+no*nv+k*no:nc*no+no*nv+(k+1)*no] = hx[index*nvir+i,passed+nvir*k:passed+nvir*k+no] # co-oo
            for i in range(nv): # cv
                for j in range(nc):
                    cv[index*nv+i,j*nv:(j+1)*nv] = hx[index*nvir+no+i,no+j*nvir:no+j*nvir+nv] # cv-cv
                    cv[index*nv+i,nc*nv+no*j:nc*nv+no*(j+1)] = hx[index*nvir+no+i,nvir*j:nvir*j+no] # cv-co
                for k in range(no):
                    cv[index*nv+i,nc*nv+nc*no+k*nv:nc*nv+nc*no+(k+1)*nv] = hx[index*nvir+no+i,passed+nvir*k+no:passed+nvir*k+no+nv] # cv-ov
                    cv[index*nv+i,dim3+k*no:dim3+(k+1)*no] = hx[index*nvir+no+i,passed+nvir*k:passed+nvir*k+no]
        passed = nc*(nvir)
        for index in range(no): # no -> no|nv
            for i in range(no):
                for j in range(no):
                    oo[i+index*no,j*no:j*no+no] = hx[index*nvir+passed+i,passed+j*nvir:passed+j*nvir+no] # oo-oo

        for i in range(nv):
            ov[i,no*nv:no*nv+no] = hx[passed+i+no,passed:passed+no]  # ov-oo
            ov[i,no*nv+no:no*nv+no*2] = hx[passed+i+no,passed+nvir:passed+nvir+no]
            for j in range(no):
                ov[i,j*nv:(j+1)*nv] = hx[passed+i+no,passed+no+j*nvir:passed+no+nv+j*nvir]   #  ov-ov
        
       
        for i in range(nv):
            ov[nv+i,no*nv:no*nv+no] = hx[nvir+passed+i+no,passed:passed+no]  # ov-oo
            ov[nv+i,no*nv+no:no*nv+no*2] = hx[nvir+passed+i+no,passed+nvir:passed+nvir+no]
            for j in range(no):
                ov[nv+i,j*nv:(j+1)*nv] = hx[nvir+passed+i+no,passed+no+j*nvir:passed+no+nv+j*nvir]

        tmp_A = np.zeros_like(hx)
        tmp_A[:dim1,:] = cv
        tmp_A[dim1:dim2,dim1:] = co
        tmp_A[dim2:dim3,dim2:] = ov
        tmp_A[dim3:,dim3:] = oo
        new_A = np.triu(tmp_A)
        new_A_T = new_A.T + new_A - np.diag(np.diagonal(tmp_A))
        np.save('hx.npy',new_A_T)
        print('hx had save as "hx.npy"')
        return None
    
    def debug_hx_dav(self,hx): # hx(n,ndim-1) with self.re
        no = self.no
        nc = self.nc
        nv = self.nv
        nvir = no+nv
        ndim = hx.shape[1]
        passed = nc*nvir
        cv = np.zeros((nc*nv,ndim))
        co = np.zeros((nc*no,ndim-nc*nv))
        ov = np.zeros((no*nv,ndim-nc*nv-nc*no))
        oo = np.zeros((no*no-1,no*no-1))
        dim1 = nc*nv
        dim2 = dim1+nc*no
        dim3 = dim2+no*nv
        for index in range(nc): # nc-> no|nv
            for i in range(no):
                for j in range(nc):
                    co[index*no+i,j*no:j*no+no] = hx[index*nvir+i,j*nvir:nvir*j+no] # co-co
                for k in range(no-1):
                    co[index*no+i,nc*no+k*nv:nc*no+(k+1)*nv] = hx[index*nvir+i,passed+nvir*k+no:passed+nvir*k+no+nv] # co-ov
                    co[index*no+i,nc*no+no*nv+k*no:nc*no+no*nv+(k+1)*no] = hx[index*nvir+i,passed+nvir*k:passed+nvir*k+no] # co-oo
                co[index*no+i,nc*no+(no-1)*nv:nc*no+(no-1)*nv+nv] = hx[index*nvir+i,passed+nvir*(no-1)+no-1:passed+nvir*(no-1)+no-1+nv]#co-ov
                co[index*no+i,nc*no+no*nv+(no-1)*no:nc*no+no*nv+(no-1)*no+no-1] = hx[index*nvir+i,passed+nvir*(no-1):passed+nvir*(no-1)+no-1]#co-oo
            for i in range(nv): # cv
                for j in range(nc):
                    cv[index*nv+i,j*nv:(j+1)*nv] = hx[index*nvir+no+i,no+j*nvir:no+j*nvir+nv] # cv-cv
                    cv[index*nv+i,nc*nv+no*j:nc*nv+no*(j+1)] = hx[index*nvir+no+i,nvir*j:nvir*j+no] # cv-co
                for k in range(no-1):
                    cv[index*nv+i,nc*nv+nc*no+k*nv:nc*nv+nc*no+(k+1)*nv] = hx[index*nvir+no+i,passed+nvir*k+no:passed+nvir*k+no+nv] # cv-ov
                    cv[index*nv+i,dim3+k*no:dim3+(k+1)*no] = hx[index*nvir+no+i,passed+nvir*k:passed+nvir*k+no] # cv-oo
                cv[index*nv+i,nc*nv+nc*no+(no-1)*nv:nc*nv+nc*no+(no-1)*nv+nv] = hx[index*nvir+no+i,passed+nvir*(no-1)+no-1:passed+nvir*(no-1)+no-1+nv] #cv-ov
                cv[index*nv+i,dim3+(no-1)*no:dim3+(no-1)*no+no-1] = hx[index*nvir+no+i,passed+nvir*(no-1):passed+nvir*(no-1)+no-1]#cv-oo

        for index in range(no-1): # no -> no|nv
            for i in range(no):
                for j in range(no-1):
                    oo[i+index*no,j*no:j*no+no] = hx[index*nvir+passed+i,passed+j*nvir:passed+j*nvir+no] # oo-oo
                oo[i+index*no,(no-1)*no:(no-1)*no+no-1] = hx[index*nvir+passed+i,passed+(no-1)*nvir:passed+(no-1)*nvir+no-1]
            #oo[(no-1)*no+index,index*no:index*no+no] = hx[passed+(no-1)*nvir+index,passed+index*nvir:passed+index*nvir+no]
            oo[(no-1)*no+index,(no-1)*no:(no-1)*no+no-1] = hx[passed+(no-1)*nvir+index,passed+(no-1)*nvir:passed+(no-1)*nvir+no-1]

        for j in range(no-1):
            for i in range(nv): # ov diag
                ov[i,j*nv:(j+1)*nv] = hx[passed+no+i,passed+no:passed+no+nv] # ov-ov, 1 electron in no
                ov[i+nv,j*nv:(j+1)*nv] = hx[passed+no-1+i+nvir,passed+no:passed+no+nv] # ov-ov
        for i in range(nv):
            ov[i,(no-1)*nv:no*nv] = hx[passed+no+i,passed+nvir+no-1:passed+nvir+no-1+nv] # ov-ov
            ov[i+nv,(no-1)*nv:no*nv] = hx[passed+no-1+i+nvir,passed+nvir+no-1:passed+nvir+no-1+nv] # ov-ov
        for j in range(no-1):
            for i in range(nv):
                ov[i,(j+1)*no*nv:(j+1)*no*nv+no] = hx[passed+no+i,passed+j*nvir:passed+j*nvir+no] #ov-oo
                ov[i+nv,(j+1)*no*nv:(j+1)*no*nv+no] = hx[passed+no-1+i+nvir,passed+j*nvir:passed+j*nvir+no]#ov-oo
        for i in range(nv):
            ov[i,no*nv*(no-1)+no:no*nv*(no-1)+no+no-1] = hx[passed+no+i,passed+nvir*(no-1):passed+nvir*(no-1)+no-1] #ov-oo
            ov[i+nv,no*nv*(no-1)+no:no*nv*(no-1)+no+no-1] = hx[passed+no-1+i+(no-1)*nvir,passed+nvir*(no-1):passed+nvir*(no-1)+no-1]

        tmp_A = np.zeros_like(hx)
        print('hx.shape ',hx.shape)
        print('oo.shape ',oo.shape)
        tmp_A[:nc*nv,:] = cv
        tmp_A[nc*nv:nc*nv+nc*no,nc*nv:] = co
        tmp_A[nc*nv+nc*no:nc*nv+nc*no+no*nv,nc*nv+nc*no:] = ov
        tmp_A[nc*nv+nc*no+no*nv:,nc*nv+nc*no+no*nv:] = oo
        new_A = np.triu(tmp_A)
        new_A_T = new_A.T + new_A - np.diag(np.diagonal(tmp_A))
        np.save('hx_dav.npy',new_A_T)
        print('hx had save as "hx_dav.npy"')
        return None
    
    def deal_v_davidson(self):
        # change davidson data form like nvir|nvir|nvir|...(alpha->beta nc|no -> no|nv)  to cv|co|ov|oo
 
        cv = np.zeros((self.nstates,self.nc,self.nv))
        co = np.zeros((self.nstates,self.nc,self.no))
        ov = np.zeros((self.nstates,self.no,self.nv))
        if self.re:
            oo = np.zeros((self.nstates,self.no*self.no-1))
        else:
            oo = np.zeros((self.nstates,self.no,self.no))
        nvir = self.no+self.nv
        passed = self.nc*nvir
        if self.nstates == (self.nc+self.no)*(self.no+self.nv):
            nstates = self.nstates-1
        else:
            nstates = self.nstates
        for state in range(nstates):
            tmp_data = self.v[:,state]
            #print(tmp_data[passed:])
            for i in range(self.nc):
                cv[state,i,:] += tmp_data[i*nvir+self.no:i*nvir+self.no+self.nv]
                co[state,i,:] += tmp_data[i*nvir:i*nvir+self.no]
            if self.re:
                for i in range(self.no-1):
                    oo[state,i*self.no:(i+1)*self.no] += tmp_data[passed+i*nvir:passed+i*nvir+self.no]
                    ov[state,i,:] += tmp_data[passed+i*nvir+self.no:passed+i*nvir+self.no+self.nv]
                oo[state,(self.no-1)*self.no:] += tmp_data[passed+(self.no-1)*nvir:passed+(self.no-1)*nvir+self.no-1]
                ov[state,self.no-1,:] += tmp_data[passed+(self.no-1)*nvir+self.no-1:]
            else:
                for i in range(self.no):
                    oo[state,i,:] += tmp_data[passed+i*nvir:passed+i*nvir+self.no]
                    ov[state,i,:] += tmp_data[passed+i*nvir+self.no:passed+i*nvir+self.no+self.nv]
        
        v = np.hstack([cv.reshape(self.nstates,-1),co.reshape(self.nstates,-1),ov.reshape(self.nstates,-1),oo.reshape(self.nstates,-1)])
        return v.T
    
    def davidson_process(self,foo,fglobal):
        #print("Davidson process...")
        vind, hdiag = self.gen_tda_operation_sf(foo,fglobal)
        precond = hdiag
        #print('precode ',precond)
        #print('init_guess.. ')
        start_t = time.time()
        x0 = self.init_guess(self.mf, self.nstates)
        end_t = time.time()
        #print(f'init_guess times use {(end_t-start_t)/3600:6.4f} hours')
        #print('x0.shape ',x0.shape)
        converged, e, x1 = lib.davidson1(vind, x0, precond,
                              tol=1e-16,
                              nroots=self.nstates,
                              max_cycle=300)
        end_time = time.time()
        #print(f'davidson time use {(end_time-end_t)/3600} hours')
        self.converged = converged
        self.e = e
        self.v = np.array(x1).T
        #print(self.v.shape)
        self.v = self.deal_v_davidson()
        print('Converged ',converged)
        return None
    
    def frozen_A(self,frozen):
        # forzen=True, drop out innermost orbital, else drop out n orbital(n=frozen)
        if type(frozen)==int:
            if f == 0:
                f=1
            else:
                f = frozen
        else:
            f=1
        # CV1
        minus_cv = self.A[f*self.nv:,f*self.nv:]
        # CO1
        dim = minus_cv.shape[0]
        kept = np.r_[0:(self.nc-f)*self.nv, (self.nc-f)*self.nv+f*self.no:dim]
        tmp_A = minus_cv[kept,:]
        tmp_A = tmp_A[:, kept]
        return tmp_A
            
    def kernel(self, nstates=1,remove=False,frozen=None,foo=1.0,d_lda=0.3,fglobal=None):
        time0 = time.time()
        self.re = remove
        nov = (self.nc+self.no) * (self.no+self.nv)
        self.nstates = min(nstates,nov)
        if fglobal is None:
            fglobal = (1-d_lda)*self.hyb + d_lda
        if remove:
            #print('fglobal',fglobal)
            if self.davidson:
                self.vects = self.get_vect()
                self.davidson_process(foo=foo,fglobal=fglobal)
            else:
                self.A = self.get_Amat(foo=foo,fglobal=fglobal)
                #np.save('diag_A.npy',self.A)
                #print('matrix saved as diag_A.nyp.')
                self.A = self.remove()
                if self.A.shape[0] < 1000:
                    e,v = scipy.linalg.eigh(self.A)
                else:
                    dim_n = self.A.shape[0]
                    nroots = min(nstates+5,dim_n)
                    e,v = scipy.sparse.linalg.eigsh(self.A,k=nroots,which='SA')
                self.e = e[:nstates]
                self.v = v[:,:nstates]
        else:
            if self.davidson:
                self.davidson_process(foo=foo,fglobal=fglobal)
            else:
                self.A = self.get_Amat(foo=foo,fglobal=fglobal)
                if frozen is not None:
                    self.A = self.frozen_A(frozen)
                if self.A.shape[0] < 1000:
                    e,v = scipy.linalg.eigh(self.A)
                else:
                    dim_n = self.A.shape[0]
                    nroots = min(dim_n,nstates+5)
                    e,v = scipy.sparse.linalg.eigsh(self.A,k=nroots,which='SA')
                self.e = e[:nstates]
                self.v = v[:,:nstates]
        time1 = time.time()
        self.times = time1-time0
        return self.e*27.21138505, self.v



if __name__ == '__main__':
    mol = gto.M(
            atom = """ Be """,
            basis = 'aug-cc-pvtz',
            charge = 0,
            spin = 2,
            verbose = 3,
            symmetry='D2h',
        )
    print('Test with Be atom.')    
    mf = dft.ROKS(mol)
    mf.xc = 'bhandhlyp'
    mf.kernel()
    sf_tda = SA_SF_TDA(mf)
    e0, values = sf_tda.kernel(nstates=10,remove=True)
    print('excited energy ',e0)
    print('Reference energy: -2.58159612  1.94501967  2.0441558   2.04415705  3.55556409  4.0395836 4.07260624  4.07260634  4.09542032  4.09542242')

