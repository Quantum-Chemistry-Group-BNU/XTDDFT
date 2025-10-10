from pyscf import gto, dft, scf, ao2mo, lib, tddft
import numpy as np
from pyscf.dft import numint
import scipy
from functools import reduce

import sys

from pyscf import gto, dft, scf, ao2mo, lib, tddft
import numpy as np
from pyscf.dft import numint
import scipy
from functools import reduce

import sys
au2ev = 27.21138505

def SF_TDA(mf,collinear=False,isf=-1):
    if isf == -1: # down
        return SF_TDA_down(mf,collinear)
    elif isf == 1: # up
        return SF_TDA_up(mf,collinear)


class SF_TDA_up():
    def __init__(self,mf,collinear=False):
        if np.array(mf.mo_coeff).ndim==3:# UKS
            mo_energy = mf.mo_energy
            mo_coeff = mf.mo_coeff
            mo_occ = mf.mo_occ
            type_u = True
            SA = 0
        else: # ROKS
            mo_energy = np.array([mf.mo_energy, mf.mo_energy])
            mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            mo_occ = np.zeros((2,len(mf.mo_coeff)))
            mo_occ[0][np.where(mf.mo_occ>=1)[0]]=1
            mo_occ[1][np.where(mf.mo_occ>=2)[0]]=1
            type_u = False
        mol = mf.mol
        nao = mol.nao_nr()


        occidx_a = np.where(mo_occ[0]==1)[0]
        viridx_a = np.where(mo_occ[0]==0)[0]
        occidx_b = np.where(mo_occ[1]==1)[0]
        viridx_b = np.where(mo_occ[1]==0)[0]
        orbo_a = mo_coeff[0][:,occidx_a]
        orbv_a = mo_coeff[0][:,viridx_a]
        orbo_b = mo_coeff[1][:,occidx_b]
        orbv_b = mo_coeff[1][:,viridx_b]
        nocc_a = orbo_a.shape[1]
        nvir_a = orbv_a.shape[1]
        nocc_b = orbo_b.shape[1]
        nvir_b = orbv_b.shape[1]
        nmo_a = nocc_a + nvir_a
        nc = nocc_b
        nv = nvir_a
        no = nocc_a-nocc_b
        delta_ij = np.eye(nocc_a)
        delta_ab = np.eye(nvir_b)

        a_b2a = np.zeros((nocc_b,nvir_a,nocc_b,nvir_a))
        if no == 0:
            dm = mf.make_rdm1()
            vhf = mf.get_veff(mf.mol, dm)
            h1e = mf.get_hcore()
            focka = h1e + vhf
            fockb = h1e + vhf
            fockA = mo_coeff[0].T @ focka @ mo_coeff[0]
            fockB = mo_coeff[1].T @ fockb @ mo_coeff[1]
        else:
            dm = mf.make_rdm1()
            vhf = mf.get_veff(mf.mol, dm)
            h1e = mf.get_hcore()
            focka = h1e + vhf[0]
            fockb = h1e + vhf[1]
            fockA = mo_coeff[0].T @ focka @ mo_coeff[0]
            fockB = mo_coeff[1].T @ fockb @ mo_coeff[1]

        def add_hf_(a_b2a, hyb=1):

            eri_mo = ao2mo.general(mol, [orbo_b,orbo_b,orbv_a,orbv_a], compact=False)
            eri_mo = eri_mo.reshape(nocc_b,nocc_b,nvir_a,nvir_a)
            a_b2a -= np.einsum('ijba->iajb', eri_mo) * hyb
            return a_b2a

        try:
            xctype = mf.xc
        except:
            xctype = None
            #eri_mo_b2a = ao2mo.general(self.mol, [self.orbo_b,self.orbo_b,self.orbv_a,self.orbv_a], compact=False)
            #eri_a_b2a = eri_mo_a2b.reshape(nocc_b,nocc_b,nvir_a,nvir_a)
            #a_b2a -= np.einsum('ijba->iajb', eri_a_b2a) * hyb
            #del eri_mo_b2a

        if xctype is not None:
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            if mf.nlc or ni.libxc.is_nlc(mf.xc):
                logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                            'deriviative is not available. Its contribution is '
                            'not included in the response function.')
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
            print('omega alpha hyb',omega, alpha, hyb)
            if hyb != 0:
                a_b2a = add_hf_(a_b2a,hyb)

            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1()
            #if np.array(mf.mo_coeff).ndim==2 and not mol.symmetry:
            if np.array(mf.mo_coeff).ndim==2:
                dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
                dm0.mo_occ = mo_occ
            make_rho = ni._gen_rho_evaluator(mf.mol, dm0, hermi=0, with_lapl=False)[0]
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype == 'LDA' and not collinear:
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                exc, vxc, fxc, kxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, omega=omega, xctype=xctype) # vxc.shape[2,1,N]
                vxc_a = vxc[0,0]*weight
                vxc_b = vxc[1,0]*weight
                fxc_ab = (vxc_a-vxc_b)/(rho0a-rho0b+1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                rho_ov_b2a = np.einsum('ri,ra->ria', rho_o_b, rho_v_a)
                rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                w_ov = np.einsum('ria,r->ria', rho_ov_b2a, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                a_b2a += iajb
                #w_ov = np.einsum('ria,r->ria', rho_ov_a2b, fxc_ab)
                #iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                #a_a2b += iajb


        elif xctype == 'GGA' and not collinear:
            ao_deriv = 1
            for ao, mask, weight, coords \
                 in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv, max_memory):#ao(4,N,nao):AO values and derivatives in x,y,z compoents in grids
                rho0a = make_rho(0, ao, mask, xctype)#(4,N):density and "density derivatives" for x,y,z components in grids
                rho0b = make_rho(1, ao, mask, xctype)
                rha = np.zeros_like(rho0a)
                rha[0] = rho0a[0]
                rhb = np.zeros_like(rho0b)
                rhb[0] = rho0b[0]
                exc, vxc, fxc, kxc = ni.eval_xc_eff(mf.xc, (rha, rhb), deriv=1, omega=omega,xctype=xctype)#vxc.shape(2,4,N)
                vxc_a = vxc[0,0]*weight #first order derivatives about \rho in \alpha
                vxc_b = vxc[1,0]*weight #\beta
                fxc_ab = (vxc_a-vxc_b)/(rho0a[0]-rho0b[0]+1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao[0], orbo_a) # (N,i)
                rho_v_a = lib.einsum('rp,pi->ri', ao[0], orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao[0], orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao[0], orbv_b)
                #rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                rho_ov_b2a = np.einsum('ri,ra->ria', rho_o_b, rho_v_a)
                w_ov = np.einsum('ria,r->ria', rho_ov_b2a, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                a_b2a += iajb
        elif xctype == None:
            add_hf_(a_b2a, hyb=1)
        self.A = (a_b2a + np.einsum('ij,ab->iajb',delta_ij[no:,no:],fockA[nc+no:,nc+no:])-np.einsum('ij,ab->iajb',fockB[:nc,:nc],delta_ab[no:,no:])).reshape((nc*nv,nc*nv))
        
    def kernel(self,nstates=1):
        e,v = scipy.linalg.eigh(self.A)
        return e[:nstates]*au2ev,v[:,:nstates]

# spin_down
class SF_TDA_down():
    def __init__(self,mf, collinear=False):

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
        nao = mol.nao_nr()
        self.collinear = collinear
        occidx_a = np.where(mo_occ[0]==1)[0]
        viridx_a = np.where(mo_occ[0]==0)[0]
        occidx_b = np.where(mo_occ[1]==1)[0]
        viridx_b = np.where(mo_occ[1]==0)[0]
        orbo_a = mo_coeff[0][:,occidx_a]
        orbv_a = mo_coeff[0][:,viridx_a]
        orbo_b = mo_coeff[1][:,occidx_b]
        orbv_b = mo_coeff[1][:,viridx_b]
        nocc_a = orbo_a.shape[1]
        nvir_a = orbv_a.shape[1]
        nocc_b = orbo_b.shape[1]
        nvir_b = orbv_b.shape[1]
        
        #a_b2a = np.zeros((nocc_b,nvir_a,nocc_b,nvir_a))
        a_a2b = np.zeros((nocc_a,nvir_b,nocc_a,nvir_b))
        delta_ij = np.eye(nocc_a)
        delta_ab = np.eye(nvir_b)
        dm = mf.make_rdm1()
        vhf = mf.get_veff(mf.mol, dm)
        h1e = mf.get_hcore()
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fockA = mo_coeff[0].T @ focka @ mo_coeff[0]
        fockB = mo_coeff[1].T @ fockb @ mo_coeff[1]
        
        #a = (a_b2a, a_a2b)
        nc = nocc_b
        nv = nvir_a
        no = nocc_a-nocc_b
        
        
        def add_hf_(a_a2b, hyb=1):
            
            #eri_a_b2a = ao2mo.general(mol, [orbo_b,orbo_b,orbv_a,orbv_a], compact=False)
            eri_mo = ao2mo.general(mol, [orbo_a,orbo_a,orbv_b,orbv_b], compact=False)

            #eri_a_b2a = eri_a_b2a.reshape(nocc_b,nocc_b,nvir_a,nvir_a)
            eri_mo = eri_mo.reshape(nocc_a,nocc_a,nvir_b,nvir_b)

            #a_b2a, a_a2b = a

            #a_b2a-= np.einsum('ijba->iajb', eri_a_b2a) * hyb
            a_a2b -= np.einsum('ijba->iajb', eri_mo) * hyb
            del eri_mo
            
        try:
            xctype = mf.xc
        except:
            xctype = None # only work for HF 
            eri_mo_a2b = ao2mo.general(mol, [orbo_a,orbo_a,orbv_b,orbv_b], compact=False)
            eri_a_a2b = eri_mo_a2b.reshape(nocc_a,nocc_a,nvir_b,nvir_b)
            a_a2b -= np.einsum('ijba->iajb', eri_a_a2b)
            del eri_mo_a2b
            
        if xctype is not None:
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            if mf.nlc or ni.libxc.is_nlc(mf.xc):
                logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                            'deriviative is not available. Its contribution is '
                            'not included in the response function.')
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
            print('omega alpha hyb',omega, alpha, hyb)
            if hyb != 0:
                add_hf_(a_a2b,hyb)

            xctype = ni._xc_type(mf.xc)
            print(xctype)
            #if mol.symmetry:
            #    #dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
            #else:
            #    #dm0 = mf.make_rdm1(mf.mo_coeff,mo_occ)
            dm0 = mf.make_rdm1()
            #if np.array(mf.mo_coeff).ndim==2 and not mol.symmetry:
            if np.array(mf.mo_coeff).ndim==2:
                dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
                dm0.mo_occ = mo_occ
            make_rho = ni._gen_rho_evaluator(mf.mol, dm0, hermi=0, with_lapl=False)[0]
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)
        
        if xctype == 'LDA' and not self.collinear:
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                exc, vxc, fxc, kxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, omega=omega, xctype=xctype) # vxc.shape[2,1,N]
                vxc_a = vxc[0,0]*weight
                vxc_b = vxc[1,0]*weight
                fxc_ab = (vxc_a-vxc_b)/(rho0a-rho0b+1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                #rho_ov_b2a = np.einsum('ri,ra->ria', rho_o_b, rho_v_a)
                rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                #w_ov = np.einsum('ria,r->ria', rho_ov_b2a, fxc_ab)
                #iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                #a_b2a += iajb
                w_ov = np.einsum('ria,r->ria', rho_ov_a2b, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                a_a2b += iajb
                
                
        elif xctype == 'GGA' and not self.collinear:
            ao_deriv = 1
            for ao, mask, weight, coords \
                 in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv, max_memory):#ao(4,N,nao):AO values and derivatives in x,y,z compoents in grids
                rho0a = make_rho(0, ao, mask, xctype)#(4,N):density and "density derivatives" for x,y,z components in grids
                rho0b = make_rho(1, ao, mask, xctype)
                rha = np.zeros_like(rho0a)
                rha[0] = rho0a[0]
                rhb = np.zeros_like(rho0b)
                rhb[0] = rho0b[0]
                exc, vxc, fxc, kxc = ni.eval_xc_eff(mf.xc, (rha, rhb), deriv=1, omega=omega,xctype=xctype)#vxc.shape(2,4,N)
                vxc_a = vxc[0,0]*weight #first order derivatives about \rho in \alpha
                vxc_b = vxc[1,0]*weight #\beta
                fxc_ab = (vxc_a-vxc_b)/(rho0a[0]-rho0b[0]+1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao[0], orbo_a) # (N,i)
                rho_v_a = lib.einsum('rp,pi->ri', ao[0], orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao[0], orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao[0], orbv_b)
                rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                w_ov = np.einsum('ria,r->ria', rho_ov_a2b, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                a_a2b += iajb
            
        self.nc = nc
        self.nv = nv
        self.no = no
        self.A = self.get_Amat(fockA,fockB,a_a2b)
        
    def get_Amat(self,fockA,fockB,a_a2b):
        nc = self.nc
        nv = self.nv
        no = self.no
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
        print('The row of A matrix ', dim)
        Amat = np.zeros((dim,dim))
        dim1 = nc*nv
        dim2 = dim1+nc*no
        dim3 = dim2+no*nv
        
        #----- Diagonal blocks -----
        #CV-CV  a_a2b(nocc_a,nvir_b,nocc_a,nvir_b) (7,3,7,3) iajb
        Amat[:dim1,:dim1]= np.einsum('ij,ab->iajb',iden_C,fockB_V).reshape(nv*nc,nv*nc)\
                           - np.einsum('ji,ab->iajb',fockA_C,iden_V).reshape(nv*nc,nv*nc) \
                           + a_a2b[:nc,no:,:nc,no:].reshape(nv*nc,nv*nc)
        # CO-CO ivju
        Amat[dim1:dim2,dim1:dim2] = np.einsum('ij,xy->ixjy',iden_C,fockB_O).reshape(no*nc,no*nc) \
                              - np.einsum('ji,xy->ixjy',fockA_C,iden_O).reshape(no*nc,no*nc) \
                              + a_a2b[:nc,:no,:nc,:no].reshape(no*nc,no*nc)
        # OV-OV vaub
        Amat[dim2:dim3,dim2:dim3] = np.einsum('xy,ab->xayb',iden_O,fockB_V).reshape(nv*no,nv*no) \
                              - np.einsum('yx,ab->xayb',fockA_O,iden_V).reshape(nv*no,nv*no) \
                              + a_a2b[nc:,no:,nc:,no:].reshape(nv*no,nv*no)
        # OO-OO utvw
        Amat[dim3:,dim3:] = np.einsum('uv,tw->utvw',iden_O,fockB_O).reshape(no*no,no*no) \
                      - np.einsum('vu,tw->utvw',fockA_O,iden_O).reshape(no*no,no*no) \
                      + a_a2b[nc:nc+no,:no,nc:nc+no,:no].reshape(no*no,no*no)
        
        #----- Off-Diagonal blocks -----
        # CV-CO, CO-CV iaju
        tmp_CV_CO = np.einsum('ij,ay->iajy',iden_C,fockB[nc+no:,nc:nc+no]).reshape(nv*nc,no*nc) \
                  + a_a2b[:nc,no:,:nc,:no].reshape(nv*nc,no*nc)
        Amat[:dim1,dim1:dim2] = tmp_CV_CO
        Amat[dim1:dim2,:dim1] = tmp_CV_CO.T
        # CV-OV, OV-CV iaub
        tmp_CV_OV = - np.einsum('yi,ab->iayb',fockA[nc:nc+no,:nc],iden_V).reshape(nv*nc,nv*no) \
                  + a_a2b[:nc,no:,nc:nc+no,no:].reshape(nv*nc,nv*no)
        Amat[:dim1,dim2:dim3] = tmp_CV_OV
        Amat[dim2:dim3,:dim1] = tmp_CV_OV.T
        # CO-OV, OV-CO ivub
        tmp_CO_OV = a_a2b[:nc,:no,nc:nc+no,no:].reshape(no*nc,nv*no)    
        Amat[dim1:dim2,dim2:dim3] = tmp_CO_OV
        Amat[dim2:dim3,dim1:dim2] = tmp_CO_OV.T
        
        #--- blocks involving OO ---  
        # CV-OO, OO-CV iauw
        tmp_CV_OO = a_a2b[:nc,no:,nc:nc+no,:no].reshape(nv*nc,no*no)
        Amat[:dim1,dim3:] = tmp_CV_OO
        Amat[dim3:,:dim1] = tmp_CV_OO.T
        # CO-OO, OO-CO  ivuw
        tmp_CO_OO = - np.einsum('yi,WZ->iWyZ',fockA[nc:nc+no,:nc],iden_O).reshape(no*nc,no*no)\
                    + a_a2b[:nc,:no,nc:nc+no,:no].reshape(no*nc,no*no)
        Amat[dim1:dim2,dim3:] = tmp_CO_OO
        Amat[dim3:,dim1:dim2] = tmp_CO_OO.T
        # OV-OO, OO-OV vauw
        tmp_OV_OO = np.einsum('yx,aZ->xayZ',iden_O,fockB[nc+no:,nc:nc+no]).reshape(nv*no,no*no) \
                   + a_a2b[nc:,no:,nc:,:no].reshape(nv*no,no*no)
        Amat[dim2:dim3,dim3:] = tmp_OV_OO
        Amat[dim3:,dim2:dim3] = tmp_OV_OO.T
        
        del a_a2b
        return Amat

    def analyse(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        Ds = []
        for nstate in range(self.nstates):
            value = self.v[:,nstate]
            x_cv_ab = value[:nc*nv].reshape(nc,nv)
            x_co_ab = value[nc*nv:nc*nv+nc*no].reshape(nc,no)
            x_ov_ab = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no,nv)
            
            x_oo_ab = value[nc*nv+nc*no+no*nv:].reshape(no,no)
            Dp_ab = 0.
            Dp_ab += sum(sum(x_cv_ab*x_cv_ab)) -sum(sum(x_oo_ab*x_oo_ab))
            for i in range(no):
                for j in range(no):
                    Dp_ab += x_oo_ab[i,i]*x_oo_ab[j,j]
                    
            print(f'Excited state {nstate+1} {self.e[nstate]*27.21138505:10.5f} eV D<S^2>={-no+1+Dp_ab:5.2f}')
            for o,v in zip(* np.where(abs(x_cv_ab)>0.1)):
                print(f'{100*x_cv_ab[o,v]**2:3.0f}% CV(ab) {o+1}a -> {v+1+self.nc+self.no}b {x_cv_ab[o,v]:10.5f}')
            for o,v in zip(* np.where(abs(x_co_ab)>0.1)):
                print(f'{100*x_co_ab[o,v]**2:3.0f}% CO(ab) {o+1}a -> {v+1+self.nc}b {x_co_ab[o,v]:10.5f}')
            for o,v in zip(* np.where(abs(x_ov_ab)>0.1)):
                print(f'{100*x_ov_ab[o,v]**2:3.0f}% OV(ab) {o+nc+1}a -> {v+1+self.nc+self.no}b {x_ov_ab[o,v]:10.5f}')
            for o,v in zip(* np.where(abs(x_oo_ab)>0.1)):
                print(f'{100*x_oo_ab[o,v]**2:3.0f}% OO(ab) {o+nc+1}a -> {v+1+self.nc}b {x_oo_ab[o,v]:10.5f}')
            print(' ')
            Ds.append(-no+1+Dp_ab)
        return None
    def kernel(self, nstates=1):
        e,v = scipy.linalg.eigh(self.A)
        self.e = e
        self.v = v
        self.nstates = nstates
        return e[:nstates]*27.21138505, v[:,:nstates]
