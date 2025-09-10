from pyscf import gto, dft, scf, ao2mo, lib, tddft,symm
from pyscf.dft.numint import _dot_ao_ao_sparse,_scale_ao_sparse,_tau_dot_sparse
from pyscf.dft.gen_grid import NBINS
import numpy as np
from pyscf.dft import numint
import scipy
from functools import reduce
from pyscf.symm import direct_prod
#import sys
#sys.path.append('/home/lenovo2/usrs/zhw/TDDFT')
from SF_TDA import *


def get_irrep_occupancy_directly(mol, mf):
    results = {}
    
    if len(mf.mo_occ)==2:  # RKS/ROKS
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

def mf_info(mf):
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
    return mo_energy,mo_occ,mo_coeff

def init_guess(mf, nstates=None, wfnsym=None):
    nstates = nstates
    mo_energy,mo_occ,mo_coeff = mf_info(mf)
    mol = mf.mol
    occidxa = np.where(mo_occ[0]>0)[0]
    occidxb = np.where(mo_occ[1]>0)[0]
    viridxa = np.where(mo_occ[0]==0)[0]
    viridxb = np.where(mo_occ[1]==0)[0]
    e_ia_b2a = (mo_energy[0][viridxa,None] - mo_energy[1][occidxb]).T
    e_ia_a2b = (mo_energy[1][viridxb,None] - mo_energy[0][occidxa]).T

    e_ia_b2a = e_ia_b2a.ravel()
    e_ia_a2b = e_ia_a2b.ravel()
    nov_b2a = e_ia_b2a.size
    nov_a2b = e_ia_a2b.size

    nstates = min(nstates, nov_a2b)
    e_threshold = np.sort(e_ia_a2b)[nstates-1]
    e_threshold += 1e-5

    # spin-down
    idx = np.where(e_ia_a2b <= e_threshold)[0]
    x0 = np.zeros((idx.size, nov_a2b))
    for i, j in enumerate(idx):
        x0[i, j] = 1  # Koopmans' excitations

    return x0

def cache_xc_kernel_sf(mf, mo_coeff, mo_occ, spin=1,max_memory=2000):
    '''Compute the fxc_sf, which can be used in SF-TDDFT/TDA
    '''
    MGGA_DENSITY_LAPL = False
    with_lapl = MGGA_DENSITY_LAPL
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    mo_energy,mo_occ,mo_coeff = mf_info(mf)

    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
    else: 
        ao_deriv = 0 # LDA

    assert mo_coeff[0].ndim == 2
    assert spin == 1

    nao = mo_coeff[0].shape[0]
    dm0 = mf.make_rdm1()
    if np.array(mf.mo_coeff).ndim==2:
        dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
        dm0.mo_occ = mo_occ
    make_rho = ni._gen_rho_evaluator(mf.mol, dm0, hermi=0, with_lapl=False)[0]

    fxc_abs = []
    for ao, mask, weight, coords \
            in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv, max_memory):
        rhoa = make_rho(0, ao, mask, xctype)
        rhob = make_rho(1, ao, mask, xctype)
        if xctype == 'LDA':
            rho = (rhoa,rhob)
        else: # GGA
            rha = np.zeros_like(rhoa)
            rhb = np.zeros_like(rhob)
            rha[0] = rhoa[0]
            rhb[0] = rhob[0]
            rho = (rha,rhb)
            rhoa = rhoa[0]
            rhob = rhob[0]
        vxc= ni.eval_xc_eff(mf.xc, rho, deriv=1, xctype=xctype)[1] # 
        vxc_a = vxc[0,0]*weight
        vxc_b = vxc[1,0]*weight
        fxc_ab = (vxc_a-vxc_b)/(np.array(rhoa)-np.array(rhob)+1e-9)
        fxc_abs += list(fxc_ab)

    fxc_abs = np.asarray(fxc_abs)
    return fxc_abs

def nr_uks_fxc_sf_tda(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,vxc=None, extype=0, max_memory=2000, verbose=None):
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if hermi != 1 and dtype != np.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)

    nao = dms.shape[-1]
    make_rhosf, nset = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)[:2]

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            p0, p1 = p1, p1 + weight.size
            _vxc = vxc[p0:p1]
            for i in range(nset):
                rho1sf = make_rhosf(i, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = rho1sf * _vxc 
                else:
                    rhosf = np.zeros_like(rho1sf)
                    rhosf[0] += rho1sf[0]
                    wv = rhosf * _vxc
                yield i, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    vmat = np.zeros((nset,nao,nao))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])
    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5
            aow = _scale_ao_sparse(ao, wv, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])

        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(nset,nao,nao)

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5
            wv[4] *= .5
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])

        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(nset,nao,nao)
        vmat += v1

    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[:,0]
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    return vmat

class SA_SF_TDA():
    def __init__(self,mf,SA=0,davidson=False):
        """SA=0: SF-TDA
           SA=1: only add diagonal block for dA
           SA=2: add all dA except for OO block
           SA=3: full dA
        """

        if np.array(mf.mo_coeff).ndim==3:# UKS
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = mf.mo_occ
            self.type_u = True
        else: # ROKS
            self.mo_energy = np.array([mf.mo_energy, mf.mo_energy])
            self.mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            self.mo_occ = np.zeros((2,len(mf.mo_coeff)))
            self.mo_occ[0][np.where(mf.mo_occ>=1)[0]]=1
            self.mo_occ[1][np.where(mf.mo_occ>=2)[0]]=1
            self.type_u = False

        self.mol = mf.mol
        self.nao = self.mol.nao_nr()
        self.mf = mf
        self.SA = SA
        self.davidson=davidson

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
        
    def get_a2b(self):
        
        mf = self.mf
        a_a2b = np.zeros((self.nocc_a,self.nvir_b,self.nocc_a,self.nvir_b))
        dm = self.mf.make_rdm1()
        vhf = self.mf.get_veff(self.mf.mol, dm)
        h1e = self.mf.get_hcore()
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fockA = self.mo_coeff[0].T @ focka @ self.mo_coeff[0]
        fockB = self.mo_coeff[1].T @ fockb @ self.mo_coeff[1]    
        
        try:
            xctype = self.mf.xc
        except:
            xctype = None
            eri_mo_a2b = ao2mo.general(self.mol, [self.orbo_a,self.orbo_a,self.orbv_b,self.orbv_b], compact=False)
            eri_mo_a2b = eri_mo_a2b.reshape(self.nocc_a,self.nocc_a,self.nvir_b,self.nvir_b)
            a_a2b -= np.einsum('ijba->iajb', eri_mo_a2b)
            
            
        if xctype is not None:
            ni = self.mf._numint
            xctype = ni._xc_type(self.mf.xc)
            ni.libxc.test_deriv_order(self.mf.xc, 2, raise_error=True)
            if self.mf.nlc or ni.libxc.is_nlc(self.mf.xc):
                logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                            'deriviative is not available. Its contribution is '
                            'not included in the response function.')
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.mol.spin)
            print('omega alpha hyb',omega, alpha, hyb)
            if hyb != 0:
                eri_mo_a2b = ao2mo.general(self.mol, [self.orbo_a,self.orbo_a,self.orbv_b,self.orbv_b], compact=False)
                eri_mo_a2b = eri_mo_a2b.reshape(self.nocc_a,self.nocc_a,self.nvir_b,self.nvir_b)
                a_a2b -= np.einsum('ijba->iajb', eri_mo_a2b) * hyb
                
                
            print(xctype)
            dm0 = self.mf.make_rdm1()
            if np.array(mf.mo_coeff).ndim==2:
                dm0.mo_coeff = (mf.mo_coeff, self.mf.mo_coeff)
                dm0.mo_occ = self.mo_occ
            make_rho = ni._gen_rho_evaluator(mf.mol, dm0, hermi=1, with_lapl=False)[0]
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)
        
        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mf.mol, mf.grids, self.nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                _, vxc, _, _ = ni.eval_xc_eff(mf.xc, rho, deriv=1, omega=omega, xctype=xctype) # vxc.shape[2,1,N]
                vxc_a = vxc[0,0]*weight
                vxc_b = vxc[1,0]*weight
                fxc_ab = (vxc_a-vxc_b)/(rho0a-rho0b+1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao, self.orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, self.orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, self.orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, self.orbv_b)
                rho_ov_b2a = np.einsum('ri,ra->ria', rho_o_b, rho_v_a)
                rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                w_ov = np.einsum('ria,r->ria', rho_ov_a2b, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                a_a2b += iajb
     
                
        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                 in ni.block_loop(mf.mol, mf.grids, self.nao, ao_deriv, max_memory):#ao(4,N,nao):AO values and derivatives in x,y,z compoents in grids
                rho0a = make_rho(0, ao, mask, xctype)#(4,N):density and "density derivatives" for x,y,z components in grids
                rho0b = make_rho(1, ao, mask, xctype)
                rha = np.zeros_like(rho0a)
                rha[0] = rho0a[0]
                rhb = np.zeros_like(rho0b)
                rhb[0] = rho0b[0]
                _, vxc, _, _ = ni.eval_xc_eff(mf.xc, (rha, rhb), deriv=1, omega=omega)#vxc.shape(2,4,N)
                vxc_a = vxc[0,0]*weight #first order derivatives about \rho in \alpha
                vxc_b = vxc[1,0]*weight #\beta
                fxc_ab = (vxc_a-vxc_b)/(rho0a[0]-rho0b[0]+1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao[0], self.orbo_a) # (N,i)
                rho_v_a = lib.einsum('rp,pi->ri', ao[0], self.orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao[0], self.orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao[0], self.orbv_b)
                rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                w_ov = np.einsum('ria,r->ria', rho_ov_a2b, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                a_a2b += iajb
                
        return a_a2b
        
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
        sf_tda = SF_TDA(mf)
        Amat = np.zeros_like(sf_tda.A)
        
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
        Amat = sf_tda.A + fglobal*Amat
        
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

    def analyse(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        Ds = []
        syms = []

        for nstate in range(self.nstates):
            m_excited = 0.
            orb1 = None
            orb2 = None
            value = self.v[:,nstate]
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
                    orb2 = v+self.nv
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
            syms.append(sym)

            if self.SA == 0 and not self.type_u:
                Dp_ab = 0.
                Dp_ab += sum(sum(x_cv_ab*x_cv_ab)) -sum(sum(x_oo_ab*x_oo_ab))
                for i in range(no):
                    for j in range(no):
                        Dp_ab += x_oo_ab[i,i]*x_oo_ab[j,j]
                ds2 = -self.no+1+Dp_ab
                print(f'Excited state {nstate+1} {self.e[nstate]*27.21138505:10.5f} eV {self.e[nstate]+self.mf.e_tot:11.8f} Hartree D<S^2>={ds2:3.2f} {sym}')
                Ds.append(ds2)
            else:
                print(f'Excited state {nstate+1} {self.e[nstate]*27.21138505:10.5f} eV {self.e[nstate]+self.mf.e_tot:11.8f} Hartree {sym}')


            print('  ')
        return Ds,syms
    
    def recoder(self): # for HF molecule in 6-31G
        print("Return first three excited vectors(CO(3->5 4->5) OO(6->5) or OO(5->5 6->6) configurations).")
        dim1 = self.nc*self.nv
        dim2 = dim1 + self.nc*self.no
        dim3 = dim2 + self.no*self.nv
        vectors = np.zeros((4,5))
        for nstate in range(4):
            value = self.v[:,nstate]
            #x_cv_ab = value[:dim1].reshape(nc,nv)
            x_co_ab = value[dim1:dim2].reshape(self.nc,self.no)
            #x_ov_ab = value[dim2:dim3].reshape(no,nv)
            if self.re:
                x_oo_ab = (self.vects @ value[dim3:].reshape(-1,1)).reshape(self.no,self.no)
            else:
                x_oo_ab = value[dim3:].reshape(self.no,self.no)
            for o,v in zip(* np.where(abs(x_co_ab)>0.1)):  
                vectors[nstate,o-self.no] = x_co_ab[o,v]
            for o,v in zip(* np.where(abs(x_oo_ab)>0.1)):
                if o == v:
                    if o+self.nc+1 == 5:
                        vectors[nstate,-2] = x_oo_ab[o,v]
                    else:
                        vectors[nstate,-1] = x_oo_ab[o,v]
                else:
                    vectors[nstate,-3] = x_oo_ab[o,v]
                    
        return vectors
    
    def gen_response_sf(self,hermi=0,max_memory=None,hf_correction=False):
        mf = self.mf
        mo_energy,mo_occ,mo_coeff = mf_info(mf)
        mol = mf.mol
        if isinstance(mf, scf.hf.KohnShamDFT) and not hf_correction:
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
            hybrid = ni.libxc.is_hybrid_xc(mf.xc)
            if max_memory is None:
                mem_now = lib.current_memory()[0]
                max_memory = max(2000, mf.max_memory*.8-mem_now)

            vxc = cache_xc_kernel_sf(mf, mo_coeff, mo_occ,1,max_memory) # XC kerkel 
            dm0 = None

            def vind(dm1):
                v1 = nr_uks_fxc_sf_tda(ni,mol, mf.grids, mf.xc, dm0, dm1, 0, hermi, # XC * dm1
                                        vxc, max_memory=max_memory)

                if not hybrid:
                    # No with_j because = 0 in spin flip part.
                    pass
                else:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 -= vk
                return v1
        if not isinstance(mf, scf.hf.KohnShamDFT) and not hf_correction : # in HF case
            def vind(dm1):
                return -mf.get_k(mol,dm1,hermi=hermi)
        if hf_correction: # for \Delta A
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
        hdiag = e_ia.ravel()

        vresp = self.gen_response_sf(hermi=0)
        if self.SA > 0:
            vresp_hf = self.gen_response_sf(hermi=0,hf_correction=True)
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

        def vind(zs0): # vector-matrix product for indexed operations
            ndim0,ndim1 = ndim # ndom0:numuber of alpha orbitals, ndim1:number of beta orbitals
            orbo,orbv = orbov # mo_coeff for alpha and beta

            #zs = zs[:,:ndim0*ndim1].reshape(-1,ndim0,ndim1) # (-1,nocca,nvirb)
            #zs = zs.reshape(-1,ndim0,ndim1) 
            zs = np.asarray(zs0).reshape(-1,ndim0,ndim1) # (-1,nocca,nvirb)
            #if self.re:
            #    oo = zs[:,nc:,:no]
            #    new_oo = self.vects@oo
            #    zs = np.einsum('nm,x->x',self.vects,zs)
            vs = np.zeros_like(zs)
            dmov = lib.einsum('xov,qv,po->xpq', zs,orbv.conj(), orbo) # (x,nmo,nmo)
            v1ao = vresp(np.asarray(dmov))   # with density and get response function
            #print('v1ao.shape ',v1ao.shape)
            vs += lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv) # (-1,nocca,nvirb)
            #np.save('v1_t.npy',v1[:,nc:,no:])
            #vs += np.einsum('ij,ab,xjb->xia',delta_ij,fockB[noccb:,noccb:],zs)-\
            #       np.einsum('ab,ij,xjb->xia',delta_ab,fockA[:nocca,:nocca],zs)
            vs += np.einsum('ab,xib->xia',fockB[noccb:,noccb:],zs)-\
                   np.einsum('ij,xja->xia',fockA[:nocca,:nocca],zs)
            vs_dA = np.zeros_like(vs)

            if self.SA > 0:
        
                cv1 = zs[:,:nc,no:]
                co1 = zs[:,:nc,:no]
                ov1 = zs[:,nc:,no:]
                oo1 = zs[:,nc:,:no]
                cv1_mo = np.einsum('xov,qv,po->xpq', cv1, orbvb[:,no:].conj(), orboa[:,:nc]) # (-1,nmo,nmo)
                co1_mo = np.einsum('xov,qv,po->xpq', co1, orbvb[:,:no].conj(), orboa[:,:nc]) # (-1,nmo,nmo)
                ov1_mo = np.einsum('xov,qv,po->xpq', ov1, orbvb[:,no:].conj(), orboa[:,nc:nc+no])
                oo1_mo = np.einsum('xov,qv,po->xpq', oo1, orbvb[:,:no].conj(), orboa[:,nc:nc+no])
                v1ao_cv1_j,v1ao_cv1_k = vresp_hf(np.asarray(cv1_mo)) # (-1,nmo,nmo)
                v1ao_co1_j,v1ao_co1_k = vresp_hf(np.asarray(co1_mo))
                v1ao_ov1_j,v1ao_ov1_k = vresp_hf(np.asarray(ov1_mo))
                v1ao_oo1_j,v1ao_oo1_k = vresp_hf(np.asarray(oo1_mo))
                v1_cv1_j = np.einsum('xpq,po,qv->xov',v1ao_cv1_j,orbo.conj(), orbv) # (-1,nocca,nvirb)
                v1_co1_j = np.einsum('xpq,po,qv->xov',v1ao_co1_j,orbo.conj(), orbv)
                v1_ov1_j = np.einsum('xpq,po,qv->xov',v1ao_ov1_j,orbo.conj(), orbv)
                v1_oo1_j = np.einsum('xpq,po,qv->xov',v1ao_oo1_j,orbo.conj(), orbv)
                v1_cv1_k = np.einsum('xpq,po,qv->xov',v1ao_cv1_k,orbo.conj(), orbv) # (-1,nocca,nvirb)
                v1_co1_k = np.einsum('xpq,po,qv->xov',v1ao_co1_k,orbo.conj(), orbv)
                v1_ov1_k = np.einsum('xpq,po,qv->xov',v1ao_ov1_k,orbo.conj(), orbv)
                v1_oo1_k = np.einsum('xpq,po,qv->xov',v1ao_oo1_k,orbo.conj(), orbv)

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
            return hx
        return vind, hdiag

    
    def davison_process(self,foo,fglobal):
        print("Davison process...")
        vind, hdiag = self.gen_tda_operation_sf(foo,fglobal)
        precond = hdiag
        x0 = init_guess(self.mf, self.nstates)
        converged, e, x1 = lib.davidson1(vind, x0, precond,
                              tol=1e-9,
                              nroots=self.nstates,
                              max_cycle=300)
        self.converged = converged
        self.e = e
        self.v = np.array(x1).T
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
        self.re = remove
        self.nstates = nstates
        if fglobal is None:
            fglobal = (1-d_lda)*self.hyb + d_lda
        if remove:
            if self.davidson:
                self.davison_process(foo=foo,fglobal=fglobal)
            else:
                self.A = self.get_Amat(foo=foo,fglobal=fglobal)
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
                self.davison_process(foo=foo,fglobal=fglobal)
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
        return self.e*27.21138505, self.v
