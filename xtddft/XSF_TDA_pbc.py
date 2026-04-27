"""Spin-adapted XSF-TDA for Gamma-point PBC references.

The main target is a Gamma-point PBC ROKS reference, matching the molecular
``XSF_TDA.py`` spin-adapted path.  UKS references are accepted for compatibility
and fall back to the SF-TDA-like ``SA=0`` default, as in the molecular code.
"""

from pyscf import lib
import numpy as np
import time
import math
import scipy
from functools import reduce
#import sys
#sys.path.append('/home/lenovo2/usrs/zhw/TDDFT')

#from line_profiler import profile

from .SF_TDA_pbc import (
    SF_TDA_down,
    _ewald_exxdiv_mo_tensor,
    _gen_uhf_tda_response_sf,
    _get_gamma_kpt,
    _get_k_gamma,
    _needs_ewald_exxdiv,
    gen_response_sf,
    get_ab_sf,
    mf_info,
)
#import sys
#sys.path.append('/home/lenovo2/usrs/zhw/software/test_git_file')
#from mc_file import _gen_uhf_tda_response_sf
au2ev = 27.21138505

def find_top10_abs_numpy(matrix):
    """
    使用NumPy找到矩阵中绝对值最大的10个元素及其坐标
    
    参数:
    matrix: numpy数组，形状为(n, m)
    
    返回:
    包含(值, 行索引, 列索引)的列表，按绝对值从大到小排序
    """
    # 计算绝对值矩阵
    abs_matrix = abs(matrix)
    
    # 在绝对值矩阵中找最大的10个元素的索引
    flat_indices = np.argpartition(abs_matrix, -10, axis=None)[-10:]
    #print(flat_indices)
    
    # 按绝对值大小排序这些索引
    flat_indices = flat_indices[np.argsort(-abs_matrix.ravel()[flat_indices])]
    
    # 转换为一维索引为二维坐标
    rows, cols = np.unravel_index(flat_indices, matrix.shape)
    
    # 创建结果列表（使用原始值，不是绝对值）
    results = [(matrix[r, c], r, c) for r, c in zip(rows, cols)]
    
    return results

def _make_spin_dm(mo_coeff, mo_occ):
    """Build alpha/beta AO density matrices from normalized MO data."""
    return np.asarray([
        (mo_coeff[spin] * mo_occ[spin]) @ mo_coeff[spin].conj().T
        for spin in range(2)
    ])


def _get_hcore_gamma(mf):
    """Return the one-electron Hamiltonian for the Gamma-point PBC object."""
    try:
        return mf.get_hcore(mf.cell, kpt=_get_gamma_kpt(mf))
    except TypeError:
        return mf.get_hcore()


def _get_veff_gamma(mf, dm):
    """Return the effective potential for a Gamma-point PBC density."""
    try:
        return mf.get_veff(mf.cell, dm, kpt=_get_gamma_kpt(mf))
    except TypeError:
        return mf.get_veff(mf.cell, dm)


def _as_spin_potential(vhf):
    """Normalize scalar/spin potentials to an alpha/beta array."""
    vhf = np.asarray(vhf)
    if vhf.ndim == 2:
        vhf = np.asarray([vhf, vhf])
    return vhf


def _get_jk_gamma(mf, dm, hermi=0):
    """Return Coulomb and exchange matrices for Gamma-point PBC density matrices."""
    try:
        return mf.get_jk(mf.cell, dm, hermi=hermi, kpt=_get_gamma_kpt(mf))
    except TypeError:
        return mf.get_jk(mf.cell, dm, hermi=hermi)


def _get_hf_fock_mo_gamma(mf, mo_coeff, mo_occ):
    """Build unscaled HF alpha/beta Fock matrices in the MO basis."""
    dm = _make_spin_dm(mo_coeff, mo_occ)
    vj, vk = _get_jk_gamma(mf, dm, hermi=1)
    vj = np.asarray(vj)
    vk = np.asarray(vk)
    coul = vj if vj.ndim == 2 else vj[0] + vj[1]
    if vk.ndim == 2:
        vk = np.asarray([vk, vk])
    vhf = coul - vk
    h1e = _get_hcore_gamma(mf)
    focka = h1e + vhf[0]
    fockb = h1e + vhf[1]
    return (
        mo_coeff[0].conj().T @ focka @ mo_coeff[0],
        mo_coeff[1].conj().T @ fockb @ mo_coeff[1],
    )


def _ao2mo_full_gamma(mf, mo_coeffs, shape):
    """Transform bare Gamma-point PBC two-electron integrals to the MO basis."""
    return mf.with_df.ao2mo(mo_coeffs, _get_gamma_kpt(mf), compact=False).reshape(shape)


def _add_exchange_ewald_gamma(mf, eri, mo_coeffs, shape):
    """Add the Gamma G=0 correction used by PBC exchange/K contractions."""
    if _needs_ewald_exxdiv(mf):
        eri = eri.copy()
        eri += _ewald_exxdiv_mo_tensor(mf, mo_coeffs, shape)
    return eri


class XSF_TDA_pbc():
    def __init__(self,mf,SA=None,davidson=True,method=0,collinear_samples=60,calculate_sp=False):
        """Gamma-point PBC XSF-TDA solver, primarily for ROKS references.

           SA=0: SF-TDA
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
            if SA is None:
                self.SA = 0
            else:
                self.SA = SA
        else: # ROKS
            self.mo_energy = np.array([mf.mo_energy, mf.mo_energy])
            self.mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            self.mo_occ = np.zeros((2,len(mf.mo_coeff)))
            self.mo_occ[0][np.where(mf.mo_occ>=1)[0]]=1
            self.mo_occ[1][np.where(mf.mo_occ>=2)[0]]=1
            self.type_u = False
            if SA is None:
                self.SA = 3
            else:
                self.SA = SA

        _get_gamma_kpt(mf)
        self.cell = mf.cell
        self.nao = self.cell.nao_nr()
        self.mf = mf
        self.collinear_samples=collinear_samples
        self.davidson=davidson
        self.method = method
        try:
            _,dsp1 = mf.spin_square()
        except AttributeError:
            dsp1 = self.cell.spin + 1
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
            self.omega, self.alpha, self.hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.cell.spin)
            print('Omega, alpha, hyb',self.omega, self.alpha, self.hyb)
        except: # HF
            xctype = None
            self.omega = 0
            self.hyb = 1.0
        self.calculate_sp = calculate_sp
        if calculate_sp: # # J. Chem. Theory Comput.2023,19,7606−7616
            self.get_sp()

    def _default_fglobal(self, d_lda=0.3, fit=True):
        """Return the spin-adaptation scale used by the kernel path."""
        if self.omega == 0:
            cx = self.hyb
        else:
            cx = self.hyb + (self.alpha-self.hyb)*math.erf(self.omega)
        fglobal = (1-d_lda)*cx + d_lda
        if self.method == 1 and fit:
            fglobal = fglobal*4*(cx-0.5)**2
        return fglobal
            
    def get_sp(self): # calculate spin polarization( J. Chem. Theory Comput.2023,19,7606−7616)
        
        if self.method == 1:
            vresp = _gen_uhf_tda_response_sf(self.mf,hermi=0,collinear_samples=self.collinear_samples)
        else:
            vresp = gen_response_sf(self.mf,hermi=0,method=self.method)
        h = self.mf.mo_coeff[:,self.nc:self.nc+1]# triplet reference
        dm1 = h @ h.T
        h_ao = vresp(np.array(dm1).reshape((1,dm1.shape[0],dm1.shape[1])))
        h_mo = np.array(self.mf.mo_coeff.T @ h_ao @ self.mf.mo_coeff).reshape((dm1.shape[0],dm1.shape[1]))
        lhhl = h_mo[self.nc+self.no,self.nc+self.no]
        # <iH|Ha> for triplet reference
        k_ao = _get_k_gamma(self.mf, dm1)
        k_mo = self.mf.mo_coeff.T @ k_ao @ self.mf.mo_coeff
        homo = np.array(k_mo[:self.nc,self.nc+self.no:]).reshape((self.nc,self.nv))
        # <iL|La>
        l = self.mf.mo_coeff[:,self.nc+1:self.nc+2] # triplet reference
        dm1 = l @ l.T
        l_ao = _get_k_gamma(self.mf, dm1)
        l_mo = self.mf.mo_coeff.T @ l_ao @ self.mf.mo_coeff
        lumo = np.array(l_mo[:self.nc,self.nc+self.no:]).reshape((self.nc,self.nv))
        homo_res = find_top10_abs_numpy(homo) # find top10 value as (values,ind_i,ind_a)
        lumo_res = find_top10_abs_numpy(lumo)
        print('=================================================')
        print(f'<LH|HL> is {lhhl:9.6f}')
        print('Top 10 value in <iH|Ha>:')
        for i in range(10):
            print(f'{i+1}  {homo_res[i][0]:9.6f}, CV is {homo_res[i][1]+1,homo_res[i][2]+self.nc+self.no+1}')
        print('Top 10 value in <iL|La>:')
        for i in range(10):
            print(f'{i+1} {lumo_res[i][0]:9.6f}, CV is {lumo_res[i][1]+1,lumo_res[i][2]+self.nc+self.no+1}')
        homo_lumo = homo - lumo
        homo_lumo_res = find_top10_abs_numpy(homo_lumo)
        print('Top 10 value in <iH|Ha>-<iL|La>:')
        for i in range(10):
            print(f'{i+1} {homo_lumo_res[i][0]:9.6f}, CV is {homo_lumo_res[i][1]+1,homo_lumo_res[i][2]+self.nc+self.no+1}, <iH|Ha> is {homo[homo_lumo_res[i][1],homo_lumo_res[i][2]]:9.6f}, <iL|La> is {lumo[homo_lumo_res[i][1],homo_lumo_res[i][2]]:9.6f}, <iH|Ha>*<iL|La> is {homo[homo_lumo_res[i][1],homo_lumo_res[i][2]] * lumo[homo_lumo_res[i][1],homo_lumo_res[i][2]]:9.6f}')
        homo_lumo = homo*lumo
        homo_lumo_res = find_top10_abs_numpy(homo_lumo)
        print('Top 10 value in <iH|Ha>*<iL|La>:')
        for i in range(10):
            print(f'{i+1} {homo_lumo_res[i][0]:9.6f}, CV is {homo_lumo_res[i][1]+1,homo_lumo_res[i][2]+self.nc+self.no+1}, <iH|Ha> is {homo[homo_lumo_res[i][1],homo_lumo_res[i][2]]:9.6f}, <iL|La> is {lumo[homo_lumo_res[i][1],homo_lumo_res[i][2]]:9.6f}, <iH|Ha>*<iL|La> is {homo[homo_lumo_res[i][1],homo_lumo_res[i][2]] * lumo[homo_lumo_res[i][1],homo_lumo_res[i][2]]:9.6f}')
        #print(f'Sum of <iH|Ha>-<iL|La> is: {np.sum(homo_lumo):9.6f}')
        #print(f'Sum of |<iH|Ha>-<iL|La>| is: {np.sum(abs(homo_lumo)):9.6f}')
        #print(f'Sum of (<iH|Ha>-<iL|La>)**2 is: {np.sum(homo_lumo**2):9.6f}')
        print('=================================================')
        
        
    def get_Amat(self,SA=None,foo=1.0,fglobal=None,projected=None,d_lda=0.3,fit=True):
        """SA=0: SF-TDA
           SA=1: only add diagonal block for dA
           SA=2: add all dA except for OO block
           SA=3: full dA

           projected=True returns the ROKS spin-projected matrix used by
           kernel(davidson=False) and the Davidson path.  By default, ROKS
           references return the projected matrix; pass projected=False to
           inspect the full CV|CO|OV|OO matrix before spin projection.
        """
        mf = self.mf
        if SA is None:
            SA = self.SA
        if projected is None:
            projected = np.array(mf.mo_coeff).ndim != 3
        if fglobal is None:
            fglobal = self._default_fglobal(d_lda=d_lda, fit=fit)
        self.fglobal = fglobal
        #a_a2b = self.get_a2b()
        nc = self.nc
        nv = self.nv
        no = self.no
        nao = nc+nv+no
        dm = self.mf.make_rdm1()
        vhf = _as_spin_potential(_get_veff_gamma(self.mf, dm))
        h1e = _get_hcore_gamma(self.mf)
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fockA = self.mo_coeff[0].conj().T @ focka @ self.mo_coeff[0]
        fockB = self.mo_coeff[1].conj().T @ fockb @ self.mo_coeff[1]
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
        if self.method == 0:
            sf_tda = SF_TDA_down(mf,method=self.method)
            sf_tda_A = sf_tda.get_Amat() 
        elif self.method == 1:
            sf_tda_A = get_ab_sf(mf).reshape(((nc+no)*(no+nv),(nc+no)*(no+nv)))
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
            self.A = sf_tda.A
            return self.A
        fockA_hf, fockB_hf = _get_hf_fock_mo_gamma(mf, self.mo_coeff, self.mo_occ)
        fockS = (fockB_hf-fockA_hf)/2
        fockS_C = fockS[:nc,:nc]
        fockS_O = fockS[nc:nc+no,nc:nc+no]
        fockS_V = fockS[nc+no:,nc+no:]
        fockS_CV = fockS[:nc,nc+no:]
        fockS_CO = fockS[:nc,nc:nc+no]
        fockS_OV = fockS[nc:nc+no,nc+no:]
        
        
        mo4 = [mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff]
        eri = _ao2mo_full_gamma(mf, mo4, (nao, nao, nao, nao))
        eri_k = _add_exchange_ewald_gamma(mf, eri, mo4, (nao, nao, nao, nao))
            
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
                                     np.einsum('avji-> iajv', eri_k[nc+no:,nc:nc+no,:nc,:nc])).reshape(nv*nc,no*nc)
            Amat[:dim1,dim1:dim2] += tmp_CV_CO
            Amat[dim1:dim2,:dim1] += tmp_CV_CO.T
        # CV-OV, OV-CV iaub
            tmp_CV_OV = (np.sqrt(1+1/(2*si))-1)*(- np.einsum('iv,ab->iavb',fockA_hf[:nc,nc:nc+no],iden_V) \
                                                -np.einsum('abvi->iavb',eri_k[nc+no:,nc+no:,nc:nc+no,:nc])).reshape(nv*nc,nv*no)
            Amat[:dim1,dim2:dim3] += tmp_CV_OV
            Amat[dim2:dim3,:dim1] += tmp_CV_OV.T
        # CO-OV, OV-CO iuvb
            tmp_CO_OV = (1/(2*si-1))*(np.einsum('uivb->iuvb',eri[nc:nc+no,:nc,nc:nc+no,nc+no:])-
                                     np.einsum('ubvi->iuvb',eri_k[nc:nc+no,nc+no:,nc:nc+no,:nc])).reshape(no*nc,nv*no)
            Amat[dim1:dim2,dim2:dim3] += tmp_CO_OV
            Amat[dim2:dim3,dim1:dim2] += tmp_CO_OV.T
        
        #--- blocks involving OO ---  
        # CV-OO, OO-CV iawv
        factor = np.sqrt((2*si+1)/(2*si-1))
        if SA > 2:
            tmp_CV_OO = -(factor-1)*np.einsum('avwi->iawv',eri_k[nc+no:,nc:nc+no,nc:nc+no,:nc]).reshape(nv*nc,no*no) + \
                         (1/si)*factor*np.einsum('ia,wv->iawv',fockS_CV,iden_O).reshape(nv*nc,no*no)
            Amat[:dim1,dim3:] += foo*tmp_CV_OO
            Amat[dim3:,:dim1] += foo*tmp_CV_OO.T
        
        # CO-OO, OO-CO  iuvw
            tmp_CO_OO = (np.sqrt(2*si/(2*si-1))-1)*(- np.einsum('wi,uv->iuwv',fockA_hf[nc:nc+no,:nc],iden_O).reshape(no*nc,no*no)\
                                                    -np.einsum('uvwi->iuwv',eri_k[nc:nc+no,nc:nc+no,nc:nc+no,:nc]).reshape(no*nc,no*no))\
                         +(1/np.sqrt(2*si*(2*si-1)))*np.einsum('iu,wv->iuwv',fockB_hf[:nc,nc:nc+no],iden_O).reshape(no*nc,no*no)
            Amat[dim1:dim2,dim3:] += foo*tmp_CO_OO
            Amat[dim3:,dim1:dim2] += foo*tmp_CO_OO.T
        # OV-OO, OO-OV uawv
            tmp_OV_OO = (np.sqrt(2*si/(2*si-1))-1)*(np.einsum('wu,av->uawv',iden_O,fockB_hf[nc+no:,nc:nc+no]).reshape(nv*no,no*no) \
                                                  - np.einsum('avwu->uawv',eri_k[nc+no:,nc:nc+no,nc:nc+no,nc:nc+no]).reshape(nv*no,no*no)) \
                         -(1/np.sqrt(2*si*(2*si-1)))*np.einsum('ua,wv->uawv',fockA_hf[nc:nc+no,nc+no:],iden_O).reshape(nv*no,no*no)
            Amat[dim2:dim3,dim3:] += foo*tmp_OV_OO
            Amat[dim3:,dim2:dim3] += foo*tmp_OV_OO.T

        # Add global scale
        self.A = sf_tda_A + fglobal*Amat
        if projected:
            if np.array(mf.mo_coeff).ndim == 3:
                return self.A
            self.vects = self.get_vect()
            return self.remove()

        return self.A
    
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
        try:
            ovlp = self.mf.get_ovlp(self.cell, kpt=_get_gamma_kpt(self.mf))
        except TypeError:
            ovlp = self.mf.get_ovlp()
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
    
    def analyse(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        Ds = []

        for nstate in range(self.nstates):
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
            #print('norm self.v',np.linalg.norm(value))
            #tmp_v1 = np.concatenate((x_cv_ab, x_co_ab),axis=1)
            #tmp_v2 = np.concatenate((x_ov_ab, x_oo_ab),axis=1)
            #tmp_v = np.concatenate((tmp_v1,tmp_v2),axis=0)
            #print('norm v',np.linalg.norm(tmp_v))
            

            for o,v in zip(* np.where(abs(x_cv_ab)>0.05)):
                print(f'{100*x_cv_ab[o,v]**2:5.2f}% CV(ab) {o+1}a -> {v+1+self.nc+self.no}b {x_cv_ab[o,v]:10.5f} ')
            for o,v in zip(* np.where(abs(x_co_ab)>0.05)):
                print(f'{100*x_co_ab[o,v]**2:5.2f}% CO(ab) {o+1}a -> {v+1+self.nc}b {x_co_ab[o,v]:10.5f} ')
            for o,v in zip(* np.where(abs(x_ov_ab)>0.05)):
                print(f'{100*x_ov_ab[o,v]**2:5.2f}% OV(ab) {o+self.nc+1}a -> {v+1+self.nc+self.no}b {x_ov_ab[o,v]:10.5f} ')
            for o,v in zip(* np.where(abs(x_oo_ab)>0.05)):
                print(f'{100*x_oo_ab[o,v]**2:5.2f}% OO(ab) {o+nc+1}a -> {v+1+self.nc}b {x_oo_ab[o,v]:10.5f} ')

            if self.SA == 0 and not self.type_u:
                Dp_ab = 0.
                Dp_ab += sum(sum(x_cv_ab*x_cv_ab)) -sum(sum(x_oo_ab*x_oo_ab))
                for i in range(no):
                    for j in range(no):
                        Dp_ab += x_oo_ab[i,i]*x_oo_ab[j,j]
                ds2 = -2*self.ground_s+1+Dp_ab
                print(f'Excited state {nstate+1} {self.e[nstate]*27.21138505:10.5f} eV {self.e[nstate]+self.mf.e_tot:11.8f} Hartree D<S^2>={ds2:3.2f}')
                Ds.append(ds2)
            elif self.type_u:
                P_ab = self.deltaS2_U(nstate)
                #ds2 = P_ab - 2*self.ground_s + 1
                ds2 = P_ab - self.no + 1
                Ds.append(ds2)
                print(f'Excited state {nstate+1} {self.e[nstate]*27.21138505:10.5f} eV {self.e[nstate]+self.mf.e_tot:11.8f} Hartree D<S^2>={ds2:3.2f}')
            else:
                print(f'Excited state {nstate+1} {self.e[nstate]*27.21138505:10.5f} eV {self.e[nstate]+self.mf.e_tot:11.8f} Hartree')

            print(f'CV(1):{100*((x_cv_ab**2).sum()):6.3f}%, CO(1):{100*((x_co_ab**2).sum()):6.3f}%, OV(1):{100*((x_ov_ab**2).sum()):6.3f}%, OO(1):{100*((x_oo_ab**2).sum()):6.3f}%')

            print('========================================')
        return Ds
    
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
        vhf = _as_spin_potential(_get_veff_gamma(mf, dm))
        h1e = _get_hcore_gamma(mf)
        fockA = h1e + vhf[0]
        fockB = h1e + vhf[1]
        fockA = mo_coeff[0].conj().T @ fockA @ mo_coeff[0]
        fockB = mo_coeff[1].conj().T @ fockB @ mo_coeff[1]
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
        #if hf_correction: # for \Delta A
        def vind(dm1):
            vj,vk = _get_jk_gamma(mf,dm1,hermi=hermi)
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
        vhf = _as_spin_potential(_get_veff_gamma(mf, dm))
        h1e = _get_hcore_gamma(mf)
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fockA = mo_coeff[0].conj().T @ focka @ mo_coeff[0]
        fockB = mo_coeff[1].conj().T @ fockb @ mo_coeff[1]

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

        if self.method == 1:
            vresp = _gen_uhf_tda_response_sf(self.mf,hermi=0,collinear_samples=self.collinear_samples)
        else:
            vresp = gen_response_sf(self.mf,hermi=0,method=self.method)

        if self.SA > 0:
            vresp_hf = self.gen_response_sf_delta_A(hermi=0)# to calculate \Delta A
            fockA_hf, fockB_hf = _get_hf_fock_mo_gamma(mf, mo_coeff, mo_occ)
            factor1 = np.sqrt((2*si+1)/(2*si))-1
            factor2 = np.sqrt((2*si+1)/(2*si-1))
            factor3 = np.sqrt((2*si)/(2*si-1))-1
            factor4 = 1/np.sqrt(2*si*(2*si-1))
            
        #@profile
        def vind(zs0,sp=False): # vector-matrix product for indexed operations
            ndim0,ndim1 = ndim # ndom0:numuber of alpha orbitals, ndim1:number of beta orbitals
            orbo,orbv = orbov # mo_coeff for alpha and beta
            #start_t = time.time()
            #print(zs0.shape)

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
            #print('dmov.shape',dmov.shape)
            v1ao = vresp(np.asarray(dmov))   # with density and get response function
            vs += lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv,optimize=True) # (-1,nocca,nvirb)
            vs += np.einsum('ab,xib->xia',fockB[noccb:,noccb:],zs,optimize=True)-\
                   np.einsum('ij,xja->xia',fockA[:nocca,:nocca],zs,optimize=True)
                
            
            vs_dA = np.zeros_like(vs)
            #end_t = time.time()
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
                #print('v1_cv1_k',v1_cv1_k)

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
                #self.sp_hh = v1_oo1_k[:,:nc,no:no+1] # for triplet reference state
                #self.sp_ll = v1_oo1_k[:,:nc,no+1:]
                #print('<iH|Ha> and <iL|La>')
                #print(v1_oo1_k[:,:nc,no:no+1],v1_oo1_k[:,:nc,no+1:])
                #print('v1_oo_k',v1_oo1_k)
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
                
            else:
                new_hx = hx.copy()
            #print('hx ',hx.shape)
            #debug = False
            #if debug:
            #    self.hx = hx
            #    if self.re:
            #        self.debug_hx_dav(hx)
            #    else:
            #        self.debug_hx(hx)
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
                              tol=1e-8,lindep=1e-9,
                              nroots=self.nstates,
                              max_cycle=1000)
        end_time = time.time()
        #print(f'davidson time use {(end_time-end_t)/3600} hours')
        #if self.calculate_sp:
        #    _ = vind(x1,sp=True)
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
            
    def kernel(self, nstates=1,remove=None,frozen=None,foo=1.0,d_lda=0.3,fglobal=None,fit=True):
        if remove is None:
            if np.array(self.mf.mo_coeff).ndim==3: # UKS
                self.re = False
            else:
                self.re = True
        else:
            self.re = remove
        nov = (self.nc+self.no) * (self.no+self.nv)
        self.nstates = min(nstates,nov)
        if fglobal is None:
            fglobal = self._default_fglobal(d_lda=d_lda, fit=fit)
        self.fglobal = fglobal
        if self.re:
            print('fglobal',fglobal)
            if self.davidson:
                self.vects = self.get_vect()
                self.davidson_process(foo=foo,fglobal=fglobal)
            else:
                self.A = self.get_Amat(foo=foo,fglobal=fglobal,projected=True)
                #np.save('diag_A.npy',self.A)
                #print('matrix saved as diag_A.nyp.')
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
        return self.e*27.21138505, self.v



XSF_TDA = XSF_TDA_pbc


if __name__ == '__main__':
    print('Import XSF_TDA_pbc and pass a Gamma-point PySCF PBC mean-field object.')
