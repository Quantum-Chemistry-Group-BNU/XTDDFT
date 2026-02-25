#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "16"
from pyscf import gto
import cupy as cp
import numpy as np
import time

au2ev = 27.21138505

import functools
from pyscf.lib import logger
from pyscf.tdscf import rhf as tdhf_cpu
from gpu4pyscf import dft, scf
from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.tdscf._uhf_resp_sf import nr_uks_fxc_sf
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.dft.numint import eval_rho2
from pyscf.dft import numint as pyscf_numint
from gpu4pyscf.tdscf._uhf_resp_sf import mcfun_eval_xc_adapter_sf

import Davidson
"""this file invoke USF-TDA and try optimize to reduce time costing"""


def __mcfun_fn_eval_xc2(ni, xc_code, xctype, rho, deriv):
    t, s = rho
    if not isinstance(t, cp.ndarray):
        t = cp.asarray(t)
    if not isinstance(s, cp.ndarray):
        s = cp.asarray(s)
    rho = cp.stack([(t + s) * .5, (t - s) * .5])
    spin = 1
    if isinstance(ni, pyscf_numint.NumInt):
        evfk = ni.eval_xc_eff(xc_code, rho.get(), deriv=deriv, xctype=xctype)
    else:
        evfk = ni.eval_xc_eff(xc_code, rho, deriv=deriv, xctype=xctype, spin=spin)
    evfk = list(evfk)
    # for order in range(1, deriv+1):
    #     if evfk[order] is not None:
    #         evfk[order] = xc_deriv_gpu.ud2ts(evfk[order])
    return evfk


def cache_xc_kernel_sf(ni, mol, grids, xc_code, mo_coeff, mo_occ,
                       collinear_samples, deriv=2, collinear='mcol'):
    '''Compute the fxc_sf, which can be used in SF-TDDFT/TDA
    '''
    xctype = ni._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 1
    else:
        ao_deriv = 0
    assert isinstance(mo_coeff, cp.ndarray)
    assert mo_coeff.ndim == 3

    nao = mo_coeff[0].shape[0]
    rhoa = []
    rhob = []

    with_lapl = False
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
        rhoa_slice = eval_rho2(_sorted_mol, ao_mask, mo_coeff[0,idx,:],
                               mo_occ[0], None, xctype, with_lapl)
        rhob_slice = eval_rho2(_sorted_mol, ao_mask, mo_coeff[1,idx,:],
                               mo_occ[1], None, xctype, with_lapl)
        rhoa.append(rhoa_slice)
        rhob.append(rhob_slice)
    rho_ab = (cp.hstack(rhoa), cp.hstack(rhob))
    if collinear == 'alda0':  # GGA ALDA0
        rha = cp.zeros_like(rho_ab[0])
        rhb = cp.zeros_like(rho_ab[1])
        rha[0] = rho_ab[0][0]
        rhb[0] = rho_ab[1][0]
        rho_ab = (rha, rhb)
        rhoa = rho_ab[0][0]
        rhob = rho_ab[1][0]
    rho_z = cp.array([rho_ab[0] + rho_ab[1],
                      rho_ab[0] - rho_ab[1]])
    eval_xc_eff = mcfun_eval_xc_adapter_sf(ni, xc_code, collinear_samples)
    if deriv == 2:
        if collinear == 'alda0':
            fxc = []
            fn_eval_xc = functools.partial(__mcfun_fn_eval_xc2, ni, xc_code, xctype)
            vxc = fn_eval_xc(rho_z, deriv)[1]
            vxc_a = vxc[0, 0] / 2
            vxc_b = vxc[1, 0] / 2
            fxc_ab = (vxc_a - vxc_b) / (cp.array(rhoa) - cp.array(rhob) + 1e-9)
            fxc += list(fxc_ab)
        else:
            vxc, fxc = eval_xc_eff(xc_code, rho_z, deriv=2, xctype=xctype)[1:3]
        return rho_ab, vxc, fxc
    elif deriv == 3:
        raise ValueError('temporary do not support deriv == 3')
        # whether_use_gpu = os.environ.get('LIBXC_ON_GPU', '0') == '1'
        # if whether_use_gpu:
        #     vxc, fxc, kxc = eval_xc_eff(xc_code, rho_z, deriv=3, xctype=xctype)[1:4]
        # else:
        #     ni_cpu = ni.to_cpu()
        #     eval_xc_eff = mcfun_eval_xc_adapter_sf(ni_cpu, xc_code, collinear_samples)
        #     vxc, fxc, kxc = eval_xc_eff(xc_code, rho_z, deriv=3, xctype=xctype)[1:4]
        # return rho_ab, vxc, fxc, kxc


def gen_uhf_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0,
                        collinear='mcol', collinear_samples=200):
    '''Generate a function to compute the product of Spin Flip UKS response function
    and UKS density matrices.
    '''
    # assert isinstance(mf, (uhf.UHF))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    assert hermi == 0

    if isinstance(mf, hf.KohnShamDFT):
        if mf.do_nlc():
            logger.warn(mf, 'NLC functional found in DFT object. Its contribution is '
                        'not included in the TDDFT response function.')

        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if collinear in ('ncol', 'mcol', 'alda0') and mf.xc != 'hf':
            fxc = cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ,
                                     collinear_samples, collinear=collinear)[2]
            if collinear == 'alda0':
                fxc_temp = cp.zeros((4,4,len(fxc)))
                fxc_temp[0, 0, :] = cp.asarray(fxc)
                fxc = fxc_temp
                del fxc_temp
        dm0 = None

        def vind(dm1):
            if collinear in ('ncol', 'mcol', 'alda0') and mf.xc != 'hf':
                v1 = nr_uks_fxc_sf(ni, mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   None, None, fxc)
            else:
                v1 = cp.zeros_like(dm1)
            if hybrid:
                # j = 0 in spin flip part.
                if omega == 0:
                    vk = mf.get_k(mol, dm1, hermi) * hyb
                elif alpha == 0: # LR=0, only SR exchange
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega) * hyb
                elif hyb == 0: # SR=0, only LR exchange
                    vk = mf.get_k(mol, dm1, hermi, omega=omega) * alpha
                else: # SR and LR exchange with different ratios
                    vk = mf.get_k(mol, dm1, hermi) * hyb
                    vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                v1 -= vk
            return v1
        return vind

    else: #HF
        def vind(dm1):
            vk = mf.get_k(mol, dm1, hermi)
            return -vk
        return vind


class SA_SF_TDA():
    def __init__(self, mf, SA=3, davidson=True, collinear='mcol'):
        """SA=0: SF-TDA
           SA=1: only add diagonal block for dA
           SA=2: add all dA except for OO block
           SA=3: full dA
        """
        print('method=0 (default) ALDA0, method=1 multicollinear, method=2 collinear')
        self.level_shift = tdhf_cpu.TDBase.level_shift
        self.conv_tol = tdhf_cpu.TDBase.conv_tol
        self.lindep = tdhf_cpu.TDBase.lindep
        self.max_cycle = tdhf_cpu.TDBase.max_cycle
        if cp.array(mf.mo_coeff).ndim == 3:  # UKS
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = mf.mo_occ
            self.SA = 0
        else:  # ROKS
            self.mo_energy = cp.stack((mf.mo_energy, mf.mo_energy), axis=0)
            self.mo_coeff = cp.stack((mf.mo_coeff, mf.mo_coeff), axis=0)
            self.mo_occ = cp.zeros((2, len(mf.mo_coeff)))
            self.mo_occ[0][cp.where(mf.mo_occ >= 1)[0]] = 1
            self.mo_occ[1][cp.where(mf.mo_occ >= 2)[0]] = 1
            self.SA = SA

        self.mol = mf.mol
        self.nao = self.mol.nao_nr()
        self.mf = mf
        # # use mol.spin calculate
        # _, dsp1 = mf.spin_square()
        # self.ground_s = (dsp1 - 1) / 2

        self.occidx_a = cp.where(self.mo_occ[0] == 1)[0]
        self.viridx_a = cp.where(self.mo_occ[0] == 0)[0]
        self.occidx_b = cp.where(self.mo_occ[1] == 1)[0]
        self.viridx_b = cp.where(self.mo_occ[1] == 0)[0]
        self.orbo_a = self.mo_coeff[0][:, self.occidx_a]
        self.orbv_a = self.mo_coeff[0][:, self.viridx_a]
        self.orbo_b = self.mo_coeff[1][:, self.occidx_b]
        self.orbv_b = self.mo_coeff[1][:, self.viridx_b]
        self.nocc_a = self.orbo_a.shape[1]
        self.nvir_a = self.orbv_a.shape[1]
        self.nocc_b = self.orbo_b.shape[1]
        self.nvir_b = self.orbv_b.shape[1]
        self.nmo_a = self.nocc_a + self.nvir_a
        self.nc = self.nocc_b
        self.nv = self.nvir_a
        self.no = self.nocc_a - self.nocc_b

        try:  # dft
            xctype = self.mf.xc
            ni = self.mf._numint
            _, _, self.hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.mol.spin)
        except:  # HF
            xctype = None
            self.hyb = 1.0

        self.collinear = collinear
        self.collinear_samples = 50
        self.extype = 1  # spin flip down, temporary setting
        self.X = True  # spin adapt

    def get_vect(self):  # construct Vmat N*(N-1)
        tmp_v = cp.zeros((self.no - 1, self.no))  # (self.no-1,self.no)
        for i in range(1, self.no):  # 1->v1 2->v2 3->v3 ...
            factor = 1 / cp.sqrt((self.no - i + 1) * (self.no - i))
            tmp = [self.no - i] + [-1] * (self.no - i)  # (N-i,-1,-1,-1 ...)
            tmp_v[i - 1][i - 1:] = cp.array(tmp) * factor
        self.vect = tmp_v.T  # N(N-1)
        # print('v ',v)
        vects = cp.eye(self.no * self.no)
        vects = vects[:, :-1]  # no*no*(no*no-1)
        index = [0]
        for i in range(1, self.no):
            index.append(i * (self.no + 1))
        # print('index ',index)
        for i in range(self.vect.shape[1]):
            vects[0::self.no + 1, index[i]] = self.vect[:, i]
        # print(vect)
        return vects

    def remove(self):
        # remove sf=si state
        dim3 = self.nc * self.nv + self.nc * self.no + self.no * self.nv
        dim = self.A.shape[0]
        self.vects = self.get_vect()  # (no*no,no*no-1)
        A = cp.zeros((dim - 1, dim - 1))
        A[:dim3, :dim3] = self.A[:dim3, :dim3]
        A[:dim3, dim3:] = self.A[:dim3, dim3:] @ self.vects
        A[dim3:, :dim3] = self.vects.T @ self.A[dim3:, :dim3]
        A[dim3:, dim3:] = self.vects.T @ self.A[dim3:, dim3:] @ self.vects

        return A

    def init_guess(self, mf, nstates):  # only spin down

        # mo_energy, mo_occ, mo_coeff = mf_info(mf)

        occidxa = cp.where(self.mo_occ[0] > 0)[0]
        occidxb = cp.where(self.mo_occ[1] > 0)[0]
        viridxa = cp.where(self.mo_occ[0] == 0)[0]
        viridxb = cp.where(self.mo_occ[1] == 0)[0]
        # e_ia_b2a = (mo_energy[0][viridxa,None] - mo_energy[1][occidxb]).T
        e_ia_a2b = (self.mo_energy[1][viridxb, None] - self.mo_energy[0][occidxa]).T
        # e_ia_a2b = cp.array(list(cv.ravel()) + list(co.ravel())+list(ov.ravel())+list(oo.ravel()))
        no = self.no
        nc = self.nc
        nv = self.nv
        nvir = no + nv

        e_ia_a2b = e_ia_a2b.ravel()
        nov_a2b = e_ia_a2b.size

        nstates = min(nstates, nov_a2b)
        e_threshold = cp.sort(e_ia_a2b)[nstates - 1]
        e_threshold += 1e-5

        # spin-down
        idx = cp.where(e_ia_a2b <= e_threshold)[0]
        x0 = cp.zeros((idx.size, nov_a2b))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        if self.re:
            x0 = x0[:, :-1]
        # print('x0.shape',x0.shape)
        return cp.array(x0)

    # def gen_response_sf_delta_A(self, hermi=0):  # only \Delta A
    #     '''
    #     generate \Delta A * x for spin adapt spin flip TDA
    #     '''
    #     mf = self.mf.copy()
    #     mol = mf.mol
    #     mf = self.mf.copy()
    #     mf.mo_occ = self.mo_occ
    #     mf.mo_coeff = self.mo_coeff
    #     mf.mo_energy = self.mo_energy
    #
    #     mo_coeff = cp.asarray(mf.mo_coeff)
    #     assert mo_coeff[0].dtype == cp.float64
    #     mo_occ = cp.asarray(mf.mo_occ)
    #
    #     # only spin flip down, spin flip up do not spin contamination
    #     occidxa = mo_occ[0] > 0
    #     viridxb = mo_occ[1] == 0
    #     orboa = mo_coeff[0][:, occidxa]
    #     orbvb = mo_coeff[1][:, viridxb]
    #     orbov = (orboa, orbvb)
    #
    #     # if hf_correction: # for \Delta A
    #     def vind(zs, ex_idx, get_j=False):
    #         orbo, orbv = orbov
    #         orbv_idx, orbo_idx = ex_idx
    #         # mo1 = contract('xov,pv->xpo', zs, orbv[:, orbv_idx])
    #         # dms = contract('xpo,qo->xpq', mo1, orbo[:, orbo_idx].conj())
    #         dms = cp.einsum('xov,qv,po->xpq', zs, orbvb[:, orbv_idx].conj(), orboa[:, orbo_idx], optimize=True)
    #         if get_j:
    #             # vjao, vkao = mf.get_jk(mol, dms, hermi=hermi)
    #             vkao = mf.get_k(mol, dms, hermi=hermi)
    #             vjao = mf.get_j(mol, dms, hermi=hermi)
    #             # vjmo = contract('xpq,qo->xpo', vjao, orbo)
    #             # vjmo = contract('xpo,pv->xov', vjmo, orbv.conj())
    #             vjmo = cp.einsum('xpq,po,qv->xov', vjao, orbo.conj(), orbv)
    #         else:
    #             vkao = mf.get_k(mol, dms, hermi=hermi)
    #             vjmo = None
    #         vkmo = cp.einsum('xpq,po,qv->xov', vkao, orbo.conj(), orbv)
    #         # vkmo = contract('xpq,qo->xpo', vkao, orbo)
    #         # vkmo = contract('xpo,pv->xov', vkmo, orbv.conj())
    #         return vjmo, vkmo
    #
    #     return vind

    def gen_response_sf_delta_A(self, hermi=0, max_memory=None):  # only \Delta A
        mf = self.mf
        mol = mf.mol

        # if hf_correction: # for \Delta A
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            return vj, vk

        return vind

    def gen_vind(self):
        '''Generate function to compute A*x for spin-flip TDDFT case.
        '''
        # mf = self._scf
        mf = self.mf
        assert isinstance(mf, scf.hf.SCF)
        if isinstance(mf.mo_coeff, (tuple, list)):
            # The to_gpu() in pyscf is not able to convert SymAdaptedUHF.mo_coeff.
            # In this case, mf.mo_coeff has the type (NPArrayWithTag, NPArrayWithTag).
            # cp.asarray() for this object leads to an error in
            # cupy._core.core._array_from_nested_sequence
            mo_coeff = cp.asarray(mf.mo_coeff[0]), cp.asarray(mf.mo_coeff[1])
        else:
            mo_coeff = cp.asarray(mf.mo_coeff)
        assert mo_coeff[0].dtype == cp.float64
        mo_energy = cp.asarray(mf.mo_energy)
        mo_occ = cp.asarray(mf.mo_occ)
        # nao, nmo = mo_coeff[0].shape

        if self.X:
            mo_energy = cp.stack((mo_energy, mo_energy), axis=0)
            mo_coeff = cp.stack((mo_coeff, mo_coeff), axis=0)
            mo_occ_temp = cp.zeros((2, len(mf.mo_coeff)))
            mo_occ_temp[0][cp.where(mo_occ >= 1)[0]] = 1
            mo_occ_temp[1][cp.where(mo_occ >= 2)[0]] = 1
            mo_occ = mo_occ_temp
            del mo_occ_temp

            # TODO(WHB): Test mf.get_fock()
            dm = self.mf.make_rdm1()
            vhf = self.mf.get_veff(self.mf.mol, dm)
            h1e = self.mf.get_hcore()
            focka_ao = h1e + vhf[0]
            fockb_ao = h1e + vhf[1]

            focka_mo = mo_coeff[0].T @ focka_ao @ mo_coeff[0]
            fockb_mo = mo_coeff[1].T @ fockb_ao @ mo_coeff[1]

        # spin flip up do not have spin comtamination,
        extype = self.extype
        if extype == 0:
            occidxb = mo_occ[1] > 0
            viridxa = mo_occ[0] ==0
            orbob = mo_coeff[1][:,occidxb]
            orbva = mo_coeff[0][:,viridxa]
            orbov = (orbob, orbva)
            e_ia = mo_energy[0][viridxa] - mo_energy[1][occidxb,None]
            hdiag = e_ia.ravel()

        elif extype == 1:
            occidxa = mo_occ[0] > 0
            viridxb = mo_occ[1] ==0
            orboa = mo_coeff[0][:,occidxa]
            orbvb = mo_coeff[1][:,viridxb]
            orbov = (orboa, orbvb)
            e_ia = mo_energy[1][viridxb] - mo_energy[0][occidxa,None]
            hdiag = e_ia.ravel()
        else:
            raise ValueError(
                f"Invalid extype = {extype}. "
                "extype must be 0 (beta->alpha spin flip up) or 1 (alpha->beta spin flip down)."
            )

        vresp = gen_uhf_response_sf(
            mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0, collinear=self.collinear,
            collinear_samples=self.collinear_samples)

        def vind(zs):
            zs = cp.asarray(zs).reshape(-1, *e_ia.shape)
            orbo, orbv = orbov
            mo1 = contract('xov,pv->xpo', zs, orbv)
            dms = contract('xpo,qo->xpq', mo1, orbo.conj())
            dms = tag_array(dms, mo1=mo1, occ_coeff=orbo)
            v1ao = vresp(dms)
            v1mo = contract('xpq,qo->xpo', v1ao, orbo)
            v1mo = contract('xpo,pv->xov', v1mo, orbv.conj())
            if self.X:
                occidxa = cp.where(mo_occ[0] == 1)[0]
                occidxb = cp.where(mo_occ[1] == 1)[0]
                v1mo += (
                    contract('ab,xib->xia', fockb_mo[len(occidxb):, len(occidxb):], zs)
                    - contract('ij,xja->xia', focka_mo[:len(occidxa), :len(occidxa)], zs)
                )
            else:
                v1mo += zs * e_ia
            return v1mo.reshape(len(v1mo), -1)

        return vind, hdiag

    def gen_tda_operation_sf(self, foo, fglobal):
        mf = self.mf.copy()
        mf.mo_occ = self.mo_occ
        mf.mo_coeff = self.mo_coeff
        mf.mo_energy = self.mo_energy

        assert (self.mo_coeff[0].dtype == cp.double)
        nao, nmo = self.mo_coeff[0].shape
        occidxa = cp.where(self.mo_occ[0] == 1)[0]
        occidxb = cp.where(self.mo_occ[1] == 1)[0]
        viridxa = cp.where(self.mo_occ[0] == 0)[0]
        viridxb = cp.where(self.mo_occ[1] == 0)[0]
        nocca = len(occidxa)
        noccb = len(occidxb)
        nvira = len(viridxa)
        nvirb = len(viridxb)
        orboa = self.mo_coeff[0][:, occidxa]
        orbob = self.mo_coeff[1][:, occidxb]
        orbva = self.mo_coeff[0][:, viridxa]
        orbvb = self.mo_coeff[1][:, viridxb]
        nc = noccb
        nv = nvira
        no = nocca - noccb
        nvir = no + nv
        si = no / 2.0
        ndim = (nocca, nvirb)
        orbov = (orboa, orbvb)
        iden_C = cp.eye(nc)
        iden_V = cp.eye(nv)
        iden_O = cp.eye(no)

        # USF-TDA
        vind_A, hdiag = self.gen_vind()

        if self.SA > 0:
            vresp_hf = self.gen_response_sf_delta_A(hermi=0)  # to calculate \Delta A
            hf = scf.ROHF(self.mf.mol)
            dm = self.mf.make_rdm1()
            vhf = hf.get_veff(hf.mol, dm)
            h1e = hf.get_hcore()
            fockA_hf = self.mf.mo_coeff.T @ (h1e + vhf[0]) @ self.mf.mo_coeff
            fockB_hf = self.mf.mo_coeff.T @ (h1e + vhf[1]) @ self.mf.mo_coeff
            factor1 = cp.sqrt((2 * si + 1) / (2 * si)) - 1
            factor2 = cp.sqrt((2 * si + 1) / (2 * si - 1))
            factor3 = cp.sqrt((2 * si) / (2 * si - 1)) - 1
            factor4 = 1 / cp.sqrt(2 * si * (2 * si - 1))

        @profile
        def vind(zs0):  # vector-matrix product for indexed operations
            zs0 = cp.asarray(zs0)
            ndim0, ndim1 = ndim  # ndom0:numuber of alpha orbitals, ndim1:number of beta orbitals
            orbo, orbv = orbov  # mo_coeff for alpha and beta

            if self.re:
                oo = cp.zeros((cp.array(zs0).shape[0], no * no - 1))  # get oo from zs0, which is no*no-1

                for i in range(no - 1):
                    # print('nc*nvir+i*nvir:nc*nvir+no+i*nvir ',nc*nvir+i*nvir,nc*nvir+no+i*nvir)
                    oo[:, i * no:(i + 1) * no] = cp.array(zs0)[:, nc * nvir + i * nvir:nc * nvir + no + i * nvir]
                # print(oo[:,(no-1)*no:].shape, zs0[:,nc*nvir+(no-1)*nvir:nc*nvir+(no-1)*nvir+no-1].shape)
                oo[:, (no - 1) * no:] = cp.array(zs0)[:, nc * nvir + (no - 1) * nvir:nc * nvir + (no - 1) * nvir + no - 1]  # no*no-1
                new_oo = cp.einsum('xy,ny->nx', self.vects, oo)  # we want the whole matrix of oo, which is no*no
                new_zs0 = cp.zeros((cp.array(zs0).shape[0], cp.array(zs0).shape[1] + 1))  # full matrix
                new_zs0[:, :nc * nvir] = cp.array(zs0)[:, :nc * nvir]
                for i in range(no - 1):
                    new_zs0[:, nc*nvir+i*nvir: nc*nvir+no+i*nvir] = new_oo[:, i * no:(i + 1) * no]
                    new_zs0[:, nc*nvir+no+i*nvir: nc*nvir+no+nv+i*nvir] = cp.array(zs0)[:, nc*nvir+no+i*nvir: nc*nvir+no+nv+i*nvir]
                new_zs0[:, nc * nvir + (no - 1) * nvir:nc * nvir + no + (no - 1) * nvir] = new_oo[:, -no:]
                new_zs0[:, nc * nvir + (no - 1) * nvir + no:] = cp.array(zs0)[:, nc * nvir + (no - 1) * nvir + no - 1:]
            else:
                new_zs0 = zs0.copy()
            zs = cp.asarray(new_zs0).reshape(-1, ndim0, ndim1)
            vs = vind_A(zs).reshape(-1, nocca, nvirb)
            vs_dA = cp.zeros_like(vs)

            if self.SA > 0:
                cv1 = zs[:, :nc, no:]
                co1 = zs[:, :nc, :no]
                ov1 = zs[:, nc:, no:]
                oo1 = zs[:, nc:, :no]

                # cv1_mo = cp.einsum('xov,qv,po->xpq', cv1, orbvb[:, no:].conj(), orboa[:, :nc], optimize=True)  # (-1,nmo,nmo)
                # co1_mo = cp.einsum('xov,qv,po->xpq', co1, orbvb[:, :no].conj(), orboa[:, :nc], optimize=True)  # (-1,nmo,nmo)
                # ov1_mo = cp.einsum('xov,qv,po->xpq', ov1, orbvb[:, no:].conj(), orboa[:, nc:nc + no], optimize=True)
                # oo1_mo = cp.einsum('xov,qv,po->xpq', oo1, orbvb[:, :no].conj(), orboa[:, nc:nc + no], optimize=True)
                mo1 = contract('xov,pv->xpo', cv1, orbvb[:, no:])
                cv1_mo = contract('xpo,qo->xpq', mo1, orboa[:, :nc].conj())
                mo1 = contract('xov,pv->xpo', co1, orbvb[:, :no])
                co1_mo = contract('xpo,qo->xpq', mo1, orboa[:, :nc].conj())
                mo1 = contract('xov,pv->xpo', ov1, orbvb[:, no:])
                ov1_mo = contract('xpo,qo->xpq', mo1, orboa[:, nc:].conj())
                mo1 = contract('xov,pv->xpo', oo1, orbvb[:, :no])
                oo1_mo = contract('xpo,qo->xpq', mo1, orboa[:, nc:].conj())
                # _, v1ao_cv1_k = vresp_hf(cp.asarray(cv1_mo))  # (-1,nmo,nmo)
                # v1ao_co1_j, v1ao_co1_k = vresp_hf(cp.asarray(co1_mo))
                # v1ao_ov1_j, v1ao_ov1_k = vresp_hf(cp.asarray(ov1_mo))
                # _, v1ao_oo1_k = vresp_hf(cp.asarray(oo1_mo))
                v1ao_j, v1ao_k = vresp_hf(cp.asarray([cv1_mo, co1_mo, ov1_mo, oo1_mo]))
                v1ao_cv1_k = v1ao_k[0, :]
                v1ao_co1_k = v1ao_k[1, :]
                v1ao_ov1_k = v1ao_k[2, :]
                v1ao_oo1_k = v1ao_k[3, :]
                v1ao_co1_j = v1ao_j[1, :]
                v1ao_ov1_j = v1ao_j[2, :]
                # # v1_cv1_j = cp.einsum('xpq,po,qv->xov',v1ao_cv1_j,orbo.conj(), orbv) # (-1,nocca,nvirb)
                # v1_co1_j = cp.einsum('xpq,po,qv->xov', v1ao_co1_j, orbo.conj(), orbv, optimize=True)
                # v1_ov1_j = cp.einsum('xpq,po,qv->xov', v1ao_ov1_j, orbo.conj(), orbv, optimize=True)
                # # v1_oo1_j = cp.einsum('xpq,po,qv->xov',v1ao_oo1_j,orbo.conj(), orbv)
                # v1_cv1_k = cp.einsum('xpq,po,qv->xov', v1ao_cv1_k, orbo.conj(), orbv, optimize=True)  # (-1,nocca,nvirb)
                # v1_co1_k = cp.einsum('xpq,po,qv->xov', v1ao_co1_k, orbo.conj(), orbv, optimize=True)
                # v1_ov1_k = cp.einsum('xpq,po,qv->xov', v1ao_ov1_k, orbo.conj(), orbv, optimize=True)
                # v1_oo1_k = cp.einsum('xpq,po,qv->xov', v1ao_oo1_k, orbo.conj(), orbv, optimize=True)
                vjmo = contract('xpq,qo->xpo', v1ao_co1_j, orbo)
                v1_co1_j = contract('xpo,pv->xov', vjmo, orbv.conj())
                vjmo = contract('xpq,qo->xpo', v1ao_ov1_j, orbo)
                v1_ov1_j = contract('xpo,pv->xov', vjmo, orbv.conj())
                vkmo = contract('xpq,qo->xpo', v1ao_cv1_k, orbo)
                v1_cv1_k = contract('xpo,pv->xov', vkmo, orbv.conj())
                vkmo = contract('xpq,qo->xpo', v1ao_co1_k, orbo)
                v1_co1_k = contract('xpo,pv->xov', vkmo, orbv.conj())
                vkmo = contract('xpq,qo->xpo', v1ao_ov1_k, orbo)
                v1_ov1_k = contract('xpo,pv->xov', vkmo, orbv.conj())
                vkmo = contract('xpq,qo->xpo', v1ao_oo1_k, orbo)
                v1_oo1_k = contract('xpo,pv->xov', vkmo, orbv.conj())

                # # optimize code
                # _, v1_cv1_k = vresp_hf(cv1, (viridxb[no:], occidxa[:nc]), False)
                # v1_co1_j, v1_co1_k = vresp_hf(co1, (viridxb[:no], occidxa[:nc]), True)
                # v1_ov1_j, v1_ov1_k = vresp_hf(ov1, (viridxb[no:], occidxa[nc:]), True)
                # _, v1_oo1_k = vresp_hf(oo1, (viridxb[:no], occidxa[nc:]), False)

                # cv1 - cv1
                # vs[:,:nc,no:] += (cp.einsum('ji,ab,xjb->xia',iden_C,fockB_hf[nc+no:,nc+no:],zs[:,:nc,no:])-\
                #                  cp.einsum('ji,ab,xjb->xia',iden_C,fockA_hf[nc+no:,nc+no:],zs[:,:nc,no:])+\
                #                  cp.einsum('ab,ji,xjb->xia',iden_V,fockB_hf[:nc,:nc],zs[:,:nc,no:])-\
                #                  cp.einsum('ab,ji,xjb->xia',iden_V,fockA_hf[:nc,:nc],zs[:,:nc,no:]))/(2*si)
                vs_dA[:, :nc, no:] += (
                    cp.einsum('ab,xib->xia', fockB_hf[nc + no:, nc + no:], zs[:, :nc, no:])
                    - cp.einsum('ab,xib->xia', fockA_hf[nc + no:, nc + no:], zs[:, :nc, no:])
                    + cp.einsum('ji,xja->xia', fockB_hf[:nc, :nc], zs[:, :nc, no:])
                    - cp.einsum('ji,xja->xia', fockA_hf[:nc, :nc], zs[:, :nc, no:])
                ) / (2 * si)
                # co1 - co1 (𝑢𝑖|𝑗𝑣)
                # vs[:,:nc,:no] += -v1_co1_j[:,:nc,:no]/(2*si-1)+\
                #                  (cp.einsum('uv,ji,xjv->xiu',iden_O,fockB_hf[:nc,:nc],zs[:,:nc,:no])-\
                #                   cp.einsum('uv,ji,xjv->xiu',iden_O,fockA_hf[:nc,:nc],zs[:,:nc,:no]))/(2*si-1)
                vs_dA[:, :nc, :no] += -v1_co1_j[:, :nc, :no] / (2 * si - 1) + (
                    cp.einsum('ji,xju->xiu', fockB_hf[:nc, :nc], zs[:, :nc, :no])
                    - cp.einsum('ji,xju->xiu', fockA_hf[:nc, :nc], zs[:, :nc, :no])
                ) / (2 * si - 1)
                # ov1 - ov1 (𝑎𝑢|𝑣𝑏)
                # vs[:,nc:,no:] += -v1_ov1_j[:,nc:,no:]/(2*si-1)+\
                #                  (cp.einsum('uv,ab,xvb->xua',iden_O,fockB_hf[nc+no:,nc+no:],zs[:,nc:,no:])-\
                #                   cp.einsum('uv,ab,xvb->xua',iden_O,fockA_hf[nc+no:,nc+no:],zs[:,nc:,no:]))/(2*si-1)
                vs_dA[:, nc:, no:] += -v1_ov1_j[:, nc:, no:] / (2 * si - 1) + (
                    cp.einsum('ab,xub->xua', fockB_hf[nc + no:, nc + no:], zs[:, nc:, no:])
                    - cp.einsum('ab,xub->xua', fockA_hf[nc + no:, nc + no:], zs[:, nc:, no:])
                ) / (2 * si - 1)

            if self.SA > 1:
                # cv1 - co1
                # vs[:,:nc,no:] += factor1*(-v1_co1_k[:,:nc,no:] + cp.einsum('ij,av,xjv->xia',iden_C,fockB_hf[nc+no:,nc:nc+no],zs[:,:nc,:no]))
                vs_dA[:, :nc, no:] += factor1 * (
                    - v1_co1_k[:, :nc, no:]
                    + cp.einsum('av,xiv->xia', fockB_hf[nc + no:, nc:nc + no], zs[:, :nc, :no])
                )
                # co1 - cv1
                # vs[:,:nc,:no] += factor1*(-v1_cv1_k[:,:nc,:no] + cp.einsum('ij,av,xia->xjv',iden_C,fockB_hf[nc+no:,nc:nc+no],zs[:,:nc,no:]))
                vs_dA[:, :nc, :no] += factor1 * (
                    - v1_cv1_k[:, :nc, :no]
                    + cp.einsum('av,xja->xjv', fockB_hf[nc + no:, nc:nc + no], zs[:, :nc, no:])
                )
                # cv1 - ov1
                # vs[:,:nc,no:] += factor1*(-v1_ov1_k[:,:nc,no:] - cp.einsum('ab,vi,xvb->xia',iden_V,fockA_hf[nc:nc+no,:nc],zs[:,nc:,no:]))
                vs_dA[:, :nc, no:] += factor1 * (
                    - v1_ov1_k[:, :nc, no:]
                    - cp.einsum('vi,xva->xia', fockA_hf[nc:nc + no, :nc], zs[:, nc:, no:])
                )
                # ov1 - cv1
                # vs[:,nc:,no:] += factor1*(-v1_cv1_k[:,nc:,no:] - cp.einsum('ab,vi,xia->xvb',iden_V,fockA_hf[nc:nc+no,:nc],zs[:,:nc,no:]))
                vs_dA[:, nc:, no:] += factor1 * (
                    -v1_cv1_k[:, nc:, no:]
                    - cp.einsum('vi,xib->xvb', fockA_hf[nc:nc + no, :nc], zs[:, :nc, no:])
                )
                # co1 - ov1
                vs_dA[:, :nc, :no] += (v1_ov1_j[:, :nc, :no] - v1_ov1_k[:, :nc, :no]) / (2 * si - 1)
                # ov1 - co1
                vs_dA[:, nc:, no:] += (v1_co1_j[:, nc:, no:] - v1_co1_k[:, nc:, no:]) / (2 * si - 1)

            if self.SA > 2:
                # cv1 - oo1
                # vs[:,:nc,no:] += foo*(-(factor2-1)*(v1_oo1_k[:,:nc,no:]) + \
                #                  (factor2/(2*si))*(cp.einsum('vw,ia,xvw->xia',iden_O,fockB_hf[:nc,nc+no:],zs[:,nc:,:no])-\
                #                                    cp.einsum('vw,ia,xvw->xia',iden_O,fockA_hf[:nc,nc+no:],zs[:,nc:,:no])))
                vs_dA[:, :nc, no:] += foo * (
                    -(factor2 - 1) * (v1_oo1_k[:, :nc, no:])
                    + (factor2 / (2 * si)) * (
                        cp.einsum('ia,xvv->xia', fockB_hf[:nc, nc + no:], zs[:, nc:, :no])
                        - cp.einsum('ia,xvv->xia', fockA_hf[:nc, nc + no:], zs[:, nc:, :no])
                    )
                )
                vs_dA[:, nc:, :no] += foo * (
                    -(factor2 - 1) * (v1_cv1_k[:, nc:, :no])
                    + (factor2 / (2 * si)) * (
                        cp.einsum('vw,ia,xia->xvw', iden_O, fockB_hf[:nc, nc + no:], zs[:, :nc, no:])
                        - cp.einsum('vw,ia,xia->xvw', iden_O, fockA_hf[:nc, nc + no:], zs[:, :nc, no:])
                    )
                )

                # co1 - oo1
                # vs[:,:nc,:no] += foo*(factor3*(-v1_oo1_k[:,:nc,:no]-cp.einsum('uv,iw,xwv->xiu',iden_O,fockA_hf[:nc,nc:nc+no],zs[:,nc:,:no]))+\
                #                 factor4*cp.einsum('vw,iu,xvw->xiu',iden_O,fockB_hf[:nc,nc:nc+no],zs[:,nc:,:no]))
                vs_dA[:, :nc, :no] += foo * (
                    factor3 * (
                        -v1_oo1_k[:, :nc, :no]
                        - cp.einsum('iw,xwu->xiu', fockA_hf[:nc, nc:nc + no], zs[:, nc:, :no])
                    )
                    + factor4 * cp.einsum('vw,iu,xvw->xiu', iden_O, fockB_hf[:nc, nc:nc + no], zs[:, nc:, :no])
                )
                # vs[:,nc:,:no] += foo*(factor3*(-v1_co1_k[:,nc:,:no]-cp.einsum('uv,iw,xiu->xwv',iden_O,fockA_hf[:nc,nc:nc+no],zs[:,:nc,:no]))+\
                #                 factor4*cp.einsum('vw,iu,xiu->xvw',iden_O,fockB_hf[:nc,nc:nc+no],zs[:,:nc,:no]))
                vs_dA[:, nc:, :no] += foo * (
                    factor3 * (
                        -v1_co1_k[:, nc:, :no]
                        - cp.einsum('iw,xiv->xwv', fockA_hf[:nc, nc:nc + no], zs[:, :nc, :no])
                    )
                    + factor4 * cp.einsum('vw,iu,xiu->xvw', iden_O, fockB_hf[:nc, nc:nc + no], zs[:, :nc, :no])
                )
                # ov1 - oo1
                # vs[:,nc:,no:] += foo*(factor3*(-v1_oo1_k[:,nc:,no:]+cp.einsum('wu,av,xwv->xua',iden_O,fockB_hf[nc+no:,nc:nc+no],zs[:,nc:,:no]))-\
                #                 factor4*(cp.einsum('vw,au,xvw->xua',iden_O,fockA_hf[nc+no:,nc:nc+no],zs[:,nc:,:no])))
                vs_dA[:, nc:, no:] += foo * (
                    factor3 * (
                        -v1_oo1_k[:, nc:, no:]
                        + cp.einsum('av,xuv->xua', fockB_hf[nc + no:, nc:nc + no], zs[:, nc:, :no])
                    )
                    - factor4 * (
                        cp.einsum('vw,au,xvw->xua', iden_O, fockA_hf[nc + no:, nc:nc + no], zs[:, nc:, :no])
                    )
                )
                # vs[:,nc:,:no] += foo*(factor3*(-v1_ov1_k[:,nc:,:no]+cp.einsum('wu,av,xua->xwv',iden_O,fockB_hf[nc+no:,nc:nc+no],zs[:,nc:,no:]))-\
                #                 factor4*(cp.einsum('vw,au,xua->xwv',iden_O,fockA_hf[nc+no:,nc:nc+no],zs[:,nc:,no:])))
                vs_dA[:, nc:, :no] += foo * (
                    factor3 * (
                        -v1_ov1_k[:, nc:, :no]
                        + cp.einsum('av,xwa->xwv', fockB_hf[nc + no:, nc:nc + no], zs[:, nc:, no:])
                )
                    - factor4 * (
                        cp.einsum('vw,au,xua->xwv', iden_O, fockA_hf[nc + no:, nc:nc + no], zs[:, nc:, no:])
                    )
                )
            vs = vs + fglobal * vs_dA
            nz = zs.shape[0]
            hx = vs.reshape(nz, -1)

            if self.re:
                new_hx = cp.zeros_like(zs0)
                new_hx[:, :nc * nvir] += hx[:, :nc * nvir]
                oo = cp.zeros((cp.array(zs0).shape[0], no * no))
                for i in range(no):
                    oo[:, i * no:(i + 1) * no] = hx[:, nc * nvir + i * nvir:nc * nvir + no + i * nvir]
                new_oo = cp.einsum('xy,nx->ny', self.vects, oo)  # no*no-1
                for i in range(no - 1):
                    new_hx[:, nc * nvir + i * nvir:nc * nvir + i * nvir + no] = new_oo[:, i * no:(i + 1) * no]
                    new_hx[:, nc * nvir + i * nvir + no:nc * nvir + i * nvir + no + nv] = hx[:,
                                                                                          nc * nvir + no + i * nvir:nc * nvir + no + nv + i * nvir]
                new_hx[:, nc * nvir + (no - 1) * nvir:nc * nvir + (no - 1) * nvir + no - 1] = new_oo[:, (no - 1) * no:]
                new_hx[:, nc * nvir + (no - 1) * nvir + no - 1:] = hx[:, nc * nvir + (no - 1) * nvir + no:]
                hx = new_hx.copy()

            else:
                new_hx = hx.copy()
            return new_hx

        return vind, hdiag

    def deal_v_davidson(self):
        # change davidson data form like nvir|nvir|nvir|...(alpha->beta nc|no -> no|nv)  to cv|co|ov|oo

        cv = cp.zeros((self.nstates, self.nc, self.nv))
        co = cp.zeros((self.nstates, self.nc, self.no))
        ov = cp.zeros((self.nstates, self.no, self.nv))
        if self.re:
            oo = cp.zeros((self.nstates, self.no * self.no - 1))
        else:
            oo = cp.zeros((self.nstates, self.no, self.no))
        nvir = self.no + self.nv
        passed = self.nc * nvir
        if self.nstates == (self.nc + self.no) * (self.no + self.nv):
            nstates = self.nstates - 1
        else:
            nstates = self.nstates
        for state in range(nstates):
            tmp_data = self.v[:, state]
            # print(tmp_data[passed:])
            for i in range(self.nc):
                cv[state, i, :] += tmp_data[i * nvir + self.no:i * nvir + self.no + self.nv]
                co[state, i, :] += tmp_data[i * nvir:i * nvir + self.no]
            if self.re:
                for i in range(self.no - 1):
                    oo[state, i * self.no:(i + 1) * self.no] += tmp_data[passed + i * nvir:passed + i * nvir + self.no]
                    ov[state, i, :] += tmp_data[passed + i * nvir + self.no:passed + i * nvir + self.no + self.nv]
                oo[state, (self.no - 1) * self.no:] += tmp_data[passed + (self.no - 1) * nvir:passed + (
                            self.no - 1) * nvir + self.no - 1]
                ov[state, self.no - 1, :] += tmp_data[passed + (self.no - 1) * nvir + self.no - 1:]
            else:
                for i in range(self.no):
                    oo[state, i, :] += tmp_data[passed + i * nvir:passed + i * nvir + self.no]
                    ov[state, i, :] += tmp_data[passed + i * nvir + self.no:passed + i * nvir + self.no + self.nv]

        v = cp.hstack([cv.reshape(self.nstates, -1), co.reshape(self.nstates, -1), ov.reshape(self.nstates, -1),
                       oo.reshape(self.nstates, -1)])
        return v.T

    def get_precond(self, hdiag):
        threshold_t=1.0e-4
        def precond(x, e, *args):
            n_states = x.shape[0]
            diagd = cp.repeat(hdiag.reshape(1,-1), n_states, axis=0)
            e = e.reshape(-1,1)
            diagd = hdiag - (e-self.level_shift)
            diagd = cp.where(abs(diagd) < threshold_t, cp.sign(diagd)*threshold_t, diagd)
            a_size = x.shape[1]//2
            diagd[:,a_size:] = diagd[:,a_size:]*(-1)
            return x/diagd
        return precond

    def davidson_process(self, foo, fglobal, gpu_davidson=True):
        # print("Davidson process...")
        if gpu_davidson:
            vind, hdiag = self.gen_tda_operation_sf(foo, fglobal)
            precond = hdiag
            x0 = self.init_guess(self.mf, self.nstates)

            # Keep all eigenvalues as SF-TDDFT allows triplet to singlet
            # "dexcitation"
            def all_eigs(w, v, nroots, envs):
                return w, v, np.arange(w.size)

            if not callable(precond):
                precond = self.get_precond(precond)

            converged, e, x1 = lr_eigh(
                vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates,pick=all_eigs,max_cycle=self.max_cycle
            )

            self.converged = converged
            self.e = e
            self.v = cp.array(x1).T
            # print(self.v.shape)
            self.v = self.deal_v_davidson()
            print('Converged ', converged)
            return None
        else:
            print('transfer date to cpu and use cpu Davidson')
            vind, hdiag = self.gen_tda_operation_sf(foo, fglobal)
            x0 = self.init_guess(self.mf, self.nstates)
            converged, e, x1 = Davidson.davidson1(
                vind, x0, hdiag, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates, max_cycle=self.max_cycle
            )
            self.converged = converged
            self.e = e
            self.v = cp.array(x1).T
            # print(self.v.shape)
            self.v = self.deal_v_davidson()
            print('Converged ', converged)
            return None

    def kernel(self, nstates=1, remove=False, frozen=None, foo=1.0, d_lda=0.3, fglobal=None, gpu_davidson=False):
        self.re = remove
        nov = (self.nc + self.no) * (self.no + self.nv)
        self.nstates = min(nstates, nov)
        if fglobal is None:
            fglobal = (1 - d_lda) * self.hyb + d_lda
            if self.collinear == 'mcol':
                fglobal = fglobal * 4 * (self.hyb - 0.5) ** 2
        if remove:
            # print('fglobal',fglobal)
            self.vects = self.get_vect()
        self.davidson_process(foo=foo, fglobal=fglobal, gpu_davidson=gpu_davidson)
        return self.e * 27.21138505, self.v


if __name__ == '__main__':
    mol = gto.M(
        # atom='geometries/invest15/geom006.xyz',
        atom='geometries/mewes_35/geom01.xyz',
        basis='def2-tzvp',
        # atom='H 0 0 0; F 0 0 1.0',
        # basis = 'cc-pvdz',
        # atom = 'O 0 0 0; O 0 0 2.07',
        # unit = 'B',
        # basis = '631g',
        charge=0,
        spin=2,
        verbose=4,
        # symmetry = 'D2h',
    )
    tscf0 = time.time()
    mf = dft.ROKS(mol)
    # mf.xc = 'bhandhlyp'
    # mf.xc = 'hf'
    mf.xc = 'b3lyp'
    mf.kernel()
    tscf1 = time.time()
    tsfd0 = time.time()
    sf_tda = SA_SF_TDA(mf, davidson=True, collinear='mcol')
    e0, values = sf_tda.kernel(nstates=10, gpu_davidson=False)
    tsfd1 = time.time()
    print('scf use      {:8.4f} s'.format(tscf1 - tscf0))
    print('xsf down use {:8.4f} s'.format(tsfd1 - tsfd0))
    print('excited energy ', e0)

