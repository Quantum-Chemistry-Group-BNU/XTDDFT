#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "16"
import cupy as cp
import numpy as np
import time
import math

import functools
from pyscf import gto
from pyscf.lib import logger
from pyscf.tdscf import rhf as tdhf_cpu
from pyscf.dft import numint as pyscf_numint
from gpu4pyscf import dft, scf
from gpu4pyscf.tdscf import uhf
from gpu4pyscf.tdscf._uhf_resp_sf import nr_uks_fxc_sf
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.dft.numint import eval_rho2
from gpu4pyscf.tdscf._uhf_resp_sf import mcfun_eval_xc_adapter_sf

import Davidson
"""this file invoke USF-TDA and try optimize to reduce time costing"""

ha2eV = 27.21138505
eVxnm = 1239.842


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
    tc = TimeCounter()
    tc.t_ex = 0.0  # exchange correlation term cost time
    tc.t_getk = 0.0

    if isinstance(mf, scf.hf.KohnShamDFT):
        if mf.do_nlc():
            logger.warn(mf, 'NLC functional found in DFT object. Its contribution is '
                        'not included in the TDDFT response function.')

        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        t_ex0 = time.perf_counter()
        if collinear in ('ncol', 'mcol', 'alda0') and mf.xc != 'hf':
            fxc = cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ,
                                     collinear_samples, collinear=collinear)[2]
            if collinear == 'alda0':
                fxc_temp = cp.zeros((4,4,len(fxc)))
                fxc_temp[0, 0, :] = cp.asarray(fxc)
                fxc = fxc_temp
                del fxc_temp
        cp.cuda.Stream.null.synchronize()
        t_ex1 = time.perf_counter()
        tc.t_ex += t_ex1 - t_ex0
        dm0 = None

        def vind(dm1):
            t_ex0 = time.perf_counter()
            if collinear in ('ncol', 'mcol', 'alda0') and mf.xc != 'hf':
                v1 = nr_uks_fxc_sf(ni, mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   None, None, fxc)
            else:
                v1 = cp.zeros_like(dm1)
            cp.cuda.Stream.null.synchronize()
            t_ex1 = time.perf_counter()
            t_getk0 = time.perf_counter()
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
            cp.cuda.Stream.null.synchronize()
            t_getk1 = time.perf_counter()
            tc.t_ex += t_ex1 - t_ex0
            tc.t_getk += t_getk1 - t_getk0
            return v1
    else: #HF
        def vind(dm1):
            t_getk0 = time.perf_counter()
            vk = mf.get_k(mol, dm1, hermi)
            cp.cuda.Stream.null.synchronize()
            t_getk1 = time.perf_counter()
            tc.t_getk += t_getk1 - t_getk0
            return -vk

    def get_total_time():
        return (tc.t_ex, tc.t_getk)
    return vind, get_total_time


class TimeCounter:
    def __init__(self):
        # init a time counter class, when count consuming time, instantiate corresponding variables
        pass


class XSF_TDA_GPU():
    def __init__(self, mf, X=3, collinear='mcol', nstates=7, gpu_davidson=False, collinear_samples=20):
        """X=0: SF-TDA
           X=1: only add diagonal block for dA
           X=2: add all dA except for OO block
           X=3: full dA

           collinear: mcol, ncol, col, alda0
        """
        self.level_shift = tdhf_cpu.TDBase.level_shift
        self.conv_tol = tdhf_cpu.TDBase.conv_tol
        self.lindep = tdhf_cpu.TDBase.lindep
        self.max_cycle = tdhf_cpu.TDBase.max_cycle
        if cp.array(mf.mo_coeff).ndim == 3:  # UKS
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = mf.mo_occ
            self.X = 0
        else:  # ROKS
            self.mo_energy = cp.stack((mf.mo_energy, mf.mo_energy), axis=0)
            self.mo_coeff = cp.stack((mf.mo_coeff, mf.mo_coeff), axis=0)
            self.mo_occ = cp.zeros((2, len(mf.mo_coeff)))
            self.mo_occ[0][cp.where(mf.mo_occ >= 1)[0]] = 1
            self.mo_occ[1][cp.where(mf.mo_occ >= 2)[0]] = 1
            self.X = X

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
        self.nc = len(self.occidx_b)
        self.nv = len(self.viridx_a)
        self.no = len(self.occidx_a) - len(self.occidx_b)
        nov = (self.nc + self.no) * (self.no + self.nv)
        self.nstates = min(nstates, nov)

        try:  # dft
            xctype = self.mf.xc
            ni = self.mf._numint
            self.omega, self.alpha, self.hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.mol.spin)
            print('Omega, alpha, hyb',self.omega, self.alpha, self.hyb)
        except:  # HF
            xctype = None
            self.hyb = 1.0

        self.collinear = collinear
        self.collinear_samples = collinear_samples
        self.extype = 1  # spin flip down, temporary setting
        self.gpu_davidson = gpu_davidson
        self.tc = TimeCounter()  # time-consuming class

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

    def init_guess(self, mf):  # only spin down
        e_ia_a2b = (self.mo_energy[1][self.viridx_b, None] - self.mo_energy[0][self.occidx_a]).T
        e_ia_a2b = e_ia_a2b.ravel()
        nov_a2b = e_ia_a2b.size

        e_threshold = cp.sort(e_ia_a2b)[self.nstates - 1]
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

    # def gen_response_sf_delta_A(self, hermi=0, max_memory=None):  # only \Delta A
    #     mf = self.mf
    #     mol = mf.mol
    #
    #     # if hf_correction: # for \Delta A
    #     def vind(dm1, get_j=False):
    #         if get_j:
    #             vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
    #         else:
    #             vj = None
    #             vk = mf.get_k(mol, dm1, hermi=hermi)
    #         return vj, vk
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
        mf = self.mf  # for convinent, do not need write self in each place
        mo_energy = self.mo_energy
        # TODO(WHB): SymAdaptedUHF.mo_coeff may cause error, refer gpu4pyscf/tdscf/uhf.py
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        assert isinstance(mf, scf.hf.SCF)
        assert mo_coeff[0].dtype == cp.float64

        if self.X:
            # TODO(WHB): Test mf.get_fock()
            dm = mf.make_rdm1()
            vhf = mf.get_veff(mf.mol, dm)
            h1e = mf.get_hcore()
            focka_ao = h1e + vhf[0]
            fockb_ao = h1e + vhf[1]

            focka_mo = mo_coeff[0].T @ focka_ao @ mo_coeff[0]
            fockb_mo = mo_coeff[1].T @ fockb_ao @ mo_coeff[1]

        # spin flip up do not have spin comtamination,
        if self.extype == 0:
            occidxb = mo_occ[1] > 0
            viridxa = mo_occ[0] ==0
            orbob = mo_coeff[1][:,occidxb]
            orbva = mo_coeff[0][:,viridxa]
            orbov = (orbob, orbva)
            e_ia = mo_energy[0][viridxa] - mo_energy[1][occidxb,None]
            hdiag = e_ia.ravel()

        elif self.extype == 1:
            occidxa = mo_occ[0] > 0
            viridxb = mo_occ[1] ==0
            orboa = mo_coeff[0][:,occidxa]
            orbvb = mo_coeff[1][:,viridxb]
            orbov = (orboa, orbvb)
            e_ia = mo_energy[1][viridxb] - mo_energy[0][occidxa,None]
            hdiag = e_ia.ravel()
        else:
            raise ValueError(
                f"Invalid extype = {self.extype}. "
                "extype must be 0 (beta->alpha spin flip up) or 1 (alpha->beta spin flip down)."
            )

        vresp, get_usf_time = gen_uhf_response_sf(
            mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0, collinear=self.collinear,
            collinear_samples=self.collinear_samples)

        def vind(zs):
            zs = cp.asarray(zs).reshape(-1, *e_ia.shape)
            orbo, orbv = orbov
            mo1 = contract('xov,pv->xpo', zs, orbv)
            dms = contract('xpo,qo->xpq', mo1, orbo.conj())
            # dms = tag_array(dms, mo1=mo1, occ_coeff=orbo)  # TODO: bug
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

        return vind, hdiag, get_usf_time

    def gen_tda_operation_sf(self, foo, fglobal):
        mf = self.mf.copy()
        mf.mo_occ = self.mo_occ
        mf.mo_coeff = self.mo_coeff
        mf.mo_energy = self.mo_energy

        self.tc.t_usf = 0.0
        self.tc.t_dA = 0.0
        self.tc.t_dA_getjk = 0.0

        assert (self.mo_coeff[0].dtype == cp.double)
        orboa = self.mo_coeff[0][:, self.occidx_a]
        orbob = self.mo_coeff[1][:, self.occidx_b]
        orbva = self.mo_coeff[0][:, self.viridx_a]
        orbvb = self.mo_coeff[1][:, self.viridx_b]
        nc = self.nc  # for convinent, do not write self each time
        nv = self.nv  # for convinent
        no = self.no  # for convinent
        nvir = no + nv  # number of beta orbitals
        nocc = nc + no  # number of alpha orbitals
        si = no / 2.0
        orbov = (orboa, orbvb)
        iden_C = cp.eye(nc)
        iden_V = cp.eye(nv)
        iden_O = cp.eye(no)

        # USF-TDA
        t_usf0 = time.perf_counter()
        vind_A, hdiag, get_usf_ex_getk_time = self.gen_vind()
        cp.cuda.Stream.null.synchronize()
        t_usf1 = time.perf_counter()
        self.tc.t_usf = t_usf1 - t_usf0

        if self.re:
            oo = cp.zeros((self.no*self.no)) # full oo
            for i in range(self.no):
                oo[i*no:(i+1)*no] = hdiag[nc*nvir+nvir*i:nc*nvir+no+nvir*i]
            new_oo = cp.einsum('x,xy->y',oo,self.vects)
            new_hdiag = cp.zeros(len(hdiag)-1)
            new_hdiag[:nc*nvir] = hdiag[:nc*nvir]
            for i in range(self.no-1):
                new_hdiag[nc*nvir+i*nvir:nc*nvir+no+i*nvir] = new_oo[i*no:(i+1)*no]
                new_hdiag[nc*nvir+no+i*nvir:nc*nvir+no+i*nvir+nv] = hdiag[nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir]
            new_hdiag[nc*nvir+(self.no-1)*nvir:nc*nvir+(self.no-1)*nvir+no-1] = new_oo[(self.no-1)*no:]
            new_hdiag[nc*nvir+(self.no-1)*nvir+no-1:] = hdiag[nc*nvir+(self.no-1)*nvir+no:]
            hdiag = new_hdiag

        if self.X > 0:
            t_dA0 = time.perf_counter()
            vresp_hf = self.gen_response_sf_delta_A(hermi=0)  # to calculate \Delta A

            t_fock0 = time.perf_counter()
            rohf = scf.ROHF(self.mf.mol)
            dm = self.mf.make_rdm1()
            vhf = rohf.get_veff(rohf.mol, dm)
            h1e = rohf.get_hcore()
            fockA_hf = self.mf.mo_coeff.T @ (h1e + vhf[0]) @ self.mf.mo_coeff
            fockB_hf = self.mf.mo_coeff.T @ (h1e + vhf[1]) @ self.mf.mo_coeff
            cp.cuda.Stream.null.synchronize()
            t_fock1 = time.perf_counter()
            factor1 = cp.sqrt((2 * si + 1) / (2 * si)) - 1
            factor2 = cp.sqrt((2 * si + 1) / (2 * si - 1))
            factor3 = cp.sqrt((2 * si) / (2 * si - 1)) - 1
            factor4 = 1 / cp.sqrt(2 * si * (2 * si - 1))
            cp.cuda.Stream.null.synchronize()
            t_dA1 = time.perf_counter()
            self.tc.t_dA += t_dA1 - t_dA0

        def vind(zs0):  # vector-matrix product for indexed operations
            t_usf0 = time.perf_counter()
            zs0 = cp.asarray(zs0)
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
            zs = cp.asarray(new_zs0).reshape(-1, nocc, nvir)
            vs = vind_A(zs).reshape(-1, nocc, nvir)
            vs_dA = cp.zeros_like(vs)
            cp.cuda.Stream.null.synchronize()
            t_usf1 = time.perf_counter()

            t_dA0 = time.perf_counter()
            if self.X > 0:
                cv1 = zs[:, :nc, no:]
                co1 = zs[:, :nc, :no]
                ov1 = zs[:, nc:, no:]
                oo1 = zs[:, nc:, :no]

                mo1 = contract('xov,pv->xpo', cv1, orbvb[:, no:])
                cv1_mo = contract('xpo,qo->xpq', mo1, orboa[:, :nc].conj())
                mo1 = contract('xov,pv->xpo', co1, orbvb[:, :no])
                co1_mo = contract('xpo,qo->xpq', mo1, orboa[:, :nc].conj())
                mo1 = contract('xov,pv->xpo', ov1, orbvb[:, no:])
                ov1_mo = contract('xpo,qo->xpq', mo1, orboa[:, nc:].conj())
                mo1 = contract('xov,pv->xpo', oo1, orbvb[:, :no])
                oo1_mo = contract('xpo,qo->xpq', mo1, orboa[:, nc:].conj())

                t_dA_getjk0 = time.perf_counter()
                # # origin code
                # _, v1ao_cv1_k = vresp_hf(cp.asarray(cv1_mo))  # (-1,nmo,nmo)
                # v1ao_co1_j, v1ao_co1_k = vresp_hf(cp.asarray(co1_mo))
                # v1ao_ov1_j, v1ao_ov1_k = vresp_hf(cp.asarray(ov1_mo))
                # _, v1ao_oo1_k = vresp_hf(cp.asarray(oo1_mo))

                # # optimize code
                # _, v1ao_cv1_k = vresp_hf(cv1_mo, False)
                # v1ao_co1_j, v1ao_co1_k = vresp_hf(co1_mo, True)
                # v1ao_ov1_j, v1ao_ov1_k = vresp_hf(ov1_mo, True)
                # _, v1ao_oo1_k = vresp_hf(oo1_mo, False)

                # calculate atomic two electronic integral one time
                v1ao_j, v1ao_k = vresp_hf(cp.asarray([cv1_mo, co1_mo, ov1_mo, oo1_mo]))
                v1ao_cv1_k = v1ao_k[0, :]
                v1ao_co1_k = v1ao_k[1, :]
                v1ao_ov1_k = v1ao_k[2, :]
                v1ao_oo1_k = v1ao_k[3, :]
                v1ao_co1_j = v1ao_j[1, :]
                v1ao_ov1_j = v1ao_j[2, :]
                cp.cuda.Stream.null.synchronize()
                t_dA_getjk1 = time.perf_counter()

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

            if self.X > 1:
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

            if self.X > 2:
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

            cp.cuda.Stream.null.synchronize()
            t_dA1 = time.perf_counter()
            self.tc.t_dA += t_dA1 - t_dA0
            self.tc.t_usf += t_usf1 - t_usf0
            if self.X > 0:
                self.tc.t_dA_getjk += t_dA_getjk1 - t_dA_getjk0
            return new_hx

        def get_total_time():
            t_xsf_fock = 0
            if self.X > 0:
                t_xsf_fock = t_fock1 - t_fock0
            t_usf_ex_getk = get_usf_ex_getk_time()
            t_usf_ex = t_usf_ex_getk[0]
            t_usf_getk = t_usf_ex_getk[1]
            return (t_usf_ex, t_usf_getk, self.tc.t_usf, self.tc.t_dA_getjk, t_xsf_fock, self.tc.t_dA)

        return vind, hdiag, get_total_time

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
                oo[state, (self.no - 1) * self.no:] += tmp_data[passed + (self.no - 1) * nvir:passed + (self.no - 1) * nvir + self.no - 1]
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

    def davidson_process(self, foo, fglobal):
        # print("Davidson process...")
        if self.gpu_davidson:
            vind, hdiag, get_time = self.gen_tda_operation_sf(foo, fglobal)
            precond = hdiag
            x0 = self.init_guess(self.mf)

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
            print('The GPU version of the Davidson iteration has not yet returned the number of iterations.')
            self.icyc = None  # TODO(WHB): gpu davidson do not return icyc
            self.converged = converged
            self.e = e
            self.v = cp.array(x1).T
            # print(self.v.shape)
            self.v = self.deal_v_davidson()
            print('Converged ', converged)
        else:
            print('transfer date to cpu and use cpu Davidson')
            vind, hdiag, get_time = self.gen_tda_operation_sf(foo, fglobal)
            x0 = self.init_guess(self.mf)
            converged, e, x1, Davidcyc = Davidson.davidson1(
                vind, x0, hdiag, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates, max_cycle=self.max_cycle
            )
            self.Davidcyc = Davidcyc
            self.converged = converged
            self.e = e
            self.v = cp.array(x1).T
            # print(self.v.shape)
            self.v = self.deal_v_davidson()
            print('Converged ', converged)
        return get_time

    def kernel(self, remove=None, frozen=None, foo=1.0, d_lda=0.3, fglobal=None):
        tsfd0 = time.perf_counter()
        if remove is None:
            if cp.array(self.mf.mo_coeff).ndim == 3:  # UKS
                self.re = False
            else:
                self.re = True
        else:
            self.re = remove
        print('remove is {}'.format(str(self.re)))
        if fglobal is None:
            if self.omega == 0:
                cx = self.hyb
            else:
                cx = self.hyb + (self.alpha-self.hyb)*math.erf(self.omega)
            fglobal = (1 - d_lda) * cx + d_lda
            if self.collinear == 'mcol':
                fglobal = fglobal*4*(cx-0.5)**2
        if self.re:
            # print('fglobal',fglobal)
            self.vects = self.get_vect()
        print('foo', foo)
        print('fglobal',fglobal)
        get_time = self.davidson_process(foo=foo, fglobal=fglobal)
        cp.cuda.Stream.null.synchronize()
        tsfd1 = time.perf_counter()
        print('there are {:8} orbitals (basis)'.format(self.mo_occ.shape[1]))
        print('Davidson iteration {:2} times'.format(self.Davidcyc[0]))
        print('='*50)
        print(f'{"num":>4} {"energy":>8}')
        for ni, ei in zip(range(self.nstates), self.e*ha2eV):
            print(f'{ni + 1:4d} {ei:8.4f}')
        print('='*50)
        t = get_time()
        t_usf_ex, t_usf_getk, t_usf, t_xsf_getjk, t_xsf_fock, t_dA = t
        print('calculate exchange correlation term use         {:8.4f} s'.format(t_usf_ex))
        print('calculate A matrix get_k use                    {:8.4f} s'.format(t_usf_getk))
        print('Ax total use                                    {:8.4f} s'.format(t_usf))
        if self.Davidcyc is not None:
            print('each Davdison iteration Ax use                  {:8.4f} s'.format(t_usf/self.Davidcyc[0]))
        if self.X > 0:
            print('calculate delta A get_jk use                    {:8.4f} s'.format(t_xsf_getjk))
            print('calculate delta A fock use                      {:8.4f} s'.format(t_xsf_fock))
            print('delta A x total use                             {:8.4f} s'.format(t_dA))
            if self.Davidcyc is not None:
                print('each Davidson iteration delta A x average use   {:8.4f} s'.format(t_dA/self.Davidcyc[0]))
            print('XSF-TDA use                                     {:8.4f} s'.format(tsfd1 - tsfd0))
        else:
            print('USF-TDA use                                     {:8.4f} s'.format(tsfd1 - tsfd0))
        return self.e * ha2eV, self.v


if __name__ == '__main__':
    path = '/home/lenovo2/users/zhw/TDDFT/SFTDA/TADF/mewes_35/geometries/24/'
    mol = gto.M(
        # atom='geometries/invest15/geom006.xyz',
        # atom='geometries/mewes_35/geom01.xyz',
        atom=path+'geom.xyz',
        # basis='def2-svp',
        basis='def2-tzvp',
        # atom='H 0 0 0; F 0 0 1.0',
        # basis = 'cc-pvdz',
        # atom = 'O 0 0 0; O 0 0 2.07',
        # unit = 'B',
        # basis = '631g',
        charge=0,
        spin=2,
        verbose=4,
        # symmetry = 'C2v',
    )
    tscf0 = time.perf_counter()
    mf = dft.ROKS(mol)
    # mf.xc = 'bhandhlyp'
    # mf.xc = 'hf'
    # mf.xc = 'pbe0'
    # mf.xc = 'camb3lyp'
    mf.xc = 'b3lyp'
    mf.init_guess = 'huckel'
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    assert mf.converged
    cp.cuda.Stream.null.synchronize()
    tscf1 = time.perf_counter()
    print('scf use      {:8.4f} s'.format(tscf1 - tscf0))
    print('='*50)
    sf_tda = XSF_TDA_GPU(mf, collinear='mcol', nstates=7, gpu_davidson=False, collinear_samples=20)
    e0, values = sf_tda.kernel()

