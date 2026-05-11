#!/usr/bin/env python
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append('../')
sys.path.append('../../')
import time
import math
import cupy as cp
import numpy as np
from functools import reduce, partial

from pyscf import gto
from pyscf.lib import logger
from pyscf.tdscf import rhf as tdhf_cpu
from pyscf.dft import numint as pyscf_numint
from pyscf.symm import direct_prod
from gpu4pyscf import dft, scf
from gpu4pyscf.scf import hf, uhf, rohf
from gpu4pyscf.tdscf._uhf_resp_sf import nr_uks_fxc_sf
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.dft.numint import eval_rho2
from gpu4pyscf.tdscf._uhf_resp_sf import mcfun_eval_xc_adapter_sf

from utils import unit, atom, utils, Davidson
"""
this file is a very completely version, include
 1. spin flip up: ROKS with SF-TDA, UKS with SF-TDA
 2. spin flip down: XSF-TDA, USF-TDA
"""


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
            fn_eval_xc = partial(__mcfun_fn_eval_xc2, ni, xc_code, xctype)
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


class TimeCounter:
    def __init__(self):
        # init a time counter class, when count consuming time,
        # instantiate corresponding variables
        pass


class XSF_TDA_GPU():
    def __init__(
            self, mf, X=3, collinear='mcol', nstates=7, extype=1,
            gpu_davidson=False, collinear_samples=20, remove=None,
            foo=1.0, d_lda=0.3, fglobal=None
        ):
        """X=0: SF-TDA (UKS with SF-TDA and ROKS with SF-TDA)
           X=1: only add diagonal block for dA
           X=2: add all dA except for OO block
           X=3: full dA

           collinear: mcol, ncol, col, alda0

           extype=0: spin flip up
           extype=1: spin flip down
        """
        self.level_shift = tdhf_cpu.TDBase.level_shift
        self.conv_tol = tdhf_cpu.TDBase.conv_tol
        self.lindep = tdhf_cpu.TDBase.lindep
        self.max_cycle = tdhf_cpu.TDBase.max_cycle
        self.mol = mf.mol
        self.mf = mf

        try:  # dft
            ni = self.mf._numint
            self.omega, self.alpha, self.hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.mol.spin)
            print('Omega, alpha, hyb',self.omega, self.alpha, self.hyb)
        except:  # HF
            self.omega = 0.0
            self.hyb = 1.0

        self.collinear = collinear
        self.collinear_samples = collinear_samples
        self.extype = extype
        self.gpu_davidson = gpu_davidson  # whether use gpu davidson solver
        self.tc = TimeCounter()  # time-consuming class

        if isinstance(mf, uhf.UHF):  # UKS
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = mf.mo_occ
            self.X = 0
        elif isinstance(mf, rohf.ROHF):  # ROKS
            self.mo_energy = cp.stack((mf.mo_energy, mf.mo_energy), axis=0)
            self.mo_coeff = cp.stack((mf.mo_coeff, mf.mo_coeff), axis=0)
            self.mo_occ = cp.zeros((2, len(mf.mo_coeff)))
            self.mo_occ[0][cp.where(mf.mo_occ >= 1)[0]] = 1
            self.mo_occ[1][cp.where(mf.mo_occ >= 2)[0]] = 1
            self.X = X

        self.occidx_a = cp.where(self.mo_occ[0] == 1)[0]
        self.viridx_a = cp.where(self.mo_occ[0] == 0)[0]
        self.occidx_b = cp.where(self.mo_occ[1] == 1)[0]
        self.viridx_b = cp.where(self.mo_occ[1] == 0)[0]
        self.nc = len(self.occidx_b)
        self.nv = len(self.viridx_a)
        self.no = len(self.occidx_a) - len(self.occidx_b)
        if self.extype == 0:
            nov = self.nc * self.nv
        elif self.extype == 1:
            nov = (self.nc + self.no) * (self.no + self.nv)
        else:
            raise ValueError(
                f"Invalid extype = {self.extype}. "
                "extype must be 0 (beta->alpha spin flip up) or 1 (alpha->beta spin flip down)."
            )
        self.nstates = min(nstates, nov)

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
            self.fglobal = (1 - d_lda) * cx + d_lda
            if self.collinear == 'mcol':
                self.fglobal = self.fglobal*4*(cx-0.5)**2
        if self.re:
            if self.extype == 0:
                pass
            elif self.extype == 1:
                self.vects = self.get_vect()
        self.foo = foo
        print('foo', self.foo)
        print('fglobal', self.fglobal)

    def get_vect(self):  # construct Vmat N*(N-1)
        tmp_v = cp.zeros((self.no - 1, self.no))  # (self.no-1,self.no)
        for i in range(1, self.no):  # 1->v1 2->v2 3->v3 ...
            factor = 1 / cp.sqrt((self.no - i + 1) * (self.no - i))
            tmp = [self.no - i] + [-1] * (self.no - i)  # (N-i,-1,-1,-1 ...)
            tmp_v[i - 1][i - 1:] = cp.array(tmp) * factor
        vect = tmp_v.T  # N(N-1)
        vects = cp.eye(self.no * self.no)
        vects = vects[:, :-1]  # no*no*(no*no-1)
        index = [0]
        for i in range(1, self.no):
            index.append(i * (self.no + 1))
        for i in range(vect.shape[1]):
            vects[0::self.no + 1, index[i]] = vect[:, i]
        return vects

    def init_guess(self):
        if self.extype == 0:
            e_ia_b2a = (self.mo_energy[0][self.viridx_a, None] - self.mo_energy[1][self.occidx_b]).T
            e_ia_b2a = e_ia_b2a.ravel()
            nov_b2a = e_ia_b2a.size
            e_threshold = cp.sort(e_ia_b2a)[self.nstates - 1]
            e_threshold += 1e-5
            idx = cp.where(e_ia_b2a <= e_threshold)[0]
            x0 = cp.zeros((idx.size, nov_b2a))
            for i, j in enumerate(idx):
                x0[i, j] = 1
        elif self.extype == 1:
            e_ia_a2b = (self.mo_energy[1][self.viridx_b, None] - self.mo_energy[0][self.occidx_a]).T
            e_ia_a2b = e_ia_a2b.ravel()
            nov_a2b = e_ia_a2b.size
            e_threshold = cp.sort(e_ia_a2b)[self.nstates - 1]
            e_threshold += 1e-5
            idx = cp.where(e_ia_a2b <= e_threshold)[0]
            x0 = cp.zeros((idx.size, nov_a2b))
            for i, j in enumerate(idx):
                x0[i, j] = 1
            if self.re:
                x0 = x0[:, :-1]
        else:
            raise ValueError
        return cp.array(x0)

    def gen_uhf_response_sf(self, hermi=0, collinear='mcol', collinear_samples=200):
        '''Generate a function to compute the product of Spin Flip UKS response function
        and UKS density matrices.
        '''
        mf = self.mf
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mol = self.mol
        assert hermi == 0
        self.tc.A_vxc = 0.0
        self.tc.A_gk = 0.0

        if isinstance(mf, hf.KohnShamDFT):
            if mf.do_nlc():
                logger.warn(mf, 'NLC functional found in DFT object. Its contribution is '
                                'not included in the TDDFT response function.')
            ni = mf._numint
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
            hybrid = ni.libxc.is_hybrid_xc(mf.xc)
            dm0 = None

            # support alda0
            if collinear == 'col' or mf.xc == 'hf':
                pass
            else:
                fxc = cache_xc_kernel_sf(
                    ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ,
                    collinear_samples, collinear=collinear
                )
                fxc = fxc[2]
                if collinear == 'alda0':
                    fxc = cp.pad(cp.asarray(fxc)[None, None], ((0, 3), (0, 3), (0, 0)))

            def vind(dm1):
                tAv0 = time.perf_counter()
                if collinear == 'col' or mf.xc == 'hf':
                    v1 = cp.zeros_like(dm1)
                else:
                    v1 = nr_uks_fxc_sf(ni, mol, mf.grids, mf.xc, dm0, dm1, 0, hermi, None, None, fxc)
                cp.cuda.Stream.null.synchronize()
                tAv1 = time.perf_counter()
                self.tc.A_vxc += tAv1 - tAv0

                tAgk0 = time.perf_counter()
                if hybrid:
                    # j = 0 in spin flip part.
                    if omega == 0:
                        vk = mf.get_k(mol, dm1, hermi) * hyb
                    elif alpha == 0:  # LR=0, only SR exchange
                        vk = mf.get_k(mol, dm1, hermi, omega=-omega) * hyb
                    elif hyb == 0:  # SR=0, only LR exchange
                        vk = mf.get_k(mol, dm1, hermi, omega=omega) * alpha
                    else:  # SR and LR exchange with different ratios
                        vk = mf.get_k(mol, dm1, hermi) * hyb
                        vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha - hyb)
                    v1 -= vk
                cp.cuda.Stream.null.synchronize()
                tAgk1 = time.perf_counter()
                self.tc.A_gk += tAgk1 - tAgk0
                return v1
        else:  # HF
            def vind(dm1):
                tAgk0 = time.perf_counter()
                vk = mf.get_k(mol, dm1, hermi)
                cp.cuda.Stream.null.synchronize()
                tAgk1 = time.perf_counter()
                self.tc.A_gk += tAgk1 - tAgk0
                return -vk
        return vind

    def gen_response_sf_delta_A(self, hermi=0, max_memory=None):  # only \Delta A
        mf = self.mf
        mol = self.mol

        # if hf_correction: # for \Delta A
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            return vj, vk

        # # if hf_correction: # for \Delta A
        # # no additional comloub integral, but extra eri
        # def vind(dm1, get_j=False):
        #     if get_j:
        #         vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
        #     else:
        #         vj = None
        #         vk = mf.get_k(mol, dm1, hermi=hermi)
        #     return vj, vk
        return vind

    def gen_vind(self):
        '''Generate function to compute A*x for spin-flip TDDFT case.
        '''
        tAp0 = time.perf_counter()
        mf = self.mf  # for convinent, do not need write self in each place
        mol = self.mol
        # TODO(WHB): SymAdaptedUHF.mo_coeff may cause error, refer gpu4pyscf/tdscf/uhf.py
        mo_energy = self.mo_energy
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        # mf.mo_energy = self.mo_energy
        # mf.mo_coeff = self.mo_coeff
        # mf.mo_occ = self.mo_occ
        occidx_a = self.occidx_a
        viridx_a = self.viridx_a
        occidx_b = self.occidx_b
        viridx_b = self.viridx_b
        nc = self.nc
        nv = self.nv
        no = self.no
        si = no / 2.0
        foo = self.foo
        fglobal = self.fglobal
        iden_O = cp.eye(no)
        assert isinstance(mf, hf.SCF)
        assert mo_coeff[0].dtype == cp.float64

        tAp_f0 = time.perf_counter()
        if isinstance(mf, rohf.ROHF):
            dm = mf.make_rdm1()
            vhf = mf.get_veff(mol, dm)
            h1e = mf.get_hcore()
            focka_ao = h1e + vhf[0]
            fockb_ao = h1e + vhf[1]
            focka_mo = mo_coeff[0].T @ focka_ao @ mo_coeff[0]
            fockb_mo = mo_coeff[1].T @ fockb_ao @ mo_coeff[1]
            if self.extype == 0:
                hdiag = (focka_mo.diagonal()[viridx_a] - fockb_mo.diagonal()[occidx_b, None]).ravel()
            elif self.extype == 1:
                hdiag = (fockb_mo.diagonal()[viridx_b] - focka_mo.diagonal()[occidx_a, None]).ravel()
        elif isinstance(mf, uhf.UHF):
            if self.extype == 0:
                e_ia = mo_energy[0][viridx_a] - mo_energy[1][occidx_b, None]
                hdiag = e_ia.ravel()
            elif self.extype == 1:
                e_ia = mo_energy[1][viridx_b] - mo_energy[0][occidx_a, None]
                hdiag = e_ia.ravel()
        cp.cuda.Stream.null.synchronize()
        tAp_f1 = time.perf_counter()
        self.tc.Ap_f = tAp_f1 - tAp_f0

        # spin flip up do not have spin comtamination,
        if self.extype == 0:
            orbob = mo_coeff[1][:, occidx_b]
            orbva = mo_coeff[0][:, viridx_a]
            orbov = (orbob, orbva)
            ndim = (len(occidx_b), len(viridx_a))
        elif self.extype == 1:
            orboa = mo_coeff[0][:, occidx_a]
            orbvb = mo_coeff[1][:, viridx_b]
            orbov = (orboa, orbvb)
            ndim = (len(occidx_a), len(viridx_b))
        else:
            raise ValueError

        if self.re:
            if self.extype == 0:
                pass
            elif self.extype == 1:
                oo = cp.zeros((no * no))  # full oo
                for i in range(no):
                    oo[i * no:(i + 1) * no] = hdiag[(nc + i) * (no + nv):(nc + i) * (no + nv) + no]
                new_oo = cp.einsum('x,xy->y', oo, self.vects)
                new_hdiag = cp.zeros(len(hdiag) - 1)
                new_hdiag[:nc * (no + nv)] = hdiag[:nc * (no + nv)]
                for i in range(no - 1):
                    new_hdiag[(nc + i) * (no + nv):(nc + i) * (no + nv) + no] \
                        = new_oo[i * no:(i + 1) * no]
                    new_hdiag[(nc + i) * (no + nv) + no:(nc + i) * (no + nv) + no + nv] \
                        = hdiag[(nc + i) * (no + nv) + no:(nc + i) * (no + nv) + no + nv]
                new_hdiag[(nc + no - 1) * (no + nv):(nc + no - 1) * (no + nv) + no - 1] = new_oo[(no - 1) * no:]
                new_hdiag[(nc + no - 1) * (no + nv) + no - 1:] = hdiag[(nc + no - 1) * (no + nv) + no:]
                hdiag = new_hdiag

        tAp_k0 = time.perf_counter()
        vresp = self.gen_uhf_response_sf(
            hermi=0, collinear=self.collinear, collinear_samples=self.collinear_samples
        )
        cp.cuda.Stream.null.synchronize()
        tAp_k1 = time.perf_counter()
        self.tc.Ap_k = tAp_k1 - tAp_k0

        cp.cuda.Stream.null.synchronize()
        tAp1 = time.perf_counter()
        self.tc.Ap = tAp1 - tAp0


        tdAp0 = time.perf_counter()
        if self.X > 0:
            if self.extype == 0:
                pass
            elif self.extype == 1:
                vresp_hf = self.gen_response_sf_delta_A(hermi=0)  # to calculate \Delta A
                mf_rohf = rohf.ROHF(self.mol)
                dm = mf.make_rdm1()
                vhf = mf_rohf.get_veff(mf_rohf.mol, dm)
                h1e = mf_rohf.get_hcore()
                fockA_hf = mo_coeff[0].T @ (h1e + vhf[0]) @ mo_coeff[0]
                fockB_hf = mo_coeff[1].T @ (h1e + vhf[1]) @ mo_coeff[1]
                factor1 = cp.sqrt((2 * si + 1) / (2 * si)) - 1
                factor2 = cp.sqrt((2 * si + 1) / (2 * si - 1))
                factor3 = cp.sqrt((2 * si) / (2 * si - 1)) - 1
                factor4 = 1 / cp.sqrt(2 * si * (2 * si - 1))
        cp.cuda.Stream.null.synchronize()
        tdAp1 = time.perf_counter()
        self.tc.dAp = tdAp1 - tdAp0

        self.tc.Adv = 0.0
        self.tc.dAdv = 0.0
        self.tc.dA_gjk = 0.0

        def vind(zs0):
            tAdv0 = time.perf_counter()  # time of A matrix Davidson
            zs0 = cp.asarray(zs0)
            if self.re:
                if self.extype == 0:
                    new_zs0 = zs0.copy()
                elif self.extype == 1:
                    oo = cp.zeros((cp.array(zs0).shape[0], no * no - 1))  # get oo from zs0, which is no*no-1

                    for i in range(no - 1):
                        # print('nc*nvir+i*nvir:nc*nvir+no+i*nvir ',nc*nvir+i*nvir,nc*nvir+no+i*nvir)
                        oo[:, i * no:(i + 1) * no] = zs0[:, (nc+i) * (no + nv):(nc+i) * (no + nv) + no]
                    # print(oo[:,(no-1)*no:].shape, zs0[:,nc*nvir+(no-1)*nvir:nc*nvir+(no-1)*nvir+no-1].shape)
                    oo[:, (no - 1) * no:] = zs0[:, (nc + no - 1) * (no + nv):(nc + no - 1) * (no + nv) + no - 1]  # no*no-1
                    new_oo = cp.einsum('xy,ny->nx', self.vects, oo)  # we want the whole matrix of oo, which is no*no
                    new_zs0 = cp.zeros((zs0.shape[0], zs0.shape[1] + 1))  # full matrix
                    new_zs0[:, :nc * (no + nv)] = zs0[:, :nc * (no + nv)]
                    for i in range(no - 1):
                        new_zs0[:, (nc + i) * (no + nv): (nc + i) * (no + nv) + no] = new_oo[:, i * no:(i + 1) * no]
                        new_zs0[:, (nc + i) * (no + nv) + no: (nc + i) * (no + nv) + no + nv] \
                            = zs0[:,(nc + i) * (no + nv) + no: (nc + i) * (no + nv) + no + nv]
                    new_zs0[:, (nc + no - 1) * (no + nv):(nc + no - 1) * (no + nv) + no] = new_oo[:, -no:]
                    new_zs0[:, (nc + no - 1) * (no + nv) + no:] = zs0[:, (nc + no - 1) * (no + nv) + no - 1:]
            else:
                new_zs0 = zs0.copy()

            zs = cp.asarray(new_zs0).reshape(-1, *ndim)
            orbo, orbv = orbov  # mo_coeff for alpha and beta
            mo1 = contract('xov,pv->xpo', zs, orbv)
            dms = contract('xpo,qo->xpq', mo1, orbo.conj())
            # https://github.com/pyscf/gpu4pyscf/issues/676
            # dms = tag_array(dms, mo1=mo1, occ_coeff=orbo)  # TODO: bug
            dms = tag_array(dms, mo1=mo1, occ_coeff=0.5*orbo)
            v1ao = vresp(dms)
            v1mo = contract('xpq,qo->xpo', v1ao, orbo)
            v1mo = contract('xpo,pv->xov', v1mo, orbv.conj())
            cp.cuda.Stream.null.synchronize()
            tAdv1 = time.perf_counter()
            self.tc.Adv += tAdv1 - tAdv0

            tdAdv0 = time.perf_counter()  # time of A matrix Davidson
            if isinstance(mf, rohf.ROHF):
                if self.extype == 0:
                    v1mo += (
                        contract('ab,xib->xia', focka_mo[len(occidx_a):, len(occidx_a):], zs)
                        - contract('ij,xja->xia', fockb_mo[:len(occidx_b), :len(occidx_b)], zs)
                    )
                elif self.extype == 1:
                    v1mo += (
                        contract('ab,xib->xia', fockb_mo[len(occidx_b):, len(occidx_b):], zs)
                        - contract('ij,xja->xia', focka_mo[:len(occidx_a), :len(occidx_a)], zs)
                    )
                    if self.X > 0:
                        cv1 = zs[:, :nc, no:]
                        co1 = zs[:, :nc, :no]
                        ov1 = zs[:, nc:, no:]
                        oo1 = zs[:, nc:, :no]
                        mo1_cv1 = contract('xov,pv->xpo', cv1, orbvb[:, no:])
                        cv1_mo = contract('xpo,qo->xpq', mo1_cv1, orboa[:, :nc].conj())
                        mo1_co1 = contract('xov,pv->xpo', co1, orbvb[:, :no])
                        co1_mo = contract('xpo,qo->xpq', mo1_co1, orboa[:, :nc].conj())
                        mo1_ov1 = contract('xov,pv->xpo', ov1, orbvb[:, no:])
                        ov1_mo = contract('xpo,qo->xpq', mo1_ov1, orboa[:, nc:].conj())
                        mo1_oo1 = contract('xov,pv->xpo', oo1, orbvb[:, :no])
                        oo1_mo = contract('xpo,qo->xpq', mo1_oo1, orboa[:, nc:].conj())

                        # calculate atomic two electronic integral one time
                        dms = cp.asarray([cv1_mo, co1_mo, ov1_mo, oo1_mo])
                        # here occ_coeff do not multiple 0.5,
                        # because when occ_coeff is a list, in df.df_jk.get_jk() default do not multiple 2.0
                        dms = tag_array(
                            dms,
                            mo1=[mo1_cv1, mo1_co1, mo1_ov1, mo1_oo1],
                            occ_coeff=[orboa[:, :nc], orboa[:, :nc], orboa[:, nc:], orboa[:, nc:]]
                        )
                        tdA_gjk0 = time.perf_counter()
                        v1ao_j, v1ao_k = vresp_hf(dms)
                        cp.cuda.Stream.null.synchronize()
                        tdA_gjk1 = time.perf_counter()
                        self.tc.dA_gjk += tdA_gjk1 - tdA_gjk0
                        # v1ao_j, v1ao_k = vresp_hf(cp.asarray([cv1_mo, co1_mo, ov1_mo, oo1_mo]))
                        v1ao_cv1_k = v1ao_k[0, :]
                        v1ao_co1_k = v1ao_k[1, :]
                        v1ao_ov1_k = v1ao_k[2, :]
                        v1ao_oo1_k = v1ao_k[3, :]
                        v1ao_co1_j = v1ao_j[1, :]
                        v1ao_ov1_j = v1ao_j[2, :]

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
                        v1mo[:, :nc, no:] += fglobal * (
                            cp.einsum('ab,xib->xia', fockB_hf[nc + no:, nc + no:], zs[:, :nc, no:])
                            - cp.einsum('ab,xib->xia', fockA_hf[nc + no:, nc + no:], zs[:, :nc, no:])
                            + cp.einsum('ji,xja->xia', fockB_hf[:nc, :nc], zs[:, :nc, no:])
                            - cp.einsum('ji,xja->xia', fockA_hf[:nc, :nc], zs[:, :nc, no:])
                        ) / (2 * si)
                        # co1 - co1 (𝑢𝑖|𝑗𝑣)
                        # vs[:,:nc,:no] += -v1_co1_j[:,:nc,:no]/(2*si-1)+\
                        #                  (cp.einsum('uv,ji,xjv->xiu',iden_O,fockB_hf[:nc,:nc],zs[:,:nc,:no])-\
                        #                   cp.einsum('uv,ji,xjv->xiu',iden_O,fockA_hf[:nc,:nc],zs[:,:nc,:no]))/(2*si-1)
                        v1mo[:, :nc, :no] += fglobal * -v1_co1_j[:, :nc, :no] / (2 * si - 1) + fglobal * (
                            cp.einsum('ji,xju->xiu', fockB_hf[:nc, :nc], zs[:, :nc, :no])
                            - cp.einsum('ji,xju->xiu', fockA_hf[:nc, :nc], zs[:, :nc, :no])
                        ) / (2 * si - 1)
                        # ov1 - ov1 (𝑎𝑢|𝑣𝑏)
                        # vs[:,nc:,no:] += -v1_ov1_j[:,nc:,no:]/(2*si-1)+\
                        #                  (cp.einsum('uv,ab,xvb->xua',iden_O,fockB_hf[nc+no:,nc+no:],zs[:,nc:,no:])-\
                        #                   cp.einsum('uv,ab,xvb->xua',iden_O,fockA_hf[nc+no:,nc+no:],zs[:,nc:,no:]))/(2*si-1)
                        v1mo[:, nc:, no:] += fglobal * -v1_ov1_j[:, nc:, no:] / (2 * si - 1) + fglobal * (
                            cp.einsum('ab,xub->xua', fockB_hf[nc + no:, nc + no:], zs[:, nc:, no:])
                            - cp.einsum('ab,xub->xua', fockA_hf[nc + no:, nc + no:], zs[:, nc:, no:])
                        ) / (2 * si - 1)

                    if self.X > 1:
                        # cv1 - co1
                        # vs[:,:nc,no:] += factor1*(-v1_co1_k[:,:nc,no:] + cp.einsum('ij,av,xjv->xia',iden_C,fockB_hf[nc+no:,nc:nc+no],zs[:,:nc,:no]))
                        v1mo[:, :nc, no:] += fglobal * factor1 * (
                            - v1_co1_k[:, :nc, no:]
                            + cp.einsum('av,xiv->xia', fockB_hf[nc + no:, nc:nc + no], zs[:, :nc, :no])
                        )
                        # co1 - cv1
                        # vs[:,:nc,:no] += factor1*(-v1_cv1_k[:,:nc,:no] + cp.einsum('ij,av,xia->xjv',iden_C,fockB_hf[nc+no:,nc:nc+no],zs[:,:nc,no:]))
                        v1mo[:, :nc, :no] += fglobal * factor1 * (
                            - v1_cv1_k[:, :nc, :no]
                            + cp.einsum('av,xja->xjv', fockB_hf[nc + no:, nc:nc + no], zs[:, :nc, no:])
                        )
                        # cv1 - ov1
                        # vs[:,:nc,no:] += factor1*(-v1_ov1_k[:,:nc,no:] - cp.einsum('ab,vi,xvb->xia',iden_V,fockA_hf[nc:nc+no,:nc],zs[:,nc:,no:]))
                        v1mo[:, :nc, no:] += fglobal * factor1 * (
                            - v1_ov1_k[:, :nc, no:]
                            - cp.einsum('vi,xva->xia', fockA_hf[nc:nc + no, :nc], zs[:, nc:, no:])
                        )
                        # ov1 - cv1
                        # vs[:,nc:,no:] += factor1*(-v1_cv1_k[:,nc:,no:] - cp.einsum('ab,vi,xia->xvb',iden_V,fockA_hf[nc:nc+no,:nc],zs[:,:nc,no:]))
                        v1mo[:, nc:, no:] += fglobal * factor1 * (
                            -v1_cv1_k[:, nc:, no:]
                            - cp.einsum('vi,xib->xvb', fockA_hf[nc:nc + no, :nc], zs[:, :nc, no:])
                        )
                        # co1 - ov1
                        v1mo[:, :nc, :no] += fglobal * (v1_ov1_j[:, :nc, :no] - v1_ov1_k[:, :nc, :no]) / (2 * si - 1)
                        # ov1 - co1
                        v1mo[:, nc:, no:] += fglobal * (v1_co1_j[:, nc:, no:] - v1_co1_k[:, nc:, no:]) / (2 * si - 1)

                    if self.X > 2:
                        # cv1 - oo1
                        # vs[:,:nc,no:] += foo*(-(factor2-1)*(v1_oo1_k[:,:nc,no:]) + \
                        #                  (factor2/(2*si))*(cp.einsum('vw,ia,xvw->xia',iden_O,fockB_hf[:nc,nc+no:],zs[:,nc:,:no])-\
                        #                                    cp.einsum('vw,ia,xvw->xia',iden_O,fockA_hf[:nc,nc+no:],zs[:,nc:,:no])))
                        v1mo[:, :nc, no:] += fglobal * foo * (
                            -(factor2 - 1) * (v1_oo1_k[:, :nc, no:])
                            + (factor2 / (2 * si)) * (
                                    cp.einsum('ia,xvv->xia', fockB_hf[:nc, nc + no:], zs[:, nc:, :no])
                                    - cp.einsum('ia,xvv->xia', fockA_hf[:nc, nc + no:], zs[:, nc:, :no])
                            )
                        )
                        v1mo[:, nc:, :no] += fglobal * foo * (
                            -(factor2 - 1) * (v1_cv1_k[:, nc:, :no])
                            + (factor2 / (2 * si)) * (
                                    cp.einsum('vw,ia,xia->xvw', iden_O, fockB_hf[:nc, nc + no:], zs[:, :nc, no:])
                                    - cp.einsum('vw,ia,xia->xvw', iden_O, fockA_hf[:nc, nc + no:], zs[:, :nc, no:])
                            )
                        )

                        # co1 - oo1
                        # vs[:,:nc,:no] += foo*(factor3*(-v1_oo1_k[:,:nc,:no]-cp.einsum('uv,iw,xwv->xiu',iden_O,fockA_hf[:nc,nc:nc+no],zs[:,nc:,:no]))+\
                        #                 factor4*cp.einsum('vw,iu,xvw->xiu',iden_O,fockB_hf[:nc,nc:nc+no],zs[:,nc:,:no]))
                        v1mo[:, :nc, :no] += fglobal * foo * (
                            factor3 * (
                                -v1_oo1_k[:, :nc, :no]
                                - cp.einsum('iw,xwu->xiu', fockA_hf[:nc, nc:nc + no], zs[:, nc:, :no])
                            )
                            + factor4 * cp.einsum('vw,iu,xvw->xiu', iden_O, fockB_hf[:nc, nc:nc + no],
                                                  zs[:, nc:, :no])
                        )
                        # vs[:,nc:,:no] += foo*(factor3*(-v1_co1_k[:,nc:,:no]-cp.einsum('uv,iw,xiu->xwv',iden_O,fockA_hf[:nc,nc:nc+no],zs[:,:nc,:no]))+\
                        #                 factor4*cp.einsum('vw,iu,xiu->xvw',iden_O,fockB_hf[:nc,nc:nc+no],zs[:,:nc,:no]))
                        v1mo[:, nc:, :no] += fglobal * foo * (
                            factor3 * (
                                -v1_co1_k[:, nc:, :no]
                                - cp.einsum('iw,xiv->xwv', fockA_hf[:nc, nc:nc + no], zs[:, :nc, :no])
                            )
                            + factor4 * cp.einsum('vw,iu,xiu->xvw', iden_O, fockB_hf[:nc, nc:nc + no], zs[:, :nc, :no])
                        )
                        # ov1 - oo1
                        # vs[:,nc:,no:] += foo*(factor3*(-v1_oo1_k[:,nc:,no:]+cp.einsum('wu,av,xwv->xua',iden_O,fockB_hf[nc+no:,nc:nc+no],zs[:,nc:,:no]))-\
                        #                 factor4*(cp.einsum('vw,au,xvw->xua',iden_O,fockA_hf[nc+no:,nc:nc+no],zs[:,nc:,:no])))
                        v1mo[:, nc:, no:] += fglobal * foo * (
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
                        v1mo[:, nc:, :no] += fglobal * foo * (
                            factor3 * (
                                -v1_ov1_k[:, nc:, :no]
                                + cp.einsum('av,xwa->xwv', fockB_hf[nc + no:, nc:nc + no], zs[:, nc:, no:])
                            )
                            - factor4 * (
                                cp.einsum('vw,au,xua->xwv', iden_O, fockA_hf[nc + no:, nc:nc + no], zs[:, nc:, no:])
                            )
                        )
            elif isinstance(mf, uhf.UHF):
                v1mo += zs * e_ia

            nz = zs.shape[0]
            hx = v1mo.reshape(nz, -1)
            if self.re:
                if self.extype == 0:
                    new_hx = hx.copy()
                elif self.extype == 1:
                    new_hx = cp.zeros_like(zs0)
                    new_hx[:, :nc * (no + nv)] += hx[:, :nc * (no + nv)]
                    oo = cp.zeros((cp.array(zs0).shape[0], no * no))
                    for i in range(no):
                        oo[:, i * no:(i + 1) * no] = hx[:, (nc + i) * (no + nv):(nc + i) * (no + nv) + no]
                    new_oo = cp.einsum('xy,nx->ny', self.vects, oo)  # no*no-1
                    for i in range(no - 1):
                        new_hx[:, (nc + i) * (no + nv):(nc + i) * (no + nv) + no] = new_oo[:, i * no:(i + 1) * no]
                        new_hx[:, (nc + i) * (no + nv) + no:(nc + i) * (no + nv) + no + nv] \
                            = hx[:, (nc + i) * (no + nv) + no:(nc + i) * (no + nv) + no + nv]
                    new_hx[:, (nc + no - 1) * (no + nv):(nc + no - 1) * (no + nv) + no - 1] = new_oo[:, (no - 1) * no:]
                    new_hx[:, (nc + no - 1) * (no + nv) + no - 1:] = hx[:, (nc + no - 1) * (no + nv) + no:]
            else:
                new_hx = hx.copy()
            cp.cuda.Stream.null.synchronize()
            tdAdv1 = time.perf_counter()  # time of A matrix Davidson
            self.tc.dAdv += tdAdv1 - tdAdv0

            return new_hx

        return vind, hdiag

    # # pyscf-forge code
    # def gen_vind(self):
    #     '''
    #     Generate function to compute A*x for spin-flip TDDFT case.
    #     '''
    #     mf = self.mf
    #     mo_energy = self.mo_energy
    #     # TODO(WHB): SymAdaptedUHF.mo_coeff may cause error, refer gpu4pyscf/tdscf/uhf.py
    #     mo_coeff = self.mo_coeff
    #     mo_occ = self.mo_occ
    #     assert isinstance(mf, scf.hf.SCF)
    #     assert mo_coeff[0].dtype == cp.float64
    #
    #     extype = self.extype
    #     if extype==0:
    #         occidxb = mo_occ[1] > 0
    #         viridxa = mo_occ[0] ==0
    #         orbob = mo_coeff[1][:,occidxb]
    #         orbva = mo_coeff[0][:,viridxa]
    #         orbov = (orbob, orbva)
    #         ndim = (int(occidxb.sum()), int(viridxa.sum()))
    #         if np.allclose(mo_coeff[0], mo_coeff[1]):
    #             dm = mf.make_rdm1()
    #             vhf = mf.get_veff(mf.mol, dm)
    #             h1e = mf.get_hcore()
    #             fock_a = h1e + vhf[0]
    #             fock_b = h1e + vhf[1]
    #             focko = orbob.conj().T @ fock_b @ orbob
    #             fockv = orbva.conj().T @ fock_a @ orbva
    #             hdiag = (fockv.diagonal()[None, :] - focko.diagonal()[:, None]).ravel()
    #         else:
    #             e_ia = (mo_energy[0][None, viridxa] - mo_energy[1][occidxb, None])
    #             hdiag = e_ia.ravel()
    #     elif extype==1:
    #         occidxa = mo_occ[0] > 0
    #         viridxb = mo_occ[1] == 0
    #         orboa = mo_coeff[0][:, occidxa]
    #         orbvb = mo_coeff[1][:, viridxb]
    #         orbov = (orboa, orbvb)
    #         ndim = (int(occidxa.sum()), int(viridxb.sum()))
    #         if np.allclose(mo_coeff[0], mo_coeff[1]):
    #             dm = mf.make_rdm1()
    #             vhf = mf.get_veff(mf.mol, dm)
    #             h1e = mf.get_hcore()
    #             fock_a = h1e + vhf[0]
    #             fock_b = h1e + vhf[1]
    #             focko = orboa.conj().T @ fock_a @ orboa
    #             fockv = orbvb.conj().T @ fock_b @ orbvb
    #             hdiag = (fockv.diagonal()[None, :] - focko.diagonal()[:, None]).ravel()
    #         else:
    #             e_ia = (mo_energy[1][None, viridxb] - mo_energy[0][occidxa, None])
    #             hdiag = e_ia.ravel()
    #     else:
    #         raise ValueError(
    #             f"Invalid extype = {extype}. "
    #             "extype must be 0 (beta->alpha spin flip up) or 1 (alpha->beta spin flip down)."
    #         )
    #
    #     # TODO: change the response function
    #     vresp, get_usf_time = gen_uhf_response_sf(
    #         mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0, collinear=self.collinear,
    #         collinear_samples=self.collinear_samples)
    #
    #     def vind(zs):
    #         zs = cp.asarray(zs).reshape(-1, *ndim)
    #         orbo, orbv = orbov
    #         mo1 = contract('xov,pv->xpo', zs, orbv)
    #         dms = contract('xpo,qo->xpq', mo1, orbo.conj())
    #         # dms = tag_array(dms, mo1=mo1, occ_coeff=orbo)  # TODO: bug
    #         dms = tag_array(dms, mo1=mo1, occ_coeff=0.5*orbo)
    #         v1ao = vresp(dms)
    #         v1mo = contract('xpq,qo->xpo', v1ao, orbo)
    #         v1mo = contract('xpo,pv->xov', v1mo, orbv.conj())
    #         if np.allclose(mo_coeff[0], mo_coeff[1]):
    #             v1mo += contract('ab,xib->xia', fockv, zs)
    #             v1mo -= contract('ji,xja->xia', focko, zs)
    #         else:
    #             v1mo += zs * e_ia
    #         return v1mo.reshape(len(v1mo), -1)
    #
    #     return vind, hdiag, get_usf_time

    def deal_v_davidson(self):
        if self.extype == 0:
            return self.v

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
                if self.extype == 0:
                    for i in range(self.no):
                        oo[state, i, :] += tmp_data[passed + i * nvir:passed + i * nvir + self.no]
                        ov[state, i, :] += tmp_data[passed + i * nvir + self.no:passed + i * nvir + self.no + self.nv]
                elif self.extype == 1:
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
            e = e.reshape(-1,1)
            diagd = hdiag - (e-self.level_shift)
            diagd = cp.where(abs(diagd) < threshold_t, cp.sign(diagd)*threshold_t, diagd)
            a_size = x.shape[1]//2
            diagd[:,a_size:] = diagd[:,a_size:]*(-1)
            return x/diagd
        return precond

    def davidson_process(self):
        if self.gpu_davidson:
            print('use gpu Davidson')
            vind, hdiag = self.gen_vind()

            tih0 = time.perf_counter()  # init guess and hdiag
            precond = hdiag
            x0 = self.init_guess()
            # Keep all eigenvalues as SF-TDDFT allows triplet to singlet
            # "dexcitation"
            def all_eigs(w, v, nroots, envs):
                return w, v, np.arange(w.size)

            if not callable(precond):
                precond = self.get_precond(precond)
            cp.cuda.Stream.null.synchronize()
            tih1 = time.perf_counter()
            self.tc.ih = tih1 - tih0

            print('remember return Davidcyc in gpu4pyscf/tdscf/_lr_eig.py')
            tdv0 = time.perf_counter()
            converged, e, x1, Davidcyc = lr_eigh(
                vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates,pick=all_eigs,max_cycle=self.max_cycle
            )
            cp.cuda.Stream.null.synchronize()
            tdv1 = time.perf_counter()
            self.tc.dv = tdv1 - tdv0

            print('The GPU version of the Davidson iteration has not yet returned the number of iterations.')
            self.Davidcyc = Davidcyc  # TODO(WHB): gpu davidson do not return icyc
            self.converged = converged
            self.e = e
            self.v = cp.array(x1).T
            self.v = self.deal_v_davidson()
            print('Davidson each state Converged ', converged)
        else:
            print('use cpu Davidson')
            vind, hdiag = self.gen_vind()
            tih0 = time.perf_counter()  # init guess and hdiag
            x0 = self.init_guess()
            cp.cuda.Stream.null.synchronize()
            tih1 = time.perf_counter()
            self.tc.ih = tih1 - tih0

            tdv0 = time.perf_counter()
            converged, e, x1, Davidcyc = Davidson.davidson1(
                vind, x0, hdiag, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates, max_cycle=self.max_cycle
            )
            cp.cuda.Stream.null.synchronize()
            tdv1 = time.perf_counter()
            self.tc.dv = tdv1 - tdv0

            self.Davidcyc = Davidcyc
            self.converged = converged
            self.e = e
            self.v = cp.array(x1).T

            tv0 = time.perf_counter()  # init guess and hdiag
            self.v = self.deal_v_davidson()
            cp.cuda.Stream.null.synchronize()
            tv1 = time.perf_counter()
            self.tc.v = tv1 - tv0
            print('Davidson each state Converged ', converged)
        return

    def kernel(self):
        t0 = time.perf_counter()
        self.davidson_process()
        print('Davidson iteration {:2} times'.format(self.Davidcyc[0]))
        print("=" * 50)

        tana0 = time.perf_counter()
        self.analyze()
        cp.cuda.Stream.null.synchronize()
        tana1 = time.perf_counter()
        self.tc.ana = tana1 - tana0
        print("=" * 50)

        print(f'{"num":>4} {"energy":>8}')
        for ni, ei in zip(range(self.nstates), self.e*unit.ha2eV):
            print(f'{ni + 1:4d} {ei:8.4f}')
        print('='*50)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        self.tc.total = t1 - t0

        print('initial time consuming')
        print('    perpare A calculation use              {:12.3f} s'.format(self.tc.Ap))
        print('        Fock (UKS is mo energy) use        {:12.3f} s'.format(self.tc.Ap_f))
        print('        vxc kernel use                     {:12.3f} s'.format(self.tc.Ap_k))
        print('    perpare delta A calculation (Fock) use {:12.3f} s'.format(self.tc.dAp))
        # print('    init guess and hdiag use         {:12.3} s'.format(self.tc.ih))
        # print('    get removed vector use           {:12.3} s'.format(self.tc.gv))
        print('Davidson process time consuming')
        print('    calculate Ax use                       {:12.3f} s'.format(self.tc.Adv))
        print('        calculate A vxc use                {:12.3f} s'.format(self.tc.A_vxc))
        print('        calculate A get_k use              {:12.3f} s'.format(self.tc.A_gk))
        print('    calculate delta Ax use                 {:12.3f} s'.format(self.tc.dAdv))
        print('        delta Ax get_jk use                {:12.3f} s'.format(self.tc.dA_gjk))
        print('    other davidson process use             {:12.3f} s'.format(self.tc.dv - self.tc.Adv - self.tc.dAdv))
        print('each Davidson iteration use                {:12.3f} s'.format(self.tc.dv / self.Davidcyc[0]))
        # print('deal Davidson eigen vector use       {:12.3} s'.format(self.tc.v))
        # print('analyze use                          {:12.3} s'.format(self.tc.ana))
        print('XSF-TDA total use                          {:12.3f} s'.format(self.tc.total))
        return self.e * unit.ha2eV, self.v

    def analyze(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        syms = []
        try:
            ground_sym = self.mf.get_wfnsym()
            print('ground_sym ', ground_sym)
        except:
            pass
        for nstate in range(self.nstates):
            print("-"*50)
            m_excited = 0.
            orb1 = None
            orb2 = None
            value = self.v[:, nstate]
            # print('self.vects.shape', self.vects.shape)
            # print(len(value))
            x_cv_ab = value[:nc * nv].reshape(nc, nv)
            if self.extype == 1:
                x_co_ab = value[nc * nv:nc * nv + nc * no].reshape(nc, no)
                x_ov_ab = value[nc * nv + nc * no:nc * nv + nc * no + no * nv].reshape(no, nv)
                if self.re:
                    x_oo_ab = (self.vects @ value[nc * nv + nc * no + no * nv:].reshape(-1, 1)).reshape(no, no)
                else:
                    x_oo_ab = value[nc * nv + nc * no + no * nv:].reshape(no, no)

            # Delta \langle S^2 \rangle
            if self.X == 0 and self.extype == 1:
                if isinstance(self.mf, rohf.ROHF):
                    Dp_ab = 0.
                    Dp_ab += sum(sum(x_cv_ab * x_cv_ab)) - sum(sum(x_oo_ab * x_oo_ab))
                    for i in range(no):
                        for j in range(no):
                            Dp_ab += x_oo_ab[i, i] * x_oo_ab[j, j]
                    ds2 = -2 * no/2 + 1 + Dp_ab
                elif isinstance(self.mf, uhf.UHF):
                    P_ab = self.deltaS2_U(nstate)
                    ds2 = P_ab - self.no + 1
                print(
                    f'D{nstate + 1}' + r"    w:" + f'{self.e[nstate] * unit.ha2eV:10.4f} eV'
                    + r"    deltaS2:" + f'{ds2:10.4f}'
                )
            else:
                print(f'D{nstate + 1}' + r"    w:" + f'{self.e[nstate] * unit.ha2eV:10.4f} eV')
            for o, v in zip(*np.where(abs(x_cv_ab) > 0.1)):
                if abs(x_cv_ab[o, v]) > m_excited:
                    m_excited = abs(x_cv_ab[o, v])
                    orb1 = o
                    orb2 = v + self.nc + self.no
                print(
                    f'{100 * x_cv_ab[o, v] ** 2:3.0f}% CV(ab) {o + 1}a -> {v + 1 + self.nc + self.no}b {x_cv_ab[o, v]:10.5f} ')
            if self.extype == 1:
                for o, v in zip(*np.where(abs(x_co_ab) > 0.1)):
                    if abs(x_co_ab[o, v]) > m_excited:
                        m_excited = abs(x_co_ab[o, v])
                        orb1 = o
                        orb2 = v + self.nc
                    print(f'{100 * x_co_ab[o, v] ** 2:3.0f}% CO(ab) {o + 1}a -> {v + 1 + self.nc}b {x_co_ab[o, v]:10.5f} ')
                for o, v in zip(*np.where(abs(x_ov_ab) > 0.1)):
                    if abs(x_ov_ab[o, v]) > m_excited:
                        m_excited = abs(x_ov_ab[o, v])
                        orb1 = o + self.nc
                        orb2 = v + self.nc + self.no
                    print(
                        f'{100 * x_ov_ab[o, v] ** 2:3.0f}% OV(ab) {o + self.nc + 1}a -> {v + 1 + self.nc + self.no}b {x_ov_ab[o, v]:10.5f} ')
                for o, v in zip(*np.where(abs(x_oo_ab) > 0.1)):
                    if abs(x_oo_ab[o, v]) > m_excited:
                        m_excited = abs(x_oo_ab[o, v])
                        orb1 = o + self.nc
                        orb2 = v + self.nc
                    print(
                        f'{100 * x_oo_ab[o, v] ** 2:3.0f}% OO(ab) {o + nc + 1}a -> {v + 1 + self.nc}b {x_oo_ab[o, v]:10.5f} ')
            if self.mol.groupname != 'C1':
                sym = self.calculate_irrep(orb1, orb2)
            else:
                sym = 'A'
            syms.append(sym)
            print('major excited configuration corresponding {}'.format(sym))

    def calculate_irrep(self, orb1, orb2):
        orb_sym = self.mf.get_orbsym(self.mf.mo_coeff)
        ground_sym = self.mf.get_wfnsym()
        # print('orb_sym',orb_sym)
        # print(orb1,orb2)
        if self.type_u:
            orb1_sym = np.array([orb_sym[0][orb1]])
            orb2_sym = np.array([orb_sym[1][orb2]])
        else:
            orb1_sym = np.array([orb_sym[orb1]])
            orb2_sym = np.array([orb_sym[orb2]])
        direct_s = direct_prod(orb1_sym, orb2_sym, self.mol.groupname)
        direct_s = direct_prod(direct_s[0], np.array(ground_sym), self.mol.groupname)
        if direct_s[0][0] >= len(self.mol.irrep_name):
            return 'A'
        else:
            return self.mol.irrep_name[direct_s[0][0]]

    def deltaS2_U(self, nstate):
        mf = self.mf
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        nc = self.nc
        no = self.no
        nv = self.nv
        occidx_a = self.occidx_a
        occidx_b = self.occidx_b
        viridx_a = self.viridx_a
        viridx_b = self.viridx_b
        mooa = mo_coeff[0][:,occidx_a]
        moob = mo_coeff[1][:,occidx_b]
        mova = mo_coeff[0][:,viridx_a]
        movb = mo_coeff[1][:,viridx_b]
        ovlp = mf.get_ovlp()
        #S = self.mol.intor('int1e_ovlp')
        #print('s ovlp',np.allclose(S,ovlp))
        # spin transfer matrix
        sab_oo = reduce(cp.dot, (mooa.conj().T, ovlp, moob))
        #Sccba = np.einsum('pq,pi,qj->ij', ovlp, moob, mooa)
        sba_oo = sab_oo.conj().T
        #print('allclose ',np.allclose(sba_oo,Sccba))
        sab_vo = reduce(cp.dot, (mova.conj().T, ovlp, moob))
        sba_vo = reduce(cp.dot, (movb.conj().T, ovlp, mooa))
        #x_ba = self.v[:,nstate].reshape((self.nc+self.no,self.no+self.nv)).transpose(1,0)
        value = self.v[:, nstate]

        x_cv_ab = value[:nc*nv].reshape(nc,nv)
        x_co_ab = value[nc*nv:nc*nv+nc*no].reshape(nc,no)
        x_ov_ab = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no,nv)
        x_oo_ab = value[nc*nv+nc*no+no*nv:].reshape(no,no)
        tmp1 = cp.hstack([x_co_ab,x_cv_ab])
        tmp2 = cp.hstack([x_oo_ab,x_ov_ab])
        x_ba = cp.concatenate([tmp1,tmp2],axis=0).transpose(1,0)
        P_ab = cp.einsum('ai,aj,jk,ki',x_ba.conj(),x_ba,sba_oo.T.conj(),sba_oo)\
                  -cp.einsum('ai,bi,kb,ak',x_ba.conj(),x_ba,sba_vo.T.conj(),sba_vo)\
                  +cp.einsum('ai,bj,jb,ai',x_ba.conj(),x_ba,sba_vo.T.conj(),sba_vo)
        return P_ab


if __name__ == '__main__':
    # path = '/home/lenovo2/users/zhw/TDDFT/SFTDA/TADF/mewes_35/geometries/30/'
    mol = gto.M(
        # atom='geometries/invest15/geom006.xyz',
        # atom='geometries/mewes_35/geom01.xyz',
        # atom=path+'geom.xyz',
        # atom = atom.hhcrqpp2,
        atom=atom.n2_,
        # atom='H 0 0 0; F 0 0 1.0',
        # atom = 'O 0 0 0; O 0 0 2.07',
        # atom=atom.ttm_vacuum,
        basis='def2-svp',
        # basis='def2-tzvp',
        # basis='cc-pvdz',
        # basis = '6-31g**',
        # unit = 'B',
        charge=1,
        spin=3,
        verbose=4,
        # symmetry = 'C2v',
    )
    tscf0 = time.perf_counter()
    mf = dft.ROKS(mol).density_fit()
    # mf = uks.UKS(mol)
    # mf = rks.RKS(mol).density_fit()
    # mf = uhf.UHF(mol).density_fit()
    # mf = roks.ROKS(mol).density_fit().SMD()
    # mf.with_solvent.eps = 35.688  # for Acetonitrile
    # mf.xc = 'bhandhlyp'
    # mf.xc = 'hf'
    # mf.xc = 'pbe0'
    # mf.xc = 'camb3lyp'
    mf.xc = 'b3lyp'
    # mf.xc = 'tpssh'
    # mf.chkfile = 'n2.chk'
    # mf.level_shift = 0.5
    # mf.damp = 0.3
    # mf.init_guess = 'atom'
    # mf.init_guess = '1e'
    # mf.init_guess = 'huckel'
    # mf.init_guess = 'vasp'
    # mf.init_guess = 'chk'
    # dm = mf.from_chk('chk/mol23_senior.chk')
    mf.max_cycle = 200
    # mf.conv_tol = 1e-10
    # mf.conv_tol_grad = 1e-6
    # mf.grids.atom_grid = (99,590)
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    assert mf.converged
    cp.cuda.Stream.null.synchronize()
    tscf1 = time.perf_counter()
    print('scf use      {:8.4f} s'.format(tscf1 - tscf0))
    print('=' * 50)
    if isinstance(mf, rohf.ROHF):
        print('num.orb    mo_energy     mo_occ')
        for me, o in zip(enumerate(mf.mo_energy), mf.mo_occ):
            ind, moei = me
            print(f'{ind + 1:5d}    {moei:10.6f}    {o:8.3f}')
        print('=' * 50)
        print('there are {:8} orbitals (basis)'.format(len(mf.mo_occ)))
    elif isinstance(mf, uhf.UHF):
        print('num.orb    mo_energy_a   mo_energy_b    mo_occ_a    mo_occ_b')
        for me, o in zip(enumerate(mf.mo_energy.T), mf.mo_occ.T):
            ind, moei = me
            print(f'{ind + 1:5d}    {moei[0]:10.6f}    {moei[1]:10.6f}    {o[0]:8.3f}    {o[1]:8.3f}')
        print('=' * 50)
        print('there are {:8} orbitals (basis)'.format(mf.mo_occ.shape[1]))

    sf_tda = XSF_TDA_GPU(
        mf, collinear='mcol', nstates=7, extype=1,
        gpu_davidson=False, collinear_samples=20
    )
    e0, values = sf_tda.kernel()

