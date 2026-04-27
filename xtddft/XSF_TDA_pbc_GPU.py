"""GPU spin-adapted XSF-TDA for Gamma-point PBC gpu4pyscf references."""

from functools import reduce
import math
from types import MethodType

import cupy as cp
import numpy as np
import scipy
from pyscf import lib

from .SF_TDA_pbc_GPU import (
    _asarray,
    _asnumpy,
    _as_spin_potential,
    _ensure_gamma_df,
    _get_gamma_kpt,
    _get_hcore_gamma,
    _get_k_gamma,
    _get_mo_fock,
    _get_veff_gamma,
    gen_response_sf,
    mf_info,
)

au2ev = 27.21138505


def make_gpu_roks_reference(roks_mf, gpu_mf=None, with_df=True, copy_grids=True):
    """Wrap a CPU PBC ROKS reference in a gpu4pyscf UKS object.

    gpu4pyscf does not implement PBC ROKS SCF.  This helper keeps the ROKS
    orbitals in their restricted 2D/1D form so XSF_TDA still recognizes a
    spin-adapted reference, while all response/J/K calls are delegated to a
    gpu4pyscf PBC UKS object.
    """
    from gpu4pyscf.pbc import dft, df as gpudf

    if gpu_mf is None:
        gpu_mf = dft.UKS(roks_mf.cell).to_gpu()
    elif hasattr(gpu_mf, "to_gpu"):
        gpu_mf = gpu_mf.to_gpu()

    for name in ("xc", "nlc", "max_memory", "verbose", "exxdiv", "e_tot", "converged"):
        if hasattr(roks_mf, name):
            try:
                setattr(gpu_mf, name, getattr(roks_mf, name))
            except Exception:
                pass

    if with_df is True:
        gpu_mf.with_df = gpudf.GDF(roks_mf.cell)
    elif with_df is not None:
        gpu_mf.with_df = with_df

    if copy_grids and hasattr(roks_mf, "grids") and hasattr(gpu_mf, "grids"):
        for name in ("level", "atom_grid", "prune", "radi_method", "becke_scheme"):
            if hasattr(roks_mf.grids, name):
                try:
                    setattr(gpu_mf.grids, name, getattr(roks_mf.grids, name))
                except Exception:
                    pass
        if getattr(roks_mf.grids, "coords", None) is not None:
            try:
                gpu_mf.grids.coords = cp.asarray(roks_mf.grids.coords)
                gpu_mf.grids.weights = cp.asarray(roks_mf.grids.weights)
            except Exception:
                pass

    gpu_mf.mo_energy = cp.asarray(roks_mf.mo_energy)
    gpu_mf.mo_coeff = cp.asarray(roks_mf.mo_coeff)
    gpu_mf.mo_occ = cp.asarray(roks_mf.mo_occ)
    gpu_mf._xtddft_reference_type = "ROKS"

    def _roks_make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        del kwargs
        old_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        old_occ = self.mo_occ if mo_occ is None else mo_occ
        coeff0, occ0 = self.mo_coeff, self.mo_occ
        try:
            self.mo_coeff = old_coeff
            self.mo_occ = old_occ
            _, occ, coeff = mf_info(self)
        finally:
            self.mo_coeff = coeff0
            self.mo_occ = occ0
        return _make_spin_dm(coeff, occ)

    gpu_mf.make_rdm1 = MethodType(_roks_make_rdm1, gpu_mf)
    return gpu_mf


def find_top10_abs_numpy(matrix):
    mat = _asnumpy(matrix)
    abs_matrix = abs(mat)
    flat_indices = np.argpartition(abs_matrix, -10, axis=None)[-10:]
    flat_indices = flat_indices[np.argsort(-abs_matrix.ravel()[flat_indices])]
    rows, cols = np.unravel_index(flat_indices, mat.shape)
    return [(mat[r, c], r, c) for r, c in zip(rows, cols)]


def _make_spin_dm(mo_coeff, mo_occ):
    return cp.asarray([
        (mo_coeff[spin] * mo_occ[spin]) @ mo_coeff[spin].conj().T
        for spin in range(2)
    ])


def _get_jk_gamma(mf, dm, hermi=0, batch=False):
    dm = _asarray(dm)
    _ensure_gamma_df(mf)
    with_df = getattr(mf, "with_df", None)
    gamma_df = bool(getattr(with_df, "is_gamma_point", False))
    if (
        batch and dm.ndim == 3
        and not gamma_df
    ):
        vjs = []
        vks = []
        for x in dm:
            vj, vk = _get_jk_gamma(mf, x, hermi=hermi, batch=False)
            vjs.append(vj)
            vks.append(vk)
        return cp.stack(vjs), cp.stack(vks)

    if dm.ndim == 3 and dm.shape[0] != 2 and not batch:
        return _get_jk_gamma(mf, dm, hermi=hermi, batch=True)

    if gamma_df:
        return with_df.get_jk(
            dm, hermi=hermi, kpts=None, with_j=True, with_k=True,
            exxdiv=getattr(mf, "exxdiv", None)
        )

    try:
        return mf.get_jk(mf.cell, dm, hermi=hermi, kpt=_get_gamma_kpt(mf))
    except TypeError:
        return mf.get_jk(mf.cell, dm, hermi=hermi)


def _get_hf_fock_mo_gamma(mf, mo_coeff, mo_occ):
    dm = _make_spin_dm(mo_coeff, mo_occ)
    vj, vk = _get_jk_gamma(mf, dm, hermi=1)
    vj, vk = _asarray(vj), _asarray(vk)
    coul = vj if vj.ndim == 2 else vj[0] + vj[1]
    if vk.ndim == 2:
        vk = cp.asarray([vk, vk])
    vhf = coul - vk
    h1e = _asarray(_get_hcore_gamma(mf))
    return (
        mo_coeff[0].conj().T @ (h1e + vhf[0]) @ mo_coeff[0],
        mo_coeff[1].conj().T @ (h1e + vhf[1]) @ mo_coeff[1],
    )


class XSF_TDA_pbc_GPU:
    def __init__(self, mf, SA=None, davidson=True, method=0,
                 collinear_samples=60, calculate_sp=False):
        """Gamma-point PBC GPU XSF-TDA solver.

        SA=0: SF-TDA
        SA=1: add diagonal delta-A blocks
        SA=2: add all delta-A except OO blocks
        SA=3: full delta-A
        """
        print("method=0 (default) ALDA0, method=1 multicollinear, method=2 collinear/no-XC")

        mo_energy, mo_occ, mo_coeff = mf_info(mf)
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.type_u = _asarray(mf.mo_coeff).ndim == 3
        if SA is None:
            self.SA = 0 if self.type_u else 3
        else:
            self.SA = SA

        _get_gamma_kpt(mf)
        self.cell = mf.cell
        self.nao = self.cell.nao_nr()
        self.mf = mf
        self.collinear_samples = collinear_samples
        self.davidson = davidson
        self.method = method
        try:
            _, dsp1 = mf.spin_square()
        except Exception:
            dsp1 = self.cell.spin + 1
        self.ground_s = (dsp1 - 1) / 2

        self.occidx_a = cp.where(self.mo_occ[0] == 1)[0]
        self.viridx_a = cp.where(self.mo_occ[0] == 0)[0]
        self.occidx_b = cp.where(self.mo_occ[1] == 1)[0]
        self.viridx_b = cp.where(self.mo_occ[1] == 0)[0]
        self.orbo_a = self.mo_coeff[0][:, self.occidx_a]
        self.orbv_a = self.mo_coeff[0][:, self.viridx_a]
        self.orbo_b = self.mo_coeff[1][:, self.occidx_b]
        self.orbv_b = self.mo_coeff[1][:, self.viridx_b]
        self.nocc_a = int(self.orbo_a.shape[1])
        self.nvir_a = int(self.orbv_a.shape[1])
        self.nocc_b = int(self.orbo_b.shape[1])
        self.nvir_b = int(self.orbv_b.shape[1])
        self.nmo_a = self.nocc_a + self.nvir_a
        self.nc = self.nocc_b
        self.nv = self.nvir_a
        self.no = self.nocc_a - self.nocc_b

        try:
            ni = self.mf._numint
            self.omega, self.alpha, self.hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.cell.spin)
            print("Omega, alpha, hyb", self.omega, self.alpha, self.hyb)
        except Exception:
            self.omega = 0
            self.alpha = 0
            self.hyb = 1.0

        self.calculate_sp = calculate_sp
        if calculate_sp:
            self.get_sp()

    def _default_fglobal(self, d_lda=0.3, fit=True):
        if self.omega == 0:
            cx = self.hyb
        else:
            cx = self.hyb + (self.alpha - self.hyb) * math.erf(self.omega)
        fglobal = (1 - d_lda) * cx + d_lda
        if self.method == 1 and fit:
            fglobal = fglobal * 4 * (cx - 0.5) ** 2
        return fglobal

    def get_sp(self):
        vresp = gen_response_sf(
            self.mf, hermi=0, method=self.method,
            collinear_samples=self.collinear_samples
        )
        h = _asarray(self.mf.mo_coeff)[:, self.nc:self.nc+1]
        dm1 = h @ h.T
        h_ao = vresp(dm1.reshape((1, dm1.shape[0], dm1.shape[1])))
        h_mo = _asarray(self.mf.mo_coeff).T @ h_ao[0] @ _asarray(self.mf.mo_coeff)
        lhhl = h_mo[self.nc + self.no, self.nc + self.no]

        k_ao = _get_k_gamma(self.mf, dm1)
        k_mo = _asarray(self.mf.mo_coeff).T @ k_ao @ _asarray(self.mf.mo_coeff)
        homo = k_mo[:self.nc, self.nc+self.no:].reshape((self.nc, self.nv))

        l = _asarray(self.mf.mo_coeff)[:, self.nc+1:self.nc+2]
        dm1 = l @ l.T
        l_ao = _get_k_gamma(self.mf, dm1)
        l_mo = _asarray(self.mf.mo_coeff).T @ l_ao @ _asarray(self.mf.mo_coeff)
        lumo = l_mo[:self.nc, self.nc+self.no:].reshape((self.nc, self.nv))

        print("=================================================")
        print(f"<LH|HL> is {float(lhhl.get()):9.6f}")
        print("Top 10 value in <iH|Ha>:")
        for i, item in enumerate(find_top10_abs_numpy(homo), start=1):
            print(f"{i}  {item[0]:9.6f}, CV is {item[1]+1,item[2]+self.nc+self.no+1}")
        print("Top 10 value in <iL|La>:")
        for i, item in enumerate(find_top10_abs_numpy(lumo), start=1):
            print(f"{i} {item[0]:9.6f}, CV is {item[1]+1,item[2]+self.nc+self.no+1}")
        print("=================================================")

    def get_vect(self):
        tmp_v = cp.zeros((self.no - 1, self.no))
        for i in range(1, self.no):
            factor = 1 / cp.sqrt((self.no - i + 1) * (self.no - i))
            tmp = [self.no - i] + [-1] * (self.no - i)
            tmp_v[i - 1][i - 1:] = cp.asarray(tmp) * factor
        self.vect = tmp_v.T
        vects = cp.eye(self.no * self.no)
        vects = vects[:, :-1]
        index = [0]
        for i in range(1, self.no):
            index.append(i * (self.no + 1))
        for i in range(self.vect.shape[1]):
            vects[0::self.no + 1, index[i]] = self.vect[:, i]
        return vects

    def remove(self):
        dim3 = self.nc * self.nv + self.nc * self.no + self.no * self.nv
        dim = self.A.shape[0]
        self.vects = self.get_vect()
        A = cp.zeros((dim - 1, dim - 1))
        A[:dim3, :dim3] = self.A[:dim3, :dim3]
        A[:dim3, dim3:] = self.A[:dim3, dim3:] @ self.vects
        A[dim3:, :dim3] = self.vects.T @ self.A[dim3:, :dim3]
        A[dim3:, dim3:] = self.vects.T @ self.A[dim3:, dim3:] @ self.vects
        return A

    def get_Amat(self, SA=None, foo=1.0, fglobal=None,
                 projected=None, d_lda=0.3, fit=True):
        """Compatibility dense matrix path via the CPU PBC implementation."""
        del foo, d_lda, fit
        from .XSF_TDA_pbc import XSF_TDA_pbc

        cpu_mf = self.mf.to_cpu() if hasattr(self.mf, "to_cpu") else self.mf
        cpu_solver = XSF_TDA_pbc(
            cpu_mf,
            SA=self.SA if SA is None else SA,
            davidson=False,
            method=self.method,
            collinear_samples=self.collinear_samples,
        )
        self.A = cp.asarray(cpu_solver.get_Amat(
            SA=SA, fglobal=fglobal, projected=projected
        ))
        return self.A

    def deltaS2_U(self, nstate):
        mo = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        occidxa = cp.where(mo_occ[0] > 0)[0]
        occidxb = cp.where(mo_occ[1] > 0)[0]
        viridxa = cp.where(mo_occ[0] == 0)[0]
        viridxb = cp.where(mo_occ[1] == 0)[0]
        mooa = mo[0][:, occidxa]
        moob = mo[1][:, occidxb]
        mova = mo[0][:, viridxa]
        movb = mo[1][:, viridxb]
        try:
            ovlp = self.mf.get_ovlp(self.cell, kpt=_get_gamma_kpt(self.mf))
        except TypeError:
            ovlp = self.mf.get_ovlp()
        sab_oo = reduce(cp.dot, (mooa.conj().T, ovlp, moob))
        sba_oo = sab_oo.conj().T
        sab_vo = reduce(cp.dot, (mova.conj().T, ovlp, moob))
        sba_vo = reduce(cp.dot, (movb.conj().T, ovlp, mooa))

        value = self.v[:, nstate]
        nc, no, nv = self.nc, self.no, self.nv
        x_cv = value[:nc*nv].reshape(nc, nv)
        x_co = value[nc*nv:nc*nv+nc*no].reshape(nc, no)
        x_ov = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no, nv)
        x_oo = value[nc*nv+nc*no+no*nv:].reshape(no, no)
        x_ba = cp.concatenate([cp.hstack([x_co, x_cv]), cp.hstack([x_oo, x_ov])], axis=0).T
        return (
            cp.einsum("ai,aj,jk,ki", x_ba.conj(), x_ba, sba_oo.T.conj(), sba_oo)
            - cp.einsum("ai,bi,kb,ak", x_ba.conj(), x_ba, sba_vo.T.conj(), sba_vo)
            + cp.einsum("ai,bj,jb,ai", x_ba.conj(), x_ba, sba_vo.T.conj(), sba_vo)
        )

    def analyse(self):
        nc, nv, no = self.nc, self.nv, self.no
        Ds = []
        for nstate in range(self.nstates):
            value = self.v[:, nstate]
            x_cv = value[:nc*nv].reshape(nc, nv)
            x_co = value[nc*nv:nc*nv+nc*no].reshape(nc, no)
            x_ov = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no, nv)
            if self.re:
                x_oo = (self.vects @ value[nc*nv+nc*no+no*nv:].reshape(-1, 1)).reshape(no, no)
            else:
                x_oo = value[nc*nv+nc*no+no*nv:].reshape(no, no)

            if self.SA == 0 and not self.type_u:
                dp_ab = cp.sum(x_cv*x_cv) - cp.sum(x_oo*x_oo) + cp.sum(cp.diag(x_oo))**2
                ds2 = -2 * self.ground_s + 1 + dp_ab
            elif self.type_u:
                ds2 = self.deltaS2_U(nstate) - self.no + 1
            else:
                ds2 = None
            if ds2 is None:
                print(f"Excited state {nstate+1} {float(self.e[nstate].get())*au2ev:10.5f} eV")
            else:
                Ds.append(float(ds2.get()))
                print(f"Excited state {nstate+1} {float(self.e[nstate].get())*au2ev:10.5f} eV D<S^2>={float(ds2.get()):3.2f}")

            for label, arr, off_o, off_v in (
                ("CV(ab)", x_cv, 1, self.nc + self.no + 1),
                ("CO(ab)", x_co, 1, self.nc + 1),
                ("OV(ab)", x_ov, nc + 1, self.nc + self.no + 1),
                ("OO(ab)", x_oo, nc + 1, self.nc + 1),
            ):
                for o, v in zip(*cp.where(abs(arr) > 0.05)):
                    oi, vi = int(o), int(v)
                    amp = float(arr[oi, vi].get())
                    print(f"{100*amp**2:5.2f}% {label} {oi+off_o}a -> {vi+off_v}b {amp:10.5f}")
            print("========================================")
        return Ds

    def init_guess(self, mf, nstates):
        mo_energy, mo_occ, mo_coeff = mf_info(mf)
        occidxa = cp.where(mo_occ[0] > 0)[0]
        viridxb = cp.where(mo_occ[1] == 0)[0]
        e_ia = (mo_energy[1][viridxb, None] - mo_energy[0][occidxa]).T.ravel()
        nov = int(e_ia.size)
        nstates = min(nstates, nov)
        e_threshold = cp.sort(e_ia)[nstates - 1] + 1e-5
        idx = cp.where(e_ia <= e_threshold)[0]
        x0 = cp.zeros((int(idx.size), nov))
        x0[cp.arange(int(idx.size)), idx] = 1.0
        if self.re:
            x0 = x0[:, :-1]
        return x0

    def gen_response_sf_delta_A(self, hermi=0, max_memory=None):
        del max_memory
        mf = self.mf

        def vind(dm1):
            return _get_jk_gamma(mf, _asarray(dm1), hermi=hermi, batch=True)

        return vind

    def gen_tda_operation_sf(self, foo, fglobal):
        mf = self.mf
        mo_energy, mo_occ, mo_coeff = self.mo_energy, self.mo_occ, self.mo_coeff
        nc, nv, no = self.nc, self.nv, self.no
        nvir = no + nv
        nocc = nc + no
        si = no / 2.0
        ndim = (nocc, nvir)
        orboa = mo_coeff[0][:, self.occidx_a]
        orbvb = mo_coeff[1][:, self.viridx_b]
        orbov = (orboa, orbvb)

        fockA, fockB = _get_mo_fock(mf, mo_coeff, mo_occ)
        e_ia = (mo_energy[1][self.viridx_b, None] - mo_energy[0][self.occidx_a]).T
        hdiag = e_ia.ravel()
        if self.re:
            oo = cp.zeros(no * no)
            for i in range(no):
                oo[i*no:(i+1)*no] = hdiag[nc*nvir+nvir*i:nc*nvir+no+nvir*i]
            new_oo = cp.einsum("x,xy->y", oo, self.vects)
            new_hdiag = cp.zeros(len(hdiag) - 1)
            new_hdiag[:nc*nvir] = hdiag[:nc*nvir]
            for i in range(no - 1):
                new_hdiag[nc*nvir+i*nvir:nc*nvir+no+i*nvir] = new_oo[i*no:(i+1)*no]
                new_hdiag[nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir] = hdiag[nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir]
            new_hdiag[nc*nvir+(no-1)*nvir:nc*nvir+(no-1)*nvir+no-1] = new_oo[(no-1)*no:]
            new_hdiag[nc*nvir+(no-1)*nvir+no-1:] = hdiag[nc*nvir+(no-1)*nvir+no:]
            hdiag = new_hdiag

        vresp = gen_response_sf(
            mf, hermi=0, method=self.method,
            collinear_samples=self.collinear_samples
        )

        if self.SA > 0:
            vresp_hf = self.gen_response_sf_delta_A(hermi=0)
            fockA_hf, fockB_hf = _get_hf_fock_mo_gamma(mf, mo_coeff, mo_occ)
            factor1 = cp.sqrt((2*si + 1)/(2*si)) - 1
            factor2 = cp.sqrt((2*si + 1)/(2*si - 1))
            factor3 = cp.sqrt((2*si)/(2*si - 1)) - 1
            factor4 = 1 / cp.sqrt(2*si*(2*si - 1))

        def vind(zs0):
            zs0 = _asarray(zs0)
            if self.re:
                oo = cp.zeros((zs0.shape[0], no*no - 1))
                for i in range(no - 1):
                    oo[:, i*no:(i+1)*no] = zs0[:, nc*nvir+i*nvir:nc*nvir+no+i*nvir]
                oo[:, (no-1)*no:] = zs0[:, nc*nvir+(no-1)*nvir:nc*nvir+(no-1)*nvir+no-1]
                new_oo = cp.einsum("xy,ny->nx", self.vects, oo)
                new_zs0 = cp.zeros((zs0.shape[0], zs0.shape[1] + 1))
                new_zs0[:, :nc*nvir] = zs0[:, :nc*nvir]
                for i in range(no - 1):
                    new_zs0[:, nc*nvir+i*nvir:nc*nvir+no+i*nvir] = new_oo[:, i*no:(i+1)*no]
                    new_zs0[:, nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir] = zs0[:, nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir]
                new_zs0[:, nc*nvir+(no-1)*nvir:nc*nvir+no+(no-1)*nvir] = new_oo[:, -no:]
                new_zs0[:, nc*nvir+(no-1)*nvir+no:] = zs0[:, nc*nvir+(no-1)*nvir+no-1:]
            else:
                new_zs0 = zs0.copy()

            zs = new_zs0.reshape(-1, *ndim)
            orbo, orbv = orbov
            # dmov = cp.einsum("xov,qv,po->xpq", zs, orbv.conj(), orbo)
            # 原来：dmov = cp.einsum("xov,qv,po->xpq", zs, orbv.conj(), orbo)
            tmp = cp.matmul(zs, orbv.conj().T)     # (x, o, q)
            dmov = cp.matmul(orbo, tmp)            # (x, p, q)
            v1ao = vresp(dmov)
            # AO response -> ov response
            # 原来：vs = cp.einsum("xpq,po,qv->xov", v1ao, orbo.conj(), orbv)
            tmp = cp.matmul(v1ao, orbv)            # (x, p, v)
            vs = cp.matmul(orbo.conj().T, tmp)     # (x, o, v)
            
            vs += cp.einsum("ab,xib->xia", fockB[self.nocc_b:, self.nocc_b:], zs, optimize=True)
            vs -= cp.einsum("ij,xja->xia", fockA[:self.nocc_a, :self.nocc_a], zs, optimize=True)
            vs_dA = cp.zeros_like(vs)

            if self.SA > 0:
                cv1 = zs[:, :nc, no:]
                co1 = zs[:, :nc, :no]
                ov1 = zs[:, nc:, no:]
                oo1 = zs[:, nc:, :no]

                cv1_mo = cp.einsum("xov,qv,po->xpq", cv1, orbvb[:, no:].conj(), orboa[:, :nc], optimize=True)
                co1_mo = cp.einsum("xov,qv,po->xpq", co1, orbvb[:, :no].conj(), orboa[:, :nc], optimize=True)
                ov1_mo = cp.einsum("xov,qv,po->xpq", ov1, orbvb[:, no:].conj(), orboa[:, nc:nc+no], optimize=True)
                oo1_mo = cp.einsum("xov,qv,po->xpq", oo1, orbvb[:, :no].conj(), orboa[:, nc:nc+no], optimize=True)

                _, v1_cv_k = vresp_hf(cv1_mo)
                v1_co_j, v1_co_k = vresp_hf(co1_mo)
                v1_ov_j, v1_ov_k = vresp_hf(ov1_mo)
                _, v1_oo_k = vresp_hf(oo1_mo)

                v1_co_j = cp.einsum("xpq,po,qv->xov", v1_co_j, orbo.conj(), orbv, optimize=True)
                v1_ov_j = cp.einsum("xpq,po,qv->xov", v1_ov_j, orbo.conj(), orbv, optimize=True)
                v1_cv_k = cp.einsum("xpq,po,qv->xov", v1_cv_k, orbo.conj(), orbv, optimize=True)
                v1_co_k = cp.einsum("xpq,po,qv->xov", v1_co_k, orbo.conj(), orbv, optimize=True)
                v1_ov_k = cp.einsum("xpq,po,qv->xov", v1_ov_k, orbo.conj(), orbv, optimize=True)
                v1_oo_k = cp.einsum("xpq,po,qv->xov", v1_oo_k, orbo.conj(), orbv, optimize=True)

                vs_dA[:, :nc, no:] += (
                    cp.einsum("ab,xib->xia", fockB_hf[nc+no:, nc+no:], zs[:, :nc, no:], optimize=True)
                    - cp.einsum("ab,xib->xia", fockA_hf[nc+no:, nc+no:], zs[:, :nc, no:], optimize=True)
                    + cp.einsum("ji,xja->xia", fockB_hf[:nc, :nc], zs[:, :nc, no:], optimize=True)
                    - cp.einsum("ji,xja->xia", fockA_hf[:nc, :nc], zs[:, :nc, no:], optimize=True)
                ) / (2*si)
                vs_dA[:, :nc, :no] += -v1_co_j[:, :nc, :no] / (2*si - 1) + (
                    cp.einsum("ji,xju->xiu", fockB_hf[:nc, :nc], zs[:, :nc, :no], optimize=True)
                    - cp.einsum("ji,xju->xiu", fockA_hf[:nc, :nc], zs[:, :nc, :no], optimize=True)
                ) / (2*si - 1)
                vs_dA[:, nc:, no:] += -v1_ov_j[:, nc:, no:] / (2*si - 1) + (
                    cp.einsum("ab,xub->xua", fockB_hf[nc+no:, nc+no:], zs[:, nc:, no:], optimize=True)
                    - cp.einsum("ab,xub->xua", fockA_hf[nc+no:, nc+no:], zs[:, nc:, no:], optimize=True)
                ) / (2*si - 1)

            if self.SA > 1:
                vs_dA[:, :nc, no:] += factor1 * (-v1_co_k[:, :nc, no:] + cp.einsum("av,xiv->xia", fockB_hf[nc+no:, nc:nc+no], zs[:, :nc, :no], optimize=True))
                vs_dA[:, :nc, :no] += factor1 * (-v1_cv_k[:, :nc, :no] + cp.einsum("av,xja->xjv", fockB_hf[nc+no:, nc:nc+no], zs[:, :nc, no:], optimize=True))
                vs_dA[:, :nc, no:] += factor1 * (-v1_ov_k[:, :nc, no:] - cp.einsum("vi,xva->xia", fockA_hf[nc:nc+no, :nc], zs[:, nc:, no:], optimize=True))
                vs_dA[:, nc:, no:] += factor1 * (-v1_cv_k[:, nc:, no:] - cp.einsum("vi,xib->xvb", fockA_hf[nc:nc+no, :nc], zs[:, :nc, no:], optimize=True))
                vs_dA[:, :nc, :no] += (v1_ov_j[:, :nc, :no] - v1_ov_k[:, :nc, :no]) / (2*si - 1)
                vs_dA[:, nc:, no:] += (v1_co_j[:, nc:, no:] - v1_co_k[:, nc:, no:]) / (2*si - 1)

            if self.SA > 2:
                iden_O = cp.eye(no)
                vs_dA[:, :nc, no:] += foo * (-(factor2 - 1) * v1_oo_k[:, :nc, no:] + (factor2/(2*si)) * (
                    cp.einsum("ia,xvv->xia", fockB_hf[:nc, nc+no:], zs[:, nc:, :no], optimize=True)
                    - cp.einsum("ia,xvv->xia", fockA_hf[:nc, nc+no:], zs[:, nc:, :no], optimize=True)
                ))
                vs_dA[:, nc:, :no] += foo * (-(factor2 - 1) * v1_cv_k[:, nc:, :no] + (factor2/(2*si)) * (
                    cp.einsum("vw,ia,xia->xvw", iden_O, fockB_hf[:nc, nc+no:], zs[:, :nc, no:], optimize=True)
                    - cp.einsum("vw,ia,xia->xvw", iden_O, fockA_hf[:nc, nc+no:], zs[:, :nc, no:], optimize=True)
                ))
                vs_dA[:, :nc, :no] += foo * (
                    factor3 * (-v1_oo_k[:, :nc, :no] - cp.einsum("iw,xwu->xiu", fockA_hf[:nc, nc:nc+no], zs[:, nc:, :no], optimize=True))
                    + factor4 * cp.einsum("vw,iu,xvw->xiu", iden_O, fockB_hf[:nc, nc:nc+no], zs[:, nc:, :no], optimize=True)
                )
                vs_dA[:, nc:, :no] += foo * (
                    factor3 * (-v1_co_k[:, nc:, :no] - cp.einsum("iw,xiv->xwv", fockA_hf[:nc, nc:nc+no], zs[:, :nc, :no], optimize=True))
                    + factor4 * cp.einsum("vw,iu,xiu->xvw", iden_O, fockB_hf[:nc, nc:nc+no], zs[:, :nc, :no], optimize=True)
                )
                vs_dA[:, nc:, no:] += foo * (
                    factor3 * (-v1_oo_k[:, nc:, no:] + cp.einsum("av,xuv->xua", fockB_hf[nc+no:, nc:nc+no], zs[:, nc:, :no], optimize=True))
                    - factor4 * cp.einsum("vw,au,xvw->xua", iden_O, fockA_hf[nc+no:, nc:nc+no], zs[:, nc:, :no], optimize=True)
                )
                vs_dA[:, nc:, :no] += foo * (
                    factor3 * (-v1_ov_k[:, nc:, :no] + cp.einsum("av,xwa->xwv", fockB_hf[nc+no:, nc:nc+no], zs[:, nc:, no:], optimize=True))
                    - factor4 * cp.einsum("vw,au,xua->xwv", iden_O, fockA_hf[nc+no:, nc:nc+no], zs[:, nc:, no:], optimize=True)
                )

            hx = (vs + fglobal * vs_dA).reshape(zs.shape[0], -1)
            if self.re:
                new_hx = cp.zeros_like(zs0)
                new_hx[:, :nc*nvir] += hx[:, :nc*nvir]
                oo = cp.zeros((zs0.shape[0], no*no))
                for i in range(no):
                    oo[:, i*no:(i+1)*no] = hx[:, nc*nvir+i*nvir:nc*nvir+no+i*nvir]
                new_oo = cp.einsum("xy,nx->ny", self.vects, oo, optimize=True)
                for i in range(no - 1):
                    new_hx[:, nc*nvir+i*nvir:nc*nvir+i*nvir+no] = new_oo[:, i*no:(i+1)*no]
                    new_hx[:, nc*nvir+i*nvir+no:nc*nvir+i*nvir+no+nv] = hx[:, nc*nvir+no+i*nvir:nc*nvir+no+nv+i*nvir]
                new_hx[:, nc*nvir+(no-1)*nvir:nc*nvir+(no-1)*nvir+no-1] = new_oo[:, (no-1)*no:]
                new_hx[:, nc*nvir+(no-1)*nvir+no-1:] = hx[:, nc*nvir+(no-1)*nvir+no:]
                return new_hx
            return hx

        return vind, hdiag

    def deal_v_davidson(self):
        cv = cp.zeros((self.nstates, self.nc, self.nv))
        co = cp.zeros((self.nstates, self.nc, self.no))
        ov = cp.zeros((self.nstates, self.no, self.nv))
        oo = cp.zeros((self.nstates, self.no*self.no - 1)) if self.re else cp.zeros((self.nstates, self.no, self.no))
        nvir = self.no + self.nv
        passed = self.nc * nvir
        nstates = self.nstates - 1 if self.nstates == (self.nc+self.no)*(self.no+self.nv) else self.nstates
        for state in range(nstates):
            tmp = self.v[:, state]
            for i in range(self.nc):
                cv[state, i, :] = tmp[i*nvir+self.no:i*nvir+self.no+self.nv]
                co[state, i, :] = tmp[i*nvir:i*nvir+self.no]
            if self.re:
                for i in range(self.no - 1):
                    oo[state, i*self.no:(i+1)*self.no] = tmp[passed+i*nvir:passed+i*nvir+self.no]
                    ov[state, i, :] = tmp[passed+i*nvir+self.no:passed+i*nvir+self.no+self.nv]
                oo[state, (self.no-1)*self.no:] = tmp[passed+(self.no-1)*nvir:passed+(self.no-1)*nvir+self.no-1]
                ov[state, self.no-1, :] = tmp[passed+(self.no-1)*nvir+self.no-1:]
            else:
                for i in range(self.no):
                    oo[state, i, :] = tmp[passed+i*nvir:passed+i*nvir+self.no]
                    ov[state, i, :] = tmp[passed+i*nvir+self.no:passed+i*nvir+self.no+self.nv]
        return cp.hstack([cv.reshape(self.nstates, -1), co.reshape(self.nstates, -1),
                          ov.reshape(self.nstates, -1), oo.reshape(self.nstates, -1)]).T

    def davidson_process(self, foo, fglobal):
        vind, hdiag = self.gen_tda_operation_sf(foo, fglobal)
        x0 = self.init_guess(self.mf, self.nstates)
        converged, e, x1 = lib.davidson1(
            lambda xs: _asnumpy(vind(cp.asarray(xs))),
            _asnumpy(x0),
            _asnumpy(hdiag),
            tol=1e-8, lindep=1e-9,
            nroots=self.nstates, max_cycle=1000,
        )
        self.converged = converged
        self.e = cp.asarray(e)
        self.v = cp.asarray(np.asarray(x1).T)
        self.v = self.deal_v_davidson()
        print("Converged ", converged)

    def frozen_A(self, frozen):
        if isinstance(frozen, int):
            f = 1 if frozen == 0 else frozen
        else:
            f = 1
        minus_cv = self.A[f*self.nv:, f*self.nv:]
        dim = minus_cv.shape[0]
        kept = cp.r_[0:(self.nc-f)*self.nv, (self.nc-f)*self.nv+f*self.no:dim]
        return minus_cv[kept][:, kept]

    def kernel(self, nstates=1, remove=None, frozen=None, foo=1.0,
               d_lda=0.3, fglobal=None, fit=True):
        if remove is None:
            self.re = not self.type_u
        else:
            self.re = remove
        nov = (self.nc + self.no) * (self.no + self.nv)
        self.nstates = min(nstates, nov)
        if fglobal is None:
            fglobal = self._default_fglobal(d_lda=d_lda, fit=fit)
        self.fglobal = fglobal
        if self.re:
            print("fglobal", fglobal)
            self.vects = self.get_vect()
        if self.davidson:
            self.davidson_process(foo=foo, fglobal=fglobal)
        else:
            self.A = self.get_Amat(foo=foo, fglobal=fglobal, projected=self.re)
            if frozen is not None:
                self.A = self.frozen_A(frozen)
            if self.A.shape[0] < 1000:
                e, v = scipy.linalg.eigh(_asnumpy(self.A))
            else:
                dim_n = self.A.shape[0]
                nroots = min(dim_n, nstates + 5)
                e, v = scipy.sparse.linalg.eigsh(_asnumpy(self.A), k=nroots, which="SA")
            self.e = cp.asarray(e[:nstates])
            self.v = cp.asarray(v[:, :nstates])
        return _asnumpy(self.e * au2ev), self.v


XSF_TDA = XSF_TDA_pbc_GPU


if __name__ == "__main__":
    print("Import XSF_TDA_pbc_GPU and pass a Gamma-point gpu4pyscf PBC mean-field object.")
