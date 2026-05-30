import numpy as np

from .base import (
    XTDDFT_base,
    _as_cpu_ctx,
    _as_cpu_mf,
    _build_initial_guess_from_gaps,
    _get_hf_mo_fock,
    _get_mo_fock,
    _get_ovlp,
    _molecular_dipole_integrals,
    _molecular_ground_dipole,
    _run_davidson,
    _system,
)
from .hxc_part import gen_response_tda
from ..utils.backend import _asarray, _asnumpy, backend, contract, require_cupy, set_backend, xp
from ..utils.unit import ha2eV

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)


def _order_pyscf2my(nc, no, nv):
    """Convert alpha|beta TDA order to CVa|OVa|COb|CVb order."""
    order = np.indices(((nc + no) * nv + nc * (no + nv),)).squeeze()
    for oi in range(nc):
        for noi in range(no):
            order = np.insert(
                order,
                (nc + no) * nv + no * oi + noi,
                (nc + no) * nv + oi * (no + nv) + noi,
            )
            order = np.delete(order, (nc + no) * nv + oi * (no + nv) + noi + 1)
    return order


def _so2st(eigvec, nc, no, nv):
    cva = eigvec[:nc * nv]
    ova = eigvec[nc * nv:(nc + no) * nv]
    cob = eigvec[(nc + no) * nv:(nc + no) * nv + nc * no]
    cvb = eigvec[(nc + no) * nv + nc * no:]
    cv0 = np.sqrt(0.5) * (cva + cvb)
    cv1 = np.sqrt(0.5) * (cva - cvb)
    return np.vstack([cv0, ova, cob, cv1])


class XTDA(XTDDFT_base):
    """Spin-conserving open-shell TDA.

    The working vector follows PySCF TDA order during Davidson:
    alpha occupied-to-virtual excitations followed by beta excitations.  Stored
    eigenvectors are reordered to CVa|OVa|COb|CVb, matching the reference XTDA
    analysis order.
    """

    def __init__(self, mf, method=0, davidson=True, davidson_backend="cpu",
                 so2st=False, dense_batch_size=64):
        if method != 0:
            raise NotImplementedError("XTDA currently implements method=0 spin-conserving response only.")
        davidson_backend = davidson_backend.lower()
        if davidson_backend not in ("cpu", "gpu", "auto"):
            raise ValueError("davidson_backend must be 'cpu', 'gpu', or 'auto'")
        super().__init__(mf, method, davidson=davidson)
        logger.info("XTDA spin-conserving TDA response")
        self.davidson_backend = "cpu" if davidson_backend == "auto" else davidson_backend
        self.so2st = so2st
        self.dense_batch_size = dense_batch_size
        self.order = _order_pyscf2my(self.nc, self.no, self.nv)
        self._raw_v = None
        self._tdm_v = None
        self.type_u = _asnumpy(self.mf.mo_coeff).ndim == 3
        self.dS2 = None

    def _result_method_label(self):
        return "spin_conserving"

    @property
    def _is_restricted_open_shell(self):
        return _asnumpy(self.mf.mo_coeff).ndim != 3 and self.no > 0

    def _split_vectors(self, zs, ctx):
        nocca, nvira = ctx.nocc_a, ctx.nvir_a
        noccb, nvirb = ctx.nocc_b, ctx.nvir_b
        za = zs[:, :nocca * nvira].reshape(zs.shape[0], nocca, nvira)
        zb = zs[:, nocca * nvira:].reshape(zs.shape[0], noccb, nvirb)
        return za, zb

    def _tag_transition_dm(self, dms, mo1a, mo1b, orboa, orbob):
        if not backend.is_gpu:
            return dms
        try:
            from gpu4pyscf.lib.cupy_helper import tag_array
        except Exception:
            return dms
        return tag_array(dms, mo1=[mo1a, mo1b], occ_coeff=[orboa, orbob])

    def _hdiag_from_fock(self, ctx, focka_mo, fockb_mo,
                         focka_hf=None, fockb_hf=None, spin=None):
        diag_a = focka_mo.diagonal()
        diag_b = fockb_mo.diagonal()
        e_ia_a = diag_a[ctx.viridx_a] - diag_a[ctx.occidx_a, None]
        e_ia_b = diag_b[ctx.viridx_b] - diag_b[ctx.occidx_b, None]

        if (
            focka_hf is not None
            and fockb_hf is not None
            and ctx.no > 0
            and ctx.nc > 0
            and ctx.nv > 0
            and spin is not None
            and abs(spin) > 1e-14
        ):
            si = 0.5 * spin
            factor_a = 0.5 * (1 - xp.sqrt((si + 1) / si) + 1 / (2 * si))
            factor_b = 0.5 * (-1 + xp.sqrt((si + 1) / si) + 1 / (2 * si))
            delta_f = fockb_hf - focka_hf
            diag_delta = delta_f.diagonal()
            core_delta = diag_delta[:ctx.nc]
            virt_delta = diag_delta[ctx.nocc_a:]

            # Li-Liu finite-spin correction only changes the CVa/CVb
            # self-block diagonals; CVa-CVb terms are off-diagonal couplings.
            e_ia_a[:ctx.nc, :] += (
                factor_a * virt_delta + factor_b * core_delta[:, None]
            )
            e_ia_b[:, -ctx.nv:] += (
                factor_b * virt_delta + factor_a * core_delta[:, None]
            )
        return xp.hstack([e_ia_a.reshape(-1), e_ia_b.reshape(-1)])

    def gen_tda_operation(self, mf=None, ctx=None):
        mf = self.mf if mf is None else mf
        ctx = self.ctx if ctx is None else ctx

        mo_coeff = _asarray(ctx.mo_coeff)
        mo_occ = _asarray(ctx.mo_occ)
        orboa = mo_coeff[0][:, ctx.occidx_a]
        orbva = mo_coeff[0][:, ctx.viridx_a]
        orbob = mo_coeff[1][:, ctx.occidx_b]
        orbvb = mo_coeff[1][:, ctx.viridx_b]

        focka_mo, fockb_mo = _get_mo_fock(mf, mo_coeff, mo_occ)
        vresp = gen_response_tda(mf, hermi=0, ctx=ctx)

        use_delta_a = (
            _asnumpy(mf.mo_coeff).ndim != 3
            and ctx.no > 0
            and _system(mf).spin != 0
        )
        if use_delta_a:
            si = 0.5 * _system(mf).spin
            if abs(si) < 1e-14:
                use_delta_a = False
            else:
                focka_hf, fockb_hf = _get_hf_mo_fock(mf, mo_coeff, mo_occ)
                factor_a = 0.5 * (1 - xp.sqrt((si + 1) / si) + 1 / (2 * si))
                factor_b = 0.5 * (-1 + xp.sqrt((si + 1) / si) + 1 / (2 * si))
                factor_ab = 0.5 / (2 * si)
        hdiag = self._hdiag_from_fock(
            ctx, focka_mo, fockb_mo,
            focka_hf=focka_hf if use_delta_a else None,
            fockb_hf=fockb_hf if use_delta_a else None,
            spin=_system(mf).spin if use_delta_a else None,
        )

        def vind(zs0):
            zs = xp.asarray(zs0)
            if zs.ndim == 1:
                zs = zs.reshape(1, -1)
            nz = zs.shape[0]
            za, zb = self._split_vectors(zs, ctx)

            mo1a = contract("xov,pv->xpo", za, orbva)
            dmsa = contract("xpo,qo->xpq", mo1a, orboa.conj())
            mo1b = contract("xov,pv->xpo", zb, orbvb)
            dmsb = contract("xpo,qo->xpq", mo1b, orbob.conj())
            dms = xp.asarray([dmsa, dmsb])
            dms = self._tag_transition_dm(dms, mo1a, mo1b, orboa, orbob)

            v1ao = vresp(dms)
            v1ao = xp.asarray(v1ao)
            v1a = contract("xpq,qo->xpo", v1ao[0], orboa)
            v1a = contract("xpo,pv->xov", v1a, orbva.conj())
            v1b = contract("xpq,qo->xpo", v1ao[1], orbob)
            v1b = contract("xpo,pv->xov", v1b, orbvb.conj())

            v1a += contract("xib,ab->xia", za, focka_mo[ctx.viridx_a[:, None], ctx.viridx_a])
            v1a -= contract("xja,ij->xia", za, focka_mo[ctx.occidx_a[:, None], ctx.occidx_a])
            v1b += contract("xib,ab->xia", zb, fockb_mo[ctx.viridx_b[:, None], ctx.viridx_b])
            v1b -= contract("xja,ij->xia", zb, fockb_mo[ctx.occidx_b[:, None], ctx.occidx_b])

            if use_delta_a:
                nc, no, nv = ctx.nc, ctx.no, ctx.nv
                cv_a = za[:, :nc, :]
                cv_b = zb[:, :, -nv:]
                v1a[:, :nc, :] += (
                    factor_a * (
                        contract("xib,ab->xia", cv_a, fockb_hf[ctx.nocc_a:, ctx.nocc_a:])
                        - contract("xib,ab->xia", cv_a, focka_hf[ctx.nocc_a:, ctx.nocc_a:])
                    )
                    + factor_b * (
                        contract("xja,ij->xia", cv_a, fockb_hf[:nc, :nc])
                        - contract("xja,ij->xia", cv_a, focka_hf[:nc, :nc])
                    )
                )
                cv_coupling = (
                    contract("xib,ab->xia", cv_b, fockb_hf[ctx.nocc_a:, ctx.nocc_a:])
                    - contract("xib,ab->xia", cv_b, focka_hf[ctx.nocc_a:, ctx.nocc_a:])
                    + contract("xja,ij->xia", cv_b, fockb_hf[:nc, :nc])
                    - contract("xja,ij->xia", cv_b, focka_hf[:nc, :nc])
                )
                v1a[:, :nc, :] -= factor_ab * cv_coupling

                v1b[:, :, -nv:] -= factor_ab * (
                    contract("xib,ab->xia", cv_a, fockb_hf[ctx.nocc_a:, ctx.nocc_a:])
                    - contract("xib,ab->xia", cv_a, focka_hf[ctx.nocc_a:, ctx.nocc_a:])
                    + contract("xja,ij->xia", cv_a, fockb_hf[:nc, :nc])
                    - contract("xja,ij->xia", cv_a, focka_hf[:nc, :nc])
                )
                v1b[:, :, -nv:] += (
                    factor_b * (
                        contract("xib,ab->xia", cv_b, fockb_hf[ctx.nocc_a:, ctx.nocc_a:])
                        - contract("xib,ab->xia", cv_b, focka_hf[ctx.nocc_a:, ctx.nocc_a:])
                    )
                    + factor_a * (
                        contract("xja,ij->xia", cv_b, fockb_hf[:nc, :nc])
                        - contract("xja,ij->xia", cv_b, focka_hf[:nc, :nc])
                    )
                )

            return xp.hstack([v1a.reshape(nz, -1), v1b.reshape(nz, -1)])

        return vind, hdiag

    def init_guess(self, nstates, hdiag=None):
        if hdiag is None:
            _, hdiag = self.gen_tda_operation()
        return _build_initial_guess_from_gaps(hdiag, nstates)

    def davidson_process(self, nstates):
        vind, hdiag = self.gen_tda_operation()
        nroots = min(nstates, int(hdiag.size))
        x0 = self.init_guess(nroots, hdiag=hdiag)
        converged, e, x1 = _run_davidson(
            self.mf, self.davidson_backend,
            vind, hdiag, x0, nroots,
        )
        self.converged = converged
        self.e = xp.asarray(e)
        raw_v = xp.asarray(_asnumpy(x1)).T
        self._raw_v = raw_v
        self.v = raw_v[self.order]
        self._tdm_v = self.v
        if self.so2st:
            self.v = xp.asarray(_so2st(_asnumpy(self.v), self.nc, self.no, self.nv))
        logger.info("XTDA Davidson converged: {}", converged)
        return self.e, self.v

    def get_Amat(self, batch_size=None):
        batch_size = self.dense_batch_size if batch_size is None else batch_size
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        mode = backend.mode
        set_backend("cpu")
        try:
            vind, hdiag = self.gen_tda_operation(mf=mf, ctx=ctx)
            ndim = int(hdiag.size)
            cols = []
            eye = np.eye(ndim)
            for p0 in range(0, ndim, batch_size):
                cols.append(_asnumpy(vind(eye[p0:p0 + batch_size])))
            amat = np.vstack(cols).T
            order = _order_pyscf2my(ctx.nc, ctx.no, ctx.nv)
            amat = amat[np.ix_(order, order)]
            self.A = np.asarray((amat + amat.T) * 0.5)
        finally:
            set_backend(mode)
        logger.info("XTDA dense A dimension: {}", self.A.shape[0])
        return self.A

    def _diagonalize_dense(self, amat, nstates):
        amat = np.asarray(_asnumpy(amat))
        if self.davidson_backend == "gpu":
            cp = require_cupy()
            e, v = cp.linalg.eigh(cp.asarray(amat))
            self.e = e[:nstates]
            self.v = v[:, :nstates]
        else:
            e, v = np.linalg.eigh(amat)
            self.e = e[:nstates]
            self.v = v[:, :nstates]
        self._tdm_v = self.v
        if self.so2st:
            self.v = xp.asarray(_so2st(_asnumpy(self.v), self.nc, self.no, self.nv))
        return self.e, self.v

    def _split_analysis_vectors(self, data=None):
        data = _asnumpy(self.v if data is None else data)
        dim1 = self.nc * self.nv
        dim2 = dim1 + self.no * self.nv
        dim3 = dim2 + self.nc * self.no
        return (
            data[:dim1].T.reshape(data.shape[1], self.nc, self.nv),
            data[dim1:dim2].T.reshape(data.shape[1], self.no, self.nv),
            data[dim2:dim3].T.reshape(data.shape[1], self.nc, self.no),
            data[dim3:].T.reshape(data.shape[1], self.nc, self.nv),
        )

    def deltaS2(self):
        cv_a, ov_a, co_b, cv_b = self._split_analysis_vectors()
        if not self.type_u:
            return (
                np.einsum("nij,nij->n", cv_a, cv_a)
                + np.einsum("nij,nij->n", cv_b, cv_b)
                - 2.0 * np.einsum("nij,nij->n", cv_a, cv_b)
            )

        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        mooa = mo_coeff[0][:, _asnumpy(ctx.occidx_a)]
        moob = mo_coeff[1][:, _asnumpy(ctx.occidx_b)]
        mova = mo_coeff[0][:, _asnumpy(ctx.viridx_a)]
        movb = mo_coeff[1][:, _asnumpy(ctx.viridx_b)]
        s = _asnumpy(_get_ovlp(mf))

        sccba = moob.conj().T @ s @ mooa
        sccab = mooa.conj().T @ s @ moob
        svcab = mova.conj().T @ s @ moob
        svcba = movb.conj().T @ s @ mooa
        svvab = mova.conj().T @ s @ movb

        nc, no = self.nc, self.no
        ds2 = (
            np.einsum("nia,nja,ki,jk->n", cv_a, cv_a, sccba[:, :nc], sccba.T[:nc, :])
            + np.einsum("nia,nja,ki,jk->n", ov_a, ov_a, sccba[:, nc:], sccba.T[nc:, :])
            + np.einsum("nia,nja,ki,jk->n", ov_a, cv_a, sccba[:, nc:], sccba.T[:nc, :])
            + np.einsum("nia,nja,ki,jk->n", cv_a, ov_a, sccba[:, :nc], sccba.T[nc:, :])
            - np.einsum("nia,nib,ak,kb->n", cv_a, cv_a, svcab, svcab.T)
            - np.einsum("nia,nib,ak,kb->n", ov_a, ov_a, svcab, svcab.T)
            + np.einsum("nia,nja,ki,jk->n", cv_b, cv_b, sccab, sccab.T)
            + np.einsum("nia,nja,ki,jk->n", co_b, co_b, sccab, sccab.T)
            - np.einsum("nia,nib,ak,kb->n", co_b, co_b, svcba[:no, :], svcba.T[:, :no])
            - np.einsum("nia,nib,ak,kb->n", cv_b, cv_b, svcba[no:, :], svcba.T[:, no:])
            - np.einsum("nia,nib,ak,kb->n", co_b, cv_b, svcba[:no, :], svcba.T[:, no:])
            - np.einsum("nia,nib,ak,kb->n", cv_b, co_b, svcba[no:, :], svcba.T[:, :no])
            - 2.0 * np.einsum("nia,njb,ji,ab->n", cv_a, cv_b, sccba[:, :nc], svvab[:, no:])
            - 2.0 * np.einsum("nia,njb,ji,ab->n", cv_a, co_b, sccba[:, :nc], svvab[:, :no])
            - 2.0 * np.einsum("nia,njb,ji,ab->n", ov_a, cv_b, sccba[:, nc:], svvab[:, no:])
            - 2.0 * np.einsum("nia,njb,ji,ab->n", ov_a, co_b, sccba[:, nc:], svvab[:, :no])
        )
        return np.real_if_close(ds2)

    def _spin_orbital_vectors(self):
        if self._tdm_v is not None:
            return _asnumpy(self._tdm_v)
        if self.so2st:
            raise NotImplementedError(
                "Transition dipoles require spin-orbital amplitudes; rerun XTDA with so2st=False."
            )
        return _asnumpy(self.v)

    def _dipole_mo_blocks(self):
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        dip_ao = _molecular_dipole_integrals(mf)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        ints_aa = np.asarray(contract("xpq,pi,qj->xij", dip_ao, mo_coeff[0].conj(), mo_coeff[0]))
        ints_bb = np.asarray(contract("xpq,pi,qj->xij", dip_ao, mo_coeff[1].conj(), mo_coeff[1]))
        return ints_aa, ints_bb, ctx, mf

    def transition_dipoles_ground(self):
        """Ground-state to excited-state transition dipoles in a.u."""
        ints_aa, ints_bb, ctx, _mf = self._dipole_mo_blocks()
        occ_a = _asnumpy(ctx.occidx_a)
        vir_a = _asnumpy(ctx.viridx_a)
        occ_b = _asnumpy(ctx.occidx_b)
        vir_b = _asnumpy(ctx.viridx_b)

        r_ov_a = ints_aa[:, occ_a][:, :, vir_a]
        r_ov_b = ints_bb[:, occ_b][:, :, vir_b]
        r_cv_a = r_ov_a[:, :self.nc, :].reshape(3, -1)
        r_ov_a = r_ov_a[:, self.nc:, :].reshape(3, -1)
        r_co_b = r_ov_b[:, :, :self.no].reshape(3, -1)
        r_cv_b = r_ov_b[:, :, self.no:].reshape(3, -1)
        dip_ordered = np.hstack([r_cv_a, r_ov_a, r_co_b, r_cv_b])

        vectors = self._spin_orbital_vectors()
        nstates = min(self.nstates, vectors.shape[1])
        return np.einsum("xd,ds->sx", dip_ordered, vectors[:, :nstates], optimize=True)

    def osc_str(self):
        trans_dip = self.transition_dipoles_ground()
        energies = _asnumpy(self.e)[:trans_dip.shape[0]]
        return (2.0 / 3.0) * energies * np.einsum("sx,sx->s", trans_dip, trans_dip)

    def _full_spin_conserving_amplitudes(self, vectors):
        cv_a, ov_a, co_b, cv_b = self._split_analysis_vectors(vectors)
        amps_a = []
        amps_b = []
        for istate in range(cv_a.shape[0]):
            xa = np.zeros((self.nc + self.no, self.nv))
            xb = np.zeros((self.nc, self.no + self.nv))
            xa[:self.nc, :] = cv_a[istate]
            xa[self.nc:, :] = ov_a[istate]
            xb[:, :self.no] = co_b[istate]
            xb[:, self.no:] = cv_b[istate]
            amps_a.append(xa)
            amps_b.append(xb)
        return amps_a, amps_b

    def transition_dipole_matrix(self, include_ground_dipole=False):
        """Excited-state to excited-state transition dipole matrix in a.u."""
        ints_aa, ints_bb, ctx, mf = self._dipole_mo_blocks()
        occ_a = _asnumpy(ctx.occidx_a)
        vir_a = _asnumpy(ctx.viridx_a)
        occ_b = _asnumpy(ctx.occidx_b)
        vir_b = _asnumpy(ctx.viridx_b)
        r_oo_a = ints_aa[:, occ_a][:, :, occ_a]
        r_vv_a = ints_aa[:, vir_a][:, :, vir_a]
        r_oo_b = ints_bb[:, occ_b][:, :, occ_b]
        r_vv_b = ints_bb[:, vir_b][:, :, vir_b]

        vectors = self._spin_orbital_vectors()
        nstates = min(self.nstates, vectors.shape[1])
        amps_a, amps_b = self._full_spin_conserving_amplitudes(vectors[:, :nstates])
        tdm = np.zeros((nstates, nstates, 3))
        for i, (a0, b0) in enumerate(zip(amps_a, amps_b)):
            for j, (a1, b1) in enumerate(zip(amps_a, amps_b)):
                tdm[i, j] = (
                    np.einsum("ia,xab,ib->x", a0, r_vv_a, a1, optimize=True)
                    - np.einsum("ia,xij,ja->x", a0, r_oo_a, a1, optimize=True)
                    + np.einsum("ia,xab,ib->x", b0, r_vv_b, b1, optimize=True)
                    - np.einsum("ia,xij,ja->x", b0, r_oo_b, b1, optimize=True)
                )
        if include_ground_dipole:
            gs = _molecular_ground_dipole(mf)
            for i in range(nstates):
                tdm[i, i] += gs
        return tdm

    def calculate_TDM(self, include_ground_dipole=True):
        gs_tdm = self.transition_dipoles_ground()
        gs_osc = self.osc_str()
        print("Ground state to Excited state transition dipole moments(a.u.)")
        print("State      X        Y        Z      OSC.")
        for i, dip in enumerate(gs_tdm):
            print(
                f" {i + 1:2d}     |GS>    "
                f"{dip[0]:>8.4f} {dip[1]:>8.4f} {dip[2]:>8.4f}  {gs_osc[i]:>8.4f} "
            )

        tdm = self.transition_dipole_matrix(include_ground_dipole=include_ground_dipole)
        energies = _asnumpy(self.e)[:tdm.shape[0]]
        osc = (2.0 / 3.0) * (
            (energies[:, None] - energies[None, :]) * np.einsum("ijx,ijx->ij", tdm, tdm)
        )
        print("\nExcited state to Excited state transition dipole moments(a.u.)")
        print("StateL StateR      X        Y        Z      f(L<-R)")
        for i in range(tdm.shape[0]):
            for j in range(tdm.shape[1]):
                print(
                    f" {i + 1:2d}     {j + 1:2d}    "
                    f"{tdm[i, j, 0]:>8.4f} {tdm[i, j, 1]:>8.4f} {tdm[i, j, 2]:>8.4f}  "
                    f"{osc[i, j]:>8.4f} "
                )
        return {"gs_tdm": gs_tdm, "gs_osc": gs_osc, "tdm": tdm, "osc": osc}

    analyze_TDM = calculate_TDM

    def kernel(self, nstates=1, save=False, save_file=None):
        self.nstates = nstates
        if self.davidson:
            self.davidson_process(nstates)
        else:
            self.get_Amat()
            self._diagonalize_dense(self.A, nstates)
        if save:
            self.save_results(save_file)
        return _asnumpy(self.e[:nstates] * ha2eV), self.v[:, :nstates]

    def analyse(self, threshold=0.1):
        energies = _asnumpy(self.e) * ha2eV
        cv_a, ov_a, co_b, cv_b = self._split_analysis_vectors()
        self.dS2 = _asnumpy(self.deltaS2())
        for istate in range(min(self.nstates, cv_a.shape[0])):
            print(
                f"D{istate + 1}    w:{energies[istate]:10.4f} eV"
                f"    d<S^2>:{self.dS2[istate]:8.4f}"
            )
            for c, v in zip(*np.where(abs(cv_a[istate]) > threshold)):
                amp = cv_a[istate, c, v]
                print(f"    CV(aa) {c + 1:3d} -> {v + 1 + self.nc + self.no:3d}    c_i: {amp:8.5f}    Per: {100 * amp**2:5.2f}%")
            for o, v in zip(*np.where(abs(ov_a[istate]) > threshold)):
                amp = ov_a[istate, o, v]
                print(f"    OV(aa) {o + 1 + self.nc:3d} -> {v + 1 + self.nc + self.no:3d}    c_i: {amp:8.5f}    Per: {100 * amp**2:5.2f}%")
            for c, o in zip(*np.where(abs(co_b[istate]) > threshold)):
                amp = co_b[istate, c, o]
                print(f"    CO(bb) {c + 1:3d} -> {o + 1 + self.nc:3d}    c_i: {amp:8.5f}    Per: {100 * amp**2:5.2f}%")
            for c, v in zip(*np.where(abs(cv_b[istate]) > threshold)):
                amp = cv_b[istate, c, v]
                print(f"    CV(bb) {c + 1:3d} -> {v + 1 + self.nc + self.no:3d}    c_i: {amp:8.5f}    Per: {100 * amp**2:5.2f}%")
            print(" ")
        return self.dS2
