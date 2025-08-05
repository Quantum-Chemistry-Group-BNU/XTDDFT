#!/usr/bin/env python
import os
import sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.append('/')
import scipy
import numpy as np
import pandas as pd
from pyscf import dft, gto, scf, lib, ao2mo, tddft
from pyscf.lib import logger
from utils import unit, atom


class XTDA:
    def __init__(self, mol, mf, nstates=10, savedata=False, basis='orbital'):
        self.mol = mol
        self.mf = mf
        self.nstates = nstates
        self.savedata = savedata
        self.basis = basis

    def kernel(self):
        if self.basis == 'tensor':
            self.x_tda = X_TDA(self.mf)
            e, v = self.x_tda.kernel(nstates=self.nstates)
            # x_tda.analyze()
            # x_tda.analyze_TDM()
            return e, v
        elif self.basis == 'orbital':
            pass
        else:
            raise ValueError('basis must be tensor or orbital')
        # Section: prepare for compute
        mo_energy = self.mf.mo_energy
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        assert mo_coeff.dtype == np.float64
        nao = self.mol.nao_nr()
        occidx_a = np.where(mo_occ >= 1)[0]
        viridx_a = np.where(mo_occ == 0)[0]
        occidx_b = np.where(mo_occ >= 2)[0]
        viridx_b = np.where(mo_occ != 2)[0]
        nocc_a = len(occidx_a)
        nvir_a = len(viridx_a)
        nocc_b = len(occidx_b)
        nvir_b = len(viridx_b)
        orbo_a = mo_coeff[:, occidx_a]
        orbv_a = mo_coeff[:, viridx_a]
        orbo_b = mo_coeff[:, occidx_b]
        orbv_b = mo_coeff[:, viridx_b]
        mo_a = np.hstack((orbo_a, orbv_a))  # equal to mo_b, equal to mo_coeff
        mo_b = np.hstack((orbo_b, orbv_b))
        nmo_a = nocc_a + nvir_a
        nmo_b = nocc_b + nvir_b

        # check XC functional hybrid proportion
        ni = self.mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.mol.spin)
        print('hyb', hyb)

        # Section: compute Fock matrix
        h1e = self.mf.get_hcore()
        dm = self.mf.make_rdm1()
        vhf = self.mf.get_veff(self.mol, dm)
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fock_a = mo_coeff.T @ focka @ mo_coeff
        fock_b = mo_coeff.T @ fockb @ mo_coeff
        delta_ij_a = np.eye(nocc_a)
        delta_ab_a = np.eye(nvir_a)
        delta_ij_b = np.eye(nocc_b)
        delta_ab_b = np.eye(nvir_b)
        delta_ij = np.eye(nocc_b)
        delta_ab = np.eye(nvir_a)

        hf = scf.ROHF(self.mol)
        # hf.kernel()
        veff = hf.get_veff(self.mol, dm)
        focka2 = mo_coeff.T @ (h1e+veff[0]) @ mo_coeff
        fockb2 = mo_coeff.T @ (h1e+veff[1]) @ mo_coeff
        fab_a2 = focka2[nocc_a:, nocc_a:]
        fab_b2 = fockb2[nocc_b:, nocc_b:]
        fij_a2 = focka2[:nocc_a, :nocc_a]
        fij_b2 = fockb2[:nocc_b, :nocc_b]

        # Section: compute two electron integral and XC integral
        # use self.mf.mol general two-electron repulsion integral
        eri = ao2mo.general(self.mf.mol, [mo_a, mo_a, mo_a, mo_a], compact=False)
        eri = eri.reshape(nmo_a, nmo_a, nmo_a, nmo_a)

        # # use this code can compare with test/test_RO_UTDA.py and remember remove Fock matrix in A matrix
        # e_ia_a = (mo_energy[viridx_a, None] - mo_energy[occidx_a]).T
        # e_ia_b = (mo_energy[viridx_b, None] - mo_energy[occidx_b]).T
        # aa = np.diag(e_ia_a.ravel()).reshape(nocc_a, nvir_a, nocc_a, nvir_a)
        # bb = np.diag(e_ia_b.ravel()).reshape(nocc_b, nvir_b, nocc_b, nvir_b)
        aa = np.zeros((nocc_a, nvir_a, nocc_a, nvir_a))
        ab = np.zeros((nocc_a, nvir_a, nocc_b, nvir_b))
        bb = np.zeros((nocc_b, nvir_b, nocc_b, nvir_b))

        aa += np.einsum('iabj->iajb', eri[:nocc_a, nocc_a:, nocc_a:, :nocc_a])
        aa -= np.einsum('ijba->iajb', eri[:nocc_a, :nocc_a, nocc_a:, nocc_a:]) * hyb

        bb += np.einsum('iabj->iajb', eri[:nocc_b, nocc_b:, nocc_b:, :nocc_b])
        bb -= np.einsum('ijba->iajb', eri[:nocc_b, :nocc_b, nocc_b:, nocc_b:]) * hyb

        ab += np.einsum('iabj->iajb', eri[:nocc_a, nocc_a:, nocc_b:, :nocc_b])
        del dm, eri, focka, fockb, h1e
        if omega != 0:  # For RSH
            with self.mol.with_range_coulomb(omega):
                eri_aa = ao2mo.general(self.mol, [orbo_a, mo_a, mo_a, mo_a], compact=False)
                eri_bb = ao2mo.general(self.mol, [orbo_b, mo_b, mo_b, mo_b], compact=False)
                eri_aa = eri_aa.reshape(nocc_a, nmo_a, nmo_a, nmo_a)
                eri_bb = eri_bb.reshape(nocc_b, nmo_b, nmo_b, nmo_b)
                k_fac = alpha - hyb
                aa -= np.einsum('ijba->iajb', eri_aa[:nocc_a,:nocc_a,nocc_a:,nocc_a:]) * k_fac
                bb -= np.einsum('ijba->iajb', eri_bb[:nocc_b,:nocc_b,nocc_b:,nocc_b:]) * k_fac

        occ = np.zeros((2, len(mo_occ)))  # deal as uks
        occ[0][:len(occidx_a)] = 1
        occ[1][:len(occidx_b)] = 1

        ni = self.mf._numint
        ni.libxc.test_deriv_order(self.mf.xc, 2, raise_error=True)
        if self.mf.do_nlc():
            logger.warn(self.mf, 'NLC functional found in DFT object.  Its second '
                            'derivative is not available. Its contribution is '
                            'not included in the response function.')

        xctype = ni._xc_type(self.mf.xc)
        dm0 = self.mf.make_rdm1(mo_coeff, occ)
        dm0.mo_coeff = (mo_coeff, mo_coeff)
        make_rho = ni._gen_rho_evaluator(self.mol, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.mf.max_memory * .8 - mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(self.mol, self.mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(self.mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc[:, 0, :, 0] * weight

                rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                rho_ov_a = np.einsum('ri,ra->ria', rho_o_a, rho_v_a)
                rho_ov_b = np.einsum('ri,ra->ria', rho_o_b, rho_v_b)

                w_ov = np.einsum('ria,r->ria', rho_ov_a, wfxc[0, 0])
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov)
                aa += iajb

                w_ov = np.einsum('ria,r->ria', rho_ov_b, wfxc[0, 1])
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov)
                ab += iajb

                w_ov = np.einsum('ria,r->ria', rho_ov_b, wfxc[1, 1])
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b, w_ov)
                bb += iajb
            del ao, coords, fxc, iajb, rho0a, rho0b, rho_o_a, rho_o_b, rho_ov_a, rho_ov_b, \
                rho_v_a, rho_v_b, w_ov, wfxc
        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(self.mol, self.mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(self.mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_o_a = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_v_a = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_o_b = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_v_b = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_a = np.einsum('xri,ra->xria', rho_o_a, rho_v_a[0])
                rho_ov_b = np.einsum('xri,ra->xria', rho_o_b, rho_v_b[0])
                rho_ov_a[1:4] += np.einsum('ri,xra->xria', rho_o_a[0], rho_v_a[1:4])
                rho_ov_b[1:4] += np.einsum('ri,xra->xria', rho_o_b[0], rho_v_b[1:4])
                w_ov_aa = np.einsum('xyr,xria->yria', wfxc[0, :, 0], rho_ov_a)
                w_ov_ab = np.einsum('xyr,xria->yria', wfxc[0, :, 1], rho_ov_a)
                w_ov_bb = np.einsum('xyr,xria->yria', wfxc[1, :, 1], rho_ov_b)

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                aa += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                bb += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                ab += iajb
            del ao, coords, fxc, iajb, rho0a, rho0b, rho_o_a, rho_o_b, rho_ov_a, rho_ov_b, \
                rho_v_a, rho_v_b, w_ov_aa, w_ov_ab, w_ov_bb, wfxc
        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_oa = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_ob = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_va = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_vb = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_a = np.einsum('xri,ra->xria', rho_oa, rho_va[0])
                rho_ov_b = np.einsum('xri,ra->xria', rho_ob, rho_vb[0])
                rho_ov_a[1:4] += np.einsum('ri,xra->xria', rho_oa[0], rho_va[1:4])
                rho_ov_b[1:4] += np.einsum('ri,xra->xria', rho_ob[0], rho_vb[1:4])
                tau_ov_a = np.einsum('xri,xra->ria', rho_oa[1:4], rho_va[1:4]) * .5
                tau_ov_b = np.einsum('xri,xra->ria', rho_ob[1:4], rho_vb[1:4]) * .5
                rho_ov_a = np.vstack([rho_ov_a, tau_ov_a[np.newaxis]])
                rho_ov_b = np.vstack([rho_ov_b, tau_ov_b[np.newaxis]])
                w_ov_aa = np.einsum('xyr,xria->yria', wfxc[0,:,0], rho_ov_a)
                w_ov_ab = np.einsum('xyr,xria->yria', wfxc[0,:,1], rho_ov_a)
                w_ov_bb = np.einsum('xyr,xria->yria', wfxc[1,:,1], rho_ov_b)

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                aa += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                bb += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                ab += iajb
            del ao, coords, fxc, iajb, rho0a, rho0b, rho_oa, rho_ob, rho_ov_a, rho_ov_b, \
                rho_va, rho_vb, w_ov_aa, w_ov_ab, w_ov_bb, wfxc, tau_ov_a, tau_ov_b
        elif xctype == 'hf':
            pass

        # Section: construct A matrix
        nc = min(nocc_a, nocc_b)
        no = abs(nocc_a - nocc_b)
        nv = min(nvir_a, nvir_b)
        dim = (nc + no) * nv + nc * (nv + no)
        A = np.zeros((dim, dim))
        si = 0.5 * self.mol.spin

        # CV(aa)-CV(aa)
        A[:nc*nv, :nc*nv] = (
            np.einsum('ij, ab -> iajb',delta_ij_a[:nc, :nc], fock_a[nc+no:, nc+no:])
            - np.einsum('ij, ab -> iajb',fock_a[:nc, :nc], delta_ab_a)
            + aa[:nc, :, :nc, :]
        ).reshape(nc*nv, nc*nv)
        # # add spin adapted correction
        # # same with below code
        # focks = fock_b2 - fock_a2
        # A[:nc * nv, :nc * nv] += (
        #     0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * np.einsum('ij, ab->iajb', delta_ij_b, focks[nc+no:, nc+no:])
        #     + 0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * np.einsum('ij, ab->iajb', focks[:nc, :nc], delta_ab_a)
        # ).reshape(nc*nv, nc*nv)
        A[:nc*nv, :nc*nv] += (
            0.5 * (1-np.sqrt((si+1)/si)+1/(2*si)) * (
                np.einsum('ij, ab->iajb', delta_ij, fab_b2[no:, no:])
                - np.einsum('ij, ab->iajb', delta_ij, fab_a2)
            )
            + 0.5 * (-1+np.sqrt((si+1)/si)+1/(2*si)) * (
                np.einsum('ab, ij->iajb', delta_ab, fij_b2)
                - np.einsum('ab, ij->iajb', delta_ab, fij_a2[:-no,:-no])
            )
        ).reshape(nc*nv, nc*nv)

        # CV(aa)-OV(aa)
        Acvaaovaa = (
            np.einsum('ij, ab -> iajb', delta_ij_a[:nc, nc:nc+no], fock_a[nc+no:, nc+no:])
            - np.einsum('ij, ab -> iajb', fock_a[:nc, nc:nc+no], delta_ab_a)
            + aa[:nc, :, nc:nc+no, :]
        ).reshape(nc*nv, no*nv)
        A[:nc*nv, nc*nv:(nc+no)*nv] = Acvaaovaa

        # CV(aa)-CO(bb)
        Acvaacobb = ab[:nc, :, :, :no].reshape(nc*nv, nc*no)
        A[:nc*nv, (nc+no)*nv:(nc+no)*nv+nc*no] = Acvaacobb

        # CV(aa)-CV(bb)
        Acvaacvbb = ab[:nc, :, :, no:no+nv].reshape(nc*nv, nc*nv)
        # add spin adapted correction
        Acvaacvbb -= (
            0.5 * 1/(2*si) * (
                np.einsum('ij, ab->iajb', delta_ij, fab_b2[no:, no:])
                - np.einsum('ij, ab->iajb', delta_ij, fab_a2)
                + np.einsum('ab, ij->iajb', delta_ab, fij_b2)
                - np.einsum('ab, ij->iajb', delta_ab, fij_a2[:-no, :-no])
            )
        ).reshape(nc*nv, nc*nv)
        A[:nc*nv, (nc+no)*nv+nc*no:dim] = Acvaacvbb

        # OV(aa)-CV(aa)
        A[nc*nv:(nc+no)*nv, :nc*nv] = Acvaaovaa.T

        # OV(aa)-OV(aa)
        A[nc*nv:(nc+no)*nv, nc*nv:(nc+no)*nv] = (
            np.einsum('ij, ab -> iajb', delta_ij_a[nc:nc+no, nc:nc+no], fock_a[nc+no:, nc+no:])
            - np.einsum('ij, ab -> iajb', fock_a[nc:nc+no, nc:nc+no], delta_ab_a)
            + aa[nc:nc+no, :, nc:nc+no, :]
        ).reshape(no*nv, no*nv)

        # OV(aa)-CO(bb)
        Aovaacobb = ab[nc:nc+no, :, :, :no].reshape(no*nv, nc*no)
        A[nc*nv:(nc+no)*nv, (nc+no)*nv:(nc+no)*nv+nc*no] = Aovaacobb

        # OV(aa)-CV(bb)
        Aovaacvbb = ab[nc:nc+no, :, :, no:no+nv].reshape(no*nv, nc*nv)
        A[nc*nv:(nc+no)*nv, (nc+no)*nv+nc*no:dim] = Aovaacvbb

        # CO(bb)-CV(aa)
        A[(nc+no)*nv:(nc+no)*nv+nc*no, :nc*nv] = Acvaacobb.T

        # CO(bb)-OV(aa)
        A[(nc+no)*nv:(nc+no)*nv+nc*no, nc*nv:(nc+no)*nv] = Aovaacobb.T

        # CO(bb)-CO(bb)
        A[(nc+no)*nv:(nc+no)*nv+nc*no, (nc+no)*nv:(nc+no)*nv+nc*no] = (
            np.einsum('ij, ab -> iajb', delta_ij_b, fock_b[nc:nc+no, nc:nc+no])
            - np.einsum('ij, ab -> iajb', fock_b[:nc, :nc], delta_ab_b[:no, :no])
            + bb[:, :no, :, :no]
        ).reshape(nc*no, nc*no)

        # CO(bb)-CV(bb)
        Acobbcvbb = (
            np.einsum('ij, ab -> iajb', delta_ij_b, fock_b[nc:nc+no, nc+no:])
            - np.einsum('ij, ab -> iajb', fock_b[:nc, :nc], delta_ab_b[:no, no:])
            + bb[:, :no, :, no:]
        ).reshape(nc*no, nc*nv)
        A[(nc+no)*nv:(nc+no)*nv+nc*no, (nc+no)*nv+nc*no:dim] = Acobbcvbb

        # CV(bb)-CV(aa)
        A[(nc+no)*nv+nc*no:dim, :nc*nv] = Acvaacvbb.T

        # CV(bb)-OV(aa)
        A[(nc+no)*nv+nc*no:dim, nc*nv:(nc+no)*nv] = Aovaacvbb.T

        # CV(bb)-CO(bb)
        A[(nc+no)*nv+nc*no:dim, (nc+no)*nv:(nc+no)*nv+nc*no] = Acobbcvbb.T

        # CV(bb)-CV(bb)
        A[(nc+no)*nv+nc*no:dim, (nc+no)*nv+nc*no:dim] = (
            np.einsum('ij, ab -> iajb', delta_ij_b, fock_b[nc+no:, nc+no:])
            - np.einsum('ij, ab -> iajb', fock_b[:nc, :nc], delta_ab_b[no:, no:])
            + bb[:, no:, :, no:]
        ).reshape(nc*nv, nc*nv)
        # add spin adapted correction
        A[(nc+no)*nv+nc*no:dim, (nc+no)*nv+nc*no:dim] += (
            0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
            np.einsum('ij, ab->iajb', delta_ij, fab_b2[no:, no:])
            - np.einsum('ij, ab->iajb', delta_ij, fab_a2)
            )
            + 0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                np.einsum('ab, ij->iajb', delta_ab, fij_b2)
                - np.einsum('ab, ij->iajb', delta_ab, fij_a2[:-no, :-no])
            )
        ).reshape(nc*nv, nc*nv)

        self.mo_coeff = mo_coeff
        self.occidx_a = occidx_a
        self.viridx_a = viridx_a
        self.occidx_b = occidx_b
        self.viridx_b = viridx_b
        self.nc = nc
        self.nv = nv
        self.no = no
        # transform pyscf order to my order
        order = np.indices(((nc+no)*nv+nc*(no+nv),)).squeeze()
        for oi in range(nc * no):
            order = np.insert(order, (nc+no)*nv+oi, (nc+no)*nv+oi*nv+oi)
            order = np.delete(order, (nc+no)*nv+oi*nv+oi+1)
        self.order = order

        self.e, self.v = scipy.linalg.eigh(A)
        # print("my xTDA result is \n{}".format(self.e[:self.nstates] * unit.ha2eV))
        self.e_eV = self.e[:self.nstates] * unit.ha2eV
        self.v = self.v[:, :self.nstates]
        self.xy_a = self.v.T[:, :nocc_a * nvir_a]  # V_alpha
        self.xy_b = self.v.T[:, nocc_a * nvir_a:]  # V_beta
        self.xycv_a = self.v.T[:, :nc*nv]  # V_cv_a
        self.xyov_a = self.v.T[:, nc*nv:(nc+no)*nv]  # V_ov_a
        self.xyco_b = self.v.T[:, (nc+no)*nv:(nc+no)*nv+nc*no]  # V_co_b
        self.xycv_b = self.v.T[:, (nc+no)*nv+nc*no:]  # V_cv_b
        # dS2 = (np.sum(self.xycv_a*self.xycv_a) + np.sum(self.xycv_b*self.xycv_b)
        #        * np.sum(self.xyco_b*self.co_b) * np.sum(self.xycv_a*self.xycv_b))
        dS2 = self.deltaS2()
        os = self.osc_str()  # oscillator strength
        # before calculate rot_str, check whether mol is chiral mol
        if gto.mole.chiral_mol(self.mol):
            rs = self.rot_str()
        else:
            rs = np.zeros(self.nstates)
            # logger.info(self.mf, 'molecule do not have chiral')
        # logger.info(self.mf, 'oscillator strength (length form) \n{}'.format(os))
        # logger.info(self.mf, 'rotatory strength (cgs unit) \n{}'.format(rs))
        # logger.info(self.mf, 'deltaS2 is \n{}'.format(dS2))
        print('my XTDA result is')
        print(f'{"num":>4} {"energy":>8} {"wav_len":>8} {"osc_str":>8} {"rot_str":>8} {"deltaS2":>8}')
        for ni, ei, wli, osi, rsi, ds2i in zip(range(self.nstates), self.e_eV, unit.eVxnm/self.e_eV, os, rs, dS2):
            print(f'{ni:4d} {ei:8.4f} {wli:8.4f} {osi:8.4f} {rsi:8.4f} {ds2i:8.4f}')
        if self.savedata:
            pd.DataFrame(
                np.concatenate((unit.eVxnm / np.expand_dims(self.e_eV, axis=1), np.expand_dims(os, axis=1)), axis=1)
            ).to_csv('uvspec_data.csv', index=False, header=None)
        return self.e[:self.nstates], os, rs, self.v

    def deltaS2(self):
        # refer to J. Chem. Phys. 134, 134101 (2011), ignore some term and some term cancel each other out
        dS2 = (np.einsum('ij,ij->i', self.xycv_a, self.xycv_a)
               + np.einsum('ij,ij->i', self.xycv_b, self.xycv_b)
               - 2 * np.einsum('ij,ij->i', self.xycv_a, self.xycv_b))
        return dS2

    def osc_str(self):
        # oscillator strength
        # f = 2/3 \omega_{0\nu} | \langle 0|\sum_s r_s |\nu\rangle |^2  # length form
        # f = 2/3 \omega_{0\nu}^{-1} | \langle 0|\sum_s \nabla_s |\nu\rangle |^2  # velocity form
        orbo_a = self.mo_coeff[:, self.occidx_a]
        orbv_a = self.mo_coeff[:, self.viridx_a]
        orbo_b = self.mo_coeff[:, self.occidx_b]
        orbv_b = self.mo_coeff[:, self.viridx_b]
        omega = self.e[:self.nstates]
        # length form oscillator strength
        dipole_ao = self.mol.intor_symmetric("int1e_r", comp=3)  # dipole moment, comp=3 is 3 axis
        dipole_mo_a = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo_a, orbv_a)
        dipole_mo_b = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo_b, orbv_b)
        dipole_mo_a = dipole_mo_a.reshape(3, -1)
        dipole_mo_b = dipole_mo_b.reshape(3, -1)[:, self.order[(self.nc+self.no)*self.nv:]-(self.nc+self.no)*self.nv]
        trans_dip_a = np.einsum('xi,yi->yx', dipole_mo_a, self.xy_a)
        trans_dip_b = np.einsum('xi,yi->yx', dipole_mo_b, self.xy_b)
        trans_dip = trans_dip_a + trans_dip_b
        f = 2. / 3. * np.einsum('s,sx,sx->s', omega, trans_dip, trans_dip)
        # np.save("osc_str_stda.npy", f)
        return f

    def rot_str(self):
        # oscillator strength
        # f = 2/3 \omega_{0\nu} | \langle 0|\sum_s r_s |\nu\rangle |^2  # length form
        # f = 2/3 \omega_{0\nu}^{-1} | \langle 0|\sum_s \nabla_s |\nu\rangle |^2  # velocity form
        orbo_a = self.mo_coeff[:, self.occidx_a]
        orbv_a = self.mo_coeff[:, self.viridx_a]
        orbo_b = self.mo_coeff[:, self.occidx_b]
        orbv_b = self.mo_coeff[:, self.viridx_b]
        omega = self.e[:self.nstates]
        dip_ele_ao = self.mol.intor('int1e_ipovlp', comp=3, hermi=2)  # transition electric dipole moment
        dip_ele_mo_a = np.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo_a, orbv_a)
        dip_ele_mo_b = np.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo_b, orbv_b)
        dip_meg_ao = self.mol.intor('int1e_cg_irxp', comp=3, hermi=2)  # transition magnetic dipole moment
        dip_meg_mo_a = np.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo_a, orbv_a)
        dip_meg_mo_b = np.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo_b, orbv_b)

        dip_ele_mo_a = dip_ele_mo_a.reshape(3, -1)
        dip_ele_mo_b = dip_ele_mo_b.reshape(3, -1)[:, self.order[(self.nc+self.no)*self.nv:]-(self.nc+self.no)*self.nv]
        trans_ele_dip_a = np.einsum('xi,yi->yx', dip_ele_mo_a, self.xy_a)
        trans_ele_dip_b = np.einsum('xi,yi->yx', dip_ele_mo_b, self.xy_b)
        trans_ele_dip = -(trans_ele_dip_a + trans_ele_dip_b)
        dip_meg_mo_a = dip_meg_mo_a.reshape(3, -1)
        dip_meg_mo_b = dip_meg_mo_b.reshape(3, -1)[:, self.order[(self.nc+self.no)*self.nv:]-(self.nc+self.no)*self.nv]
        trans_meg_dip_a = np.einsum('xi,yi->yx', dip_meg_mo_a, self.xy_a)
        trans_meg_dip_b = np.einsum('xi,yi->yx', dip_meg_mo_b, self.xy_b)
        trans_meg_dip = 0.5 * (trans_meg_dip_a + trans_meg_dip_b)
        # in Gaussian and ORCA, do not multiply constant
        # f = 1./unit.c * np.einsum('s,sx,sx->s', 1./omega, trans_ele_dip, trans_meg_dip)
        f = np.einsum('s,sx,sx->s', 1. / omega, trans_ele_dip, trans_meg_dip)
        f = f / unit.cgs2au  # transform atom unit to cgs unit
        # np.save("rot_str_stda.npy", f)
        return f

    def analyze(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        print(nc)
        print(no)
        print(nv)
        for nstate in range(self.nstates):
            value = self.v[:,nstate]
            x_cv_aa = value[:nc*nv].reshape(nc,nv)
            x_ov_aa = value[nc*nv:(nc+no)*nv].reshape(no,nv)
            x_co_bb = value[(nc+no)*nv:(nc+no)*nv+nc*no].reshape(nc,no)
            x_cv_bb = value[(nc+no)*nv+nc*no:].reshape(nc, nv)
            print(f'Excited state {nstate + 1} {self.e[nstate] * unit.ha2eV:10.5f} eV')
            for o, v in zip(*np.where(abs(x_cv_aa) > 0.1)):
                print(f'CV(aa) {o + 1}a -> {v + 1 + nc+no}a {x_cv_aa[o, v]:10.5f}')
            for o, v in zip(*np.where(abs(x_ov_aa) > 0.1)):
                print(f'OV(aa) {nc+o + 1}a -> {v + 1+nc+no}a {x_ov_aa[o, v]:10.5f}')
            for o, v in zip(*np.where(abs(x_co_bb) > 0.1)):
                print(f'CO(bb) {o + 1}b -> {v + 1+nc}b {x_co_bb[o, v]:10.5f}')
            for o, v in zip(*np.where(abs(x_cv_bb) > 0.1)):
                print(f'CV(bb) {o + 1}b -> {v + 1 + nc+no}b {x_cv_bb[o, v]:10.5f}')


# spin-conserving spin-adapted x-tda
def _charge_center(mol):
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    return np.einsum('z,zr->r', charges, coords) / charges.sum()


class X_TDA():
    def __init__(self, mf, s_tda=False):

        if np.array(mf.mo_coeff).ndim == 3:  # UKS
            mo_energy = mf.mo_energy
            mo_coeff = mf.mo_coeff
            mo_occ = mf.mo_occ
        else:  # ROKS
            mo_energy = np.array([mf.mo_energy, mf.mo_energy])
            mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            mo_occ = np.zeros((2, len(mf.mo_coeff)))
            mo_occ[0][np.where(mf.mo_occ >= 1)[0]] = 1
            mo_occ[1][np.where(mf.mo_occ >= 2)[0]] = 1

        self.s_tda = s_tda
        self.mol = mf.mol
        self.mf = mf
        nao = self.mol.nao_nr()
        occidx_a = np.where(mo_occ[0] == 1)[0]
        viridx_a = np.where(mo_occ[0] == 0)[0]
        occidx_b = np.where(mo_occ[1] == 1)[0]
        viridx_b = np.where(mo_occ[1] == 0)[0]
        orbo_a = mo_coeff[0][:, occidx_a]
        orbv_a = mo_coeff[0][:, viridx_a]
        orbo_b = mo_coeff[1][:, occidx_b]
        orbv_b = mo_coeff[1][:, viridx_b]
        nocc_a = orbo_a.shape[1]
        nvir_a = orbv_a.shape[1]
        nocc_b = orbo_b.shape[1]
        nvir_b = orbv_b.shape[1]
        mo_a = np.hstack((orbo_a, orbv_a))
        mo_b = np.hstack((orbo_b, orbv_b))
        nmo_a = nocc_a + nvir_a
        nmo_b = nocc_b + nvir_b

        self.eri_mo = ao2mo.general(self.mol, [mo_a, mo_a, mo_a, mo_a], compact=False)  # iajb
        self.eri_mo = self.eri_mo.reshape(nmo_a, nmo_a, nmo_a, nmo_a)
        self.nc = nocc_b
        self.nv = nvir_a
        self.no = abs(nvir_a - nvir_b)
        print('nc ', self.nc)
        print('no ', self.no)
        print('nv ', self.nv)

        dm = mf.make_rdm1()
        vhf = mf.get_veff(mf.mol, dm)
        h1e = mf.get_hcore()
        focka = mo_coeff[0].T @ (h1e + vhf[0]) @ mo_coeff[0]
        fockb = mo_coeff[1].T @ (h1e + vhf[1]) @ mo_coeff[1]

        fab_a = focka[nocc_a:, nocc_a:]
        fab_b = fockb[nocc_b:, nocc_b:]
        fij_a = focka[:nocc_a, :nocc_a]
        fij_b = fockb[:nocc_b, :nocc_b]

        if not s_tda:
            hf = scf.ROHF(self.mol)
            veff = hf.get_veff(self.mol, dm)
            focka2 = mo_coeff[0].T @ (h1e + veff[0]) @ mo_coeff[0]
            fockb2 = mo_coeff[1].T @ (h1e + veff[1]) @ mo_coeff[1]
            fab_a2 = focka2[nocc_a:, nocc_a:]
            fab_b2 = fockb2[nocc_b:, nocc_b:]
            fij_a2 = focka2[:nocc_a, :nocc_a]
            fij_b2 = fockb2[:nocc_b, :nocc_b]

        def get_kxc():
            occ = np.zeros((2, len(mo_occ)))
            occ[0][:len(occidx_a)] = 1
            occ[1][:len(occidx_b)] = 1

            from pyscf.dft import xc_deriv
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            if mf.nlc or ni.libxc.is_nlc(mf.xc):
                logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                                'deriviative is not available. Its contribution is '
                                'not included in the response function.')
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
            print('omega alpha hyb', omega, alpha, hyb)

            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1(mf.mo_coeff, mo_occ)
            # dm0 = mf.make_rdm1(mo_coeff0, mo_occ)
            if np.array(mf.mo_coeff).ndim == 2:
                dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
                # dm0.mo_coeff = (mo_coeff0, mo_coeff0)
            make_rho = ni._gen_rho_evaluator(mf.mol, dm0, hermi=1, with_lapl=False)[0]
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory * .8 - mem_now)
            print('xctype', xctype)

            if xctype == 'LDA':
                ao_deriv = 0
                for ao, mask, weight, coords \
                        in ni.block_loop(self.mol, mf.grids, nao, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, xctype)
                    rho0b = make_rho(1, ao, mask, xctype)
                    rho = (rho0a, rho0b)
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]  # (2,1,2,1,N)
                    wfxc = fxc[:, 0, :, 0] * weight  # (2,2,N)

                    rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                    rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                    rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                    rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                    rho_ov_a = np.einsum('ri,ra->ria', rho_o_a, rho_v_a)
                    rho_ov_b = np.einsum('ri,ra->ria', rho_o_b, rho_v_b)

                    w_ov = np.einsum('ria,r->ria', rho_ov_a, wfxc[0, 0])
                    iajb = lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov)
                    a_aa = iajb
                    # b_aa = iajb

                    w_ov = np.einsum('ria,r->ria', rho_ov_b, wfxc[0, 1])
                    iajb = lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov)
                    a_ab = iajb
                    # b_ab = iajb

                    w_ov = np.einsum('ria,r->ria', rho_ov_b, wfxc[1, 1])
                    iajb = lib.einsum('ria,rjb->iajb', rho_ov_b, w_ov)
                    a_bb = iajb
                    # b_bb = iajb

            elif xctype == 'GGA':
                ao_deriv = 1
                for ao, mask, weight, coords \
                        in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv,
                                         max_memory):  # ao(4,N,nao): AO values and x,y,z compoents in grids
                    rho0a = make_rho(0, ao, mask,
                                     xctype)  # (4,N): density and "density derivatives" for x,y,z components in grids
                    rho0b = make_rho(1, ao, mask, xctype)
                    rho = (rho0a, rho0b)
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, omega=omega, xctype=xctype)[
                        2]  # second order derivatives about\rho
                    wfxc = fxc * weight
                    rho_o_a = lib.einsum('xrp,pi->xri', ao, orbo_a)  # AO values to MO values in grids
                    rho_v_a = lib.einsum('xrp,pi->xri', ao, orbv_a)
                    rho_o_b = lib.einsum('xrp,pi->xri', ao, orbo_b)
                    rho_v_b = lib.einsum('xrp,pi->xri', ao, orbv_b)
                    rho_ov_a = np.einsum('xri,ra->xria', rho_o_a, rho_v_a[0])  # \rho for \alpha
                    rho_ov_b = np.einsum('xri,ra->xria', rho_o_b, rho_v_b[0])
                    rho_ov_a[1:4] += np.einsum('ri,xra->xria', rho_o_a[0],
                                               rho_v_a[1:4])  # (4,N,i,a): \rho values for occupied and virtual orbitals
                    rho_ov_b[1:4] += np.einsum('ri,xra->xria', rho_o_b[0], rho_v_b[1:4])
                    w_ov_aa = np.einsum('xyr,xria->yria', wfxc[0, :, 0], rho_ov_a)  # (4,N,i,a) for \alpha\alpha
                    w_ov_ab = np.einsum('xyr,xria->yria', wfxc[0, :, 1], rho_ov_a)
                    w_ov_bb = np.einsum('xyr,xria->yria', wfxc[1, :, 1], rho_ov_b)

                    iajb = lib.einsum('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                    a_aa = iajb
                    # b_aa = iajb

                    iajb = lib.einsum('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                    a_bb = iajb
                    # b_bb = iajb

                    iajb = lib.einsum('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                    a_ab = iajb
                    # b_ab = iajb
            return a_aa, a_ab, a_bb, hyb

        try:
            xctype = mf.xc
        except:
            xctype = None

        if xctype is not None:
            a_aa, a_ab, a_bb, hyb = get_kxc()
        else:
            hyb = 1

        aa = np.zeros((nocc_a, nvir_a, nocc_a, nvir_a))
        ab = np.zeros((nocc_a, nvir_a, nocc_b, nvir_b))
        ba = np.zeros((nocc_b, nvir_b, nocc_a, nvir_a))
        bb = np.zeros((nocc_b, nvir_b, nocc_b, nvir_b))
        aa += np.einsum('iabj->iajb', self.eri_mo[:nocc_a, nocc_a:, nocc_a:, :nocc_a])
        aa -= hyb * np.einsum('ijba->iajb', self.eri_mo[:nocc_a, :nocc_a, nocc_a:, nocc_a:])
        bb += np.einsum('iabj->iajb', self.eri_mo[:nocc_b, nocc_b:, nocc_b:, :nocc_b])
        bb -= hyb * np.einsum('ijba->iajb', self.eri_mo[:nocc_b, :nocc_b, nocc_b:, nocc_b:])
        ab += np.einsum('iabj->iajb', self.eri_mo[:nocc_a, nocc_a:, nocc_b:, :nocc_b])
        ba += np.einsum('iabj->iajb', self.eri_mo[:nocc_b, nocc_b:, nocc_a:, :nocc_a])

        if xctype is not None:
            aa += a_aa
            ab += a_ab
            bb += a_bb
            ba += a_ab.transpose(2, 3, 0, 1)
        if not s_tda:
            info_eris = (aa, ab, ba, bb, fij_a, fij_b, fab_a, fab_b, fij_a2, fij_b2, fab_a2, fab_b2)
        else:
            info_eris = (aa, ab, ba, bb, fij_a, fij_b, fab_a, fab_b)

        self.A = self.get_matA(info_eris)
        self.nocc_a = nocc_a
        self.nocc_b = nocc_b
        self.nvir_a = nvir_a
        self.nvir_b = nvir_b

    def get_matA(self, info_eris):
        dim = 2 * self.nc * self.nv + self.nc * self.no + self.nv * self.no
        print('Dimension of A matrix ', dim)
        A = np.zeros((dim, dim))
        no = self.no
        nv = self.nv
        nc = self.nc
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv
        iden_C = np.identity(nc)
        iden_O = np.identity(no)
        iden_V = np.identity(nv)

        if not self.s_tda:
            aa, ab, ba, bb, fij_a, fij_b, fab_a, fab_b, fij_a2, fij_b2, fab_a2, fab_b2 = info_eris
        else:
            aa, ab, ba, bb, fij_a, fij_b, fab_a, fab_b = info_eris

        # CV(0)-CV(0) iajb
        A[:dim1, :dim1] += 0.5 * (
                    aa[:-no, :, :-no, :] + bb[:, no:, :, no:] + ab[:-no, :, :, no:] + ba[:, no:, :-no, :] + \
                    np.einsum('ij,ab -> iajb', iden_C, fab_a) - np.einsum('ji,ab->iajb', fij_a[:-no, :-no], iden_V) + \
                    np.einsum('ij,ab-> iajb', iden_C, fab_b[no:, no:]) - np.einsum('ji,ab->iajb', fij_b, iden_V)).reshape(nc * nv, nc * nv)
        # CV(0)-CO(0) iajv
        A_cv0co0 = (1 / np.sqrt(2)) * (ab[:-no, :, :, :no] + bb[:, no:, :, :no] + np.einsum('ij,av->iajv', iden_C, fab_b[no:, :no])).reshape(nc * nv, nc * no)
        A[:dim1, dim1:dim2] += A_cv0co0
        A[dim1:dim2, :dim1] += A_cv0co0.T

        # CV(0)-OV(0) iavb
        A_cv0ov0 = (1 / np.sqrt(2)) * (
                    aa[:-no, :, nc:, :] + ba[:, no:, nc:, :] - np.einsum('vi,ab->iavb', fij_a[nc:, :nc], iden_V)).reshape(nc * nv, nv * no)
        A[:dim1, dim2:dim3] += A_cv0ov0
        A[dim2:dim3, :dim1] += A_cv0ov0.T

        # CV(0)-CV(1)
        si = -np.sqrt((0.5 * self.mol.spin + 1) / (0.5 * self.mol.spin))
        if self.s_tda:
            A_cv0cv1 = si * 0.5 * (
                        aa[:-no, :, :-no, :] - bb[:, no:, :, no:] - ab[:-no, :, :, no:] + ba[:, no:, :-no, :] + \
                        (np.einsum('ij,ab -> iajb', iden_C, fab_a) - np.einsum('ab,ij->iajb', iden_V, fij_a[:-no, :-no])) + \
                        (-np.einsum('ij,ab-> iajb', iden_C, fab_b[no:, no:]) + np.einsum('ab,ij->iajb', iden_V, fij_b))).reshape((nc * nv, nc * nv))
        else:
            A_cv0cv1 = 0.5 * (aa[:nc, :, :nc, :] - bb[:, no:, :, no:] - ab[:nc, :, :, no:] + ba[:, no:, :nc, :] + \
                              +np.einsum('ij,ab -> iajb', iden_C, fab_a) - np.einsum('ab,ij->iajb', iden_V, fij_a[:nc, :nc]) + \
                              -np.einsum('ij,ab-> iajb', iden_C, fab_b[no:, no:]) + np.einsum('ab,ij->iajb', iden_V,fij_b)).reshape((nc * nv, nc * nv)) + \
                       0.5 * (1 + si) * (
                                   np.einsum('ij,ab->iajb', iden_C, fab_b2[no:, no:]) - np.einsum('ij,ab->iajb', iden_C, fab_a2) - \
                                   np.einsum('ab,ij->iajb', iden_V, fij_b2) + np.einsum('ab,ij->iajb', iden_V, fij_a2[:-no, :-no])).reshape(nc * nv, nc * nv)
        A[:dim1, dim3:] += A_cv0cv1
        A[dim3:, :dim1] += A_cv0cv1.T

        # CO(0)-CO(0)
        A[dim1:dim2, dim1:dim2] += bb[:, :no, :, :no].reshape(nc * no, nc * no) + (np.einsum('ij, ab->iajb', iden_C, fab_b[:no, :no].reshape(no, no)) - np.einsum('ab,ji->iajb', iden_O.reshape(no, no), fij_b)).reshape(nc * no, nc * no)

        # CO(0)-OV(0)
        A_co0ov0 = ba[:, :no, nc:, :].reshape(nc * no, nv * no)
        A[dim1:dim2, dim2:dim3] += A_co0ov0
        A[dim2:dim3, dim1:dim2] += A_co0ov0.T

        # CO(0)-CV(1)
        if self.s_tda:
            si = -np.sqrt((0.5 * self.mol.spin + 1) / (0.5 * self.mol.spin))
        else:
            si = 1
        A_co0cv1 = (1 / np.sqrt(2)) * si * (ba[:, :no, :-no, :] - bb[:, :no, :, no:] - np.einsum('ij,ub->iujb', iden_C, fab_b[:no, no:])).reshape(nc * no, nc * nv)
        A[dim1:dim2, dim3:] += A_co0cv1
        A[dim3:, dim1:dim2] += A_co0cv1.T

        # OV(0)-OV(0)
        A[dim2:dim3, dim2:dim3] += aa[nc:, :, nc:, :].reshape(nv * no, nv * no) + \
                                   (np.einsum('ji,ab->iajb', iden_O.reshape(no, no), fab_a) - \
                                    np.einsum('ab,ji->iajb', iden_V, fij_a[nc:, nc:].reshape(no, no))).reshape(nv * no, nv * no)

        # OV(0)-CV(1)
        A_ov0cv1 = (1 / np.sqrt(2)) * si * (aa[nc:, :, :-no, :] - ab[nc:, :, :, no:] - np.einsum('ab,ju->uajb', iden_V, fij_a[:-no, nc:])).reshape(nv * no, nc * nv)
        A[dim2:dim3, dim3:] += A_ov0cv1
        A[dim3:, dim2:dim3] += A_ov0cv1.T

        # CV(1)-CV(1) iajb
        if self.s_tda:
            A[dim3:, dim3:] += 0.5 * (
                        aa[:-no, :, :-no, :] + bb[:, no:, :, no:] - ab[:-no, :, :, no:] - ba[:, no:, :-no, :] + \
                        (1 - 1 / (0.5 * mol.spin)) * np.einsum('ij,ab->iajb', iden_C, fab_a) - \
                        (1 + 1 / (0.5 * mol.spin)) * np.einsum('ab,ij->iajb', iden_V, fij_a[:-no, :-no]) + \
                        (1 + 1 / (0.5 * mol.spin)) * np.einsum('ij,ab->iajb', iden_C, fab_b[no:, no:]) - \
                        (1 - 1 / (0.5 * mol.spin)) * np.einsum('ab,ij->iajb', iden_V, fij_b)).reshape(nc * nv, nc * nv)
        else:
            A[dim3:, dim3:] += 0.5 * (
                        aa[:-no, :, :-no, :] + bb[:, no:, :, no:] - ab[:-no, :, :, no:] - ba[:, no:, :-no, :] + \
                        np.einsum('ij,ab->iajb', iden_C, fab_a) - np.einsum('ab,ij->iajb', iden_V, fij_a[:-no, :-no]) + \
                        np.einsum('ij,ab->iajb', iden_C, fab_b[no:, no:]) - np.einsum('ab,ij->iajb', iden_V, fij_b)).reshape(nc * nv, nc * nv) + \
                               0.5 * (1 / (0.5 * self.mol.spin)) * (
                                           np.einsum('ij,ab->iajb', iden_C, fab_b2[no:, no:]) - np.einsum('ij,ab->iajb', iden_C, fab_a2) + \
                                           np.einsum('ab,ij->iajb', iden_V, fij_b2) - np.einsum('ab,ij->iajb', iden_V, fij_a2[:-no, :-no])).reshape(nc * nv, nc * nv)
        return A

    def analyze_TDM(self):
        with self.mol.with_common_orig(_charge_center(self.mol)):
            ints = self.mol.intor_symmetric('int1e_r', comp=3)  # (3,nao,nao)
        ints_mo = np.einsum('xpq,pi,qj->xij', ints, self.mf.mo_coeff, self.mf.mo_coeff)
        print("Ground state to Excited state transition dipole moments(Au)")
        print('X    Y    Z    OSC.')
        nc = self.nc
        nv = self.nv
        no = self.no
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv
        for i in range(len(self.e)):
            cv0 = self.values[:, i][:dim1].reshape(nc, nv)
            co0 = self.values[:, i][dim1:dim2].reshape(nc, no)
            ov0 = self.values[:, i][dim2:dim3].reshape(no, nv)
            cv1 = self.values[:, i][dim3:].reshape(nc, nv)
            hcv0 = np.einsum('...ia,ia->...', ints_mo[:, :nc, nc + no:], cv0)
            hco0 = np.einsum('...iv,iv->...', ints_mo[:, :nc, nc:nc + no], co0)
            hov0 = np.einsum('...va,va->...', ints_mo[:, nc:nc + no, nc + no:], ov0)
            tdm = np.sqrt(2) * hcv0 + hco0 + hov0
            osc = (2 / 3) * self.e[i] * (tdm[0] ** 2 + tdm[1] ** 2 + tdm[2] ** 2)
            print(i + 1, tdm, osc)

        dip_elec = -self.mf.dip_moment(unit='au')
        nuc_charges = self.mol.atom_charges()
        nuc_coords = self.mol.atom_coords()
        dip_nuc = np.einsum('i,ix->x', nuc_charges, nuc_coords)
        gs = dip_elec + dip_nuc
        print('Ground state dipole moment (in Debye) ', gs)
        print(' ')
        print("Excited state to Excited state transition dipole moments(Au)")
        print('State   State   X    Y    Z    OSC.')
        si = self.mol.spin / 2
        iden_C = np.identity(self.nc)
        iden_O = np.identity(self.no)
        iden_V = np.identity(self.nv)
        dim1 = self.nc * self.nv
        dim2 = dim1 + self.nc * self.no
        dim3 = dim2 + self.no * self.nv
        for i in range(len(self.e)):
            s0_cv0 = self.values[:, i][:dim1].reshape(self.nc, self.nv)
            s0_co0 = self.values[:, i][dim1:dim2].reshape(self.nc, self.no)
            s0_ov0 = self.values[:, i][dim2:dim3].reshape(self.no, self.nv)
            s0_cv1 = self.values[:, i][dim3:].reshape(self.nc, self.nv)
            for j in range(len(self.e)):
                s1_cv0 = self.values[:, j][:dim1].reshape(self.nc, self.nv)
                s1_co0 = self.values[:, j][dim1:dim2].reshape(self.nc, self.no)
                s1_ov0 = self.values[:, j][dim2:dim3].reshape(self.no, self.nv)
                s1_cv1 = self.values[:, j][dim3:].reshape(self.nc, self.nv)
                if i == j:
                    continue

                # diagonal
                # cv0-cv0 delta_ij*r_ab - delta_ab*r_ij
                h_cv0_cv0 = np.einsum('ia,xba,jb,ij->x', s0_cv0, ints_mo[:, self.nc + self.no:, self.nc + self.no:], s1_cv0, iden_C) - \
                            np.einsum('ia,xji,jb,ab->x', s0_cv0, ints_mo[:, :self.nc, :self.nc], s1_cv0, iden_V)
                # print(h_cv0_cv0)
                # co0-co0 delta_ij*r_uv - delta_uv*r_ij
                h_co0_co0 = np.einsum('iu,xvu,jv,ij->x', s0_co0, ints_mo[:, self.nc:self.nc + self.no, self.nc:self.nc + self.no], s1_co0, iden_C) - \
                            np.einsum('iu,xji,jv,uv->x', s0_co0, ints_mo[:, :self.nc, :self.nc], s1_co0, iden_O)
                # print(h_co0_co0)
                # ov0-ov0 delta_uv*r_ab - delta_ab*r_uv
                h_ov0_ov0 = np.einsum('ua,xba,vb,uv->x', s0_ov0, ints_mo[:, self.nc + self.no:, self.nc + self.no:], s1_ov0, iden_O) - \
                            np.einsum('ua,xuv,vb,ab->x', s0_ov0, ints_mo[:, self.nc:self.nc + self.no, self.nc:self.nc + self.no], s1_ov0, iden_V)
                # print(h_ov0_ov0)
                # cv1-cv1 delta_ij*r_ab - delta_ab*r_ij
                h_cv1_cv1 = np.einsum('ia,xba,jb,ij->x', s0_cv1, ints_mo[:, self.nc + self.no:, self.nc + self.no:], s1_cv1, iden_C) - \
                            np.einsum('ia,xji,jb,ab->x', s0_cv1, ints_mo[:, :self.nc, :self.nc], s1_cv1, iden_V)
                # print(h_cv1_cv1)

                # off-diagonal
                factor = np.sqrt((si + 1) / (2 * si))
                # factor = 0.5
                # cv0-co0 1/sqrt{2} * delta_ij*r_av
                h_cv0_co0 = np.einsum('ia,xva,jv,ij->x', s0_cv0, ints_mo[:, self.nc:self.nc + self.no, self.nc + self.no:], s1_co0, iden_C) / np.sqrt(2)
                # print(h_cv0_co0)
                # cv0-ov0 -1/sqrt{2} * delta_ab*r_iv
                h_cv0_ov0 = -np.einsum('ia,xvi,vb,ab->x', s0_cv0, ints_mo[:, self.nc:self.nc + self.no, :self.nc], s1_ov0, iden_V) / np.sqrt(2)
                # print(h_cv0_ov0)

                # co0 -cv0
                h_co0_cv0 = np.einsum('iu,xbu,jb,ij->x', s0_co0, ints_mo[:, self.nc + self.no:, self.nc:self.nc + self.no], s1_cv0, iden_C) / np.sqrt(2)
                # co0-cv1 -1/sqrt{2} * delta_ij*r_ub
                h_co0_cv1 = -factor * np.einsum('iu,xbu,jb,ij->x', s0_co0, ints_mo[:, self.nc + self.no:, self.nc:self.nc + self.no], s1_cv1, iden_C)
                # print(h_co0_cv1)

                # ov0 - cv0
                h_ov0_cv0 = -np.einsum('ua,xju,jb,ab->x', s0_ov0, ints_mo[:, :self.nc, self.nc:self.nc + self.no], s1_cv0, iden_V) / np.sqrt(2)
                # ov0-cv1 -1/sqrt{2} * delta_ab*r_ju
                h_ov0_cv1 = -factor * np.einsum('ua,xju,jb,ab->x', s0_ov0, ints_mo[:, :self.nc, self.nc:self.nc + self.no], s1_cv1, iden_V)
                # print(h_ov0_cv1)

                # cv1-co0
                h_cv1_co0 = -factor * np.einsum('ia,xva,jv,ij->x', s0_cv1, ints_mo[:, self.nc:self.nc + self.no, self.nc + self.no:], s1_co0, iden_C)
                # cv1-ov1
                h_cv1_ov0 = -factor * np.einsum('ia,xvi,vb,ab->x', s0_cv1, ints_mo[:, self.nc:self.nc + self.no, :self.nc], s1_ov0, iden_V)

                tdm = h_cv0_cv0 + h_co0_co0 + h_ov0_ov0 + h_cv1_cv1 + h_cv0_co0 + h_cv0_ov0 + h_co0_cv0 + h_co0_cv1 + h_ov0_cv0 + h_ov0_cv1 + h_cv1_co0 + h_cv1_ov0
                # tdm=h_cv0_cv0+h_co0_co0+h_ov0_ov0+h_cv1_cv1+h_ov0_cv1
                if i == j:
                    tdm += gs
                osc = (2 / 3) * (self.e[i] - self.e[j]) * (tdm[0] ** 2 + tdm[1] ** 2 + tdm[2] ** 2)
                print(f'{i + 1:2d} {j + 1:2d} {tdm[0]:>8.4f} {tdm[1]:>8.4f} {tdm[2]:>8.4f} {osc:>8.4f}')

    def analyze(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv
        for i in range(len(self.e)):
            value = self.values[:, i]
            # print(lib.norm(value)) # 1
            x_cv0 = value[:dim1].reshape(nc, nv)
            x_co0 = value[dim1:dim2].reshape(nc, no)
            x_ov0 = value[dim2:dim3].reshape(no, nv)
            x_cv1 = value[dim3:].reshape(nc, nv)
            Dp_ab = 0.
            print(f'Excited state {i + 1} {self.e[i] * 27.21138505:12.5f} eV')

            for o, v in zip(*np.where(abs(x_cv0) > 0.1)):
                # print(f'CV(0) {o+1}b -> {v+1+self.nc+self.no}b {x_cv_bb[o,v]:10.5f} {100*x_cv_bb[o,v]**2:2.2f}%')
                print(
                    f'CV(0) {o + 1}b -> {v + 1 + self.nc + self.no}b {x_cv0[o, v]:10.5f} {100 * x_cv0[o, v] ** 2:2.2f}%')
            for o, v in zip(*np.where(abs(x_co0) > 0.1)):
                print(f'CO(0) {o + 1}b -> {v + 1 + self.nc}b {x_co0[o, v]:10.5f} {100 * x_co0[o, v] ** 2:5.2f}%')
            for o, v in zip(*np.where(abs(x_ov0) > 0.1)):
                print(
                    f'OV(0) {o + self.nc + 1}a -> {v + 1 + self.nc + self.no}a {x_ov0[o, v]:10.5f} {100 * x_ov0[o, v] ** 2:5.2f}%')
            for o, v in zip(*np.where(abs(x_cv1) > 0.1)):
                # print(f'CV(1) {o+1}a -> {v+1+self.nc+self.no}a {x_cv_aa[o,v]:10.5f} {100*x_cv_aa[o,v]**2:5.2f}%')
                print(
                    f'CV(1) {o + 1}a -> {v + 1 + self.nc + self.no}a {x_cv1[o, v]:10.5f} {100 * x_cv1[o, v] ** 2:5.2f}%')
            print(' ')

    def kernel(self, nstates=1):
        e, v = scipy.linalg.eigh(self.A)
        self.e = e[:nstates]
        self.values = v[:, :nstates]
        self.nstates = nstates
        return self.e * 27.21138505, self.values

if __name__ == "__main__":
    mol = gto.M(
        # atom=atom.ch2o,
        atom=atom.n2,
        # atom=atom.ch2o_vacuum,
        # atom=atom.ch2o_Cyclohexane,
        # atom=atom.ch2o_DiethylEther,
        # atom=atom.ch2o_TetraHydroFuran,
        # basis='cc-pvdz',
        basis='def2-tzvpp',
        # basis='aug-cc-pvtz',
        # basis='sto-3g',
        unit='A',
        # unit='B',
        charge=1,
        spin=1,
        verbose=4
    )
    # path = '/home/whb/Documents/TDDFT/orcabasis/'
    # # bse.convert_formatted_basis_file(path+'sto-3g.bas', path+'sto-3g.nw')
    # # with open(path + "sto-3g.nw", "r") as f:
    # # bse.convert_formatted_basis_file(path+'tzvp.bas', path+'tzvp.nw')
    # with open(path + "tzvp.nw", "r") as f:
    #     basis = f.read()
    # mol.basis = basis
    # mol.build()

    # # add solvents
    # # t_dft0 = time.time()
    # mf = dft.ROKS(mol).SMD()
    # # mf.with_solvent.method = 'COSMO'  # C-PCM, SS(V)PE, COSMO, IEF-PCM
    # # in https://gaussian.com/scrf/ solvents entry, give different eps for different solvents
    # mf.with_solvent.eps = 2.0165  # for Cyclohexane 环己烷
    # # mf.with_solvent.eps = 4.2400  # for DiethylEther 乙醚
    # # mf.with_solvent.eps = 7.4257  # for TetraHydroFuran 四氢呋喃

    mf = dft.ROKS(mol)  # use ROKS is for use ROKS orbital, in paper assume ROKS orbital same with UKS orbital
    # xc = 'svwn'
    # xc = 'blyp'
    # xc = 'b3lyp'
    xc = 'cam-b3lyp'
    # xc = '0.50*HF + 0.50*B88 + GGA_C_LYP'  # BHHLYP
    # xc = 'pbe0'
    # xc = 'pbe38'
    # xc = 'hf'
    mf.xc = xc
    # xc = 'bhhlyp'
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.kernel()
    xtda = XTDA(mol, mf)
    xtda.add_xtda = True
    xtda.nstates = 20
    # tddft.TDA(mf) have no refer meaning
    e_eV, os, rs, v = xtda.kernel()

    # import pandas as pd
    # pd.DataFrame(e_eV).to_csv(xc + 'xTDA.csv')
