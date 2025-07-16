#!/usr/bin/env python
import os
import sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.append('/')
import time
import scipy
import numpy as np
import pandas as pd
from pyscf import dft, gto, scf, tddft, lib, ao2mo
from pyscf.lib import logger

from sTDA import tools
from utils import atom, unit


class UTDA:
    def __init__(self, mol, mf, nstates=10, savedata=False):
        self.mol = mol
        self.mf = mf
        self.nstates = nstates
        self.savedata = savedata

    def pyscf_tda(self, conv_tol=1e-5, is_analyze=False):
        td = tddft.TDA(self.mf)
        # td.singlet = self.singlet
        td.nstates = self.nstates  # calculate nstates excited states
        td.conv_tol = conv_tol  # pyscf default is 1e-5, not 1e-9
        # td.deg_eia_thresh = 1e-9
        e, xy = td.kernel()
        v = np.empty((xy[0][0][0].reshape(-1).shape[0]+xy[0][0][1].reshape(-1).shape[0], 0))
        for vec in xy:
            x, y = vec
            v = np.concatenate(
                (v, np.expand_dims(np.concatenate((x[0].reshape(-1), x[1].reshape(-1))), axis=1)),
                axis=1
            )
        if is_analyze:
            td.analyze()
        # os = td.oscillator_strength(gauge="velocity")
        os = td.oscillator_strength(gauge="length")
        # logger.info(self.mf, "oscillator strength \n{}".format(os))
        # # transform v in pyscf order to my order
        nc, no, nv = tools.get_cov(self.mf)
        order = tools.order_pyscf2my(nc, no, nv)
        v = v[order, :]
        print('my UTDA result is')
        print(f'{"num":>4} {"energy":>8} {"osc_str":>8}')
        for ni, ei, osi in zip(range(self.nstates), e*unit.ha2eV, os):
            print(f'{ni:4d} {ei:8.4f} {osi:8.4f}')
        # np.save('energy_utda.npy', e)
        # np.save('osc_str_utda.npy', os)
        return e, os, v

    def pyscf_get_ab(self):
        from pyscf.tdscf.uhf import get_ab

        mo_occ = self.mf.mo_occ
        occidx_a = np.where(mo_occ[0] == 1)[0]
        viridx_a = np.where(mo_occ[0] == 0)[0]
        occidx_b = np.where(mo_occ[1] == 1)[0]
        viridx_b = np.where(mo_occ[1] == 0)[0]
        nocc_a = len(occidx_a)
        nvir_a = len(viridx_a)
        nocc_b = len(occidx_b)
        nvir_b = len(viridx_b)

        a, b = get_ab(self.mf)
        a_aa = a[0].reshape((nocc_a * nvir_a, nocc_a * nvir_a))
        a_ab = a[1].reshape((nocc_a * nvir_a, nocc_b * nvir_b))
        a_bb = a[2].reshape((nocc_b * nvir_b, nocc_b * nvir_b))
        a = np.block([[a_aa, a_ab],
                      [a_ab.T, a_bb]])
        e, v = scipy.linalg.eigh(a)
        print("use pyscf get_ab func, result is \n{}".format(e[:self.nstates] * unit.ha2eV))

    def kernel(self):
        # Section: prepare for compute
        mo_energy = self.mf.mo_energy
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        assert mo_coeff.dtype == np.float64
        nao = self.mol.nao_nr()
        occidx_a = np.where(mo_occ[0] == 1)[0]
        viridx_a = np.where(mo_occ[0] == 0)[0]
        occidx_b = np.where(mo_occ[1] == 1)[0]
        viridx_b = np.where(mo_occ[1] == 0)[0]
        nocc_a = len(occidx_a)
        nvir_a = len(viridx_a)
        nocc_b = len(occidx_b)
        nvir_b = len(viridx_b)
        orbo_a = mo_coeff[0][:, occidx_a]
        orbv_a = mo_coeff[0][:, viridx_a]
        orbo_b = mo_coeff[1][:, occidx_b]
        orbv_b = mo_coeff[1][:, viridx_b]
        mo_a = np.hstack((orbo_a, orbv_a))
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
        fock_a = mo_a.T @ focka @ mo_a
        fock_b = mo_b.T @ fockb @ mo_b
        delta_ij_a = np.eye(nocc_a)
        delta_ab_a = np.eye(nvir_a)
        delta_ij_b = np.eye(nocc_b)
        delta_ab_b = np.eye(nvir_b)

        # Section: compute two electron integral and XC integral
        # use self.mf.mol general two-electron repulsion integral
        eri_aa = ao2mo.general(self.mol, [orbo_a, mo_a, mo_a, mo_a], compact=False)
        eri_ab = ao2mo.general(self.mol, [orbo_a, mo_a, mo_b, mo_b], compact=False)
        eri_bb = ao2mo.general(self.mol, [orbo_b, mo_b, mo_b, mo_b], compact=False)
        eri_aa = eri_aa.reshape(nocc_a, nmo_a, nmo_a, nmo_a)
        eri_ab = eri_ab.reshape(nocc_a, nmo_a, nmo_b, nmo_b)
        eri_bb = eri_bb.reshape(nocc_b, nmo_b, nmo_b, nmo_b)

        # # use orbital energy replace Fock matrix
        # e_ia_a = (mo_energy[0][viridx_a, None] - mo_energy[0][occidx_a]).T
        # e_ia_b = (mo_energy[1][viridx_b, None] - mo_energy[1][occidx_b]).T
        # aa = np.diag(e_ia_a.ravel()).reshape(nocc_a, nvir_a, nocc_a, nvir_a)
        # bb = np.diag(e_ia_b.ravel()).reshape(nocc_b, nvir_b, nocc_b, nvir_b)
        aa = np.zeros((nocc_a, nvir_a, nocc_a, nvir_a))
        ab = np.zeros((nocc_a, nvir_a, nocc_b, nvir_b))
        bb = np.zeros((nocc_b, nvir_b, nocc_b, nvir_b))

        aa += np.einsum('iabj->iajb', eri_aa[:nocc_a, nocc_a:, nocc_a:, :nocc_a])
        aa -= np.einsum('ijba->iajb', eri_aa[:nocc_a, :nocc_a, nocc_a:, nocc_a:]) * hyb

        bb += np.einsum('iabj->iajb', eri_bb[:nocc_b, nocc_b:, nocc_b:, :nocc_b])
        bb -= np.einsum('ijba->iajb', eri_bb[:nocc_b, :nocc_b, nocc_b:, nocc_b:]) * hyb

        ab += np.einsum('iabj->iajb', eri_ab[:nocc_a, nocc_a:, nocc_b:, :nocc_b])
        # del dm, eri, focka, fockb, h1e, mo_a, mo_b,

        from pyscf.dft import xc_deriv
        ni = self.mf._numint
        ni.libxc.test_deriv_order(self.mf.xc, 2, raise_error=True)
        if self.mf.do_nlc():
            logger.warn(self.mf, 'NLC functional found in DFT object.  Its second '
                                 'derivative is not available. Its contribution is '
                                 'not included in the response function.')

        xctype = ni._xc_type(self.mf.xc)
        dm0 = self.mf.make_rdm1(mo_coeff, mo_occ)
        # dm0.mo_coeff = (mo_coeff, mo_coeff)
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
        elif xctype == 'hf':
            pass

        # Section: construct A matrix
        nc = min(nocc_a, nocc_b)
        no = abs(nocc_a - nocc_b)
        nv = min(nvir_a, nvir_b)
        dim = (nc + no) * nv + nc * (nv + no)
        A = np.zeros((dim, dim))
        # CV(aa)-CV(aa)
        A[:nc * nv, :nc * nv] = (
                np.einsum('ij, ab -> iajb', delta_ij_a[:nc, :nc], fock_a[nc + no:, nc + no:])
                - np.einsum('ij, ab -> iajb', fock_a[:nc, :nc], delta_ab_a)
                + aa[:nc, :, :nc, :]
        ).reshape(nc * nv, nc * nv)

        # CV(aa)-OV(aa)
        Acvaaovaa = (
                np.einsum('ij, ab -> iajb', delta_ij_a[:nc, nc:nc + no], fock_a[nc + no:, nc + no:])
                - np.einsum('ij, ab -> iajb', fock_a[:nc, nc:nc + no], delta_ab_a)
                + aa[:nc, :, nc:nc + no, :]
        ).reshape(nc * nv, no * nv)
        A[:nc * nv, nc * nv:(nc + no) * nv] = Acvaaovaa

        # CV(aa)-CO(bb)
        Acvaacobb = ab[:nc, :, :, :no].reshape(nc * nv, nc * no)
        A[:nc * nv, (nc + no) * nv:(nc + no) * nv + nc * no] = Acvaacobb

        # CV(aa)-CV(bb)
        Acvaacvbb = ab[:nc, :, :, no:no + nv].reshape(nc * nv, nc * nv)
        A[:nc * nv, (nc + no) * nv + nc * no:dim] = Acvaacvbb

        # OV(aa)-CV(aa)
        A[nc * nv:(nc + no) * nv, :nc * nv] = Acvaaovaa.T

        # OV(aa)-OV(aa)
        A[nc * nv:(nc + no) * nv, nc * nv:(nc + no) * nv] = (
                np.einsum('ij, ab -> iajb', delta_ij_a[nc:nc + no, nc:nc + no], fock_a[nc + no:, nc + no:])
                - np.einsum('ij, ab -> iajb', fock_a[nc:nc + no, nc:nc + no], delta_ab_a)
                + aa[nc:nc + no, :, nc:nc + no, :]
        ).reshape(no * nv, no * nv)

        # OV(aa)-CO(bb)
        Aovaacobb = ab[nc:nc + no, :, :, :no].reshape(no * nv, nc * no)
        A[nc * nv:(nc + no) * nv, (nc + no) * nv:(nc + no) * nv + nc * no] = Aovaacobb

        # OV(aa)-CV(bb)
        Aovaacvbb = ab[nc:nc + no, :, :, no:no + nv].reshape(no * nv, nc * nv)
        A[nc * nv:(nc + no) * nv, (nc + no) * nv + nc * no:dim] = Aovaacvbb

        # CO(bb)-CV(aa)
        A[(nc + no) * nv:(nc + no) * nv + nc * no, :nc * nv] = Acvaacobb.T

        # CO(bb)-OV(aa)
        A[(nc + no) * nv:(nc + no) * nv + nc * no, nc * nv:(nc + no) * nv] = Aovaacobb.T

        # CO(bb)-CO(bb)
        A[(nc + no) * nv:(nc + no) * nv + nc * no, (nc + no) * nv:(nc + no) * nv + nc * no] = (
                np.einsum('ij, ab -> iajb', delta_ij_b, fock_b[nc:nc + no, nc:nc + no])
                - np.einsum('ij, ab -> iajb', fock_b[:nc, :nc], delta_ab_b[:no, :no])
                + bb[:, :no, :, :no]
        ).reshape(nc * no, nc * no)

        # CO(bb)-CV(bb)
        Acobbcvbb = (
                np.einsum('ij, ab -> iajb', delta_ij_b, fock_b[nc:nc + no, nc + no:])
                - np.einsum('ij, ab -> iajb', fock_b[:nc, :nc], delta_ab_b[:no, no:])
                + bb[:, :no, :, no:]
        ).reshape(nc * no, nc * nv)
        A[(nc + no) * nv:(nc + no) * nv + nc * no, (nc + no) * nv + nc * no:dim] = Acobbcvbb

        # CV(bb)-CV(aa)
        A[(nc + no) * nv + nc * no:dim, :nc * nv] = Acvaacvbb.T

        # CV(bb)-OV(aa)
        A[(nc + no) * nv + nc * no:dim, nc * nv:(nc + no) * nv] = Aovaacvbb.T

        # CV(bb)-CO(bb)
        A[(nc + no) * nv + nc * no:dim, (nc + no) * nv:(nc + no) * nv + nc * no] = Acobbcvbb.T

        # CV(bb)-CV(bb)
        A[(nc + no) * nv + nc * no:dim, (nc + no) * nv + nc * no:dim] = (
                np.einsum('ij, ab -> iajb', delta_ij_b, fock_b[nc + no:, nc + no:])
                - np.einsum('ij, ab -> iajb', fock_b[:nc, :nc], delta_ab_b[no:, no:])
                + bb[:, no:, :, no:]
        ).reshape(nc * nv, nc * nv)

        # define some class variety, simple to use
        self.mo_coeff_a = mo_a
        self.mo_coeff_b = mo_b
        self.occidx_a = occidx_a
        self.viridx_a = viridx_a
        self.occidx_b = occidx_b
        self.viridx_b = viridx_b
        self.nc = nc
        self.nv = nv
        self.no = no
        # transform pyscf order to my order
        order = np.indices(((nc+no)*nv+nc*(no+nv),)).squeeze()
        for oi in range(nc*no):
            order = np.insert(order, (nc+no)*nv+oi, (nc+no)*nv+oi*nv+oi)
            order = np.delete(order, (nc+no)*nv+oi*nv+oi+1)
        self.order = order

        self.e, self.v = scipy.linalg.eigh(A)
        # print("use my code (block construct A matrix), result is \n{}".format(self.e[:self.nstates] * unit.ha2eV))
        self.e_eV = self.e[:self.nstates] * unit.ha2eV
        self.v = self.v[:, :self.nstates]
        self.xy_a = self.v.T[:, :nocc_a * nvir_a]  # V_alpha
        self.xy_b = self.v.T[:, nocc_a * nvir_a:]  # V_beta
        self.xycv_a = self.v.T[:, :nc * nv]  # V_cv_a
        self.xyov_a = self.v.T[:, nc * nv:(nc + no) * nv]  # V_ov_a
        self.xyco_b = self.v.T[:, (nc + no) * nv:(nc + no) * nv + nc * no]  # V_co_b
        self.xycv_b = self.v.T[:, (nc + no) * nv + nc * no:]  # V_cv_b
        os = self.osc_str()  # oscillator strength
        # before calculate rot_str, check whether mol is chiral mol
        if gto.mole.chiral_mol(self.mol):
            rs = self.rot_str()
        else:
            rs = np.zeros(self.nstates)
            # logger.info(self.mf, 'molecule do not have chiral')
        dS2 = self.deltaS2()
        # logger.info(self.mf, 'oscillator strength (length form) \n{}'.format(self.os))
        # logger.info(self.mf, 'rotatory strength (cgs unit) \n{}'.format(self.rs))
        # logger.info(self.mf, 'deltaS2 is \n{}'.format(dS2))
        print('UTDA result is')
        print(f'{"num":>4} {"energy":>8} {"wav_len":>8} {"osc_str":>8} {"rot_str":>8} {"deltaS2":>8}')
        for ni, ei, wli, osi, rsi, ds2i in zip(range(self.nstates), self.e_eV, unit.eVxnm/self.e_eV, os, rs, dS2):
            print(f'{ni:4d} {ei:8.4f} {wli:8.4f} {osi:8.4f} {rsi:8.4f} {ds2i:8.4f}')
        if self.savedata:
            pd.DataFrame(
                np.concatenate((unit.eVxnm / np.expand_dims(self.e_eV, axis=1), np.expand_dims(os, axis=1)), axis=1)
            ).to_csv('uvspec_data.csv', index=False, header=None)
        return self.e_eV, os, rs, self.v

    def deltaS2(self):
        orbo_a = self.mo_coeff_a[:, self.occidx_a]
        orbv_a = self.mo_coeff_a[:, self.viridx_a]
        orbo_b = self.mo_coeff_b[:, self.occidx_b]
        orbv_b = self.mo_coeff_b[:, self.viridx_b]
        S = self.mol.intor('int1e_ovlp')
        Sccba = np.einsum('pq,pi,qj->ij', S, orbo_b, orbo_a)
        Sccab = np.einsum('pq,pi,qj->ij', S, orbo_a, orbo_b)
        Svcab = np.einsum('pq,pi,qj->ij', S, orbv_a, orbo_b)
        Svcba = np.einsum('pq,pi,qj->ij', S, orbv_b, orbo_a)
        Svvab = np.einsum('pq,pi,qj->ij', S, orbv_a, orbv_b)
        # # error, pyscf order do not same with mine, but difficult to adjust
        # dS2 = (
        #     np.einsum('nia,nja,ki,jk->n',
        #               self.xy_a.reshape(self.nstates, self.nc+self.no, self.nv),
        #               self.xy_a.reshape(self.nstates, self.nc+self.no, self.nv),
        #               Sccba, Sccba.T)
        #     - np.einsum('nia,nib,ak,kb->n',
        #                 self.xy_a.reshape(self.nstates, self.nc+self.no, self.nv),
        #                 self.xy_a.reshape(self.nstates, self.nc+self.no, self.nv),
        #                 Svcab, Svcab.T)
        #     + np.einsum('nia,nja,ki,jk->n',
        #                 self.xy_b.reshape(self.nstates, self.nc, self.no+self.nv),
        #                 self.xy_b.reshape(self.nstates, self.nc, self.no+self.nv),
        #                 Sccab, Sccab.T)
        #     - np.einsum('nia,nib,ak,kb->n',
        #                 self.xy_b.reshape(self.nstates, self.nc, self.no+self.nv),
        #                 self.xy_b.reshape(self.nstates, self.nc, self.no+self.nv),
        #                 Svcba, Svcba.T)
        #     - 2*np.einsum('nia,njb,ji,ab->n',
        #                   self.xy_a.reshape(self.nstates, self.nc+self.no, self.nv),
        #                   self.xy_b.reshape(self.nstates, self.nc, self.no+self.nv),
        #                   Sccba, Svvab)
        # )
        xycv_a = self.xycv_a.reshape(self.nstates, self.nc, self.nv)
        xyov_a = self.xyov_a.reshape(self.nstates, self.no, self.nv)
        xyco_b = self.xyco_b.reshape(self.nstates, self.nc, self.no)
        xycv_b = self.xycv_b.reshape(self.nstates, self.nc, self.nv)
        dS2 = (
            np.einsum('nia,nja,ki,jk->n', xycv_a,xycv_a, Sccba[:, :self.nc], Sccba.T[:self.nc, :])  # first term cvacva
            + np.einsum('nia,nja,ki,jk->n', xyov_a, xyov_a, Sccba[:, self.nc:], Sccba.T[self.nc:, :])  # first term ovaova
            + np.einsum('nia,nja,ki,jk->n', xyov_a, xycv_a, Sccba[:, self.nc:], Sccba.T[:self.nc, :])  # first term ovacva
            + np.einsum('nia,nja,ki,jk->n', xycv_a, xyov_a, Sccba[:, :self.nc], Sccba.T[self.nc:, :])  # first term cvaova
            - np.einsum('nia,nib,ak,kb->n', xycv_a, xycv_a, Svcab, Svcab.T)  # second term  cva
            - np.einsum('nia,nib,ak,kb->n', xyov_a, xyov_a, Svcab, Svcab.T)  # second term ova
            + np.einsum('nia,nja,ki,jk->n', xycv_b, xycv_b, Sccab, Sccab.T)  # third term  cvb
            + np.einsum('nia,nja,ki,jk->n', xyco_b, xyco_b, Sccab, Sccab.T)  # third term cob
            - np.einsum('nia,nib,ak,kb->n', xyco_b, xyco_b, Svcba[:self.no, :], Svcba.T[:, :self.no])  # forth term cobcob
            - np.einsum('nia,nib,ak,kb->n', xycv_b, xycv_b, Svcba[self.no:, :], Svcba.T[:, self.no:])  # forth term cvbcvb
            - np.einsum('nia,nib,ak,kb->n', xyco_b, xycv_b, Svcba[:self.no, :], Svcba.T[:, self.no:])  # forth term cobcvb
            - np.einsum('nia,nib,ak,kb->n', xycv_b, xyco_b, Svcba[self.no:, :], Svcba.T[:, :self.no])  # forth term cvbcob
            - 2 * np.einsum('nia,njb,ji,ab->n', xycv_a, xycv_b, Sccba[:, :self.nc], Svvab[:, self.no:])  # fifth term cvacvb
            - 2 * np.einsum('nia,njb,ji,ab->n', xycv_a, xyco_b, Sccba[:, :self.nc], Svvab[:, :self.no])  # fifth term cvacob
            - 2 * np.einsum('nia,njb,ji,ab->n', xyov_a, xycv_b, Sccba[:, self.nc:], Svvab[:, self.no:])  # fifth term ovacvb
            - 2 * np.einsum('nia,njb,ji,ab->n', xyov_a, xyco_b, Sccba[:, self.nc:], Svvab[:, :self.no])  # fifth term ovacob
        )
        return dS2

    def osc_str(self):
        # oscillator strength
        # f = 2/3 \omega_{0\nu} | \langle 0|\sum_s r_s |\nu\rangle |^2  # length form
        # f = 2/3 \omega_{0\nu}^{-1} | \langle 0|\sum_s \nabla_s |\nu\rangle |^2  # velocity form
        orbo_a = self.mo_coeff_a[:, self.occidx_a]
        orbv_a = self.mo_coeff_a[:, self.viridx_a]
        orbo_b = self.mo_coeff_b[:, self.occidx_b]
        orbv_b = self.mo_coeff_b[:, self.viridx_b]
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
        orbo_a = self.mo_coeff_a[:, self.occidx_a]
        orbv_a = self.mo_coeff_a[:, self.viridx_a]
        orbo_b = self.mo_coeff_b[:, self.occidx_b]
        orbv_b = self.mo_coeff_b[:, self.viridx_b]
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


if __name__ == "__main__":
    mol = gto.M(
        # atom=atom.n2,  # unit is Angstrom
        # atom=atom.ch2o,
        # atom=atom.ch2s,
        # atom=atom.ch2o_vacuum,
        # atom=atom.ch2o_cyclohexane,
        # atom=atom.ch2o_diethylether,
        atom=atom.ch2o_thf,
        basis='cc-pvdz',
        # basis='aug-cc-pvtz',
        # basis='6-31g*',
        # basis='sto-3g',
        # unit='B',
        unit='A',
        charge=1,
        spin=1,
        verbose=4
    )
    # path = '/home/whb/Documents/TDDFT/orcabasis/'
    # # bse.convert_formatted_basis_file(path+'tzvp.bas', path+'tzvp.nw')
    # # with open(path+"tzvp.nw", "r") as f:
    # # bse.convert_formatted_basis_file(path+'ccpvdz.bas', path+'ccpvdz.nw')
    # with open(path+"ccpvdz.nw", "r") as f:
    #     basis = f.read()
    # mol.basis = basis
    # mol.build()

    # add solvents
    t_dft0 = time.time()
    # mf = dft.UKS(mol).SMD()
    mf = dft.UKS(mol).PCM()
    mf.with_solvent.method = 'IEF-PCM'  # C-PCM, SS(V)PE, COSMO, IEF-PCM
    # in https://gaussian.com/scrf/ solvents entry, give different eps for different solvents
    # mf.with_solvent.eps = 2.0165  # for Cyclohexane 环己烷
    # mf.with_solvent.eps = 4.2400  # for DiethylEther 乙醚
    mf.with_solvent.eps = 7.4257  # for TetraHydroFuran 四氢呋喃

    # mf = dft.UKS(mol)
    # xc = 'svwn'
    # xc = 'blyp'
    xc = 'b3lyp'
    # xc = 'pbe0'
    # xc = 'pbe38'
    # xc = 'hf'
    mf.xc = xc
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.conv_check = False
    mf.kernel()
    utda = UTDA(mol, mf)
    utda.nstates = 20
    # utda.pyscf_tda()
    # utda.pyscf_get_ab()

    t0 = time.time()
    e_eV = utda.kernel()
    t1 = time.time()
    print("utda use {} s".format(t1-t0))

    # import pandas as pd
    # pd.DataFrame(e_eV).to_csv(xc + 'UTDA.csv')
