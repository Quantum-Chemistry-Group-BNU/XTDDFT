#!/usr/bin/env python
import os
import sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.append('../')
import scipy
import numpy as np
import pandas as pd
from pyscf import dft, gto, scf, lib, ao2mo, tddft
from pyscf.lib import logger
from utils import unit, atom, tools
# TODO(WHB): this file only calculate alpha spin larger than beta spin


class XTDA:
    def __init__(self, mol, mf, nstates=10, add_xtda=True):
        self.mol = mol
        self.mf = mf
        self.add_xtda = add_xtda
        self.nstates = nstates

    def my_tda(self):
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

        if self.add_xtda:
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
        del dm, eri, focka, fockb, h1e, mo_a, mo_b,

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
        if self.add_xtda:
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
        if self.add_xtda:
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
        if self.add_xtda:
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
        print(f'{"num":>4} {"energy":>8} {"osc_str":>8} {"rot_str":>8} {"deltaS2":>8}')
        for ni, ei, wli, osi, rsi, ds2i in zip(range(self.nstates), self.e_eV, unit.eVxnm/self.e_eV, os, rs, dS2):
            print(f'{ni:4d} {ei:8.4f} {wli:8.4f} {osi:8.4f} {rsi:8.4f} {ds2i:8.4f}')
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

if __name__ == "__main__":
    mol = gto.M(
        # atom=atom.ch2o,
        # atom=atom.n2,
        atom=atom.ch2o_vacuum,
        # atom=atom.ch2o_Cyclohexane,
        # atom=atom.ch2o_DiethylEther,
        # atom=atom.ch2o_TetraHydroFuran,
        basis='cc-pvdz',
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
    xc = 'b3lyp'
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
    e_eV, os, rs, v = xtda.my_tda()

    # import pandas as pd
    # pd.DataFrame(e_eV).to_csv(xc + 'xTDA.csv')

    # mol = gto.M(
    #     # atom=atom.perylene,
    #     atom=atom.ch2s,
    #     # atom=atom.c2h4foh,  # unit is Angstrom
    #     # atom = atom.indigo,
    #     # unit="A",
    #     unit="B",  # https://doi.org/10.1016/j.comptc.2014.02.023 use bohr
    #     # basis='def2-TZVP',
    #     # basis='sto-3g',
    #     spin=1,
    #     charge=1,
    #     verbose=4
    # )
    # path = '/home/whb/Documents/TDDFT/orcabasis/'
    # # bse.convert_formatted_basis_file(path+'tzvp.bas', path+'tzvp.nw')
    # with open(path + "sto-3g.nw", "r") as f:
    # # with open(path + "tzvp.nw", "r") as f:
    #     # bse.convert_formatted_basis_file('../orcabasis/sto-3g.bas', '../orcabasis/sto-3g.nw')
    #     # with open("../orcabasis/sto-3g.nw", "r") as f:
    #     basis = f.read()
    # mol.basis = basis
    # mol.build()
    #
    # # mf = scf.UHF(mol)
    # # mf.conv_tol = 1e-11
    # # mf.conv_tol_grad = 1e-10
    # # mf.max_cycle = 200
    # # mf.conv_check = False
    # # mf.kernel()
    # # print("="*100)
    #
    # t_dft0 = time.time()
    # mf = dft.ROKS(mol)
    # # mf.init_guess = '1e'
    # # mf.init_guess = 'atom'
    # # mf.init_guess = 'huckel'
    # mf.conv_tol = 1e-11
    # mf.conv_tol_grad = 1e-8
    # mf.max_cycle = 200
    # # xc = 'b3lyp'
    # xc = 'PBE0'
    # # xc = 'PBE1PBE'
    # # xc = 'BLYP'
    # mf.xc = xc
    # # mf.grids.level = 9
    # mf.conv_check = False
    # mf.kernel()
    # t_dft1 = time.time()
    # print("dft use {} s".format(t_dft1 - t_dft0))
    #
    # xtda = xtda(mol, mf)
    # xtda.my_xtda()
