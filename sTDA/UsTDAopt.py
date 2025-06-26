#!/usr/bin/env python
import os
import sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.append('../')
import time
import scipy
import numpy as np
import pandas as pd
import basis_set_exchange as bse
from numba import jit
from pyscf import dft, gto, scf, tddft
from pyscf.lib import logger

from utils.eta import eta
from utils import atom, unit, tools
'''lots of code can optimize,
1. according to pyscf constructing method, use index to divide A matrix to each block
2. create qAkcv_aa ... qBkcv_bb instead of create qAk_aa ... qBj_bb
3. construct A matrix block line by block line
'''


def iaia_f(qAk_ss, qAj_ss, qBk_tt, qBj_tt, gamma_k, gamma_j, fock_vir, n1, fock_occ, n2):
    # _ss, _tt represent spin, like qAk_aa ...
    # return iaiak is for add diagonal correction
    iaiak = np.einsum('nmia, nmia, nm->ia', qAk_ss[:, None, ...], qBk_tt[None, ...], gamma_k)
    iaiaj = np.einsum('nmii, nmaa, nm->ia', qAj_ss[:, None, ...], qBj_tt[None, ...], gamma_j)
    iaiak = iaiak.reshape(-1)
    iaiaj = iaiaj.reshape(-1)
    fock = np.einsum('i, a->ia', np.ones(n1), np.diag(fock_vir))
    fock -= np.einsum('i, a->ia', np.diag(fock_occ), np.ones(n2))
    fock = fock.reshape(-1)
    iaia = iaiak.reshape(-1) - iaiaj.reshape(-1) + fock.reshape(-1)
    return iaia, iaiak


def devide_csf_p_n(iaia, t):
    # t: t_P, self.truncate
    pcsf = np.where(iaia * unit.ha2eV <= t)
    ncsf = np.where(iaia * unit.ha2eV > t)
    return pcsf, ncsf


def devide_csf_ps(pcsf, ncsf, scsf):
    pscsfcv_a_i = np.concatenate((pcsf[0], ncsf[0][scsf[scsf<len(ncsf[0])]]), axis=0)
    pscsfcv_a_a = np.concatenate((pcsf[1], ncsf[1][scsf[scsf<len(ncsf[1])]]), axis=0)
    pscsfcv_a_ind = np.argsort(pscsfcv_a_i, stable=True)
    pscsfcv_a_i = pscsfcv_a_i[pscsfcv_a_ind]
    pscsfcv_a_a = pscsfcv_a_a[pscsfcv_a_ind]
    scsf -= len(ncsf[0])
    scsf = np.delete(scsf, np.where(scsf<0))
    return pscsfcv_a_i, pscsfcv_a_a, scsf


def iajb_f(pcsfi, pcsfa, qAk1, qAj1, qBk1, qBj1, qBk2,
           n11, n12, n21, n22, fock_occ, fock_vir,
           ncsf11, ncsf12, ncsf21, ncsf22, ncsf31, ncsf32, ncsf41, ncsf42,
           iajb, iaia_ncsf, iaia, gamma_k, gamma_j, iajbtype):
    """no use, very difficult to all condition"""
    for i, j in zip(pcsfi, pcsfa):
        iajbk1 = np.einsum('nm, nmi, nm->i',
                           qAk1[:, None, i, j], qBk1[None, :, ncsf11, ncsf12], gamma_k)
        iajbj1 = np.einsum('nmi, nmi, nm->i',
                           qAj1[:, None, i, ncsf11], qBj1[None, :, j, ncsf12], gamma_j)
        iajbk2 = np.einsum('nm, nmi, nm->i',
                           qAk1[:, None, i, j], qBk1[None, :, ncsf21, ncsf22], gamma_k)
        iajbj2 = np.einsum('nmi, nmi, nm->i',
                           qAj1[:, None, i, ncsf21], qBj1[None, :, j, ncsf22], gamma_j)
        iajbk3 = np.einsum('nm, nmi, nm->i',
                           qAk1[:, None, i, j], qBk2[None, :, ncsf31, ncsf32], gamma_k)
        iajbk4 = np.einsum('nm, nmi, nm->i',
                           qAk1[:, None, i, j], qBk2[None, :, ncsf41, ncsf42], gamma_k)
        iajbf1 = np.einsum('i, i->i', np.eye(n11)[i, ncsf11], fock_vir[j, ncsf12])
        iajbf1 -= np.einsum('i, i->i', fock_occ[i, ncsf11], np.eye(n12)[j, ncsf12])
        iajbf2 = np.einsum('i, i->i', np.eye(n21)[i, ncsf21], fock_vir[j, ncsf22])
        iajbf2 -= np.einsum('i, i->i', fock_occ[i, ncsf21], np.eye(n22)[j, ncsf22])
        iajb1 = iajbk1 - iajbj1 + iajbf1
        iajb2 = iajbk2 - iajbj2 + iajbf2
        if iajbtype == 'a':
            iajbline = np.concatenate((iajb1, iajb2, iajbk3, iajbk4), axis=0)
        elif iajbtype == 'b':
            iajbline = np.concatenate((iajbk3, iajbk4, iajb1, iajb2), axis=0)
        else:
            raise "iajbtype input error"
        iajb += iajbline ** 2 / (iaia_ncsf - iaia[i, j] + 1e-10)
    return iajb


class UsTDA:
    def __init__(self, mol, mf, truncate=20.0, cas=True, nstates=10, correct=False):
        self.mol = mol
        self.mf = mf
        self.nstates = nstates
        self.cas = cas
        self.truncate = truncate
        self.correct = correct

    def info(self):
        mo_occ = self.mf.mo_occ
        occidx_a = np.where(mo_occ[0] == 1)[0]
        viridx_a = np.where(mo_occ[0] == 0)[0]
        occidx_b = np.where(mo_occ[1] == 1)[0]
        viridx_b = np.where(mo_occ[1] == 0)[0]
        nocc_a = len(occidx_a)
        nvir_a = len(viridx_a)
        nocc_b = len(occidx_b)
        nvir_b = len(viridx_b)
        nc = min(nocc_a, nocc_b)
        nv = min(nvir_a, nvir_b)
        no = nocc_a + nvir_a - nc - nv
        logger.info(self.mf, "nc is {}".format(nc))
        logger.info(self.mf, "nv is {}".format(nv))
        logger.info(self.mf, "no is {}".format(no))
        logger.info(self.mf, "A matrix dim is {}".format((nc + no) * nv + nc * (no + nv)))
        logger.info(self.mf, 'nelectron is {}'.format(self.mol.nelectron))
        logger.info(self.mf, 'natm is {}'.format(self.mol.natm))

    def gamma(self, hyb):
        R = gto.inter_distance(self.mol)  # internuclear distance array, unit is bohr
        alpha1 = 1.42
        alpha2 = 0.48
        beta1 = 0.20
        beta2 = 1.83
        alpha = alpha1 + hyb * alpha2
        beta = beta1 + hyb * beta2  # close shell parameter, https://doi.org/10.1063/1.4811331
        # beta = hyb + 0.3  # open shell parameter, 10.1021/acs.jpca.9b03176
        eta_ele = []  # Note: eta's unit is eV
        for e in self.mol.elements:
            # # eta_ele unit is eV, transform eV to hartree is fine.
            # # inside unit do not need transformation,
            # # or transform inside unit and do not need to transform eV to hartree
            # # eta_ele.append(2 * eta[e] * unit.bohr / unit.ha2eV)  # why do not transform Angstrom to Bohr
            # refer to ORCA origin code: https://github.com/grimme-lab/std2/blob/master/stda.f
            eta_ele.append(2 * eta[e] / unit.ha2eV)
        eta_ele = (np.array(eta_ele)[:, np.newaxis] + np.array(eta_ele)[np.newaxis, :]) / 2
        gj = (1. / (R ** beta + (hyb * eta_ele) ** (-beta))) ** (1. / beta)
        # gj = (1. / (R ** beta + (1.4 * hyb * eta_ele) ** (-beta))) ** (1. / beta)
        gk = (1. / (R ** alpha + eta_ele ** (-alpha))) ** (1. / alpha)
        return gj, gk

    def my_stda(self):
        mo_energy = self.mf.mo_energy
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        assert mo_coeff[0].dtype == np.float64
        mol = self.mf.mol
        occidx_a = np.where(mo_occ[0] == 1)[0]
        viridx_a = np.where(mo_occ[0] == 0)[0]
        occidx_b = np.where(mo_occ[1] == 1)[0]
        viridx_b = np.where(mo_occ[1] == 0)[0]

        mo_coeff_a = mo_coeff[0]
        mo_coeff_b = mo_coeff[1]
        nocc_a = len(occidx_a)
        nvir_a = len(viridx_a)
        nocc_b = len(occidx_b)
        nvir_b = len(viridx_b)

        # check XC functional hybrid proportion
        ni = self.mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, mol.spin)
        print('hyb', hyb)
        # record full A matrix diagonal element, to calculate overlap
        pscsf_fdiag_a = np.zeros((nocc_a, nvir_a), dtype=bool)
        pscsf_fdiag_b = np.zeros((nocc_b, nvir_b), dtype=bool)

        no = self.mol.nelec[0] - self.mol.nelec[1]
        nc = min(nocc_a, nocc_b)
        nv = min(nvir_a, nvir_b)

        # CAS process
        if self.cas:
            delta_mo_occ = mo_occ[0] - mo_occ[1]
            upeomo = np.nonzero(delta_mo_occ)[0]  # un-paired electron occpied molecular orbital
            somo_high = mo_energy[0, upeomo[-1]]
            somo_low = mo_energy[0, upeomo[0]]
            sumo_high = mo_energy[1, upeomo[-1]]
            sumo_low = mo_energy[1, upeomo[0]]
            deps = (1. + 0.8 * hyb) * self.truncate / unit.ha2eV
            vthr_a = (deps * 2 + somo_high)
            othr_a = (somo_low - deps * 2)
            vthr_b = (deps * 2 + sumo_high)
            othr_b = (sumo_low - deps * 2)
            assert (vthr_a > othr_a) and (vthr_b > othr_b)
            nc_a = len(np.nonzero((mo_energy[0] > othr_a) & (mo_energy[0] < somo_low))[0])
            nc_b = len(np.nonzero((mo_energy[1] > othr_b) & (mo_energy[1] < sumo_low))[0])
            nc = max(nc_a, nc_b)
            nv_a = len(np.nonzero((mo_energy[0]<vthr_a) & (mo_energy[0]>sumo_high))[0])
            nv_b = len(np.nonzero((mo_energy[0]<vthr_b) & (mo_energy[0]>somo_high))[0])
            nv = max(nv_a, nv_b)
            act_orb = (upeomo[0]-nc, upeomo[-1]+nv)  # activate space range, activate orbital
            mo_occ_a = mo_occ[0, act_orb[0]:act_orb[1]+1]
            mo_occ_b = mo_occ[1, act_orb[0]:act_orb[1]+1]
            occidx_a = np.where(mo_occ_a > 0.999)[0]
            viridx_a = np.where(mo_occ_a < 0.001)[0]
            occidx_b = np.where(mo_occ_b > 0.999)[0]
            viridx_b = np.where(mo_occ_b < 0.001)[0]
            nocc_a = len(occidx_a)
            nvir_a = len(viridx_a)
            nocc_b = len(occidx_b)
            nvir_b = len(viridx_b)
            mo_coeff_a = mo_coeff_a[:, act_orb[0]:act_orb[1]+1]
            mo_coeff_b = mo_coeff_b[:, act_orb[0]:act_orb[1]+1]

        fock = self.mf.get_fock()
        fock_a = mo_coeff_a.T @ fock[0] @ mo_coeff_a
        fock_b = mo_coeff_b.T @ fock[1] @ mo_coeff_b

        # use sTDA method construct two-electron intergral
        gamma_j, gamma_k = self.gamma(hyb)
        # AO basis function shell(1s,2s ... 4d) range and index(index in overlap for each atom) range
        ao_slices = self.mol.aoslice_by_atom()
        S = self.mol.intor('int1e_ovlp')  # AO overlap, symmetry matrix
        eigvals, eigvecs = np.linalg.eigh(S)
        S_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_prime_a = S_sqrt @ mo_coeff_a
        C_prime_b = S_sqrt @ mo_coeff_b

        # use sTDA method construct two-electron intergral
        qAk_aa = []
        qAj_aa = []
        qAk_bb = []
        qAj_bb = []
        qBj_aa = []
        qBj_bb = []
        for (shl0,shl1,p0,p1) in ao_slices:
            qAk_aa.append(np.einsum('mi, ma->ia', C_prime_a[p0:p1, :nocc_a], C_prime_a[p0:p1, nocc_a:]))  # exchange
            qAj_aa.append(np.einsum('mi, mj->ij', C_prime_a[p0:p1, :nocc_a], C_prime_a[p0:p1, :nocc_a]))  # coulomb
            qAk_bb.append(np.einsum('mi, ma->ia', C_prime_b[p0:p1, :nocc_b], C_prime_b[p0:p1, nocc_b:]))  # exchange
            qAj_bb.append(np.einsum('mi, mj->ij', C_prime_b[p0:p1, :nocc_b], C_prime_b[p0:p1, :nocc_b]))  # coulomb
            qBj_aa.append(np.einsum('ma, mb->ab', C_prime_a[p0:p1, nocc_a:], C_prime_a[p0:p1, nocc_a:]))  # coulomb
            qBj_bb.append(np.einsum('ma, mb->ab', C_prime_b[p0:p1, nocc_b:], C_prime_b[p0:p1, nocc_b:]))  # coulomb
        qAk_aa = np.array(qAk_aa)
        qBk_aa = np.array(qAk_aa)
        qAk_bb = np.array(qAk_bb)
        qBk_bb = np.array(qAk_bb)
        qAj_aa = np.array(qAj_aa)
        qBj_aa = np.array(qBj_aa)
        qAj_bb = np.array(qAj_bb)
        qBj_bb = np.array(qBj_bb)
        fock_a_occ = fock_a[:nocc_a, :nocc_a]
        fock_a_vir = fock_a[nocc_a:, nocc_a:]
        fock_b_occ = fock_b[:nocc_b, :nocc_b]
        fock_b_vir = fock_b[nocc_b:, nocc_b:]
        del S_sqrt, S, eigvals, eigvecs, C_prime_a, C_prime_b, mo_energy, mo_coeff, mo_occ,\
            fock, fock_a, fock_b, p0, p1, shl0, shl1, ao_slices, nocc_a, nocc_b, nvir_a, nvir_b
        if self.truncate:
            # Section: construct diagonal A matrix
            # # CV(aa)-CV(aa)
            # iaiakcv_a = np.einsum('nmia, nmia, nm->ia', qAk_aa[:, None, :nc, :], qBk_aa[None, :, :nc, :], gamma_k)
            # iaiajcv_a = np.einsum('nmii, nmaa, nm->ia', qAj_aa[:, None, :nc, :nc], qBj_aa[None, ...], gamma_j)
            # iaiakcv_a = iaiakcv_a.reshape(-1)
            # iaiajcv_a = iaiajcv_a.reshape(-1)
            # # OV(aa)-OV(aa)
            # iaiakov_a = np.einsum('nmua, nmua, nm->ua', qAk_aa[:, None, nc:, :], qBk_aa[None, :, nc:, :], gamma_k)
            # iaiajov_a = np.einsum('nmuu, nmaa, nm->ua', qAj_aa[:, None, nc:, nc:], qBj_aa[None, ...], gamma_j)
            # iaiakov_a = iaiakov_a.reshape(-1)
            # iaiajov_a = iaiajov_a.reshape(-1)
            # # CO(bb)-CO(bb)
            # iaiakco_b = np.einsum('nmiu, nmiu, nm->iu', qAk_bb[:, None, :, :no], qBk_bb[None, :, :, :no], gamma_k)
            # iaiajco_b = np.einsum('nmii, nmuu, nm->iu', qAj_bb[:, None, ...], qBj_bb[None, :, :no, :no], gamma_j)
            # iaiakco_b = iaiakco_b.reshape(-1)
            # iaiajco_b = iaiajco_b.reshape(-1)
            # # CV(bb)-CV(bb)
            # iaiakcv_b = np.einsum('nmia, nmia, nm->ia', qAk_bb[:, None, :, no:], qBk_bb[None, :, :, no:], gamma_k)
            # iaiajcv_b = np.einsum('nmii, nmaa, nm->ia', qAj_bb[:, None, ...], qBj_bb[None, :, no:, no:], gamma_j)
            # iaiakcv_b = iaiakcv_b.reshape(-1)
            # iaiajcv_b = iaiajcv_b.reshape(-1)
            # # combine each block
            # iaiak = np.concatenate((iaiakcv_a, iaiakov_a, iaiakco_b, iaiakcv_b), axis=0)
            # iaiaj = np.concatenate((iaiajcv_a, iaiajov_a, iaiajco_b, iaiajcv_b), axis=0)
            # # add diagonal correction
            # if self.correct:
            #     delta_max = 0.5 / unit.ha2eV
            #     sigma_k = 0.1 / unit.ha2eV
            #     delta_k = delta_max / (1 + (iaiak / sigma_k) ** 4)
            #     iaiak += delta_k
            # # A matrix main diagonal element (do not include fock)
            # iaia = iaiak - iaiaj
            # del iaiakcv_a, iaiajcv_a, iaiakov_a, iaiajov_a, iaiakco_b, iaiajco_b, iaiakcv_b, iaiajcv_b, iaiak, iaiaj
            # # fock contribution
            # fockcv_a = np.einsum('i, a->ia', np.ones(nc), np.diag(fock_a_vir))  # fockcv_a_a
            # fockcv_a -= np.einsum('i, a->ia', np.diag(fock_a_occ[:nc, :nc]), np.ones(nv))  # fockcv_a_i
            # fockcv_a = fockcv_a.reshape(-1)
            # fockov_a = np.einsum('u, a->ua', np.ones(no), np.diag(fock_a_vir))  # fockov_a_a
            # fockov_a -= np.einsum('u, a->ua', np.diag(fock_a_occ[nc:, nc:]), np.ones(nv))  # fockov_a_u
            # fockov_a = fockov_a.reshape(-1)
            # fockco_b = np.einsum('i, u->iu', np.ones(nc), np.diag(fock_b_vir[:no, :no]))  # fockco_b_u
            # fockco_b -= np.einsum('i, u->iu', np.diag(fock_b_occ), np.ones(no))  # fockco_b_i
            # fockco_b = fockco_b.reshape(-1)
            # fockcv_b = np.einsum('i, a->ia', np.ones(nc), np.diag(fock_b_vir[no:, no:]))  # fockcv_b_a
            # fockcv_b -= np.einsum('i, a->ia', np.diag(fock_b_occ), np.ones(nv))  # fockcv_b_i
            # fockcv_b = fockcv_b.reshape(-1)
            # iaia += np.concatenate((fockcv_a, fockov_a, fockco_b, fockcv_b), axis=0)
            # del fockcv_a, fockov_a, fockco_b, fockcv_b

            # # encapsulate to a function, same with upper code
            # CV(aa)-CV(aa)
            iaiacv_a, iaiakcv_a = iaia_f(qAk_aa[:, :nc, :], qAj_aa[:, :nc, :nc],
                                         qBk_aa[:, :nc, :], qBj_aa[...], gamma_k, gamma_j,
                                         fock_a_vir, nc, fock_a_occ[:nc, :nc], nv)
            # OV(aa)-OV(aa)
            iaiaov_a, iaiakov_a = iaia_f(qAk_aa[:, nc:, :], qAj_aa[:, nc:, nc:],
                                         qBk_aa[:, nc:, :], qBj_aa[...], gamma_k, gamma_j,
                                         fock_a_vir, no, fock_a_occ[nc:, nc:], nv)
            # CO(bb)-CO(bb)
            iaiaco_b, iaiakco_b = iaia_f(qAk_bb[:, :, :no], qAj_bb[...],
                                         qBk_bb[:, :, :no], qBj_bb[:, :no, :no], gamma_k, gamma_j,
                                         fock_b_vir[:no, :no], nc, fock_b_occ, no)
            # CV(bb)-CV(bb)
            iaiacv_b, iaiakcv_b = iaia_f(qAk_bb[:, :, no:], qAj_bb[...],
                                         qBk_bb[:, :, no:], qBj_bb[:, no:, no:], gamma_k, gamma_j,
                                         fock_b_vir[no:, no:], nc, fock_b_occ, nv)
            iaiak = np.concatenate((iaiakcv_a, iaiakov_a, iaiakco_b, iaiakcv_b), axis=0)
            iaia = np.concatenate((iaiacv_a, iaiaov_a, iaiaco_b, iaiacv_b), axis=0)
            if self.correct:
                delta_max = 0.5 / unit.ha2eV
                sigma_k = 0.1 / unit.ha2eV
                delta_k = delta_max / (1 + (iaiak / sigma_k) ** 4)
                iaia += delta_k
            del iaiacv_a, iaiaov_a, iaiaco_b, iaiacv_b, iaiakcv_a, iaiakov_a, iaiakco_b, iaiakcv_b, iaiak

            # Section: select P-CSF and construct nondiagonal A matrix
            iaiacv_a = iaia[:nc*nv].reshape(nc, nv)
            iaiaov_a = iaia[nc*nv:(nc+no)*nv].reshape(no, nv)
            iaiaco_b = iaia[(nc+no)*nv:(nc+no)*nv+nc*no].reshape(nc, no)
            iaiacv_b = iaia[(nc+no)*nv+nc*no:].reshape(nc, nv)
            # pcsfcv_a = np.where(iaiacv_a * unit.ha2eV <= self.truncate)
            # ncsfcv_a = np.where(iaiacv_a * unit.ha2eV > self.truncate)
            # pcsfov_a = np.where(iaiaov_a * unit.ha2eV <= self.truncate)
            # ncsfov_a = np.where(iaiaov_a * unit.ha2eV > self.truncate)
            # pcsfco_b = np.where(iaiaco_b * unit.ha2eV <= self.truncate)
            # ncsfco_b = np.where(iaiaco_b * unit.ha2eV > self.truncate)
            # pcsfcv_b = np.where(iaiacv_b * unit.ha2eV <= self.truncate)
            # ncsfcv_b = np.where(iaiacv_b * unit.ha2eV > self.truncate)
            pcsfcv_a, ncsfcv_a = devide_csf_p_n(iaiacv_a, self.truncate)
            pcsfov_a, ncsfov_a = devide_csf_p_n(iaiaov_a, self.truncate)
            pcsfco_b, ncsfco_b = devide_csf_p_n(iaiaco_b, self.truncate)
            pcsfcv_b, ncsfcv_b = devide_csf_p_n(iaiacv_b, self.truncate)
            iajb = np.zeros(len(ncsfcv_a[0])+len(ncsfov_a[0])+len(ncsfco_b[0])+len(ncsfcv_b[0]))
            iaia_ncsf = np.concatenate((iaiacv_a[ncsfcv_a[0], ncsfcv_a[1]], iaiaov_a[ncsfov_a[0], ncsfov_a[1]],
                                   iaiaco_b[ncsfco_b[0], ncsfco_b[1]], iaiacv_b[ncsfcv_b[0], ncsfcv_b[1]]), axis=0)
            for i, j in zip(pcsfcv_a[0], pcsfcv_a[1]):
                iajbcv_a = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, i, j], qBk_aa[None, :, ncsfcv_a[0], ncsfcv_a[1]], gamma_k)
                iajbcv_a -= np.einsum('nmi, nmi, nm->i', qAj_aa[:, None, i, ncsfcv_a[0]], qBj_aa[None, :, j, ncsfcv_a[1]], gamma_j)
                iajbcv_a += np.einsum('i, i->i', np.eye(nc)[i, ncsfcv_a[0]], fock_a_vir[j, ncsfcv_a[1]])
                iajbcv_a -= np.einsum('i, i->i', fock_a_occ[i, ncsfcv_a[0]], np.eye(nv)[j, ncsfcv_a[1]])
                # Note: calculate ov_a, so add nc and ncsfov_a, others is same
                iajbov_a = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, i, j], qBk_aa[None, :, nc+ncsfov_a[0], ncsfov_a[1]], gamma_k)
                iajbov_a -= np.einsum('nmi, nmi, nm->i', qAj_aa[:, None, i, nc+ncsfov_a[0]], qBj_aa[None, :, j, ncsfov_a[1]], gamma_j)
                iajbov_a += np.einsum('i, i->i', np.eye(nc + no)[i, nc + ncsfov_a[0]], fock_a_vir[j, ncsfov_a[1]])
                iajbov_a -= np.einsum('i, i->i', fock_a_occ[i, nc + ncsfov_a[0]], np.eye(nv)[j, ncsfov_a[1]])
                iajbkco_b = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, i, j], qBk_bb[None, :, ncsfco_b[0], ncsfco_b[1]], gamma_k)
                iajbkcv_b = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, i, j], qBk_bb[None, :, ncsfcv_b[0], no+ncsfcv_b[1]], gamma_k)
                iajbline = np.concatenate((iajbcv_a, iajbov_a, iajbkco_b, iajbkcv_b), axis=0)
                iajb += iajbline**2/(iaia_ncsf-iaiacv_a[i, j]+1e-10)
            for i, j in zip(pcsfov_a[0], pcsfov_a[1]):
                iajbcv_a = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, nc+i, j], qBk_aa[None, :, ncsfcv_a[0], ncsfcv_a[1]], gamma_k)
                iajbcv_a -= np.einsum('nmi, nmi, nm->i', qAj_aa[:, None, nc+i, ncsfcv_a[0]], qBj_aa[None, :, j, ncsfcv_a[1]], gamma_j)
                iajbcv_a += np.einsum('i, i->i', np.eye(nc+no)[nc+i, ncsfcv_a[0]], fock_a_vir[j, ncsfcv_a[1]])
                iajbcv_a -= np.einsum('i, i->i', fock_a_occ[nc+i, ncsfcv_a[0]], np.eye(nv)[j, ncsfcv_a[1]])
                iajbov_a = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, nc+i, j], qBk_aa[None, :, nc+ncsfov_a[0], ncsfov_a[1]], gamma_k)
                iajbov_a -= np.einsum('nmi, nmi, nm->i', qAj_aa[:, None, nc+i, nc+ncsfov_a[0]], qBj_aa[None, :, j, ncsfov_a[1]], gamma_j)
                iajbov_a += np.einsum('i, i->i', np.eye(nc+no)[nc+i, nc+ncsfov_a[0]], fock_a_vir[j, ncsfov_a[1]])
                iajbov_a -= np.einsum('i, i->i', fock_a_occ[nc+i, nc+ncsfov_a[0]], np.eye(nv)[j, ncsfov_a[1]])
                iajbkco_b = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, nc+i, j], qBk_bb[None, :, ncsfco_b[0], ncsfco_b[1]], gamma_k)
                iajbkcv_b = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, nc+i, j], qBk_bb[None, :, ncsfcv_b[0], no+ncsfcv_b[1]], gamma_k)
                iajbline = np.concatenate((iajbcv_a, iajbov_a, iajbkco_b, iajbkcv_b), axis=0)
                # Note: iaiaov_a[i, j] can not use nc+i as index, so iajb_f do not suit for this iteration
                iajb += iajbline ** 2 / (iaia_ncsf - iaiaov_a[i, j] + 1e-10)
            for i, j in zip(pcsfco_b[0], pcsfco_b[1]):
                iajbkcv_a = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, j], qBk_aa[None, :, ncsfcv_a[0], ncsfcv_a[1]], gamma_k)
                iajbkov_a = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, j], qBk_aa[None, :, nc+ncsfov_a[0], ncsfov_a[1]], gamma_k)
                iajbco_b = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, j], qBk_bb[None, :, ncsfco_b[0], ncsfco_b[1]], gamma_k)
                iajbco_b -= np.einsum('nmi, nmi, nm->i', qAj_bb[:, None, i, ncsfco_b[0]], qBj_bb[None, :, j, ncsfco_b[1]], gamma_j)
                iajbco_b += np.einsum('i, i->i', np.eye(nc)[i, ncsfco_b[0]], fock_b_vir[j, ncsfco_b[1]])
                iajbco_b -= np.einsum('i, i->i', fock_b_occ[i, ncsfco_b[0]], np.eye(no+nv)[j, ncsfco_b[1]])
                iajbcv_b = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, j], qBk_bb[None, :, ncsfcv_b[0], no+ncsfcv_b[1]], gamma_k)
                iajbcv_b -= np.einsum('nmi, nmi, nm->i', qAj_bb[:, None, i, ncsfcv_b[0]], qBj_bb[None, :, j, no+ncsfcv_b[1]], gamma_j)
                iajbcv_b += np.einsum('i, i->i', np.eye(nc)[i, ncsfcv_b[0]], fock_b_vir[j, no+ncsfcv_b[1]])
                iajbcv_b -= np.einsum('i, i->i', fock_b_occ[i, ncsfcv_b[0]], np.eye(no+nv)[j, no+ncsfcv_b[1]])
                iajbline = np.concatenate((iajbkcv_a, iajbkov_a, iajbco_b, iajbcv_b), axis=0)
                iajb += iajbline ** 2 / (iaia_ncsf - iaiaco_b[i, j] + 1e-10)
            for i, j in zip(pcsfcv_b[0], pcsfcv_b[1]):
                iajbkcv_a = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, no+j], qBk_aa[None, :, ncsfcv_a[0], ncsfcv_a[1]], gamma_k)
                iajbkov_a = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, no+j], qBk_aa[None, :, nc + ncsfov_a[0], ncsfov_a[1]], gamma_k)
                iajbco_b = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, no+j], qBk_bb[None, :, ncsfco_b[0], ncsfco_b[1]], gamma_k)
                iajbco_b -= np.einsum('nmi, nmi, nm->i', qAj_bb[:, None, i, ncsfco_b[0]], qBj_bb[None, :, no+j, ncsfco_b[1]], gamma_j)
                iajbco_b += np.einsum('i, i->i', np.eye(nc)[i, ncsfco_b[0]], fock_b_vir[no+j, ncsfco_b[1]])
                iajbco_b -= np.einsum('i, i->i', fock_b_occ[i, ncsfco_b[0]], np.eye(no + nv)[no+j, ncsfco_b[1]])
                iajbcv_b = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, no+j], qBk_bb[None, :, ncsfcv_b[0], no + ncsfcv_b[1]], gamma_k)
                iajbcv_b -= np.einsum('nmi, nmi, nm->i', qAj_bb[:, None, i, ncsfcv_b[0]], qBj_bb[None, :, no+j, no + ncsfcv_b[1]], gamma_j)
                iajbcv_b += np.einsum('i, i->i', np.eye(nc)[i, ncsfcv_b[0]], fock_b_vir[no+j, no + ncsfcv_b[1]])
                iajbcv_b -= np.einsum('i, i->i', fock_b_occ[i, ncsfcv_b[0]], np.eye(no + nv)[no+j, no + ncsfcv_b[1]])
                iajbline = np.concatenate((iajbkcv_a, iajbkov_a, iajbco_b, iajbcv_b), axis=0)
                # Note: iaiacv_b[i, j] can not use no+j as index, so iajb_f do not suit for this iteration
                iajb += iajbline ** 2 / (iaia_ncsf - iaiacv_b[i, j] + 1e-10)
            del iajbkcv_a, iajbkov_a, iajbkco_b, iajbkcv_b, iajbcv_a, iajbov_a, iajbco_b, iajbcv_b,\
                iaia, iajbline, iaia_ncsf, iaiacv_a, iaiaov_a, iaiaco_b, iaiacv_b
            scsf = np.where(iajb>=1e-4)[0]
            # # CV(aa)
            # pscsfcv_a_i = np.concatenate((pcsfcv_a[0], ncsfcv_a[0][scsf[scsf<len(ncsfcv_a[0])]]), axis=0)
            # pscsfcv_a_a = np.concatenate((pcsfcv_a[1], ncsfcv_a[1][scsf[scsf<len(ncsfcv_a[1])]]), axis=0)
            # pscsfcv_a_ind = np.argsort(pscsfcv_a_i, stable=True)
            # pscsfcv_a_i = pscsfcv_a_i[pscsfcv_a_ind]
            # pscsfcv_a_a = pscsfcv_a_a[pscsfcv_a_ind]
            # scsf -= len(ncsfcv_a[0])
            # scsf = np.delete(scsf, np.where(scsf<0))
            # # OV(aa)
            # pscsfov_a_i = np.concatenate((pcsfov_a[0], ncsfov_a[0][scsf[scsf<len(ncsfov_a[0])]]), axis=0)
            # pscsfov_a_a = np.concatenate((pcsfov_a[1], ncsfov_a[1][scsf[scsf<len(ncsfov_a[1])]]), axis=0)
            # pscsfov_a_ind = np.argsort(pscsfov_a_i, stable=True)
            # pscsfov_a_i = pscsfov_a_i[pscsfov_a_ind]
            # pscsfov_a_a = pscsfov_a_a[pscsfov_a_ind]
            # scsf -= len(ncsfov_a[0])
            # scsf = np.delete(scsf, np.where(scsf < 0))
            # # CO(bb)
            # pscsfco_b_i = np.concatenate((pcsfco_b[0], ncsfco_b[0][scsf[scsf<len(ncsfco_b[0])]]), axis=0)
            # pscsfco_b_a = np.concatenate((pcsfco_b[1], ncsfco_b[1][scsf[scsf<len(ncsfco_b[1])]]), axis=0)
            # pscsfco_b_ind = np.argsort(pscsfco_b_i, stable=True)
            # pscsfco_b_i = pscsfco_b_i[pscsfco_b_ind]
            # pscsfco_b_a = pscsfco_b_a[pscsfco_b_ind]
            # scsf -= len(ncsfco_b[0])
            # scsf = np.delete(scsf, np.where(scsf < 0))
            # # CV(bb)
            # pscsfcv_b_i = np.concatenate((pcsfcv_b[0], ncsfcv_b[0][scsf[scsf<len(ncsfcv_b[0])]]), axis=0)
            # pscsfcv_b_a = np.concatenate((pcsfcv_b[1], ncsfcv_b[1][scsf[scsf<len(ncsfcv_b[1])]]), axis=0)
            # pscsfcv_b_ind = np.argsort(pscsfcv_b_i, stable=True)
            # pscsfcv_b_i = pscsfcv_b_i[pscsfcv_b_ind]
            # pscsfcv_b_a = pscsfcv_b_a[pscsfcv_b_ind]

            # # encapsulate to a function, same with upper code
            # CV(aa)
            pscsfcv_a_i, pscsfcv_a_a, scsf = devide_csf_ps(pcsfcv_a, ncsfcv_a, scsf)
            # OV(aa)
            pscsfov_a_i, pscsfov_a_a, scsf = devide_csf_ps(pcsfov_a, ncsfov_a, scsf)
            # CO(bb)
            pscsfco_b_i, pscsfco_b_a, scsf = devide_csf_ps(pcsfco_b, ncsfco_b, scsf)
            # CV(bb)
            pscsfcv_b_i, pscsfcv_b_a, scsf = devide_csf_ps(pcsfcv_b, ncsfcv_b, scsf)
            Adim = len(pscsfcv_a_i) + len(pscsfov_a_i) + len(pscsfco_b_i) + len(pscsfcv_b_i)
            pcsfdim = len(pcsfcv_a[0]) + len(pcsfov_a[0]) + len(pcsfco_b[0]) + len(pcsfcv_b[0])
            scsfdim = len(pscsfcv_a_i) + len(pscsfov_a_i) + len(pscsfco_b_i) + len(pscsfcv_b_i) - pcsfdim
            ncsfdim = len(iajb) - scsfdim
            logger.info(self.mf, 'A matrix dimension is {}'.format(Adim))
            logger.info(self.mf, '{} CSFs in pcsf'.format(pcsfdim))
            logger.info(self.mf, '{} CSFs in scsf'.format(scsfdim))
            logger.info(self.mf, '{} CSFs in ncsf'.format(ncsfdim))
        else:
            Adim = (nc+no)*nv+nc*(no+nv)
            logger.info(self.mf, 'no * nv is {}'.format(Adim))
            pscsfcv_a_i, pscsfcv_a_a = np.indices((nc, nv))
            pscsfov_a_i, pscsfov_a_a = np.indices((no, nv))
            pscsfco_b_i, pscsfco_b_a = np.indices((nc, no))
            pscsfcv_b_i, pscsfcv_b_a = np.indices((nc, nv))
            pscsfcv_a_i = pscsfcv_a_i.reshape(-1)
            pscsfcv_a_a = pscsfcv_a_a.reshape(-1)
            pscsfov_a_i = pscsfov_a_i.reshape(-1)
            pscsfov_a_a = pscsfov_a_a.reshape(-1)
            pscsfco_b_i = pscsfco_b_i.reshape(-1)
            pscsfco_b_a = pscsfco_b_a.reshape(-1)
            pscsfcv_b_i = pscsfcv_b_i.reshape(-1)
            pscsfcv_b_a = pscsfcv_b_a.reshape(-1)

        # Section: calculate overlap need transform different order to same, so here calculate order
        pscsf_fdiag_a[pscsfcv_a_i+pscsf_fdiag_a.shape[0]-nc-no, pscsfcv_a_a] = True
        pscsf_fdiag_a[pscsfov_a_i+pscsf_fdiag_a.shape[0]-nc-no+nc, pscsfov_a_a] = True
        pscsf_fdiag_b[pscsfco_b_i+pscsf_fdiag_b.shape[0]-nc, pscsfco_b_a] = True
        pscsf_fdiag_b[pscsfcv_b_i+pscsf_fdiag_b.shape[0]-nc, no+pscsfcv_b_a] = True
        # Note: pscsf_fdiag order is pyscf order, so if calculate overlap, adjust pyscf csf order to my order
        pscsf_fdiag = np.concatenate((pscsf_fdiag_a.reshape(-1), pscsf_fdiag_b.reshape(-1)))
        nc_old, no_old, nv_old = tools.get_cov(self.mf)
        order = tools.order_pyscf2my(nc_old, no_old, nv_old)
        pscsf_fdiag = pscsf_fdiag[order]

        # Section: construct A
        A = np.zeros((Adim, Adim))
        qAkcv_aa = qAk_aa[:, pscsfcv_a_i, pscsfcv_a_a]
        qAkov_aa = qAk_aa[:, nc+pscsfov_a_i, pscsfov_a_a]
        qAkco_bb = qAk_bb[:, pscsfco_b_i, pscsfco_b_a]
        qAkcv_bb = qAk_bb[:, pscsfcv_b_i, no+pscsfcv_b_a]
        qBkcv_aa = np.array(qAkcv_aa)
        qBkov_aa = np.array(qAkov_aa)
        qBkco_bb = np.array(qAkco_bb)
        qBkcv_bb = np.array(qAkcv_bb)
        lcva = len(pscsfcv_a_i)
        lova = len(pscsfov_a_i)
        lcob = len(pscsfco_b_i)
        lcvb = len(pscsfcv_b_i)
        if self.correct:
            delta_max = 0.5 / unit.ha2eV
            sigma_k = 0.1 / unit.ha2eV
        # CV(aa)-CV(aa)
        int2ekcv_a = np.einsum('nmi, nma, nm->ia', qAkcv_aa[:, None, ...], qBkcv_aa[None, ...], gamma_k)
        int2ejcv_a = np.zeros_like(int2ekcv_a)
        for i, a, p in zip(pscsfcv_a_i, pscsfcv_a_a, range(lcva)):
            int2ejcv_a[p, :] = np.einsum('nmp, nmp, nm->p', qAj_aa[:, None, i, pscsfcv_a_i], qBj_aa[None, :, a, pscsfcv_a_a], gamma_j)
            A[p, :lcva] -= np.einsum('p, p->p', fock_a_occ[i, pscsfcv_a_i], np.eye(nv)[a, pscsfcv_a_a])
            A[p, :lcva] += np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_a_i], fock_a_vir[a, pscsfcv_a_a])
        A[:lcva, :lcva] += int2ekcv_a - int2ejcv_a
        if self.correct:
            delta_k = delta_max / (1 + (np.diag(int2ekcv_a) / sigma_k) ** 4)
            A[:lcva, :lcva] += np.diag(delta_k)
        del int2ekcv_a, int2ejcv_a
        # CV(aa)-OV(aa)
        int2ekcvaova = np.einsum('nmi, nma, nm->ia', qAkcv_aa[:, None, ...], qBkov_aa[None, ...], gamma_k)
        int2ejcvaova = np.zeros_like(int2ekcvaova)
        for i, a, p in zip(pscsfcv_a_i, pscsfcv_a_a, range(lcva)):
            int2ejcvaova[p, :] = np.einsum('nmp, nmp, nm->p', qAj_aa[:, None, i, nc+pscsfov_a_i], qBj_aa[None, :, a, pscsfov_a_a], gamma_j)
            A[p, lcva:lcva+lova] -= np.einsum('p, p->p', fock_a_occ[i, nc+pscsfov_a_i], np.eye(nv)[a, pscsfov_a_a])
            A[p, lcva:lcva+lova] += np.einsum('p, p->p', np.eye(nc+no)[i, nc+pscsfov_a_i], fock_a_vir[a, pscsfov_a_a])
        A[:lcva, lcva:lcva+lova] += int2ekcvaova - int2ejcvaova
        del int2ekcvaova, int2ejcvaova
        # CV(aa)-CO(bb)
        A[:lcva, lcva+lova:lcva+lova+lcob] = np.einsum('nmi, nma, nm->ia', qAkcv_aa[:, None, ...], qBkco_bb[None, ...], gamma_k)
        # CV(aa)-CV(bb)
        A[:lcva, lcva+lova+lcob:] = np.einsum('nmi, nma, nm->ia', qAkcv_aa[:, None, ...], qBkcv_bb[None, ...], gamma_k)
        # OV(aa)-CV(aa)
        A[lcva:lcva+lova, :lcva] = A[:lcva, lcva:lcva+lova].T
        # OV(aa)-OV(aa)
        int2ekov_a = np.einsum('nmi, nma, nm->ia', qAkov_aa[:, None, ...], qBkov_aa[None, ...], gamma_k)
        int2ejov_a = np.zeros_like(int2ekov_a)
        for i, a, p in zip(pscsfov_a_i, pscsfov_a_a, range(lova)):
            int2ejov_a[p, :] = np.einsum('nmp, nmp, nm->p', qAj_aa[:, None, nc+i, nc+pscsfov_a_i], qBj_aa[None, :, a, pscsfov_a_a], gamma_j)
            A[lcva+p, lcva:lcva+lova] -= np.einsum('p, p->p', fock_a_occ[nc+i, nc+pscsfov_a_i], np.eye(nv)[a, pscsfov_a_a])
            A[lcva+p, lcva:lcva+lova] += np.einsum('p, p->p', np.eye(nc+no)[nc+i, nc+pscsfov_a_i], fock_a_vir[a, pscsfov_a_a])
        A[lcva:lcva+lova, lcva:lcva+lova] += int2ekov_a - int2ejov_a
        if self.correct:
            delta_k = delta_max / (1 + (np.diag(int2ekov_a) / sigma_k) ** 4)
            A[lcva:lcva+lova, lcva:lcva+lova] += np.diag(delta_k)
        del int2ekov_a, int2ejov_a
        # OV(aa)-CO(bb)
        A[lcva:lcva+lova, lcva+lova:lcva+lova+lcob] = np.einsum('nmi, nma, nm->ia', qAkov_aa[:, None, ...], qBkco_bb[None, ...], gamma_k)
        # OV(aa)-CV(bb)
        A[lcva:lcva+lova, lcva+lova+lcob:] = np.einsum('nmi, nma, nm->ia', qAkov_aa[:, None, ...], qBkcv_bb[None, ...], gamma_k)
        # CO(bb)-CV(aa)
        A[lcva+lova:lcva+lova+lcob, :lcva] = A[:lcva, lcva+lova:lcva+lova+lcob].T
        # CO(bb)-OV(aa)
        A[lcva+lova:lcva+lova+lcob, lcva:lcva+lova] = A[lcva:lcva+lova, lcva+lova:lcva+lova+lcob].T
        # CO(bb)-CO(bb)
        int2ekco_b = np.einsum('nmi, nma, nm->ia', qAkco_bb[:, None, ...], qBkco_bb[None, ...], gamma_k)
        int2ejco_b = np.zeros_like(int2ekco_b)
        for i, a, p in zip(pscsfco_b_i, pscsfco_b_a, range(lcob)):
            int2ejco_b[p, :] = np.einsum('nmp, nmp, nm->p', qAj_bb[:, None, i, pscsfco_b_i], qBj_bb[None, :, a, pscsfco_b_a], gamma_j)
            A[lcva+lova+p, lcva+lova:lcva+lova+lcob] -= np.einsum('p, p->p', fock_b_occ[i, pscsfco_b_i], np.eye(no+nv)[a, pscsfco_b_a])
            A[lcva+lova+p, lcva+lova:lcva+lova+lcob] += np.einsum('p, p->p', np.eye(nc)[i, pscsfco_b_i], fock_b_vir[a, pscsfco_b_a])
        A[lcva+lova:lcva+lova+lcob, lcva+lova:lcva+lova+lcob] += int2ekco_b - int2ejco_b
        if self.correct:
            delta_k = delta_max / (1 + (np.diag(int2ekco_b) / sigma_k) ** 4)
            A[lcva+lova:lcva+lova+lcob, lcva+lova:lcva+lova+lcob] += np.diag(delta_k)
        del int2ekco_b, int2ejco_b
        # CO(bb)-CV(bb)
        int2ekcobcvb = np.einsum('nmi, nma, nm->ia', qAkco_bb[:, None, ...], qBkcv_bb[None, ...], gamma_k)
        int2ejcobcvb = np.zeros_like(int2ekcobcvb)
        for i, a, p in zip(pscsfco_b_i, pscsfco_b_a, range(lcob)):
            int2ejcobcvb[p, :] = np.einsum('nmp, nmp, nm->p', qAj_bb[:, None, i, pscsfcv_b_i], qBj_bb[None, :, a, no+pscsfcv_b_a], gamma_j)
            A[lcva+lova+p, lcva+lova+lcob:] -= np.einsum('p, p->p', fock_b_occ[i, pscsfcv_b_i], np.eye(no+nv)[a, no+pscsfcv_b_a])
            A[lcva+lova+p, lcva+lova+lcob:] += np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_b_i], fock_b_vir[a, no+pscsfcv_b_a])
        A[lcva+lova:lcva+lova+lcob, lcva+lova+lcob:] += int2ekcobcvb - int2ejcobcvb
        del int2ekcobcvb, int2ejcobcvb
        # CV(bb)-CV(aa)
        A[lcva+lova+lcob:, :lcva] = A[:lcva, lcva+lova+lcob:].T
        # CV(bb)-OV(aa)
        A[lcva+lova+lcob:, lcva:lcva+lova] = A[lcva:lcva+lova, lcva+lova+lcob:].T
        # CV(bb)-CO(bb)
        A[lcva+lova+lcob:, lcva+lova:lcva+lova+lcob] = A[lcva+lova:lcva+lova+lcob, lcva+lova+lcob:].T
        # CV(bb)-CV(bb)
        int2ekcv_b = np.einsum('nmi, nma, nm->ia', qAkcv_bb[:, None, ...], qBkcv_bb[None, ...], gamma_k)
        int2ejcv_b = np.zeros_like(int2ekcv_b)
        for i, a, p in zip(pscsfcv_b_i, pscsfcv_b_a, range(lcvb)):
            int2ejcv_b[p, :] = np.einsum('nmp, nmp, nm->p', qAj_bb[:, None, i, pscsfcv_b_i], qBj_bb[None, :, no+a, no+pscsfcv_b_a], gamma_j)
            A[lcva+lova+lcob+p, lcva+lova+lcob:] -= np.einsum('p, p->p', fock_b_occ[i, pscsfcv_b_i], np.eye(no+nv)[no+a, no+pscsfcv_b_a])
            A[lcva+lova+lcob+p, lcva+lova+lcob:] += np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_b_i], fock_b_vir[no+a, no+pscsfcv_b_a])
        A[lcva+lova+lcob:, lcva+lova+lcob:] += int2ekcv_b - int2ejcv_b
        if self.correct:
            delta_k = delta_max / (1 + (np.diag(int2ekcv_b) / sigma_k) ** 4)
            A[lcva+lova+lcob:, lcva+lova+lcob:] += np.diag(delta_k)
        del int2ekcv_b, int2ejcv_b

        # define some class variety, simple to use
        self.mo_coeff_a = mo_coeff_a
        self.mo_coeff_b = mo_coeff_b
        self.occidx_a = occidx_a
        self.viridx_a = viridx_a
        self.occidx_b = occidx_b
        self.viridx_b = viridx_b
        self.no = no
        self.nc = nc
        self.nv = nv
        self.pscsfcv_a_i = pscsfcv_a_i
        self.pscsfcv_a_a = pscsfcv_a_a
        self.pscsfov_a_i = pscsfov_a_i
        self.pscsfov_a_a = pscsfov_a_a
        self.pscsfco_b_i = pscsfco_b_i
        self.pscsfco_b_a = pscsfco_b_a
        self.pscsfcv_b_i = pscsfcv_b_i
        self.pscsfcv_b_a = pscsfcv_b_a

        self.e, self.v = scipy.linalg.eigh(A)
        self.e_eV = self.e[:self.nstates] * unit.ha2eV
        # logger.info(self.mf, "my stda result is \n{}".format(self.e_eV))
        self.v = self.v[:, :self.nstates]
        if self.truncate:
            self.xycv_a = self.v.T[:, :lcva]  # V_cv_a
            self.xyov_a = self.v.T[:, lcva:lcva+lova]  # V_ov_a
            self.xyco_b = self.v.T[:, lcva+lova:lcva+lova+lcob]  # V_co_b
            self.xycv_b = self.v.T[:, lcva+lova+lcob:]  # V_cv_b
        else:
            self.xycv_a = self.v.T[:, :nc*nv]  # V_cv_a
            self.xyov_a = self.v.T[:, nc*nv:(nc+no)*nv]  # V_ov_a
            self.xyco_b = self.v.T[:, (nc+no)*nv:(nc+no)*nv+nc*no]  # V_co_b
            self.xycv_b = self.v.T[:, (nc+no)*nv+nc*no:]  # V_cv_b
        os = self.osc_str()  # oscillator strength
        # before calculate rot_str, check whether mol is chiral mol
        if gto.mole.chiral_mol(mol):
            rs = self.rot_str()
        else:
            rs = np.zeros(self.nstates)
            # logger.info(self.mf, 'molecule do not have chiral')
        dS2 = self.deltaS2()
        # logger.info(self.mf, 'oscillator strength (length form) \n{}'.format(self.os))
        # logger.info(self.mf, 'rotatory strength (cgs unit) \n{}'.format(self.rs))
        # logger.info(self.mf, 'deltaS2 is \n{}'.format(dS2))
        print('my UsTDA result is')
        print(f'{"num":>4} {"energy":>8} {"osc_str":>8} {"rot_str":>8} {"deltaS2":>8}')
        for ni, ei, wli, osi, rsi, ds2i in zip(range(self.nstates), self.e_eV, unit.eVxnm/self.e_eV, os, rs, dS2):
            print(f'{ni:4d} {ei:8.4f} {wli:8.4f} {osi:8.4f} {rsi:8.4f} {ds2i:8.4f}')
        pd.DataFrame(
            np.concatenate((unit.eVxnm / np.expand_dims(self.e_eV, axis=1), np.expand_dims(os, axis=1)), axis=1)
        ).to_csv('uvspec_data.csv', index=False, header=None)
        # np.save("energy_ustda.npy", self.e)
        return self.e_eV, os, rs, self.v, pscsf_fdiag

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
        if self.truncate:
            # here can not use truncate csfs to calculate dS2,
            # so transform to norm (not truncate) format to calculate
            xycv_a = np.zeros((self.nstates, self.nc, self.nv))
            xyov_a = np.zeros((self.nstates, self.no, self.nv))
            xyco_b = np.zeros((self.nstates, self.nc, self.no))
            xycv_b = np.zeros((self.nstates, self.nc, self.nv))
            xycv_a[:, self.pscsfcv_a_i, self.pscsfcv_a_a] = self.xycv_a
            xyov_a[:, self.pscsfov_a_i, self.pscsfov_a_a] = self.xyov_a
            xyco_b[:, self.pscsfco_b_i, self.pscsfco_b_a] = self.xyco_b
            xycv_b[:, self.pscsfcv_b_i, self.pscsfcv_b_a] = self.xycv_b
        else:
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
        if self.truncate:
            dipole_mocv_a = dipole_mo_a[:, self.pscsfcv_a_i, self.pscsfcv_a_a]
            dipole_moov_a = dipole_mo_a[:, self.nc+self.pscsfov_a_i, self.pscsfov_a_a]
            dipole_moco_b = dipole_mo_b[:, self.pscsfco_b_i, self.pscsfco_b_a]
            dipole_mocv_b = dipole_mo_b[:, self.pscsfcv_b_i, self.no+self.pscsfcv_b_a]
        else:
            dipole_mocv_a = dipole_mo_a[:, :self.nc, :].reshape(3, -1)
            dipole_moov_a = dipole_mo_a[:, self.nc:, :].reshape(3, -1)
            dipole_moco_b = dipole_mo_b[:, :, :self.no].reshape(3, -1)
            dipole_mocv_b = dipole_mo_b[:, :, self.no:].reshape(3, -1)
        trans_dipcv_a = np.einsum('xi,yi->yx', dipole_mocv_a, self.xycv_a)
        trans_dipov_a = np.einsum('xi,yi->yx', dipole_moov_a, self.xyov_a)
        trans_dipco_b = np.einsum('xi,yi->yx', dipole_moco_b, self.xyco_b)
        trans_dipcv_b = np.einsum('xi,yi->yx', dipole_mocv_b, self.xycv_b)
        trans_dip = trans_dipcv_a + trans_dipov_a + trans_dipco_b + trans_dipcv_b
        f = 2. / 3. * np.einsum('s,sx,sx->s', omega, trans_dip, trans_dip)
        # np.save("osc_str_ustda.npy", f)
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
        if self.truncate:
            dip_ele_mocv_a = dip_ele_mo_a[:, self.pscsfcv_a_i, self.pscsfcv_a_a]
            dip_ele_moov_a = dip_ele_mo_a[:, self.nc+self.pscsfov_a_i, self.pscsfov_a_a]
            dip_ele_moco_b = dip_ele_mo_b[:, self.pscsfco_b_i, self.pscsfco_b_a]
            dip_ele_mocv_b = dip_ele_mo_b[:, self.pscsfcv_b_i, self.no+self.pscsfcv_b_a]
            dip_meg_mocv_a = dip_meg_mo_a[:, self.pscsfcv_a_i, self.pscsfcv_a_a]
            dip_meg_moov_a = dip_meg_mo_a[:, self.nc+self.pscsfov_a_i, self.pscsfov_a_a]
            dip_meg_moco_b = dip_meg_mo_b[:, self.pscsfco_b_i, self.pscsfco_b_a]
            dip_meg_mocv_b = dip_meg_mo_b[:, self.pscsfcv_b_i, self.no+self.pscsfcv_b_a]
        else:
            dip_ele_mocv_a = dip_ele_mo_a[:, :self.nc, :].reshape(3, -1)
            dip_ele_moov_a = dip_ele_mo_a[:, self.nc:, :].reshape(3, -1)
            dip_ele_moco_b = dip_ele_mo_b[:, :, :self.no].reshape(3, -1)
            dip_ele_mocv_b = dip_ele_mo_b[:, :, self.no:].reshape(3, -1)
            dip_meg_mocv_a = dip_meg_mo_a[:, :self.nc, :].reshape(3, -1)
            dip_meg_moov_a = dip_meg_mo_a[:, self.nc:, :].reshape(3, -1)
            dip_meg_moco_b = dip_meg_mo_b[:, :, :self.no].reshape(3, -1)
            dip_meg_mocv_b = dip_meg_mo_b[:, :, self.no:].reshape(3, -1)
        trans_ele_dipcv_a = np.einsum('xi,yi->yx', dip_ele_mocv_a, self.xycv_a)
        trans_ele_dipov_a = np.einsum('xi,yi->yx', dip_ele_moov_a, self.xyov_a)
        trans_ele_dipco_b = np.einsum('xi,yi->yx', dip_ele_moco_b, self.xyco_b)
        trans_ele_dipcv_b = np.einsum('xi,yi->yx', dip_ele_mocv_b, self.xycv_b)
        trans_ele_dip = -(trans_ele_dipcv_a + trans_ele_dipov_a + trans_ele_dipco_b + trans_ele_dipcv_b)
        trans_meg_dipcv_a = np.einsum('xi,yi->yx', dip_meg_mocv_a, self.xycv_a)
        trans_meg_dipov_a = np.einsum('xi,yi->yx', dip_meg_moov_a, self.xyov_a)
        trans_meg_dipco_b = np.einsum('xi,yi->yx', dip_meg_moco_b, self.xyco_b)
        trans_meg_dipcv_b = np.einsum('xi,yi->yx', dip_meg_mocv_b, self.xycv_b)
        trans_meg_dip = 0.5 * (trans_meg_dipcv_a + trans_meg_dipov_a + trans_meg_dipco_b + trans_meg_dipcv_b)
        # in Gaussian and ORCA, do not multiply constant
        # f = 1./unit.c * np.einsum('s,sx,sx->s', 1./omega, trans_ele_dip, trans_meg_dip)
        f = np.einsum('s,sx,sx->s', 1. / omega, trans_ele_dip, trans_meg_dip)
        f = f / unit.cgs2au  # transform atom unit to cgs unit
        # np.save("rot_str_stda.npy", f)
        return f


if __name__ == "__main__":
    mol = gto.M(
        # atom=atom.perylene,
        # atom=atom.n2,  # unit is Angstrom
        # atom=atom.ch2o,
        # atom=atom.ch2o_vacuum,
        atom=atom.ch2o_cyclohexane,
        # atom=atom.ch2o_diethylether,
        # atom=atom.ch2o_thf,
        # atom=atom.ch2s,
        # atom=atom.c2h4foh,  # unit is Angstrom
        # atom = atom.indigo,
        # atom=atom.ttm1cz,  # unit is Angstrom
        unit="A",
        # unit="B",  # https://doi.org/10.1016/j.comptc.2014.02.023 use bohr
        # basis='aug-cc-pvtz',
        # basis='sto-3g',
        basis='cc-pvdz',
        spin=1,
        charge=1,
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

    # add solvents
    t_dft0 = time.time()
    mf = dft.UKS(mol).SMD()
    # mf.with_solvent.method = 'COSMO'  # C-PCM, SS(V)PE, COSMO, IEF-PCM
    # in https://gaussian.com/scrf/ solvents entry, give different eps for different solvents
    mf.with_solvent.eps = 2.0165  # for Cyclohexane 环己烷
    # mf.with_solvent.eps = 4.2400  # for DiethylEther 乙醚
    # mf.with_solvent.eps = 7.4257  # for TetraHydroFuran 四氢呋喃

    # t_dft0 = time.time()
    # mf = dft.UKS(mol)
    # mf.init_guess = '1e'
    # mf.init_guess = 'atom'
    # mf.init_guess = 'huckel'
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    # xc = 'svwn'
    # xc = 'blyp'
    xc = 'b3lyp'
    # xc = 'pbe0'
    # xc = 'pbe38'
    # xc = '0.50*HF + 0.50*B88 + GGA_C_LYP'  # BHHLYP
    # xc = 'hf'
    mf.xc = xc
    # xc = 'bhhlyp'
    # mf.grids.level = 9
    mf.conv_check = False
    mf.kernel()
    t_dft1 = time.time()
    print("dft use {} s".format(t_dft1 - t_dft0))
    # os.environ["OMP_NUM_THREADS"] = "1"  # test one core time consuming

    ustda = UsTDA(mol, mf, nstates=10)
    ustda.info()

    # print("=" * 50)
    # t0 = time.time()
    # ustda.nstates = 12
    # ustda.truncate = 0.0
    # ustda.cas = False
    # e_eV, os, rs, v, pscsf = ustda.my_stda()
    # t1 = time.time()
    # print("ustda use {} s".format(t1-t0))

    # print("="*50)
    # t0 = time.time()
    # ustda.nstates = 12
    # ustda.truncate = 20.0
    # ustda.cas = False
    # ustda.my_stda()
    # t1 = time.time()
    # print("ustda use {} s".format(t1 - t0))

    print("="*50)
    t0 = time.time()
    ustda.nstates = 12
    ustda.truncate = 20.0
    ustda.cas = True
    e_eV, os, rs, v, pscsf = ustda.my_stda()
    t1 = time.time()
    print("ustda use {} s".format(t1 - t0))

    # import pandas as pd
    # pd.DataFrame(e_eV).to_csv(xc + 'UsTDA.csv')

    # from UTDA import UTDA
    # utda = UTDA(mol, mf)
    # print('='*50)
    # t0 = time.time()
    # utda.nstates = 12
    # # e_eV = utda.my_utda()
    # utda.pyscf_tda()
    # t1 = time.time()
    # print("utda use {} s".format(t1 - t0))
