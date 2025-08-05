#!/usr/bin/env python
import sys
sys.path.append('../')
import time
import scipy
import numpy as np
import pandas as pd
import basis_set_exchange as bse
from numba import jit
from pyscf import dft, gto, scf
from pyscf.lib import logger

from eta import eta
from utils import atom, unit
"""not update for a long time, so somewhere may not be as satisfactory as XsTDA and UsTDA"""


@jit(nopython=True)
def int2e_f(pscsf_i, pscsf_a, qAj, qBj, gamma_j):
    int2e = np.zeros_like(pscsf_i, dtype=np.float64)
    for p in range(len(pscsf_i)):
        int2e[p] = np.sum(qAj[:, :, p] * qBj[:, :, p] * gamma_j)
    return int2e


class sTDA:
    def __init__(self, mol, mf, singlet=True, nstates=5, cas=True,
                 truncate=20.0, correct=False, savedata=False):
        self.mol = mol
        self.mf = mf
        self.singlet = singlet
        self.nstates = nstates
        self.cas = cas
        self.truncate = truncate
        self.correct = correct
        self.savedata = savedata

    def info(self):
        mo_occ = self.mf.mo_occ
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        logger.info(self.mf, "no is {}".format(nocc))
        logger.info(self.mf, "nv is {}".format(nvir))
        logger.info(self.mf, "no * nv is {}".format(nocc * nvir))
        logger.info(self.mf, 'nelectron is {}'.format(self.mol.nelectron))
        logger.info(self.mf, 'natm is {}'.format(self.mol.natm))

    def gamma(self, hyb):
        R = gto.inter_distance(self.mol)  # internuclear distance array, unit is bohr
        alpha1 = 1.42
        alpha2 = 0.48
        beta1 = 0.20
        beta2 = 1.83
        alpha = alpha1 + hyb * alpha2
        beta = beta1 + hyb * beta2
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

    def kernel(self):
        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        assert (mo_coeff.dtype == np.double)
        mo_energy = self.mf.mo_energy
        mo_occ = self.mf.mo_occ
        # nao, nmo = mo_coeff.shape
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        nocc = len(occidx)
        nvir = len(viridx)

        # check XC functional hybrid proportion
        ni = self.mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, mol.spin)
        print('hyb', hyb)
        # record full A matrix diagonal element, to calculate overlap
        pscsf_fdiag = np.zeros((nocc, nvir), dtype=bool)

        # CAS process
        if self.cas:
            homo = mo_energy[nocc-1]
            lomo = mo_energy[nocc]
            deps = (1. + 0.8 * hyb) * self.truncate / unit.ha2eV
            vthr = (deps * 2 + homo) * unit.ha2eV
            othr = (lomo - deps * 2) * unit.ha2eV
            assert vthr > othr
            index = np.where((mo_energy*unit.ha2eV>othr) & (mo_energy*unit.ha2eV<vthr))[0]
            mo_occ = mo_occ[index]
            occidx = np.where(mo_occ == 2)[0]
            viridx = np.where(mo_occ == 0)[0]
            # occidx = index[np.array(mo_occ, dtype=bool)]  # according to index
            # viridx = index[~np.array(mo_occ, dtype=bool)]  # according to index
            nocc = len(occidx)
            nvir = len(viridx)
            mo_coeff = mo_coeff[:, index]

        # # Fock matrix
        # dm = self.mf.make_rdm1()
        # vhf = self.mf.get_veff(mf.mol, dm)  # Most time-consuming
        # h1e = self.mf.get_hcore()
        # fock = h1e + vhf
        fock = self.mf.get_fock()
        fock = mo_coeff.T @ fock @ mo_coeff
        delta_ij = np.eye(nocc)
        delta_ab = np.eye(nvir)

        # use sTDA method construct two-electron intergral
        gamma_j, gamma_k = self.gamma(hyb)
        # AO basis function shell(1s,2s ... 4d) range and index(index in overlap for each atom) range
        ao_slices = mol.aoslice_by_atom()
        S = mol.intor('int1e_ovlp')  # AO overlap, symmetry matrix
        eigvals, eigvecs = np.linalg.eigh(S)
        S_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_prime = S_sqrt @ mo_coeff

        # construct A matrix
        qAk = []
        qAj = []
        qBj = []
        C_occ = C_prime[:, :nocc]
        C_vir = C_prime[:, nocc:]
        for (shl0, shl1, p0, p1) in ao_slices:
            C_occ_atom = C_occ[p0:p1]
            C_vir_atom = C_vir[p0:p1]
            if self.singlet:
                qAk.append(np.einsum('mi, ma -> ia', C_occ_atom, C_vir_atom))
            qAj.append(np.einsum('mi, mj -> ij', C_occ_atom, C_occ_atom))
            qBj.append(np.einsum('ma, mb -> ab', C_vir_atom, C_vir_atom))
        if self.singlet:
            qAk = np.array(qAk)
            qBk = np.array(qAk)  # do not remove np.array(), this will make qAk and qBk use same memory (change qBk will change qAk)
        qAj = np.array(qAj)
        qBj = np.array(qBj)
        fockocc = fock[:nocc, :nocc]
        fockvir = fock[nocc:, nocc:]
        if self.truncate:
            # int2e_j[i, a, i, a] = qAj[atom, i, i] * qBj[atom, a, a]
            iaia_j = np.einsum('nmii, nmaa, nm->ia', qAj[:, None, ...], qBj[None, ...], gamma_j)
            if self.singlet:
                # int2e_k[i, a, i, a] = qAk[atom, i, a] * qBk[atom, i, a]
                iaia_k = np.einsum('nmia, nmia, nm->ia', qAk[:, None, ...], qBk[None, ...], gamma_k)
                if self.correct:
                    delta_max = 0.5 / unit.ha2eV
                    sigma_k = 0.1 / unit.ha2eV
                    delta_k = delta_max / (1 + (iaia_k / sigma_k) ** 4)
                    iaia_k += delta_k
                iaia = 2 * iaia_k - iaia_j  # A matrix main diagonal element (do not include fock)
            else:
                iaia = -iaia_j
            iaia -= np.einsum('i, a->ia', np.diag(fockocc), np.ones(nvir))  # add fock contribution
            iaia += np.einsum('i, a->ia', np.ones(nocc), np.diag(fockvir))  # add fock contribution
            del iaia_j

            # get truncation space
            pcsf = np.where(iaia * unit.ha2eV < self.truncate)
            ncsf = np.where(iaia * unit.ha2eV > self.truncate)
            iajb = np.zeros(ncsf[0].shape[0])
            for i, j in zip(pcsf[0], pcsf[1]):
                # coulomb term in non-diagonal A matrix element
                iajb_j = np.einsum('nmi, nmi, nm->i',
                                   qAj[:, None, i, ncsf[0]], qBj[None, :, j, ncsf[1]], gamma_j)
                # fock term in non-diagonal A matrix element
                iajb_f_o = np.einsum('i, i->i', fockocc[i, ncsf[0]], np.eye(nvir)[j, ncsf[1]])
                iajb_f_v = np.einsum('i, i->i', np.eye(nocc)[i, ncsf[0]], fockvir[j, ncsf[1]])
                if self.singlet:
                    # exchange term in non-diagonal A matrix element
                    iajb_k = np.einsum('nm, nmi, nm->i',
                                        qAk[:, None, i, j], qBk[None, :, ncsf[0], ncsf[1]], gamma_k)
                    iajb += (2 * iajb_k - iajb_j - iajb_f_o + iajb_f_v) ** 2 / (iaia[ncsf[0], ncsf[1]] - iaia[i, j] + 1e-10)
                else:
                    iajb += (-iajb_j - iajb_f_o + iajb_f_v) ** 2 / (iaia[ncsf[0], ncsf[1]] - iaia[i, j] + 1e-10)
            scsf = np.where(iajb > 1e-4)  # 1D array
            pscsf_i = np.concatenate((pcsf[0], ncsf[0][scsf[0]]), axis=0)
            pscsf_a = np.concatenate((pcsf[1], ncsf[1][scsf[0]]), axis=0)
            pscsf_ind = np.argsort(pscsf_i, stable=True)
            pscsf_i = pscsf_i[pscsf_ind]
            pscsf_a = pscsf_a[pscsf_ind]
            logger.info(self.mf, 'no * nv is {}'.format(len(pcsf[0]) + len(scsf[0])))
            logger.info(self.mf, '{} CSFs in pcsf'.format(len(pcsf[0])))
            logger.info(self.mf, '{} CSFs in scsf'.format(len(scsf[0])))
            logger.info(self.mf, '{} CSFs in ncsf'.format(len(ncsf[0]) - len(scsf[0])))
            del iajb, iajb_j, iajb_f_o, iajb_f_v, pcsf, scsf, ncsf
        else:
            logger.info(self.mf, 'no * nv is {}'.format(nocc * nvir))
            pscsf_i, pscsf_a = np.indices((nocc, nvir))
            pscsf_i = pscsf_i.reshape(-1)
            pscsf_a = pscsf_a.reshape(-1)
        pscsf_diag = pscsf_i*nvir + pscsf_a  # in os and rs calculation will be used to truncated
        pscsf_fdiag[pscsf_i+pscsf_fdiag.shape[0]-nocc, pscsf_a] = True  # full A matrix diagonal element
        pscsf_fdiag = pscsf_fdiag.reshape(-1)
        A = np.zeros((pscsf_i.shape[0], pscsf_a.shape[0]))
        # # line by line construct int2e (1/2)
        # int2e_k = np.zeros((pscsf_i.shape[0], pscsf_a.shape[0]))
        int2e_j = np.zeros((pscsf_i.shape[0], pscsf_a.shape[0]))
        for i, a, p in zip(pscsf_i, pscsf_a, range(len(pscsf_i))):
            # int2e_k[p, :] = np.einsum('nmp, nmp, nm->p',
            #                           qAk[:, None, i, pscsf_a], qBk[None, :, i, pscsf_a], gamma_k)
            # int2e_j[p, :] = np.einsum('nmp, nmp, nm->p',
            #                           qAj[:, None, i, pscsf_i], qBj[None, :, a, pscsf_a], gamma_j)
            # # line by line construct int2e (2/2)
            # int2e_k[p, :] = int2e_f(pscsf_i, pscsf_a, qAk[:, None, i, pscsf_a], qBk[None, :, i, pscsf_a], gamma_k)
            int2e_j[p, :] = int2e_f(pscsf_i, pscsf_a, qAj[:, None, i, pscsf_i], qBj[None, :, a, pscsf_a], gamma_j)
            A[p, :] -= np.einsum('p, p->p', fockocc[i, pscsf_i], np.eye(nvir)[a, pscsf_a])
            A[p, :] += np.einsum('p, p->p', np.eye(nocc)[i, pscsf_i], fockvir[a, pscsf_a])
        if self.singlet:
            qAk = qAk[:, pscsf_i, pscsf_a]
            qBk = qAk
            int2e_k = np.einsum('nmi, nma, nm->ia', qAk[:, None, ...], qBk[None, ...], gamma_k)
            if self.correct:
                delta_max = 0.5 / unit.ha2eV
                sigma_k = 0.1 / unit.ha2eV
                delta_k = delta_max / (1 + (np.diag(int2e_k) / sigma_k) ** 4)
                A += np.diag(delta_k)
            A += 2 * int2e_k - int2e_j
        else:
            A += -int2e_j

        # # construct int2e_j, but these code take too much memory
        # xx_i, yy_i = np.meshgrid(pscsf_i, pscsf_i, indexing='ij')  # For easy indexing
        # xx_a, yy_a = np.meshgrid(pscsf_a, pscsf_a, indexing='ij')  # For easy indexing
        # # do not use below code calculate is that below code take too much memory
        # qAj = qAj[:, xx_i, yy_i]
        # qBj = qBj[:, xx_a, yy_a]
        # int2e_j = np.einsum('nmia, nmia, nm->ia', qAj[:, None, ...], qBj[None, ...], gamma_j)
        # # This is also the case with the following code (better than upper code but still take too much memory)
        # int2e_j = np.zeros_like(int2e_k)
        # for i in range(mol.natm):
        #     for j in range(mol.natm):
        #         int2e_j += np.einsum('ia, ia->ia', qAj[i, xx_i, yy_i], qBj[j, xx_a, yy_a]) * gamma_j[i, j]
        #
        # A -= np.einsum('ia, ia->ia', fockocc[xx_i, yy_i], np.eye(nvir)[xx_a, yy_a])
        # A += np.einsum('ia, ia->ia', np.eye(nocc)[xx_i, yy_i], fockvir[xx_a, yy_a])

        # define some class variety, simple to use
        self.mo_coeff = mo_coeff
        self.occidx = occidx
        self.viridx = viridx
        self.nc = nocc
        self.nv = nvir
        self.dim = self.nc * self.nv
        self.pscsf = pscsf_diag

        # qAk = []
        # qAj = []
        # qBk = []
        # qBj = []
        # int2e_j = np.zeros((nocc, nocc, nvir, nvir))  # two-electron integral
        # int2e_k = np.zeros((nocc, nvir, nocc, nvir))
        # for (shl0,shl1,p0,p1) in ao_slices:
        #     qAk.append(np.einsum('mi, ma->ia', C_prime[p0:p1, :nocc], C_prime[p0:p1, nocc:]))  # exchange
        #     qAj.append(np.einsum('mi, mj->ij', C_prime[p0:p1, :nocc], C_prime[p0:p1, :nocc]))  # coulomb
        #     qBk.append(np.einsum('mj, mb->jb', C_prime[p0:p1, :nocc], C_prime[p0:p1, nocc:]))  # exchange
        #     qBj.append(np.einsum('ma, mb->ab', C_prime[p0:p1, nocc:], C_prime[p0:p1, nocc:]))  # coulomb
        # for i in range(mol.natm):
        #     for j in range(mol.natm):
        #         int2e_j += np.einsum('ij,ab->ijab', qAj[i], qBj[j]) * gamma_j[i, j]
        #         int2e_k += np.einsum('ia,jb->iajb', qAk[i], qBk[j]) * gamma_k[i, j]
        # int2e_j = np.einsum('ijab->iajb', int2e_j)
        # del qAk, qAj, qBk, qBj
        #
        # # Construct and diagonal A matrix
        # A = np.zeros((self.dim, self.dim))
        # # CV-CV
        # A += (
        #     (np.einsum('ij,ab -> iajb', delta_ij, fock[self.nc:, self.nc:])
        #      - np.einsum('ab,ij->iajb', delta_ab, fock[:self.nc, :self.nc])
        #      + 2 * int2e_k - int2e_j)
        # ).reshape(self.dim, self.dim)
        #
        # if self.correct:
        #     delta_max = 0.5 / unit.ha2eV
        #     sigma_k = 0.1 / unit.ha2eV
        #     iaia = np.diag(int2e_k.reshape(self.dim, self.dim))
        #     delta_k = delta_max / (1 + (iaia / sigma_k)**4)
        #     A += np.diag(delta_k)
        #
        # if self.truncate:
        #     e = np.diag(A)
        #     pcsf = e * unit.ha2eV < self.truncate
        #     inv_pcsf = ~pcsf  # scsf + ncsf  # ~ is logical NOT operator
        #     scsf = np.zeros(np.count_nonzero(inv_pcsf))
        #     scsf_cond = np.nonzero(inv_pcsf)[0]  # store scsf condition result
        #     for ev, indv in zip(e[pcsf], np.nonzero(pcsf)[0]):
        #         scsf += A[inv_pcsf, indv]**2 / (e[inv_pcsf] - ev + 1e-10)
        #     logger.info(mf, '{} CSFs in pcsf'.format(np.count_nonzero(pcsf)))
        #     pcsf[scsf_cond[scsf > 1e-4]] = True  # pcsf + scsf
        #     logger.info(mf, '{} CSFs in scsf'.format(np.count_nonzero(scsf>1e-4)))
        #     logger.info(mf, '{} CSFs in ncsf'.format(np.count_nonzero(~pcsf)))
        #     A = A[pcsf][:, pcsf]
        #     self.pscsf = pcsf

        self.e, self.v = scipy.linalg.eigh(A)
        # Note: when remove minor value, 'osc_str' and 'rot_str' truncate code have error.
        #  In my code do not consider remove
        # self.e = self.e[np.where(self.e > 0)[0]]  # avoid appear negetive value
        # self.v = self.v[:, np.where(self.e > 0)[0]]
        self.e_eV = self.e[:self.nstates] * unit.ha2eV
        self.v = self.v[:, :self.nstates]
        # logger.info(self.mf, "my stda result is \n{}".format(self.e_eV))
        if self.singlet:
            os = self.osc_str()  # oscillator strength
            # before calculate rot_str, check whether mol is chiral mol
            if gto.mole.chiral_mol(mol):
                rs = self.rot_str()
            else:
                rs = np.zeros(self.nstates)
                # logger.info(self.mf, 'molecule do not have chiral')
        else:
            os = np.zeros(self.nstates)
            rs = np.zeros(self.nstates)
        # logger.info(self.mf, 'oscillator strength (length form) \n{}'.format(self.os))
        # logger.info(self.mf, 'rotatory strength (cgs unit) \n{}'.format(self.rs))
        print('my sTDA result is')
        print(f'{"num":>4} {"energy":>8} {"wav_len":>8} {"osc_str":>8} {"rot_str":>8}')
        for ni, ei, wli, osi, rsi in zip(range(self.nstates), self.e_eV, unit.eVxnm / self.e_eV, os, rs):
            print(f'{ni:4d} {ei:8.4f} {wli:8.4f} {osi:8.4f} {rsi:8.4f}')
        if self.savedata:
            pd.DataFrame(
                np.concatenate((unit.eVxnm / np.expand_dims(self.e_eV, axis=1), np.expand_dims(os, axis=1)), axis=1)
            ).to_csv('uvspec_data.csv', index=False, header=None)
        # np.save("energy_stda.npy", self.e)
        return self.e_eV, os, rs, self.v, pscsf_fdiag


    def osc_str(self):
        # oscillator strength
        # f = 2/3 \omega_{0\nu} | \langle 0|\sum_s r_s |\nu\rangle |^2  # length form
        # f = 2/3 \omega_{0\nu}^{-1} | \langle 0|\sum_s \nabla_s |\nu\rangle |^2  # velocity form
        orbv = self.mo_coeff[:, self.viridx]
        orbo = self.mo_coeff[:, self.occidx]
        omega = self.e[:self.nstates]
        # length form oscillator strength
        dipole_ao = self.mol.intor_symmetric("int1e_r", comp=3)  # dipole moment, comp=3 is 3 axis
        dipole_mo = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo, orbv)

        # pyscf X^{+}X=1/2, but here X^{+}X=1, so to make result same, here multiple a constant
        xy = self.v.T * np.sqrt(2)  # '.T' make 'reshape' is right
        dipole_mo = dipole_mo.reshape(3, self.dim)[:, self.pscsf]
        trans_dip = np.einsum('xi,yi->yx', dipole_mo, xy)
        f = 2. / 3. * np.einsum('s,sx,sx->s', omega, trans_dip, trans_dip)
        # np.save("osc_str_stda.npy", f)

        # # velocity form oscillator strength (without truncation)
        # dipole_ao = self.mol.intor('int1e_ipovlp', comp=3, hermi=2)  # dipole moment, (nabla \|\)
        # dipole_mo = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo, orbv)
        # trans_dip = -np.einsum('xij,yij->yx', dipole_mo, xy)  # Note: dipole is (-i nabla \|\), so here have a '-'
        # f = 2. / 3. * np.einsum('s,sx,sx->s', 1. / omega, trans_dip, trans_dip)
        return f

    def rot_str(self):
        # rotatory strength, only implement velocity form
        orbv = self.mo_coeff[:, self.viridx]
        orbo = self.mo_coeff[:, self.occidx]
        omega = self.e[:self.nstates]
        # dipole_ao = self.mol.intor_symmetric("int1e_r", comp=3)  # dipole moment, comp=3 is 3 axis
        # dipole_mo = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo, orbv)
        # trans_ele_dip = np.einsum('xij,yij->yx', dipole_mo, xy)  # length form
        dip_ele_ao = self.mol.intor('int1e_ipovlp', comp=3, hermi=2)  # transition electric dipole moment
        dip_ele_mo = np.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo, orbv)
        dip_meg_ao = self.mol.intor('int1e_cg_irxp', comp=3, hermi=2)  # transition magnetic dipole moment
        dip_meg_mo = np.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo, orbv)

        # pyscf X^{+}X=1/2, but here X^{+}X=1, so to make result same, here multiple a constant
        xy = self.v.T * np.sqrt(2)  # '.T' make 'reshape' is right
        dip_ele_mo = dip_ele_mo.reshape(3, self.dim)[:, self.pscsf]
        trans_ele_dip = -np.einsum('xi,yi->yx', dip_ele_mo, xy)  # velocity form
        dip_meg_mo = dip_meg_mo.reshape(3, self.dim)[:, self.pscsf]
        trans_meg_dip = 0.5 * np.einsum('xi,yi->yx', dip_meg_mo, xy)  # remove '-' to keep same with Gaussian and ORCA
        # in Gaussian and ORCA, do not multiply constant
        # f = 1./unit.c * np.einsum('s,sx,sx->s', 1./omega, trans_ele_dip, trans_meg_dip)
        f = np.einsum('s,sx,sx->s', 1. / omega, trans_ele_dip, trans_meg_dip)
        f = f / unit.cgs2au  # transform atom unit to cgs unit
        # np.save("rot_str_stda.npy", f)
        return f

    def analyze(self):
        for i in range(self.nstates):
            print(f'Excited state {i+1} {self.e_eV[i]:12.5f} eV')
            v = self.v[:, i].reshape(self.nc, self.nv)
            for o,v in zip(* np.where(abs(v)>0.1)):
                print(f'{o+1}a -> {v+1+self.nc}a {self.v[o, v]:12.5f}')


if __name__ == "__main__":
    mol = gto.M(
        # atom=atom.perylene,
        # atom=atom.ch2s,  # unit is Angstrom
        # atom=atom.ch2o,
        atom=atom.ch2o_vacuum,
        # atom=atom.n2,  # unit is Angstrom
        # atom=atom.c2h4foh,  # unit is Angstrom
        # atom = atom.indigo,
        # atom=atom.ttm1cz,  # unit is Angstrom
        unit="A",
        # unit = "B",  # https://doi.org/10.1016/j.comptc.2014.02.023 use bohr
        # basis='aug-cc-pvtz',
        # basis='sto-3g',
        basis = "cc-pvdz",
        spin=0,
        charge=0,
        verbose=4
    )
    # path = '/home/whb/Documents/TDDFT/orcabasis/'
    # # bse.convert_formatted_basis_file(path+'tzvp.bas', path+'tzvp.nw')
    # with open(path+"ccpvdz.nw", "r") as f:
    # # bse.convert_formatted_basis_file('../orcabasis/sto-3g.bas', '../orcabasis/sto-3g.nw')
    # # with open("../orcabasis/sto-3g.nw", "r") as f:
    #     basis = f.read()
    # mol.basis = basis
    # mol.build()

    t_dft0 = time.time()
    mf = dft.RKS(mol)
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
    print("dft use {} s".format(t_dft1-t_dft0))
    # os.environ["OMP_NUM_THREADS"] = "1"  # test one core time consuming

    # truncation Threshold unit is eV
    stda = sTDA(mol, mf, singlet=True, nstates=10, cas=True, truncate=20, correct=False)
    stda.info()

    # # three kind approximation
    # # A12
    # print("=" * 50)
    # t_stda0 = time.time()
    # stda.nstates = 12
    # stda.truncate = 0  # eV, do not set to small or raise error, same with ORCA
    # stda.cas = False
    # stda.correct = False
    # e_eV, os, rs, v, pscsf = stda.kernel()
    # t_stad1 = time.time()
    # print("stda use {} s".format(t_stad1-t_stda0))

    # # A123
    # print("=" * 50)
    # t_stda0 = time.time()
    # stda.nstates = 12
    # stda.truncate = 20  # eV, do not set to small or raise error, same with ORCA
    # stda.cas = False
    # stda.correct = False
    # e_eV, os, rs, v, pscsf = stda.kernel()
    # t_stad1 = time.time()
    # print("stda use {} s".format(t_stad1 - t_stda0))
    # stda.analyze()

    # A1234
    print("=" * 50)
    t_stda0 = time.time()
    stda.nstates = 12
    stda.truncate = 20  # eV, do not set to small or raise error, same with ORCA
    stda.cas = True
    stda.correct = False
    e_eV, os, rs, v, pscsf = stda.kernel()
    t_stad1 = time.time()
    print("stda use {} s".format(t_stad1 - t_stda0))

    # print("=" * 50)
    # t_stda0 = time.time()
    # stda.nstates = 12
    # stda.truncate = 20  # eV, do not set to small or raise error, same with ORCA
    # stda.cas = True
    # stda.correct = True
    # e_eV, os, rs, v, pscsf = stda.kernel()
    # t_stad1 = time.time()
    # print("stda use {} s".format(t_stad1 - t_stda0))
