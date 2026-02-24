#!/usr/bin/env python
import os
import sys

os.environ["OMP_NUM_THREADS"] = "16"
sys.path.append('../')
sys.path.append('../../')
import time
import scipy
import numpy as np
import pandas as pd
# import basis_set_exchange as bse
# from numba import jit
from joblib import Parallel, delayed
from pyscf import dft, gto, scf, tddft, lo
from pyscf.lib import logger
from pyscf.tools import molden, cubegen

from eta import eta
from utils import unit, atom, utils

'''U and X opt file select same CSF in CVaa and CVbb'''


def accumulator_sum(generator):
    # add joblib output, when use 'return_as=generator'
    result = 0
    for value in generator:
        result += value
    return result


def iaia_f(qAk_ss, qAj_ss, qBk_tt, qBj_tt, gamma_k, gamma_j, fock_vir, n1, fock_occ, n2):
    # _ss, _tt represent spin, like qAk_aa ...
    # return iaiak is for add diagonal correction
    iaiak = np.einsum('nmia, nmia, nm->ia', qAk_ss[:, None, ...], qBk_tt[None, ...], gamma_k, optimize=True)
    iaiaj = np.einsum('nmii, nmaa, nm->ia', qAj_ss[:, None, ...], qBj_tt[None, ...], gamma_j, optimize=True)
    iaiak = iaiak.reshape(-1)
    iaiaj = iaiaj.reshape(-1)
    fock = np.einsum('i, a->ia', np.ones(n1), np.diag(fock_vir))
    fock -= np.einsum('i, a->ia', np.diag(fock_occ), np.ones(n2))
    fock = fock.reshape(-1)
    iaia = iaiak.reshape(-1) - iaiaj.reshape(-1) + fock.reshape(-1)
    return iaia, iaiak


def devide_csf_p_n(iaia, t):
    # t: t_P, self.Emax
    pcsf = np.where(iaia * unit.ha2eV <= t)
    ncsf = np.where(iaia * unit.ha2eV > t)
    return pcsf, ncsf


def devide_csf_ps(pcsf, ncsf, scsf):
    pscsfcv_a_i = np.concatenate((pcsf[0], ncsf[0][scsf[scsf < len(ncsf[0])]]), axis=0)
    pscsfcv_a_a = np.concatenate((pcsf[1], ncsf[1][scsf[scsf < len(ncsf[1])]]), axis=0)
    # pscsfcv_a_ind = np.argsort(pscsfcv_a_i, stable=True)  # numpy version: 2.2.6
    pscsfcv_a_ind = np.argsort(pscsfcv_a_i, kind="stable")
    pscsfcv_a_i = pscsfcv_a_i[pscsfcv_a_ind]
    pscsfcv_a_a = pscsfcv_a_a[pscsfcv_a_ind]
    scsf -= len(ncsf[0])
    scsf = np.delete(scsf, np.where(scsf < 0))
    return pscsfcv_a_i, pscsfcv_a_a, scsf


def intersec(nc, csfcv_a, csfcv_b):
    csfcv0 = np.array([], dtype=np.int64)  # temp name, no actually meaning
    csfcv1 = np.array([], dtype=np.int64)  # temp name, no actually meaning
    for i in range(nc):
        inda = np.where(csfcv_a[0] == i)[0]
        indb = np.where(csfcv_b[0] == i)[0]
        csfcvi = np.intersect1d(csfcv_a[1][inda], csfcv_b[1][indb])
        csfcv1 = np.concatenate((csfcv1, csfcvi))
        csfcv0 = np.concatenate((csfcv0, np.ones(csfcvi.shape[0], dtype=np.int64) * i))
    return (csfcv0, csfcv1)


def union(nc, csfcv_a, csfcv_b):
    csfcv0 = np.array([], dtype=np.int64)  # temp name, no actually meaning
    csfcv1 = np.array([], dtype=np.int64)  # temp name, no actually meaning
    for i in range(nc):
        inda = np.where(csfcv_a[0] == i)[0]
        indb = np.where(csfcv_b[0] == i)[0]
        csfcvi = np.union1d(csfcv_a[1][inda], csfcv_b[1][indb])
        csfcv1 = np.concatenate((csfcv1, csfcvi))
        csfcv0 = np.concatenate((csfcv0, np.ones(csfcvi.shape[0], dtype=np.int64) * i))
    return (csfcv0, csfcv1)


def _cvaacvaa_ndc(i, j, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2):
    # CV(aa)-CV(aa) non-diagonal correct term
    line = (
            0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
            np.einsum('i, i->i', np.eye(nc)[i, ncsf[0]], fab_b2[no + j, no + ncsf[1]], optimize=True)
            - np.einsum('i, i->i', np.eye(nc)[i, ncsf[0]], fab_a2[j, ncsf[1]], optimize=True)
    )
            + 0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                    np.einsum('i, i->i', np.eye(no + nv)[no + j, no + ncsf[1]], fij_b2[i, ncsf[0]], optimize=True)
                    - np.einsum('i, i->i', np.eye(nv)[j, ncsf[1]], fij_a2[i, ncsf[0]], optimize=True)
            )
    )
    return line


def _cvaacvbb_ndc(i, j, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2):
    # CV(aa)-CV(bb) non-diagonal correct term
    line = (
            0.5 * 1 / (2 * si) * (
            np.einsum('i, i->i', np.eye(nc)[i, ncsf[0]], fab_b2[no + j, no + ncsf[1]], optimize=True)
            - np.einsum('i, i->i', np.eye(nc)[i, ncsf[0]], fab_a2[j, ncsf[1]], optimize=True)
            + np.einsum('i, i->i', np.eye(no + nv)[no + j, no + ncsf[1]], fij_b2[i, ncsf[0]], optimize=True)
            - np.einsum('i, i->i', np.eye(nv)[j, ncsf[1]], fij_a2[i, ncsf[0]], optimize=True)
    )
    )
    return line


def _cvbbcvbb_ndc(i, j, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2):
    # CV(bb)-CV(bb) non-diagonal correct term
    line = (
            0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
            np.einsum('i, i->i', np.eye(nc)[i, ncsf[0]], fab_b2[no + j, no + ncsf[1]], optimize=True)
            - np.einsum('i, i->i', np.eye(nc)[i, ncsf[0]], fab_a2[j, ncsf[1]], optimize=True)
    )
            + 0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                    np.einsum('i, i->i', np.eye(no + nv)[no + j, no + ncsf[1]], fij_b2[i, ncsf[0]], optimize=True)
                    - np.einsum('i, i->i', np.eye(nv)[j, ncsf[1]], fij_a2[i, ncsf[0]], optimize=True)
            )
    )
    return line


# def iajb_f(i, a, qAk1, qAj1, qBk1, qBj1, qBk2,
#            n11, n12, n21, n22, fock_occ, fock_vir,
#            ncsf11, ncsf12, ncsf21, ncsf22, ncsf31, ncsf32, ncsf41, ncsf42,
#            iaia_ncsf, iaia, gamma_k, gamma_j, iajbtype, nc_c=0, no_c=0,
#            nc=None, no=None, nv=None, ncsf=None, si=None, fij_a2=None, fij_b2=None, fab_a2=None, fab_b2=None):
#     # TODO(WHB): add comment
#     eye11 = np.eye(n11)
#     eye12 = np.eye(n12)
#     eye21 = np.eye(n21)
#     eye22 = np.eye(n22)
#     iajb1 = np.einsum('nm, nmi, nm->i',
#                       qAk1[:, None, nc_c + i, no_c + a], qBk1[None, :, ncsf11, ncsf12], gamma_k, optimize=True)
#     iajb1 -= np.einsum('nmi, nmi, nm->i',
#                        qAj1[:, None, nc_c + i, ncsf11], qBj1[None, :, no_c + a, ncsf12], gamma_j, optimize=True)
#     iajb1 += np.einsum('i, i->i', eye11[nc_c + i, ncsf11], fock_vir[no_c + a, ncsf12], optimize=True)
#     iajb1 -= np.einsum('i, i->i', fock_occ[nc_c + i, ncsf11], eye12[no_c + a, ncsf12], optimize=True)
#     iajb2 = np.einsum('nm, nmi, nm->i',
#                       qAk1[:, None, nc_c + i, no_c + a], qBk1[None, :, ncsf21, ncsf22], gamma_k, optimize=True)
#     iajb2 -= np.einsum('nmi, nmi, nm->i',
#                        qAj1[:, None, nc_c + i, ncsf21], qBj1[None, :, no_c + a, ncsf22], gamma_j, optimize=True)
#     iajb2 += np.einsum('i, i->i', eye21[nc_c + i, ncsf21], fock_vir[no_c + a, ncsf22], optimize=True)
#     iajb2 -= np.einsum('i, i->i', fock_occ[nc_c + i, ncsf21], eye22[no_c + a, ncsf22], optimize=True)
#     iajb3 = np.einsum('nm, nmi, nm->i',
#                       qAk1[:, None, nc_c + i, no_c + a], qBk2[None, :, ncsf31, ncsf32], gamma_k, optimize=True)
#     iajb4 = np.einsum('nm, nmi, nm->i',
#                       qAk1[:, None, nc_c + i, no_c + a], qBk2[None, :, ncsf41, ncsf42], gamma_k, optimize=True)
#     if iajbtype == 'cva' and fij_a2 is not None:
#         iajb1 += _cvaacvaa_ndc(i, a, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
#         iajb4 += _cvaacvbb_ndc(i, a, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
#     elif iajbtype == 'cvb' and fij_a2 is not None:
#         iajb2 += _cvbbcvbb_ndc(i, a, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
#         iajb3 += _cvaacvbb_ndc(i, a, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
#     if 'a' in iajbtype:
#         iajbline = np.concatenate((iajb1, iajb2, iajb3, iajb4), axis=0)
#     elif 'b' in iajbtype:
#         iajbline = np.concatenate((iajb3, iajb4, iajb1, iajb2), axis=0)
#     else:
#         raise "iajbtype input error"
#     iajb = iajbline ** 2 / (iaia_ncsf - iaia[i, a] + 1e-10)
#     return iajb


# # debug test
# def iajb_f(i, a, qAk1, qAj1, qBk1, qBj1, qBk2,
#            n11, n12, n21, n22, fock_occ, fock_vir,
#            ncsf11, ncsf12, ncsf21, ncsf22, ncsf31, ncsf32, ncsf41, ncsf42,
#            iaia_ncsf, iaia, gamma_k, gamma_j, iajbtype, nc_c=0, no_c=0,
#            nc=None, no=None, nv=None, ncsf=None, si=None, fij_a2=None, fij_b2=None, fab_a2=None, fab_b2=None):
#     # TODO(WHB): add comment
#     eye11 = np.eye(n11)
#     eye12 = np.eye(n12)
#     eye21 = np.eye(n21)
#     eye22 = np.eye(n22)
#     iajb1_eri = np.einsum('nm, nmi, nm->i',
#                       qAk1[:, None, nc_c + i, no_c + a], qBk1[None, :, ncsf11, ncsf12], gamma_k, optimize=True)
#     iajb1_eri -= np.einsum('nmi, nmi, nm->i',
#                        qAj1[:, None, nc_c + i, ncsf11], qBj1[None, :, no_c + a, ncsf12], gamma_j, optimize=True)
#     iajb1_fock = np.einsum('i, i->i', eye11[nc_c + i, ncsf11], fock_vir[no_c + a, ncsf12], optimize=True)
#     iajb1_fock -= np.einsum('i, i->i', fock_occ[nc_c + i, ncsf11], eye12[no_c + a, ncsf12], optimize=True)
#     iajb1 = iajb1_eri + iajb1_fock
#     iajb2_eri = np.einsum('nm, nmi, nm->i',
#                       qAk1[:, None, nc_c + i, no_c + a], qBk1[None, :, ncsf21, ncsf22], gamma_k, optimize=True)
#     iajb2_eri -= np.einsum('nmi, nmi, nm->i',
#                        qAj1[:, None, nc_c + i, ncsf21], qBj1[None, :, no_c + a, ncsf22], gamma_j, optimize=True)
#     iajb2_fock = np.einsum('i, i->i', eye21[nc_c + i, ncsf21], fock_vir[no_c + a, ncsf22], optimize=True)
#     iajb2_fock -= np.einsum('i, i->i', fock_occ[nc_c + i, ncsf21], eye22[no_c + a, ncsf22], optimize=True)
#     iajb2 = iajb2_eri+iajb2_fock
#     iajb3 = np.einsum('nm, nmi, nm->i',
#                       qAk1[:, None, nc_c + i, no_c + a], qBk2[None, :, ncsf31, ncsf32], gamma_k, optimize=True)
#     iajb4 = np.einsum('nm, nmi, nm->i',
#                       qAk1[:, None, nc_c + i, no_c + a], qBk2[None, :, ncsf41, ncsf42], gamma_k, optimize=True)
#     if iajbtype == 'cva' and fij_a2 is not None:
#         iajb1 += _cvaacvaa_ndc(i, a, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
#         iajb4 += _cvaacvbb_ndc(i, a, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
#     elif iajbtype == 'cvb' and fij_a2 is not None:
#         iajb2 += _cvbbcvbb_ndc(i, a, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
#         iajb3 += _cvaacvbb_ndc(i, a, ncsf, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
#     if 'a' in iajbtype:
#         iajbline = np.concatenate((iajb1, iajb2, iajb3, iajb4), axis=0)
#         iajbfock = np.concatenate((iajb1_fock, iajb2_fock, np.zeros_like(iajb3), np.zeros_like(iajb4)), axis=0)
#         iajberi = np.concatenate((iajb1_eri, iajb2_eri, iajb3, iajb4), axis=0)
#     elif 'b' in iajbtype:
#         iajbline = np.concatenate((iajb3, iajb4, iajb1, iajb2), axis=0)
#         iajbfock = np.concatenate((np.zeros_like(iajb3), np.zeros_like(iajb4), iajb1_fock, iajb2_fock), axis=0)
#         iajberi = np.concatenate((iajb3, iajb4, iajb1_eri, iajb2_eri), axis=0)
#     else:
#         raise "iajbtype input error"
#     iajb = iajbline ** 2 / (iaia_ncsf - iaia[i, a] + 1e-10)
#     iajbfock_output = iajbfock ** 2 / (iaia_ncsf - iaia[i, a] + 1e-10)
#     index = np.where(iajb > 1e-4)
#     print('-'*50)
#     print('length of index', len(index[0]))
#     print("Fock dominant {}".format(len(np.where(np.abs(iajbfock[index]) > np.abs(iajberi[index]))[0])))
#     print("eri dominant {}".format(len(np.where(np.abs(iajbfock[index]) <= np.abs(iajberi[index]))[0])))
#     print('Fock {}'.format(iajbfock[index]))
#     print('Eri {}'.format(iajberi[index]))
#     print('numerator {}'.format((iajbline ** 2)[index]))
#     print('denominator {}'.format(iaia_ncsf[index] - iaia[i, a]))
#     return iajb, iajbfock_output


# do not add fock and delta A
def iajb_f(i, a, qAk1, qAj1, qBk1, qBj1, qBk2,
           ncsf11, ncsf12, ncsf21, ncsf22, ncsf31, ncsf32, ncsf41, ncsf42,
           iaia_ncsf, iaia, gamma_k, gamma_j, iajbtype, nc_c=0, no_c=0, iajb_path=False):
    # TODO(WHB): add comment
    # after test, exchange integral use optimize=True will be accelater while coulomb do not.
    #  It may cause by input array but here I obey my test result using optimize=True.
    iajb1 = np.einsum('nm, nmi, nm->i',
                      qAk1[:, None, nc_c + i, no_c + a], qBk1[None, :, ncsf11, ncsf12], gamma_k, optimize=iajb_path)
    iajb1 -= np.einsum('nmi, nmi, nm->i',
                       qAj1[:, None, nc_c + i, ncsf11], qBj1[None, :, no_c + a, ncsf12], gamma_j)
    iajb2 = np.einsum('nm, nmi, nm->i',
                      qAk1[:, None, nc_c + i, no_c + a], qBk1[None, :, ncsf21, ncsf22], gamma_k, optimize=iajb_path)
    iajb2 -= np.einsum('nmi, nmi, nm->i',
                       qAj1[:, None, nc_c + i, ncsf21], qBj1[None, :, no_c + a, ncsf22], gamma_j)
    iajb3 = np.einsum('nm, nmi, nm->i',
                      qAk1[:, None, nc_c + i, no_c + a], qBk2[None, :, ncsf31, ncsf32], gamma_k, optimize=iajb_path)
    iajb4 = np.einsum('nm, nmi, nm->i',
                      qAk1[:, None, nc_c + i, no_c + a], qBk2[None, :, ncsf41, ncsf42], gamma_k, optimize=iajb_path)
    if 'a' in iajbtype:
        iajbline = np.concatenate((iajb1, iajb2, iajb3, iajb4), axis=0)
    elif 'b' in iajbtype:
        iajbline = np.concatenate((iajb3, iajb4, iajb1, iajb2), axis=0)
    else:
        raise "iajbtype input error"
    iajb = iajbline ** 2 / (iaia_ncsf - iaia[i, a] + 1e-10)
    return iajb


def cAcvacva(i, a, c, pscsfcv_a_i, pscsfcv_a_a, qAk_aa, qBk_aa, qAj_aa, qBj_aa, gamma_k, gamma_j,
             fock_a_occ, fock_a_vir, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2,
             correct, delta_max, sigma_k):
    # this function contract cv(aa)-cv(aa) block of A matrix
    A = np.einsum('nm, nmp, nm->p',
                  qAk_aa[:, None, i, a], qBk_aa[None, :, pscsfcv_a_i, pscsfcv_a_a], gamma_k, optimize=True)
    if correct:
        A[c] += delta_max / (1 + (A[c] / sigma_k) ** 4)
    A -= np.einsum('nmp, nmp, nm->p',
                   qAj_aa[:, None, i, pscsfcv_a_i], qBj_aa[None, :, a, pscsfcv_a_a], gamma_j)
    A -= np.einsum('p, p->p', fock_a_occ[i, pscsfcv_a_i], np.eye(nv)[a, pscsfcv_a_a], optimize=True)
    A += np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_a_i], fock_a_vir[a, pscsfcv_a_a], optimize=True)
    if fij_a2 is not None:
        A += (
                0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_a_i], fab_b2[no + a, no + pscsfcv_a_a], optimize=True)
                - np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_a_i], fab_a2[a, pscsfcv_a_a], optimize=True)
        )
                + 0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                        np.einsum('p, p->p', np.eye(no + nv)[no + a, no + pscsfcv_a_a], fij_b2[i, pscsfcv_a_i], optimize=True)
                        - np.einsum('p, p->p', np.eye(nv)[a, pscsfcv_a_a], fij_a2[i, pscsfcv_a_i], optimize=True)
                )
        )
    return A


def cAcvacvb(i, a, pscsfcv_b_i, pscsfcv_b_a, qAk_aa, qBk_bb, gamma_k,
             nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2):
    A = np.einsum('nm, nmp, nm->p', qAk_aa[:, None, i, a],
                  qBk_bb[None, :, pscsfcv_b_i, no + pscsfcv_b_a], gamma_k, optimize=True)
    if fij_a2 is not None:
        A += (0.5 * 1 / (2 * si) * (
                np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_b_i], fab_b2[no + a, no + pscsfcv_b_a], optimize=True)
                - np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_b_i], fab_a2[a, pscsfcv_b_a], optimize=True)
                + np.einsum('p, p->p', np.eye(no + nv)[no + a, no + pscsfcv_b_a], fij_b2[i, pscsfcv_b_i], optimize=True)
                - np.einsum('p, p->p', np.eye(nv)[a, pscsfcv_b_a], fij_a2[i, pscsfcv_b_i], optimize=True)
        ))
    return A


def cAcvbcvb(i, a, c, pscsfcv_b_i, pscsfcv_b_a, qAk_bb, qBk_bb, qAj_bb, qBj_bb, gamma_k, gamma_j,
             fock_b_occ, fock_b_vir, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2,
             correct, delta_max, sigma_k):
    A = np.einsum('nm, nmp, nm->p',
                  qAk_bb[:, None, i, no + a], qBk_bb[None, :, pscsfcv_b_i, no + pscsfcv_b_a], gamma_k, optimize=True)
    if correct:
        A[c] += delta_max / (1 + (A[c] / sigma_k) ** 4)
    A -= np.einsum('nmp, nmp, nm->p',
                   qAj_bb[:, None, i, pscsfcv_b_i], qBj_bb[None, :, no + a, no + pscsfcv_b_a], gamma_j)
    A -= np.einsum('p, p->p', fock_b_occ[i, pscsfcv_b_i], np.eye(no + nv)[no + a, no + pscsfcv_b_a], optimize=True)
    A += np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_b_i], fock_b_vir[no + a, no + pscsfcv_b_a], optimize=True)
    if fij_a2 is not None:
        A += (
                0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_b_i], fab_b2[no + a, no + pscsfcv_b_a], optimize=True)
                - np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_b_i], fab_a2[a, pscsfcv_b_a], optimize=True)
        )
                + 0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                        np.einsum('p, p->p', np.eye(no + nv)[no + a, no + pscsfcv_b_a], fij_b2[i, pscsfcv_b_i], optimize=True)
                        - np.einsum('p, p->p', np.eye(nv)[a, pscsfcv_b_a], fij_a2[i, pscsfcv_b_i], optimize=True)
                )
        )
    return A


def constractA(pscsfcv_a_i, pscsfcv_a_a, pscsfcv_b_i, pscsfcv_b_a, nc, no, nv, gamma_k, gamma_j,
               qAk_aa, qBk_aa, qAj_aa, qBj_aa, qAk_bb, qBk_bb, qAj_bb, qBj_bb,
               fock_a_occ, fock_a_vir, fock_b_occ, fock_b_vir, si, fij_a2, fij_b2, fab_a2, fab_b2, njob,
               correct, delta_max, sigma_k):
    Acvacva = Parallel(n_jobs=njob, backend='threading')(
        delayed(cAcvacva)(
            i, a, ind, pscsfcv_a_i, pscsfcv_a_a, qAk_aa, qBk_aa, qAj_aa, qBj_aa, gamma_k, gamma_j,
            fock_a_occ, fock_a_vir, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2, correct, delta_max, sigma_k
        ) for (ind, i), a in zip(enumerate(pscsfcv_a_i), pscsfcv_a_a)
    )
    Acvacvb = Parallel(n_jobs=njob, backend='threading')(
        delayed(cAcvacvb)(
            i, a, pscsfcv_b_i, pscsfcv_b_a, qAk_aa, qBk_bb, gamma_k,
            nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2,
        ) for i, a in zip(pscsfcv_a_i, pscsfcv_a_a)
    )
    Acvbcvb = Parallel(n_jobs=njob, backend='threading')(
        delayed(cAcvbcvb)(
            i, a, ind, pscsfcv_b_i, pscsfcv_b_a, qAk_bb, qBk_bb, qAj_bb, qBj_bb, gamma_k, gamma_j,
            fock_b_occ, fock_b_vir, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2, correct, delta_max, sigma_k
        ) for (ind, i), a in zip(enumerate(pscsfcv_b_i), pscsfcv_b_a)
    )
    return Acvacva, Acvacvb, Acvbcvb


class OSsTDA:
    def __init__(self, mol, mf, spinadapt=True, Emax=10.0, tp=1e-4, cas=True, nstates=10, union=True, correct=False,
                 paramtype='os', savedata=False, readinfo=False):
        self.mol = mol
        self.mf = mf
        self.nstates = nstates
        self.cas = cas
        self.spinadapt = spinadapt
        self.Emax = Emax
        self.tp = tp
        self.union = union
        self.correct = correct
        self.paramtype = paramtype
        self.savedata = savedata
        self.readinfo = readinfo
        self.njob = int(os.environ.get("OMP_NUM_THREADS"))
        # self.njob = 1

    def info(self, mo_occ=None, mo_energy=None):
        if mo_occ is None: mo_occ = self.mf.mo_occ
        if mo_energy is None: mo_energy = self.mf.mo_energy
        if self.spinadapt:
            occidx_a = np.where(mo_occ >= 1)[0]
            viridx_a = np.where(mo_occ == 0)[0]
            occidx_b = np.where(mo_occ >= 2)[0]
            viridx_b = np.where(mo_occ != 2)[0]
        else:
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
        print("nc is {}".format(nc))
        print("nv is {}".format(nv))
        print("no is {}".format(no))
        print("A matrix dim is {}".format((nc + no) * nv + nc * (no + nv)))
        print('nelectron is {}'.format(self.mol.nelectron))
        print('natm is {}'.format(self.mol.natm))
        print('=' * 50)
        if self.spinadapt:
            print('num.orb    mo_energy     mo_occ')
            for me, o in zip(enumerate(mo_energy), mo_occ):
                ind, moei = me
                print(f'{ind + 1:5d}    {moei:10.4f}    {o:8.3f}')
        else:
            print('num.orb    mo_energy_a   mo_energy_b    mo_occ_a    mo_occ_b')
            for me, o in zip(enumerate(mo_energy.T), mo_occ.T):
                ind, moei = me
                print(f'{ind + 1:5d}    {moei[0]:10.4f}    {moei[1]:10.4f}    {o[0]:8.3f}    {o[1]:8.3f}')

    def gamma(self, hyb):
        R = gto.inter_distance(self.mol)  # internuclear distance array, unit is bohr
        eta_ele = []  # Note: eta's unit is eV
        for e in self.mol.elements:
            # # eta_ele unit is eV, transform eV to hartree is fine.
            # # inside unit do not need transformation,
            # # or transform inside unit and do not need to transform eV to hartree
            # # eta_ele.append(2 * eta[e] * unit.bohr / unit.ha2eV)  # why do not transform Angstrom to Bohr
            # refer to ORCA origin code: https://github.com/grimme-lab/std2/blob/master/stda.f
            eta_ele.append(2 * eta[e] / unit.ha2eV)
        eta_ele = (np.array(eta_ele)[:, np.newaxis] + np.array(eta_ele)[np.newaxis, :]) / 2
        if self.paramtype == 'cs':
            beta1 = 0.20
            beta2 = 1.83
            beta = beta1 + hyb * beta2
            gj = (1. / (R ** beta + (hyb * eta_ele) ** (-beta))) ** (1. / beta)
        elif self.paramtype == "os":
            beta = hyb + 0.3
            gj = (1. / (R ** beta + (1.4 * hyb * eta_ele) ** (-beta))) ** (1. / beta)
        else:
            raise ValueError
        alpha1 = 1.42
        alpha2 = 0.48
        alpha = alpha1 + hyb * alpha2
        gk = (1. / (R ** alpha + eta_ele ** (-alpha))) ** (1. / alpha)
        return gj, gk

    # def wrapper_iajb_cva(self, i, a):
    #     return iajb_f(i, a, self.qAk_aa, self.qAj_aa, self.qBk_aa, self.qBj_aa, self.qBk_bb,
    #                 self.ncsfcv_a0, self.ncsfcv_a1, self.ncsfov_a0, self.ncsfov_a1, self.ncsfco_b0, self.ncsfco_b1, self.ncsfcv_b0, self.ncsfcv_b1,
    #                 self.iaia_ncsf, self.iaiacv_a, self.gamma_k, self.gamma_j, self.cva, self.zero, self.zero,)
    #
    # def wrapper_iajb_ova(self, i, a):
    #     return iajb_f(i, a, self.qAk_aa, self.qAj_aa, self.qBk_aa, self.qBj_aa, self.qBk_bb,
    #                 self.ncsfcv_a0, self.ncsfcv_a1, self.ncsfov_a0, self.ncsfov_a1, self.ncsfco_b0, self.ncsfco_b1, self.ncsfcv_b0, self.ncsfcv_b1,
    #                 self.iaia_ncsf, self.iaiaov_a, self.gamma_k, self.gamma_j, self.ova, self.nc, self.zero,)
    #
    # def wrapper_iajb_cob(self, i, a):
    #     return iajb_f(i, a, self.qAk_bb, self.qAj_bb, self.qBk_bb, self.qBj_bb, self.qBk_aa,
    #                 self.ncsfco_b0, self.ncsfco_b1, self.ncsfcv_b0, self.ncsfcv_b1, self.ncsfcv_a0, self.ncsfcv_a1, self.ncsfov_a0, self.ncsfov_a1,
    #                 self.iaia_ncsf, self.iaiaco_b, self.gamma_k, self.gamma_j, self.cob, self.zero, self.zero,)
    #
    # def wrapper_iajb_cvb(self, i, a):
    #     return iajb_f(i, a, self.qAk_bb, self.qAj_bb, self.qBk_bb, self.qBj_bb, self.qBk_aa,
    #                 self.ncsfco_b0, self.ncsfco_b1, self.ncsfcv_b0, self.ncsfcv_b1, self.ncsfcv_a0, self.ncsfcv_a1, self.ncsfov_a0, self.ncsfov_a1,
    #                 self.iaia_ncsf, self.iaiacv_b, self.gamma_k, self.gamma_j, self.cva, self.zero, self.no,)

    def kernel(self, mo_occ=None, mo_coeff=None, mo_energy=None, h1e=None, dm=None, vhf=None, hyb=None):
        # Section: prepare for compute
        t_all_0 = time.time()  # record kernel begin time
        if mo_occ is None: mo_occ = self.mf.mo_occ
        if mo_coeff is None: mo_coeff = self.mf.mo_coeff
        if mo_energy is None: mo_energy = self.mf.mo_energy
        t0 = time.time()
        if h1e is None: h1e = self.mf.get_hcore()
        if dm is None: dm = self.mf.make_rdm1()
        if vhf is None: vhf = self.mf.get_veff(self.mol, dm)
        # read vhf do not add v_solvent because when save vhf add v_solvent
        if getattr(self.mf, 'with_solvent', None) is not None:
            vhf += vhf.v_solvent
        t1 = time.time()
        if self.spinadapt:
            mo_coeff = np.array((mo_coeff, mo_coeff))
            mo_occ_temp = np.zeros((2, len(mo_occ)))
            mo_occ_temp[0, np.where(mo_occ>=1)[0]] = 1
            mo_occ_temp[1, np.where(mo_occ==2)[0]] = 1
            mo_occ = mo_occ_temp
            del mo_occ_temp
        else:  # sU-TDA
            pass

            # # test ch2o b3lyp/sto-3g
            # orca2pyscf_index = [0, 1, 3, 4, 2, 5, 6, 8, 9, 7, 10, 11]
            # mo_coeff[0] = pd.read_csv('mo_coeff_a.csv', sep='[,\s]+', header=None, engine='python').to_numpy()[orca2pyscf_index]
            # mo_coeff[1] = pd.read_csv('mo_coeff_b.csv', sep='[,\s]+', header=None, engine='python').to_numpy()[orca2pyscf_index]
        assert mo_coeff.dtype == np.float64
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

        # Section: compute Fock matrix
        t2 = time.time()
        # if self.readinfo:
        #     assert h1e is not None
        #     assert dm is not None
        #     assert vhf is not None
        #     assert hyb is not None
        # else:
        #     h1e = self.mf.get_hcore()
        #     dm = self.mf.make_rdm1()
        #     vhf = self.mf.get_veff(self.mol, dm)
        #     if getattr(self.mf, 'with_solvent', None) is not None:
        #         vhf += vhf.v_solvent
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fock_a = mo_coeff_a.T @ focka @ mo_coeff_a
        fock_b = mo_coeff_b.T @ fockb @ mo_coeff_b
        t3 = time.time()
        # fock = self.mf.get_fock()   # teacher mentioned the method of constructing the Fock matrix.

        if self.spinadapt:
            mo_energy = np.array((np.sort(np.diag(fock_a)), np.sort(np.diag(fock_b))))
        else:
            pass
        # mo_energy = np.array((np.sort(np.diag(fock_a)), np.sort(np.diag(fock_b))))

        nc = min(nocc_a, nocc_b)
        no = abs(nocc_a - nocc_b)
        nv = min(nvir_a, nvir_b)

        # check XC functional hybrid proportion
        if not self.readinfo:
            ni = self.mf._numint
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.mol.spin)
            print("=" * 50)
            print('omega is {}, alpha is {}, hyb is {}'.format(omega, alpha, hyb))
        # # record full A matrix diagonal element, to calculate overlap
        # pscsf_fdiag_a = np.zeros((nocc_a, nvir_a), dtype=bool)
        # pscsf_fdiag_b = np.zeros((nocc_b, nvir_b), dtype=bool)

        # CAS process
        if self.cas:
            delta_mo_occ = mo_occ[0] - mo_occ[1]
            upeomo = np.nonzero(delta_mo_occ)[0]  # un-paired electron occpied molecular orbital
            somo_high = mo_energy[0, upeomo[-1]]
            somo_low = mo_energy[0, upeomo[0]]
            sumo_high = mo_energy[1, upeomo[-1]]
            sumo_low = mo_energy[1, upeomo[0]]
            deps = (1. + 0.8 * hyb) * self.Emax / unit.ha2eV
            vthr_a = (deps * 2 + somo_high)
            othr_a = (somo_low - deps * 2)
            vthr_b = (deps * 2 + sumo_high)
            othr_b = (sumo_low - deps * 2)
            assert (vthr_a > othr_a) and (vthr_b > othr_b)
            nc_a = len(np.nonzero((mo_energy[0] > othr_a) & (mo_energy[0] < somo_low))[0])
            nc_b = len(np.nonzero((mo_energy[1] > othr_b) & (mo_energy[1] < sumo_low))[0])
            nc = max(nc_a, nc_b)
            nv_a = len(np.nonzero((mo_energy[0] < vthr_a) & (mo_energy[0] > somo_high))[0])
            nv_b = len(np.nonzero((mo_energy[1] < vthr_b) & (mo_energy[1] > sumo_high))[0])
            nv = max(nv_a, nv_b)
            act_orb = (upeomo[0] - nc, upeomo[-1] + nv)  # activate space range, activate orbital
            mo_occ_a = mo_occ[0, act_orb[0]:act_orb[1] + 1]
            mo_occ_b = mo_occ[1, act_orb[0]:act_orb[1] + 1]
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
            print('before additional selection')
            print('occ. MO cut-off (eV) (alpha, beta): {} {}'.format(othr_a * unit.ha2eV, othr_b * unit.ha2eV))
            print('vir. MO cut-off (eV) (alpha, beta): {} {}'.format(vthr_a * unit.ha2eV, vthr_b * unit.ha2eV))
            print('occ. MOs (alpha, beta): {} {}'.format(nc_a + no, nc_b))
            print('vir. MOs (alpha, beta): {} {}'.format(nv_a, nv_b + no))
            print('The number of additional alpha orbitals selected: {}'.format(nv_b - nv_a))
            print('The number of additional beta orbitals selected: {}'.format(nc_a - nc_b))
            print('after additional selection')
            print('occ. MO cut-off (eV) (alpha, beta): {} {}'.format(othr_a * unit.ha2eV, mo_energy[1, act_orb[0]] * unit.ha2eV))
            print('vir. MO cut-off (eV) (alpha, beta): {} {}'.format(mo_energy[0, act_orb[1]] * unit.ha2eV, vthr_b * unit.ha2eV))
            print('occ. MOs (alpha, beta): {} {}'.format(nocc_a, nocc_b))
            print('vir. MOs (alpha, beta): {} {}'.format(nvir_a, nvir_b))

            # # orca choose active space
            # delta_mo_occ = mo_occ[0] - mo_occ[1]
            # upeomo = np.nonzero(delta_mo_occ)[0]  # un-paired electron occpied molecular orbital
            # somo_high = mo_energy[0, upeomo[-1]]
            # lomo = mo_energy[0, upeomo[-1]+1]
            # deps = (1. + 0.8 * hyb) * self.Emax / unit.ha2eV
            # vthr = (deps * 2 + somo_high) * unit.ha2eV
            # othr = (lomo - deps * 2) * unit.ha2eV
            # assert vthr > othr
            # index_a = np.where((mo_energy[0] * unit.ha2eV > othr) & (mo_energy[0] * unit.ha2eV < vthr))[0]
            # mo_occ_a = mo_occ[0, index_a]
            # mo_occ_b = mo_occ[1, index_a]
            # occidx_a = np.where(mo_occ_a > 0.999)[0]
            # viridx_a = np.where(mo_occ_a < 0.001)[0]
            # occidx_b = np.where(mo_occ_b > 0.999)[0]
            # viridx_b = np.where(mo_occ_b < 0.001)[0]
            # nocc_a = len(occidx_a)
            # nvir_a = len(viridx_a)
            # nocc_b = len(occidx_b)
            # nvir_b = len(viridx_b)
            # nc = min(nocc_a, nocc_b)
            # nv = min(nvir_a, nvir_b)
            # mo_coeff_a = mo_coeff_a[:, index_a]
            # mo_coeff_b = mo_coeff_b[:, index_a]
        if self.cas:
            self.frozen_nc = act_orb[0]
            # self.frozen_nc = index_a[0]-1  # orca choose active space
        else:
            self.frozen_nc = 0

        t4 = time.time()
        fock_a = mo_coeff_a.T @ focka @ mo_coeff_a
        fock_b = mo_coeff_b.T @ fockb @ mo_coeff_b
        # # test use orbital energy calculate A matrix
        # fock_a = np.diag(mo_energy[0, index_a])
        # fock_b = np.diag(mo_energy[1, index_a])
        if self.spinadapt:
            pterrorFock_a = np.argwhere(np.abs(np.triu(fock_a, k=1)) > 1e-2)
            pterrorFock_b = np.argwhere(np.abs(np.triu(fock_b, k=1)) > 1e-2)
            print('number of upper triangle abs(Fock_a) bigger than 1e-2 element : {}'.format(len(pterrorFock_a)))
            print('number of upper triangle abs(Fock_b) bigger than 1e-2 element : {}'.format(len(pterrorFock_b)))
            pterrorFock_a = np.argwhere(np.abs(np.triu(fock_a, k=1)) > 1e-3)
            pterrorFock_b = np.argwhere(np.abs(np.triu(fock_b, k=1)) > 1e-3)
            print('number of upper triangle abs(Fock_a) bigger than 1e-3 element : {}'.format(len(pterrorFock_a)))
            print('number of upper triangle abs(Fock_b) bigger than 1e-3 element : {}'.format(len(pterrorFock_b)))
            del pterrorFock_a, pterrorFock_b
        # # debug test
        # print('num.orb    fock_diag_a   fock_diag_b    mo_occ_a    mo_occ_b')
        # i = 0
        # for moea, moeb, oa, ob in zip(np.sort(np.diag(fock_a)), np.sort(np.diag(fock_b)), mo_occ_a, mo_occ_b):
        #     print(f'{i + 1:5d}    {moea:10.4f}    {moeb:10.4f}    {oa:8.3f}    {ob:8.3f}')
        #     i += 1

        if self.spinadapt:
            # spin adapted correct term use ROHF fock
            hf = scf.ROHF(self.mol)
            veff = hf.get_veff(self.mol, dm)
            # hf.kernel()
            focka2 = mo_coeff_a.T @ (h1e + veff[0]) @ mo_coeff_a
            fockb2 = mo_coeff_b.T @ (h1e + veff[1]) @ mo_coeff_b
            fab_a2 = focka2[nocc_a:, nocc_a:]
            fab_b2 = fockb2[nocc_b:, nocc_b:]
            fij_a2 = focka2[:nocc_a, :nocc_a]
            fij_b2 = fockb2[:nocc_b, :nocc_b]
        else:
            hf = None
            veff = None
            focka2 = None
            fockb2 = None
            fab_a2 = None
            fab_b2 = None
            fij_a2 = None
            fij_b2 = None
        t5 = time.time()
        t_getfock = t5 - t4 + t3 - t2 + t1 - t0

        # use sTDA method construct two-electron intergral
        gamma_j, gamma_k = self.gamma(hyb)
        # AO basis function shell(1s,2s ... 4d) range and index(index in overlap for each atom) range
        ao_slices = self.mol.aoslice_by_atom()
        S = self.mol.intor('int1e_ovlp')  # AO overlap, symmetry matrix

        # # use orca overlap, test ch2o b3lyp/sto-3g
        # orca2pyscf_index = [0, 1, 3, 4, 2, 5, 6, 8, 9, 7, 10, 11]
        # S = pd.read_csv('overlap.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
        # S = S[orca2pyscf_index, :][:, orca2pyscf_index]

        eigvals, eigvecs = np.linalg.eigh(S)
        S_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_prime_a = S_sqrt @ mo_coeff_a
        C_prime_b = S_sqrt @ mo_coeff_b

        # Section: compute two electron integral and XC integral
        # use sTDA method construct two-electron intergral
        qAk_aa = []
        qAj_aa = []
        qAk_bb = []
        qAj_bb = []
        qBj_aa = []
        qBj_bb = []
        for (shl0, shl1, p0, p1) in ao_slices:
            # exchange
            qAk_aa.append(np.einsum('mi, ma->ia', C_prime_a[p0:p1, :nocc_a], C_prime_a[p0:p1, nocc_a:]))
            # coulomb
            qAj_aa.append(np.einsum('mi, mj->ij', C_prime_a[p0:p1, :nocc_a], C_prime_a[p0:p1, :nocc_a]))
            # exchange
            qAk_bb.append(np.einsum('mi, ma->ia', C_prime_b[p0:p1, :nocc_b], C_prime_b[p0:p1, nocc_b:]))
            # coulomb
            qAj_bb.append(np.einsum('mi, mj->ij', C_prime_b[p0:p1, :nocc_b], C_prime_b[p0:p1, :nocc_b]))
            # coulomb
            qBj_aa.append(np.einsum('ma, mb->ab', C_prime_a[p0:p1, nocc_a:], C_prime_a[p0:p1, nocc_a:]))
            # coulomb
            qBj_bb.append(np.einsum('ma, mb->ab', C_prime_b[p0:p1, nocc_b:], C_prime_b[p0:p1, nocc_b:]))
        qAk_aa = np.array(qAk_aa)
        qBk_aa = np.array(qAk_aa)  # for easy to understand and easy to compare with formula
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
        del S_sqrt, S, eigvals, eigvecs, C_prime_a, C_prime_b, mo_energy, mo_occ, mo_coeff, \
            focka, fockb, fock_a, fock_b, p0, p1, shl0, shl1, ao_slices, nocc_a, nocc_b, nvir_a, nvir_b, \
            h1e, dm, vhf, focka2, fockb2, veff, hf, delta_mo_occ, act_orb
        # until this line, use optimize=True in np.einsum, max memory usage is 18 Gb
        si = 0.5 * self.mol.spin  # coefficient in spin adapted correct term
        delta_max = 0  # correct term
        sigma_k = 0  # correct term
        if self.Emax:
            # select P-CSF
            t0 = time.time()
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
            if self.spinadapt:
                # CV(aa)-CV(aa) add spin adapted correct term
                iaiacv_a += (
                        0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                        np.einsum('i, a->ia', np.ones(nc), np.diag(fab_b2[no:, no:]))
                        - np.einsum('i, a->ia', np.ones(nc), np.diag(fab_a2))
                )
                        + 0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                                np.einsum('a, i->ia', np.ones(nv), np.diag(fij_b2))
                                - np.einsum('a, i->ia', np.ones(nv), np.diag(fij_a2[:nc, :nc]))
                        )
                ).reshape(-1)
                # CV(bb)-CV(bb) add spin adapted correct term
                iaiacv_b += (
                        0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                        np.einsum('i, a->ia', np.ones(nc), np.diag(fab_b2[no:, no:]))
                        - np.einsum('i, a->ia', np.ones(nc), np.diag(fab_a2))
                )
                        + 0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                                np.einsum('a, i->ia', np.ones(nv), np.diag(fij_b2))
                                - np.einsum('a, i->ia', np.ones(nv), np.diag(fij_a2[:nc, :nc]))
                        )
                ).reshape(-1)
            # concatenate each block diagonal element
            iaiak = np.concatenate((iaiakcv_a, iaiakov_a, iaiakco_b, iaiakcv_b), axis=0)
            iaia = np.concatenate((iaiacv_a, iaiaov_a, iaiaco_b, iaiacv_b), axis=0)
            if self.correct:
                delta_max = 0.5 / unit.ha2eV
                sigma_k = 0.1 / unit.ha2eV
                delta_k = delta_max / (1 + (iaiak / sigma_k) ** 4)
                iaia += delta_k
            del iaiacv_a, iaiaov_a, iaiaco_b, iaiacv_b, iaiakcv_a, iaiakov_a, iaiakco_b, iaiakcv_b, iaiak
            iaiacv_a = iaia[:nc * nv].reshape(nc, nv)
            iaiaov_a = iaia[nc * nv:(nc + no) * nv].reshape(no, nv)
            iaiaco_b = iaia[(nc + no) * nv:(nc + no) * nv + nc * no].reshape(nc, no)
            iaiacv_b = iaia[(nc + no) * nv + nc * no:].reshape(nc, nv)
            pcsfcv_a, ncsfcv_a = devide_csf_p_n(iaiacv_a, self.Emax)
            pcsfov_a, ncsfov_a = devide_csf_p_n(iaiaov_a, self.Emax)
            pcsfco_b, ncsfco_b = devide_csf_p_n(iaiaco_b, self.Emax)
            pcsfcv_b, ncsfcv_b = devide_csf_p_n(iaiacv_b, self.Emax)
            print("before union, {} PCSFs in CV(aa)".format(len(pcsfcv_a[0])))
            print("before union, {} PCSFs in CV(bb)".format(len(pcsfcv_b[0])))
            # union is for transfer spin orbital basis to spin tensor basis
            if self.union:
                pcsfcv_a = pcsfcv_b = union(nc, pcsfcv_a, pcsfcv_b)
                ncsfcv_a = ncsfcv_b = intersec(nc, ncsfcv_a, ncsfcv_b)
            t1 = time.time()
            t_pcsf = t1 - t0

            # select S-CSF
            t0 = time.time()
            iajb = np.zeros(len(ncsfcv_a[0]) + len(ncsfov_a[0]) + len(ncsfco_b[0]) + len(ncsfcv_b[0]))
            iaia_ncsf = np.concatenate((iaiacv_a[ncsfcv_a[0], ncsfcv_a[1]], iaiaov_a[ncsfov_a[0], ncsfov_a[1]],
                                        iaiaco_b[ncsfco_b[0], ncsfco_b[1]], iaiacv_b[ncsfcv_b[0], ncsfcv_b[1]]), axis=0)
            # # do not add fock and delta A
            # # monitor = utils.MemoryMonitor()
            # self.qAk_aa = qAk_aa
            # self.qAj_aa = qAj_aa
            # self.qAk_bb = qAk_bb
            # self.qAj_bb = qAj_bb
            # self.qBk_aa = qBk_aa
            # self.qBj_aa = qBj_aa
            # self.qBk_bb = qBk_bb
            # self.qBj_bb = qBj_bb
            # self.ncsfcv_a0 = ncsfcv_a[0]
            # self.ncsfcv_a1 = ncsfcv_a[1]
            # self.ncsfov_a0 = nc + ncsfov_a[0]
            # self.ncsfov_a1 = ncsfov_a[1]
            # self.ncsfco_b0 = ncsfco_b[0]
            # self.ncsfco_b1 = ncsfco_b[1]
            # self.ncsfcv_b0 = ncsfcv_b[0]
            # self.ncsfcv_b1 = no + ncsfcv_b[1]
            # self.iaia_ncsf = iaia_ncsf
            # self.iaiacv_a = iaiacv_a
            # self.iaiaov_a = iaiaov_a
            # self.iaiaco_b = iaiaco_b
            # self.iaiacv_b = iaiacv_b
            # self.gamma_k = gamma_k
            # self.gamma_j = gamma_j
            # self.cva = 'cva'
            # self.ova = 'ova'
            # self.cob = 'cob'
            # self.cvb = 'cvb'
            # self.zero = 0
            # self.nc = nc
            # self.no = no
            # iajb_sjob = Parallel(n_jobs=self.njob, return_as="generator")(
            #     delayed(self.wrapper_iajb_cva)(i, a) for i, a in zip(pcsfcv_a[0], pcsfcv_a[1])
            # )
            # iajb += accumulator_sum(iajb_sjob)
            # iajb_sjob = Parallel(n_jobs=self.njob, return_as="generator")(
            #     delayed(self.wrapper_iajb_ova)(i, a) for i, a in zip(pcsfov_a[0], pcsfov_a[1])
            # )
            # iajb += accumulator_sum(iajb_sjob)
            # iajb_sjob = Parallel(n_jobs=self.njob, return_as="generator")(
            #     delayed(self.wrapper_iajb_cob)(i, a) for i, a in zip(pcsfco_b[0], pcsfco_b[1])
            # )
            # iajb += accumulator_sum(iajb_sjob)
            # iajb_sjob = Parallel(n_jobs=self.njob, return_as="generator")(
            #     delayed(self.wrapper_iajb_cvb)(i, a) for i, a in zip(pcsfcv_b[0], pcsfcv_b[1])
            # )
            # iajb += accumulator_sum(iajb_sjob)
            # # time.sleep(0.2)
            # # monitor.join()
            # # peak = max(monitor.memory_buffer) / 1e6
            # # print(f"Peak memory usage: {peak:.2f}MB")
            # del self.qAk_aa, self.qAj_aa, self.qAk_bb, self.qAj_bb, self.qBk_aa, self.qBj_aa, self.qBk_bb, self.qBj_bb, \
            #     self.ncsfcv_a0, self.ncsfcv_a1, self.ncsfov_a0, self.ncsfov_a1, \
            #     self.ncsfco_b0, self.ncsfco_b1, self.ncsfcv_b0, self.ncsfcv_b1, self.iaia_ncsf, \
            #     self.iaiacv_a, self.iaiaov_a, self.iaiaco_b, self.iaiacv_b, self.gamma_k, self.gamma_j, \
            #     self.cva, self.ova, self.cob, self.cvb, self.zero, self.nc, self.no

            iajb_path = np.einsum_path(
                'nm, nmi, nm->i',
                qAk_bb[:, None, pcsfco_b[0][0], pcsfco_b[1][0]],
                qBk_bb[None, :, ncsfco_b[0], ncsfco_b[1]],
                gamma_k, optimize=True
            )
            iajb_sjob = Parallel(n_jobs=self.njob, return_as="generator")(
                delayed(iajb_f)(
                    i, a, qAk_aa, qAj_aa, qBk_aa, qBj_aa, qBk_bb,
                    ncsfcv_a[0], ncsfcv_a[1], nc + ncsfov_a[0], ncsfov_a[1], ncsfco_b[0], ncsfco_b[1], ncsfcv_b[0], no + ncsfcv_b[1],
                    iaia_ncsf, iaiacv_a, gamma_k, gamma_j, 'cva', 0, 0, iajb_path[0]
                ) for i, a in zip(pcsfcv_a[0], pcsfcv_a[1])
            )
            iajb += accumulator_sum(iajb_sjob)
            iajb_sjob = Parallel(n_jobs=self.njob, return_as="generator")(
                delayed(iajb_f)(
                    i, a, qAk_aa, qAj_aa, qBk_aa, qBj_aa, qBk_bb,
                    ncsfcv_a[0], ncsfcv_a[1], nc + ncsfov_a[0], ncsfov_a[1], ncsfco_b[0], ncsfco_b[1], ncsfcv_b[0], no + ncsfcv_b[1],
                    iaia_ncsf, iaiaov_a, gamma_k, gamma_j, 'ova', nc, 0, iajb_path[0]
                ) for i, a in zip(pcsfov_a[0], pcsfov_a[1])
            )
            iajb += accumulator_sum(iajb_sjob)
            iajb_sjob = Parallel(n_jobs=self.njob, return_as="generator")(
                delayed(iajb_f)(
                    i, a, qAk_bb, qAj_bb, qBk_bb, qBj_bb, qBk_aa,
                    ncsfco_b[0], ncsfco_b[1], ncsfcv_b[0], no + ncsfcv_b[1], ncsfcv_a[0], ncsfcv_a[1], nc + ncsfov_a[0], ncsfov_a[1],
                    iaia_ncsf, iaiaco_b, gamma_k, gamma_j, 'cob', 0, 0, iajb_path[0]
                ) for i, a in zip(pcsfco_b[0], pcsfco_b[1])
            )
            iajb += accumulator_sum(iajb_sjob)
            iajb_sjob = Parallel(n_jobs=self.njob, return_as="generator")(
                delayed(iajb_f)(
                    i, a, qAk_bb, qAj_bb, qBk_bb, qBj_bb, qBk_aa,
                    ncsfco_b[0], ncsfco_b[1], ncsfcv_b[0], no + ncsfcv_b[1], ncsfcv_a[0], ncsfcv_a[1], nc + ncsfov_a[0], ncsfov_a[1],
                    iaia_ncsf, iaiacv_b, gamma_k, gamma_j, 'cvb', 0, no, iajb_path[0]
                ) for i, a in zip(pcsfcv_b[0], pcsfcv_b[1])
            )
            iajb += accumulator_sum(iajb_sjob)

            # # select SCSF add Fock and Delta A
            # # iajbfock = np.empty((0, iaia_ncsf.shape[0]))
            # iajb_sjob = Parallel(n_jobs=self.njob, backend='threading')(
            #     delayed(iajb_f)(
            #         i, a, qAk_aa, qAj_aa, qBk_aa, qBj_aa, qBk_bb,
            #         nc, nv, nc + no, nv, fock_a_occ, fock_a_vir,
            #         ncsfcv_a[0], ncsfcv_a[1], nc + ncsfov_a[0], ncsfov_a[1], ncsfco_b[0], ncsfco_b[1], ncsfcv_b[0], no + ncsfcv_b[1],
            #         iaia_ncsf, iaiacv_a, gamma_k, gamma_j, 'cva', 0, 0,
            #         nc, no, nv, ncsfcv_a, si, fij_a2, fij_b2, fab_a2, fab_b2
            #     ) for i, a in zip(pcsfcv_a[0], pcsfcv_a[1])
            # )
            # iajb += np.sum(iajb_sjob, axis=0)
            # # for i in range(len(iajb_sjob)):
            # #     iajb += iajb_sjob[i][0]
            # #     iajbfock = np.concatenate((iajbfock, np.expand_dims(iajb_sjob[i][1], axis=0)), axis=0)
            # # print('='*50)
            # iajb_sjob = Parallel(n_jobs=self.njob, backend='threading')(
            #     delayed(iajb_f)(
            #         i, a, qAk_aa, qAj_aa, qBk_aa, qBj_aa, qBk_bb,
            #         nc + no, nv, nc + no, nv, fock_a_occ, fock_a_vir,
            #         ncsfcv_a[0], ncsfcv_a[1], nc + ncsfov_a[0], ncsfov_a[1], ncsfco_b[0], ncsfco_b[1], ncsfcv_b[0], no + ncsfcv_b[1],
            #         iaia_ncsf, iaiaov_a, gamma_k, gamma_j, 'ova', nc, 0
            #     ) for i, a in zip(pcsfov_a[0], pcsfov_a[1])
            # )
            # iajb += np.sum(iajb_sjob, axis=0)
            # # for i in range(len(iajb_sjob)):
            # #     iajb += iajb_sjob[i][0]
            # #     iajbfock = np.concatenate((iajbfock, np.expand_dims(iajb_sjob[i][1], axis=0)), axis=0)
            # # print('=' * 50)
            # iajb_sjob = Parallel(n_jobs=self.njob, backend='threading')(
            #     delayed(iajb_f)(
            #         i, a, qAk_bb, qAj_bb, qBk_bb, qBj_bb, qBk_aa,
            #         nc, no + nv, nc, no + nv, fock_b_occ, fock_b_vir,
            #         ncsfco_b[0], ncsfco_b[1], ncsfcv_b[0], no + ncsfcv_b[1], ncsfcv_a[0], ncsfcv_a[1], nc + ncsfov_a[0], ncsfov_a[1],
            #         iaia_ncsf, iaiaco_b, gamma_k, gamma_j, 'cob'
            #     ) for i, a in zip(pcsfco_b[0], pcsfco_b[1])
            # )
            # iajb += np.sum(iajb_sjob, axis=0)
            # # for i in range(len(iajb_sjob)):
            # #     iajb += iajb_sjob[i][0]
            # #     iajbfock = np.concatenate((iajbfock, np.expand_dims(iajb_sjob[i][1], axis=0)), axis=0)
            # # print('=' * 50)
            # iajb_sjob = Parallel(n_jobs=self.njob, backend='threading')(
            #     delayed(iajb_f)(
            #         i, a, qAk_bb, qAj_bb, qBk_bb, qBj_bb, qBk_aa,
            #         nc, no + nv, nc, no + nv, fock_b_occ, fock_b_vir,
            #         ncsfco_b[0], ncsfco_b[1], ncsfcv_b[0], no + ncsfcv_b[1], ncsfcv_a[0], ncsfcv_a[1], nc + ncsfov_a[0], ncsfov_a[1],
            #         iaia_ncsf, iaiacv_b, gamma_k, gamma_j, 'cvb', 0, no,
            #         nc, no, nv, ncsfcv_b, si, fij_a2, fij_b2, fab_a2, fab_b2
            #     ) for i, a in zip(pcsfcv_b[0], pcsfcv_b[1])
            # )
            # iajb += np.sum(iajb_sjob, axis=0)
            # # for i in range(len(iajb_sjob)):
            # #     iajb += iajb_sjob[i][0]
            # #     iajbfock = np.concatenate((iajbfock, np.expand_dims(iajb_sjob[i][1], axis=0)), axis=0)
            # # iajbfock_sum = np.sum(iajbfock, axis=0)
            # # print("add fock extra select SCSF : {}".format(np.sum(iajbfock_sum > 1e-4)))
            # print('=' * 50)


            # # not para
            # iajb = iajb_f(
            #     pcsfcv[0], pcsfcv[1], qAk_aa, qAj_aa, qBk_aa, qBj_aa, qBk_bb,
            #     nc, nv, nc + no, nv, fock_a_occ, fock_a_vir,
            #     ncsfcv[0], ncsfcv[1], nc+ncsfov_a[0], ncsfov_a[1], ncsfco_b[0], ncsfco_b[1], ncsfcv[0], no+ncsfcv[1],
            #     iajb, iaia_ncsf, iaiacv_a, gamma_k, gamma_j, 'cva', 0, 0,
            #     nc, no, nv, ncsfcv, si, fij_a2, fij_b2, fab_a2, fab_b2
            # )
            # iajb = iajb_f(
            #     pcsfov_a[0], pcsfov_a[1], qAk_aa, qAj_aa, qBk_aa, qBj_aa, qBk_bb,
            #     nc + no, nv, nc + no, nv, fock_a_occ, fock_a_vir,
            #     ncsfcv[0], ncsfcv[1], nc+ncsfov_a[0], ncsfov_a[1], ncsfco_b[0], ncsfco_b[1], ncsfcv[0], no+ncsfcv[1],
            #     iajb, iaia_ncsf, iaiaov_a, gamma_k, gamma_j, 'ova', nc, 0
            # )
            # iajb = iajb_f(
            #     pcsfco_b[0], pcsfco_b[1], qAk_bb, qAj_bb, qBk_bb, qBj_bb, qBk_aa,
            #     nc, no + nv, nc, no + nv, fock_b_occ, fock_b_vir,
            #     ncsfco_b[0], ncsfco_b[1], ncsfcv[0], no+ncsfcv[1], ncsfcv[0], ncsfcv[1], nc+ncsfov_a[0], ncsfov_a[1],
            #     iajb, iaia_ncsf, iaiaco_b, gamma_k, gamma_j, 'cob'
            # )
            # iajb = iajb_f(
            #     pcsfcv[0], pcsfcv[1], qAk_bb, qAj_bb, qBk_bb, qBj_bb, qBk_aa,
            #     nc, no + nv, nc, no + nv, fock_b_occ, fock_b_vir,
            #     ncsfco_b[0], ncsfco_b[1], ncsfcv[0], no+ncsfcv[1], ncsfcv[0], ncsfcv[1], nc+ncsfov_a[0], ncsfov_a[1],
            #     iajb, iaia_ncsf, iaiacv_b, gamma_k, gamma_j, 'cvb', 0, no,
            #     nc, no, nv, ncsfcv, si, fij_a2, fij_b2, fab_a2, fab_b2
            # )
            # del iaia, iaia_ncsf, iaiacv_a, iaiaov_a, iaiaco_b, iaiacv_b

            # # origin code
            # for i, j in zip(pcsfcv_a[0], pcsfcv_a[1]):
            #     iajbcv_a = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, i, j], qBk_aa[None, :, ncsfcv_a[0], ncsfcv_a[1]], gamma_k)
            #     iajbcv_a -= np.einsum('nmi, nmi, nm->i', qAj_aa[:, None, i, ncsfcv_a[0]], qBj_aa[None, :, j, ncsfcv_a[1]], gamma_j)
            #     # iajbcv_a += np.einsum('i, i->i', np.eye(nc)[i, ncsfcv_a[0]], fock_a_vir[j, ncsfcv_a[1]])
            #     # iajbcv_a -= np.einsum('i, i->i', fock_a_occ[i, ncsfcv_a[0]], np.eye(nv)[j, ncsfcv_a[1]])
            #     # if self.spinadapt:
            #     #     iajbcv_a += _cvaacvaa_ndc(i, j, ncsfcv_a, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
            #     # Note: calculate ov_a, so add nc and ncsfov_a, others is same
            #     iajbov_a = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, i, j], qBk_aa[None, :, nc+ncsfov_a[0], ncsfov_a[1]], gamma_k)
            #     iajbov_a -= np.einsum('nmi, nmi, nm->i', qAj_aa[:, None, i, nc+ncsfov_a[0]], qBj_aa[None, :, j, ncsfov_a[1]], gamma_j)
            #     # iajbov_a += np.einsum('i, i->i', np.eye(nc + no)[i, nc+ncsfov_a[0]], fock_a_vir[j, ncsfov_a[1]])
            #     # iajbov_a -= np.einsum('i, i->i', fock_a_occ[i, nc+ncsfov_a[0]], np.eye(nv)[j, ncsfov_a[1]])
            #     iajbkco_b = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, i, j], qBk_bb[None, :, ncsfco_b[0], ncsfco_b[1]], gamma_k)
            #     iajbkcv_b = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, i, j], qBk_bb[None, :, ncsfcv_b[0], no+ncsfcv_b[1]], gamma_k)
            #     # if self.spinadapt:
            #     #     iajbkcv_b += _cvaacvbb_ndc(i, j, ncsfcv_b, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
            #     iajbline = np.concatenate((iajbcv_a, iajbov_a, iajbkco_b, iajbkcv_b), axis=0)
            #     iajb += iajbline**2/(iaia_ncsf-iaiacv_a[i, j]+1e-10)
            # for i, j in zip(pcsfov_a[0], pcsfov_a[1]):
            #     iajbcv_a = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, nc+i, j], qBk_aa[None, :, ncsfcv_a[0], ncsfcv_a[1]], gamma_k)
            #     iajbcv_a -= np.einsum('nmi, nmi, nm->i', qAj_aa[:, None, nc+i, ncsfcv_a[0]], qBj_aa[None, :, j, ncsfcv_a[1]], gamma_j)
            #     # iajbcv_a += np.einsum('i, i->i', np.eye(nc+no)[nc+i, ncsfcv_a[0]], fock_a_vir[j, ncsfcv_a[1]])
            #     # iajbcv_a -= np.einsum('i, i->i', fock_a_occ[nc+i, ncsfcv_a[0]], np.eye(nv)[j, ncsfcv_a[1]])
            #     iajbov_a = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, nc+i, j], qBk_aa[None, :, nc+ncsfov_a[0], ncsfov_a[1]], gamma_k)
            #     iajbov_a -= np.einsum('nmi, nmi, nm->i', qAj_aa[:, None, nc+i, nc+ncsfov_a[0]], qBj_aa[None, :, j, ncsfov_a[1]], gamma_j)
            #     # iajbov_a += np.einsum('i, i->i', np.eye(nc+no)[nc+i, nc+ncsfov_a[0]], fock_a_vir[j, ncsfov_a[1]])
            #     # iajbov_a -= np.einsum('i, i->i', fock_a_occ[nc+i, nc+ncsfov_a[0]], np.eye(nv)[j, ncsfov_a[1]])
            #     iajbkco_b = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, nc+i, j], qBk_bb[None, :, ncsfco_b[0], ncsfco_b[1]], gamma_k)
            #     iajbkcv_b = np.einsum('nm, nmi, nm->i', qAk_aa[:, None, nc+i, j], qBk_bb[None, :, ncsfcv_b[0], no+ncsfcv_b[1]], gamma_k)
            #     iajbline = np.concatenate((iajbcv_a, iajbov_a, iajbkco_b, iajbkcv_b), axis=0)
            #     # Note: iaiaov_a[i, j] can not use nc+i as index, so iajb_f do not suit for this iteration
            #     iajb += iajbline ** 2 / (iaia_ncsf - iaiaov_a[i, j] + 1e-10)
            # for i, j in zip(pcsfco_b[0], pcsfco_b[1]):
            #     iajbkcv_a = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, j], qBk_aa[None, :, ncsfcv_a[0], ncsfcv_a[1]], gamma_k)
            #     iajbkov_a = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, j], qBk_aa[None, :, nc+ncsfov_a[0], ncsfov_a[1]], gamma_k)
            #     iajbco_b = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, j], qBk_bb[None, :, ncsfco_b[0], ncsfco_b[1]], gamma_k)
            #     iajbco_b -= np.einsum('nmi, nmi, nm->i', qAj_bb[:, None, i, ncsfco_b[0]], qBj_bb[None, :, j, ncsfco_b[1]], gamma_j)
            #     # iajbco_b += np.einsum('i, i->i', np.eye(nc)[i, ncsfco_b[0]], fock_b_vir[j, ncsfco_b[1]])
            #     # iajbco_b -= np.einsum('i, i->i', fock_b_occ[i, ncsfco_b[0]], np.eye(no + nv)[j, ncsfco_b[1]])
            #     iajbcv_b = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, j], qBk_bb[None, :, ncsfcv_b[0], no+ncsfcv_b[1]], gamma_k)
            #     iajbcv_b -= np.einsum('nmi, nmi, nm->i', qAj_bb[:, None, i, ncsfcv_b[0]], qBj_bb[None, :, j, no+ncsfcv_b[1]], gamma_j)
            #     # iajbcv_b += np.einsum('i, i->i', np.eye(nc)[i, ncsfcv_b[0]], fock_b_vir[j, no+ncsfcv_b[1]])
            #     # iajbcv_b -= np.einsum('i, i->i', fock_b_occ[i, ncsfcv_b[0]], np.eye(no+nv)[j, no+ncsfcv_b[1]])
            #     iajbline = np.concatenate((iajbkcv_a, iajbkov_a, iajbco_b, iajbcv_b), axis=0)
            #     iajb += iajbline ** 2 / (iaia_ncsf - iaiaco_b[i, j] + 1e-10)
            # for i, j in zip(pcsfcv_b[0], pcsfcv_b[1]):
            #     iajbkcv_a = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, no+j], qBk_aa[None, :, ncsfcv_a[0], ncsfcv_a[1]], gamma_k)
            #     # if self.spinadapt:
            #     #     iajbkcv_a += _cvaacvbb_ndc(i, j, ncsfcv_a, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)  # _cvaacvbb = _cvbbcvaa
            #     iajbkov_a = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, no+j], qBk_aa[None, :, nc + ncsfov_a[0], ncsfov_a[1]], gamma_k)
            #     iajbco_b = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, no+j], qBk_bb[None, :, ncsfco_b[0], ncsfco_b[1]], gamma_k)
            #     iajbco_b -= np.einsum('nmi, nmi, nm->i', qAj_bb[:, None, i, ncsfco_b[0]], qBj_bb[None, :, no+j, ncsfco_b[1]], gamma_j)
            #     # iajbco_b += np.einsum('i, i->i', np.eye(nc)[i, ncsfco_b[0]], fock_b_vir[no+j, ncsfco_b[1]])
            #     # iajbco_b -= np.einsum('i, i->i', fock_b_occ[i, ncsfco_b[0]], np.eye(no + nv)[no+j, ncsfco_b[1]])
            #     iajbcv_b = np.einsum('nm, nmi, nm->i', qAk_bb[:, None, i, no+j], qBk_bb[None, :, ncsfcv_b[0], no + ncsfcv_b[1]], gamma_k)
            #     iajbcv_b -= np.einsum('nmi, nmi, nm->i', qAj_bb[:, None, i, ncsfcv_b[0]], qBj_bb[None, :, no+j, no + ncsfcv_b[1]], gamma_j)
            #     # iajbcv_b += np.einsum('i, i->i', np.eye(nc)[i, ncsfcv_b[0]], fock_b_vir[no+j, no + ncsfcv_b[1]])
            #     # iajbcv_b -= np.einsum('i, i->i', fock_b_occ[i, ncsfcv_b[0]], np.eye(no + nv)[no+j, no + ncsfcv_b[1]])
            #     # if self.spinadapt:
            #     #     iajbcv_b += _cvbbcvbb_ndc(i, j, ncsfcv_b, nc, no, nv, si, fij_a2, fij_b2, fab_a2, fab_b2)
            #     iajbline = np.concatenate((iajbkcv_a, iajbkov_a, iajbco_b, iajbcv_b), axis=0)
            #     # Note: iaiacv_b[i, j] can not use no+j as index, so iajb_f do not suit for this iteration
            #     iajb += iajbline ** 2 / (iaia_ncsf - iaiacv_b[i, j] + 1e-10)
            # del iajbkcv_a, iajbkov_a, iajbkco_b, iajbkcv_b, iajbcv_a, iajbov_a, iajbco_b, iajbcv_b,\
            #     iaia, iajbline, iaia_ncsf, iaiacv_a, iaiaov_a, iaiaco_b, iaiacv_b

            scsf = np.where(iajb >= 1e-4)[0]
            # # encapsulate to a function, same with upper code
            # CV(aa)
            pscsfcv_a_i, pscsfcv_a_a, scsf = devide_csf_ps(pcsfcv_a, ncsfcv_a, scsf)
            # OV(aa)
            pscsfov_a_i, pscsfov_a_a, scsf = devide_csf_ps(pcsfov_a, ncsfov_a, scsf)
            # CO(bb)
            pscsfco_b_i, pscsfco_b_a, scsf = devide_csf_ps(pcsfco_b, ncsfco_b, scsf)
            # CV(bb)
            pscsfcv_b_i, pscsfcv_b_a, scsf = devide_csf_ps(pcsfcv_b, ncsfcv_b, scsf)
            print("before union, {} SCSFs in CV(aa)".format(len(pscsfcv_a_i) - len(pcsfcv_a[0])))
            print("before union, {} SCSFs in CV(bb)".format(len(pscsfcv_b_i) - len(pcsfcv_b[0])))
            if self.union:
                pscsfcv_a_i, pscsfcv_a_a = union(nc, (pscsfcv_a_i, pscsfcv_a_a), (pscsfcv_b_i, pscsfcv_b_a))
            pscsfcv_b_i, pscsfcv_b_a = pscsfcv_a_i, pscsfcv_a_a

            Adim = len(pscsfcv_a_i) + len(pscsfov_a_i) + len(pscsfco_b_i) + len(pscsfcv_b_i)
            pcsfdim = len(pcsfcv_a[0]) + len(pcsfov_a[0]) + len(pcsfco_b[0]) + len(pcsfcv_b[0])
            scsfdim = len(pscsfcv_a_i) + len(pscsfov_a_i) + len(pscsfco_b_i) + len(pscsfcv_b_i) - pcsfdim
            ncsfdim = len(iajb) - scsfdim
            print('After select, A matrix dimension is {}'.format(Adim))
            print('{} CSFs in pcsf ({} in CV(aa), {} in OV(aa), {} in CO(bb), {} in CV(bb))'.format(
                pcsfdim, len(pcsfcv_a[0]), len(pcsfov_a[0]), len(pcsfco_b[0]), len(pcsfcv_b[0])
            ))
            print('{} CSFs in scsf ({} in CV(aa), {} in OV(aa), {} in CO(bb), {} in CV(bb))'.format(
                scsfdim, len(pscsfcv_a_i) - len(pcsfcv_a[0]), len(pscsfov_a_i) - len(pcsfov_a[0]),
                         len(pscsfco_b_i) - len(pcsfco_b[0]), len(pscsfcv_b_i) - len(pcsfcv_b[0])
            ))
            print('{} CSFs in ncsf ({} in CV(aa), {} in OV(aa), {} in CO(bb), {} in CV(bb))'.format(
                ncsfdim, nc * nv - len(pscsfcv_a_i), no * nv - len(pscsfov_a_i), nc * no - len(pscsfco_b_i),
                         nc * nv - len(pscsfcv_b_i)
            ))

            t1 = time.time()
            t_scsf = t1 - t0
        else:
            Adim = (nc + no) * nv + nc * (no + nv)
            print('no * nv is {}'.format(Adim))
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

        # # Section: calculate overlap need transform different order to same, so here calculate order
        # pscsf_fdiag_a[pscsfcv_a_i + pscsf_fdiag_a.shape[0] - nc - no, pscsfcv_a_a] = True
        # pscsf_fdiag_a[pscsfov_a_i + pscsf_fdiag_a.shape[0] - nc - no + nc, pscsfov_a_a] = True
        # pscsf_fdiag_b[pscsfco_b_i + pscsf_fdiag_b.shape[0] - nc, pscsfco_b_a] = True
        # pscsf_fdiag_b[pscsfcv_b_i + pscsf_fdiag_b.shape[0] - nc, no + pscsfcv_b_a] = True
        # # Note: pscsf_fdiag order is pyscf order, so if calculate overlap, adjust pyscf csf order to my order
        # #  when use pscsf_fdiag, return add this variable
        # pscsf_fdiag = np.concatenate((pscsf_fdiag_a.reshape(-1), pscsf_fdiag_b.reshape(-1)))
        # nc_old, no_old, nv_old = utils.get_cov(self.mf)
        # order = utils.order_pyscf2my(nc_old, no_old, nv_old)
        # pscsf_fdiag = pscsf_fdiag[order]

        # Section: construct A
        t0 = time.time()
        A = np.zeros((Adim, Adim))
        # qAkcv_aa = qAk_aa[:, pscsfcv_i, pscsfcv_a]
        # qAkov_aa = qAk_aa[:, nc+pscsfov_a_i, pscsfov_a_a]
        # qAkco_bb = qAk_bb[:, pscsfco_b_i, pscsfco_b_a]
        # qAkcv_bb = qAk_bb[:, pscsfcv_i, no+pscsfcv_a]
        # qBkcv_aa = np.array(qAkcv_aa)
        # qBkov_aa = np.array(qAkov_aa)
        # qBkco_bb = np.array(qAkco_bb)
        # qBkcv_bb = np.array(qAkcv_bb)
        lcva = len(pscsfcv_a_i)
        lova = len(pscsfov_a_i)
        lcob = len(pscsfco_b_i)
        lcvb = len(pscsfcv_b_i)
        # if self.correct:
        #     delta_max = 0.5 / unit.ha2eV
        #     sigma_k = 0.1 / unit.ha2eV
        # # # CV(aa)-CV(aa)
        # # # do not line by line construct int2ekcv_a is for self.correct
        # # int2ekcv_a = np.einsum('nmi, nma, nm->ia', qAkcv_aa[:, None, ...], qBkcv_aa[None, ...], gamma_k)
        # for i, a, p in zip(pscsfcv_i, pscsfcv_a, range(lcva)):
        #     A[p, :lcva] += np.einsum('nm, nmp, nm->p', qAk_aa[:, None, i, a], qBk_aa[None, :, pscsfcv_i, pscsfcv_a], gamma_k)
        #     A[p, :lcva] -= np.einsum('nmp, nmp, nm->p', qAj_aa[:, None, i, pscsfcv_i], qBj_aa[None, :, a, pscsfcv_a], gamma_j)
        #     A[p, :lcva] -= np.einsum('p, p->p', fock_a_occ[i, pscsfcv_i], np.eye(nv)[a, pscsfcv_a])
        #     A[p, :lcva] += np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_i], fock_a_vir[a, pscsfcv_a])
        #     A[p, :lcva] += (
        #         0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
        #             np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_i], fab_b2[no+a, no+pscsfcv_a])
        #             - np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_i], fab_a2[a, pscsfcv_a])
        #         )
        #         + 0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
        #                 np.einsum('p, p->p', np.eye(no+nv)[no+a, no+pscsfcv_a], fij_b2[i, pscsfcv_i])
        #                 - np.einsum('p, p->p', np.eye(nv)[a, pscsfcv_a], fij_a2[i, pscsfcv_i])
        #         )
        #     )
        # # A[:lcva, :lcva] += int2ekcv_a
        # # if self.correct:
        # #     delta_k = delta_max / (1 + (np.diag(int2ekcv_a) / sigma_k) ** 4)
        # #     A[:lcva, :lcva] += np.diag(delta_k)
        # # del int2ekcv_a

        # CV(aa)-CV(aa), CV(aa)-CV(bb), CV(bb)-CV(bb)
        Acvacva, Acvacvb, Acvbcvb = constractA(
            pscsfcv_a_i, pscsfcv_a_a, pscsfcv_b_i, pscsfcv_b_a, nc, no, nv, gamma_k, gamma_j,
            qAk_aa, qBk_aa, qAj_aa, qBj_aa, qAk_bb, qBk_bb, qAj_bb, qBj_bb,
            fock_a_occ, fock_a_vir, fock_b_occ, fock_b_vir, si, fij_a2, fij_b2, fab_a2, fab_b2, self.njob,
            self.correct, delta_max, sigma_k
        )
        A[:lcva, :lcva] = Acvacva
        A[:lcva, lcva + lova + lcob:] = Acvacvb
        A[lcva + lova + lcob:, lcva + lova + lcob:] = Acvbcvb
        del Acvacva, Acvacvb, Acvbcvb

        # # CV(aa)-OV(aa)
        # int2ekcvaova = np.einsum('nmi, nma, nm->ia', qAkcv_aa[:, None, ...], qBkov_aa[None, ...], gamma_k)
        # int2ejcvaova = np.zeros_like(int2ekcvaova)
        for i, a, p in zip(pscsfcv_a_i, pscsfcv_a_a, range(lcva)):
            # int2ejcvaova[p, :] = np.einsum('nmp, nmp, nm->p', qAj_aa[:, None, i, nc+pscsfov_a_i], qBj_aa[None, :, a, pscsfov_a_a], gamma_j)
            A[p, lcva:lcva + lova] += np.einsum(
                'nm, nmp, nm->p',
                qAk_aa[:, None, i, a], qBk_aa[None, :, nc + pscsfov_a_i, pscsfov_a_a], gamma_k, optimize=True
            )
            A[p, lcva:lcva + lova] -= np.einsum(
                'nmp, nmp, nm->p',
                qAj_aa[:, None, i, nc + pscsfov_a_i], qBj_aa[None, :, a, pscsfov_a_a], gamma_j
            )
            A[p, lcva:lcva + lova] -= np.einsum(
                'p, p->p', fock_a_occ[i, nc + pscsfov_a_i], np.eye(nv)[a, pscsfov_a_a], optimize=True
            )
            A[p, lcva:lcva + lova] += np.einsum(
                'p, p->p', np.eye(nc + no)[i, nc + pscsfov_a_i], fock_a_vir[a, pscsfov_a_a], optimize=True
            )
        # A[:lcva, lcva:lcva+lova] += int2ekcvaova - int2ejcvaova
        # del int2ekcvaova, int2ejcvaova

        # # CV(aa)-CO(bb)
        # A[:lcva, lcva+lova:lcva+lova+lcob] = np.einsum('nmi, nma, nm->ia', qAkcv_aa[:, None, ...], qBkco_bb[None, ...], gamma_k)
        for i, a, p in zip(pscsfcv_a_i, pscsfcv_a_a, range(lcva)):
            A[p, lcva + lova:lcva + lova + lcob] += np.einsum(
                'nm, nmp, nm->p',
                qAk_aa[:, None, i, a], qBk_bb[None, :, pscsfco_b_i, pscsfco_b_a], gamma_k, optimize=True
            )

        # # # CV(aa)-CV(bb)
        # # A[:lcva, lcva+lova+lcob:] = np.einsum('nmi, nma, nm->ia', qAkcv_aa[:, None, ...], qBkcv_bb[None, ...], gamma_k)
        # for i, a, p in zip(pscsfcv_i, pscsfcv_a, range(lcva)):
        #     A[p, lcva+lova+lcob:] += np.einsum('nm, nmp, nm->p', qAk_aa[:, None, i, a], qBk_bb[None, :, pscsfcv_i, no+pscsfcv_a], gamma_k)
        #     A[p, lcva+lova+lcob:] += (0.5 * 1 / (2 * si) * (
        #         np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_i], fab_b2[no+a, no+pscsfcv_a])
        #         - np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_i], fab_a2[a, pscsfcv_a])
        #         + np.einsum('p, p->p', np.eye(no+nv)[no+a, no+pscsfcv_a], fij_b2[i, pscsfcv_i])
        #         - np.einsum('p, p->p', np.eye(nv)[a, pscsfcv_a], fij_a2[i, pscsfcv_i])
        #     ))

        # OV(aa)-CV(aa)
        A[lcva:lcva + lova, :lcva] = A[:lcva, lcva:lcva + lova].T

        # # OV(aa)-OV(aa)
        int2ekov_a = np.einsum(
            'nmi, nma, nm->ia',
            qAk_aa[:, None, nc + pscsfov_a_i, pscsfov_a_a], qBk_aa[None, :, nc + pscsfov_a_i, pscsfov_a_a], gamma_k, optimize=True,
        )
        # int2ejov_a = np.zeros_like(int2ekov_a)
        for i, a, p in zip(pscsfov_a_i, pscsfov_a_a, range(lova)):
            # # int2ejov_a[p, :] = np.einsum('nmp, nmp, nm->p', qAj_aa[:, None, nc+i, nc+pscsfov_a_i], qBj_aa[None, :, a, pscsfov_a_a], gamma_j)
            # # for add correct term, do not line by line calculate
            # A[lcva+p, lcva:lcva+lova] += np.einsum(
            #     'nm, nmp, nm->p',
            #     qAk_aa[:, None, nc+i, a], qBk_aa[None, :, nc+pscsfov_a_i, pscsfov_a_a], gamma_k, optimize=True
            # )
            A[lcva + p, lcva:lcva + lova] -= np.einsum(
                'nmp, nmp, nm->p',
                qAj_aa[:, None, nc + i, nc + pscsfov_a_i], qBj_aa[None, :, a, pscsfov_a_a], gamma_j
            )
            A[lcva + p, lcva:lcva + lova] -= np.einsum(
                'p, p->p', fock_a_occ[nc + i, nc + pscsfov_a_i], np.eye(nv)[a, pscsfov_a_a], optimize=True
            )
            A[lcva + p, lcva:lcva + lova] += np.einsum(
                'p, p->p', np.eye(nc + no)[nc + i, nc + pscsfov_a_i], fock_a_vir[a, pscsfov_a_a], optimize=True
            )
        A[lcva:lcva + lova, lcva:lcva + lova] += int2ekov_a
        if self.correct:
            delta_k = delta_max / (1 + (np.diag(int2ekov_a) / sigma_k) ** 4)
            A[lcva:lcva + lova, lcva:lcva + lova] += np.diag(delta_k)
        del int2ekov_a

        # # OV(aa)-CO(bb)
        # A[lcva:lcva+lova, lcva+lova:lcva+lova+lcob] = np.einsum('nmi, nma, nm->ia', qAkov_aa[:, None, ...], qBkco_bb[None, ...], gamma_k)
        for i, a, p in zip(pscsfov_a_i, pscsfov_a_a, range(lova)):
            A[lcva + p, lcva + lova:lcva + lova + lcob] += np.einsum(
                'nm, nmp, nm->p',
                qAk_aa[:, None, nc + i, a], qBk_bb[None, :, pscsfco_b_i, pscsfco_b_a], gamma_k, optimize=True
            )

        # # OV(aa)-CV(bb)
        # A[lcva:lcva+lova, lcva+lova+lcob:] = np.einsum('nmi, nma, nm->ia', qAkov_aa[:, None, ...], qBkcv_bb[None, ...], gamma_k)
        for i, a, p in zip(pscsfov_a_i, pscsfov_a_a, range(lova)):
            A[lcva + p, lcva + lova + lcob:] += np.einsum(
                'nm, nmp, nm->p',
                qAk_aa[:, None, nc + i, a], qBk_bb[None, :, pscsfcv_b_i, no + pscsfcv_b_a], gamma_k, optimize=True
            )

        # CO(bb)-CV(aa)
        A[lcva + lova:lcva + lova + lcob, :lcva] = A[:lcva, lcva + lova:lcva + lova + lcob].T

        # CO(bb)-OV(aa)
        A[lcva + lova:lcva + lova + lcob, lcva:lcva + lova] = A[lcva:lcva + lova, lcva + lova:lcva + lova + lcob].T

        # # CO(bb)-CO(bb)
        int2ekco_b = np.einsum(
            'nmi, nma, nm->ia',
            qAk_bb[:, None, pscsfco_b_i, pscsfco_b_a], qBk_bb[None, :, pscsfco_b_i, pscsfco_b_a], gamma_k, optimize=True
        )
        # int2ejco_b = np.zeros_like(int2ekco_b)
        for i, a, p in zip(pscsfco_b_i, pscsfco_b_a, range(lcob)):
            # # int2ejco_b[p, :] = np.einsum('nmp, nmp, nm->p', qAj_bb[:, None, i, pscsfco_b_i], qBj_bb[None, :, a, pscsfco_b_a], gamma_j)
            # # for add correct term, do not line by line calculate
            # A[lcva+lova+p, lcva+lova:lcva+lova+lcob] += np.einsum(
            #     'nm, nmp, nm->p',
            #     qAk_bb[:, None, i, a],qBk_bb[None, :, pscsfco_b_i, pscsfco_b_a], gamma_k, optimize=True
            # )
            A[lcva + lova + p, lcva + lova:lcva + lova + lcob] -= np.einsum(
                'nmp, nmp, nm->p',
                qAj_bb[:, None, i, pscsfco_b_i], qBj_bb[None, :, a, pscsfco_b_a], gamma_j
            )
            A[lcva + lova + p, lcva + lova:lcva + lova + lcob] -= np.einsum(
                'p, p->p', fock_b_occ[i, pscsfco_b_i], np.eye(no + nv)[a, pscsfco_b_a], optimize=True
            )
            A[lcva + lova + p, lcva + lova:lcva + lova + lcob] += np.einsum(
                'p, p->p', np.eye(nc)[i, pscsfco_b_i], fock_b_vir[a, pscsfco_b_a], optimize=True
            )
        A[lcva + lova:lcva + lova + lcob, lcva + lova:lcva + lova + lcob] += int2ekco_b
        if self.correct:
            delta_k = delta_max / (1 + (np.diag(int2ekco_b) / sigma_k) ** 4)
            A[lcva + lova:lcva + lova + lcob, lcva + lova:lcva + lova + lcob] += np.diag(delta_k)
        del int2ekco_b

        # # CO(bb)-CV(bb)
        # int2ekcobcvb = np.einsum('nmi, nma, nm->ia', qAkco_bb[:, None, ...], qBkcv_bb[None, ...], gamma_k)
        # int2ejcobcvb = np.zeros_like(int2ekcobcvb)
        for i, a, p in zip(pscsfco_b_i, pscsfco_b_a, range(lcob)):
            # int2ejcobcvb[p, :] = np.einsum('nmp, nmp, nm->p', qAj_bb[:, None, i, pscsfcv_i], qBj_bb[None, :, a, no+pscsfcv_a], gamma_j)
            A[lcva + lova + p, lcva + lova + lcob:] += np.einsum(
                'nm, nmp, nm->p',
                qAk_bb[:, None, i, a], qBk_bb[None, :, pscsfcv_b_i, no + pscsfcv_b_a], gamma_k, optimize=True
            )
            A[lcva + lova + p, lcva + lova + lcob:] -= np.einsum(
                'nmp, nmp, nm->p',
                qAj_bb[:, None, i, pscsfcv_b_i], qBj_bb[None, :, a, no + pscsfcv_b_a], gamma_j
            )
            A[lcva + lova + p, lcva + lova + lcob:] -= np.einsum(
                'p, p->p', fock_b_occ[i, pscsfcv_b_i], np.eye(no + nv)[a, no + pscsfcv_b_a], optimize=True
            )
            A[lcva + lova + p, lcva + lova + lcob:] += np.einsum(
                'p, p->p', np.eye(nc)[i, pscsfcv_b_i], fock_b_vir[a, no + pscsfcv_b_a], optimize=True
            )
        # A[lcva+lova:lcva+lova+lcob, lcva+lova+lcob:] += int2ekcobcvb - int2ejcobcvb
        # del int2ekcobcvb, int2ejcobcvb

        # CV(bb)-CV(aa)
        A[lcva + lova + lcob:, :lcva] = A[:lcva, lcva + lova + lcob:].T

        # CV(bb)-OV(aa)
        A[lcva + lova + lcob:, lcva:lcva + lova] = A[lcva:lcva + lova, lcva + lova + lcob:].T

        # CV(bb)-CO(bb)
        A[lcva + lova + lcob:, lcva + lova:lcva + lova + lcob] = A[lcva + lova:lcva + lova + lcob, lcva + lova + lcob:].T
        t1 = time.time()
        t_cA = t1 - t0

        # # # CV(bb)-CV(bb)
        # # int2ekcv_b = np.einsum('nmi, nma, nm->ia', qAkcv_bb[:, None, ...], qBkcv_bb[None, ...], gamma_k)
        # # int2ejcv_b = np.zeros_like(int2ekcv_b)
        # for i, a, p in zip(pscsfcv_i, pscsfcv_a, range(lcvb)):
        #     # int2ejcv_b[p, :] = np.einsum('nmp, nmp, nm->p', qAj_bb[:, None, i, pscsfcv_i], qBj_bb[None, :, no+a, no+pscsfcv_a], gamma_j)
        #     A[lcva+lova+lcob+p, lcva+lova+lcob:] += np.einsum('nm, nmp, nm->p', qAk_bb[:, None, i, no+a], qBk_bb[None, :, pscsfcv_i, no+pscsfcv_a], gamma_k)
        #     A[lcva+lova+lcob+p, lcva+lova+lcob:] -= np.einsum('nmp, nmp, nm->p', qAj_bb[:, None, i, pscsfcv_i], qBj_bb[None, :, no+a, no+pscsfcv_a], gamma_j)
        #     A[lcva+lova+lcob+p, lcva+lova+lcob:] -= np.einsum('p, p->p', fock_b_occ[i, pscsfcv_i], np.eye(no+nv)[no+a, no+pscsfcv_a])
        #     A[lcva+lova+lcob+p, lcva+lova+lcob:] += np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_i], fock_b_vir[no+a, no+pscsfcv_a])
        #     A[lcva+lova+lcob+p, lcva+lova+lcob:] += (
        #         0.5 * (-1 + np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
        #             np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_i], fab_b2[no+a, no+pscsfcv_a])
        #             - np.einsum('p, p->p', np.eye(nc)[i, pscsfcv_i], fab_a2[a, pscsfcv_a])
        #         )
        #         + 0.5 * (1 - np.sqrt((si + 1) / si) + 1 / (2 * si)) * (
        #             np.einsum('p, p->p', np.eye(no+nv)[no+a, no+pscsfcv_a], fij_b2[i, pscsfcv_i])
        #             - np.einsum('p, p->p', np.eye(nv)[a, pscsfcv_a], fij_a2[i, pscsfcv_i])
        #         )
        #     )
        # # # A[lcva+lova+lcob:, lcva+lova+lcob:] += int2ekcv_b - int2ejcv_b
        # # A[lcva+lova+lcob:, lcva+lova+lcob:] += int2ekcv_b
        # # if self.correct:
        # #     delta_k = delta_max / (1 + (np.diag(int2ekcv_b) / sigma_k) ** 4)
        # #     A[lcva+lova+lcob:, lcva+lova+lcob:] += np.diag(delta_k)
        # # del int2ekcv_b

        # define some class variety, simple to use
        self.mo_coeff_a = mo_coeff_a
        self.mo_coeff_b = mo_coeff_b
        self.occidx_a = occidx_a
        self.viridx_a = viridx_a
        self.occidx_b = occidx_b
        self.viridx_b = viridx_b
        self.nc = nc
        self.no = no
        self.nv = nv
        self.pscsfcv_a_i = pscsfcv_a_i
        self.pscsfcv_a_a = pscsfcv_a_a
        self.pscsfov_a_i = pscsfov_a_i
        self.pscsfov_a_a = pscsfov_a_a
        self.pscsfco_b_i = pscsfco_b_i
        self.pscsfco_b_a = pscsfco_b_a
        self.pscsfcv_b_i = pscsfcv_b_i
        self.pscsfcv_b_a = pscsfcv_b_a
        self.lcva = lcva
        self.lova = lova
        self.lcob = lcob
        self.lcvb = lcvb

        t0 = time.time()
        self.e, self.v = scipy.linalg.eigh(A)
        t1 = time.time()
        t_diagA = t1 - t0
        self.e_eV = self.e[:self.nstates] * unit.ha2eV
        # # TODO(WHB): use eigen vector in spin tensor basis
        # self.v = utils.so2st(self.v[:, :self.nstates], lcva=lcva, lova=lova, lcob=lcob, lcvb=lcvb)
        # self.v = utils.so2st(self.v[:, :self.nstates], nc, no, nv)
        self.v = self.v[:, :self.nstates]
        if self.Emax:
            self.xycv_a = self.v.T[:, :lcva]  # V_cv_a
            self.xyov_a = self.v.T[:, lcva:lcva + lova]  # V_ov_a
            self.xyco_b = self.v.T[:, lcva + lova:lcva + lova + lcob]  # V_co_b
            self.xycv_b = self.v.T[:, lcva + lova + lcob:]  # V_cv_b
        else:
            self.xycv_a = self.v.T[:, :nc * nv]  # V_cv_a
            self.xyov_a = self.v.T[:, nc * nv:(nc + no) * nv]  # V_ov_a
            self.xyco_b = self.v.T[:, (nc + no) * nv:(nc + no) * nv + nc * no]  # V_co_b
            self.xycv_b = self.v.T[:, (nc + no) * nv + nc * no:]  # V_cv_b
        t0 = time.time()
        self.os = self.osc_str()  # oscillator strength
        t1 = time.time()
        t_os = t1 - t0
        # before calculate rot_str, check whether mol is chiral mol
        t0 = time.time()
        if gto.mole.chiral_mol(self.mol):
            rs = self.rot_str()
        else:
            rs = np.zeros(self.nstates)
        t1 = time.time()
        t_rs = t1 - t0
        t0 = time.time()
        self.dS2 = self.deltaS2()
        t1 = time.time()
        t_dS2 = t1 - t0
        if self.spinadapt:
            print('my sXTDA result is')
        else:
            print('my sUTDA result is')
        print(f'{"num":>4} {"energy":>8} {"wav_len":>8} {"osc_str":>8} {"rot_str":>8} {"deltaS2":>8}')
        for ni, ei, wli, osi, rsi, ds2i in zip(range(self.nstates), self.e_eV, unit.eVxnm / self.e_eV, self.os, rs, self.dS2):
            print(f'{ni + 1:4d} {ei:8.4f} {wli:8.4f} {osi:8.4f} {rsi:8.4f} {ds2i:8.4f}')
        if self.savedata:
            pd.DataFrame(
                np.concatenate((unit.eVxnm / np.expand_dims(self.e_eV, axis=1), np.expand_dims(self.os, axis=1)),
                               axis=1)
            ).to_csv('uvspec_data.csv', index=False, header=None)
        # np.save("energy_xstda.npy", self.e)
        t_all_1 = time.time()  # record kernel end time
        t_all = t_all_1 - t_all_0
        print('=' * 50)
        print("contract Fock matrix use          {:8.4f} s".format(t_getfock))
        if self.Emax:
            print("select PCSF use                   {:8.4f} s".format(t_pcsf))
            print("select SCSF use                   {:8.4f} s".format(t_scsf))
        print("contract A matrix use             {:8.4f} s".format(t_cA))
        print("diagonal A matrix use             {:8.4f} s".format(t_diagA))
        print("calculate oscillator strength use {:8.4f} s".format(t_os))
        print("calculate rotator strength use    {:8.4f} s".format(t_rs))
        print("calculate delta S^2 use           {:8.4f} s".format(t_dS2))
        if self.spinadapt:
            print("sX-TDA use                        {:8.4f} s".format(t_all))
        else:
            print("sU-TDA use                        {:8.4f} s".format(t_all))
        return self.e_eV, self.os, rs, self.v

    def deltaS2(self):
        # refer to J. Chem. Phys. 134, 134101 (2011), ignore some term and some term cancel each other out
        if self.spinadapt:
            dS2 = (np.einsum('ij,ij->i', self.xycv_a, self.xycv_a)
                   + np.einsum('ij,ij->i', self.xycv_b, self.xycv_b)
                   - 2 * np.einsum('ij,ij->i', self.xycv_a, self.xycv_b))
        else:
            orbo_a = self.mo_coeff_a[:, self.occidx_a]
            orbv_a = self.mo_coeff_a[:, self.viridx_a]
            orbo_b = self.mo_coeff_b[:, self.occidx_b]
            orbv_b = self.mo_coeff_b[:, self.viridx_b]
            S = self.mol.intor('int1e_ovlp')
            Sccba = np.einsum('pq,pi,qj->ij', S, orbo_b, orbo_a, optimize=True)
            Sccab = np.einsum('pq,pi,qj->ij', S, orbo_a, orbo_b, optimize=True)
            Svcab = np.einsum('pq,pi,qj->ij', S, orbv_a, orbo_b, optimize=True)
            Svcba = np.einsum('pq,pi,qj->ij', S, orbv_b, orbo_a, optimize=True)
            Svvab = np.einsum('pq,pi,qj->ij', S, orbv_a, orbv_b, optimize=True)
            if self.Emax:
                # here can not use truncate csfs to calculate dS2,z
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
                    np.einsum('nia,nja,ki,jk->n', xycv_a, xycv_a, Sccba[:, :self.nc], Sccba.T[:self.nc, :], optimize=True)  # first term cvacva
                    + np.einsum('nia,nja,ki,jk->n', xyov_a, xyov_a, Sccba[:, self.nc:], Sccba.T[self.nc:, :], optimize=True)  # first term ovaova
                    + np.einsum('nia,nja,ki,jk->n', xyov_a, xycv_a, Sccba[:, self.nc:], Sccba.T[:self.nc, :], optimize=True)  # first term ovacva
                    + np.einsum('nia,nja,ki,jk->n', xycv_a, xyov_a, Sccba[:, :self.nc], Sccba.T[self.nc:, :], optimize=True)  # first term cvaova
                    - np.einsum('nia,nib,ak,kb->n', xycv_a, xycv_a, Svcab, Svcab.T, optimize=True)  # second term  cva
                    - np.einsum('nia,nib,ak,kb->n', xyov_a, xyov_a, Svcab, Svcab.T, optimize=True)  # second term ova
                    + np.einsum('nia,nja,ki,jk->n', xycv_b, xycv_b, Sccab, Sccab.T, optimize=True)  # third term  cvb
                    + np.einsum('nia,nja,ki,jk->n', xyco_b, xyco_b, Sccab, Sccab.T, optimize=True)  # third term cob
                    - np.einsum('nia,nib,ak,kb->n', xyco_b, xyco_b, Svcba[:self.no, :], Svcba.T[:, :self.no], optimize=True)  # forth term cobcob
                    - np.einsum('nia,nib,ak,kb->n', xycv_b, xycv_b, Svcba[self.no:, :], Svcba.T[:, self.no:], optimize=True)  # forth term cvbcvb
                    - np.einsum('nia,nib,ak,kb->n', xyco_b, xycv_b, Svcba[:self.no, :], Svcba.T[:, self.no:], optimize=True)  # forth term cobcvb
                    - np.einsum('nia,nib,ak,kb->n', xycv_b, xyco_b, Svcba[self.no:, :], Svcba.T[:, :self.no], optimize=True)  # forth term cvbcob
                    - 2 * np.einsum('nia,njb,ji,ab->n', xycv_a, xycv_b, Sccba[:, :self.nc], Svvab[:, self.no:], optimize=True)  # fifth term cvacvb
                    - 2 * np.einsum('nia,njb,ji,ab->n', xycv_a, xyco_b, Sccba[:, :self.nc], Svvab[:, :self.no], optimize=True)  # fifth term cvacob
                    - 2 * np.einsum('nia,njb,ji,ab->n', xyov_a, xycv_b, Sccba[:, self.nc:], Svvab[:, self.no:], optimize=True)  # fifth term ovacvb
                    - 2 * np.einsum('nia,njb,ji,ab->n', xyov_a, xyco_b, Sccba[:, self.nc:], Svvab[:, :self.no], optimize=True)  # fifth term ovacob
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
        dipole_mo_a = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo_a, orbv_a, optimize=True)
        dipole_mo_b = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo_b, orbv_b, optimize=True)
        if self.Emax:
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
        f = 2. / 3. * np.einsum('s,sx,sx->s', omega, trans_dip, trans_dip, optimize=True)
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
        dip_ele_mo_a = np.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo_a, orbv_a, optimize=True)
        dip_ele_mo_b = np.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo_b, orbv_b, optimize=True)
        dip_meg_ao = self.mol.intor('int1e_cg_irxp', comp=3, hermi=2)  # transition magnetic dipole moment
        dip_meg_mo_a = np.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo_a, orbv_a, optimize=True)
        dip_meg_mo_b = np.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo_b, orbv_b, optimize=True)
        if self.Emax:
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
        f = np.einsum('s,sx,sx->s', 1. / omega, trans_ele_dip, trans_meg_dip, optimize=True)
        f = f / unit.cgs2au  # transform atom unit to cgs unit
        # np.save("rot_str_stda.npy", f)
        return f

    def analyze(self, out_filename='UsTDA'):
        nc = self.nc
        nv = self.nv
        no = self.no
        fnc = self.frozen_nc
        out_cube = [0, np.argmax(self.os)]  # only output 1st excited state and max os excited state orbital
        orbital = []  # record output orbital number
        if self.Emax:
            self.v = utils.so2st(self.v[:, :self.nstates], lcva=self.lcva, lova=self.lova, lcob=self.lcob, lcvb=self.lcvb)
        else:
            self.v = utils.so2st(self.v[:, :self.nstates], nc, no, nv)
        print("=" * 50)
        for nstate in range(self.nstates):
            value = self.v[:,nstate]
            x_cv_aa = value[:self.lcva]
            x_ov_aa = value[self.lcva:self.lcva+self.lova]
            x_co_bb = value[self.lcva+self.lova:self.lcva+self.lova+self.lcob]
            x_cv_bb = value[self.lcva+self.lova+self.lcob:]
            # print(f'Excited state {nstate + 1} {self.e[nstate] * unit.ha2eV:10.5f} eV')
            print(
                f'D{nstate + 1}' + r"    w:" + f'{self.e[nstate] * unit.ha2eV:10.4f} eV'
                + r"    d<S^2>:" + f'{self.dS2[nstate]:8.4f}'
                + r"    f:" + f'{self.os[nstate]:8.4f}'
            )
            print(
                r"CV(0): " + f"{np.sum(x_cv_aa ** 2 * 100):5.2f}%"
                + r"    OV(0): " + f"{np.sum(x_ov_aa ** 2 * 100):5.2f}%"
                + r"    CO(0): " + f"{np.sum(x_co_bb ** 2 * 100):5.2f}%"
                + r"    CV(1): " + f"{np.sum(x_cv_bb ** 2 * 100):5.2f}%"
            )
            # out_excittype = np.argmax((np.max(abs(x_cv_aa)), np.max(abs(x_ov_aa)), np.max(abs(x_co_bb)), np.max(abs(x_cv_bb))))
            if self.Emax:
                for i in np.where(abs(x_cv_aa) > 0.1)[0]:
                    print(f'    CV(0) {fnc+self.pscsfcv_a_i[i]+1:3d}a -> {fnc+self.pscsfcv_a_a[i]+1+nc+no:3d}a'
                          + f'    c_i: {x_cv_aa[i]:8.5f}    Per: {100*x_cv_aa[i]**2:5.2f}%')
                    # if nstate in out_cube and out_excittype == 0 and i == np.argmax(abs(x_cv_aa)):
                    if nstate in out_cube:
                        orbital.append(fnc+self.pscsfcv_a_i[i])
                        orbital.append(fnc+self.pscsfcv_a_a[i]+nc+no)
                for i in np.where(abs(x_ov_aa) > 0.1)[0]:
                    print(f'    OV(0) {fnc+self.pscsfov_a_i[i]+1+nc:3d}a -> {fnc+self.pscsfov_a_a[i]+1+nc+no:3d}a'
                          + f'    c_i: {x_ov_aa[i]:8.5f}    Per: {100*x_ov_aa[i]**2:5.2f}%')
                    if nstate in out_cube:
                        orbital.append(fnc+self.pscsfov_a_i[i]+nc)
                        orbital.append(fnc+self.pscsfov_a_a[i]+nc+no)
                for i in np.where(abs(x_co_bb) > 0.1)[0]:
                    print(f'    CO(0) {fnc+self.pscsfco_b_i[i]+1:3d}b -> {fnc+self.pscsfco_b_a[i]+1+nc:3d}b'
                          + f'    c_i: {x_co_bb[i]:8.5f}    Per: {100*x_co_bb[i]**2:5.2f}%')
                    if nstate in out_cube:
                        orbital.append(fnc+self.pscsfco_b_i[i])
                        orbital.append(fnc+self.pscsfco_b_a[i]+nc)
                for i in np.where(abs(x_cv_bb) > 0.1)[0]:
                    print(f'    CV(1) {fnc+self.pscsfcv_b_i[i]+1:3d}b -> {fnc+self.pscsfcv_b_a[i]+1+nc+no:3d}b'
                          + f'    c_i: {x_cv_bb[i]:8.5f}    Per: {100*x_cv_bb[i]**2:5.2f}%')
                    if nstate in out_cube:
                        orbital.append(fnc+self.pscsfcv_b_i[i])
                        orbital.append(fnc+self.pscsfcv_b_a[i]+nc+no)
            else:
                for o, v in zip(*np.where(abs(x_cv_aa) > 0.1)):
                    print(f'    CV(0) {fnc+o + 1:3d}a -> {fnc+v + 1 + nc+no:3d}a'
                          + f'    c_i: {x_cv_aa[o, v]:8.5f}    Per: {100*x_cv_aa[o, v]**2:5.2f}%')
                    if nstate in out_cube:
                        orbital.append(fnc+o)
                        orbital.append(fnc+v+nc+no)
                for o, v in zip(*np.where(abs(x_ov_aa) > 0.1)):
                    print(f'    OV(0) {fnc+nc+o + 1:3d}a -> {fnc+v + 1+nc+no:3d}a'
                          + f'     c_i: {x_ov_aa[o, v]:8.5f}    Per: {100*x_ov_aa[o, v]**2:5.2f}%')
                    if nstate in out_cube:
                        orbital.append(fnc+nc+o)
                        orbital.append(fnc+v+nc+no)
                for o, v in zip(*np.where(abs(x_co_bb) > 0.1)):
                    print(f'    CO(0) {fnc+o + 1:3d}b -> {fnc+v + 1+nc:3d}b'
                          + f'    c_i: {x_co_bb[o, v]:8.5f}    Per: {100*x_co_bb[o, v]**2:5.2f}%')
                    if nstate in out_cube:
                        orbital.append(fnc+o)
                        orbital.append(fnc+v+nc)
                for o, v in zip(*np.where(abs(x_cv_bb) > 0.1)):
                    print(f'    CV(1) {fnc+o + 1:3d}b -> {fnc+v + 1 + nc+no:3d}b'
                          + f'    c_i: {x_cv_bb[o, v]:8.5f}    Per: {100*x_cv_bb[o, v]**2:5.2f}%')
                    if nstate in out_cube:
                        orbital.append(fnc+o)
                        orbital.append(fnc+v+nc+no)
        # orbital = np.unique(np.array(orbital))  # only include excited orbital
        orbital = np.arange(np.min(orbital), np.max(orbital)+1, dtype=np.int64)
        for i in orbital:
            # here use self.mf.mo_coeff is for better construct molecular orbital
            if self.readinfo:
                print('read information do not output cube file')
            else:
                if self.spinadapt:
                    cubegen.orbital(mol, out_filename + str(i + 1) + '.cube', self.mf.mo_coeff[:, i])
                else:
                    cubegen.orbital(self.mol, out_filename + str(i + 1) + 'alpha.cube', self.mf.mo_coeff[0, :, i])
                    cubegen.orbital(self.mol, out_filename + str(i + 1) + 'beta.cube', self.mf.mo_coeff[1, :, i])
        # export molden file
        c_loc_orth = lo.orth.orth_ao(self.mol)
        molden.from_mo(self.mol, out_filename+'.molden', c_loc_orth)


    # def osc_str(self):
    #     # oscillator strength
    #     # f = 2/3 \omega_{0\nu} | \langle 0|\sum_s r_s |\nu\rangle |^2  # length form
    #     # f = 2/3 \omega_{0\nu}^{-1} | \langle 0|\sum_s \nabla_s |\nu\rangle |^2  # velocity form
    #     orbo_a = self.mo_coeff[:, self.occidx_a]
    #     orbv_a = self.mo_coeff[:, self.viridx_a]
    #     orbo_b = self.mo_coeff[:, self.occidx_b]
    #     orbv_b = self.mo_coeff[:, self.viridx_b]
    #     omega = self.e[:self.nstates]
    #     # length form oscillator strength
    #     dipole_ao = self.mol.intor_symmetric("int1e_r", comp=3)  # dipole moment, comp=3 is 3 axis
    #     dipole_mo_a = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo_a, orbv_a, optimize=True)
    #     dipole_mo_b = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo_b, orbv_b, optimize=True)
    #     if self.Emax:
    #         dipole_mocv_a = dipole_mo_a[:, self.pscsfcv_i, self.pscsfcv_a]
    #         dipole_moov_a = dipole_mo_a[:, self.nc + self.pscsfov_a_i, self.pscsfov_a_a]
    #         dipole_moco_b = dipole_mo_b[:, self.pscsfco_b_i, self.pscsfco_b_a]
    #         dipole_mocv_b = dipole_mo_b[:, self.pscsfcv_i, self.no + self.pscsfcv_a]
    #     else:
    #         dipole_mocv_a = dipole_mo_a[:, :self.nc, :].reshape(3, -1)
    #         dipole_moov_a = dipole_mo_a[:, self.nc:, :].reshape(3, -1)
    #         dipole_moco_b = dipole_mo_b[:, :, :self.no].reshape(3, -1)
    #         dipole_mocv_b = dipole_mo_b[:, :, self.no:].reshape(3, -1)
    #     trans_dipcv_a = np.einsum('xi,yi->yx', dipole_mocv_a, self.xycv_a, optimize=True)
    #     trans_dipov_a = np.einsum('xi,yi->yx', dipole_moov_a, self.xyov_a, optimize=True)
    #     trans_dipco_b = np.einsum('xi,yi->yx', dipole_moco_b, self.xyco_b, optimize=True)
    #     trans_dipcv_b = np.einsum('xi,yi->yx', dipole_mocv_b, self.xycv_b, optimize=True)
    #     trans_dip = trans_dipcv_a + trans_dipov_a + trans_dipco_b + trans_dipcv_b
    #     f = 2. / 3. * np.einsum('s,sx,sx->s', omega, trans_dip, trans_dip, optimize=True)
    #     # np.save("osc_str_xstda.npy", f)
    #     return f
    #
    # def rot_str(self):
    #     # oscillator strength
    #     # f = 2/3 \omega_{0\nu} | \langle 0|\sum_s r_s |\nu\rangle |^2  # length form
    #     # f = 2/3 \omega_{0\nu}^{-1} | \langle 0|\sum_s \nabla_s |\nu\rangle |^2  # velocity form
    #     orbo_a = self.mo_coeff[:, self.occidx_a]
    #     orbv_a = self.mo_coeff[:, self.viridx_a]
    #     orbo_b = self.mo_coeff[:, self.occidx_b]
    #     orbv_b = self.mo_coeff[:, self.viridx_b]
    #     omega = self.e[:self.nstates]
    #     dip_ele_ao = self.mol.intor('int1e_ipovlp', comp=3, hermi=2)  # transition electric dipole moment
    #     dip_ele_mo_a = np.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo_a, orbv_a, optimize=True)
    #     dip_ele_mo_b = np.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo_b, orbv_b, optimize=True)
    #     dip_meg_ao = self.mol.intor('int1e_cg_irxp', comp=3, hermi=2)  # transition magnetic dipole moment
    #     dip_meg_mo_a = np.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo_a, orbv_a, optimize=True)
    #     dip_meg_mo_b = np.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo_b, orbv_b, optimize=True)
    #     if self.Emax:
    #         dip_ele_mocv_a = dip_ele_mo_a[:, self.pscsfcv_i, self.pscsfcv_a]
    #         dip_ele_moov_a = dip_ele_mo_a[:, self.nc + self.pscsfov_a_i, self.pscsfov_a_a]
    #         dip_ele_moco_b = dip_ele_mo_b[:, self.pscsfco_b_i, self.pscsfco_b_a]
    #         dip_ele_mocv_b = dip_ele_mo_b[:, self.pscsfcv_i, self.no + self.pscsfcv_a]
    #         dip_meg_mocv_a = dip_meg_mo_a[:, self.pscsfcv_i, self.pscsfcv_a]
    #         dip_meg_moov_a = dip_meg_mo_a[:, self.nc + self.pscsfov_a_i, self.pscsfov_a_a]
    #         dip_meg_moco_b = dip_meg_mo_b[:, self.pscsfco_b_i, self.pscsfco_b_a]
    #         dip_meg_mocv_b = dip_meg_mo_b[:, self.pscsfcv_i, self.no + self.pscsfcv_a]
    #     else:
    #         dip_ele_mocv_a = dip_ele_mo_a[:, :self.nc, :].reshape(3, -1)
    #         dip_ele_moov_a = dip_ele_mo_a[:, self.nc:, :].reshape(3, -1)
    #         dip_ele_moco_b = dip_ele_mo_b[:, :, :self.no].reshape(3, -1)
    #         dip_ele_mocv_b = dip_ele_mo_b[:, :, self.no:].reshape(3, -1)
    #         dip_meg_mocv_a = dip_meg_mo_a[:, :self.nc, :].reshape(3, -1)
    #         dip_meg_moov_a = dip_meg_mo_a[:, self.nc:, :].reshape(3, -1)
    #         dip_meg_moco_b = dip_meg_mo_b[:, :, :self.no].reshape(3, -1)
    #         dip_meg_mocv_b = dip_meg_mo_b[:, :, self.no:].reshape(3, -1)
    #     trans_ele_dipcv_a = np.einsum('xi,yi->yx', dip_ele_mocv_a, self.xycv_a, optimize=True)
    #     trans_ele_dipov_a = np.einsum('xi,yi->yx', dip_ele_moov_a, self.xyov_a, optimize=True)
    #     trans_ele_dipco_b = np.einsum('xi,yi->yx', dip_ele_moco_b, self.xyco_b, optimize=True)
    #     trans_ele_dipcv_b = np.einsum('xi,yi->yx', dip_ele_mocv_b, self.xycv_b, optimize=True)
    #     trans_ele_dip = -(trans_ele_dipcv_a + trans_ele_dipov_a + trans_ele_dipco_b + trans_ele_dipcv_b)
    #     trans_meg_dipcv_a = np.einsum('xi,yi->yx', dip_meg_mocv_a, self.xycv_a, optimize=True)
    #     trans_meg_dipov_a = np.einsum('xi,yi->yx', dip_meg_moov_a, self.xyov_a, optimize=True)
    #     trans_meg_dipco_b = np.einsum('xi,yi->yx', dip_meg_moco_b, self.xyco_b, optimize=True)
    #     trans_meg_dipcv_b = np.einsum('xi,yi->yx', dip_meg_mocv_b, self.xycv_b, optimize=True)
    #     trans_meg_dip = 0.5 * (trans_meg_dipcv_a + trans_meg_dipov_a + trans_meg_dipco_b + trans_meg_dipcv_b)
    #     # in Gaussian and ORCA, do not multiply constant
    #     # f = 1./unit.c * np.einsum('s,sx,sx->s', 1./omega, trans_ele_dip, trans_meg_dip)
    #     f = np.einsum('s,sx,sx->s', 1. / omega, trans_ele_dip, trans_meg_dip, optimize=True)
    #     f = f / unit.cgs2au  # transform atom unit to cgs unit
    #     # np.save("rot_str_stda.npy", f)
    #     return f
    #
    # def analyze(self, out_filename='XsTDA'):
    #     nc = self.nc
    #     nv = self.nv
    #     no = self.no
    #     fnc = self.frozen_nc
    #     if self.Emax:
    #         self.v = utils.so2st(self.v[:, :self.nstates], lcva=self.lcva, lova=self.lova, lcob=self.lcob,
    #                              lcvb=self.lcvb)
    #     else:
    #         self.v = utils.so2st(self.v[:, :self.nstates], nc, no, nv)
    #     out_cube = [0, np.argmax(self.os)]  # only output 1st excited state and max os excited state orbital
    #     orbital = []  # record output orbital number
    #     print("=" * 50)
    #     for nstate in range(self.nstates):
    #         value = self.v[:, nstate]
    #         x_cv_aa = value[:self.lcva]
    #         x_ov_aa = value[self.lcva:self.lcva + self.lova]
    #         x_co_bb = value[self.lcva + self.lova:self.lcva + self.lova + self.lcob]
    #         x_cv_bb = value[self.lcva + self.lova + self.lcob:]
    #         # print(f'Excited state {nstate + 1} {self.e[nstate] * unit.ha2eV:10.5f} eV')
    #
    #         print(
    #             f'D{nstate + 1}' + r"    w:" + f'{self.e[nstate] * unit.ha2eV:10.4f} eV'
    #             + r"    d<S^2>:" + f'{self.dS2[nstate]:8.4f}'
    #             + r"    f:" + f'{self.os[nstate]:8.4f}'
    #         )
    #         print(
    #             r"CV(0): " + f"{np.sum(x_cv_aa ** 2 * 100):5.2f}%"
    #             + r"    OV(0): " + f"{np.sum(x_ov_aa ** 2 * 100):5.2f}%"
    #             + r"    CO(0): " + f"{np.sum(x_co_bb ** 2 * 100):5.2f}%"
    #             + r"    CV(1): " + f"{np.sum(x_cv_bb ** 2 * 100):5.2f}%"
    #         )
    #
    #         # out_excittype = np.argmax((np.max(abs(x_cv_aa)), np.max(abs(x_ov_aa)), np.max(abs(x_co_bb)), np.max(abs(x_cv_bb))))
    #         if self.Emax:
    #             for i in np.where(abs(x_cv_aa) > 0.1)[0]:
    #                 print(f'    CV(0) {fnc + self.pscsfcv_i[i] + 1:3d} -> {fnc + self.pscsfcv_a[i] + 1 + nc + no:3d}'
    #                       + f'    c_i: {x_cv_aa[i]:8.5f}    Per: {100 * x_cv_aa[i] ** 2:5.2f}%')
    #                 # if nstate in out_cube and out_excittype == 0 and i == np.argmax(abs(x_cv_aa)):
    #                 if nstate in out_cube:
    #                     orbital.append(fnc + self.pscsfcv_i[i])
    #                     orbital.append(fnc + self.pscsfcv_a[i] + nc + no)
    #             for i in np.where(abs(x_ov_aa) > 0.1)[0]:
    #                 print(
    #                     f'    OV(0) {fnc + self.pscsfov_a_i[i] + 1 + nc:3d} -> {fnc + self.pscsfov_a_a[i] + 1 + nc + no:3d}'
    #                     + f'    c_i: {x_ov_aa[i]:8.5f}    Per: {100 * x_ov_aa[i] ** 2:5.2f}%')
    #                 if nstate in out_cube:
    #                     orbital.append(fnc + self.pscsfov_a_i[i] + nc)
    #                     orbital.append(fnc + self.pscsfov_a_a[i] + nc + no)
    #             for i in np.where(abs(x_co_bb) > 0.1)[0]:
    #                 print(f'    CO(0) {fnc + self.pscsfco_b_i[i] + 1:3d} -> {fnc + self.pscsfco_b_a[i] + 1 + nc:3d}'
    #                       + f'    c_i: {x_co_bb[i]:8.5f}    Per: {100 * x_co_bb[i] ** 2:5.2f}%')
    #                 if nstate in out_cube:
    #                     orbital.append(fnc + self.pscsfco_b_i[i])
    #                     orbital.append(fnc + self.pscsfco_b_a[i] + nc)
    #             for i in np.where(abs(x_cv_bb) > 0.1)[0]:
    #                 print(f'    CV(1) {fnc + self.pscsfcv_i[i] + 1:3d} -> {fnc + self.pscsfcv_a[i] + 1 + nc + no:3d}'
    #                       + f'    c_i: {x_cv_bb[i]:8.5f}    Per: {100 * x_cv_bb[i] ** 2:5.2f}%')
    #                 if nstate in out_cube:
    #                     orbital.append(fnc + self.pscsfcv_i[i])
    #                     orbital.append(fnc + self.pscsfcv_a[i] + nc + no)
    #         else:
    #             for o, v in zip(*np.where(abs(x_cv_aa) > 0.1)):
    #                 print(f'    CV(0) {fnc + o + 1:3d} -> {fnc + v + 1 + nc + no:3d}'
    #                       + f'    c_i: {x_cv_aa[o, v]:8.5f}    Per: {100 * x_cv_aa[o, v] ** 2:5.2f}%')
    #                 if nstate in out_cube:
    #                     orbital.append(fnc + o)
    #                     orbital.append(fnc + v + nc + no)
    #             for o, v in zip(*np.where(abs(x_ov_aa) > 0.1)):
    #                 print(f'    OV(0) {fnc + nc + o + 1:3d} -> {fnc + v + 1 + nc + no:3d}'
    #                       + f'     c_i: {x_ov_aa[o, v]:8.5f}    Per: {100 * x_ov_aa[o, v] ** 2:5.2f}%')
    #                 if nstate in out_cube:
    #                     orbital.append(fnc + nc + o)
    #                     orbital.append(fnc + v + nc + no)
    #             for o, v in zip(*np.where(abs(x_co_bb) > 0.1)):
    #                 print(f'    CO(0) {fnc + o + 1:3d} -> {fnc + v + 1 + nc:3d}'
    #                       + f'    c_i: {x_co_bb[o, v]:8.5f}    Per: {100 * x_co_bb[o, v] ** 2:5.2f}%')
    #                 if nstate in out_cube:
    #                     orbital.append(fnc + o)
    #                     orbital.append(fnc + v + nc)
    #             for o, v in zip(*np.where(abs(x_cv_bb) > 0.1)):
    #                 print(f'    CV(1) {fnc + o + 1:3d} -> {fnc + v + 1 + nc + no:3d}'
    #                       + f'    c_i: {x_cv_bb[o, v]:8.5f}    Per: {100 * x_cv_bb[o, v] ** 2:5.2f}%')
    #                 if nstate in out_cube:
    #                     orbital.append(fnc + o)
    #                     orbital.append(fnc + v + nc + no)
    #     # orbital = np.unique(np.array(orbital))  # only include excited orbital
    #     orbital = np.arange(np.min(orbital), np.max(orbital) + 1, dtype=np.int64)
    #     for i in orbital:
    #         cubegen.orbital(mol, out_filename + str(i + 1) + '.cube', self.mf.mo_coeff[:, i])
    #     # export molden file
    #     c_loc_orth = lo.orth.orth_ao(mol)
    #     molden.from_mo(mol, out_filename + '.molden', c_loc_orth)


if __name__ == "__main__":
    mol = gto.M(
        # atom=atom.n2_,  # unit is Angstrom
        atom=atom.ch3,
        # atom=atom.ch2 xo_vacuum,
        # atom=atom.ttm_toluene,
        # atom=atom.bispytm_toluene,
        # atom=atom.ttm3ncz_toluene,
        # atom=atom.mttm2_toluene,
        # atom=atom.hhcrqpp2,
        # atom=atom.ptm3ncz_cyclohexane,
        # atom=atom.g3ttm_toluene,
        unit="A",
        # unit="B",  # https://doi.org/10.1016/j.comptc.2014.02.023 use bohr
        basis='aug-cc-pvtz',
        # basis='cc-pvtz',
        # basis='6-31g**',
        # cart = True,
        spin=1,
        charge=0,
        # symmetry=True,
        verbose=4
    )
    # # path = '/home/whb/Documents/TDDFT/orcabasis/'
    # path = './'
    # # bse.convert_formatted_basis_file(path+'sto-3g.bas', path+'sto-3g.nw')
    # # bse.convert_formatted_basis_file(path+'tzvp.bas', path+'tzvp.nw')
    # with open(path + "sto-3g.nw", "r") as f:
    # # with open(path + "tzvp.nw", "r") as f:
    #     basis = f.read()
    # mol.basis = basis
    # mol.build()
    
    # # add solvents
    # t_dft0 = time.time()
    # # mf = dft.ROKS(mol).SMD()
    # mf = dft.UKS(mol).SMD()
    # # mf = dft.ROKS(mol).PCM()
    # # mf.with_solvent.method = 'COSMO'  # C-PCM, SS(V)PE, COSMO, IEF-PCM
    # # in https://gaussian.com/scrf/ solvents entry, give different eps for different solvents
    # # mf.with_solvent.eps = 2.0165  # for Cyclohexane 环己烷
    # mf.with_solvent.eps = 2.3741  # for toluene 甲苯
    # # mf.with_solvent.eps = 35.688  # for Acetonitrile 乙腈
    
    t_dft0 = time.time()
    mf = dft.ROKS(mol)
    # mf = dft.UKS(mol)
    mf.conv_tol = 1e-8  # same with orca tightscf criterion
    mf.conv_tol_grad = 1e-5  # same with orca tightscf criterion
    mf.max_cycle = 200
    # xc = 'svwn'
    # xc = 'blyp'
    # xc = 'tpssh'
    # xc = 'wb97xd'
    # xc = 'b3lyp'
    # xc = 'pbe0'
    # xc = 'tpss0'  # ax=0.25, same with pbe0
    # xc = 'pbe38'
    xc = '0.50*HF + 0.50*B88 + GGA_C_LYP'  # BHHLYP
    # xc = 'hf'
    mf.xc = xc
    xc = 'bhhlyp'
    # mf.grids.level = 9
    # mf.conv_check = False  # here must set True, default is True. Or cause error
    if 'Cr' in mol.elements:  # for my test, this only for hhcrqpp2
        mf.level_shift = 0.6  # when add level shift, mo_energy will change, do not same with Fock diagonal element
    mf.kernel()
    # mo1, mo2 = mf.stability()  # stable test
    # mf.kernel(mo_coeff=mo1)  # if do not stable, use other wave function as initial guess
    mf.analyze()
    t_dft1 = time.time()
    save = False
    if save:
        mf.dump_chk("./scf.chk")
        t_fock0 = time.time()
        h1e = mf.get_hcore()
        dm = mf.make_rdm1()
        vhf = mf.get_veff(mol, dm)
        if getattr(mf, 'with_solvent', None) is not None:
            vhf += vhf.v_solvent
        t_fock1 = time.time()
        print("contract Fock matrix use          {:8.4f} s".format(t_fock1 - t_fock0))
        scf_dic = {'h1e': h1e, 'dm':dm, 'vhf':vhf}
        np.save('scf.npy', scf_dic)
    print("dft use {} s".format(t_dft1 - t_dft0))
    print('=' * 50)

    # os.environ["OMP_NUM_THREADS"] = "1"  # test one core time-consuming
    monitor = utils.MemoryMonitor()
    read = False
    if read:
        mol, mf = scf.chkfile.load_scf('scf.chk')
        mo_occ = mf['mo_occ']
        mo_coeff = mf['mo_coeff']
        mo_energy = mf['mo_energy']
        scf_info = np.load('scf.npy', allow_pickle=True).item()
        h1e = scf_info['h1e']
        dm = scf_info['dm']
        vhf = scf_info['vhf']
    osstda = OSsTDA(mol, mf)
    # if isinstance(mf, dft.roks.ROKS):
    #     osstda.spinadapt = True
    #     print('spinadapt is True')
    # else:
    #     osstda.spinadapt = False
    #     print('spinadapt is False')
    osstda.spinadapt = True
    osstda.nstates = 12
    osstda.cas = True
    osstda.Emax = 10.0
    osstda.union = True
    osstda.correct = False
    osstda.paramtype = 'os'
    osstda.readinfo = read
    if read:
        # osstda.info(mo_occ=mo_occ, mo_energy=mo_energy)
        e_eV, os, rs, v = osstda.kernel(mo_occ, mo_coeff, mo_energy, h1e, dm, vhf, hyb=0.25)
    else:
        # osstda.info()
        e_eV, os, rs, v = osstda.kernel()
    # osstda.analyze(out_filename='toluene-XsTDA-')
    monitor.join()
    peak = max(monitor.memory_buffer) / 1e6
    print(f"Peak memory usage: {peak:.2f}MB")
