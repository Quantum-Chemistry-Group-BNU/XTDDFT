#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import numpy as np
from typing import List
from pyscf import gto, scf, dft, lib, __config__
from pyscf.lib import logger
from pyscf.dft import numint2c
from pyscf.grad import rohf as rohf_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.grad import tduks as tduks_grad

from XTDA import XTDA
from utils import atom, unit, utils


def excited_energy(atom, spec, state, xc='b3lyp'):
    """excited energy = mf.e_tot + td.e[state-1], unit of td.e is Hartree"""
    mol = gto.M(atom=atom, verbose=0, **spec)
    mf = dft.ROKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.grids.level = 5
    mf.kernel()

    # Request +2 extra roots, make davidson itersion more stable
    td = XTDA(mol, mf, nstates=max(state, 1) + 2, basis='orbital', use_Davidson=True)
    td.kernel()
    return mf.e_tot + td.e[state-1]


def fd_gradient(atoms, state, xc='b3lyp', h=1e-5):
    """finite difference truncate to second order, (-3E0+4E+ - E++)/(2h), unit of h is Angstrom"""
    assert isinstance(atoms, List) and (len(atoms[0]) == 2)
    print('***** The molecular coordinates of the finite difference input are in angstroms *****')
    natm = len(atoms)
    spec = dict(charge=0, spin=2, basis='6-31g')
    g = np.zeros((natm, 3))
    h_au = h * unit.au2ang

    E0 = excited_energy(atoms, spec, state, xc)
    for i in range(natm):
        for d in range(3):
            atoms_h = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_h[i][1][d] += h
            Eh = excited_energy(atoms_h, spec, state, xc)
            atoms_2h = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_2h[i][1][d] += 2 * h
            Ehh = excited_energy(atoms_2h, spec, state, xc)
            g[i, d] = (-3*E0 + 4*Eh - Ehh) / (2*h_au)  # energy unit is Hartree/bohr
    return g


# dmov, dmoo in AO-representation
# Note spin-trace is applied for fxc, kxc
#TODO: to include the response of grids
def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True,
                        with_kxc=True, max_memory=2000):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    # Below two line have changed. For a ROKS reference mf.mo_coeff is 2-D 
    # (restricted); read the doubled pseudo-UKS arrays the SF_TDA_up solver
    # stores. For UKS these equal mf's.
    mo_coeff = getattr(td_grad.base, 'mo_coeff', mf.mo_coeff)
    mo_occ = getattr(td_grad.base, 'mo_occ', mf.mo_occ)
    nao = mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # dmvo ~ reduce(numpy.dot, (orbv, Xai, orbo.T))
    dmvo = [(dmvo[0] + dmvo[0].T) * .5, # because K_{ia,jb} == K_{ia,jb}
            (dmvo[1] + dmvo[1].T) * .5]

    f1vo = np.zeros((2,4,nao,nao))
    deriv = 2
    if dmoo is not None:
        f1oo = np.zeros((2,4,nao,nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = np.zeros((2,4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = np.zeros((2,4,nao,nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = tdrks_grad._lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = tdrks_grad._gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = tdrks_grad._mgga_eval_mat_, 2
        logger.warn(td_grad, 'TDUKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    if mf.do_nlc():
        raise NotImplementedError("TDDFT gradient with NLC contribution is not supported yet. "
                                  "Please set exclude_nlc field of tdscf object to True, "
                                  "which will turn off NLC contribution in the whole TDDFT calculation.")

    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        if xctype == 'LDA':
            ao0 = ao[0]
        else:
            ao0 = ao
        rho = (ni.eval_rho2(mol, ao0, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl=False),
               ni.eval_rho2(mol, ao0, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl=False))
        vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]

        rho1 = np.asarray((
            ni.eval_rho(mol, ao0, dmvo[0], mask, xctype, hermi=1, with_lapl=False),
            ni.eval_rho(mol, ao0, dmvo[1], mask, xctype, hermi=1, with_lapl=False)))
        if xctype == 'LDA':
            rho1 = rho1[:,np.newaxis]
        wv = np.einsum('axg,axbyg,g->byg', rho1, fxc, weight)
        fmat_(mol, f1vo[0], ao, wv[0], mask, shls_slice, ao_loc)
        fmat_(mol, f1vo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None:
            rho2 = np.asarray((
                ni.eval_rho(mol, ao0, dmoo[0], mask, xctype, hermi=1, with_lapl=False),
                ni.eval_rho(mol, ao0, dmoo[1], mask, xctype, hermi=1, with_lapl=False)))
            if xctype == 'LDA':
                rho2 = rho2[:,np.newaxis]
            wv = np.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_vxc:
            wv = vxc * weight
            fmat_(mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_kxc:
            wv = np.einsum('axg,byg,axbyczg,g->czg', rho1, rho1, kxc, weight)
            fmat_(mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[:,1:] *= -1
    if f1oo is not None: f1oo[:,1:] *= -1
    if v1ao is not None: v1ao[:,1:] *= -1
    if k1ao is not None: k1ao[:,1:] *= -1
    return f1vo, f1oo, v1ao, k1ao


def grad_elec(td, atmlst=None, max_memory=2000,
              verbose=logger.INFO):
    """ROKS spin-conserving XTDA excitation-energy nuclear derivative."""
    log = logger.new_logger(td, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mf = td.base._scf
    mol = td.mol
    mo_energy = np.array((mf.mo_energy, mf.mo_energy))
    mo_coeff = np.array((mf.mo_coeff, mf.mo_coeff))
    mo_occ = np.zeros((2,len(mf.mo_coeff)))
    mo_occ[0][np.where(mf.mo_occ>=1)[0]]=1
    mo_occ[1][np.where(mf.mo_occ>=2)[0]]=1
    nao = mo_coeff[0].shape[0]
    occidx_a = np.where(mo_occ[0] == 1)[0]
    viridx_a = np.where(mo_occ[0] == 0)[0]
    occidx_b = np.where(mo_occ[1] == 1)[0]
    viridx_b = np.where(mo_occ[1] == 0)[0]
    nc = len(occidx_b)
    nv = len(viridx_a)
    no = len(occidx_a) - len(occidx_b)
    ni = nc+no  # convenient use, alpha occ
    orbv_a = mo_coeff[0][:, ni:]
    orbo_a = mo_coeff[0][:, :ni]
    orbv_b = mo_coeff[1][:, nc:]
    orbo_b = mo_coeff[1][:, :nc]
    v = td.v[:, td.state-1]
    v_cva = v[:nc*nv].reshape(nc, nv)
    v_ova = v[nc*nv:(nc+no)*nv].reshape(no, nv)
    v_cob = v[(nc+no)*nv:(nc+no)*nv+nc*no].reshape(nc, no)
    v_cvb = v[(nc+no)*nv+nc*no:].reshape(nc, nv)
    va = np.vstack((v_cva, v_ova)).T
    vb = np.hstack((v_cob, v_cvb)).T

    # 1. internel variable
    dvva = lib.einsum('ai,bi->ab', va, va)  # T_{ab}^{\alpha}
    dvvb = lib.einsum('ai,bi->ab', vb, vb)
    dooa = -lib.einsum('ai,aj->ij', va, va)  # T_{ij}^{\alpha}
    doob = -lib.einsum('ai,aj->ij', vb, vb)
    dmza = orbv_a @ dvva @ orbv_a.T + orbo_a @ dooa @ orbo_a.T  # T_{\mu\nu}^{\alpha}
    dmzb = orbv_b @ dvvb @ orbv_b.T + orbo_b @ doob @ orbo_b.T  # T_{\mu\nu}^{\beta}
    dmxa = orbv_a @ va @ orbo_a.T  # X_{\mu\nu}^{\alpha}
    dmxb = orbv_b @ vb @ orbo_b.T  # X_{\mu\nu}^{\beta}
    # correct coefficient
    s = mol.spin / 2
    c1 = 1 - np.sqrt((s + 1) / s) + 1 / (2 * s)
    c2 = -1 + np.sqrt((s + 1) / s) + 1 / (2 * s)
    c3 = -1 / (2 * s)
    # correct dm
    va_cc = va[:, :nc]
    vb_cc = vb[no:, :]
    dmcc = lib.einsum('ai,aj->ij', va_cc, va_cc) * c2
    dmcc += lib.einsum('ai,aj->ij', va_cc, vb_cc) * c3
    dmcc += lib.einsum('ai,aj->ij', vb_cc, va_cc) * c3
    dmcc += lib.einsum('ai,aj->ij', vb_cc, vb_cc) * c1
    dmvv = lib.einsum('ai,bi->ab', va_cc, va_cc) * c1
    dmvv += lib.einsum('ai,bi->ab', va_cc, vb_cc) * c3
    dmvv += lib.einsum('ai,bi->ab', vb_cc, va_cc) * c3
    dmvv += lib.einsum('ai,bi->ab', vb_cc, vb_cc) * c2
    dmcorrao = orbo_b @ dmcc @ orbo_b.T + orbv_a @ dmvv @ orbv_a.T  # dm correct ao
    scorrao = orbo_a[:, nc:ni] @ (np.eye(no)/2) @ orbo_a[:, nc:ni].T  # s correct ao
    # ROKS fock 
    dm = mf.make_rdm1()
    vhf = mf.get_veff(mol, dm)
    h1e = mf.get_hcore()
    focka = h1e + vhf[0]
    fockb = h1e + vhf[1]
    fockamo = mo_coeff[0].T @ focka @ mo_coeff[0]
    fockbmo = mo_coeff[1].T @ fockb @ mo_coeff[1]
    # symmetrized Fock blocks, perpare for solve Z-vector equation
    fockacc = (fockamo[:nc, :nc] + fockamo[:nc, :nc].T) / 2
    fockaoc = (fockamo[nc:ni, :nc] + fockamo[:nc, nc:ni].T) / 2
    fockavc = (fockamo[ni:, :nc] + fockamo[:nc, ni:].T) / 2
    fockavv = (fockamo[ni:, ni:] + fockamo[ni:, ni:].T) / 2
    fockaoo = (fockamo[nc:ni, nc:ni] + fockamo[nc:ni, nc:ni].T) / 2
    fockbcc = (fockbmo[:nc, :nc] + fockbmo[:nc, :nc].T) / 2
    fockbvc = (fockbmo[ni:, :nc] + fockbmo[:nc, ni:].T) / 2
    fockbvv = (fockbmo[ni:, ni:] + fockbmo[ni:, ni:].T) / 2
    fockbvo = (fockbmo[ni:, nc:ni] + fockbmo[nc:ni, ni:].T) / 2
    fockboo = (fockbmo[nc:ni, nc:ni] + fockbmo[nc:ni, nc:ni].T) / 2

    # 2. functional derivative, include derivative respect to mo_coeff and coordinate
    tdro = _RO2U(td, mf)
    ni_ = mf._numint
    omega, alpha, hyb = ni_.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # f1vo: f^{xc}[X], f1oo: f^{xc}[T], vxc1: v^{xc}[\rho], k1ao: g^{xc}[X,X] 
    # and their derivative
    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(
        tdro, mf.xc, (dmxa, dmxb), (dmza, dmzb), True, True, max_memory
    )

    # 3.1 construct Q matrix
    if hyb > 0:
        dm = (dmza, dmzb, dmxa, dmxb,)
        # TODO(WHB): dmza and dmzb is hermitian but dmxa and dmxb is not.
        #  how it influence time usage.
        vj, vk = mf.get_jk(mol, dm, hermi=0)  # g_{\mu\nu}^{\sigma}[T]
        vk *= hyb
        veff0doo = vj[0] + vj[1] - vk[:2] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = (orbv_a.T @ veff0doo[0] @ orbo_a) * 2
        wvob = (orbv_b.T @ veff0doo[1] @ orbo_b) * 2
        veff0moa = mo_coeff[0].T @ (vj[2] + vj[3] - vk[2] + f1vo[0, 0]) @ mo_coeff[0]
        veff0mob = mo_coeff[1].T @ (vj[2] + vj[3] - vk[3] + f1vo[1, 0]) @ mo_coeff[1]
        # TODO(WHB): check repeat calculate
        wvoa += lib.einsum('ba,bi->ai', veff0moa[ni:, ni:], va) * 2
        wvoa += lib.einsum('ij,aj->ai', dooa, fockamo[ni:, :ni]) * 2
        wvoa -= lib.einsum('ij,aj->ai', veff0moa[:ni, :ni], va) * 2
        wvoa -= lib.einsum('ab,ib->ai', dvva, fockamo[:ni, ni:]) * 2
        wvob += lib.einsum('ba,bi->ai', veff0mob[nc:, nc:], vb) * 2
        wvob += lib.einsum('ij,aj->ai', doob, fockbmo[nc:, :nc]) * 2
        wvob -= lib.einsum('ij,aj->ai', veff0mob[:nc, :nc], vb) * 2
        wvob -= lib.einsum('ab,ib->ai', dvvb, fockbmo[:nc, nc:]) * 2
    else:
        dm = (dmza, dmzb, dmxa, dmxb,)
        vj= mf.get_j(mol, dm, hermi=0)  # g_{\mu\nu}^{\sigma}[T]
        veff0doo = vj[0] + vj[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = (orbv_a.T @ veff0doo[0] @ orbo_a) * 2
        wvob = (orbv_b.T @ veff0doo[1] @ orbo_b) * 2
        veff0moa = mo_coeff[0].T @ (vj[2] + vj[3] + f1vo[0, 0]) @ mo_coeff[0]
        veff0mob = mo_coeff[1].T @ (vj[2] + vj[3] + f1vo[1, 0]) @ mo_coeff[1]
        wvoa += lib.einsum('ba,bi->ai', veff0moa[ni:, ni:], va) * 2
        wvoa += lib.einsum('ij,aj->ai', dooa, fockamo[ni:, :ni]) * 2
        wvoa -= lib.einsum('ij,aj->ai', veff0moa[:ni, :ni], va) * 2
        wvoa -= lib.einsum('ab,ib->ai', dvva, fockamo[:ni, ni:]) * 2
        wvob += lib.einsum('ba,bi->ai', veff0mob[nc:, nc:], vb) * 2
        wvob += lib.einsum('ij,aj->ai', doob, fockbmo[nc:, :nc]) * 2
        wvob -= lib.einsum('ij,aj->ai', veff0mob[:nc, :nc], vb) * 2
        wvob -= lib.einsum('ab,ib->ai', dvvb, fockbmo[:nc, nc:]) * 2

    wcca = (orbo_a.T @ veff0doo[0] @ orbo_a) * 2
    wcca += lib.einsum('ik,jk->ij', dooa, fockamo[:ni, :ni]) * 2
    wcca += lib.einsum('aj,ai->ij', veff0moa[ni:, :ni], va) * 2
    wvvb = lib.einsum('ac,bc->ab', dvvb, fockbmo[nc:, nc:]) * 2
    wvvb += lib.einsum('bi,ai->ab', veff0mob[nc:, :nc], vb) * 2
    wvc = wvoa[:, :nc] + wvob[no:, :]  # (Q_{ia} - Q_{ai})
    wvo = wvoa[:, nc:] - (wvvb - wvvb.T)[no:, :no]  # (Q_{ta} - Q_{at})
    woc = wvob[:no, :] - (wcca - wcca.T)[nc:, :nc]  # (Q_{it} - Q_{ti})

    # 3.2 add correct term
    hfc = scf.ROHF(mol)
    FSao = hfc.get_k(mol, (scorrao, dmcorrao), hermi=0)
    FSmo = mo_coeff[0].T @ FSao @ mo_coeff[0]
    wvc += lib.einsum('ij,aj->ai', dmcc, FSmo[0, ni:, :nc]) * 2
    wvc -= lib.einsum('ab,ib->ai', dmvv, FSmo[0, :nc, ni:]) * 2
    wvo += FSmo[1, ni:, nc:ni]
    wvo -= lib.einsum('ab,tb->at', dmvv, FSmo[0, nc:ni, ni:]) * 2
    woc += lib.einsum('ij,tj->ti', dmcc, FSmo[0, nc:ni, :nc]) * 2
    woc -= FSmo[1, nc:ni, :nc]
    w = np.hstack((wvc.ravel(), wvo.ravel(), woc.ravel()))

    # 4. constuct G[Z^S] and solve Z-vector equation
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)  # invoke UKS function, same with upper
    def matvec(x):  # ROHF VC/VO/OC orbital Hessian (same as down-rohf)
        xvc = x[:nv*nc].reshape(nv, nc)
        xvo = x[nv*nc:nv*nc+nv*no].reshape(nv, no)
        xoc = x[nv*nc+nv*no:].reshape(no, nc)
        xa = np.hstack((xvc, xvo))
        xb = np.vstack((xoc, xvc))
        dm1 = np.zeros((2, nao, nao))
        dma = np.einsum('ka,ai,il->kl', orbv_a, xa, orbo_a.T)
        dmb = np.einsum('ka,ai,il->kl', orbv_b, xb, orbo_b.T)
        dm1[0] += (dma + dma.T) / 2  # Z^{S,\alpha}
        dm1[1] += (dmb + dmb.T) / 2  # Z^{S,\beta}
        v1 = vresp(dm1)  # G_{\mu\nu}^{\sigma}[Z^S]
        v1a = np.einsum('ak,kl,li->ai', orbv_a.T, v1[0], orbo_a)
        v1b = np.einsum('ak,kl,li->ai', orbv_b.T, v1[1], orbo_b)
        vvc = (v1a[:, :nc] + v1b[no:, :])  # G_{a'i'}^{\alpha}[Z^S]+G_{a'i'}^{\beta}[Z^S]
        voc = v1b[:no, :]  # G_{t'i'}^{\beta}[Z^S]
        vvo = v1a[:, nc:]  # G_{a't'}^{\alpha}[Z^S]
        Fxvc = np.zeros((nv, nc))
        Fxvc -= np.einsum('bi,ab->ai', xvc, fockavv)
        Fxvc -= np.einsum('bi,ab->ai', xvc, fockbvv)
        Fxvc += np.einsum('aj,ji->ai', xvc, fockacc)
        Fxvc += np.einsum('aj,ji->ai', xvc, fockbcc)
        Fxvc -= np.einsum('ti,at->ai', xoc, fockbvo)
        Fxvc += np.einsum('at,ti->ai', xvo, fockaoc)
        Fxvc -= vvc * 2
        Fxvo = np.zeros((nv, no))
        Fxvo -= np.einsum('ti,ai->at', xoc, fockbvc)
        Fxvo += np.einsum('ai,ti->at', xvc, fockaoc)
        Fxvo -= np.einsum('bt,ba->at', xvo, fockavv)
        Fxvo += np.einsum('au,tu->at', xvo, fockaoo)
        Fxvo -= vvo * 2
        Fxoc = np.zeros((no, nc))
        Fxoc -= np.einsum('ui,tu->ti', xoc, fockboo)
        Fxoc += np.einsum('tj,ij->ti', xoc, fockbcc)
        Fxoc -= np.einsum('ai,at->ti', xvc, fockbvo)
        Fxoc += np.einsum('at,ai->ti', xvo, fockavc)
        Fxoc -= voc * 2
        return np.hstack((Fxvc.ravel(), Fxvo.ravel(), Fxoc.ravel()))

    # Z, instead of 1/2 Z
    z = lib.solve(
        matvec, w, tol=1e-12, max_cycle=td.cphf_max_cycle,
        dot=np.dot, lindep=td.dsolve_lindep, verbose=0, tol_residual=None
    )
    zvc = z[:nv*nc].reshape(nv, nc)
    zvo = z[nv*nc:nv*nc+nv*no].reshape(nv, no)
    zoc = z[nv*nc+nv*no:].reshape(no, nc)
    z1a = np.hstack((zvc, zvo))
    z1b = np.vstack((zoc, zvc))
    time1 = log.timer('Z-vector equation solver', *time0)

    # 5.1 construct complete W matrix
    # correct term make Za not equal Zb need construct full W matrix
    z1ao = np.empty((2, nao, nao))
    z1ao[0] = orbv_a @ z1a @ orbo_a.T
    z1ao[1] = orbv_b @ z1b @ orbo_b.T
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)) / 2)
    im0a = np.zeros((nao, nao))
    im0b = np.zeros((nao, nao))
    im0a[:ni, :ni] += fockamo[:ni, :ni]  # ground state
    im0b[:nc, :nc] += fockbmo[:nc, :nc]  # ground state
    im0a[:ni, :ni] += (orbo_a.T @ veff0doo[0] @ orbo_a)
    im0a[:ni, :ni] += lib.einsum('ik,jk->ij', dooa, fockamo[:ni, :ni])
    im0a[:ni, :ni] += lib.einsum('aj,ai->ij', veff0moa[ni:, :ni], va)
    im0a[ni:, ni:] = lib.einsum('ac,bc->ab', dvva, fockamo[ni:, ni:])
    im0a[ni:, ni:] += lib.einsum('bi,ai->ab', veff0moa[ni:, :ni], va)
    im0a[:ni, ni:] = (orbo_a.T @ veff0doo[0] @ orbv_a)
    im0a[:ni, ni:] += lib.einsum('ij,aj->ia', dooa, fockamo[ni:, :ni])
    im0a[:ni, ni:] += lib.einsum('ba,bi->ia', veff0moa[ni:, ni:], va)
    im0a[ni:, :ni] = lib.einsum('ab,ib->ai', dvva, fockamo[:ni, ni:])
    im0a[ni:, :ni] += lib.einsum('ij,aj->ai', veff0moa[:ni, :ni], va)
    im0b[:nc, :nc] += (orbo_b.T @ veff0doo[1] @ orbo_b)
    im0b[:nc, :nc] += lib.einsum('ik,jk->ij', doob, fockbmo[:nc, :nc])
    im0b[:nc, :nc] += lib.einsum('aj,ai->ij', veff0mob[nc:, :nc], vb)
    im0b[nc:, nc:] = lib.einsum('ac,bc->ab', dvvb, fockbmo[nc:, nc:])
    im0b[nc:, nc:] += lib.einsum('bi,ai->ab', veff0mob[nc:, :nc], vb)
    im0b[:nc, nc:] = (orbo_b.T @ veff0doo[1] @ orbv_b)
    im0b[:nc, nc:] += lib.einsum('ij,aj->ia', doob, fockbmo[nc:, :nc])
    im0b[:nc, nc:] += lib.einsum('ba,bi->ia', veff0mob[nc:, nc:], vb)
    im0b[nc:, :nc] = lib.einsum('ab,ib->ai', dvvb, fockbmo[:nc, nc:])
    im0b[nc:, :nc] += lib.einsum('ij,aj->ai', veff0mob[:nc, :nc], vb)
    im0 = im0a + im0b

    # 5.2 add Z
    im0[:ni, :] += (orbo_a.T @ veff[0] @ mo_coeff[0])
    im0[:nc, :] += (orbo_b.T @ veff[1] @ mo_coeff[1])
    im0[:nc, ni:] += lib.einsum('bi,ba->ia', z1a[:, :nc], fockamo[ni:, ni:]) / 2
    im0[ni:, :ni] += lib.einsum('aj,ij->ai', z1a, fockamo[:ni, :ni]) / 2
    im0[nc:ni, :nc] += lib.einsum('at,ai->ti', z1a[:, nc:], fockamo[ni:, :nc]) / 2
    im0[nc:ni, ni:] += lib.einsum('bt,ba->ta', z1a[:, nc:], fockamo[ni:, ni:]) / 2
    im0[ni:, :nc] += lib.einsum('aj,ij->ai', z1b[no:], fockbmo[:nc, :nc]) / 2
    im0[:nc, nc:] += lib.einsum('bi,ba->ia', z1b, fockbmo[nc:, nc:]) / 2
    im0[nc:ni, :nc] += lib.einsum('tj,ij->ti', z1b[:no], fockbmo[:nc, :nc]) / 2
    im0[nc:ni, ni:] += lib.einsum('ti,ai->ta', z1b[:no], fockbmo[ni:, :nc]) / 2

    # 5.3 add correct term
    scorrmo = np.zeros((nao, nao))
    scorrmo[nc:ni, nc:ni] = np.eye(no) / 2
    im0[:nc, :] += lib.einsum('ij,pj->ip', dmcc, FSmo[0, :, :nc])
    im0[ni:, :] += lib.einsum('ab,pb->ap', dmvv, FSmo[0, :, ni:])
    im0[nc:ni, :] += FSmo[1, nc:ni, :] / 2

    im0 = mo_coeff[0] @ im0 @ mo_coeff[0].T

    # 6. derivative of coordinate
    dmz1dvva = (z1ao[0] + z1ao[0].T) / 2 + dmza  # T_{\mu\nu}^{\alpha} + Z^{S,\alpha}
    dmz1doob = (z1ao[1] + z1ao[1].T) / 2 + dmzb  # T_{\mu\nu}^{\beta} + Z^{S,\beta}

    mf_grad = td.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    oo0a = orbo_a @ orbo_a.T
    oo0b = orbo_b @ orbo_b.T
    as_dm1 = oo0a + oo0b + dmz1dvva + dmz1doob

    if hyb > 0:
        dm = (oo0a, dmz1dvva+dmz1dvva.T, dmxa, dmxa.T,
              oo0b, dmz1doob+dmz1doob.T, dmxb, dmxb.T)
        vj, vk = td.get_jk(mol, dm)
        vj = vj.reshape(2,4,3,nao,nao)
        vk = vk.reshape(2,4,3,nao,nao) * hyb
        # if omega != 0:
        #     vk += td.get_k(mol, dm, omega=omega).reshape(2,4,3,nao,nao) * (alpha-hyb)
        veff1 = vj[0] + vj[1] - vk
    else:
        dm = (oo0a, dmz1dvva+dmz1dvva.T, dmxa, dmxa.T,
              oo0b, dmz1doob+dmz1doob.T, dmxb, dmxb.T)
        vj = td.get_j(mol, dm).reshape(2,4,3,nao,nao)
        veff1 = vj[0] + vj[1]

    vkc = td.get_k(mol, (scorrao, dmcorrao), hermi=0)  # correct exchange potential

    # fxcz1_grad = _RO2U(td, mf)
    fxcz1 = tduks_grad._contract_xc_kernel(
        tdro, mf.xc, z1ao, None, False, False, max_memory)[0]
    veff1[:,0] += vxc1[:,1:]
    veff1[:,1] +=(f1oo[:,1:] + fxcz1[:,1:] + k1ao[:,1:]) * 2 # *2 for dmz1doo+dmz1oo.T
    veff1[:,2] += f1vo[:,1:]
    veff1[:,3] += f1vo[:,1:]
    veff1a, veff1b = veff1
    time1 = log.timer('2e AO integral derivatives', *time1)

    # 7. combine upper result
    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        h1ao = hcore_deriv(ia)
        de[k] = lib.einsum('xpq,pq->x', h1ao, as_dm1)
        de[k] += lib.einsum('xpq,pq->x', veff1a[0,:,p0:p1], oo0a[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', veff1b[0,:,p0:p1], oo0b[p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1a[0,:,p0:p1], oo0a[:,p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1b[0,:,p0:p1], oo0b[:,p0:p1])

        de[k] += lib.einsum('xpq,pq->x', veff1a[0,:,p0:p1], dmz1dvva[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', veff1b[0,:,p0:p1], dmz1doob[p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1a[0,:,p0:p1], dmz1dvva[:,p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1b[0,:,p0:p1], dmz1doob[:,p0:p1])

        de[k] -= lib.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= lib.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        de[k] += lib.einsum('xij,ij->x', veff1a[1,:,p0:p1], oo0a[p0:p1])
        de[k] += lib.einsum('xij,ij->x', veff1b[1,:,p0:p1], oo0b[p0:p1])
        de[k] += lib.einsum('xij,ij->x', veff1a[2,:,p0:p1], dmxa[p0:p1,:])*2
        de[k] += lib.einsum('xij,ij->x', veff1b[2,:,p0:p1], dmxb[p0:p1,:])*2
        de[k] += lib.einsum('xji,ij->x', veff1a[3,:,p0:p1], dmxa[:,p0:p1])*2
        de[k] += lib.einsum('xji,ij->x', veff1b[3,:,p0:p1], dmxb[:,p0:p1])*2

        # add correct term
        de[k] += lib.einsum('xpq,pq->x', vkc[0, :, p0:p1], dmcorrao[p0:p1])
        de[k] += lib.einsum('xpq,qp->x', vkc[0, :, p0:p1], dmcorrao[:, p0:p1])
        de[k] += lib.einsum('xpq,pq->x', vkc[1, :, p0:p1], scorrao[p0:p1])
        de[k] += lib.einsum('xpq,qp->x', vkc[1, :, p0:p1], scorrao[:, p0:p1])

        # de[k] += td.extra_force(ia, locals())  # extension, here always zero
    log.timer('SF-up-TDA(ROKS) nuclear gradients', *time0)
    return de


class _RO2U:
    '''Wrap ROKS mf, make mo_coeff and mo_occ have same dim with UKS'''
    def __init__(self, td_grad, mf):
        self.mol = td_grad.mol
        self.verbose = getattr(td_grad, 'verbose', logger.INFO)
        self.stdout = getattr(td_grad, 'stdout', sys.stdout)
        self.base = td_grad
        self.base._scf = mf
        mo_energy = np.array((mf.mo_energy, mf.mo_energy))
        mo_coeff = np.array((mf.mo_coeff, mf.mo_coeff))
        mo_occ = np.zeros((2,len(mf.mo_coeff)))
        mo_occ[0][np.where(mf.mo_occ>=1)[0]]=1
        mo_occ[1][np.where(mf.mo_occ>=2)[0]]=1
        self.base._scf.mo_occ = mo_occ
        self.base._scf.mo_coeff = mo_coeff
        self.base._scf.mo_energy = mo_energy


class SC_gradient(rohf_grad.Gradients):
    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20) + 20
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)
    dsolve_lindep = getattr(__config__, 'lib_linalg_helper_dsolve_lindep', 1e-13)

    def __init__(self, td, state=1):
        self.base = td
        self.base._scf = td.mf
        self.mol = td.mol
        self.v = td.v
        self.state = state
        self.de = None  # gradient of molecule
        self.atmlst = None  # which atom will be calculate gradient
        
    def grad_elec(self, atmlst=None):
        if self.v is None:
            print('***** have not do excited energy calculation *****')
            print('Below, doing excited energy calculation ...')
            self.td.kernel()
        return grad_elec(self, atmlst=atmlst)
    
    def kernel(self, atmlst=None):
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        de = self.grad_elec(atmlst=atmlst)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        self._finalize()
        return self.de

    def _finalize(self):
        print(
            '--------- %s gradients for state %d ----------', 
            self.base.__class__.__name__,
            self.state
        )
        self._write(self.mol, self.de, self.atmlst)
        print('----------------------------------------------')


if __name__ == "__main__":
    ch2o = '''
        H   0.000000   0.934473  -0.588078
        H   0.000000  -0.934473  -0.588078
        C   0.000000   0.000000   0.000000
        O   0.000000   0.000000   1.221104
    '''
    # finite difference use this form coordinate
    ch2o = utils.parse_xyz_string(ch2o)

    mol = gto.M(
        atom = ch2o,
        spin = 2,
        charge = 0,
        basis = '6-31g',
        verbose=3,
    )
    mf = scf.ROKS(mol)
    xc = 'b3lyp'
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.grids.level = 5
    mf.kernel()

    td = XTDA(mol, mf, nstates=3, basis='orbital', use_Davidson=False)
    td.kernel()

    sfu_roks_td = SC_gradient(td, state=1)
    sfu_roks_td.kernel()

    # g_fd = fd_gradient(ch2o, 1, xc=xc)
    # print('finite-diff:\n', g_fd)

