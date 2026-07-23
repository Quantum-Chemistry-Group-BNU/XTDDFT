#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import numpy as np
from typing import List
from pyscf import gto, scf, dft, lib, __config__
from pyscf.lib import logger
from pyscf.dft import numint2c
from pyscf.scf import ucphf
from pyscf.grad import rohf as rohf_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.grad import tduks as tduks_grad
from pyscf.sftda.numint2c_sftd import mcfun_eval_xc_adapter_sf

from SF_TDA import SF_TDA_up
from utils import atom, unit, utils


def excited_energy(atom, spec, state, xc='b3lyp', method=1, cs=50):
    """excited energy = mf.e_tot + td.e[state-1], unit of td.e is Hartree"""
    mol = gto.M(atom=atom, verbose=3, **spec)
    mf = dft.ROKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.grids.level = 5
    mf.kernel()

    td = SF_TDA_up(mf, method=method, davidson=True)
    td.collinear_samples = cs
    # Request +2 extra roots, make davidson itersion more stable
    td.kernel(nstates=max(state, 1) + 2)
    return mf.e_tot + td.e[state-1]


def fd_gradient(atoms, state, xc='b3lyp', h=1e-5, method=1, cs=50):
    """finite difference truncate to second order, (-3E0+4E+ - E++)/(2h), unit of h is Angstrom"""
    assert isinstance(atoms, List) and (len(atoms[0]) == 2)
    print('***** The molecular coordinates of the finite difference input are in angstroms *****')
    natm = len(atoms)
    spec = dict(charge=0, spin=2, basis='6-31g')
    g = np.zeros((natm, 3))
    h_au = h * unit.au2ang

    E0 = excited_energy(atoms, spec, state, xc, method, cs)
    for i in range(natm):
        for d in range(3):
            atoms_h = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_h[i][1][d] += h
            Eh = excited_energy(atoms_h, spec, state, xc, method, cs)
            atoms_2h = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_2h[i][1][d] += 2 * h
            Ehh = excited_energy(atoms_2h, spec, state, xc, method, cs)
            g[i, d] = (-3*E0 + 4*Eh - Ehh) / (2*h_au)  # energy unit is Hartree/bohr
    return g


def _contract_xc_kernel(td_grad, xc_code, dmt, dmoo=None,
                          with_vxc=True, with_kxc=True, max_memory=2000):
    """Spin-flip XC-kernel contraction.

    Args:
        dmt : (nao, nao) transition-density AO matrix (b2a/a2b spin-flip block).
        td_grad : SF-TDA/TDDFT gradient object. Reads `.mol`, `.base._scf`
            (the UKS mean-field), and `.base.collinear_samples`.
        xc_code : XC functional string.
        dmoo : optional 2-tuple of (nao, nao) relaxed occ-occ densities.

    Returns:
        f1vo : (4, nao, nao)
        f1oo : (2, 4, nao, nao) or None
        v1ao : (2, 4, nao, nao) or None
        k1ao : (2, 4, nao, nao) or None
    """
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

    dmvo = (dmt + dmt.T) * 0.5

    f1vo = np.zeros((4, nao, nao))
    deriv = 2
    if dmoo is not None:
        f1oo = np.zeros((2, 4, nao, nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = np.zeros((2, 4, nao, nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = np.zeros((2, 4, nao, nao))
        deriv = 3
    else:
        k1ao = None

    if td_grad.base.collinear_samples > 0:
        # create a mc object to use mcfun.
        nimc = numint2c.NumInt2C()
        nimc.collinear = 'mcol'
        nimc.collinear_samples = td_grad.base.collinear_samples
        eval_xc_eff = mcfun_eval_xc_adapter_sf(nimc, xc_code)

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

    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        if xctype == 'LDA':
            ao0 = ao[0]
        else:
            ao0 = ao
        rho = (
            ni.eval_rho2(mol, ao0, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl=False),
            ni.eval_rho2(mol, ao0, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl=False),
        )
        if td_grad.base.collinear_samples > 0:
            rho_z = np.array([rho[0] + rho[1], rho[0] - rho[1]])
            fxc_sf, kxc_sf = eval_xc_eff(xc_code, rho_z, deriv, xctype=xctype)[2:4]
            kxc_sf = np.stack((kxc_sf[:, :, 0] + kxc_sf[:, :, 1], kxc_sf[:, :, 0] - kxc_sf[:, :, 1]), axis=2)
            rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False)
            if xctype == 'LDA':
                rho1 = rho1[np.newaxis]
            wv = lib.einsum('yg,xyg,g->xg', rho1, 2 * fxc_sf, weight)  # 2 for f_xx + f_yy
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if with_kxc:
                wv = lib.einsum('xg,yg,xyczg,g->czg', rho1, rho1, 2 * kxc_sf, weight)
                fmat_(mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
                fmat_(mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None or with_vxc:
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv=2, spin=1)[1:]

        if dmoo is not None:
            rho2 = np.asarray(
                (
                    ni.eval_rho(mol, ao0, dmoo[0], mask, xctype, hermi=1, with_lapl=False),
                    ni.eval_rho(mol, ao0, dmoo[1], mask, xctype, hermi=1, with_lapl=False),
                )
            )
            if xctype == 'LDA':
                rho2 = rho2[:, np.newaxis]
            wv = lib.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if with_vxc:
            wv = vxc * weight
            fmat_(mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    if f1oo is not None:
        f1oo[:, 1:] *= -1
    if v1ao is not None:
        v1ao[:, 1:] *= -1
    if k1ao is not None:
        k1ao[:, 1:] *= -1
    return f1vo, f1oo, v1ao, k1ao


def grad_elec(td, atmlst=None, max_memory=2000, verbose=logger.INFO):
    """electronic part of spin flip up TDA gradient in ROKS reference state"""
    log = logger.new_logger(td, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td.mol
    mf = td.base._scf
    sftda = td.base
    mo_occ = sftda.mo_occ
    mo_coeff = sftda.mo_coeff
    mo_energy = sftda.mo_energy
    occidx_a = np.where(mo_occ[0] == 1)[0]
    viridx_a = np.where(mo_occ[0] == 0)[0]
    occidx_b = np.where(mo_occ[1] == 1)[0]
    viridx_b = np.where(mo_occ[1] == 0)[0]
    nc = len(occidx_b)
    nv = len(viridx_a)
    no = len(occidx_a) - len(occidx_b)
    nao = mo_coeff[0].shape[0]
    orbv_a = mo_coeff[0][:, (nc+no):]
    orbo_a = mo_coeff[0][:, :(nc+no)]
    orbv_b = mo_coeff[1][:, nc:]
    orbo_b = mo_coeff[1][:, :nc]
    v = td.v[:, td.state-1].reshape(nc, nv)  # eigen vector of TDA

    # 1. internel variable
    dvva = lib.einsum('ia, ib->ab', v, v)  # T_{ab}
    doob = -lib.einsum('ia, ja->ij', v, v)  # T_{ij}
    dmzvva = orbv_a @ dvva @ orbv_a.T  # T_{\mu\nu}^{\alpha}
    dmzoob = orbo_b @ doob @ orbo_b.T  # T_{\mu\nu}^{\beta}
    dmt = orbo_b @ v @ orbv_a.T  # X_{\mu\nu}^{\beta\alpha}
    dm = mf.make_rdm1()
    dmSx = (dmt + dmt.T) / 2  # X_{\mu\nu}^{S}
    dmAx = (dmt - dmt.T) / 2  # X_{\mu\nu}^{A}
    vhf = mf.get_veff(mol, dm)
    h1e = mf.get_hcore()
    focka = h1e + vhf[0]
    fockb = h1e + vhf[1]
    fockamo = mo_coeff[0].T @ focka @ mo_coeff[0]
    fockbmo = mo_coeff[1].T @ fockb @ mo_coeff[1]
    # symmetrized Fock blocks, perpare for solve Z-vector equation
    fockacc = (fockamo[:nc, :nc] + fockamo[:nc, :nc].T) / 2
    fockaoc = (fockamo[nc:nc+no, :nc] + fockamo[:nc, nc:nc+no].T) / 2
    fockavc = (fockamo[nc+no:, :nc] + fockamo[:nc, nc+no:].T) / 2
    fockavv = (fockamo[nc+no:, nc+no:] + fockamo[nc+no:, nc+no:].T) / 2
    fockaoo = (fockamo[nc:nc+no, nc:nc+no] + fockamo[nc:nc+no, nc:nc+no].T) / 2
    fockbcc = (fockbmo[:nc, :nc] + fockbmo[:nc, :nc].T) / 2
    fockbvc = (fockbmo[nc+no:, :nc] + fockbmo[:nc, nc+no:].T) / 2
    fockbvv = (fockbmo[nc+no:, nc+no:] + fockbmo[nc+no:, nc+no:].T) / 2
    fockbvo = (fockbmo[nc+no:, nc:nc+no] + fockbmo[nc:nc+no, nc+no:].T) / 2
    fockboo = (fockbmo[nc:nc+no, nc:nc+no] + fockbmo[nc:nc+no, nc:nc+no].T) / 2

    # 2. functional derivate, include derivate respect to mo_coeff and coordinate
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # f1vo: f^{xc}[X], f1oo: f^{xc}[T], vxc1: v^{xc}[\rho], k1ao: g^{xc}[X,X]
    # and their derivative respect to coordinate
    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(
        td, mf.xc, dmt, (dmzvva, dmzoob), True, True, max_memory)
    
    # 3. construct Q matrix, similar with UKS reference, only difference is add fock
    # i for alpha is nc+no, i for beta is nc, a for alpha is nv, a for beta is no+nv
    if hyb > 0:
        vj, vk = mf.get_jk(mol, (dmzvva, dmzoob), hermi=1)  # g_{\mu\nu}^{\sigma}[T]
        vk = vk * hyb
        vk1 = mf.get_k(mol, dmt, hermi=0) * hyb  # c_x(\mu\lambda|\kappa\nu)
        # G_{\mu\nu}^{\sigma}[T] + g^{xc}[X,X]
        veff0doo = vj[0] + vj[1] - vk + f1oo[:, 0] + k1ao[:, 0]
        wvoa = orbv_a.T @ veff0doo[0] @ orbo_a  # 1/2 Q_{ia}^{\alpha}
        wvob = orbv_b.T @ veff0doo[1] @ orbo_b  # part of 1/2 Q_{ia}^{\beta}
        veff0mo = mo_coeff[1].T @ (f1vo[0] - vk1) @ mo_coeff[0]  # collinear f1vo[0]=0
        wvoa -= lib.einsum('jk,jc->ck', veff0mo[:nc, :(nc+no)], v)  # part of 1/2 Q_{ai}^{\alpha}
        wvoa -= lib.einsum('ac,ka->ck', dvva, fockamo[:(nc+no), (nc+no):])  # part of 1/2 Q_{ai}^{\alpha}
        wvob += lib.einsum('ac,ka->ck', veff0mo.T[(nc+no):, nc:], v)  # part of 1/2 Q_{ia}^{\beta}
        wvob += lib.einsum('jk,jc->ck', doob, fockbmo[:nc, nc:])  # part of 1/2 Q_{ia}^{\beta}

    else:
        vj = mf.get_j(mol, (dmzvva, dmzoob), hermi=1)  # (2, nao, nao)
        veff0doo = vj[0] + vj[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = orbv_a.T @ veff0doo[0] @ orbo_a
        wvob = orbv_b.T @ veff0doo[1] @ orbo_b
        veff0mo = mo_coeff[1].T @ f1vo[0] @ mo_coeff[0]
        wvoa -= lib.einsum('jk,jc->ck', veff0mo[:nc, :(nc+no)], v)
        wvoa -= lib.einsum('ac,ka->ck', dvva, fockamo[:(nc+no), (nc+no):])
        wvob += lib.einsum('ac,ka->ck', veff0mo.T[(nc+no):, nc:], v)
        wvob += lib.einsum('ac,ka->ck', doob, fockbmo[:nc, nc:])

    wvc = wvoa[:, :nc] + wvob[no:, :]  # 1/2 (Q_{ia} - Q_{ai})
    wvo = wvoa[:, nc:]  # 1/2 (Q_{at} - Q_{ta})
    woc = wvob[:no, :]  # 1/2 (Q_{ti} - Q_{it})
    w = np.hstack((wvc.ravel(), wvo.ravel(), woc.ravel())) * 2

    # 4. constuct G[Z^S] and solve Z-vector equation
    # _uks = dft.UKS(mol)
    # _uks.xc = mf.xc
    # _uks.grids = mf.grids
    # _uks.mo_coeff = mo_coeff
    # _uks.mo_occ = mo_occ
    # vresp = _uks.gen_response(mo_coeff, mo_occ, hermi=1)
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

    # 5. construct W matrix
    z1ao = np.empty((2, nao, nao))
    z1ao[0] = orbv_a @ z1a @ orbo_a.T
    z1ao[1] = orbv_b @ z1b @ orbo_b.T
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)) / 2)  # G_{\mu\nu}^{\sigma}[Z^S]

    im0a = np.zeros((nao, nao))
    im0b = np.zeros((nao, nao))
    im0a[:(nc+no), :(nc+no)] += fockamo[:(nc+no), :(nc+no)]  # ground state
    im0b[:nc, :nc] += fockbmo[:nc, :nc]  # ground state
    im0a[:(nc+no), :(nc+no)] += orbo_a.T @ (veff0doo[0] + veff[0]) @ orbo_a
    im0a[(nc+no):, (nc+no):] = lib.einsum('ac,bc->ab', dvva, fockamo[(nc+no):, (nc+no):])
    im0a[(nc+no):, (nc+no):] += lib.einsum('ia,ib->ab', v, veff0mo[:nc, (nc+no):])
    im0a[(nc+no):, :(nc+no)] = lib.einsum('aj,ij->ai', z1a, fockamo[:(nc+no), :(nc+no)])
    im0a[(nc+no):, :(nc+no)] += lib.einsum('ac,ic->ai', dvva, fockamo[:(nc+no), (nc+no):]) * 2  # ROKS particle term
    im0a[(nc+no):, :(nc+no)] += lib.einsum('ia,ij->aj', v, veff0mo[:nc, :(nc+no)]) * 2
    im0b[:nc, :nc] += orbo_b.T @ (veff0doo[1] + veff[1]) @ orbo_b
    im0b[:nc, :nc] += lib.einsum('ik,kj->ij', doob, fockbmo[:nc, :nc])
    im0b[:nc, :nc] += lib.einsum('ia,ja->ij', v, veff0mo[:nc, (nc+no):])
    im0b[nc:, :nc] = lib.einsum('aj,ij->ai', z1b, fockbmo[:nc, :nc])
    im0b[nc:nc+no, :nc] += lib.einsum('bt,bi->ti', zvo, fockavc)  # ROHF VO->OC coupling
    im0 = mo_coeff[0] @ (im0a+im0b) @ mo_coeff[0].T  # C^{\alpha}=C^{\beta}

    # 6. derivative of coordinate
    dmz1dvva = (z1ao[0] + z1ao[0].T) / 2 + dmzvva  # T_{\mu\nu}^{\alpha} + Z^{S,\alpha}
    dmz1doob = (z1ao[1] + z1ao[1].T) / 2 + dmzoob  # T_{\mu\nu}^{\beta} + Z^{S,\beta}

    mf_grad = td.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    oo0a = orbo_a @ orbo_a.T
    oo0b = orbo_b @ orbo_b.T
    as_dm1 = oo0a + oo0b + dmz1dvva + dmz1doob

    if hyb > 0:
        vj, vk = td.get_jk(mol, (oo0a, dmz1dvva, oo0b, dmz1doob))
        vj = vj.reshape(2, 2, 3, nao, nao)
        vk = vk.reshape(2, 2, 3, nao, nao) * hyb
        veff1 = vj[0] + vj[1] - vk
        vk1 = -td.get_k(mol, (dmt, dmt.T)) * hyb
    else:
        vj = td.get_j(mol, (oo0a, dmz1dvva, oo0b, dmz1doob))
        vj = vj.reshape(2, 2, 3, nao, nao)
        veff1 = vj[0] + vj[1]
        veff1 = np.stack((veff1, veff1))

    fxcz1_grad = _RO2U(td, mf, sftda)
    fxcz1 = tduks_grad._contract_xc_kernel(
        fxcz1_grad, mf.xc, z1ao, None, False, False, max_memory)[0]
    veff1[:, 0] += vxc1[:, 1:]
    veff1[:, 1] += (f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:])
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
        de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], oo0a[p0:p1]) * 2
        de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], oo0b[p0:p1]) * 2

        de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], dmz1dvva[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], dmz1doob[p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1a[0, :, p0:p1], dmz1dvva[:, p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1b[0, :, p0:p1], dmz1doob[:, p0:p1])
        de[k] += lib.einsum('xij,ij->x', veff1a[1, :, p0:p1], oo0a[p0:p1])*2
        de[k] += lib.einsum('xij,ij->x', veff1b[1, :, p0:p1], oo0b[p0:p1])*2
        if hyb > 0:
            de[k] += lib.einsum('xpq,pq->x', vk1[0, :, p0:p1], dmt[p0:p1]) * 2
            de[k] += lib.einsum('xpq,pq->x', vk1[1, :, p0:p1], dmt.T[p0:p1]) * 2
        if td.base.collinear_samples > 0:
            de[k] += np.einsum('xpq,pq->x', f1vo[1:, p0:p1], dmt[p0:p1]) * 2
            de[k] += np.einsum('xpq,pq->x', f1vo[1:, p0:p1], dmt.T[p0:p1]) * 2
        de[k] -= np.einsum('xpq,pq->x', s1[:, p0:p1], im0[p0:p1])
        de[k] -= np.einsum('xpq,qp->x', s1[:, p0:p1], im0[:, p0:p1])
        # de[k] += td.extra_force(ia, locals())  # extension, here always zero
    log.timer('SF-up-TDA(ROKS) nuclear gradients', *time0)
    return de


class _RO2U:
    '''Wrap ROKS mf, make mo_coeff and mo_occ have same dim with UKS'''
    def __init__(self, td_grad, mf, sftda):
        self.mol = td_grad.mol
        self.verbose = getattr(td_grad, 'verbose', logger.INFO)
        self.stdout = getattr(td_grad, 'stdout', sys.stdout)
        self.base = td_grad
        self.base._scf = mf
        self.base._scf.mo_occ = sftda.mo_occ
        self.base._scf.mo_coeff = sftda.mo_coeff
        self.base.collinear_samples = sftda.collinear_samples


class SFU_gradient(rohf_grad.Gradients):
    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20) + 20
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)
    dsolve_lindep = getattr(__config__, 'lib_linalg_helper_dsolve_lindep', 1e-13)

    def __init__(self, td, method=1, state=1):
        self.base = td
        self.base._scf = td.mf
        self.mol = td.mol
        self.v = td.v
        self.state = state
        self.method = method
        self.de = None  # gradient of molecule
        self.atmlst = None  # which atom will be calculate gradient
        if method == 1:
            self.collinear_samples = 20
        elif method == 2:
            self.collinear_samples = -1
        else:
            raise NotImplementedError("ALDA0 and Noncollinear kernel do not implement")
        
    def grad_elec(self, atmlst=None):
        if self.v is None:
            print('***** have not do excited energy calculation *****')
            print('Below, doing excited energy calculation ...')
            self.td.kernel(states=max(self.state, 1)+2)
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

    td = SF_TDA_up(mf, method=2, davidson=True)
    td.collinear_samples = -1
    td.kernel(nstates=max(1, 1) + 2)

    sfu_roks_td = SFU_gradient(td, method=2)
    sfu_roks_td.kernel()

    g_fd = fd_gradient(ch2o, 1, xc=xc, method=2)
    print('finite-diff:\n', g_fd)

