#!/usr/bin/env python
import sys
import math
from pyscf import scf, lib, __config__
from pyscf.lib import logger
from pyscf.dft import numint2c
from pyscf.grad import rohf as rohf_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.grad import tduks as tduks_grad
from pyscf.sftda.numint2c_sftd import mcfun_eval_xc_adapter_sf

from ...utils.backend import asnumpy
from ._backend import array_module, is_gpu_mf, nuclear_gradient


def jk_energies_per_atom(
        mf, dm_list, j_factor=None, k_factor=None,
        omega=None, lr_factor=None, sr_factor=None,
        hermi=0, sum_results=False, verbose=None
    ):
    """
    Computes a set of first-order derivatives of J/K contributions for each
    element (density matrix or a pair of density matrices) in dm_pairs.

    This function supports evaluating multiple sets of energy derivatives in a
    single call. Additionally, for each set, the two density matrices for the
    four-index Coulomb integrals can be different.

    Args:
        dm_list :
            A list of density-matrix-pairs [[dm, dm], [dm, dm], ...].
            Each element corresponds to one set of energy derivative.
        j_factor :
            A list of factors for Coulomb (J) term
        k_factor :
            A list of factors for Coulomb (K) term
        hermi :
            No effects
        sum_results : bool
            If True, aggregate all sets of derivatives into a single result.

    Returns:
        An array of shape (*, Natm, 3) if sum_results is False; otherwise,
        an array of shape (Natm, 3).
    """
    from gpu4pyscf.grad.tdrhf import _jk_energies_per_atom
    xp = array_module(mf)
    vhfopt = mf._opt_gpu.get(omega)
    if vhfopt is None:
        from gpu4pyscf.scf.jk import _VHFOpt
        # For LDA and GGA, only mf._opt_jengine is initialized
        mol = mf.mol
        with mol.with_range_coulomb(omega):
            vhfopt = mf._opt_gpu[omega] = _VHFOpt(mol, mf.direct_scf_tol).build()
    if isinstance(dm_list, xp.ndarray) and dm_list.ndim == 2:
        dm_list = dm_list[None]
    ejk = _jk_energies_per_atom(vhfopt, dm_list, j_factor, k_factor,
                                omega=omega, lr_factor=lr_factor, sr_factor=sr_factor,
                                sum_results=sum_results, verbose=verbose)
    return ejk


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
    gpu = is_gpu_mf(mf)
    xp = array_module(mf)

    # Below two line have changed. For a ROKS reference mf.mo_coeff is 2-D 
    # (restricted); read the doubled pseudo-UKS arrays the SF_TDA_up solver
    # stores. For UKS these equal mf's.
    mo_coeff = xp.asarray(getattr(td_grad.base, 'mo_coeff', mf.mo_coeff))
    mo_occ = xp.asarray(getattr(td_grad.base, 'mo_occ', mf.mo_occ))
    nao = mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmvo = xp.asarray((dmt + dmt.T) * 0.5)
    if dmoo is not None:
        dmoo = xp.asarray(dmoo)

    eval_mol = mol
    if gpu:
        if not all(xcfun.on_gpu
                   for xcfun, _ in ni._init_xcfuns(xc_code, spin=1)):
            raise NotImplementedError(
                f"GPU analytic gradients require GPU-native LibXC components; "
                f"{xc_code!r} would use a CPU fallback"
            )
        from gpu4pyscf.grad import tdrks as gpu_tdrks_grad
        from gpu4pyscf.lib.cupy_helper import contract as gpu_contract
        from gpu4pyscf.tdscf._uhf_resp_sf import (
            mcfun_eval_xc_adapter_sf as gpu_mcfun_eval_xc_adapter_sf,
        )

        opt = getattr(ni, "gdftopt", None)
        if opt is None:
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
        eval_mol = opt._sorted_mol
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])
        dmvo = opt.sort_orbitals(dmvo, axis=[0, 1])
        if dmoo is not None:
            dmoo = opt.sort_orbitals(dmoo, axis=[1, 2])

    f1vo = xp.zeros((4, nao, nao))
    deriv = 2
    if dmoo is not None:
        f1oo = xp.zeros((2, 4, nao, nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = xp.zeros((2, 4, nao, nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = xp.zeros((2, 4, nao, nao))
        deriv = 3
    else:
        k1ao = None

    if td_grad.base.collinear_samples > 0:
        if gpu:
            eval_xc_eff = gpu_mcfun_eval_xc_adapter_sf(
                ni, xc_code, td_grad.base.collinear_samples
            )
        else:
            nimc = numint2c.NumInt2C()
            nimc.collinear = 'mcol'
            nimc.collinear_samples = td_grad.base.collinear_samples
            eval_xc_eff = mcfun_eval_xc_adapter_sf(nimc, xc_code)

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = (
            (gpu_tdrks_grad._lda_eval_mat_ if gpu else tdrks_grad._lda_eval_mat_), 1
        )
    elif xctype == 'GGA':
        # gpu4pyscf v1.8.1
        fmat_, ao_deriv = (
            (gpu_tdrks_grad._gga_eval_mat_ if gpu else tdrks_grad._gga_eval_mat_), 2
        )
        # # gpu4pyscf v1.7.0
        # fmat_, ao_deriv = (
        #     (gpu_tduks_sf_grad._gga_eval_mat_ if gpu else tdrks_grad._gga_eval_mat_), 2
        # )
    elif xctype == 'MGGA':
        if gpu:
            raise NotImplementedError("GPU analytic gradients do not support MGGA")
        fmat_, ao_deriv = tdrks_grad._mgga_eval_mat_, 2
        logger.warn(td_grad, 'TDUKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    if gpu:
        block_loop = ni.block_loop(eval_mol, grids, nao, ao_deriv)
    else:
        block_loop = ni.block_loop(mol, grids, nao, ao_deriv, max_memory)

    for ao, mask, weight, coords in block_loop:
        if xctype == 'LDA':
            ao0 = ao[0]
        else:
            ao0 = ao
        if gpu:
            coeff_a = mo_coeff[0, mask]
            coeff_b = mo_coeff[1, mask]
            dmvo_block = dmvo[mask[:, None], mask]
        else:
            coeff_a, coeff_b = mo_coeff
            dmvo_block = dmvo
        rho = (
            ni.eval_rho2(eval_mol, ao0, coeff_a, mo_occ[0], mask, xctype, with_lapl=False),
            ni.eval_rho2(eval_mol, ao0, coeff_b, mo_occ[1], mask, xctype, with_lapl=False),
        )
        if td_grad.base.collinear_samples > 0:
            rho_z = xp.asarray([rho[0] + rho[1], rho[0] - rho[1]])
            fxc_sf, kxc_sf = eval_xc_eff(xc_code, rho_z, deriv, xctype=xctype)[2:4]
            rho1 = ni.eval_rho(eval_mol, ao0, dmvo_block, mask, xctype, hermi=1, with_lapl=False)
            if xctype == 'LDA':
                rho1 = rho1[xp.newaxis]
                if gpu:
                    tmp = gpu_contract('yg,xyg->xg', rho1, 2 * fxc_sf)
                    wv = gpu_contract('xg,g->xg', tmp, weight)
                else:
                    wv = xp.einsum('yg,xyg,g->xg', rho1, 2 * fxc_sf, weight)
            elif gpu:
                # # gpu4pyscf v1.7.0
                # wv = gpu_tduks_sf_grad.uks_sf_gga_wv1(rho1, fxc_sf, weight)

                # gpu4pyscf v1.7.0
                wv = xp.einsum('yg,xyg->xg', rho1, 2.0 * fxc_sf) * weight
            else:
                wv = xp.einsum('yg,xyg,g->xg', rho1, 2 * fxc_sf, weight)
            fmat_(eval_mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if with_kxc:
                if gpu and xctype == 'GGA':
                    # # gpu4pyscf v1.7.0
                    # gv = gpu_tduks_sf_grad.uks_sf_gga_wv2_p(
                    #     rho1, kxc_sf, weight
                    # )

                    # gpu4pyscf v1.8.1
                    gv = xp.einsum('xg,yg,xyvzg->vzg', rho1, rho1, 2.0 * kxc_sf, optimize=True) * weight
                    wv = xp.stack((gv[0] + gv[1], gv[0] - gv[1]))
                else:
                    kxc_sf = xp.stack(
                        (kxc_sf[:, :, 0] + kxc_sf[:, :, 1],
                         kxc_sf[:, :, 0] - kxc_sf[:, :, 1]),
                        axis=2,
                    )
                    if gpu:
                        tmp = gpu_contract('xg,yg->xyg', rho1, rho1)
                        tmp = gpu_contract('xyg,xyczg->czg', tmp, 2 * kxc_sf)
                        wv = gpu_contract('czg,g->czg', tmp, weight)
                    else:
                        wv = xp.einsum(
                            'xg,yg,xyczg,g->czg', rho1, rho1,
                            2 * kxc_sf, weight,
                        )
                fmat_(eval_mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
                fmat_(eval_mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None or with_vxc:
            vxc, fxc, kxc = ni.eval_xc_eff(
                xc_code, xp.asarray(rho) if gpu else rho, deriv=2, spin=1
            )[1:]

        if dmoo is not None:
            if gpu:
                dmoo_a = dmoo[0, mask[:, None], mask]
                dmoo_b = dmoo[1, mask[:, None], mask]
            else:
                dmoo_a, dmoo_b = dmoo
            rho2 = xp.asarray(
                (
                    ni.eval_rho(eval_mol, ao0, dmoo_a, mask, xctype, hermi=1, with_lapl=False),
                    ni.eval_rho(eval_mol, ao0, dmoo_b, mask, xctype, hermi=1, with_lapl=False),
                )
            )
            if xctype == 'LDA':
                rho2 = rho2[:, xp.newaxis]
            if gpu:
                tmp = gpu_contract('axg,axbyg->byg', rho2, fxc)
                wv = gpu_contract('byg,g->byg', tmp, weight)
                # # gpu4pyscf v1.7.0
                # if xctype == 'GGA':
                #     wv[:, 0] *= .5
            else:
                wv = xp.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(eval_mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(eval_mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if with_vxc:
            wv = vxc * weight
            # # gpu4pyscf v1.7.0
            # if gpu and xctype == 'GGA':
            #     wv[:, 0] *= .5
            fmat_(eval_mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(eval_mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    if gpu:
        f1vo = opt.unsort_orbitals(f1vo, axis=[1, 2])
    if f1oo is not None:
        f1oo[:, 1:] *= -1
        if gpu:
            f1oo = opt.unsort_orbitals(f1oo, axis=[2, 3])
    if v1ao is not None:
        v1ao[:, 1:] *= -1
        if gpu:
            v1ao = opt.unsort_orbitals(v1ao, axis=[2, 3])
    if k1ao is not None:
        k1ao[:, 1:] *= -1
        if gpu:
            k1ao = opt.unsort_orbitals(k1ao, axis=[2, 3])
    return f1vo, f1oo, v1ao, k1ao


def grad_elec(td, fglobal=None, fit=True,d_lda=0.3, atmlst=None, max_memory=2000, verbose=logger.INFO):
    """electronic part of spin flip up TDA gradient in ROKS reference state"""
    log = logger.new_logger(td, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td.mol
    mf = td.base._scf
    sftda = td.base
    re = sftda.re
    xp = array_module(mf)
    gpu = is_gpu_mf(mf)
    mo_energy = xp.asarray(sftda.mo_energy)
    mo_coeff = xp.asarray(sftda.mo_coeff)
    mo_occ = xp.asarray(sftda.mo_occ)
    occidx_a = xp.where(mo_occ[0] == 1)[0]
    viridx_a = xp.where(mo_occ[0] == 0)[0]
    occidx_b = xp.where(mo_occ[1] == 1)[0]
    viridx_b = xp.where(mo_occ[1] == 0)[0]
    nc = len(occidx_b)
    nv = len(viridx_a)
    no = len(occidx_a) - len(occidx_b)
    ni = nc + no  # convenient use, alpha occ
    na = no + nv  # convenient use, beta vir
    nao = mo_coeff[0].shape[0]
    orbv_a = mo_coeff[0][:, ni:]
    orbo_a = mo_coeff[0][:, :ni]
    orbv_b = mo_coeff[1][:, nc:]
    orbo_b = mo_coeff[1][:, :nc]
    v = xp.asarray(td.v[:, td.state - 1])
    v_cv = v[:nc*nv].reshape(nc, nv)
    v_co = v[nc*nv:nc*na].reshape(nc, no)
    v_ov = v[nc*na:nc*na+no*nv].reshape(no, nv)
    if re:
        v_oo = xp.einsum(
            'xy,y->x', xp.asarray(sftda.vects), v[nc*na+no*nv:]
        )
    else:
        v_oo = v[nc*na+no*nv:]
    v_oo = v_oo.reshape(no, no)
    v_ca = xp.hstack((v_co, v_cv))
    v_oa = xp.hstack((v_oo, v_ov))
    v = xp.vstack((v_ca, v_oa)).T

    # 1. internel variable
    dooa = -xp.einsum("ai,aj->ij", v, v)  # T_{ij}
    dvvb = xp.einsum("ai,bi->ab", v, v)  # T_{ab}
    dmzooa = orbo_a @ dooa @ orbo_a.T  # T_{\mu\nu}^{\alpha}
    dmzvvb = orbv_b @ dvvb @ orbv_b.T  # T_{\mu\nu}^{\beta}
    dmt = orbv_b @ v @ orbo_a.T  # X_{\mu\nu}^{\alpha\beta}

    # correct coefficient
    si = mol.spin / 2
    c1 = 1 / si
    c2 = 2 / (2 * si - 1)
    c3 = math.sqrt((2 * si + 1)/(2 * si)) - 1
    c4 = math.sqrt((2 * si + 1)/(2 * si - 1)) - 1
    c5 = math.sqrt((2 * si)/(2 * si - 1)) - 1
    c6 = 1 / math.sqrt(2 * si * (2 * si - 1))
    dmcc = xp.einsum('aj,ai->ij', v[no:, :nc], v[no:, :nc]) * c1
    dmcc += xp.einsum('tj,ti->ij', v[:no, :nc], v[:no, :nc]) * c2
    dmvv = xp.einsum('bi,ai->ab', v[no:, :nc], v[no:, :nc]) * c1
    dmvv += xp.einsum('bt,at->ab', v[no:, nc:ni], v[no:, nc:ni]) * c2
    dmoca = -xp.einsum('au,ai->ui', v[no:, nc:ni], v[no:, :nc]) * c3
    dmoca -= xp.einsum('tu,ti->ui', v[:no, nc:ni], v[:no, :nc]) * c5
    dmocb = v[:no, :nc] * xp.trace(v[:no, nc:ni]) * c6
    dmvoa = -v[no:, nc:ni] * xp.trace(v[:no, nc:ni]) * c6
    dmvob = xp.einsum('ui,ai->au', v[:no, :nc], v[no:, :nc]) * c3
    dmvob += xp.einsum('ut,at->au', v[:no, nc:ni], v[no:, nc:ni]) * c5
    scorrao = orbo_a[:, nc:ni] @ (xp.eye(no)/2) @ orbo_a[:, nc:ni].T

    dmS = xp.zeros((nao, nao))  # contract with F^{S}
    dmHFa = xp.zeros_like(dmS)  # contract with F^{HF,\alpha}
    dmHFb = xp.zeros_like(dmS)  # contract with F^{HF,\beta}
    dmS[:nc, :nc] = dmcc
    dmS[ni:, ni:] = dmvv
    # dmS[ni:, :nc] = 2.0 * dmvc  # always zero
    dmHFa[:nc, nc:ni] = dmoca.T  # T_{iu}^{CO,\alpha} in \Delta Q_{ip}
    dmHFa[nc:ni, ni:] = dmvoa.T  # T_{ta}^{OV,\alpha} in \Delta Q_{tp}
    dmHFa[nc:ni, :nc] = dmoca  # T_{iu}^{CO,\alpha} in \Delta Q_{tp}
    dmHFa[ni:, nc:ni] = dmvoa  # T_{ta}^{OV,\alpha} in \Delta Q_{ap}
    dmHFb[:nc, nc:ni] = dmocb.T
    dmHFb[nc:ni, ni:] = dmvob.T
    dmHFb[nc:ni, :nc] = dmocb
    dmHFb[ni:, nc:ni] = dmvob
    dmSao = mo_coeff[0] @ dmS @ mo_coeff[0].T
    dmHFaao = mo_coeff[0] @ dmHFa @ mo_coeff[0].T
    dmHFbao = mo_coeff[0] @ dmHFb @ mo_coeff[0].T
    dmtcv = orbv_a @ v[no:, :nc] @ orbo_b.T
    dmtco = orbv_b[:, :no] @ v[:no, :nc] @ orbo_b.T
    dmtov = orbv_a @ v[no:, nc:ni] @ orbo_a[:, nc:ni].T
    dmtoo = orbv_b[:, :no] @ v[:no, nc:ni] @ orbo_a[:, nc:ni].T

    dm = xp.asarray(mf.make_rdm1())
    vhf = xp.asarray(mf.get_veff(mol, dm))
    h1e = xp.asarray(mf.get_hcore())
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

    # 2. functional derivate, include derivate respect to mo_coeff and coordinate
    tdro = _RO2U(td, mf, sftda)
    ni_ = mf._numint
    ni_.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni_.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    if fglobal is None:
        if omega == 0:
            cx = hyb
        else:
            cx = hyb + (alpha - hyb) * math.erf(omega)
        fglobal = (1 - d_lda) * cx + d_lda
        if sftda.method == 1 and fit:
            fglobal = fglobal * 4 * (cx - 0.5) ** 2
    # f1vo: f^{xc}[X], f1oo: f^{xc}[T], vxc1: v^{xc}[\rho], k1ao: g^{xc}[X,X]
    # and their derivative respect to coordinate
    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(
        td, mf.xc, dmt, (dmzooa, dmzvvb), True, True, max_memory)

    # 3.1 construct Q matrix
    with_k = ni_.libxc.is_hybrid_xc(mf.xc)
    if with_k:
        vj, vk = mf.get_jk(
            mol, xp.stack((dmzooa, dmzvvb)), hermi=1
        )
        vj = xp.asarray(vj)
        vk = xp.asarray(vk) * hyb
        vk1 = xp.asarray(mf.get_k(mol, dmt, hermi=0)) * hyb
        # G_{\mu\nu}^{\sigma}[T] + g^{xc}[X,X]
        veff0doo = vj[0] + vj[1] - vk + f1oo[:, 0] + k1ao[:, 0]
        wvoa = orbv_a.T @ veff0doo[0] @ orbo_a  # part of 1/2 Q_{ia}^{\alpha}
        wvob = orbv_b.T @ veff0doo[1] @ orbo_b  # 1/2 Q_{ia}^{\beta}
        veff0mo = mo_coeff[1].T @ (f1vo[0] - vk1) @ mo_coeff[0]  # collinear f1vo[0]=0
        wvoa += xp.einsum('ba,bi->ai', veff0mo[nc:, ni:], v)
        wvoa += xp.einsum('ij,aj->ai', dooa, fockamo[ni:, :ni])
        wvob -= xp.einsum('ij,aj->ai', veff0mo[:nc, :ni], v)
        wvob -= xp.einsum('ab,ib->ai', dvvb, fockbmo[:nc, nc:])
    else:
        vj = xp.asarray(mf.get_j(
            mol, xp.stack((dmzooa, dmzvvb)), hermi=1
        ))
        veff0doo = vj[0] + vj[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = orbv_a.T @ veff0doo[0] @ orbo_a
        wvob = orbv_b.T @ veff0doo[1] @ orbo_b
        veff0mo = mo_coeff[1].T @ f1vo[0] @ mo_coeff[0]
        wvoa += xp.einsum('ba,bi->ai', veff0mo[nc:, ni:], v)
        wvoa += xp.einsum('ij,aj->ai', dooa, fockamo[ni:, :ni])
        wvob -= xp.einsum('ij,aj->ai', veff0mo[:nc, :ni], v)
        wvob -= xp.einsum('ab,ib->ai', dvvb, fockbmo[:nc, nc:])

    wcca = orbo_a.T @ veff0doo[0] @ orbo_a
    wcca += xp.einsum('ik,jk->ij', dooa, fockamo[:ni, :ni])
    wcca += xp.einsum('aj,ai->ij', veff0mo[nc:, :ni], v)
    wvvb = xp.einsum('ac,bc->ab', dvvb, fockbmo[nc:, nc:])
    wvvb += xp.einsum('bi,ai->ab', veff0mo[nc:, :ni], v)
    wvc = wvoa[:, :nc] + wvob[no:, :]  # 1/2 (Q_{ia} - Q_{ai})
    wvo = wvoa[:, nc:] - (wvvb - wvvb.T)[no:, :no]  # 1/2 (Q_{ta} - Q_{at})
    woc = wvob[:no, :] - (wcca - wcca.T)[nc:, :nc]  # 1/2 (Q_{it} - Q_{ti})

    # 3.2 add correct term
    if gpu:
        from gpu4pyscf.scf import rohf as gpu_rohf

        hfc = gpu_rohf.ROHF(mol)
    else:
        hfc = scf.ROHF(mol)
    vhfc = xp.asarray(hfc.get_veff(mol, dm))
    h1ec = xp.asarray(hfc.get_hcore())
    fockac = h1ec + vhfc[0]
    fockbc = h1ec + vhfc[1]
    fockacmo = mo_coeff[0].T @ fockac @ mo_coeff[0]
    fockbcmo = mo_coeff[1].T @ fockbc @ mo_coeff[1]
    dm = xp.stack((scorrao, dmSao, dmHFaao, dmHFbao,
                   dmtcv, dmtco, dmtov, dmtoo))
    vjcao, vkcao = hfc.get_jk(mol, dm, hermi=0)
    vjcao = xp.asarray(vjcao)
    vkcao = xp.asarray(vkcao)
    vjc = mo_coeff[0].T @ vjcao @ mo_coeff[0]
    vkc = mo_coeff[0].T @ vkcao @ mo_coeff[0]
    # T_{pq}^{X}F_{pq}^{S} + T_{pq}^{Y,\sigma}F_{pq}^{HF,\sigma}, X\in {CC,VV,CV}, Y\in {CO,OV}
    dq = dmS @ vkc[0] + dmHFa @ fockacmo + dmHFb @ fockbcmo
    # next 2 lines use double count dmHFa, dmHFb
    dq[:ni, :] += orbo_a.T @ (vjcao[2] + vjcao[3] - vkcao[2]) @ mo_coeff[0]
    dq[:nc, :] += orbo_b.T @ (vjcao[2] + vjcao[3] - vkcao[3]) @ mo_coeff[0]
    dq[nc:ni, :] += orbo_a[:, nc:ni].T @ vkcao[1] @ mo_coeff[0] / 2  # K[T]

    x_cv = -c3 * vkcao[5] - c3 * vkcao[6] - c4 * vkcao[7]
    x_co = -c3 * vkcao[4] - 0.5 * c2 * vkcao[6] - c5 * vkcao[7] - 0.5 * c2 * vjcao[5] + 0.5 * c2 * vjcao[6]
    x_ov = -c3 * vkcao[4] - 0.5 * c2 * vkcao[5] - c5 * vkcao[7] - 0.5 * c2 * vjcao[6] + 0.5 * c2 * vjcao[5]
    x_oo = -c4 * vkcao[4] - c5 * vkcao[5] - c5 * vkcao[6]
    x_cv_i = orbo_b.T @ x_cv.T @ mo_coeff[0]
    x_cv_a = orbv_a.T @ x_cv @ mo_coeff[0]
    x_co_i = orbo_b.T @ x_co.T @ mo_coeff[0]
    x_co_t = orbv_b[:, :no].T @ x_co @ mo_coeff[0]
    x_ov_t = orbv_b[:, :no].T @ x_ov.T @ mo_coeff[0]
    x_ov_v = orbv_a.T @ x_ov @ mo_coeff[0]
    x_oo_t = orbo_a[:, nc:ni].T @ x_oo @ mo_coeff[0]
    x_oo_u = orbo_a[:, nc:ni].T @ x_oo.T @ mo_coeff[0]

    dq[ni:, :] += xp.einsum('ip,ai->ap', x_cv_i, v[no:, :nc])
    dq[:nc, :] += xp.einsum('ap,ai->ip', x_cv_a, v[no:, :nc])
    dq[:nc, :] += xp.einsum('tp,ti->ip', x_co_t, v[:no, :nc])
    dq[nc:ni, :] += xp.einsum('ip,ti->tp', x_co_i, v[:no, :nc])
    dq[nc:ni, :] += xp.einsum('ap,at->tp', x_ov_v, v[no:, nc:ni])
    dq[ni:, :] += xp.einsum('tp,at->ap', x_ov_t, v[no:, nc:ni])
    dq[nc:ni, :] += xp.einsum('vp,vt->tp', x_oo_t, v[:no, nc:ni])
    dq[nc:ni, :] += xp.einsum('tp,vt->vp', x_oo_u, v[:no, nc:ni])

    dq = fglobal * dq
    wvc += dq[:nc, ni:].T - dq[ni:, :nc]
    woc += dq[:nc, nc:ni].T - dq[nc:ni, :nc]
    wvo += dq[nc:ni, ni:].T - dq[ni:, nc:ni]

    w = xp.hstack((wvc.ravel(), wvo.ravel(), woc.ravel())) * 2

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
        xa = xp.hstack((xvc, xvo))
        xb = xp.vstack((xoc, xvc))
        dm1 = xp.zeros((2, nao, nao))
        dma = xp.einsum('ka,ai,il->kl', orbv_a, xa, orbo_a.T)
        dmb = xp.einsum('ka,ai,il->kl', orbv_b, xb, orbo_b.T)
        dm1[0] += (dma + dma.T) / 2  # Z^{S,\alpha}
        dm1[1] += (dmb + dmb.T) / 2  # Z^{S,\beta}
        v1 = vresp(dm1)  # G_{\mu\nu}^{\sigma}[Z^S]
        v1a = xp.einsum('ak,kl,li->ai', orbv_a.T, v1[0], orbo_a)
        v1b = xp.einsum('ak,kl,li->ai', orbv_b.T, v1[1], orbo_b)
        vvc = (v1a[:, :nc] + v1b[no:, :])  # G_{a'i'}^{\alpha}[Z^S]+G_{a'i'}^{\beta}[Z^S]
        voc = v1b[:no, :]  # G_{t'i'}^{\beta}[Z^S]
        vvo = v1a[:, nc:]  # G_{a't'}^{\alpha}[Z^S]
        Fxvc = xp.zeros((nv, nc))
        Fxvc -= xp.einsum('bi,ab->ai', xvc, fockavv)
        Fxvc -= xp.einsum('bi,ab->ai', xvc, fockbvv)
        Fxvc += xp.einsum('aj,ji->ai', xvc, fockacc)
        Fxvc += xp.einsum('aj,ji->ai', xvc, fockbcc)
        Fxvc -= xp.einsum('ti,at->ai', xoc, fockbvo)
        Fxvc += xp.einsum('at,ti->ai', xvo, fockaoc)
        Fxvc -= vvc * 2
        Fxvo = xp.zeros((nv, no))
        Fxvo -= xp.einsum('ti,ai->at', xoc, fockbvc)
        Fxvo += xp.einsum('ai,ti->at', xvc, fockaoc)
        Fxvo -= xp.einsum('bt,ba->at', xvo, fockavv)
        Fxvo += xp.einsum('au,tu->at', xvo, fockaoo)
        Fxvo -= vvo * 2
        Fxoc = xp.zeros((no, nc))
        Fxoc -= xp.einsum('ui,tu->ti', xoc, fockboo)
        Fxoc += xp.einsum('tj,ij->ti', xoc, fockbcc)
        Fxoc -= xp.einsum('ai,at->ti', xvc, fockbvo)
        Fxoc += xp.einsum('at,ai->ti', xvo, fockavc)
        Fxoc -= voc * 2
        return xp.hstack((Fxvc.ravel(), Fxvo.ravel(), Fxoc.ravel()))

    # Z, instead of 1/2 Z
    if gpu:
        from cupyx.scipy.sparse.linalg import LinearOperator, gmres

        operator = LinearOperator((w.size, w.size), matvec=matvec, dtype=w.dtype)
        z, _ = gmres(operator, w, tol=1e-12,
            maxiter=td.cphf_max_cycle)
        # residual = float(xp.linalg.norm(matvec(z) - w)) / max(
        #     float(xp.linalg.norm(w)), 1e-30
        # )
        # if info != 0 or residual > td.cphf_conv_tol:
        #     raise RuntimeError(
        #         f"Z-vector GMRES failed: info={info}, residual={residual:.3e}"
        #     )
    else:
        z = lib.solve(
            matvec, w, tol=1e-12, max_cycle=td.cphf_max_cycle,
            dot=xp.dot, lindep=td.dsolve_lindep, verbose=0,
            tol_residual=None,
        )
    # # more accurate solver than lib.solve
    # operator = LinearOperator((w.size, w.size), matvec=matvec, dtype=w.dtype)
    # z, _ = gmres(operator, w, rtol=1e-12,  atol=1e-13,
    #     maxiter=200, restart=min(50, w.size))

    zvc = z[:nv*nc].reshape(nv, nc)
    zvo = z[nv*nc:nv*nc+nv*no].reshape(nv, no)
    zoc = z[nv*nc+nv*no:].reshape(no, nc)
    z1a = xp.hstack((zvc, zvo))
    z1b = xp.vstack((zoc, zvc))
    time1 = log.timer('Z-vector equation solver', *time0)

    # 5.1 construct W matrix
    z1ao = xp.empty((2, nao, nao))
    z1ao[0] = orbv_a @ z1a @ orbo_a.T
    z1ao[1] = orbv_b @ z1b @ orbo_b.T
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)) / 2)  # G_{\mu\nu}^{\sigma}[Z^S]
    im0a = xp.zeros((nao, nao))
    im0b = xp.zeros((nao, nao))
    im0a[:ni, :ni] += fockamo[:ni, :ni]  # ground state
    im0b[:nc, :nc] += fockbmo[:nc, :nc]  # ground state
    im0a[:ni, :ni] += orbo_a.T @ veff0doo[0] @ orbo_a
    im0a[:ni, :ni] += xp.einsum('aj,ai->ij', veff0mo[nc:, :ni], v)
    im0a[:ni, :ni] += xp.einsum('ik,jk->ij', dooa, fockamo[:ni, :ni])
    im0a[:ni, ni:] = orbo_a.T @ veff0doo[0] @ orbv_a
    im0a[:ni, ni:] += xp.einsum('ba,bi->ia', veff0mo[nc:, ni:], v)
    im0a[:ni, ni:] += xp.einsum('ij,aj->ia', dooa, fockamo[ni:, :ni])
    im0b[:nc, :nc] += orbo_b.T @ veff0doo[1] @ orbo_b
    im0b[:nc, nc:] = orbo_b.T @ veff0doo[1] @ orbv_b
    im0b[nc:, nc:] = xp.einsum('ac,bc->ab', dvvb, fockbmo[nc:, nc:])
    im0b[nc:, nc:] += xp.einsum('bi,ai->ab', veff0mo[nc:, :ni], v)
    im0b[nc:, :nc] += xp.einsum('ab,ib->ai', dvvb, fockbmo[:nc, nc:])  # ROKS particle term
    im0b[nc:, :nc] += xp.einsum('ij,aj->ai', veff0mo[:nc, :ni], v)
    im0 = im0a + im0b

    # 5.2 add Z
    im0[:ni, :] += (orbo_a.T @ veff[0] @ mo_coeff[0])
    im0[:nc, :] += (orbo_b.T @ veff[1] @ mo_coeff[1])
    im0[:nc, ni:] += xp.einsum('bi,ba->ia', z1a[:, :nc], fockamo[ni:, ni:]) / 2
    im0[ni:, :ni] += xp.einsum('aj,ij->ai', z1a, fockamo[:ni, :ni]) / 2
    im0[nc:ni, :nc] += xp.einsum('at,ai->ti', z1a[:, nc:], fockamo[ni:, :nc]) / 2
    im0[nc:ni, ni:] += xp.einsum('bt,ba->ta', z1a[:, nc:], fockamo[ni:, ni:]) / 2
    im0[ni:, :nc] += xp.einsum('aj,ij->ai', z1b[no:], fockbmo[:nc, :nc]) / 2
    im0[:nc, nc:] += xp.einsum('bi,ba->ia', z1b, fockbmo[nc:, nc:]) / 2
    im0[nc:ni, :nc] += xp.einsum('tj,ij->ti', z1b[:no], fockbmo[:nc, :nc]) / 2
    im0[nc:ni, ni:] += xp.einsum('ti,ai->ta', z1b[:no], fockbmo[ni:, :nc]) / 2

    # 5.3 add correct term
    im0 += dq
    im0 = mo_coeff[0] @ im0 @ mo_coeff[0].T

    # 6. derivative of coordinate
    dmz1dooa = (z1ao[0] + z1ao[0].T) / 2 + dmzooa  # T_{\mu\nu}^{\alpha} + Z^{S,\alpha}
    dmz1dvvb = (z1ao[1] + z1ao[1].T) / 2 + dmzvvb  # T_{\mu\nu}^{\beta} + Z^{S,\beta}
    oo0a = orbo_a @ orbo_a.T
    oo0b = orbo_b @ orbo_b.T
    oo0j = oo0a + oo0b  # contract with F^{HF,\sigma}
    dmHFao = dmHFaao + dmHFbao
    as_dm1 = oo0a + oo0b + dmz1dooa + dmz1dvvb  # follow CPU version naming convention
    as_dm1 = (as_dm1 + as_dm1.T) * 0.5

    if gpu:
        from gpu4pyscf.grad import tduks as gpu_tduks_grad
        from gpu4pyscf.grad import rhf as gpu_rhf_grad

        mf_grad = gpu_rhf_grad.Gradients(mf)
        h1 = xp.asarray(mf_grad.get_hcore(mol))
        s1 = xp.asarray(mf_grad.get_ovlp(mol))
        dh_ground_and_td = gpu_rhf_grad.contract_h1e_dm(mol, h1, as_dm1, hermi=1)
        ds = gpu_rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0)
        dh1e_ground_and_td = gpu_rhf_grad.int3c2e.get_dh1e(mol, as_dm1)  # 1/r like terms

        dmz1doo = dmz1dooa + dmz1dvvb
        oo0 = oo0a + oo0b
        if with_k:
            # # density fitting derivative will use 
            # if hasattr(dmt, 'symmetrize'):
            #     dmt_T = tag_array(dmt.T, factor_l=dmt.factor_r, factor_r=dmt.factor_l)
            # else:
            #     dmt_T = dmt.T
            # dms = [[_tag_factorize_dm(dmz1doo + oo0, hermi=1), _tag_factorize_dm(oo0, hermi=1)],
            #     [_tag_factorize_dm(dmz1dooa + oo0a, hermi=1), oo0a],
            #     [_tag_factorize_dm(dmz1doob + oo0b, hermi=1), oo0b],
            #     [dmt, dmt_T]]

            dmt_T = dmt.T
            dms = [[2 * dmz1doo + oo0, oo0],
                [2 * dmz1dooa + oo0a, oo0a],
                [2 * dmz1dvvb + oo0b, oo0b],
                [dmt, dmt_T]]
            j_factors = [0.5, 0, 0, 0]
            k_factors = [0, hyb, hyb, 2 * hyb]
            dvhf = jk_energies_per_atom(mf, dms, j_factors, k_factors, sum_results=True)
        else:
            dms = [[2 * dmz1doo + oo0, oo0]]
            j_factors = [0.5]
            k_factors = [0]
            dvhf = jk_energies_per_atom(mf, dms, j_factors, k_factors, sum_results=True)

        # if with_k and omega != 0:
        #     j_factors = [0, 0, 0]
        #     k_factors = [alpha - hyb, alpha - hyb, 2 * (alpha - hyb)]
        #     dvhf += td.jk_energies_per_atom(dms[1:], j_factors, k_factors, omega=omega, sum_results=True)
        # time1 = log.timer('2e AO integral derivatives', *time1)

        z1ao = z1ao.view(xp.ndarray)
        fxcz1 = gpu_tduks_grad._contract_xc_kernel(
            tdro, mf.xc, z1ao, None, False, False
        )[0]
        veff1_0 = vxc1[:, 1:]
        veff1_1 = f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:]
        veff1_0_a, veff1_0_b = veff1_0
        veff1_1_a, veff1_1_b = veff1_1

        de = dh_ground_and_td + xp.asnumpy(dh1e_ground_and_td) - ds + 2 * dvhf
        dveff1_0 = gpu_rhf_grad.contract_h1e_dm(mol, veff1_0_a, oo0a + dmz1dooa, hermi=0)
        dveff1_0 += gpu_rhf_grad.contract_h1e_dm(mol, veff1_0_b, oo0b + dmz1dvvb, hermi=0)
        dveff1_1 = gpu_rhf_grad.contract_h1e_dm(mol, veff1_1_a, oo0a, hermi=1)
        dveff1_1 += gpu_rhf_grad.contract_h1e_dm(mol, veff1_1_b, oo0b, hermi=1)
        dveff1_2 = gpu_rhf_grad.contract_h1e_dm(mol, f1vo[1:], dmt, hermi=0) * 2
        de += dveff1_0 + dveff1_1 + dveff1_2

        correction = gpu_rhf_grad.contract_h1e_dm(mol, h1, dmHFao, hermi=1)
        correction += gpu_rhf_grad.int3c2e.get_dh1e(mol, dmHFao).get()
        dm_delta = dmtco - dmtov
        dm_pairs = [
            [scorrao, dmSao], [oo0j, dmHFao], [oo0a, dmHFaao],
            [oo0b, dmHFbao], [dm_delta, dm_delta],
            # K_il[A_jk] is contracted with B_il in the CPU branch, so the
            # pair-energy API must receive B.T as its second density.
            [dmtcv, dmtco.T], [dmtcv, dmtov.T], [dmtco, dmtov.T],
            [dmtcv, dmtoo.T], [dmtco, dmtoo.T], [dmtov, dmtoo.T]
        ]

        # _jk_energies_per_atom uses energy-derivative normalization:
        # J(A, B) = J_cpu_cross(A, B) / 2 and
        # K(A, B) = -K_cpu_cross(A, B) / 4.
        j_factors = [0.0, 2.0, 0.0, 0.0, -c2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        k_factors = [-4.0, 0.0, 4.0, 4.0, 0.0, 8.0 * c3, 8.0 * c3,
            4.0 * c2, 8.0 * c4, 8.0 * c5, 8.0 * c5]
        correction += jk_energies_per_atom(mf, dm_pairs,
            j_factor=j_factors, k_factor=k_factors, sum_results=True,)

        de += correction * fglobal
        de = xp.asarray(de)
    else:
        mf_grad = td.base._scf.nuc_grad_method()
        hcore_deriv = mf_grad.hcore_generator(mol)
        s1 = mf_grad.get_ovlp(mol)
        as_dm1 = oo0a + oo0b + dmz1dooa + dmz1dvvb

        if with_k:
            vj, vk = td.get_jk(mol, (oo0a, dmz1dooa, oo0b, dmz1dvvb))
            vj = vj.reshape(2, 2, 3, nao, nao)
            vk = vk.reshape(2, 2, 3, nao, nao) * hyb
            veff1 = vj[0] + vj[1] - vk
            vk1 = -td.get_k(mol, (dmt, dmt.T)) * hyb
        else:
            vj = td.get_j(mol, (oo0a, dmz1dooa, oo0b, dmz1dvvb))
            vj = vj.reshape(2, 2, 3, nao, nao)
            veff1 = vj[0] + vj[1]
            veff1 = xp.stack((veff1, veff1))

        dm = (scorrao, dmSao, dmHFaao, dmHFbao, dmtcv, dmtco, dmtov, dmtoo,
               oo0a, oo0b, oo0j, dmtcv.T, dmtco.T, dmtov.T, dmtoo.T)
        vjcg, vkcg = td.get_jk(mol, dm, hermi=0)  # correct potential gradient

        fxcz1 = tduks_grad._contract_xc_kernel(
            tdro, mf.xc, z1ao, None, False, False, max_memory)[0]
        veff1[:, 0] += vxc1[:, 1:]
        veff1[:, 1] += (f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:])
        veff1a, veff1b = veff1
        time1 = log.timer('2e AO integral derivatives', *time1)

        # 7. combine upper result
        if atmlst is None:
            atmlst = range(mol.natm)
        offsetdic = mol.offset_nr_by_atom()
        de = xp.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = offsetdic[ia]
            h1ao = hcore_deriv(ia)
            de[k] = lib.einsum('xpq,pq->x', h1ao, as_dm1)
            de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], oo0a[p0:p1]) * 2
            de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], oo0b[p0:p1]) * 2

            de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], dmz1dooa[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], dmz1dvvb[p0:p1])
            de[k] += lib.einsum('xpq,qp->x', veff1a[0, :, p0:p1], dmz1dooa[:, p0:p1])
            de[k] += lib.einsum('xpq,qp->x', veff1b[0, :, p0:p1], dmz1dvvb[:, p0:p1])
            de[k] += lib.einsum('xij,ij->x', veff1a[1, :, p0:p1], oo0a[p0:p1])*2
            de[k] += lib.einsum('xij,ij->x', veff1b[1, :, p0:p1], oo0b[p0:p1])*2
            if with_k:
                de[k] += lib.einsum('xpq,pq->x', vk1[0, :, p0:p1], dmt[p0:p1]) * 2
                de[k] += lib.einsum('xpq,pq->x', vk1[1, :, p0:p1], dmt.T[p0:p1]) * 2
            if td.base.collinear_samples > 0:
                de[k] += lib.einsum('xpq,pq->x', f1vo[1:, p0:p1], dmt[p0:p1]) * 2
                de[k] += lib.einsum('xpq,pq->x', f1vo[1:, p0:p1], dmt.T[p0:p1]) * 2
            de[k] -= lib.einsum('xpq,pq->x', s1[:, p0:p1], im0[p0:p1])
            de[k] -= lib.einsum('xpq,qp->x', s1[:, p0:p1], im0[:, p0:p1])

            # correct term
            # dm = (scorrao, dmSao, dmHFaao, dmHFbao, dmtcv, dmtco, dmtov, dmtoo, oo0a, oo0b, oo0j)
            # \partial F_{\mu\nu}^{S} / \partial\xi
            de[k] += lib.einsum("xpq,pq->x", vkcg[0, :, p0:p1], dmSao[p0:p1]) * fglobal
            de[k] += lib.einsum("xpq,pq->x", vkcg[0, :, p0:p1], dmSao.T[p0:p1]) * fglobal
            de[k] += lib.einsum("xpq,pq->x", vkcg[1, :, p0:p1], scorrao[p0:p1]) * fglobal
            de[k] += lib.einsum("xpq,pq->x", vkcg[1, :, p0:p1], scorrao.T[p0:p1]) * fglobal
            # \partial F_{\mu\nu}^{HF,\sigma} / \partial\xi
            de[k] += lib.einsum('xpq,pq->x', h1ao, dmHFao) * fglobal
            de[k] += lib.einsum("xpq,pq->x", vjcg[10, :, p0:p1], dmHFao[p0:p1]) * fglobal
            de[k] += lib.einsum("xpq,pq->x", vjcg[10, :, p0:p1], dmHFao.T[p0:p1]) * fglobal
            de[k] += lib.einsum("xpq,pq->x", vjcg[2, :, p0:p1] + vjcg[3, :, p0:p1], oo0j[p0:p1]) * fglobal
            de[k] += lib.einsum("xpq,pq->x", vjcg[2, :, p0:p1] + vjcg[3, :, p0:p1], oo0j.T[p0:p1]) * fglobal
            de[k] -= lib.einsum('xpq,pq->x', vkcg[8, :, p0:p1], dmHFaao[p0:p1]) * fglobal
            de[k] -= lib.einsum('xpq,pq->x', vkcg[8, :, p0:p1], dmHFaao.T[p0:p1]) * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[2, :, p0:p1], oo0a[p0:p1]) * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[2, :, p0:p1], oo0a.T[p0:p1]) * fglobal
            de[k] -= lib.einsum('xpq,pq->x', vkcg[9, :, p0:p1], dmHFbao[p0:p1]) * fglobal
            de[k] -= lib.einsum('xpq,pq->x', vkcg[9, :, p0:p1], dmHFbao.T[p0:p1]) * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[3, :, p0:p1], oo0b[p0:p1]) * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[3, :, p0:p1], oo0b.T[p0:p1]) * fglobal
            # \partial J^{ao} / \partial\xi
            de[k] -= lib.einsum("xpq,pq->x", vjcg[5, :, p0:p1], dmtco[p0:p1]) * c2 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vjcg[5, :, p0:p1], dmtco.T[p0:p1]) * c2 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vjcg[6, :, p0:p1], dmtov[p0:p1]) * c2 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vjcg[6, :, p0:p1], dmtov.T[p0:p1]) * c2 * fglobal
            de[k] += lib.einsum("xpq,pq->x", vjcg[6, :, p0:p1], dmtco[p0:p1]) * c2 * fglobal
            de[k] += lib.einsum("xpq,pq->x", vjcg[6, :, p0:p1], dmtco.T[p0:p1]) * c2 * fglobal
            de[k] += lib.einsum("xpq,pq->x", vjcg[5, :, p0:p1], dmtov[p0:p1]) * c2 * fglobal
            de[k] += lib.einsum("xpq,pq->x", vjcg[5, :, p0:p1], dmtov.T[p0:p1]) * c2 * fglobal
            # \partial K^{ao} / \partial\xi
            de[k] -= lib.einsum("xpq,pq->x", vkcg[4, :, p0:p1], dmtco[p0:p1]) * 2 * c3 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[11, :, p0:p1], dmtco.T[p0:p1]) * 2 * c3 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[5, :, p0:p1], dmtcv[p0:p1]) * 2 * c3 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[12, :, p0:p1], dmtcv.T[p0:p1]) * 2 * c3 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[4, :, p0:p1], dmtov[p0:p1]) * 2 * c3 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[11, :, p0:p1], dmtov.T[p0:p1]) * 2 * c3 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[6, :, p0:p1], dmtcv[p0:p1]) * 2 * c3 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[13, :, p0:p1], dmtcv.T[p0:p1]) * 2 * c3 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[5, :, p0:p1], dmtov[p0:p1]) * c2 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[12, :, p0:p1], dmtov.T[p0:p1]) * c2 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[6, :, p0:p1], dmtco[p0:p1]) * c2 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[13, :, p0:p1], dmtco.T[p0:p1]) * c2 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[4, :, p0:p1], dmtoo[p0:p1]) * 2 * c4 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[11, :, p0:p1], dmtoo.T[p0:p1]) * 2 * c4 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[7, :, p0:p1], dmtcv[p0:p1]) * 2 * c4 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[14, :, p0:p1], dmtcv.T[p0:p1]) * 2 * c4 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[5, :, p0:p1], dmtoo[p0:p1]) * 2 * c5 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[12, :, p0:p1], dmtoo.T[p0:p1]) * 2 * c5 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[7, :, p0:p1], dmtco[p0:p1]) * 2 * c5 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[14, :, p0:p1], dmtco.T[p0:p1]) * 2 * c5 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[6, :, p0:p1], dmtoo[p0:p1]) * 2 * c5 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[13, :, p0:p1], dmtoo.T[p0:p1]) * 2 * c5 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[7, :, p0:p1], dmtov[p0:p1]) * 2 * c5 * fglobal
            de[k] -= lib.einsum("xpq,pq->x", vkcg[14, :, p0:p1], dmtov.T[p0:p1]) * 2 * c5 * fglobal
            # de[k] += td.extra_force(ia, locals())  # extension, here always zero
    log.timer('SF-up-TDA(ROKS) nuclear gradients', *time0)
    return de


class _RO2U:
    '''Wrap ROKS mf, make mo_coeff and mo_occ have same dim with UKS'''
    def __init__(self, td_grad, mf, sftda):
        xp = array_module(mf)
        self.mol = td_grad.mol
        self.verbose = getattr(td_grad, 'verbose', logger.INFO)
        self.stdout = getattr(td_grad, 'stdout', sys.stdout)
        self.base = self
        self._scf = mf.copy()
        self._scf.mo_occ = xp.asarray(sftda.mo_occ)
        self._scf.mo_coeff = xp.asarray(sftda.mo_coeff)
        self._scf.mo_energy = xp.asarray(sftda.mo_energy)
        self.collinear_samples = sftda.collinear_samples
        self.exclude_nlc = True


class SFD_gradient(rohf_grad.Gradients):
    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20) + 20
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)
    dsolve_lindep = getattr(__config__, 'lib_linalg_helper_dsolve_lindep', 1e-13)

    def __init__(self, td, method=1, state=1):
        self.base = td
        self.base._scf = td.mf
        self.base.exclude_nlc = True
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
        return grad_elec(self, atmlst=atmlst)

    def kernel(self, atmlst=None):
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        de = self.grad_elec(atmlst=atmlst)
        self.de = de + nuclear_gradient(self.base.mf, atmlst)
        self._finalize()
        return self.de

    def _finalize(self):
        logger.note(self,
            '--------- %s gradients for state %d ----------',
            self.base.__class__.__name__,
            self.state
        )
        self._write(self.mol, asnumpy(self.de), self.atmlst)
        logger.note(self, '----------------------------------------------')
