#!/usr/bin/env python
import sys
from pyscf import lib, __config__
from pyscf.lib import logger
from pyscf.grad import uhf as uhf_grad
from pyscf.grad import rohf as rohf_grad
from pyscf.grad import tdrks as tdrks_grad

from ..xtda import _st2so
from ...utils.backend import asnumpy
from ._backend import array_module, is_gpu_mf, nuclear_gradient, ucphf_module


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
    gpu = is_gpu_mf(mf)
    xp = array_module(mf)
    mo_coeff = xp.asarray(mf.mo_coeff)
    mo_occ = xp.asarray(mf.mo_occ)
    nao = mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmvo = xp.asarray(((dmvo[0] + dmvo[0].T) * .5,
                       (dmvo[1] + dmvo[1].T) * .5))
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

        opt = getattr(ni, "gdftopt", None)
        if opt is None:
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
        eval_mol = opt._sorted_mol
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])
        dmvo = opt.sort_orbitals(dmvo, axis=[1, 2])
        if dmoo is not None:
            dmoo = opt.sort_orbitals(dmoo, axis=[1, 2])

    f1vo = xp.zeros((2,4,nao,nao))
    deriv = 2
    if dmoo is not None:
        f1oo = xp.zeros((2,4,nao,nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = xp.zeros((2,4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = xp.zeros((2,4,nao,nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = (
            (gpu_tdrks_grad._lda_eval_mat_ if gpu else tdrks_grad._lda_eval_mat_), 1
        )
    elif xctype == 'GGA':
        fmat_, ao_deriv = (
            (gpu_tdrks_grad._gga_eval_mat_ if gpu else tdrks_grad._gga_eval_mat_), 2
        )
    elif xctype == 'MGGA':
        fmat_, ao_deriv = (
            (gpu_tdrks_grad._mgga_eval_mat_ if gpu else tdrks_grad._mgga_eval_mat_), 2
        )
        logger.warn(td_grad, 'TDUKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    if mf.do_nlc():
        raise NotImplementedError("TDDFT gradient with NLC contribution is not supported yet. "
                                  "Please set exclude_nlc field of tdscf object to True, "
                                  "which will turn off NLC contribution in the whole TDDFT calculation.")

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
            dmvo_a = dmvo[0, mask[:, None], mask]
            dmvo_b = dmvo[1, mask[:, None], mask]
        else:
            coeff_a, coeff_b = mo_coeff
            dmvo_a, dmvo_b = dmvo

        rho = xp.asarray((
            ni.eval_rho2(eval_mol, ao0, coeff_a, mo_occ[0], mask, xctype, with_lapl=False),
            ni.eval_rho2(eval_mol, ao0, coeff_b, mo_occ[1], mask, xctype, with_lapl=False),
        ))
        #TODO(WHB): libxc gpu version used in gpu4pyscf may have problem
        vxc, fxc, kxc = ni.eval_xc_eff(
            xc_code, rho, deriv, xctype=xctype
        )[1:]

        rho1 = xp.asarray((
            ni.eval_rho(eval_mol, ao0, dmvo_a, mask, xctype, hermi=1, with_lapl=False),
            ni.eval_rho(eval_mol, ao0, dmvo_b, mask, xctype, hermi=1, with_lapl=False),
        ))
        if xctype == 'LDA':
            rho1 = rho1[:,xp.newaxis].copy()
        if gpu:
            tmp = gpu_contract('axg,axbyg->byg', rho1, fxc)
            wv = gpu_contract('byg,g->byg', tmp, weight)
        else:
            wv = xp.einsum('axg,axbyg,g->byg', rho1, fxc, weight)
        fmat_(eval_mol, f1vo[0], ao, wv[0], mask, shls_slice, ao_loc)
        fmat_(eval_mol, f1vo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None:
            if gpu:
                dmoo_a = dmoo[0, mask[:, None], mask]
                dmoo_b = dmoo[1, mask[:, None], mask]
            else:
                dmoo_a, dmoo_b = dmoo
            rho2 = xp.asarray((
                ni.eval_rho(eval_mol, ao0, dmoo_a, mask, xctype, hermi=1, with_lapl=False),
                ni.eval_rho(eval_mol, ao0, dmoo_b, mask, xctype, hermi=1, with_lapl=False),
            ))
            if xctype == 'LDA':
                rho2 = rho2[:,xp.newaxis].copy()
            if gpu:
                tmp = gpu_contract('axg,axbyg->byg', rho2, fxc)
                wv = gpu_contract('byg,g->byg', tmp, weight)
            else:
                wv = xp.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(eval_mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(eval_mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_vxc:
            wv = vxc * weight
            fmat_(eval_mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(eval_mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_kxc:
            if gpu:
                tmp = gpu_contract('axg,axbyczg->byczg', rho1, kxc)
                tmp = gpu_contract('byg,byczg->czg', rho1, tmp)
                wv = gpu_contract('czg,g->czg', tmp, weight)
            else:
                wv = xp.einsum(
                    'axg,byg,axbyczg,g->czg', rho1, rho1, kxc, weight
                )
            fmat_(eval_mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(eval_mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[:,1:] *= -1
    if gpu:
        f1vo = opt.unsort_orbitals(f1vo, axis=[2, 3])
    if f1oo is not None:
        f1oo[:,1:] *= -1
        if gpu:
            f1oo = opt.unsort_orbitals(f1oo, axis=[2, 3])
    if v1ao is not None:
        v1ao[:,1:] *= -1
        if gpu:
            v1ao = opt.unsort_orbitals(v1ao, axis=[2, 3])
    if k1ao is not None:
        k1ao[:,1:] *= -1
        if gpu:
            k1ao = opt.unsort_orbitals(k1ao, axis=[2, 3])
    return f1vo, f1oo, v1ao, k1ao



def grad_elec(td, atmlst=None, max_memory=2000,
              verbose=logger.INFO):
    """ROKS spin-conserving XTDA excitation-energy nuclear derivative."""
    log = logger.new_logger(td, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mf = td.base._scf
    mol = td.mol
    xp = array_module(mf)
    gpu = is_gpu_mf(mf)
    restricted_energy = xp.asarray(mf.mo_energy)
    restricted_coeff = xp.asarray(mf.mo_coeff)
    restricted_occ = xp.asarray(mf.mo_occ)
    mo_energy = xp.stack((restricted_energy, restricted_energy))
    mo_coeff = xp.stack((restricted_coeff, restricted_coeff))
    mo_occ = xp.stack((restricted_occ >= 1, restricted_occ >= 2)).astype(float)
    nao = mo_coeff[0].shape[0]
    occidx_a = xp.where(mo_occ[0] == 1)[0]
    viridx_a = xp.where(mo_occ[0] == 0)[0]
    occidx_b = xp.where(mo_occ[1] == 1)[0]
    viridx_b = xp.where(mo_occ[1] == 0)[0]
    nc = len(occidx_b)
    nv = len(viridx_a)
    no = len(occidx_a) - len(occidx_b)
    ni = nc+no  # convenient use, alpha occ
    orbv_a = mo_coeff[0][:, ni:]
    orbo_a = mo_coeff[0][:, :ni]
    orbv_b = mo_coeff[1][:, nc:]
    orbo_b = mo_coeff[1][:, :nc]
    # XTDA return spin tensor basis vector, here use spin orbital basis
    v = xp.asarray(_st2so(xp.asarray(td.v), nc, no, nv))
    v = v[:, td.state-1]
    v_cva = v[:nc*nv].reshape(nc, nv)
    v_ova = v[nc*nv:(nc+no)*nv].reshape(no, nv)
    v_cob = v[(nc+no)*nv:(nc+no)*nv+nc*no].reshape(nc, no)
    v_cvb = v[(nc+no)*nv+nc*no:].reshape(nc, nv)
    va = xp.vstack((v_cva, v_ova)).T
    vb = xp.hstack((v_cob, v_cvb)).T

    # 1. internel variable
    dvva = xp.einsum('ai,bi->ab', va, va)  # T_{ab}^{\alpha}
    dvvb = xp.einsum('ai,bi->ab', vb, vb)
    dooa = -xp.einsum('ai,aj->ij', va, va)  # T_{ij}^{\alpha}
    doob = -xp.einsum('ai,aj->ij', vb, vb)
    dmza = orbv_a @ dvva @ orbv_a.T + orbo_a @ dooa @ orbo_a.T  # T_{\mu\nu}^{\alpha}
    dmzb = orbv_b @ dvvb @ orbv_b.T + orbo_b @ doob @ orbo_b.T  # T_{\mu\nu}^{\beta}
    dmxa = orbv_a @ va @ orbo_a.T  # X_{\mu\nu}^{\alpha}
    dmxb = orbv_b @ vb @ orbo_b.T  # X_{\mu\nu}^{\beta}
    # correct coefficient
    s = mol.spin / 2
    c1 = 1 - xp.sqrt((s + 1) / s) + 1 / (2 * s)
    c2 = -1 + xp.sqrt((s + 1) / s) + 1 / (2 * s)
    c3 = -1 / (2 * s)
    # correct dmz
    va_cc = va[:, :nc]
    vb_cc = vb[no:, :]
    dmcc = xp.einsum('ai,aj->ij', va_cc, va_cc) * c2
    dmcc += xp.einsum('ai,aj->ij', va_cc, vb_cc) * c3
    dmcc += xp.einsum('ai,aj->ij', vb_cc, va_cc) * c3
    dmcc += xp.einsum('ai,aj->ij', vb_cc, vb_cc) * c1
    dmvv = xp.einsum('ai,bi->ab', va_cc, va_cc) * c1
    dmvv += xp.einsum('ai,bi->ab', va_cc, vb_cc) * c3
    dmvv += xp.einsum('ai,bi->ab', vb_cc, va_cc) * c3
    dmvv += xp.einsum('ai,bi->ab', vb_cc, vb_cc) * c2
    dmcorrao = orbo_b @ dmcc @ orbo_b.T + orbv_a @ dmvv @ orbv_a.T  # dm correct ao
    scorrao = orbo_a[:, nc:ni] @ (xp.eye(no)/2) @ orbo_a[:, nc:ni].T  # s correct ao
    # ROKS fock 
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

    # 2. functional derivative, include derivative respect to mo_coeff and coordinate
    tdro = _RO2U(td, mf)
    ni_ = mf._numint
    ni_.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni_.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # f1vo: f^{xc}[X], f1oo: f^{xc}[T], vxc1: v^{xc}[\rho], k1ao: g^{xc}[X,X] 
    # and their derivative
    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(
        tdro, mf.xc, (dmxa, dmxb), (dmza, dmzb), True, True, max_memory
    )

    # 3.1 construct Q matrix
    with_k = ni_.libxc.is_hybrid_xc(mf.xc)
    if with_k:
        dm = xp.stack((dmza, dmzb, dmxa, dmxb))
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
        wvoa += xp.einsum('ba,bi->ai', veff0moa[ni:, ni:], va) * 2
        wvoa += xp.einsum('ij,aj->ai', dooa, fockamo[ni:, :ni]) * 2
        wvoa -= xp.einsum('ij,aj->ai', veff0moa[:ni, :ni], va) * 2
        wvoa -= xp.einsum('ab,ib->ai', dvva, fockamo[:ni, ni:]) * 2
        wvob += xp.einsum('ba,bi->ai', veff0mob[nc:, nc:], vb) * 2
        wvob += xp.einsum('ij,aj->ai', doob, fockbmo[nc:, :nc]) * 2
        wvob -= xp.einsum('ij,aj->ai', veff0mob[:nc, :nc], vb) * 2
        wvob -= xp.einsum('ab,ib->ai', dvvb, fockbmo[:nc, nc:]) * 2
    else:
        dm = xp.stack((dmza, dmzb, dmxa, dmxb))
        vj= mf.get_j(mol, dm, hermi=0)  # g_{\mu\nu}^{\sigma}[T]
        veff0doo = vj[0] + vj[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = (orbv_a.T @ veff0doo[0] @ orbo_a) * 2
        wvob = (orbv_b.T @ veff0doo[1] @ orbo_b) * 2
        veff0moa = mo_coeff[0].T @ (vj[2] + vj[3] + f1vo[0, 0]) @ mo_coeff[0]
        veff0mob = mo_coeff[1].T @ (vj[2] + vj[3] + f1vo[1, 0]) @ mo_coeff[1]
        wvoa += xp.einsum('ba,bi->ai', veff0moa[ni:, ni:], va) * 2
        wvoa += xp.einsum('ij,aj->ai', dooa, fockamo[ni:, :ni]) * 2
        wvoa -= xp.einsum('ij,aj->ai', veff0moa[:ni, :ni], va) * 2
        wvoa -= xp.einsum('ab,ib->ai', dvva, fockamo[:ni, ni:]) * 2
        wvob += xp.einsum('ba,bi->ai', veff0mob[nc:, nc:], vb) * 2
        wvob += xp.einsum('ij,aj->ai', doob, fockbmo[nc:, :nc]) * 2
        wvob -= xp.einsum('ij,aj->ai', veff0mob[:nc, :nc], vb) * 2
        wvob -= xp.einsum('ab,ib->ai', dvvb, fockbmo[:nc, nc:]) * 2

    wcca = (orbo_a.T @ veff0doo[0] @ orbo_a) * 2
    wcca += xp.einsum('ik,jk->ij', dooa, fockamo[:ni, :ni]) * 2
    wcca += xp.einsum('aj,ai->ij', veff0moa[ni:, :ni], va) * 2
    wvvb = xp.einsum('ac,bc->ab', dvvb, fockbmo[nc:, nc:]) * 2
    wvvb += xp.einsum('bi,ai->ab', veff0mob[nc:, :nc], vb) * 2
    wvc = wvoa[:, :nc] + wvob[no:, :]  # (Q_{ia} - Q_{ai})
    wvo = wvoa[:, nc:] - (wvvb - wvvb.T)[no:, :no]  # (Q_{ta} - Q_{at})
    woc = wvob[:no, :] - (wcca - wcca.T)[nc:, :nc]  # (Q_{it} - Q_{ti})

    # 3.2 add correct term
    if gpu:
        FSao = mf.get_k(mol, xp.stack((scorrao, dmcorrao)), hermi=0)
    else:
        from pyscf import scf

        FSao = scf.ROHF(mol).get_k(
            mol, xp.stack((scorrao, dmcorrao)), hermi=0
        )
    FSao = xp.asarray(FSao)
    FSmo = mo_coeff[0].T @ FSao @ mo_coeff[0]
    wvc += xp.einsum('ij,aj->ai', dmcc, FSmo[0, ni:, :nc]) * 2
    wvc -= xp.einsum('ab,ib->ai', dmvv, FSmo[0, :nc, ni:]) * 2
    wvo += FSmo[1, ni:, nc:ni]
    wvo -= xp.einsum('ab,tb->at', dmvv, FSmo[0, nc:ni, ni:]) * 2
    woc += xp.einsum('ij,tj->ti', dmcc, FSmo[0, nc:ni, :nc]) * 2
    woc -= FSmo[1, nc:ni, :nc]
    w = xp.hstack((wvc.ravel(), wvo.ravel(), woc.ravel()))

    # 4. constuct G[Z^S] and solve Z-vector equation
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
    else:
        z = lib.solve(
            matvec, w, tol=1e-12, max_cycle=td.cphf_max_cycle,
            lindep=td.dsolve_lindep,
        )
    zvc = z[:nv*nc].reshape(nv, nc)
    zvo = z[nv*nc:nv*nc+nv*no].reshape(nv, no)
    zoc = z[nv*nc+nv*no:].reshape(no, nc)
    z1a = xp.hstack((zvc, zvo))
    z1b = xp.vstack((zoc, zvc))
    time1 = log.timer('Z-vector equation solver', *time0)

    # 5.1 construct complete W matrix
    # correct term make Za not equal Zb need construct full W matrix
    z1ao = xp.empty((2, nao, nao))
    z1ao[0] = orbv_a @ z1a @ orbo_a.T
    z1ao[1] = orbv_b @ z1b @ orbo_b.T
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)) / 2)
    im0a = xp.zeros((nao, nao))
    im0b = xp.zeros((nao, nao))
    im0a[:ni, :ni] += fockamo[:ni, :ni]  # ground state
    im0b[:nc, :nc] += fockbmo[:nc, :nc]  # ground state
    im0a[:ni, :ni] += (orbo_a.T @ veff0doo[0] @ orbo_a)
    im0a[:ni, :ni] += xp.einsum('ik,jk->ij', dooa, fockamo[:ni, :ni])
    im0a[:ni, :ni] += xp.einsum('aj,ai->ij', veff0moa[ni:, :ni], va)
    im0a[ni:, ni:] = xp.einsum('ac,bc->ab', dvva, fockamo[ni:, ni:])
    im0a[ni:, ni:] += xp.einsum('bi,ai->ab', veff0moa[ni:, :ni], va)
    im0a[:ni, ni:] = (orbo_a.T @ veff0doo[0] @ orbv_a)
    im0a[:ni, ni:] += xp.einsum('ij,aj->ia', dooa, fockamo[ni:, :ni])
    im0a[:ni, ni:] += xp.einsum('ba,bi->ia', veff0moa[ni:, ni:], va)
    im0a[ni:, :ni] = xp.einsum('ab,ib->ai', dvva, fockamo[:ni, ni:])
    im0a[ni:, :ni] += xp.einsum('ij,aj->ai', veff0moa[:ni, :ni], va)
    im0b[:nc, :nc] += (orbo_b.T @ veff0doo[1] @ orbo_b)
    im0b[:nc, :nc] += xp.einsum('ik,jk->ij', doob, fockbmo[:nc, :nc])
    im0b[:nc, :nc] += xp.einsum('aj,ai->ij', veff0mob[nc:, :nc], vb)
    im0b[nc:, nc:] = xp.einsum('ac,bc->ab', dvvb, fockbmo[nc:, nc:])
    im0b[nc:, nc:] += xp.einsum('bi,ai->ab', veff0mob[nc:, :nc], vb)
    im0b[:nc, nc:] = (orbo_b.T @ veff0doo[1] @ orbv_b)
    im0b[:nc, nc:] += xp.einsum('ij,aj->ia', doob, fockbmo[nc:, :nc])
    im0b[:nc, nc:] += xp.einsum('ba,bi->ia', veff0mob[nc:, nc:], vb)
    im0b[nc:, :nc] = xp.einsum('ab,ib->ai', dvvb, fockbmo[:nc, nc:])
    im0b[nc:, :nc] += xp.einsum('ij,aj->ai', veff0mob[:nc, :nc], vb)
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
    scorrmo = xp.zeros((nao, nao))
    scorrmo[nc:ni, nc:ni] = xp.eye(no) / 2
    im0[:nc, :] += xp.einsum('ij,pj->ip', dmcc, FSmo[0, :, :nc])
    im0[ni:, :] += xp.einsum('ab,pb->ap', dmvv, FSmo[0, :, ni:])
    im0[nc:ni, :] += FSmo[1, nc:ni, :] / 2

    im0 = mo_coeff[0] @ im0 @ mo_coeff[0].T

    # 6. derivative of coordinate
    dmz1dvva = (z1ao[0] + z1ao[0].T) / 2 + dmza  # T_{\mu\nu}^{\alpha} + Z^{S,\alpha}
    dmz1doob = (z1ao[1] + z1ao[1].T) / 2 + dmzb  # T_{\mu\nu}^{\beta} + Z^{S,\beta}

    oo0a = orbo_a @ orbo_a.T
    oo0b = orbo_b @ orbo_b.T
    as_dm1 = oo0a + oo0b + dmz1dvva + dmz1doob  # follow CPU version naming convention
    as_dm1 = (as_dm1 + as_dm1.T) * 0.5
    fxcz1 = _contract_xc_kernel(
        tdro, mf.xc, z1ao, None, False, False, max_memory
    )[0]

    if gpu:
        from gpu4pyscf.grad import tduks as gpu_tduks_grad
        from gpu4pyscf.grad import rhf as gpu_rhf_grad

        mf_grad = gpu_rhf_grad.Gradients(mf)
        h1 = xp.asarray(mf_grad.get_hcore(mol))
        s1 = xp.asarray(mf_grad.get_ovlp(mol))
        dh_ground_and_td = gpu_rhf_grad.contract_h1e_dm(mol, h1, as_dm1, hermi=1)
        ds = gpu_rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0)
        dh1e_ground_and_td = gpu_rhf_grad.int3c2e.get_dh1e(mol, as_dm1)  # 1/r like terms

        get_veff = gpu_tduks_grad.Gradients.get_veff
        k_factor = hyb if with_k else 0.0
        dvhf = get_veff(
            td, mol,
            xp.stack(((dmz1dvva + dmz1dvva.T) * .5 + oo0a,
                      (dmz1doob + dmz1doob.T) * .5 + oo0b)),
            1.0, k_factor, hermi=1,
        )
        dvhf -= get_veff(
            td, mol,
            xp.stack(((dmz1dvva + dmz1dvva.T) * .5,
                      (dmz1doob + dmz1doob.T) * .5)),
            1.0, k_factor, hermi=1,
        )
        dvhf += 2 * get_veff(
            td, mol,
            xp.stack(((dmxa + dmxa.T) * .5, (dmxb + dmxb.T) * .5)),
            1.0, k_factor, hermi=1,
        )
        dvhf -= 2 * get_veff(
            td, mol,
            xp.stack(((dmxa - dmxa.T) * .5, (dmxb - dmxb.T) * .5)),
            0.0, k_factor, hermi=2,
        )
        de = dh_ground_and_td + xp.asnumpy(dh1e_ground_and_td) - ds + 2 * dvhf

        dveff1_0 = gpu_rhf_grad.contract_h1e_dm(mol, vxc1[0, 1:], oo0a + dmz1dvva, hermi=0)
        dveff1_0 += gpu_rhf_grad.contract_h1e_dm(mol, vxc1[1, 1:], oo0b + dmz1doob, hermi=0)
        veff1_1 = f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:]
        dveff1_1 = gpu_rhf_grad.contract_h1e_dm(mol, veff1_1[0], oo0a, hermi=1)
        dveff1_1 += gpu_rhf_grad.contract_h1e_dm(mol, veff1_1[1], oo0b, hermi=1)
        dveff1_2 = gpu_rhf_grad.contract_h1e_dm(mol, f1vo[0, 1:] * 2, dmxa, hermi=0)
        dveff1_2 += gpu_rhf_grad.contract_h1e_dm(mol, f1vo[1, 1:] * 2, dmxb, hermi=0)

        # correct term
        dcorr = get_veff(td, mol, scorrao + dmcorrao, j_factor=0.0, k_factor=-1.0, hermi=1)
        dcorr -= get_veff(td, mol, scorrao, j_factor=0.0, k_factor=-1.0, hermi=1)
        dcorr -= get_veff(td, mol, dmcorrao, j_factor=0.0, k_factor=-1.0, hermi=1)
        de += dveff1_0 + dveff1_1 + dveff1_2 + 2 * dcorr
        de = xp.asarray(de)
        if atmlst is not None:
            de = de[xp.asarray(tuple(atmlst), dtype=int)]
    else:
        mf_grad = td.base._scf.nuc_grad_method()
        hcore_deriv = mf_grad.hcore_generator(mol)
        s1 = mf_grad.get_ovlp(mol)
        as_dm1 = oo0a + oo0b + dmz1dvva + dmz1doob
        dm = xp.stack((oo0a, dmz1dvva+dmz1dvva.T, dmxa, dmxa.T,
                       oo0b, dmz1doob+dmz1doob.T, dmxb, dmxb.T))
        if with_k:
            vj, vk = td.get_jk(mol, dm)
            vj = vj.reshape(2,4,3,nao,nao)
            vk = vk.reshape(2,4,3,nao,nao) * hyb
            # if omega != 0:
            #     vk += td.get_k(mol, dm, omega=omega).reshape(2,4,3,nao,nao) * (alpha-hyb)
            veff1 = vj[0] + vj[1] - vk
        else:
            vj = td.get_j(mol, dm).reshape(2,4,3,nao,nao)
            veff1 = vj[0] + vj[1]

        vkc = td.get_k(
            mol, xp.stack((scorrao, dmcorrao)), hermi=0
        )  # correct exchange potential
        veff1[:,0] += vxc1[:,1:]
        veff1[:,1] +=(f1oo[:,1:] + fxcz1[:,1:] + k1ao[:,1:]) * 2 # *2 for dmz1doo+dmz1oo.T
        veff1[:,2] += f1vo[:,1:]
        veff1[:,3] += f1vo[:,1:]
        veff1a, veff1b = veff1
        time1 = log.timer('2e AO integral derivatives', *time1)

        if atmlst is None:
            atmlst = range(mol.natm)
        offsetdic = mol.offset_nr_by_atom()
        de = xp.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = offsetdic[ia]
            h1ao = hcore_deriv(ia)
            de[k] = xp.einsum('xpq,pq->x', h1ao, as_dm1)
            de[k] += xp.einsum('xpq,pq->x', veff1a[0,:,p0:p1], oo0a[p0:p1])
            de[k] += xp.einsum('xpq,pq->x', veff1b[0,:,p0:p1], oo0b[p0:p1])
            de[k] += xp.einsum('xpq,qp->x', veff1a[0,:,p0:p1], oo0a[:,p0:p1])
            de[k] += xp.einsum('xpq,qp->x', veff1b[0,:,p0:p1], oo0b[:,p0:p1])

            de[k] += xp.einsum('xpq,pq->x', veff1a[0,:,p0:p1], dmz1dvva[p0:p1])
            de[k] += xp.einsum('xpq,pq->x', veff1b[0,:,p0:p1], dmz1doob[p0:p1])
            de[k] += xp.einsum('xpq,qp->x', veff1a[0,:,p0:p1], dmz1dvva[:,p0:p1])
            de[k] += xp.einsum('xpq,qp->x', veff1b[0,:,p0:p1], dmz1doob[:,p0:p1])

            de[k] -= xp.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
            de[k] -= xp.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

            de[k] += xp.einsum('xij,ij->x', veff1a[1,:,p0:p1], oo0a[p0:p1])
            de[k] += xp.einsum('xij,ij->x', veff1b[1,:,p0:p1], oo0b[p0:p1])
            de[k] += xp.einsum('xij,ij->x', veff1a[2,:,p0:p1], dmxa[p0:p1,:])*2
            de[k] += xp.einsum('xij,ij->x', veff1b[2,:,p0:p1], dmxb[p0:p1,:])*2
            de[k] += xp.einsum('xji,ij->x', veff1a[3,:,p0:p1], dmxa[:,p0:p1])*2
            de[k] += xp.einsum('xji,ij->x', veff1b[3,:,p0:p1], dmxb[:,p0:p1])*2

            # add correct term
            de[k] += xp.einsum('xpq,pq->x', vkc[0, :, p0:p1], dmcorrao[p0:p1])
            de[k] += xp.einsum('xpq,qp->x', vkc[0, :, p0:p1], dmcorrao[:, p0:p1])
            de[k] += xp.einsum('xpq,pq->x', vkc[1, :, p0:p1], scorrao[p0:p1])
            de[k] += xp.einsum('xpq,qp->x', vkc[1, :, p0:p1], scorrao[:, p0:p1])

            # de[k] += td.extra_force(ia, locals())  # extension, here always zero
    log.timer('SF-up-TDA(ROKS) nuclear gradients', *time0)
    return de


class _RO2U:
    '''Wrap ROKS mf, make mo_coeff and mo_occ have same dim with UKS'''
    def __init__(self, td_grad, mf):
        xp = array_module(mf)
        self.mol = td_grad.mol
        self.verbose = getattr(td_grad, 'verbose', logger.INFO)
        self.stdout = getattr(td_grad, 'stdout', sys.stdout)
        self.base = self
        self._scf = mf.copy()
        restricted_occ = xp.asarray(mf.mo_occ)
        self._scf.mo_occ = xp.stack(
            (restricted_occ >= 1, restricted_occ >= 2)
        ).astype(float)
        restricted_coeff = xp.asarray(mf.mo_coeff)
        self._scf.mo_coeff = xp.stack((restricted_coeff, restricted_coeff))
        restricted_energy = xp.asarray(mf.mo_energy)
        self._scf.mo_energy = xp.stack((restricted_energy, restricted_energy))


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
        self.verbose = 3
        
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
