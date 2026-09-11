#!/usr/bin/env python
from types import SimpleNamespace

from pyscf import __config__
from pyscf.lib import logger
from pyscf.dft import numint2c
from pyscf.grad import uhf as uhf_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.grad import tduks as tduks_grad
from pyscf.sftda.numint2c_sftd import mcfun_eval_xc_adapter_sf

from ...utils.backend import asnumpy
from ._backend import array_module, is_gpu_mf, nuclear_gradient, ucphf_module


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
        from gpu4pyscf.grad import tduks_sf as gpu_tduks_sf_grad
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
        fmat_, ao_deriv = (
            (gpu_tduks_sf_grad._gga_eval_mat_ if gpu else tdrks_grad._gga_eval_mat_), 2
        )
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
                wv = gpu_tduks_sf_grad.uks_sf_gga_wv1(rho1, fxc_sf, weight)
            else:
                wv = xp.einsum('yg,xyg,g->xg', rho1, 2 * fxc_sf, weight)
            fmat_(eval_mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if with_kxc:
                if gpu and xctype == 'GGA':
                    gv = gpu_tduks_sf_grad.uks_sf_gga_wv2_p(
                        rho1, kxc_sf, weight
                    )
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
                if xctype == 'GGA':
                    wv[:, 0] *= .5
            else:
                wv = xp.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(eval_mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(eval_mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if with_vxc:
            wv = vxc * weight
            if gpu and xctype == 'GGA':
                wv[:, 0] *= .5
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


def grad_elec(td, atmlst=None, max_memory=2000, verbose=logger.INFO):
    """electronic part of spin flip up TDA gradient in UKS reference state"""
    log = logger.new_logger(td, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td.mol
    mf = td.base._scf
    xp = array_module(mf)
    gpu = is_gpu_mf(mf)
    mo_coeff = xp.asarray(mf.mo_coeff)
    mo_energy = xp.asarray(mf.mo_energy)
    mo_occ = xp.asarray(mf.mo_occ)
    occidx_a = xp.where(mo_occ[0] > 0)[0]
    occidx_b = xp.where(mo_occ[1] > 0)[0]
    viridx_a = xp.where(mo_occ[0] == 0)[0]
    viridx_b = xp.where(mo_occ[1] == 0)[0]
    nc = len(occidx_b)
    nv = len(viridx_a)
    no = len(occidx_a) - len(occidx_b)
    ni = nc + no  # convenient use, alpha occ
    na = no + nv  # convenient use, beta vir
    orboa = mo_coeff[0][:, :ni]
    orbob = mo_coeff[1][:, :nc]
    orbva = mo_coeff[0][:, ni:]
    orbvb = mo_coeff[1][:, nc:]
    nao = mo_coeff[0].shape[0]

    v = xp.asarray(td.v[:, td.state - 1])
    v_cv = v[:nc*nv].reshape(nc, nv)
    v_co = v[nc*nv:nc*na].reshape(nc, no)
    v_ov = v[nc*na:nc*na+no*nv].reshape(no, nv)
    v_oo = v[nc*na+no*nv:].reshape(no, no)
    v_ca = xp.hstack((v_co, v_cv))
    v_oa = xp.hstack((v_oo, v_ov))
    v = xp.vstack((v_ca, v_oa)).T

    # 1. internel variable
    dooa = -xp.einsum("ai,aj->ij", v, v)  # T_{ij}
    dvvb = xp.einsum("ai,bi->ab", v, v)  # T_{ab}
    dmzooa = orboa @ dooa @ orboa.T  # T_{\mu\nu}^{\alpha}
    dmzoob = orbvb @ dvvb @ orbvb.T  # T_{\mu\nu}^{\beta}
    dmt = orbvb @ v @ orboa.T  # X_{\mu\nu}^{\alpha\beta}

    # 2. functional derivative, include derivative respect to mo_coeff and coordinate
    ni_ = mf._numint
    ni_.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni_.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # f1vo: f^{xc}[X], f1oo: f^{xc}[T], vxc1: v^{xc}[\rho], k1ao: g^{xc}[X,X] 
    # and their derivative
    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(
        td, mf.xc, dmt, (dmzooa, dmzoob), True, True, max_memory)

    # 3. construct Q matrix
    with_k = ni_.libxc.is_hybrid_xc(mf.xc)
    if with_k:
        vj0, vk0 = mf.get_jk(
            mol, xp.stack((dmzooa, dmzoob)), hermi=1
        )
        vj0 = xp.asarray(vj0)
        vk1 = xp.asarray(mf.get_k(mol, dmt, hermi=0)) * hyb
        vk0 = xp.asarray(vk0) * hyb
        # if omega != 0:
        #     vk0 += mf.get_k(
        #         mol, (dmzooa, dmzoob), hermi=1, omega=omega
        #     ) * (alpha - hyb)
        #     vk1 += mf.get_k(mol, dmt, hermi=0, omega=omega) * (alpha - hyb)

        veff0doo = vj0[0] + vj0[1] - vk0 + f1oo[:, 0] + k1ao[:, 0]
        wvoa = orbva.T @ veff0doo[0] @ orboa  # part of Q_{ia}^{\alpha}
        wvob = orbvb.T @ veff0doo[1] @ orbob  # Q_{ia}^{\beta}
        veff0mo = mo_coeff[1].T @ (f1vo[0] - vk1) @ mo_coeff[0]
        wvoa += xp.einsum("ac,ak->ck", veff0mo[nc:, ni:], v)  # part of Q_{ia}^{\alpha}
        wvob -= xp.einsum("kj,cj->ck", veff0mo[:nc, :ni], v)  # Q_{ai}^{\beta}
    else:
        vj0 = xp.asarray(mf.get_j(
            mol, xp.stack((dmzooa, dmzoob)), hermi=1
        ))
        veff0doo = vj0[0] + vj0[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = orbva.T @ veff0doo[0] @ orboa
        wvob = orbvb.T @ veff0doo[1] @ orbob
        veff0mo = mo_coeff[1].T @ f1vo[0] @ mo_coeff[0]
        wvoa += xp.einsum("ac,ak->ck", veff0mo[nc:, ni:], v)
        wvob -= xp.einsum("kj,cj->ck", veff0mo.T[:ni, :nc], v)

    # 4. constuct G[Z^S] and solve Z-vector equation
    vresp = mf.gen_response(hermi=1)
    def fvind(x):
        xa = x[0, :ni*nv].reshape(nv, ni)
        xb = x[0, ni*nv:].reshape(na, nc)
        dma = orbva @ xa @ orboa.T
        dmb = orbvb @ xb @ orbob.T
        dm1 = xp.stack((dma + dma.T, dmb + dmb.T))
        v1 = vresp(dm1)  # G_{\mu\nu}^{\sigma}[Z^T]
        v1a = orbva.T @ v1[0] @ orboa
        v1b = orbvb.T @ v1[1] @ orbob
        return xp.hstack((v1a.ravel(), v1b.ravel()))

    # 1/2 Z
    ucphf = ucphf_module(mf)
    z1a, z1b = ucphf.solve(
        fvind, mo_energy, mo_occ, (wvoa, wvob),
        max_cycle=td.cphf_max_cycle, tol=td.cphf_conv_tol)[0]
    time1 = log.timer("Z-vector using UCPHF solver", *time0)

    z1ao = xp.empty((2, nao, nao))
    z1ao[0] = orbva @ z1a @ orboa.T
    z1ao[1] = orbvb @ z1b @ orbob.T
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)))

    # 5.1 construct complete W matrix
    im0a = xp.zeros((nao, nao))
    im0b = xp.zeros((nao, nao))
    im0a[:ni, :ni] = orboa.T @ (veff0doo[0] + veff[0]) @ orboa
    im0a[:ni, :ni] += xp.einsum("aj,ai->ij", veff0mo[nc:, :ni], v)
    im0b[:nc, :nc] = orbob.T @ (veff0doo[1] + veff[1]) @ orbob
    im0b[nc:, nc:] = xp.einsum("bi,ai->ab", veff0mo[nc:, :ni], v)
    im0b[nc:, :nc] = xp.einsum("ij,aj->ai", veff0mo[:nc, :ni], v) * 2

    # 5.2 construct W matrix, Fock part
    zeta_a = (mo_energy[0][:, None] + mo_energy[0]) * 0.5
    zeta_b = (mo_energy[1][:, None] + mo_energy[1]) * 0.5
    zeta_a[ni:, :ni] = mo_energy[0][:ni]
    zeta_b[nc:, :nc] = mo_energy[1][:nc]
    zeta_a[:ni, ni:] = mo_energy[0][ni:]
    zeta_b[:nc, nc:] = mo_energy[1][nc:]
    dm1a = xp.zeros((nao, nao))
    dm1b = xp.zeros((nao, nao))
    dm1a[:ni, :ni] = dooa
    dm1b[nc:, nc:] = dvvb
    dm1a[ni:, :ni] = z1a * 2
    dm1b[nc:, :nc] = z1b * 2
    dm1a[:ni, :ni] += xp.eye(ni)  # ground state
    dm1b[:nc, :nc] += xp.eye(nc)  # ground state
    im0a = mo_coeff[0] @ (im0a + zeta_a * dm1a) @ mo_coeff[0].T
    im0b = mo_coeff[1] @ (im0b + zeta_b * dm1b) @ mo_coeff[1].T
    im0 = im0a + im0b

    # 6. derivative of coordinate
    dmz1dooa = 4 * z1ao[0] + 2 * dmzooa  # 2(T_{\mu\nu}^{\alpha}+Z_{\mu\nu}^{\alpha})
    dmz1doob = 4 * z1ao[1] + 2 * dmzoob
    oo0a = orboa @ orboa.T
    oo0b = orbob @ orbob.T
    if gpu:
        from gpu4pyscf.grad import tduks as gpu_tduks_grad, uhf as gpu_uhf_grad

        gpu_rhf_grad = gpu_uhf_grad.rhf_grad
        mf_grad = gpu_uhf_grad.Gradients(mf)
        h1 = xp.asarray(mf_grad.get_hcore(mol))
        s1 = xp.asarray(mf_grad.get_ovlp(mol))
        dh_ground = xp.asarray(
            gpu_rhf_grad.contract_h1e_dm(mol, h1, oo0a + oo0b, hermi=1)
        )
        dmz1doo = dmz1dooa + dmz1doob
        dh_td = xp.asarray(
            gpu_rhf_grad.contract_h1e_dm(mol, h1, dmz1doo * .5, hermi=0)
        )
        ds = xp.asarray(gpu_rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0))

        dh1e_ground = xp.asarray(gpu_rhf_grad.int3c2e.get_dh1e(
            mol, oo0a + oo0b
        ))
        dmz1doo_sym = (dmz1doo + dmz1doo.T) * .25
        dh1e_td = xp.asarray(gpu_rhf_grad.int3c2e.get_dh1e(
            mol, dmz1doo_sym
        ))
        if len(mol._ecpbas) > 0:
            dh1e_ground += xp.asarray(
                gpu_rhf_grad.get_dh1e_ecp(mol, oo0a + oo0b)
            )
            dh1e_td += xp.asarray(
                gpu_rhf_grad.get_dh1e_ecp(mol, dmz1doo_sym)
            )
        if mol._pseudo:
            raise NotImplementedError(
                "Pseudopotential gradient not supported for molecular system yet"
            )

        get_veff = gpu_tduks_grad.Gradients.get_veff
        td_density = xp.stack((
            (dmz1dooa + dmz1dooa.T) * .25,
            (dmz1doob + dmz1doob.T) * .25,
        ))
        k_factor = hyb if with_k else 0.0
        dvhf = xp.asarray(get_veff(
            td, mol, td_density + xp.stack((oo0a, oo0b)),
            1.0, k_factor, hermi=1,
        ))
        dvhf -= xp.asarray(get_veff(
            td, mol, td_density, 1.0, k_factor, hermi=1,
        ))
        if with_k:
            dvhf += xp.asarray(get_veff(
                td, mol, xp.stack(((dmt + dmt.T) * .5,
                                   (dmt + dmt.T) * .5)),
                0.0, k_factor, hermi=1,
            ))
            dvhf -= xp.asarray(get_veff(
                td, mol, xp.stack(((dmt - dmt.T) * .5,
                                   (dmt - dmt.T) * .5)),
                0.0, k_factor, hermi=2,
            ))

        gpu_xc_grad = SimpleNamespace(
            mol=mol,
            base=SimpleNamespace(_scf=mf, exclude_nlc=True),
        )
        fxcz1 = gpu_tduks_grad._contract_xc_kernel(
            gpu_xc_grad, mf.xc, 2 * z1ao, None, False, False
        )[0]
        dveff1_0 = xp.asarray(gpu_rhf_grad.contract_h1e_dm(
            mol, vxc1[0, 1:], oo0a + dmz1dooa * .5, hermi=0
        ))
        dveff1_0 += xp.asarray(gpu_rhf_grad.contract_h1e_dm(
            mol, vxc1[1, 1:], oo0b + dmz1doob * .5, hermi=0
        ))
        veff1_1 = (f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:]) * 4
        dveff1_1 = xp.asarray(gpu_rhf_grad.contract_h1e_dm(
            mol, veff1_1[0], oo0a, hermi=1
        )) * .25
        dveff1_1 += xp.asarray(gpu_rhf_grad.contract_h1e_dm(
            mol, veff1_1[1], oo0b, hermi=1
        )) * .25
        dveff1_2 = xp.zeros_like(dvhf)
        if td.base.collinear_samples > 0:
            dveff1_2 = xp.asarray(gpu_rhf_grad.contract_h1e_dm(
                mol, f1vo[1:], dmt, hermi=0
            )) * 2
        de = (
            dh_ground + dh_td - ds + dh1e_ground + dh1e_td + 2 * dvhf
            + dveff1_0 + dveff1_1 + dveff1_2
        )
        if atmlst is not None:
            de = de[xp.asarray(tuple(atmlst), dtype=int)]
    else:
        mf_grad = mf.nuc_grad_method()
        hcore_deriv = mf_grad.hcore_generator(mol)
        s1 = mf_grad.get_ovlp(mol)
        as_dm1 = oo0a + oo0b + (dmz1dooa + dmz1doob) * .5

        dm = xp.stack((oo0a, dmz1dooa + dmz1dooa.T,
                       oo0b, dmz1doob + dmz1doob.T))
        if with_k:
            vj, vk = td.get_jk(mol, dm, hermi=1)
            vj = vj.reshape(2, 2, 3, nao, nao)
            vk = vk.reshape(2, 2, 3, nao, nao) * hyb
            vk1 = -td.get_k(mol, xp.stack((dmt, dmt.T))) * hyb
            veff1 = vj[0] + vj[1] - vk
        else:
            vj = td.get_j(mol, dm, hermi=1).reshape(2, 2, 3, nao, nao)
            veff1 = xp.stack((vj[0] + vj[1], vj[0] + vj[1]))

        fxcz1 = tduks_grad._contract_xc_kernel(
            td, mf.xc, 2 * z1ao, None, False, False, max_memory
        )[0]
        veff1[:, 0] += vxc1[:, 1:]
        veff1[:, 1] += (f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:]) * 4
        veff1a, veff1b = veff1
        time1 = log.timer("2e AO integral derivatives", *time1)

        if atmlst is None:
            atmlst = range(mol.natm)
        offsetdic = mol.offset_nr_by_atom()
        de = xp.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = offsetdic[ia]
            h1ao = hcore_deriv(ia)
            de[k] = xp.einsum("xpq,pq->x", h1ao, as_dm1)
            de[k] += xp.einsum("xpq,pq->x", veff1a[0, :, p0:p1], oo0a[p0:p1]) * 2
            de[k] += xp.einsum("xpq,pq->x", veff1b[0, :, p0:p1], oo0b[p0:p1]) * 2
            de[k] -= xp.einsum("xpq,pq->x", s1[:, p0:p1], im0[p0:p1])
            de[k] -= xp.einsum("xqp,pq->x", s1[:, p0:p1], im0[:, p0:p1])
            de[k] += xp.einsum("xpq,pq->x", veff1a[0, :, p0:p1], dmz1dooa[p0:p1]) * .5
            de[k] += xp.einsum("xpq,pq->x", veff1b[0, :, p0:p1], dmz1doob[p0:p1]) * .5
            de[k] += xp.einsum("xpq,qp->x", veff1a[0, :, p0:p1], dmz1dooa[:, p0:p1]) * .5
            de[k] += xp.einsum("xpq,qp->x", veff1b[0, :, p0:p1], dmz1doob[:, p0:p1]) * .5
            de[k] += xp.einsum("xij,ij->x", veff1a[1, :, p0:p1], oo0a[p0:p1]) * .5
            de[k] += xp.einsum("xij,ij->x", veff1b[1, :, p0:p1], oo0b[p0:p1]) * .5
            if td.base.collinear_samples > 0:
                de[k] += xp.einsum("xpq,pq->x", f1vo[1:, p0:p1], dmt[p0:p1]) * 2
                de[k] += xp.einsum("xpq,pq->x", f1vo[1:, p0:p1], dmt.T[p0:p1]) * 2
            if with_k:
                de[k] += xp.einsum("xpq,pq->x", vk1[0, :, p0:p1], dmt[p0:p1]) * 2
                de[k] += xp.einsum("xpq,pq->x", vk1[1, :, p0:p1], dmt.T[p0:p1]) * 2

    log.timer('SF-down-TDA nuclear gradients', *time0)
    return de


class SFD_gradient(uhf_grad.Gradients):
    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20) + 20
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)

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
