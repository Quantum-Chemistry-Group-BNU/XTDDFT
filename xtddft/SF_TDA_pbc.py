"""Further-refactored spin-flip TDA implementation.

This module keeps the public API of ``SF_TDA_1.py`` but reduces repeated code
by factoring out:

1. MO-space normalization and orbital partitioning
2. Fock/XC preparation helpers
3. Shared Davidson setup and post-processing
4. Shared scalar spin-flip kernel contraction logic

The multicollinear helper routines are preserved so that the explicit matrix
path and the Davidson path stay compatible with the existing implementation.
"""

import functools
from types import SimpleNamespace

import numpy as np
import scipy
from pyscf import dft, lib, scf
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc.dft import numint as pbc_numint
from pyscf.pbc.dft import numint2c as pbc_numint2c
from pyscf.dft import numint, numint2c, xc_deriv
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import _dot_ao_ao_sparse, _scale_ao_sparse, _tau_dot_sparse

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    class _LoggerFallback:
        def info(self, msg, *args):
            logging.getLogger(__name__).info(msg.format(*args) if args else msg)

        def warning(self, msg, *args):
            logging.getLogger(__name__).warning(msg.format(*args) if args else msg)

    logger = _LoggerFallback()

ha2eV = 27.2113834


def _get_gamma_kpt(mf):
    """Return the Gamma k-point and reject general k-point calculations."""
    if hasattr(mf, 'kpts'):
        kpts = np.asarray(mf.kpts).reshape(-1, 3)
        if len(kpts) != 1 or np.linalg.norm(kpts[0]) > 1e-9:
            raise NotImplementedError('SF_TDA_pbc.py currently supports Gamma-point PBC only.')
        return kpts[0]
    kpt = np.asarray(getattr(mf, 'kpt', np.zeros(3))).reshape(3)
    if np.linalg.norm(kpt) > 1e-9:
        raise NotImplementedError('SF_TDA_pbc.py currently supports Gamma-point PBC only.')
    return kpt


def _is_pbc_ks(mf):
    """Whether an SCF object is a PBC Kohn-Sham object."""
    return isinstance(mf, (pbc_dft.KohnShamDFT, dft.KohnShamDFT))


def _get_k_gamma(mf, dm, hermi=0, omega=None):
    """Evaluate Gamma-point PBC exchange for one or more AO density matrices."""
    kwargs = {'hermi': hermi, 'kpt': _get_gamma_kpt(mf)}
    if omega is not None and abs(omega) > 1e-14:
        kwargs['omega'] = omega
    return mf.get_k(mf.cell, dm, **kwargs)


def _needs_ewald_exxdiv(mf, omega=None):
    """Whether explicit MO exchange integrals need the Gamma G=0 correction."""
    _get_gamma_kpt(mf)
    return (
        (omega is None or abs(omega) < 1e-14)
        and getattr(mf, 'exxdiv', None) == 'ewald'
    )


def _ewald_exxdiv_mo_tensor(mf, mo_coeffs, shape):
    """Return the MO-basis G=0 exchange-divergence correction tensor."""
    from pyscf.pbc import tools

    kpt = _get_gamma_kpt(mf)
    overlap = mf.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpt)
    madelung = tools.pbc.madelung(mf.cell, kpt)
    left = mo_coeffs[0].conj().T @ overlap @ mo_coeffs[1]
    right = mo_coeffs[2].conj().T @ overlap @ mo_coeffs[3]
    return madelung * np.einsum('ij,ab->ijab', left, right).reshape(shape)


def SF_TDA(mf, isf=-1, davidson=True, method=0):
    """Return the spin-up or spin-down SF-TDA solver selected by ``isf``."""
    print('method=0 (default) ALDA0, method=1 multicollinear, method=2 collinear')
    if isf == -1:
        return SF_TDA_down(mf, method, davidson)
    if isf == 1:
        return SF_TDA_up(mf, method, davidson)
    raise ValueError(f'Unsupported isf={isf!r}; expected -1 or 1.')


def mf_info(mf):
    """Normalize UKS/ROKS orbital data into a two-spin representation."""
    if np.array(mf.mo_coeff).ndim == 3:
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
    else:
        mo_energy = np.array([mf.mo_energy, mf.mo_energy])
        mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
        mo_occ = np.zeros((2, len(mf.mo_coeff)))
        mo_occ[0][np.where(mf.mo_occ >= 1)[0]] = 1
        mo_occ[1][np.where(mf.mo_occ >= 2)[0]] = 1
    return mo_energy, mo_occ, mo_coeff


def _build_spin_orbital_spaces(mo_coeff, mo_occ):
    """Collect occupied/virtual orbital partitions and commonly used dimensions."""
    occidx_a = np.where(mo_occ[0] == 1)[0]
    viridx_a = np.where(mo_occ[0] == 0)[0]
    occidx_b = np.where(mo_occ[1] == 1)[0]
    viridx_b = np.where(mo_occ[1] == 0)[0]

    orbo_a = mo_coeff[0][:, occidx_a]
    orbv_a = mo_coeff[0][:, viridx_a]
    orbo_b = mo_coeff[1][:, occidx_b]
    orbv_b = mo_coeff[1][:, viridx_b]

    nc = len(occidx_b)
    nv = len(viridx_a)
    no = len(occidx_a) - nc
    return SimpleNamespace(
        occidx_a=occidx_a,
        viridx_a=viridx_a,
        occidx_b=occidx_b,
        viridx_b=viridx_b,
        orbo_a=orbo_a,
        orbv_a=orbv_a,
        orbo_b=orbo_b,
        orbv_b=orbv_b,
        nc=nc,
        nv=nv,
        no=no,
        nocc_a=len(occidx_a),
        nvir_a=len(viridx_a),
        nocc_b=len(occidx_b),
        nvir_b=len(viridx_b),
        nao=mo_coeff[0].shape[0],
    )


def _build_sf_context(mf, mo_energy=None, mo_occ=None, mo_coeff=None):
    """Create a reusable spin-flip context object from an SCF reference."""
    if mo_energy is None or mo_occ is None or mo_coeff is None:
        mo_energy, mo_occ, mo_coeff = mf_info(mf)
    spaces = _build_spin_orbital_spaces(mo_coeff, mo_occ)
    ctx = {
        'mf': mf,
        'cell': mf.cell,
        'mo_energy': mo_energy,
        'mo_occ': mo_occ,
        'mo_coeff': mo_coeff,
    }
    ctx.update(vars(spaces))
    return SimpleNamespace(**ctx)


def _make_reference_dm(mf, mo_occ):
    """Build the reference density matrix used by the scalar XC kernel routines."""
    dm0 = mf.make_rdm1()
    if np.array(mf.mo_coeff).ndim == 2:
        dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
        dm0.mo_occ = mo_occ
    return dm0


def _get_mo_fock(mf, mo_coeff):
    """Return alpha/beta Fock matrices transformed to the MO basis."""
    dm = mf.make_rdm1()
    vhf = np.asarray(mf.get_veff(mf.cell, dm))
    h1e = mf.get_hcore()
    if vhf.ndim == 3:
        focka_ao = h1e + vhf[0]
        fockb_ao = h1e + vhf[1]
    else:
        focka_ao = h1e + vhf
        fockb_ao = h1e + vhf
    focka_mo = mo_coeff[0].conj().T @ focka_ao @ mo_coeff[0]
    fockb_mo = mo_coeff[1].conj().T @ fockb_ao @ mo_coeff[1]
    return focka_mo, fockb_mo


def _prepare_xc_data(mf, mo_occ, max_memory=None):
    """Prepare objects needed by the scalar ALDA0 spin-flip XC kernel."""
    xc_code = getattr(mf, 'xc', None)
    if xc_code is None:
        return None

    ni = mf._numint
    ni.libxc.test_deriv_order(xc_code, 2, raise_error=True)
    if getattr(mf, 'nlc', None) or ni.libxc.is_nlc(xc_code):
        logger.warning('NLC functional found. Its second derivative is not '
                       'available and is not included in the response function.')

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc_code, mf.cell.spin)
    xctype = ni._xc_type(xc_code)
    dm0 = _make_reference_dm(mf, mo_occ)
    make_rho = ni._gen_rho_evaluator(mf.cell, dm0, hermi=0, with_lapl=False)[0]
    mem_now = lib.current_memory()[0]
    if max_memory is None:
        max_memory = max(2000, mf.max_memory * .8 - mem_now)
    return SimpleNamespace(
        ni=ni,
        xctype=xctype,
        omega=omega,
        alpha=alpha,
        hyb=hyb,
        make_rho=make_rho,
        max_memory=max_memory,
    )


def _lda_reduced_density(rho):
    """Keep only the scalar density channel when evaluating a GGA kernel in ALDA0."""
    rho_lda = np.zeros_like(rho)
    rho_lda[0] = rho[0]
    return rho_lda


def _iter_scalar_kernel_blocks(mf, xc_data):
    """Yield AO values and scalarized spin-flip kernel weights on each grid block."""
    if xc_data is None:
        return

    ni = xc_data.ni
    xctype = xc_data.xctype
    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype in ('GGA', 'MGGA'):
        ao_deriv = 1
    else:
        ao_deriv = None
    if ao_deriv is None:
        return

    kpt = _get_gamma_kpt(mf)
    for ao, _, mask, weight, coords in ni.block_loop(
            mf.cell, mf.grids, mf.cell.nao_nr(), ao_deriv,
            kpt, None, xc_data.max_memory):
        rho_a = xc_data.make_rho(0, ao, mask, xctype)
        rho_b = xc_data.make_rho(1, ao, mask, xctype)
        if xctype == 'LDA':
            rho = (rho_a, rho_b)
            ao_values = ao
            denom = rho_a - rho_b
        else:
            rho = (_lda_reduced_density(rho_a), _lda_reduced_density(rho_b))
            ao_values = ao[0]
            denom = rho_a[0] - rho_b[0]
        vxc = ni.eval_xc_eff(
            mf.xc, rho, deriv=1, omega=xc_data.omega, xctype=xctype
        )[1]
        fxc_ab = (vxc[0, 0] - vxc[1, 0]) * weight
        fxc_ab /= denom + 1e-9
        yield ao_values, fxc_ab


def _contract_scalar_kernel(amat, ao_values, fxc_ab, orbo, orbv):
    """Accumulate one scalar XC-kernel block into a spin-flip interaction tensor."""
    rho_occ = lib.einsum('rp,pi->ri', ao_values, orbo)
    rho_vir = lib.einsum('rp,pa->ra', ao_values, orbv)
    rho_ov = np.einsum('ri,ra->ria', rho_occ, rho_vir, optimize=True)
    amat += np.einsum('ria,r,rjb->iajb', rho_ov, fxc_ab, rho_ov, optimize=True)
    return amat


def _add_hf_exchange_tensor(amat, mf, orbo, orbv, nocc, nvir, hyb=1.0, omega=None):
    """Add short-range or range-separated HF exchange to an interaction tensor."""
    if abs(hyb) < 1e-14:
        return amat
    if nocc == 0 or nvir == 0:
        return amat

    kpt = _get_gamma_kpt(mf)

    def build_eri(with_df):
        return with_df.ao2mo([orbo, orbo, orbv, orbv], kpt, compact=False)

    if omega is None or abs(omega) < 1e-14:
        eri_mo = build_eri(mf.with_df)
    else:
        with mf.with_df.range_coulomb(omega) as rsh_df:
            eri_mo = build_eri(rsh_df)
    eri_mo = eri_mo.reshape(nocc, nocc, nvir, nvir)
    if _needs_ewald_exxdiv(mf, omega):
        eri_mo += _ewald_exxdiv_mo_tensor(
            mf, [orbo, orbo, orbv, orbv], eri_mo.shape
        )
    amat -= np.einsum('ijba->iajb', eri_mo, optimize=True) * hyb
    return amat


def _pair_sector(occ_slice, vir_slice, nocc, nvir):
    """Describe one occupied-virtual sector inside a spin-flip tensor."""
    return SimpleNamespace(occ=occ_slice, vir=vir_slice, nocc=nocc, nvir=nvir)


def _reshape_sector_tensor(tensor, left_sector, right_sector=None):
    """Extract and reshape one tensor block between two occupied-virtual sectors."""
    if right_sector is None:
        right_sector = left_sector
    block = tensor[left_sector.occ, left_sector.vir, right_sector.occ, right_sector.vir]
    return block.reshape(
        left_sector.nocc * left_sector.nvir,
        right_sector.nocc * right_sector.nvir,
    )


def _pair_hessian_block(occ_fock, vir_fock, tensor_block):
    """Assemble one diagonal occupied-virtual block of the explicit A matrix."""
    nocc = occ_fock.shape[0]
    nvir = vir_fock.shape[0]
    return (
        np.einsum('ij,ab->iajb', np.eye(nocc), vir_fock, optimize=True)
        - np.einsum('ji,ab->iajb', occ_fock, np.eye(nvir), optimize=True)
        + tensor_block.reshape(nocc, nvir, nocc, nvir)
    ).reshape(nocc * nvir, nocc * nvir)


def _vir_coupling_block(shared_occ, vir_coupling, tensor_block):
    """Assemble an off-diagonal block coupled through the virtual-space Fock term."""
    return (
        np.einsum('ij,ay->iajy', np.eye(shared_occ), vir_coupling, optimize=True)
        + tensor_block.reshape(shared_occ, vir_coupling.shape[0], shared_occ, vir_coupling.shape[1])
    ).reshape(shared_occ * vir_coupling.shape[0], shared_occ * vir_coupling.shape[1])


def _occ_coupling_block(occ_coupling, shared_vir, tensor_block, sign=-1.0):
    """Assemble an off-diagonal block coupled through the occupied-space Fock term."""
    return (
        sign * np.einsum('yi,ab->iayb', occ_coupling, np.eye(shared_vir), optimize=True)
        + tensor_block.reshape(occ_coupling.shape[1], shared_vir, occ_coupling.shape[0], shared_vir)
    ).reshape(occ_coupling.shape[1] * shared_vir, occ_coupling.shape[0] * shared_vir)


def _set_symmetric_block(matrix, left_slice, right_slice, block):
    """Write one block and its transpose into a symmetric dense matrix."""
    matrix[left_slice, right_slice] = block
    if left_slice != right_slice:
        matrix[right_slice, left_slice] = block.T


def _spinflip_gaps(ctx, isf):
    """Return flattened Koopmans-like spin-flip energy gaps for the target channel."""
    if isf == 1:
        return (ctx.mo_energy[0][ctx.viridx_a, None]
                - ctx.mo_energy[1][ctx.occidx_b]).T.ravel()
    if isf == -1:
        return (ctx.mo_energy[1][ctx.viridx_b, None]
                - ctx.mo_energy[0][ctx.occidx_a]).T.ravel()
    raise ValueError(f'Unsupported isf={isf!r}; expected -1 or 1.')


def _build_initial_guess_from_gaps(gaps, nstates):
    """Build Davidson guesses from the lowest diagonal energy gaps."""
    nov = gaps.size
    nroots = min(nstates, nov)
    e_threshold = np.sort(gaps)[nroots - 1] + 1e-5
    idx = np.where(gaps <= e_threshold)[0]
    x0 = np.zeros((idx.size, nov))
    for row, col in enumerate(idx):
        x0[row, col] = 1.0
    return x0


def _reorder_down_vectors(ctx, v, remove=False):
    """Map Davidson vectors from (occ_a, vir_b) order to CV|CO|OV|OO order."""
    nstates = v.shape[1]
    nc = ctx.nc
    nv = ctx.nv
    no = ctx.no
    nvir = no + nv
    passed = nc * nvir

    cv = np.zeros((nstates, nc, nv))
    co = np.zeros((nstates, nc, no))
    ov = np.zeros((nstates, no, nv))
    if remove:
        oo = np.zeros((nstates, no * no - 1))
    else:
        oo = np.zeros((nstates, no, no))

    for state in range(nstates):
        tmp_data = v[:, state]
        for i in range(nc):
            co[state, i, :] = tmp_data[i*nvir:i*nvir + no]
            cv[state, i, :] = tmp_data[i*nvir + no:i*nvir + no + nv]
        if remove:
            for i in range(no - 1):
                oo[state, i*no:(i + 1)*no] = tmp_data[passed + i*nvir:passed + i*nvir + no]
                ov[state, i, :] = tmp_data[passed + i*nvir + no:passed + i*nvir + no + nv]
            oo[state, (no - 1)*no:] = tmp_data[passed + (no - 1)*nvir:passed + (no - 1)*nvir + no - 1]
            ov[state, no - 1, :] = tmp_data[passed + (no - 1)*nvir + no - 1:]
        else:
            for i in range(no):
                oo[state, i, :] = tmp_data[passed + i*nvir:passed + i*nvir + no]
                ov[state, i, :] = tmp_data[passed + i*nvir + no:passed + i*nvir + no + nv]

    return np.hstack([
        cv.reshape(nstates, -1),
        co.reshape(nstates, -1),
        ov.reshape(nstates, -1),
        oo.reshape(nstates, -1),
    ]).T


def cache_xc_kernel_sf(mf, mo_coeff, mo_occ, spin=1, max_memory=2000, isf=-1):
    """Cache the scalar spin-flip XC kernel used by ALDA0 response functions."""
    del mo_coeff, isf
    xc_data = _prepare_xc_data(mf, mo_occ, max_memory=max_memory)
    if xc_data is None:
        return np.asarray([])
    if xc_data.xctype not in ('LDA', 'GGA', 'MGGA'):
        return np.asarray([])
    assert spin == 1
    fxc_abs = []
    for ao_values, fxc_ab in _iter_scalar_kernel_blocks(mf, xc_data):
        del ao_values
        fxc_abs.extend(fxc_ab.tolist())
    return np.asarray(fxc_abs)


def nr_uks_fxc_sf_tda(ni, cell, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                      vxc=None, extype=0, kpt=None, max_memory=2000, verbose=None):
    del dm0, relativity, extype, verbose

    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype in ('GGA', 'MGGA'):
        ao_deriv = 1
    else:
        raise NotImplementedError

    if kpt is None:
        kpt = np.zeros(3)
    else:
        kpt = np.asarray(kpt).reshape(3)

    dtype = dms.dtype if isinstance(dms, np.ndarray) else np.result_type(*dms)
    nao = dms.shape[-1]
    make_rhosf, nset = ni._gen_rho_evaluator(cell, dms, hermi, False)[:2]

    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc_nr()

    v_hermi = 0
    if pbc_numint.is_zero(kpt) and dtype == np.double:
        v_hermi = 1

    vmat = [0] * nset
    p1 = 0
    for ao_k1, ao_k2, mask, weight, coords in ni.block_loop(
            cell, grids, nao, ao_deriv, kpt, None, max_memory):
        p0, p1 = p1, p1 + weight.size
        _vxc = vxc[p0:p1]

        for i in range(nset):
            rho1sf = make_rhosf(i, ao_k1, mask, xctype)
            if xctype == 'LDA':
                wv = (rho1sf * _vxc)[np.newaxis]
            else:
                rhosf = np.zeros_like(rho1sf)
                rhosf[0] = rho1sf[0]
                wv = rhosf * _vxc
            vmat[i] += ni._vxc_mat(cell, ao_k1, wv, mask, xctype,
                                   shls_slice, ao_loc, v_hermi)

    vmat = np.stack(vmat)
    if v_hermi == 1:
        vmat = vmat + vmat.conj().swapaxes(-2, -1)
    if nset == 1:
        vmat = vmat.reshape(dms.shape)
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    return vmat

def gen_response_sf(mf, hermi=0, max_memory=None, method=0):
    mo_energy, mo_occ, mo_coeff = mf_info(mf)
    del mo_energy
    cell = mf.cell
    kpt = _get_gamma_kpt(mf)

    if _is_pbc_ks(mf):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, cell.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory * .8 - mem_now)

        vxc = cache_xc_kernel_sf(
            mf, mo_coeff, mo_occ, 1, max_memory, isf=-1
        ) if method == 0 else None

        def vind(dm1):
            if method == 0:
                v1 = nr_uks_fxc_sf_tda(
                    ni, cell, mf.grids, mf.xc, None, dm1, 0, hermi,
                    vxc=vxc, kpt=kpt, max_memory=max_memory
                )
            else:
                v1 = np.zeros_like(dm1)

            if hybrid:
                vk = _get_k_gamma(mf, dm1, hermi=hermi) * hyb
                if abs(omega) > 1e-10:
                    vk += _get_k_gamma(mf, dm1, hermi=hermi, omega=omega) * (alpha - hyb)
                v1 -= vk
            return v1
    else:
        def vind(dm1):
            return -_get_k_gamma(mf, dm1, hermi=hermi)

    return vind


def gen_tda_operation_sf(mf, isf, method):
    """Build the matrix-vector product and diagonal preconditioner for SF-TDA."""
    ctx = _build_sf_context(mf)
    assert ctx.mo_coeff[0].dtype == np.double
    focka_mo, fockb_mo = _get_mo_fock(mf, ctx.mo_coeff)

    if isf == -1:
        ndim = (ctx.nocc_a, ctx.nvir_b)
        orbo, orbv = ctx.orbo_a, ctx.orbv_b
        occ_block = focka_mo[:ctx.nocc_a, :ctx.nocc_a]
        vir_block = fockb_mo[ctx.nc:, ctx.nc:]
    elif isf == 1:
        ndim = (ctx.nocc_b, ctx.nvir_a)
        orbo, orbv = ctx.orbo_b, ctx.orbv_a
        occ_block = fockb_mo[:ctx.nocc_b, :ctx.nocc_b]
        vir_block = focka_mo[ctx.nocc_a:, ctx.nocc_a:]
    else:
        raise ValueError(f'Unsupported isf={isf!r}; expected -1 or 1.')

    hdiag = _spinflip_gaps(ctx, isf)
    if method == 1:
        vresp = _gen_uhf_tda_response_sf(mf, hermi=0, collinear_samples=50)
    else:
        vresp = gen_response_sf(mf, hermi=0, method=method)

    def vind(zs0):
        zs = np.asarray(zs0).reshape(-1, *ndim)
        dmov = lib.einsum('xov,qv,po->xpq', zs, orbv.conj(), orbo)
        v1ao = vresp(np.asarray(dmov))
        vs = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        vs += np.einsum('ab,xib->xia', vir_block, zs, optimize=True)
        vs -= np.einsum('ij,xja->xia', occ_block, zs, optimize=True)
        return vs.reshape(zs.shape[0], -1)

    return vind, hdiag

def deal_v_davidson2(mf, nstates, v):
    """Reshape spin-up Davidson vectors into the compact CV block layout."""
    ctx = _build_sf_context(mf)
    return np.asarray(v).T.reshape(nstates, ctx.nc, ctx.nv)


def deal_v_davidson(mf, nstates, v, remove=False):
    """Reorder spin-down Davidson vectors into the CV|CO|OV|OO layout."""
    del nstates
    return _reorder_down_vectors(_build_sf_context(mf), np.asarray(v), remove=remove)


def init_guess(mf, nstates, isf=-1):
    """Construct Davidson initial guesses from the lowest spin-flip gaps."""
    return _build_initial_guess_from_gaps(_spinflip_gaps(_build_sf_context(mf), isf), nstates)


def davidson_process(mf, nstates, method, isf=-1):
    """Solve the SF-TDA eigenproblem with Davidson and return energies/vectors."""
    vind, hdiag = gen_tda_operation_sf(mf, isf, method)

    ndim = hdiag.size
    nroots = min(nstates, ndim)

    x0 = init_guess(mf, nroots, isf)
    converged, e, x1 = lib.davidson1(
        vind, x0, hdiag, tol=1e-7, lindep=1e-14,
        nroots=nroots, max_cycle=3000
    )

    v = np.asarray(x1).T
    if isf == -1:
        v = _reorder_down_vectors(_build_sf_context(mf), v)

    logger.info('SF-TDA converged: {}', converged)
    return e, v



class _SpinFlipTDABase:
    """Shared implementation for the spin-up and spin-down SF-TDA solvers."""

    isf = None
    xc_warning_label = 'SF_TDA'

    def __init__(self, mf, method, davidson=True):
        self.mf = mf
        self.cell = mf.cell
        self.method = method
        self.davidson = davidson
        self.ctx = _build_sf_context(mf)
        self.mo_energy = self.ctx.mo_energy
        self.mo_occ = self.ctx.mo_occ
        self.mo_coeff = self.ctx.mo_coeff
        self.nao = self.ctx.nao
        self.occidx_a = self.ctx.occidx_a
        self.viridx_a = self.ctx.viridx_a
        self.occidx_b = self.ctx.occidx_b
        self.viridx_b = self.ctx.viridx_b
        self.orbo_a = self.ctx.orbo_a
        self.orbv_a = self.ctx.orbv_a
        self.orbo_b = self.ctx.orbo_b
        self.orbv_b = self.ctx.orbv_b
        self.nc = self.ctx.nc
        self.nv = self.ctx.nv
        self.no = self.ctx.no
        self.nocc_a = self.ctx.nocc_a
        self.nvir_a = self.ctx.nvir_a
        self.nocc_b = self.ctx.nocc_b
        self.nvir_b = self.ctx.nvir_b
        self._fock_mo = None
        self.A = None
        self.e = None
        self.v = None
        self.nstates = None
        self.converged = None

    def _get_fock_mo(self):
        """Return and cache alpha/beta Fock matrices in the MO basis."""
        if self._fock_mo is None:
            self._fock_mo = _get_mo_fock(self.mf, self.mo_coeff)
        return self._fock_mo

    def _prepare_xc(self):
        """Build the scalar XC-kernel helper bundle for the current SCF reference."""
        return _prepare_xc_data(self.mf, self.mo_occ)

    def _exchange_orbitals(self):
        """Return occupied/virtual orbital blocks used in the HF exchange tensor."""
        raise NotImplementedError

    def _interaction_shape(self):
        """Return the tensor shape of the explicit interaction block."""
        raise NotImplementedError

    def _multicollinear_matrix(self):
        """Return the method=1 dense matrix in the solver's analysis layout."""
        raise NotImplementedError

    def _finalize_explicit_matrix(self, interaction_tensor):
        """Convert the interaction tensor into the dense matrix used by eigensolvers."""
        raise NotImplementedError

    def _postprocess_davidson_vectors(self, v):
        """Map Davidson vectors into the same layout used by ``analyse``."""
        return v

    def _build_interaction_tensor(self):
        """Assemble the common XC/HF interaction tensor shared by both channels."""
        amat = np.zeros(self._interaction_shape())
        orbo, orbv, nocc, nvir = self._exchange_orbitals()
        xc_data = self._prepare_xc()

        if xc_data is None:
            _add_hf_exchange_tensor(amat, self.mf, orbo, orbv, nocc, nvir, hyb=1.0)
            return amat

        _add_hf_exchange_tensor(amat, self.mf, orbo, orbv, nocc, nvir, hyb=xc_data.hyb)
        if abs(xc_data.omega) > 1e-10:
            _add_hf_exchange_tensor(
                amat, self.mf, orbo, orbv, nocc, nvir,
                hyb=xc_data.alpha - xc_data.hyb, omega=xc_data.omega
            )

        if xc_data.xctype in ('LDA', 'GGA') and self.method != 2:
            for ao_values, fxc_ab in _iter_scalar_kernel_blocks(self.mf, xc_data):
                _contract_scalar_kernel(amat, ao_values, fxc_ab, orbo, orbv)
        elif xc_data.xctype not in ('LDA', 'GGA', 'HF', 'NLC'):
            logger.warning('XC type {} is not implemented in {}; '
                           'only Fock and HF-exchange terms were included.',
                           xc_data.xctype, self.xc_warning_label)
        return amat

    def get_Amat(self):
        """Build and store the explicit SF-TDA A matrix."""
        self.A = self._finalize_explicit_matrix(self._build_interaction_tensor())
        return self.A

    def _init_guess(self, nstates):
        """Construct Davidson guesses from the lowest orbital energy gaps."""
        return _build_initial_guess_from_gaps(_spinflip_gaps(self.ctx, self.isf), nstates)

    def davidson_process(self, nstates):
        """Solve the current SF-TDA eigenproblem with Davidson iteration."""
        vind, hdiag = gen_tda_operation_sf(self.mf, isf=self.isf, method=self.method)
        x0 = self._init_guess(nstates)
        converged, e, x1 = lib.davidson1(
            vind, x0, hdiag, tol=1e-7, lindep=1e-14,
            nroots=nstates, max_cycle=3000
        )
        self.converged = converged
        self.e = e
        self.v = self._postprocess_davidson_vectors(np.array(x1).T)
        print('Converged ', converged)
        return self.e, self.v

    def kernel(self, nstates=1):
        """Run the selected diagonalization path and return excitation energies/vectors."""
        self.nstates = nstates
        if self.davidson:
            self.davidson_process(nstates)
        else:
            if self.method == 1:
                self.A = self._multicollinear_matrix()
            else:
                self.get_Amat()
            self.e, self.v = scipy.linalg.eigh(self.A)
        return self.e[:nstates] * ha2eV, self.v[:, :nstates]


class SF_TDA_up(_SpinFlipTDABase):
    """Spin-up SF-TDA solver with shared helper-based assembly."""

    isf = 1
    xc_warning_label = 'SF_TDA_up'

    def _exchange_orbitals(self):
        """Return the beta-occupied / alpha-virtual orbital blocks."""
        return self.orbo_b, self.orbv_a, self.nc, self.nv

    def _interaction_shape(self):
        """Return the spin-up interaction tensor shape."""
        return self.nc, self.nv, self.nc, self.nv

    def _finalize_explicit_matrix(self, interaction_tensor):
        """Add one-particle Fock terms and reshape the spin-up A matrix."""
        focka_mo, fockb_mo = self._get_fock_mo()
        return _pair_hessian_block(
            fockb_mo[:self.nc, :self.nc],
            focka_mo[self.nocc_a:, self.nocc_a:],
            interaction_tensor,
        )

    def _multicollinear_matrix(self):
        """Build the multicollinear dense A matrix for the spin-up channel."""
        amat = get_ab_sf(self.mf, return_both=True)[0]
        return amat.reshape(self.nc * self.nv, self.nc * self.nv)

    def analyse(self):
        """Print the dominant CV(beta->alpha) components of each excited state."""
        for nstate in range(self.nstates):
            value = self.v[:, nstate]
            x_cv = value[:self.nc * self.nv].reshape(self.nc, self.nv)
            print(f'Excited state {nstate+1} {self.e[nstate]*ha2eV:10.5f} eV')
            for occ, vir in zip(*np.where(abs(x_cv) > 0.1)):
                vir_label = vir + 1 + self.nc + self.no
                print(f'{100*x_cv[occ, vir]**2:3.0f}% CV(ab) '
                      f'{occ+1}a -> {vir_label}b {x_cv[occ, vir]:10.5f}')
            print(' ')


class SF_TDA_down(_SpinFlipTDABase):
    """Spin-down SF-TDA solver with shared helper-based assembly."""

    isf = -1
    xc_warning_label = 'SF_TDA_down'

    def _exchange_orbitals(self):
        """Return the alpha-occupied / beta-virtual orbital blocks."""
        return self.orbo_a, self.orbv_b, self.nocc_a, self.nvir_b

    def _interaction_shape(self):
        """Return the spin-down interaction tensor shape."""
        return self.nocc_a, self.nvir_b, self.nocc_a, self.nvir_b

    def _assemble_full_amat(self, a_a2b):
        """Assemble the full CV|CO|OV|OO spin-down A matrix from tensor blocks."""
        nc = self.nc
        nv = self.nv
        no = self.no
        focka_mo, fockb_mo = self._get_fock_mo()

        iden_C = np.eye(nc)
        iden_O = np.eye(no)
        iden_V = np.eye(nv)
        fockA_C = focka_mo[:nc, :nc]
        fockA_O = focka_mo[nc:nc+no, nc:nc+no]
        fockB_O = fockb_mo[nc:nc+no, nc:nc+no]
        fockB_V = fockb_mo[nc+no:, nc+no:]

        cv = _pair_sector(slice(0, nc), slice(no, no + nv), nc, nv)
        co = _pair_sector(slice(0, nc), slice(0, no), nc, no)
        ov = _pair_sector(slice(nc, nc + no), slice(no, no + nv), no, nv)
        oo = _pair_sector(slice(nc, nc + no), slice(0, no), no, no)

        dim = (nc + no) * (nv + no)
        amat = np.zeros((dim, dim))
        cv_slice = slice(0, cv.nocc * cv.nvir)
        co_slice = slice(cv_slice.stop, cv_slice.stop + co.nocc * co.nvir)
        ov_slice = slice(co_slice.stop, co_slice.stop + ov.nocc * ov.nvir)
        oo_slice = slice(ov_slice.stop, ov_slice.stop + oo.nocc * oo.nvir)

        _set_symmetric_block(
            amat,
            cv_slice,
            cv_slice,
            _pair_hessian_block(fockA_C, fockB_V, _reshape_sector_tensor(a_a2b, cv).reshape(nc, nv, nc, nv)),
        )
        _set_symmetric_block(
            amat,
            co_slice,
            co_slice,
            _pair_hessian_block(fockA_C, fockB_O, _reshape_sector_tensor(a_a2b, co).reshape(nc, no, nc, no)),
        )
        _set_symmetric_block(
            amat,
            ov_slice,
            ov_slice,
            _pair_hessian_block(fockA_O, fockB_V, _reshape_sector_tensor(a_a2b, ov).reshape(no, nv, no, nv)),
        )
        _set_symmetric_block(
            amat,
            oo_slice,
            oo_slice,
            _pair_hessian_block(fockA_O, fockB_O, _reshape_sector_tensor(a_a2b, oo).reshape(no, no, no, no)),
        )

        _set_symmetric_block(
            amat,
            cv_slice,
            co_slice,
            _vir_coupling_block(
                nc,
                fockb_mo[nc + no:, nc:nc + no],
                _reshape_sector_tensor(a_a2b, cv, co),
            ),
        )
        _set_symmetric_block(
            amat,
            cv_slice,
            ov_slice,
            _occ_coupling_block(
                focka_mo[nc:nc + no, :nc],
                nv,
                _reshape_sector_tensor(a_a2b, cv, ov),
            ),
        )
        _set_symmetric_block(
            amat,
            co_slice,
            ov_slice,
            _reshape_sector_tensor(a_a2b, co, ov),
        )
        _set_symmetric_block(
            amat,
            cv_slice,
            oo_slice,
            _reshape_sector_tensor(a_a2b, cv, oo),
        )
        _set_symmetric_block(
            amat,
            co_slice,
            oo_slice,
            _occ_coupling_block(
                focka_mo[nc:nc + no, :nc],
                no,
                _reshape_sector_tensor(a_a2b, co, oo),
            ),
        )
        _set_symmetric_block(
            amat,
            ov_slice,
            oo_slice,
            _vir_coupling_block(
                no,
                fockb_mo[nc + no:, nc:nc + no],
                _reshape_sector_tensor(a_a2b, ov, oo),
            ),
        )
        return amat

    def _finalize_explicit_matrix(self, interaction_tensor):
        """Assemble and return the dense spin-down A matrix."""
        return self._assemble_full_amat(interaction_tensor)

    def _multicollinear_matrix(self):
        """Build the multicollinear dense A matrix for the spin-down channel."""
        amat = get_ab_sf(self.mf)
        dim = (self.nc + self.no) * (self.nv + self.no)
        return amat.reshape(dim, dim)

    def _postprocess_davidson_vectors(self, v):
        """Reorder Davidson vectors into the CV|CO|OV|OO analysis layout."""
        return _reorder_down_vectors(self.ctx, v)

    def analyse(self):
        """Print the dominant spin-down amplitudes and D<S^2> estimate."""
        nc = self.nc
        nv = self.nv
        no = self.no
        for nstate in range(self.nstates):
            value = self.v[:, nstate]
            x_cv_ab = value[:nc*nv].reshape(nc, nv)
            x_co_ab = value[nc*nv:nc*nv+nc*no].reshape(nc, no)
            x_ov_ab = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no, nv)
            x_oo_ab = value[nc*nv+nc*no+no*nv:].reshape(no, no)
            dp_ab = np.sum(x_cv_ab * x_cv_ab) - np.sum(x_oo_ab * x_oo_ab)
            dp_ab += np.sum(np.diag(x_oo_ab)) ** 2

            print(f'Excited state {nstate+1} {self.e[nstate]*ha2eV:10.5f} eV D<S^2>={-no+1+dp_ab:5.2f}')
            for o, v in zip(*np.where(abs(x_cv_ab) > 0.1)):
                print(f'{100*x_cv_ab[o, v]**2:3.0f}% CV(ab) {o+1}a -> {v+1+self.nc+self.no}b {x_cv_ab[o, v]:10.5f}')
            for o, v in zip(*np.where(abs(x_co_ab) > 0.1)):
                print(f'{100*x_co_ab[o, v]**2:3.0f}% CO(ab) {o+1}a -> {v+1+self.nc}b {x_co_ab[o, v]:10.5f}')
            for o, v in zip(*np.where(abs(x_ov_ab) > 0.1)):
                print(f'{100*x_ov_ab[o, v]**2:3.0f}% OV(ab) {o+nc+1}a -> {v+1+self.nc+self.no}b {x_ov_ab[o, v]:10.5f}')
            for o, v in zip(*np.where(abs(x_oo_ab) > 0.1)):
                print(f'{100*x_oo_ab[o, v]**2:3.0f}% OO(ab) {o+nc+1}a -> {v+1+self.nc}b {x_oo_ab[o, v]:10.5f}')
            print(' ')


def _gen_uhf_tda_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0,
                             collinear_samples=60, max_memory=None, method=1):
    """Generate the multicollinear spin-flip response function."""
    del method
    mo_energy, mo_occ, mo_coeff = mf_info(mf)
    del mo_energy
    cell = mf.cell
    kpt = _get_gamma_kpt(mf)

    ni = pbc_numint2c.NumInt2C()
    ni.collinear = 'mcol'
    ni.collinear_samples = collinear_samples
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    if mf.nlc or ni.libxc.is_nlc(mf.xc):
        logger.warning('NLC functional found in DFT object. Its second '
                       'derivative is not available. Its contribution is '
                       'not included in the response function.')
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, cell.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    fxc = cache_xc_kernel_sf_mc(
        ni, cell, mf.grids, mf.xc, mo_coeff, mo_occ, deriv=2, spin=1,
        kpt=kpt)[2]
    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * .8 - mem_now)

    def vind(dm1):
        in2 = pbc_numint.NumInt()
        v1 = nr_uks_fxc_sf_tda_mc(
            in2, cell, mf.grids, mf.xc, None, dm1, 0, hermi,
            None, None, fxc, kpt=kpt, max_memory=max_memory
        )
        if hybrid:
            vk = _get_k_gamma(mf, dm1, hermi=hermi) * hyb
            if omega > 1e-10:
                vk += _get_k_gamma(mf, dm1, hermi=hermi, omega=omega) * (alpha - hyb)
            v1 -= vk
        return v1

    return vind


def __mcfun_fn_eval_xc(ni, xc_code, xctype, rho, deriv):
    """Evaluate XC derivatives and convert them to tensor-spin form."""
    evfk = ni.eval_xc_eff(xc_code, rho, deriv=deriv, xctype=xctype)
    for order in range(1, deriv + 1):
        if evfk[order] is not None:
            evfk[order] = xc_deriv.ud2ts(evfk[order])
    return evfk


def mcfun_eval_xc_adapter_sf(ni, xc_code):
    """Wrap ``mcfun.eval_xc_eff_sf`` in a PySCF-compatible callback."""
    try:
        import mcfun
    except ImportError as exc:
        raise ImportError(
            'This feature requires mcfun library.\n'
            'Try install mcfun with `pip install mcfun`'
        ) from exc

    xctype = ni._xc_type(xc_code)
    fn_eval_xc = functools.partial(__mcfun_fn_eval_xc, ni, xc_code, xctype)
    nproc = lib.num_threads()

    def eval_xc_eff(xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None):
        del xc_code, omega, xctype, verbose
        return mcfun.eval_xc_eff_sf(
            fn_eval_xc, rho, deriv,
            collinear_samples=ni.collinear_samples, workers=nproc
        )

    return eval_xc_eff


def cache_xc_kernel_sf_mc(self, cell, grids, xc_code, mo_coeff, mo_occ,
                          deriv=2, spin=1, kpt=None, max_memory=2000):
    """Compute the multicollinear spin-flip XC kernel."""
    del deriv
    MGGA_DENSITY_LAPL = False
    xctype = self._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
    else:
        ao_deriv = 0
    with_lapl = MGGA_DENSITY_LAPL

    assert mo_coeff[0].ndim == 2
    assert spin == 1

    nao = mo_coeff[0].shape[0]
    rhoa = []
    rhob = []
    if kpt is None:
        kpt = np.zeros(3)
    ni = pbc_numint.NumInt()
    for ao, _, mask, weight, coords in self.block_loop(
            cell, grids, nao, ao_deriv, kpt, None, max_memory):
        del weight, coords
        rhoa.append(ni.eval_rho2(cell, ao, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl))
        rhob.append(ni.eval_rho2(cell, ao, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl))
    rho_ab = np.asarray((np.hstack(rhoa), np.hstack(rhob)))
    rho_tmz = np.zeros_like(rho_ab) + 1e-11
    rho_tmz[0] += rho_ab[0] + rho_ab[1]
    rho_tmz[1] += rho_ab[0] - rho_ab[1]
    eval_xc = mcfun_eval_xc_adapter_sf(self, xc_code)
    return eval_xc(xc_code, rho_tmz, deriv=2, xctype=xctype)


def nr_uks_fxc_sf_tda_mc(ni, cell, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                         rho0=None, vxc=None, fxc=None, extype=0,
                         kpt=None, max_memory=2000, verbose=None):
    """Apply the multicollinear spin-flip XC kernel to AO density perturbations."""
    del dm0, relativity, rho0, vxc, extype, verbose
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if hermi != 1 and dtype != np.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)
    nao = dms.shape[-1]
    if kpt is None:
        kpt = np.zeros(3)
    make_rhosf, nset = ni._gen_rho_evaluator(cell, dms, hermi, False)[:2]

    def block_loop(ao_deriv):
        p1 = 0
        for ao, _, mask, weight, coords in ni.block_loop(
                cell, grids, nao, ao_deriv, kpt, None, max_memory):
            del coords
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[..., p0:p1]
            for i in range(nset):
                rho1sf = make_rhosf(i, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = rho1sf * _fxc[0, 0] * 2.0 * weight
                else:
                    wv = lib.einsum('bg,abg->ag', rho1sf, _fxc * 2.0) * weight
                yield i, ao, mask, wv

    ao_loc = cell.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    pair_mask = cell.get_overlap_cond() < -np.log(ni.cutoff)
    vmat = np.zeros((nset, nao, nao))
    aow = None

    if xctype == 'LDA':
        for i, ao, mask, wv in block_loop(0):
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc, hermi, vmat[i])
    elif xctype == 'GGA':
        for i, ao, mask, wv in block_loop(1):
            wv[0] *= .5
            aow = _scale_ao_sparse(ao, wv, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])
        vmat = lib.hermi_sum(vmat.reshape(-1, nao, nao), axes=(0, 2, 1)).reshape(nset, nao, nao)
    elif xctype == 'MGGA':
        ao_deriv = 1
        v1 = np.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5
            wv[4] *= .5
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])
        vmat = lib.hermi_sum(vmat.reshape(-1, nao, nao), axes=(0, 2, 1)).reshape(nset, nao, nao)
        vmat += v1

    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[:, 0]
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    return vmat


def _initialize_mc_diagonal_blocks(mf, ctx, mo_energy):
    """Build the diagonal part of the multicollinear spin-flip A tensors."""
    if np.allclose(mf.mo_coeff[0], mf.mo_coeff[1]):
        fock_ao_a, fock_ao_b = mf.get_fock()
        fock_oo_a = ctx.orbo_a.T @ fock_ao_a @ ctx.orbo_a
        fock_vv_a = ctx.orbv_a.T @ fock_ao_a @ ctx.orbv_a
        fock_oo_b = ctx.orbo_b.T @ fock_ao_b @ ctx.orbo_b
        fock_vv_b = ctx.orbv_b.T @ fock_ao_b @ ctx.orbv_b

        a_b2a = np.zeros((ctx.nocc_b, ctx.nvir_a, ctx.nocc_b, ctx.nvir_a))
        a_a2b = np.zeros((ctx.nocc_a, ctx.nvir_b, ctx.nocc_a, ctx.nvir_b))
        a_b2a += np.einsum('ik,ab->iakb', np.eye(ctx.nocc_b), fock_vv_a)
        a_b2a -= np.einsum('ac,ik->iakc', np.eye(ctx.nvir_a), fock_oo_b.T)
        a_a2b += np.einsum('ik,ab->iakb', np.eye(ctx.nocc_a), fock_vv_b)
        a_a2b -= np.einsum('ac,ik->iakc', np.eye(ctx.nvir_b), fock_oo_a.T)
    else:
        e_ia_b2a = (mo_energy[0][ctx.viridx_a, None] - mo_energy[1][ctx.occidx_b]).T
        e_ia_a2b = (mo_energy[1][ctx.viridx_b, None] - mo_energy[0][ctx.occidx_a]).T
        a_b2a = np.diag(e_ia_b2a.ravel()).reshape(ctx.nocc_b, ctx.nvir_a, ctx.nocc_b, ctx.nvir_a)
        a_a2b = np.diag(e_ia_a2b.ravel()).reshape(ctx.nocc_a, ctx.nvir_b, ctx.nocc_a, ctx.nvir_b)
    return a_b2a, a_a2b


def _add_mc_hf_terms(mf, ctx, a, b, hyb=1.0, omega=None):
    """Add Gamma-point PBC HF exchange contributions to multicollinear A/B tensors."""
    if abs(hyb) < 1e-14:
        return

    kpt = _get_gamma_kpt(mf)

    def build_eri_block(with_df, coeffs, shape):
        if any(coeff.shape[1] == 0 for coeff in coeffs):
            return np.zeros(shape, dtype=np.result_type(*coeffs))
        eri = with_df.ao2mo(coeffs, kpt, compact=False)
        eri = eri.reshape(shape)
        if _needs_ewald_exxdiv(mf, omega):
            eri += _ewald_exxdiv_mo_tensor(mf, coeffs, shape)
        return eri

    def build_eri(with_df):
        return (
            build_eri_block(
                with_df,
                [ctx.orbo_b, ctx.orbo_b, ctx.orbv_a, ctx.orbv_a],
                (ctx.nocc_b, ctx.nocc_b, ctx.nvir_a, ctx.nvir_a)),
            build_eri_block(
                with_df,
                [ctx.orbo_a, ctx.orbo_a, ctx.orbv_b, ctx.orbv_b],
                (ctx.nocc_a, ctx.nocc_a, ctx.nvir_b, ctx.nvir_b)),
            build_eri_block(
                with_df,
                [ctx.orbo_b, ctx.orbv_b, ctx.orbo_a, ctx.orbv_a],
                (ctx.nocc_b, ctx.nvir_b, ctx.nocc_a, ctx.nvir_a)),
            build_eri_block(
                with_df,
                [ctx.orbo_a, ctx.orbv_a, ctx.orbo_b, ctx.orbv_b],
                (ctx.nocc_a, ctx.nvir_a, ctx.nocc_b, ctx.nvir_b)),
        )

    if omega is None or abs(omega) < 1e-14:
        eri_a_b2a, eri_a_a2b, eri_b_b2a, eri_b_a2b = build_eri(mf.with_df)
    else:
        with mf.with_df.range_coulomb(omega) as rsh_df:
            eri_a_b2a, eri_a_a2b, eri_b_b2a, eri_b_a2b = build_eri(rsh_df)

    a_b2a, a_a2b = a
    b_b2a, b_a2b = b

    a_b2a -= np.einsum('ijba->iajb', eri_a_b2a) * hyb
    a_a2b -= np.einsum('ijba->iajb', eri_a_a2b) * hyb
    b_b2a -= np.einsum('ibja->iajb', eri_b_b2a) * hyb
    b_a2b -= np.einsum('ibja->iajb', eri_b_a2b) * hyb



def _mc_scalar_transition_density(ao, orbo, orbv):
    """Build the scalar occupied-virtual transition density on one grid block."""
    rho_o = lib.einsum('rp,pi->ri', ao, orbo)
    rho_v = lib.einsum('rp,pi->ri', ao, orbv)
    return np.einsum('ri,ra->ria', rho_o, rho_v)


def _mc_gradient_transition_density(ao, orbo, orbv, with_tau=False):
    """Build the GGA/MGGA occupied-virtual transition density on one grid block."""
    rho_o = lib.einsum('xrp,pi->xri', ao, orbo)
    rho_v = lib.einsum('xrp,pi->xri', ao, orbv)
    rho_ov = np.einsum('xri,ra->xria', rho_o, rho_v[0])
    rho_ov[1:4] += np.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
    if with_tau:
        tau_ov = np.einsum('xri,xra->ria', rho_o[1:4], rho_v[1:4]) * .5
        rho_ov = np.vstack([rho_ov, tau_ov[np.newaxis]])
    return rho_ov


def _accumulate_mc_kernel_pair(same_target, cross_target, rho_ref, rho_other, kernel_block):
    """Accumulate one multicollinear kernel block into same-spin and cross-spin tensors."""
    if rho_ref.ndim == 3:
        w_ov = np.einsum('ria,r->ria', rho_ref, kernel_block * 2.0)
        same_target += lib.einsum('ria,rjb->iajb', rho_ref, w_ov)
        cross_target += lib.einsum('ria,rjb->iajb', rho_other, w_ov)
    else:
        w_ov = np.einsum('xyr,xria->yria', kernel_block * 2.0, rho_ref)
        same_target += lib.einsum('xria,xrjb->iajb', w_ov, rho_ref)
        cross_target += lib.einsum('xria,xrjb->iajb', w_ov, rho_other)


def get_ab_sf(mf, mo_energy=None, mo_coeff=None, mo_occ=None,
              collinear_samples=30, return_both=False):
    """Construct the multicollinear spin-flip A matrix."""
    if isinstance(mf, scf.rohf.ROHF) or isinstance(mf, scf.hf_symm.SymAdaptedROHF):
        if isinstance(mf, dft.roks.ROKS) or isinstance(mf, dft.rks_symm.SymAdaptedROKS):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    ctx = _build_sf_context(mf, mo_energy=mo_energy, mo_occ=mo_occ, mo_coeff=mo_coeff)
    cell = mf.cell
    kpt = _get_gamma_kpt(mf)

    a_b2a, a_a2b = _initialize_mc_diagonal_blocks(mf, ctx, mo_energy)

    b_b2a = np.zeros((ctx.nocc_b, ctx.nvir_a, ctx.nocc_a, ctx.nvir_b))
    b_a2b = np.zeros((ctx.nocc_a, ctx.nvir_b, ctx.nocc_b, ctx.nvir_a))
    a = (a_b2a, a_a2b)
    b = (b_b2a, b_a2b)

    if _is_pbc_ks(mf):
        ni0 = mf._numint
        ni = pbc_numint2c.NumInt2C()
        ni.collinear = 'mcol'
        ni.collinear_samples = collinear_samples
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, cell.spin)

        _add_mc_hf_terms(mf, ctx, a, b, hyb=hyb)

        if omega != 0:
            _add_mc_hf_terms(mf, ctx, a, b, hyb=alpha - hyb, omega=omega)

        if collinear_samples < 0:
            return a, b

        xctype = ni._xc_type(mf.xc)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * .8 - mem_now)
        fxc = cache_xc_kernel_sf_mc(
            ni, cell, mf.grids, mf.xc, mo_coeff, mo_occ,
            deriv=2, spin=1, kpt=kpt, max_memory=max_memory)[2]
        p0, p1 = 0, 0

        if xctype == 'LDA':
            for ao, _, mask, weight, coords in ni0.block_loop(
                    cell, mf.grids, ctx.nao, 0, kpt, None, max_memory):
                del mask, coords
                p0 = p1
                p1 += weight.shape[0]
                wfxc = fxc[0, 0][..., p0:p1] * weight

                rho_ov_b2a = _mc_scalar_transition_density(
                    ao, ctx.orbo_b, ctx.orbv_a)
                rho_ov_a2b = _mc_scalar_transition_density(
                    ao, ctx.orbo_a, ctx.orbv_b)
                _accumulate_mc_kernel_pair(a_b2a, b_a2b,
                                           rho_ov_b2a, rho_ov_a2b, wfxc)
                _accumulate_mc_kernel_pair(a_a2b, b_b2a,
                                           rho_ov_a2b, rho_ov_b2a, wfxc)

        elif xctype in ('GGA', 'MGGA'):
            with_tau = xctype == 'MGGA'
            for ao, _, mask, weight, coords in ni.block_loop(
                    cell, mf.grids, ctx.nao, 1, kpt, None, max_memory):
                del mask, coords
                p0 = p1
                p1 += weight.shape[0]
                wfxc = fxc[..., p0:p1] * weight

                rho_ov_b2a = _mc_gradient_transition_density(
                    ao, ctx.orbo_b, ctx.orbv_a, with_tau=with_tau)
                rho_ov_a2b = _mc_gradient_transition_density(
                    ao, ctx.orbo_a, ctx.orbv_b, with_tau=with_tau)
                _accumulate_mc_kernel_pair(a_b2a, b_b2a,
                                           rho_ov_b2a, rho_ov_a2b, wfxc)
                _accumulate_mc_kernel_pair(a_a2b, b_a2b,
                                           rho_ov_a2b, rho_ov_b2a, wfxc)

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')
    else:
        _add_mc_hf_terms(mf, ctx, a, b)

    if return_both:
        return a
    return a[1]
