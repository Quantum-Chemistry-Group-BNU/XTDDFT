import numpy as np
import scipy, sys
from pyscf import gto, dft, scf, ao2mo, lib, tddft
from pyscf.dft import numint, numint2c, xc_deriv
from functools import reduce
import functools
from pyscf.lib import logger
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import _dot_ao_ao_sparse, _scale_ao_sparse, _tau_dot_sparse

# import sys
# sys.path.append('/home/lenovo2/usrs/zhw/software/test_git_file')
# from scf_genrep_sftd import _gen_uhf_tda_response_sf

# from utils import unit
ha2eV = 27.2113834


def SF_TDA(mf, isf=-1, davidson=True, method=0):
    print('method=0 (default) ALDA0, method=1 multicollinear, method=2 collinear')
    if isf == -1:  # down
        return SF_TDA_down(mf, method, davidson)
    elif isf == 1:  # up
        return SF_TDA_up(mf, method, davidson)


def mf_info(mf):
    if np.array(mf.mo_coeff).ndim == 3:  # UKS
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
    else:  # ROKS
        mo_energy = np.array([mf.mo_energy, mf.mo_energy])
        mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
        mo_occ = np.zeros((2, len(mf.mo_coeff)))
        mo_occ[0][np.where(mf.mo_occ >= 1)[0]] = 1
        mo_occ[1][np.where(mf.mo_occ >= 2)[0]] = 1
    return mo_energy, mo_occ, mo_coeff


def cache_xc_kernel_sf(mf, mo_coeff, mo_occ, spin=1, max_memory=2000, isf=-1, method=0):  # for ALDA0 and noncollinear
    '''Compute the fxc_sf, which can be used in SF-TDDFT/TDA
    '''
    MGGA_DENSITY_LAPL = False
    with_lapl = MGGA_DENSITY_LAPL
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    mo_energy, mo_occ, mo_coeff = mf_info(mf)

    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
    else:
        ao_deriv = 0  # LDA

    assert mo_coeff[0].ndim == 2
    assert spin == 1

    nao = mo_coeff[0].shape[0]
    dm0 = mf.make_rdm1()
    if np.array(mf.mo_coeff).ndim == 2:
        dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
        dm0.mo_occ = mo_occ
    make_rho = ni._gen_rho_evaluator(mf.mol, dm0, hermi=0, with_lapl=False)[0]

    fxc_abs = []
    # if isf == -1: #
    for ao, mask, weight, coords \
            in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv, max_memory):
        rhoa = make_rho(0, ao, mask, xctype)
        rhob = make_rho(1, ao, mask, xctype)
        if xctype == 'LDA':
            rho = (rhoa, rhob)
        elif method == 0:  # GGA ALDA0
            rha = np.zeros_like(rhoa)
            rhb = np.zeros_like(rhob)
            rha[0] = rhoa[0]
            rhb[0] = rhob[0]
            rho = (rha, rhb)
            rhoa = rhoa[0]
            rhob = rhob[0]
        elif method == 3:
            rho = (rhoa, rhob)
        vxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, xctype=xctype)[1]  #
        vxc_a = vxc[0, 0] * weight
        vxc_b = vxc[1, 0] * weight
        fxc_ab = (vxc_a - vxc_b) / (np.array(rhoa) - np.array(rhob) + 1e-9)
        fxc_abs += list(fxc_ab)

    fxc_abs = np.asarray(fxc_abs)
    return fxc_abs


def nr_uks_fxc_sf_tda(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0, vxc=None, extype=0, max_memory=2000,
                      verbose=None):
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if hermi != 1 and dtype != np.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)

    nao = dms.shape[-1]
    make_rhosf, nset = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)[:2]

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            p0, p1 = p1, p1 + weight.size
            _vxc = vxc[p0:p1]
            for i in range(nset):
                rho1sf = make_rhosf(i, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = rho1sf * _vxc
                else:
                    rhosf = np.zeros_like(rho1sf)
                    rhosf[0] += rho1sf[0]
                    wv = rhosf * _vxc
                yield i, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    vmat = np.zeros((nset, nao, nao))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])
    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5
            aow = _scale_ao_sparse(ao, wv, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])

        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(-1, nao, nao), axes=(0, 2, 1)).reshape(nset, nao, nao)

    elif xctype == 'MGGA':
        # assert not MGGA_DENSITY_LAPL
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


def gen_tda_operation_sf(mf, isf, method):
    if np.array(mf.mo_coeff).ndim == 3:  # UKS
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
    else:  # ROKS
        mo_energy = np.array([mf.mo_energy, mf.mo_energy])
        mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
        mo_occ = np.zeros((2, len(mf.mo_coeff)))
        mo_occ[0][np.where(mf.mo_occ >= 1)[0]] = 1
        mo_occ[1][np.where(mf.mo_occ >= 2)[0]] = 1

    mol = mf.mol
    assert (mo_coeff[0].dtype == np.double)
    nao, nmo = mo_coeff[0].shape
    occidxa = np.where(mo_occ[0] == 1)[0]
    occidxb = np.where(mo_occ[1] == 1)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]
    # nc = noccb
    # nv = nvira
    # print('nc,nv',nc,nv)
    # no = nocca - noccb
    # nvir = no+nv
    # iden_C = np.eye(nc)
    # iden_V = np.eye(nv)
    # iden_O = np.eye(no)
    # delta_ij = np.eye(nocca)
    # delta_ab = np.eye(nvirb)

    dm = mf.make_rdm1()
    vhf = mf.get_veff(mf.mol, dm)
    h1e = mf.get_hcore()
    focka = h1e + vhf[0]
    fockb = h1e + vhf[1]
    fockA = mo_coeff[0].T @ focka @ mo_coeff[0]
    fockB = mo_coeff[1].T @ fockb @ mo_coeff[1]

    if isf == -1:  # spin down
        e_ia = (mo_energy[1][viridxb, None] - mo_energy[0][occidxa]).T
        hdiag = e_ia.ravel()
        ndim = (nocca, nvirb)
        orbov = (orboa, orbvb)
    elif isf == 1:  # spin up
        e_ai = (mo_energy[0][viridxa, None] - mo_energy[1][occidxb]).T
        hdiag = e_ai.ravel()
        ndim = (noccb, nvira)
        orbov = (orbob, orbva)
    if method == 1:
        vresp = _gen_uhf_tda_response_sf(mf, hermi=0, collinear_samples=50)
    else:
        vresp = gen_response_sf(mf, hermi=0, method=method)

    # @profile
    # print('isf=',isf)
    def vind(zs0):  # vector-matrix product for indexed operations
        ndim0, ndim1 = ndim  # ndom0:numuber of occ orbitals, ndim1:number of vir orbitals
        orbo, orbv = orbov  # mo_coeff for alpha and beta
        zs = np.asarray(zs0).reshape(-1, ndim0, ndim1)
        # print('zs.shape ',zs.shape)
        # print(zs)
        vs = np.zeros_like(zs)
        dmov = lib.einsum('xov,qv,po->xpq', zs, orbv.conj(), orbo)  # (x,nmo,nmo)
        v1ao = vresp(np.asarray(dmov))  # with density and get response function
        vs += lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)  # (-1,nocca,nvirb)
        if isf == -1:  # spin down
            vs += np.einsum('ab,xib->xia', fockB[noccb:, noccb:], zs, optimize=True) - \
                  np.einsum('ij,xja->xia', fockA[:nocca, :nocca], zs, optimize=True)
            # vs += lib.einsum('ov,xov->xov', e_ia, zs)
        elif isf == 1:  # spin up
            vs += np.einsum('ab,xib->xia', fockA[nocca:, nocca:], zs, optimize=True) - \
                  np.einsum('ij,xja->xia', fockB[:noccb, :noccb], zs, optimize=True)
        # print('zs.shape',zs.shape)
        hx = vs.reshape(zs.shape[0], -1)
        return hx

    return vind, hdiag


def gen_response_sf(mf, hermi=0, max_memory=None, method=0):
    mo_energy, mo_occ, mo_coeff = mf_info(mf)
    mol = mf.mol
    if isinstance(mf, scf.hf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory * .8 - mem_now)
        if method == 0:
            vxc = cache_xc_kernel_sf(mf, mo_coeff, mo_occ, 1, max_memory, isf=-1, method=method)  # XC kerkel
        dm0 = None

        def vind(dm1):
            if method == 0:
                v1 = nr_uks_fxc_sf_tda(ni, mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,  # XC * dm1
                                       vxc, max_memory=max_memory)
            else:
                v1 = np.zeros_like(dm1)

            if not hybrid:
                # No with_j because = 0 in spin flip part.
                pass
            else:
                vk = mf.get_k(mol, dm1, hermi=hermi)
                vk *= hyb
                if omega > 1e-10:  # For range separated Coulomb
                    vk += mf.get_k(mol, dm1, hermi, omega) * (alpha - hyb)
                v1 -= vk
            return v1
    if not isinstance(mf, scf.hf.KohnShamDFT):  # in HF case
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)
    # if hf_correction: # for \Delta A
    #    def vind(dm1):
    #        vj,vk = mf.get_jk(mol,dm1,hermi=hermi)
    #        return vj,vk
    return vind


def deal_v_davidson2(mf, nstates, v):
    _, mo_occ, _ = mf_info(mf)
    occidx_a = np.where(mo_occ[0] == 1)[0]
    viridx_a = np.where(mo_occ[0] == 0)[0]
    occidx_b = np.where(mo_occ[1] == 1)[0]
    # viridx_b = np.where(mo_occ[1]==0)[0]
    nc = len(occidx_b)
    nv = len(viridx_a)
    no = len(occidx_a) - nc
    cv = np.zeros((nstates, nc, nv))
    for state in range(nstates):
        tmp_data = v[:, state]
        for i in range(nc):
            cv[state, i, :] += tmp_data[i * nv:i * nv + nv]
    return cv


def deal_v_davidson(mf, nstates, v, remove=False):
    # change davidson data form like nvir|nvir|nvir|...(alpha->beta nc|no -> no|nv)  to cv|co|ov|oo
    _, mo_occ, _ = mf_info(mf)
    occidx_a = np.where(mo_occ[0] == 1)[0]
    viridx_a = np.where(mo_occ[0] == 0)[0]
    occidx_b = np.where(mo_occ[1] == 1)[0]
    # viridx_b = np.where(mo_occ[1]==0)[0]
    nc = len(occidx_b)
    nv = len(viridx_a)
    no = len(occidx_a) - nc
    cv = np.zeros((nstates, nc, nv))
    co = np.zeros((nstates, nc, no))
    ov = np.zeros((nstates, no, nv))
    if remove:
        oo = np.zeros((nstates, no * no - 1))
    else:
        oo = np.zeros((nstates, no, no))
    nvir = no + nv
    passed = nc * nvir
    if nstates == (nc + no) * (no + nv):
        nstates = nstates - 1
    else:
        nstates = nstates
    for state in range(nstates):
        tmp_data = v[:, state]
        # print(tmp_data[passed:])
        for i in range(nc):
            cv[state, i, :] += tmp_data[i * nvir + no:i * nvir + no + nv]
            co[state, i, :] += tmp_data[i * nvir:i * nvir + no]
        if remove:
            for i in range(no - 1):
                oo[state, i * no:(i + 1) * no] += tmp_data[passed + i * nvir:passed + i * nvir + no]
                ov[state, i, :] += tmp_data[passed + i * nvir + no:passed + i * nvir + no + nv]
            oo[state, (no - 1) * no:] += tmp_data[passed + (no - 1) * nvir:passed + (no - 1) * nvir + no - 1]
            ov[state, no - 1, :] += tmp_data[passed + (no - 1) * nvir + no - 1:]
        else:
            for i in range(no):
                oo[state, i, :] += tmp_data[passed + i * nvir:passed + i * nvir + no]
                ov[state, i, :] += tmp_data[passed + i * nvir + no:passed + i * nvir + no + nv]

    v = np.hstack([cv.reshape(nstates, -1), co.reshape(nstates, -1), ov.reshape(nstates, -1), oo.reshape(nstates, -1)])
    return v.T


def init_guess(mf, nstates, isf=-1):
    mo_energy, mo_occ, mo_coeff = mf_info(mf)

    occidxa = np.where(mo_occ[0] > 0)[0]
    occidxb = np.where(mo_occ[1] > 0)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    # print('nc,nv',len(occidxb),len(viridxa))
    if isf == 1:
        e_ia_b2a = (mo_energy[0][viridxa, None] - mo_energy[1][occidxb]).T
        e_ia_b2a = e_ia_b2a.ravel()
        nov_b2a = e_ia_b2a.size
        nstates = min(nstates, nov_b2a)
        e_threshold = np.sort(e_ia_b2a)[nstates - 1]
        e_threshold += 1e-5
        idx = np.where(e_ia_b2a <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov_b2a))
        for i, j in enumerate(idx):
            x0[i, j] = 1
    elif isf == -1:
        e_ia_a2b = (mo_energy[1][viridxb, None] - mo_energy[0][occidxa]).T
        e_ia_a2b = e_ia_a2b.ravel()
        nov_a2b = e_ia_a2b.size
        nstates = min(nstates, nov_a2b)
        e_threshold = np.sort(e_ia_a2b)[nstates - 1]
        e_threshold += 1e-5

        idx = np.where(e_ia_a2b <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov_a2b))
        for i, j in enumerate(idx):
            x0[i, j] = 1
    return x0


def davidson_process(mf, nstates, method, isf=-1):
    vind, hdiag = gen_tda_operation_sf(mf, isf, method)
    precond = hdiag
    # start_t = time.time()
    x0 = init_guess(mf, nstates, isf)
    # print(x0)
    # print('x0.shape',x0.shape)
    # end_t = time.time()
    # print(f'init_guess times use {(end_t-start_t)/3600:6.4f} hours')
    # print('x0.shape ',x0.shape)
    converged, e, x1 = lib.davidson1(vind, x0, precond,
                                     tol=1e-7, lindep=1e-14,
                                     nroots=nstates,
                                     max_cycle=3000)
    # end_time = time.time()
    # print(f'davidson time use {(end_time-end_t)/3600} hours')
    v = np.array(x1).T
    # print(self.v.shape)
    if isf == -1:
        v = deal_v_davidson(mf, nstates, v)
    # elif isf == 1:
    #    pass
    # v = deal_v_davidson2(mf,nstates,v)
    print('Converged ', converged)
    return e, v


class SF_TDA_up():
    def __init__(self, mf, method, davidson=False):
        if np.array(mf.mo_coeff).ndim == 3:  # UKS
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = mf.mo_occ
            # self.type_u = True

        else:  # ROKS
            self.mo_energy = np.array([mf.mo_energy, mf.mo_energy])
            self.mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            self.mo_occ = np.zeros((2, len(mf.mo_coeff)))
            self.mo_occ[0][np.where(mf.mo_occ >= 1)[0]] = 1
            self.mo_occ[1][np.where(mf.mo_occ >= 2)[0]] = 1
            # self.type_u = False
        mol = mf.mol
        self.mf = mf
        self.nao = mol.nao_nr()
        self.method = method
        self.davidson = davidson
        self.collinear = False
        occidx_a = np.where(self.mo_occ[0] == 1)[0]
        viridx_a = np.where(self.mo_occ[0] == 0)[0]
        occidx_b = np.where(self.mo_occ[1] == 1)[0]
        viridx_b = np.where(self.mo_occ[1] == 0)[0]
        self.orbo_a = self.mo_coeff[0][:, occidx_a]
        self.orbv_a = self.mo_coeff[0][:, viridx_a]
        self.orbo_b = self.mo_coeff[1][:, occidx_b]
        self.orbv_b = self.mo_coeff[1][:, viridx_b]
        nocc_a = self.orbo_a.shape[1]
        nvir_a = self.orbv_a.shape[1]
        nocc_b = self.orbo_b.shape[1]
        nvir_b = self.orbv_b.shape[1]
        nmo_a = nocc_a + nvir_a
        self.nc = nocc_b
        self.nv = nvir_a
        self.no = nocc_a - nocc_b
        # delta_ij = np.eye(nocc_a)
        # delta_ab = np.eye(nvir_b)

    def get_Amat(self):
        delta_ij = np.eye(self.nc + self.no)
        delta_ab = np.eye(self.nv + self.no)
        mf = self.mf
        mol = self.mf.mol
        a_b2a = np.zeros((self.nc, self.nv, self.nc, self.nv))
        if self.no == 0:
            dm = mf.make_rdm1()
            vhf = mf.get_veff(mf.mol, dm)
            h1e = mf.get_hcore()
            focka = h1e + vhf
            fockb = h1e + vhf
            fockA = self.mo_coeff[0].T @ focka @ self.mo_coeff[0]
            fockB = self.mo_coeff[1].T @ fockb @ self.mo_coeff[1]
        else:
            dm = mf.make_rdm1()
            vhf = mf.get_veff(mf.mol, dm)
            h1e = mf.get_hcore()
            focka = h1e + vhf[0]
            fockb = h1e + vhf[1]
            fockA = self.mo_coeff[0].T @ focka @ self.mo_coeff[0]
            fockB = self.mo_coeff[1].T @ fockb @ self.mo_coeff[1]

        def add_hf_(a_b2a, hyb=1):

            eri_mo = ao2mo.general(mol, [self.orbo_b, self.orbo_b, self.orbv_a, self.orbv_a], compact=False)
            eri_mo = eri_mo.reshape(self.nc, self.nc, self.nv, self.nv)
            a_b2a -= np.einsum('ijba->iajb', eri_mo) * hyb
            return a_b2a

        try:
            xctype = mf.xc
        except:
            xctype = None
            # eri_mo_b2a = ao2mo.general(self.mol, [self.orbo_b,self.orbo_b,self.orbv_a,self.orbv_a], compact=False)
            # eri_a_b2a = eri_mo_a2b.reshape(nocc_b,nocc_b,nvir_a,nvir_a)
            # a_b2a -= np.einsum('ijba->iajb', eri_a_b2a) * hyb
            # del eri_mo_b2a

        if xctype is not None:
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            if mf.nlc or ni.libxc.is_nlc(mf.xc):
                logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                                'deriviative is not available. Its contribution is '
                                'not included in the response function.')
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
            # print('omega alpha hyb',omega, alpha, hyb)
            if hyb != 0:
                a_b2a = add_hf_(a_b2a, hyb)

            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1()
            # if np.array(mf.mo_coeff).ndim==2 and not mol.symmetry:
            if np.array(mf.mo_coeff).ndim == 2:
                dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
                dm0.mo_occ = self.mo_occ
            make_rho = ni._gen_rho_evaluator(mf.mol, dm0, hermi=0, with_lapl=False)[0]
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory * .8 - mem_now)

        if xctype == 'LDA' and not self.collinear:
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mf.mol, mf.grids, self.nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                exc, vxc, fxc, kxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, omega=omega, xctype=xctype)  # vxc.shape[2,1,N]
                vxc_a = vxc[0, 0] * weight
                vxc_b = vxc[1, 0] * weight
                fxc_ab = (vxc_a - vxc_b) / (rho0a - rho0b + 1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao, self.orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, self.orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, self.orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, self.orbv_b)
                rho_ov_b2a = np.einsum('ri,ra->ria', rho_o_b, rho_v_a)
                # rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                w_ov = np.einsum('ria,r->ria', rho_ov_b2a, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                a_b2a += iajb
                # w_ov = np.einsum('ria,r->ria', rho_ov_a2b, fxc_ab)
                # iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                # a_a2b += iajb


        elif xctype == 'GGA' and not self.collinear:
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mf.mol, mf.grids, self.nao, ao_deriv,
                                     max_memory):  # ao(4,N,nao):AO values and derivatives in x,y,z compoents in grids
                rho0a = make_rho(0, ao, mask,
                                 xctype)  # (4,N):density and "density derivatives" for x,y,z components in grids
                rho0b = make_rho(1, ao, mask, xctype)
                rha = np.zeros_like(rho0a)
                rha[0] = rho0a[0]
                rhb = np.zeros_like(rho0b)
                rhb[0] = rho0b[0]
                exc, vxc, fxc, kxc = ni.eval_xc_eff(mf.xc, (rha, rhb), deriv=1, omega=omega,
                                                    xctype=xctype)  # vxc.shape(2,4,N)
                vxc_a = vxc[0, 0] * weight  # first order derivatives about \rho in \alpha
                vxc_b = vxc[1, 0] * weight  # \beta
                fxc_ab = (vxc_a - vxc_b) / (rho0a[0] - rho0b[0] + 1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao[0], self.orbo_a)  # (N,i)
                rho_v_a = lib.einsum('rp,pi->ri', ao[0], self.orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao[0], self.orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao[0], self.orbv_b)
                # rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                rho_ov_b2a = np.einsum('ri,ra->ria', rho_o_b, rho_v_a)
                w_ov = np.einsum('ria,r->ria', rho_ov_b2a, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                a_b2a += iajb
        elif xctype == None:
            add_hf_(a_b2a, hyb=1)
        self.A = (a_b2a + np.einsum('ij,ab->iajb', delta_ij[self.no:, self.no:],
                                    fockA[self.nc + self.no:, self.nc + self.no:]) \
                  - np.einsum('ij,ab->iajb', fockB[:self.nc, :self.nc], delta_ab[self.no:, self.no:])).reshape(
            (self.nc * self.nv, self.nc * self.nv))

    def kernel(self, nstates=1):
        # if nstates==None:
        #    nstates=self.nstates
        self.nstates = nstates
        if self.davidson:
            self.e, self.v = davidson_process(self.mf, nstates, self.method, isf=1)
        else:
            if self.method == 1:  # multicollinear
                self.A = get_ab_sf(self.mf)[0]
            else:
                self.get_Amat()
            self.e, self.v = scipy.linalg.eigh(self.A)
        return self.e[:nstates] * ha2eV, self.v[:, :nstates]

    def analyse(self):
        nc = self.nc
        nv = self.nv
        for nstate in range(self.nstates):
            value = self.v[:, nstate]
            x_cv_ab = value[:nc * nv].reshape(nc, nv)
            print(f'Excited state {nstate + 1} {self.e[nstate] * ha2eV:10.5f} eV')
            for o, v in zip(*np.where(abs(x_cv_ab) > 0.1)):
                print(
                    f'{100 * x_cv_ab[o, v] ** 2:3.0f}% CV(ab) {o + 1}a -> {v + 1 + self.nc + self.no}b {x_cv_ab[o, v]:10.5f}')
            print(' ')


# spin_down
class SF_TDA_down():
    def __init__(self, mf, method, davidson=True):

        if np.array(mf.mo_coeff).ndim == 3:  # UKS
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = mf.mo_occ
        else:  # ROKS
            self.mo_energy = np.array([mf.mo_energy, mf.mo_energy])
            self.mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            self.mo_occ = np.zeros((2, len(mf.mo_coeff)))
            self.mo_occ[0][np.where(mf.mo_occ >= 1)[0]] = 1
            self.mo_occ[1][np.where(mf.mo_occ >= 2)[0]] = 1

        mol = mf.mol
        self.mf = mf
        self.nao = mol.nao_nr()
        self.davidson = davidson
        self.method = method
        occidx_a = np.where(self.mo_occ[0] == 1)[0]
        viridx_a = np.where(self.mo_occ[0] == 0)[0]
        occidx_b = np.where(self.mo_occ[1] == 1)[0]
        viridx_b = np.where(self.mo_occ[1] == 0)[0]
        self.orbo_a = self.mo_coeff[0][:, occidx_a]
        self.orbv_a = self.mo_coeff[0][:, viridx_a]
        self.orbo_b = self.mo_coeff[1][:, occidx_b]
        self.orbv_b = self.mo_coeff[1][:, viridx_b]
        nocc_a = self.orbo_a.shape[1]
        nvir_a = self.orbv_a.shape[1]
        nocc_b = self.orbo_b.shape[1]
        nvir_b = self.orbv_b.shape[1]
        self.nc = nocc_b
        self.nv = nvir_a
        self.no = nocc_a - nocc_b

    def get_Amat(self):
        a_a2b = np.zeros((self.nc + self.no, self.nv + self.no, self.nc + self.no, self.nv + self.no))
        nocc_a = self.nc + self.no
        nvir_b = self.nv + self.no
        mol = self.mf.mol
        delta_ij = np.eye(nocc_a)
        delta_ab = np.eye(nvir_b)
        dm = self.mf.make_rdm1()
        vhf = self.mf.get_veff(mol, dm)
        h1e = self.mf.get_hcore()
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fockA = self.mo_coeff[0].T @ focka @ self.mo_coeff[0]
        fockB = self.mo_coeff[1].T @ fockb @ self.mo_coeff[1]

        nc = self.nc
        nv = self.nv
        no = self.no

        def add_hf_(a_a2b, hyb=1):

            # eri_a_b2a = ao2mo.general(mol, [orbo_b,orbo_b,orbv_a,orbv_a], compact=False)
            eri_mo = ao2mo.general(mol, [self.orbo_a, self.orbo_a, self.orbv_b, self.orbv_b], compact=False)

            # eri_a_b2a = eri_a_b2a.reshape(nocc_b,nocc_b,nvir_a,nvir_a)
            eri_mo = eri_mo.reshape(nocc_a, nocc_a, nvir_b, nvir_b)

            # a_b2a, a_a2b = a

            # a_b2a-= np.einsum('ijba->iajb', eri_a_b2a) * hyb
            a_a2b -= np.einsum('ijba->iajb', eri_mo) * hyb
            del eri_mo

        try:
            xctype = self.mf.xc
        except:
            xctype = None  # only work for HF
            eri_mo_a2b = ao2mo.general(mol, [self.orbo_a, self.orbo_a, self.orbv_b, self.orbv_b], compact=False)
            eri_a_a2b = eri_mo_a2b.reshape(nocc_a, nocc_a, nvir_b, nvir_b)
            a_a2b -= np.einsum('ijba->iajb', eri_a_a2b)
            del eri_mo_a2b

        if xctype is not None:
            ni = self.mf._numint
            ni.libxc.test_deriv_order(self.mf.xc, 2, raise_error=True)
            if self.mf.nlc or ni.libxc.is_nlc(self.mf.xc):
                logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                                'deriviative is not available. Its contribution is '
                                'not included in the response function.')
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, self.mf.mol.spin)
            # print('omega alpha hyb',omega, alpha, hyb)
            if hyb != 0:
                add_hf_(a_a2b, hyb)

            xctype = ni._xc_type(self.mf.xc)
            # print(xctype)
            # if mol.symmetry:
            #    #dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
            # else:
            #    #dm0 = mf.make_rdm1(mf.mo_coeff,mo_occ)
            dm0 = self.mf.make_rdm1()
            # if np.array(mf.mo_coeff).ndim==2 and not mol.symmetry:
            if np.array(self.mf.mo_coeff).ndim == 2:
                dm0.mo_coeff = (self.mf.mo_coeff, self.mf.mo_coeff)
                dm0.mo_occ = self.mo_occ
            make_rho = ni._gen_rho_evaluator(self.mf.mol, dm0, hermi=0, with_lapl=False)[0]
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, self.mf.max_memory * .8 - mem_now)

        if xctype == 'LDA' and self.method != 2:
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, self.mf.grids, self.nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                exc, vxc, fxc, kxc = ni.eval_xc_eff(self.mf.xc, rho, deriv=1, omega=omega,
                                                    xctype=xctype)  # vxc.shape[2,1,N]
                vxc_a = vxc[0, 0] * weight
                vxc_b = vxc[1, 0] * weight
                fxc_ab = (vxc_a - vxc_b) / (rho0a - rho0b + 1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao, self.orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, self.orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, self.orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, self.orbv_b)
                rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                w_ov = np.einsum('ria,r->ria', rho_ov_a2b, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                a_a2b += iajb


        elif xctype == 'GGA' and self.method != 2:
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, self.mf.grids, self.nao, ao_deriv,
                                     max_memory):  # ao(4,N,nao):AO values and derivatives in x,y,z compoents in grids
                rho0a = make_rho(0, ao, mask,
                                 xctype)  # (4,N):density and "density derivatives" for x,y,z components in grids
                rho0b = make_rho(1, ao, mask, xctype)
                rha = np.zeros_like(rho0a)
                rha[0] = rho0a[0]
                rhb = np.zeros_like(rho0b)
                rhb[0] = rho0b[0]
                exc, vxc, fxc, kxc = ni.eval_xc_eff(self.mf.xc, (rha, rhb), deriv=1, omega=omega,
                                                    xctype=xctype)  # vxc.shape(2,4,N)
                vxc_a = vxc[0, 0] * weight  # first order derivatives about \rho in \alpha
                vxc_b = vxc[1, 0] * weight  # \beta
                fxc_ab = (vxc_a - vxc_b) / (rho0a[0] - rho0b[0] + 1e-9)
                rho_o_a = lib.einsum('rp,pi->ri', ao[0], self.orbo_a)  # (N,i)
                rho_v_a = lib.einsum('rp,pi->ri', ao[0], self.orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao[0], self.orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao[0], self.orbv_b)
                rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)
                w_ov = np.einsum('ria,r->ria', rho_ov_a2b, fxc_ab)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                a_a2b += iajb

        iden_C = np.identity(nc)
        iden_O = np.identity(no)
        iden_V = np.identity(nv)
        fockA_C = fockA[:nc, :nc]
        fockA_O = fockA[nc:nc + no, nc:nc + no]
        fockA_V = fockA[nc + no:, nc + no:]
        fockB_C = fockB[:nc, :nc]
        fockB_O = fockB[nc:nc + no, nc:nc + no]
        fockB_V = fockB[nc + no:, nc + no:]
        dim = (nc + no) * (nv + no)
        print('The dims of A matrix ', dim)
        Amat = np.zeros((dim, dim))
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv

        # ----- Diagonal blocks -----
        # CV-CV  a_a2b(nocc_a,nvir_b,nocc_a,nvir_b) (7,3,7,3) iajb
        Amat[:dim1, :dim1] = np.einsum('ij,ab->iajb', iden_C, fockB_V).reshape(nv * nc, nv * nc) \
                             - np.einsum('ji,ab->iajb', fockA_C, iden_V).reshape(nv * nc, nv * nc) \
                             + a_a2b[:nc, no:, :nc, no:].reshape(nv * nc, nv * nc)
        # CO-CO ivju
        Amat[dim1:dim2, dim1:dim2] = np.einsum('ij,xy->ixjy', iden_C, fockB_O).reshape(no * nc, no * nc) \
                                     - np.einsum('ji,xy->ixjy', fockA_C, iden_O).reshape(no * nc, no * nc) \
                                     + a_a2b[:nc, :no, :nc, :no].reshape(no * nc, no * nc)
        # OV-OV vaub
        Amat[dim2:dim3, dim2:dim3] = np.einsum('xy,ab->xayb', iden_O, fockB_V).reshape(nv * no, nv * no) \
                                     - np.einsum('yx,ab->xayb', fockA_O, iden_V).reshape(nv * no, nv * no) \
                                     + a_a2b[nc:, no:, nc:, no:].reshape(nv * no, nv * no)
        # OO-OO utvw
        Amat[dim3:, dim3:] = np.einsum('uv,tw->utvw', iden_O, fockB_O).reshape(no * no, no * no) \
                             - np.einsum('vu,tw->utvw', fockA_O, iden_O).reshape(no * no, no * no) \
                             + a_a2b[nc:nc + no, :no, nc:nc + no, :no].reshape(no * no, no * no)

        # ----- Off-Diagonal blocks -----
        # CV-CO, CO-CV iaju
        tmp_CV_CO = np.einsum('ij,ay->iajy', iden_C, fockB[nc + no:, nc:nc + no]).reshape(nv * nc, no * nc) \
                    + a_a2b[:nc, no:, :nc, :no].reshape(nv * nc, no * nc)
        Amat[:dim1, dim1:dim2] = tmp_CV_CO
        Amat[dim1:dim2, :dim1] = tmp_CV_CO.T
        # CV-OV, OV-CV iaub
        tmp_CV_OV = - np.einsum('yi,ab->iayb', fockA[nc:nc + no, :nc], iden_V).reshape(nv * nc, nv * no) \
                    + a_a2b[:nc, no:, nc:nc + no, no:].reshape(nv * nc, nv * no)
        Amat[:dim1, dim2:dim3] = tmp_CV_OV
        Amat[dim2:dim3, :dim1] = tmp_CV_OV.T
        # CO-OV, OV-CO ivub
        tmp_CO_OV = a_a2b[:nc, :no, nc:nc + no, no:].reshape(no * nc, nv * no)
        Amat[dim1:dim2, dim2:dim3] = tmp_CO_OV
        Amat[dim2:dim3, dim1:dim2] = tmp_CO_OV.T

        # --- blocks involving OO ---
        # CV-OO, OO-CV iauw
        tmp_CV_OO = a_a2b[:nc, no:, nc:nc + no, :no].reshape(nv * nc, no * no)
        Amat[:dim1, dim3:] = tmp_CV_OO
        Amat[dim3:, :dim1] = tmp_CV_OO.T
        # CO-OO, OO-CO  ivuw
        tmp_CO_OO = - np.einsum('yi,WZ->iWyZ', fockA[nc:nc + no, :nc], iden_O).reshape(no * nc, no * no) \
                    + a_a2b[:nc, :no, nc:nc + no, :no].reshape(no * nc, no * no)
        Amat[dim1:dim2, dim3:] = tmp_CO_OO
        Amat[dim3:, dim1:dim2] = tmp_CO_OO.T
        # OV-OO, OO-OV vauw
        tmp_OV_OO = np.einsum('yx,aZ->xayZ', iden_O, fockB[nc + no:, nc:nc + no]).reshape(nv * no, no * no) \
                    + a_a2b[nc:, no:, nc:, :no].reshape(nv * no, no * no)
        Amat[dim2:dim3, dim3:] = tmp_OV_OO
        Amat[dim3:, dim2:dim3] = tmp_OV_OO.T

        del a_a2b
        return Amat

    def analyse(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        Ds = []
        for nstate in range(self.nstates):
            value = self.v[:, nstate]
            x_cv_ab = value[:nc * nv].reshape(nc, nv)
            x_co_ab = value[nc * nv:nc * nv + nc * no].reshape(nc, no)
            x_ov_ab = value[nc * nv + nc * no:nc * nv + nc * no + no * nv].reshape(no, nv)

            x_oo_ab = value[nc * nv + nc * no + no * nv:].reshape(no, no)
            Dp_ab = 0.
            Dp_ab += sum(sum(x_cv_ab * x_cv_ab)) - sum(sum(x_oo_ab * x_oo_ab))
            for i in range(no):
                for j in range(no):
                    Dp_ab += x_oo_ab[i, i] * x_oo_ab[j, j]

            print(f'Excited state {nstate + 1} {self.e[nstate] * ha2eV:10.5f} eV D<S^2>={-no + 1 + Dp_ab:5.2f}')
            for o, v in zip(*np.where(abs(x_cv_ab) > 0.1)):
                print(
                    f'{100 * x_cv_ab[o, v] ** 2:3.0f}% CV(ab) {o + 1}a -> {v + 1 + self.nc + self.no}b {x_cv_ab[o, v]:10.5f}')
            for o, v in zip(*np.where(abs(x_co_ab) > 0.1)):
                print(f'{100 * x_co_ab[o, v] ** 2:3.0f}% CO(ab) {o + 1}a -> {v + 1 + self.nc}b {x_co_ab[o, v]:10.5f}')
            for o, v in zip(*np.where(abs(x_ov_ab) > 0.1)):
                print(
                    f'{100 * x_ov_ab[o, v] ** 2:3.0f}% OV(ab) {o + nc + 1}a -> {v + 1 + self.nc + self.no}b {x_ov_ab[o, v]:10.5f}')
            for o, v in zip(*np.where(abs(x_oo_ab) > 0.1)):
                print(
                    f'{100 * x_oo_ab[o, v] ** 2:3.0f}% OO(ab) {o + nc + 1}a -> {v + 1 + self.nc}b {x_oo_ab[o, v]:10.5f}')
            print(' ')
            Ds.append(-no + 1 + Dp_ab)
        return None

    def kernel(self, nstates=1):
        self.nstates = nstates
        if self.davidson:
            self.e, self.v = davidson_process(self.mf, nstates, isf=-1, method=self.method)
        else:
            if self.method == 1:  # multicollinear function
                A = get_ab_sf(self.mf)[1]
                dim = (self.nc + self.no) * (self.nv + self.no)
                self.A = A.reshape((dim, dim))
            else:
                self.A = self.get_Amat()
            self.e, self.v = scipy.linalg.eigh(self.A)
        return self.e[:nstates] * ha2eV, self.v[:, :nstates]


# code from pyscf-forge to construct multicollinear functional
def _gen_uhf_tda_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0, collinear_samples=50, max_memory=None):
    '''Generate a function to compute the product of Spin Flip UKS response function
    and UKS density matrices.
    '''
    mo_energy, mo_occ, mo_coeff = mf_info(mf)
    # assert isinstance(mf, (uhf.UHF))
    # if mo_coeff is None: mo_coeff = mf.mo_coeff
    # if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol

    ni = numint2c.NumInt2C()
    ni.collinear = 'mcol'
    ni.collinear_samples = collinear_samples
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    if mf.nlc or ni.libxc.is_nlc(mf.xc):
        logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    # mf can be pbc.dft.UKS object with multigrid
    if (not hybrid and
            'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
        raise NotImplementedError("Spin Flip TDDFT doesn't support pbc calculations.")

    fxc = cache_xc_kernel_sf_mc(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
    # print('fxs')
    # print(fxc)
    dm0 = None

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * .8 - mem_now)

    def vind(dm1):
        in2 = numint.NumInt()
        v1 = nr_uks_fxc_sf_tda_mc(in2, mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                  None, None, fxc, max_memory=max_memory)
        if not hybrid:
            # No with_j because = 0 in spin flip part.
            pass
        else:
            vk = mf.get_k(mol, dm1, hermi=hermi)
            vk *= hyb
            if omega > 1e-10:  # For range separated Coulomb
                vk += mf.get_k(mol, dm1, hermi, omega) * (alpha - hyb)
            v1 -= vk
        return v1

    return vind


def __mcfun_fn_eval_xc(ni, xc_code, xctype, rho, deriv):
    evfk = ni.eval_xc_eff(xc_code, rho, deriv=deriv, xctype=xctype)
    for order in range(1, deriv + 1):
        if evfk[order] is not None:
            evfk[order] = xc_deriv.ud2ts(evfk[order])
    return evfk


# This function can be merged with pyscf.dft.numint2c.mcfun_eval_xc_adapter()
# This function should be a class function in the Numint2c class.
def mcfun_eval_xc_adapter_sf(ni, xc_code):
    '''Wrapper to generate the eval_xc function required by mcfun

    Kwargs:
        dim: int
            eval_xc_eff_sf is for mc collinear sf tddft/ tda case.add().
    '''

    try:
        import mcfun
    except ImportError:
        raise ImportError('This feature requires mcfun library.\n'
                          'Try install mcfun with `pip install mcfun`')

    xctype = ni._xc_type(xc_code)
    fn_eval_xc = functools.partial(__mcfun_fn_eval_xc, ni, xc_code, xctype)
    nproc = lib.num_threads()

    def eval_xc_eff(xc_code, rho, deriv=1, omega=None, xctype=None,
                    verbose=None):
        return mcfun.eval_xc_eff_sf(
            fn_eval_xc, rho, deriv,
            collinear_samples=ni.collinear_samples, workers=nproc)

    return eval_xc_eff


# This function should be a class function in the Numint2c class.
def cache_xc_kernel_sf_mc(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=1, max_memory=2000):
    '''Compute the fxc_sf, which can be used in SF-TDDFT/TDA
    '''
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

    ni = numint.NumInt()
    for ao, mask, weight, coords \
            in self.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        rhoa.append(ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl))
        rhob.append(ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl))
    rho_ab = (np.hstack(rhoa), np.hstack(rhob))
    rho_ab = np.asarray(rho_ab)
    rho_tmz = np.zeros_like(rho_ab) + 1e-9
    rho_tmz[0] += rho_ab[0] + rho_ab[1]
    rho_tmz[1] += rho_ab[0] - rho_ab[1]
    eval_xc = mcfun_eval_xc_adapter_sf(self, xc_code)
    fxc_sf = eval_xc(xc_code, rho_tmz, deriv=2, xctype=xctype)
    return fxc_sf


def nr_uks_fxc_sf_tda_mc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0, rho0=None,
                         vxc=None, fxc=None, extype=0, max_memory=2000, verbose=None):
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if hermi != 1 and dtype != np.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)

    nao = dms.shape[-1]
    make_rhosf, nset = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)[:2]

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[..., p0:p1]
            for i in range(nset):
                rho1sf = make_rhosf(i, ao, mask, xctype)
                if xctype == 'LDA':
                    # *2.0 becausue kernel xx,yy parts.
                    wv = rho1sf * _fxc[0, 0] * 2.0 * weight
                else:
                    # *2.0 becausue kernel xx,yy parts.
                    wv = lib.einsum('bg,abg->ag', rho1sf, _fxc * 2.0) * weight
                yield i, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    vmat = np.zeros((nset, nao, nao))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])
    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5
            aow = _scale_ao_sparse(ao, wv, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])

        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(-1, nao, nao), axes=(0, 2, 1)).reshape(nset, nao, nao)

    elif xctype == 'MGGA':
        # assert not MGGA_DENSITY_LAPL
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


# construct full A matrix for mulitcollinear function
def get_ab_sf(mf, mo_energy=None, mo_coeff=None, mo_occ=None, collinear_samples=50):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)

    Spin symmetry is not considered in the returned A, B lists.
    List A has two items: (A_baba, A_abab).
    List B has two items: (B_baab, B_abba).
    '''
    # if mo_energy is None: mo_energy = mf.mo_energy
    # if mo_coeff is None: mo_coeff = mf.mo_coeff
    # if mo_occ is None: mo_occ = mf.mo_occ
    mo_energy, mo_occ, mo_coeff = mf_info(mf)

    mol = mf.mol
    nao = mol.nao_nr()
    occidx_a = np.where(mo_occ[0] == 1)[0]
    viridx_a = np.where(mo_occ[0] == 0)[0]
    occidx_b = np.where(mo_occ[1] == 1)[0]
    viridx_b = np.where(mo_occ[1] == 0)[0]
    orbo_a = mo_coeff[0][:, occidx_a]
    orbv_a = mo_coeff[0][:, viridx_a]
    orbo_b = mo_coeff[1][:, occidx_b]
    orbv_b = mo_coeff[1][:, viridx_b]
    nocc_a = orbo_a.shape[1]
    nvir_a = orbv_a.shape[1]
    nocc_b = orbo_b.shape[1]
    nvir_b = orbv_b.shape[1]
    no = nocc_a - nocc_b
    dm = mf.make_rdm1()
    vhf = mf.get_veff(mf.mol, dm)
    h1e = mf.get_hcore()
    focka = h1e + vhf[0]
    fockb = h1e + vhf[1]
    fockA = mo_coeff[0].T @ focka @ mo_coeff[0]
    fockB = mo_coeff[1].T @ fockb @ mo_coeff[1]
    delta_ij = np.eye(nocc_a)
    delta_ab = np.eye(nvir_b)

    e_ia_b2a = (mo_energy[0][viridx_a, None] - mo_energy[1][occidx_b]).T
    e_ia_a2b = (mo_energy[1][viridx_b, None] - mo_energy[0][occidx_a]).T

    a_b2a = np.diag(e_ia_b2a.ravel()).reshape(nocc_b, nvir_a, nocc_b, nvir_a)
    a_a2b = np.diag(e_ia_a2b.ravel()).reshape(nocc_a, nvir_b, nocc_a, nvir_b)
    # a_b2a = (np.einsum('ij,ab->iajb',delta_ij[no:,no:],fockA[nocc_a:,nocc_a:])\
    #      -np.einsum('ij,ab->iajb',fockB[:nocc_b,:nocc_b],delta_ab[no:,no:])).reshape((nocc_b,nvir_a,nocc_b,nvir_a))
    # a_a2b = (np.einsum('ij,ab->iajb',delta_ij,fockB[nocc_b:,nocc_b:])\
    #      -np.einsum('ij,ab ->iajb',fockA[:nocc_a,:nocc_a],delta_ab)).reshape((nocc_a,nvir_b,nocc_a,nvir_b))
    # b_b2a = np.zeros((nocc_b,nvir_a,nocc_a,nvir_b))
    # b_a2b = np.zeros((nocc_a,nvir_b,nocc_b,nvir_a))
    a = (a_b2a, a_a2b)

    # b = (b_b2a, b_a2b)

    def add_hf_(a, b, hyb=1):
        # In spin flip TDA/ TDDFT, hartree potential is zero.
        # A : iabj ---> ijba; B : iajb ---> ibja
        eri_a_b2a = ao2mo.general(mol, [orbo_b, orbo_b, orbv_a, orbv_a], compact=False)
        eri_a_a2b = ao2mo.general(mol, [orbo_a, orbo_a, orbv_b, orbv_b], compact=False)
        # eri_b_b2a = ao2mo.general(mol, [orbo_b,orbv_b,orbo_a,orbv_a], compact=False)
        # eri_b_a2b = ao2mo.general(mol, [orbo_a,orbv_a,orbo_b,orbv_b], compact=False)

        eri_a_b2a = eri_a_b2a.reshape(nocc_b, nocc_b, nvir_a, nvir_a)
        eri_a_a2b = eri_a_a2b.reshape(nocc_a, nocc_a, nvir_b, nvir_b)
        # eri_b_b2a = eri_b_b2a.reshape(nocc_b,nvir_b,nocc_a,nvir_a)
        # eri_b_a2b = eri_b_a2b.reshape(nocc_a,nvir_a,nocc_b,nvir_b)

        a_b2a, a_a2b = a
        # b_b2a, b_a2b = b

        a_b2a -= np.einsum('ijba->iajb', eri_a_b2a) * hyb
        a_a2b -= np.einsum('ijba->iajb', eri_a_a2b) * hyb
        # b_b2a-= np.einsum('ibja->iajb', eri_b_b2a) * hyb
        # b_a2b-= np.einsum('ibja->iajb', eri_b_a2b) * hyb

    if isinstance(mf, scf.hf.KohnShamDFT):
        from pyscf.dft import xc_deriv
        from pyscf.dft import numint2c
        ni0 = mf._numint
        ni = numint2c.NumInt2C()
        ni.collinear = 'mcol'
        ni.collinear_samples = collinear_samples
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                            'deriviative is not available. Its contribution is '
                            'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        add_hf_(a, hyb)

        xctype = ni._xc_type(mf.xc)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * .8 - mem_now)

        # it should be optimized, which is the disadvantage of mc approach.
        fxc = cache_xc_kernel_sf_mc(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ, deriv=2, spin=1)[2]
        p0, p1 = 0, 0  # the two parameters are used for counts the batch of grids.

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni0.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                p0 = p1
                p1 += weight.shape[0]
                wfxc = fxc[0, 0][..., p0:p1] * weight

                rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                rho_ov_b2a = np.einsum('ri,ra->ria', rho_o_b, rho_v_a)
                rho_ov_a2b = np.einsum('ri,ra->ria', rho_o_a, rho_v_b)

                w_ov = np.einsum('ria,r->ria', rho_ov_b2a, wfxc * 2.0)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                a_b2a += iajb
                # iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                # b_a2b += iajb

                w_ov = np.einsum('ria,r->ria', rho_ov_a2b, wfxc * 2.0)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a2b, w_ov)
                a_a2b += iajb
                # iajb = lib.einsum('ria,rjb->iajb', rho_ov_b2a, w_ov)
                # b_b2a += iajb

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                p0 = p1
                p1 += weight.shape[0]
                wfxc = fxc[..., p0:p1] * weight

                rho_o_a = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_v_a = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_o_b = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_v_b = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_b2a = np.einsum('xri,ra->xria', rho_o_b, rho_v_a[0])
                rho_ov_a2b = np.einsum('xri,ra->xria', rho_o_a, rho_v_b[0])
                rho_ov_b2a[1:4] += np.einsum('ri,xra->xria', rho_o_b[0], rho_v_a[1:4])
                rho_ov_a2b[1:4] += np.einsum('ri,xra->xria', rho_o_a[0], rho_v_b[1:4])

                w_ov = np.einsum('xyr,xria->yria', wfxc * 2.0, rho_ov_b2a)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                a_b2a += iajb
                # iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                # b_b2a += iajb

                w_ov = np.einsum('xyr,xria->yria', wfxc * 2.0, rho_ov_a2b)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                a_a2b += iajb
                # iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                # b_a2b += iajb

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                p0 = p1
                p1 += weight.shape[0]
                wfxc = fxc[..., p0:p1] * weight

                rho_oa = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_ob = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_va = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_vb = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_b2a = np.einsum('xri,ra->xria', rho_ob, rho_va[0])
                rho_ov_a2b = np.einsum('xri,ra->xria', rho_oa, rho_vb[0])
                rho_ov_b2a[1:4] += np.einsum('ri,xra->xria', rho_ob[0], rho_va[1:4])
                rho_ov_a2b[1:4] += np.einsum('ri,xra->xria', rho_oa[0], rho_vb[1:4])
                tau_ov_b2a = np.einsum('xri,xra->ria', rho_ob[1:4], rho_va[1:4]) * .5
                tau_ov_a2b = np.einsum('xri,xra->ria', rho_oa[1:4], rho_vb[1:4]) * .5
                rho_ov_b2a = np.vstack([rho_ov_b2a, tau_ov_b2a[np.newaxis]])
                rho_ov_a2b = np.vstack([rho_ov_a2b, tau_ov_a2b[np.newaxis]])

                w_ov = np.einsum('xyr,xria->yria', wfxc * 2.0, rho_ov_b2a)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                a_b2a += iajb
                # iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                # b_b2a += iajb

                w_ov = np.einsum('xyr,xria->yria', wfxc * 2.0, rho_ov_a2b)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a2b)
                a_a2b += iajb
                # iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                # b_a2b += iajb
    else:
        add_hf_(a)

    return a


if __name__ == '__main__':
    mol = gto.M(
        atom="""O 0 0 2.07; O 0 0 0""",
        basis='6-31G',
        unit='B',
        # basis='sto3g',
        charge=0,
        spin=2,
        verbose=4,
        # cart=True,
        # symmetry='C2v'
    )
    mf = dft.UKS(mol)
    # mf = scf.RHF(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    sf_tda = SF_TDA(mf, isf=1, method=1)
    e, v = sf_tda.kernel(nstates=10)
    print(e)

    sf_tda = SF_TDA(mf, isf=-1, method=1)
    e, v = sf_tda.kernel(nstates=10)
    print(e)