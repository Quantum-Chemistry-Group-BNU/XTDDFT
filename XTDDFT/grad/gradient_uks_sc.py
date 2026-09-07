#!/usr/bin/env python
import numpy as np
from pyscf import lib, __config__
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.grad import uhf as uhf_grad
from pyscf.grad import tdrks as tdrks_grad


#
# Given Y = 0, TDHF gradients (XAX+XBY+YBX+YAY)^1 turn to TDA gradients (XAX)^1
#
def grad_elec(td, atmlst=None, max_memory=2000, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDDFT nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    '''
    log = logger.new_logger(td, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mf = td.base._scf
    mol = td.mol
    order = td.base.order
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]
    occidx_a = np.where(mo_occ[0] == 1)[0]
    viridx_a = np.where(mo_occ[0] == 0)[0]
    occidx_b = np.where(mo_occ[1] == 1)[0]
    viridx_b = np.where(mo_occ[1] == 0)[0]
    nc = len(occidx_b)
    nv = len(viridx_a)
    no = len(occidx_a) - len(occidx_b)
    orbv_a = mo_coeff[0][:, (nc+no):]
    orbo_a = mo_coeff[0][:, :(nc+no)]
    orbv_b = mo_coeff[1][:, nc:]
    orbo_b = mo_coeff[1][:, :nc]
    v = td.v[:, td.state-1]
    v_cva = v[:nc*nv].reshape(nc, nv)
    v_ova = v[nc*nv:(nc+no)*nv].reshape(no, nv)
    v_cob = v[(nc+no)*nv:(nc+no)*nv+nc*no].reshape(nc, no)
    v_cvb = v[(nc+no)*nv+nc*no:].reshape(nc, nv)
    # convert CV(aa)|OV(aa)|CO(bb)|CV(bb) to pyscf order
    va = np.vstack((v_cva, v_ova)).T
    vb = np.hstack((v_cob, v_cvb)).T

    dvva = lib.einsum('ai, bi->ab', va, va)  # T_{ab}^{\alpha}
    dvvb = lib.einsum('ai, bi->ab', vb, vb)
    dooa = -lib.einsum('ai, aj->ij', va, va)  # T_{ij}^{\alpha}
    doob = -lib.einsum('ai, aj->ij', vb, vb)
    dmza = orbv_a @ dvva @ orbv_a.T + orbo_a @ dooa @ orbo_a.T  # T_{\mu\nu}^{\alpha}
    dmzb = orbv_b @ dvvb @ orbv_b.T + orbo_b @ doob @ orbo_b.T  # T_{\mu\nu}^{\beta}
    dmxa = orbv_a @ va @ orbo_a.T  # X_{\mu\nu}^{\alpha}
    dmxb = orbv_b @ vb @ orbo_b.T

    # 2. functional derivative, include derivative respect to mo_coeff and coordinate
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # f1vo: f^{xc}[X], f1oo: f^{xc}[T], vxc1: v^{xc}[\rho], k1ao: g^{xc}[X,X] 
    # and their derivative
    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(
        td, mf.xc, (dmxa, dmxb), (dmza, dmzb), True, True, max_memory
    )

    # 3. construct Q matrix
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    if with_k:
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
        wvoa += lib.einsum('ba,bi->ai', veff0moa[(nc+no):, (nc+no):], va) * 2
        wvoa -= lib.einsum('ij,aj->ai', veff0moa[:(nc+no), :(nc+no)], va) * 2
        wvob += lib.einsum('ba,bi->ai', veff0mob[nc:, nc:], vb) * 2
        wvob -= lib.einsum('ij,aj->ai', veff0mob[:nc, :nc], vb) * 2
    else:
        dm = (dmza, dmzb, dmxa, dmxb,)
        vj= mf.get_j(mol, dm, hermi=0)  # g_{\mu\nu}^{\sigma}[T]
        veff0doo = vj[0] + vj[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = (orbv_a.T @ veff0doo[0] @ orbo_a) * 2
        wvob = (orbv_b.T @ veff0doo[1] @ orbo_b) * 2
        veff0moa = mo_coeff[0].T @ (vj[2] + vj[3] + f1vo[0, 0]) @ mo_coeff[0]
        veff0mob = mo_coeff[1].T @ (vj[2] + vj[3] + f1vo[1, 0]) @ mo_coeff[1]
        wvoa += lib.einsum('ba,bi->ai', veff0moa[(nc+no):, (nc+no):], va) * 2
        wvoa -= lib.einsum('ij,aj->ai', veff0moa[:(nc+no), :(nc+no)], va) * 2
        wvob += lib.einsum('ba,bi->ai', veff0mob[nc:, nc:], vb) * 2
        wvob -= lib.einsum('ij,aj->ai', veff0mob[:nc, :nc], vb) * 2

    # 4. constuct G[Z^S] and solve Z-vector equation
    vresp = mf.gen_response(hermi=1)
    def fvind(x):
        dm1 = np.empty((2,nao,nao))
        xa = x[0, :(nc+no)*nv].reshape(nv, nc+no)
        xb = x[0, (nc+no)*nv:].reshape(no+nv, nc)
        dma = orbv_a @ xa @ orbo_a.T
        dmb = orbv_b @ xb @ orbo_b.T
        dm1[0] = (dma + dma.T) / 2
        dm1[1] = (dmb + dmb.T) / 2
        v1 = vresp(dm1)  # G_{\mu\nu}[Z^{S}]
        v1a = orbv_a.T @ v1[0]*2 @ orbo_a
        v1b = orbv_b.T @ v1[1]*2 @ orbo_b
        return np.hstack((v1a.ravel(), v1b.ravel()))
    # Z instead of 1/2 Z
    z1a, z1b = ucphf.solve(
        fvind, mo_energy, mo_occ, (wvoa, wvob),
        max_cycle=td.cphf_max_cycle, tol=td.cphf_conv_tol
    )[0]
    time1 = log.timer('Z-vector using UCPHF solver', *time0)
    z1ao = np.empty((2, nao, nao))
    z1ao[0] = orbv_a @ z1a @ orbo_a.T
    z1ao[1] = orbv_b @ z1b @ orbo_b.T
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)) * .5)  # G_{\mu\nu}[Z^{S}]

    # 5.1. construct W matrix, without Fock part
    im0a = np.zeros((nao, nao))
    im0b = np.zeros((nao, nao))
    im0a[:(nc+no),:(nc+no)] = orbo_a.T @ (veff0doo[0]+veff[0]) @ orbo_a
    im0a[:(nc+no),:(nc+no)] += lib.einsum('ak,ai->ki', veff0moa[(nc+no):,:(nc+no)], va)
    im0a[(nc+no):,(nc+no):] = lib.einsum('ci,ai->ac', veff0moa[(nc+no):,:(nc+no)], va)
    im0a[(nc+no):,:(nc+no)] = lib.einsum('ki,ai->ak', veff0moa[:(nc+no),:(nc+no)], va) * 2
    im0b[:nc,:nc] = orbo_b.T @ (veff0doo[1]+veff[1]) @ orbo_b
    im0b[:nc,:nc] += lib.einsum('ak,ai->ki', veff0mob[nc:,:nc], vb)
    im0b[nc:,nc:] = lib.einsum('ci,ai->ac', veff0mob[nc:,:nc], vb)
    im0b[nc:,:nc] = lib.einsum('ki,ai->ak', veff0mob[:nc,:nc], vb) * 2

    # 5.2 construct W matrix, Fock part
    zeta_a = (mo_energy[0][:, None] + mo_energy[0]) * .5
    zeta_b = (mo_energy[1][:, None] + mo_energy[1]) * .5
    zeta_a[(nc+no):, :(nc+no)] = mo_energy[0][:(nc+no)]
    zeta_b[nc:, :nc] = mo_energy[1][:nc]
    zeta_a[:(nc+no), (nc+no):] = mo_energy[0][(nc+no):]
    zeta_b[:nc, nc:] = mo_energy[1][nc:]
    dm1a = np.zeros((nao, nao))
    dm1b = np.zeros((nao, nao))
    dm1a[:(nc+no), :(nc+no)] = dooa
    dm1b[:nc, :nc] = doob
    dm1a[(nc+no):, (nc+no):] = dvva
    dm1b[nc:, nc:] = dvvb
    dm1a[(nc+no):, :(nc+no)] = z1a
    dm1b[nc:, :nc] = z1b
    dm1a[:(nc+no), :(nc+no)] += np.eye((nc+no)) # for ground state
    dm1b[:nc, :nc] += np.eye(nc)
    im0a = mo_coeff[0] @ (im0a+zeta_a*dm1a) @ mo_coeff[0].T
    im0b = mo_coeff[1] @ (im0b+zeta_b*dm1b) @ mo_coeff[1].T
    im0 = im0a + im0b

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dmz1dooa = z1ao[0] + dmza  # Z_{\mu\nu}^{\alpha} + T_{\mu\nu}^{\alpha}
    dmz1doob = z1ao[1] + dmzb
    oo0a = orbo_a @ orbo_a.T
    oo0b = orbo_b @ orbo_b.T
    as_dm1 = oo0a + oo0b + dmz1dooa + dmz1doob

    if with_k:
        dm = (oo0a, dmz1dooa+dmz1dooa.T, dmxa, dmxa.T,
              oo0b, dmz1doob+dmz1doob.T, dmxb, dmxb.T)
        vj, vk = td.get_jk(mol, dm)
        vj = vj.reshape(2,4,3,nao,nao)
        vk = vk.reshape(2,4,3,nao,nao) * hyb
        # if omega != 0:
        #     vk += td.get_k(mol, dm, omega=omega).reshape(2,4,3,nao,nao) * (alpha-hyb)
        veff1 = vj[0] + vj[1] - vk
    else:
        dm = (oo0a, dmz1dooa+dmz1dooa.T, dmxa, dmxa.T,
              oo0b, dmz1doob+dmz1doob.T, dmxb, dmxb.T)
        vj = td.get_j(mol, dm).reshape(2,4,3,nao,nao)
        veff1 = vj[0] + vj[1]

    fxcz1 = _contract_xc_kernel(
        td, mf.xc, z1ao, None, False, False, max_memory
    )[0]

    veff1[:,0] += vxc1[:,1:]
    veff1[:,1] +=(f1oo[:,1:] + fxcz1[:,1:] + k1ao[:,1:]) * 2 # *2 for dmz1doo+dmz1oo.T
    veff1[:,2] += f1vo[:,1:]
    veff1[:,3] += f1vo[:,1:]
    veff1a, veff1b = veff1
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = np.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        de[k] = lib.einsum('xpq,pq->x', h1ao, as_dm1)

        de[k] += lib.einsum('xpq,pq->x', veff1a[0,:,p0:p1], oo0a[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', veff1b[0,:,p0:p1], oo0b[p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1a[0,:,p0:p1], oo0a[:,p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1b[0,:,p0:p1], oo0b[:,p0:p1])

        de[k] += lib.einsum('xpq,pq->x', veff1a[0,:,p0:p1], dmz1dooa[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', veff1b[0,:,p0:p1], dmz1doob[p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1a[0,:,p0:p1], dmz1dooa[:,p0:p1])
        de[k] += lib.einsum('xpq,qp->x', veff1b[0,:,p0:p1], dmz1doob[:,p0:p1])

        de[k] -= lib.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= lib.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        de[k] += lib.einsum('xij,ij->x', veff1a[1,:,p0:p1], oo0a[p0:p1])
        de[k] += lib.einsum('xij,ij->x', veff1b[1,:,p0:p1], oo0b[p0:p1])
        de[k] += lib.einsum('xij,ij->x', veff1a[2,:,p0:p1], dmxa[p0:p1,:])*2
        de[k] += lib.einsum('xij,ij->x', veff1b[2,:,p0:p1], dmxb[p0:p1,:])*2
        de[k] += lib.einsum('xji,ij->x', veff1a[3,:,p0:p1], dmxa[:,p0:p1])*2
        de[k] += lib.einsum('xji,ij->x', veff1b[3,:,p0:p1], dmxb[:,p0:p1])*2
        # de[k] += td.extra_force(ia, locals())  # extension, here always zero

    log.timer('TDUKS nuclear gradients', *time0)
    return de


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

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
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


class SC_gradient(uhf_grad.Gradients):
    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20) + 20
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)

    def __init__(self, td, state=1):
        self.base = td
        self.base._scf = td.mf
        self.mol = td.mol
        self.v = td.v
        self.state = state
        self.de = None  # gradient of molecule
        self.atmlst = None  # which atom will be calculate gradient
        
    def grad_elec(self, atmlst=None):
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
        logger.note(self,
            '--------- %s gradients for state %d ----------', 
            self.base.__class__.__name__,
            self.state
        )
        self._write(self.mol, self.de, self.atmlst)
        logger.note(self, '----------------------------------------------')


