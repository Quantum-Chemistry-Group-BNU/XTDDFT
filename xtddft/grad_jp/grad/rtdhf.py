#!/usr/bin/env python

from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.scf import cphf
from pyscf import __config__
from pyscf import tdscf

def grad_elec(td_grad, x_y, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO, with_nlc=True):
    '''
    Electronic part of TDA, TDHF nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.
        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
        with_nlc : bool
            Whether to include Nuclear-Lagrangian contributions (orbital response terms).
            If False, only Hellmann-Feynman terms are computed.
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    assert td_grad.base.frozen is None

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    x, y = x_y
    xpy = (x+y).reshape(nocc,nvir).T
    xmy = (x-y).reshape(nocc,nvir).T
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    dvv = (numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)) /2
    doo = -(numpy.einsum('ai,aj->ij', xpy, xpy) + numpy.einsum('ai,aj->ij', xmy, xmy)) /2
    dmxpy = reduce(numpy.dot, (orbv, xpy, orbo.T))
    dmxmy = reduce(numpy.dot, (orbv, xmy, orbo.T))
    
    
    dmzoo = reduce(numpy.dot, (orbo, doo, orbo.T))
    dmzoo+= reduce(numpy.dot, (orbv, dvv, orbv.T))
    dmxpyS = (dmxpy+dmxpy.T) / 2
    dmxmyA = (dmxmy-dmxmy.T) / 2

    vj, vk = mf.get_jk(mol, (dmzoo, dmxpyS, dmxmyA), hermi=0)
    veff0doo = vj[0] * 2 - vk[0]
    wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2
        
    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
    veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
    wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
        
    veff = -vk[2]
    veff0mom = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 2
    wvo += numpy.einsum('ac,ai->ci', veff0mom[nocc:,nocc:], xmy) * 2

       
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind(x):  # For singlet, closed shell ground state
        dm = reduce(numpy.dot, (orbv, 2 * x.reshape(nvir,nocc), orbo.T))
        v1ao = vresp((dm+dm.T)/2) * 2
        return reduce(numpy.dot, (orbv.T, v1ao, orbo)).ravel()
        
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.cphf_max_cycle,
                    tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao = reduce(numpy.dot, (orbv, z1, orbo.T))
    veff = vresp(z1ao+z1ao.T)

    im0 = numpy.zeros((nmo,nmo))
    im0[:nocc,:nocc] = reduce(numpy.dot, (orbo.T, veff0doo+veff, orbo)) * 2
    im0[:nocc,:nocc]+= numpy.einsum('ak,ai->ki', veff0mop[nocc:,:nocc], xpy) * 2
    im0[:nocc,:nocc]+= numpy.einsum('ak,ai->ki', veff0mom[nocc:,:nocc], xmy) * 2
    im0[nocc:,nocc:] = numpy.einsum('ci,ai->ac', veff0mop[nocc:,:nocc], xpy) * 2
    im0[nocc:,nocc:]+= numpy.einsum('ci,ai->ac', veff0mom[nocc:,:nocc], xmy) * 2
    im0[nocc:,:nocc] = numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 4
    im0[nocc:,:nocc]+= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 4
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) / 2
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo * 2
    dm1[nocc:,nocc:] = dvv * 2
    dm1[nocc:,:nocc] = z1 * 2
    dm1[:nocc,:nocc] += numpy.eye(nocc) * 2 # for ground state
    im0 = reduce(numpy.dot, (mo_coeff, im0+zeta*dm1, mo_coeff.T))

        
    dmz1doo = (z1ao + z1ao.T)/2 + dmzoo

    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    oo0 = reduce(numpy.dot, (orbo, orbo.T))

    vj, vk = td_grad.get_jk(mol, (oo0, dmz1doo, dmxpyS, dmxmyA))       
    vj = vj.reshape(-1,3,nao,nao)
    vk = vk.reshape(-1,3,nao,nao)
    vhf1 = -vk
    if singlet:
        vhf1 += vj * 2
    else:
        vhf1[:2] += vj[:2]*2
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h1ao = hcore_deriv(ia)
        de[k] = numpy.einsum('xpq,pq->x', h1ao, 2 * oo0)  # 基态HF部分
        
        # 基态电子排斥部分
        de[k] += numpy.einsum('xpq,pq->x', vhf1[0,:,p0:p1],  oo0[p0:p1]) * 2
        de[k] += numpy.einsum('xqp,pq->x', vhf1[0,:,p0:p1],  oo0[:,p0:p1]) * 2
        # 激发态响应部分
        de[k] += numpy.einsum('xpq,pq->x', h1ao, dmz1doo) * 2
        de[k] += numpy.einsum('xpq,pq->x', vhf1[0,:,p0:p1], dmz1doo[p0:p1]) * 2
        de[k] += numpy.einsum('xqp,pq->x', vhf1[0,:,p0:p1], dmz1doo[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1[1,:,p0:p1], oo0[p0:p1]) * 2
        de[k] += numpy.einsum('xji,ij->x', vhf1[1,:,p0:p1], oo0[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1[2,:,p0:p1], (dmxpyS)[p0:p1]) * 4
        de[k] += numpy.einsum('xji,ij->x', vhf1[2,:,p0:p1], (dmxpyS)[:,p0:p1]) * 4
        de[k] += numpy.einsum('xij,ij->x', vhf1[3,:,p0:p1], (dmxmyA)[p0:p1]) * 4
        de[k] -= numpy.einsum('xji,ij->x', vhf1[3,:,p0:p1], (dmxmyA)[:,p0:p1]) * 4
        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        
        # 额外项（如QM/MM等）
        de[k] += td_grad.extra_force(ia, locals())

    log.timer('TDHF nuclear gradients', *time0)
    return de


def as_scanner(td_grad, state=1):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns energy and first order nuclear derivatives.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    nuc-grad object and SCF object (DIIS, conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

    >>> from pyscf import gto, scf, tdscf, grad
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
    >>> td_grad_scanner = scf.RHF(mol).apply(tdscf.TDA).nuc_grad_method().as_scanner()
    >>> e_tot, grad = td_grad_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
    >>> e_tot, grad = td_grad_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    from pyscf import gto
    if isinstance(td_grad, lib.GradScanner):
        return td_grad

    if state == 0:
        return td_grad.base._scf.nuc_grad_method().as_scanner()

    logger.info(td_grad, 'Create scanner for %s', td_grad.__class__)
    name = td_grad.__class__.__name__ + TDSCF_GradScanner.__name_mixin__
    return lib.set_class(TDSCF_GradScanner(td_grad, state),
                         (TDSCF_GradScanner, td_grad.__class__), name)

class TDSCF_GradScanner(lib.GradScanner):
    _keys = {'e_tot'}

    def __init__(self, g, state):
        lib.GradScanner.__init__(self, g)
        if state is not None:
            self.state = state

    def __call__(self, mol_or_geom, state=None, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)

        if state is None:
            state = self.state
        else:
            self.state = state

        td_scanner = self.base
        td_scanner(mol)
# TODO: Check root flip.  Maybe avoid the initial guess in TDHF otherwise
# large error may be found in the excited states amplitudes
        de = self.kernel(state=state, **kwargs)
        e_tot = self.e_tot[state-1]
        return e_tot, de

    @property
    def converged(self):
        td_scanner = self.base
        return all((td_scanner._scf.converged,
                    td_scanner.converged[self.state]))


class Gradients(rhf_grad.GradientsBase):

    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20)
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)
    
    # 添加 with_nlc 控制
    with_nlc = getattr(__config__, 'grad_tdrhf_Gradients_with_nlc', True)

    _keys = {
        'cphf_max_cycle', 'cphf_conv_tol', 'with_nlc',
        'mol', 'base', 'chkfile', 'state', 'atmlst', 'de',
    }
    

    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.chkfile = td.chkfile
        self.max_memory = td.max_memory
        self.state = 1  # of which the gradients to be computed.
        self.atmlst = None
        self.de = None
        self.with_nlc = True  # 默认包含NLC项

        if getattr(td._scf, 'with_df', None):
            raise NotImplementedError('Nuclear Gradients for DF-TDDFT')
    @lib.with_doc(grad_elec.__doc__)
    
    def grad_elec(self, xy, singlet, atmlst=None, with_nlc=None):
        if with_nlc is None:
            with_nlc = self.with_nlc
        return grad_elec(self, xy, singlet, atmlst, self.max_memory, 
                        self.verbose)

    def kernel(self, xy=None, state=None, singlet=None, atmlst=None, with_nlc=None):
        '''
        Args:
        state : int
            Excited state ID.  state = 1 means the first excited state.
        with_nlc : bool or None
            Whether to include Nuclear-Lagrangian contributions.
            If None, use self.with_nlc
        '''
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state

            if state == 0:
                logger.warn(self, 'state=0 found in the input. '
                        'Gradients of ground state is computed.')
                return self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)

            if self.base.xy is None:
                self.base.run()
            xy = self.base.xy[state-1]

        if singlet is None: 
            singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        
        if with_nlc is None:
            with_nlc = self.with_nlc
        else:
            self.with_nlc = with_nlc

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(xy, singlet, atmlst, with_nlc=with_nlc)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de
    
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** LR %s gradients for %s ********',
             self.base.__class__, self.base._scf.__class__)
        log.info('cphf_conv_tol = %g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('with_nlc = %s', self.with_nlc)
        log.info('chkfile = %s', self.chkfile)
        log.info('State ID = %d', self.state)
        log.info('max_memory %d MB (current use %d MB)',
             self.max_memory, lib.current_memory()[0])
        log.info('\n')
        return self

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------- %s gradients for state %d ----------',
                        self.base.__class__.__name__, self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = as_scanner

    to_gpu = lib.to_gpu

from pyscf.tdscf.rhf import TDHF

class TD_HF(TDHF):
    def __init__(self, mf):
        super().__init__(mf)
        self.frozen = None


    def nuc_grad_method(self):
        return Gradients(self)






from pyscf import gto, scf
mol = gto.M(atom=''' O 0 0 0;
                    H 1 0 -0.5;
                    H -1 0 -0,5
            ''',basis = 'cc-pvdz')
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.max_cycle = 200
energy = mf.kernel()
td = TD_HF(mf)
td.conv_tol = 1e-12
td.tol = 1e-10 
td.max_cycle = 1000
td.kernel()
my_grad = td.nuc_grad_method().kernel()
