#!/usr/bin/env python

from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import uhf as uhf_grad
from pyscf.scf import ucphf
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

    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    (xa, xb), (ya, yb) = x_y
    xpya = (xa+ya).reshape(nocca,nvira).T
    xpyb = (xb+yb).reshape(noccb,nvirb).T
    xmya = (xa-ya).reshape(nocca,nvira).T
    xmyb = (xb-yb).reshape(noccb,nvirb).T

    dvva = (numpy.einsum('ai,bi->ab', xpya, xpya) + numpy.einsum('ai,bi->ab', xmya, xmya)) /2
    dvvb = (numpy.einsum('ai,bi->ab', xpyb, xpyb) + numpy.einsum('ai,bi->ab', xmyb, xmyb)) /2
    dooa = -(numpy.einsum('ai,aj->ij', xpya, xpya) + numpy.einsum('ai,aj->ij', xmya, xmya)) /2
    doob = -(numpy.einsum('ai,aj->ij', xpyb, xpyb) + numpy.einsum('ai,aj->ij', xmyb, xmyb)) /2
    dmxpya = reduce(numpy.dot, (orbva, xpya, orboa.T))
    dmxpyb = reduce(numpy.dot, (orbvb, xpyb, orbob.T))
    dmxmya = reduce(numpy.dot, (orbva, xmya, orboa.T))
    dmxmyb = reduce(numpy.dot, (orbvb, xmyb, orbob.T))
    dmzooa = reduce(numpy.dot, (orboa, dooa, orboa.T))
    dmzoob = reduce(numpy.dot, (orbob, doob, orbob.T))
    dmzooa+= reduce(numpy.dot, (orbva, dvva, orbva.T))
    dmzoob+= reduce(numpy.dot, (orbvb, dvvb, orbvb.T))
    
    dmxpyaS = (dmxpya+dmxpya.T) / 2
    dmxpybS = (dmxpyb+dmxpyb.T) / 2
    dmxmyaA = (dmxmya-dmxmya.T) / 2
    dmxmybA = (dmxmyb-dmxmyb.T) / 2

    vj, vk = mf.get_jk(mol, (dmzooa, dmxpyaS, dmxmyaA,
                             dmzoob, dmxpybS, dmxmybA), hermi=0)
    vj = vj.reshape(2,3,nao,nao)
    vk = vk.reshape(2,3,nao,nao)
    veff0doo = vj[0,0]+vj[1,0] - vk[:,0]
    wvoa = reduce(numpy.dot, (orbva.T, veff0doo[0], orboa)) * 2
    wvob = reduce(numpy.dot, (orbvb.T, veff0doo[1], orbob)) * 2
    veff = vj[0,1]+vj[1,1] - vk[:,1]
    veff0mopa = reduce(numpy.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
    veff0mopb = reduce(numpy.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
    wvoa -= numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
    wvob -= numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
    wvoa += numpy.einsum('ac,ai->ci', veff0mopa[nocca:,nocca:], xpya) * 2
    wvob += numpy.einsum('ac,ai->ci', veff0mopb[noccb:,noccb:], xpyb) * 2
    veff = -vk[:,2]
    veff0moma = reduce(numpy.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
    veff0momb = reduce(numpy.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
    wvoa -= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya) * 2
    wvob -= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb) * 2
    wvoa += numpy.einsum('ac,ai->ci', veff0moma[nocca:,nocca:], xmya) * 2
    wvob += numpy.einsum('ac,ai->ci', veff0momb[noccb:,noccb:], xmyb) * 2

       
    vresp = mf.gen_response(hermi=1)    
    def fvind(x):
        dm1 = numpy.empty((2,nao,nao))
        xa = x[0,:nvira*nocca].reshape(nvira,nocca)
        xb = x[0,nvira*nocca:].reshape(nvirb,noccb)
        dma = reduce(numpy.dot, (orbva, xa, orboa.T))
        dmb = reduce(numpy.dot, (orbvb, xb, orbob.T))
        dm1[0] = (dma + dma.T)/2
        dm1[1] = (dmb + dmb.T)/2
        v1 = vresp(dm1) * 2
        v1a = reduce(numpy.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(numpy.dot, (orbvb.T, v1[1], orbob))
        return numpy.hstack((v1a.ravel(), v1b.ravel()))
    z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa,wvob),
                           max_cycle=td_grad.cphf_max_cycle,
                           tol=td_grad.cphf_conv_tol)[0]
    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    z1ao = numpy.empty((2,nao,nao))
    z1ao[0] = reduce(numpy.dot, (orbva, z1a, orboa.T))
    z1ao[1] = reduce(numpy.dot, (orbvb, z1b, orbob.T))
    veff = vresp((z1ao+z1ao.transpose(0,2,1)) * .5)

    im0a = numpy.zeros((nmoa,nmoa))
    im0b = numpy.zeros((nmob,nmob))
    im0a[:nocca,:nocca] = reduce(numpy.dot, (orboa.T, veff0doo[0]+veff[0], orboa))
    im0b[:noccb,:noccb] = reduce(numpy.dot, (orbob.T, veff0doo[1]+veff[1], orbob))
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', veff0mopa[nocca:,:nocca], xpya)
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', veff0mopb[noccb:,:noccb], xpyb)
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', veff0moma[nocca:,:nocca], xmya)
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', veff0momb[noccb:,:noccb], xmyb)
    im0a[nocca:,nocca:] = numpy.einsum('ci,ai->ac', veff0mopa[nocca:,:nocca], xpya)
    im0b[noccb:,noccb:] = numpy.einsum('ci,ai->ac', veff0mopb[noccb:,:noccb], xpyb)
    im0a[nocca:,nocca:]+= numpy.einsum('ci,ai->ac', veff0moma[nocca:,:nocca], xmya)
    im0b[noccb:,noccb:]+= numpy.einsum('ci,ai->ac', veff0momb[noccb:,:noccb], xmyb)
    im0a[nocca:,:nocca] = numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
    im0b[noccb:,:noccb] = numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
    im0a[nocca:,:nocca]+= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya) * 2
    im0b[noccb:,:noccb]+= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb) * 2

    zeta_a = (mo_energy[0][:,None] + mo_energy[0]) * .5
    zeta_b = (mo_energy[1][:,None] + mo_energy[1]) * .5
    zeta_a[nocca:,:nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:,:noccb] = mo_energy[1][:noccb]
    dm1a = numpy.zeros((nmoa,nmoa))
    dm1b = numpy.zeros((nmob,nmob))
    dm1a[:nocca,:nocca] = dooa
    dm1b[:noccb,:noccb] = doob
    dm1a[nocca:,nocca:] = dvva
    dm1b[noccb:,noccb:] = dvvb
    dm1a[nocca:,:nocca] = z1a
    dm1b[noccb:,:noccb] = z1b
    dm1a[:nocca,:nocca] += numpy.eye(nocca)
    dm1b[:noccb,:noccb] += numpy.eye(noccb)
    im0a = reduce(numpy.dot, (mo_coeff[0], im0a+zeta_a*dm1a, mo_coeff[0].T))
    im0b = reduce(numpy.dot, (mo_coeff[1], im0b+zeta_b*dm1b, mo_coeff[1].T))
    im0 = im0a + im0b


        
    dmz1dooa = (z1ao[0] + z1ao[0].T)/2 + dmzooa
    dmz1doob = (z1ao[1] + z1ao[1].T)/2 + dmzoob

    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    oo0a = reduce(numpy.dot, (orboa, orboa.T))
    oo0b = reduce(numpy.dot, (orbob, orbob.T))
    oo0T = oo0a + oo0b
    vj, vk = td_grad.get_jk(mol, (oo0a, dmz1dooa, dmxpyaS, dmxmyaA,
                                  oo0b, dmz1doob, dmxpybS, dmxmybA))       
    vj = vj.reshape(2,4,3,nao,nao)
    vk = vk.reshape(2,4,3,nao,nao)
    vhf1a,vhf1b = vj[0] + vj[1] - vk
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h1ao = hcore_deriv(ia)
        de[k] = numpy.einsum('xpq,pq->x', h1ao, oo0T)  
        de[k] += numpy.einsum('xpq,pq->x', vhf1a[0,:,p0:p1],  oo0a[p0:p1])
        de[k] += numpy.einsum('xqp,pq->x', vhf1a[0,:,p0:p1],  oo0a[:,p0:p1])
        de[k] += numpy.einsum('xpq,pq->x', vhf1b[0,:,p0:p1],  oo0b[p0:p1])
        de[k] += numpy.einsum('xqp,pq->x', vhf1b[0,:,p0:p1],  oo0b[:,p0:p1])
        # 激发态响应部分
        de[k] += numpy.einsum('xpq,pq->x', h1ao, dmz1dooa)
        de[k] += numpy.einsum('xpq,pq->x', h1ao, dmz1doob)
        de[k] += numpy.einsum('xpq,pq->x', vhf1a[0,:,p0:p1], dmz1dooa[p0:p1])
        de[k] += numpy.einsum('xqp,pq->x', vhf1a[0,:,p0:p1], dmz1dooa[:,p0:p1])
        de[k] += numpy.einsum('xpq,pq->x', vhf1b[0,:,p0:p1], dmz1doob[p0:p1])
        de[k] += numpy.einsum('xqp,pq->x', vhf1b[0,:,p0:p1], dmz1doob[:,p0:p1])
        de[k] += numpy.einsum('xij,ij->x', vhf1a[1,:,p0:p1], oo0a[p0:p1])
        de[k] += numpy.einsum('xji,ij->x', vhf1a[1,:,p0:p1], oo0a[:,p0:p1])
        de[k] += numpy.einsum('xij,ij->x', vhf1b[1,:,p0:p1], oo0b[p0:p1])
        de[k] += numpy.einsum('xji,ij->x', vhf1b[1,:,p0:p1], oo0b[:,p0:p1])
        de[k] += numpy.einsum('xij,ij->x', vhf1a[2,:,p0:p1], (dmxpyaS)[p0:p1]) * 2
        de[k] += numpy.einsum('xji,ij->x', vhf1a[2,:,p0:p1], (dmxpyaS)[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1b[2,:,p0:p1], (dmxpybS)[p0:p1]) * 2
        de[k] += numpy.einsum('xji,ij->x', vhf1b[2,:,p0:p1], (dmxpybS)[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1a[3,:,p0:p1], (dmxmyaA)[p0:p1]) * 2
        de[k] -= numpy.einsum('xji,ij->x', vhf1a[3,:,p0:p1], (dmxmyaA)[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1b[3,:,p0:p1], (dmxmybA)[p0:p1]) * 2
        de[k] -= numpy.einsum('xji,ij->x', vhf1b[3,:,p0:p1], (dmxmybA)[:,p0:p1]) * 2
       
        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])
        

        
        # 额外项（如QM/MM等）
        de[k] += td_grad.extra_force(ia, locals())

    log.timer('TDHF nuclear gradients', *time0)
    return de




class MyGradients(uhf_grad.Gradients):

    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20)
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)
    
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

    to_gpu = lib.to_gpu

from pyscf.tdscf.uhf import TDA

class MyTDHF(TDA):
    def __init__(self, mf):
        super().__init__(mf)
        self.frozen = None


    def nuc_grad_method(self):
        return MyGradients(self)
    



from pyscf import scf
mol = gto.M(atom=''' H 0. 0. 0.; 
            F 0. 0. 0.97;
            ''',charge = 0 , spin=0 ,basis = '6-31g')
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.max_cycle = 1000
mf.kernel()


td = MyTDHF(mf)  
td.conv_tol = 1e-10
td.max_cycle = 200
td.kernel()

my_grad = td.nuc_grad_method().kernel()

