from functools import reduce
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from TDA.SF_TDA import SF_TDA_down
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.grad import uhf as uhf_grad
from pyscf import __config__

def grad_elec(td_grad, singlet=True, atmlst = None,
              max_memory=2000, verbose=logger.INFO,with_nlc=None, state_idx=1):
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
    DSOLVE_LINDEP = getattr(__config__, 'lib_linalg_helper_dsolve_lindep', 1e-13)

    mol = td_grad.mol
    mf = td_grad.base._scf
    sftda = td_grad.base
    mo_occ = sftda.mo_occ
    mo_coeff = sftda.mo_coeff
    occidxa = numpy.where(mo_occ[0] == 1)[0]
    viridxa = numpy.where(mo_occ[0] == 0)[0]
    occidxb = numpy.where(mo_occ[1] == 1)[0]
    viridxb = numpy.where(mo_occ[1] == 0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]  #C_oa
    orbva = mo_coeff[0][:,viridxa]
    orbob = mo_coeff[1][:,occidxb]
    orbvb = mo_coeff[1][:,viridxb]  #C_vb
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb
    nc = sftda.nc  #double occupied number
    no = sftda.no  #single occupied number
    nv = sftda.nv  #virtual occupied number
    

    value = sftda.v[:, state_idx]
    
    x_cv_ab = value[:nc*nv].reshape(nc, nv)
    x_co_ab = value[nc*nv:nc*nv+nc*no].reshape(nc, no)
    x_ov_ab = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no, nv)
    x_oo_ab = value[nc*nv+nc*no+no*nv:].reshape(no, no)

    x_ab = numpy.zeros((nocca,nvirb))
    x_ab[:nc,:no]+= x_co_ab
    x_ab[:nc,no:]+= x_cv_ab
    x_ab[nc:,:no]+= x_oo_ab
    x_ab[nc:,no:]+= x_ov_ab
    x_ab = x_ab.T



    fock_ao = mf.get_fock()  #fock(ao)
    focka = fock_ao[0]  #focka(ao)
    fockb = fock_ao[1]  #fockb(ao)
    
    fockamo = reduce(numpy.dot, (mo_coeff[0].T, focka, mo_coeff[0]))  #focka(mo)
    fockbmo = reduce(numpy.dot, (mo_coeff[1].T, fockb, mo_coeff[1]))  #fockb(mo)
  

    #T_a
    ta = -numpy.einsum('pq,ps->qs', x_ab, x_ab)
    #T_b
    tb = numpy.einsum('pq,rq->pr', x_ab, x_ab)
    #T(ao)
    dmta = numpy.einsum('kq,qs,sl->kl',orboa, ta, orboa.T)
    dmtb = numpy.einsum('kp,pr,rl->kl',orbvb, tb, orbvb.T)
    #X(ao)
    dmxab = numpy.einsum('ka,ai,il->kl',orbvb, x_ab, orboa.T)  
    dmSx = (dmxab + dmxab.T) / 2
    dmAx = (dmxab - dmxab.T) / 2   

                            
                            


 

    
    vj, vk = mf.get_jk(mol, (dmta, dmtb, dmSx, dmAx), hermi=0)
    vj = vj.reshape(4,nao,nao)
    vk = vk.reshape(4,nao,nao)

    vkmop = numpy.einsum('pk,kl,lq->pq', mo_coeff[1].T, vk[2], mo_coeff[0])
    vkmom = numpy.einsum('pk,kl,lq->pq', mo_coeff[1].T, vk[3], mo_coeff[0])
    #G[T](ao)
    veff0dooa = vj[0]+vj[1] - vk[0]  
    veff0doob = vj[0]+vj[1] - vk[1]
    #wvoa = -(Q^alpha_ia - Q^alpha_ai)
    #= -(2*G^alpha_ia[T] + 2*T^alpha_ik*F^alpha_ak - 2*X_pi*K_pa[X])
    wvoa = numpy.einsum('ik,kl,la->ai', orboa.T, veff0dooa, orbva) * 2  
    wvoa+= numpy.einsum('ik,ak->ai', ta, fockamo[nocca:,:nocca]) * 2
    wvoa-= numpy.einsum('bi,ba->ai', x_ab, vkmop[noccb:,nocca:]) * 2
    wvoa-= numpy.einsum('bi,ba->ai', x_ab, vkmom[noccb:,nocca:]) * 2
    #wvob = -(Q^beta_ia - Q^beta_ai)
    #= -(2*G^beta_ia[T] - 2*T^beta_ac*F^beta_ic + 2*X_aq*K_iq[X])
    wvob = numpy.einsum('ik,kl,la->ai', orbob.T, veff0doob, orbvb) * 2
    wvob-= numpy.einsum('ac,ic->ai', tb, fockbmo[:noccb,noccb:]) * 2               
    wvob+= numpy.einsum('aj,ij->ai', x_ab, vkmop[:noccb,:nocca]) * 2  
    wvob+= numpy.einsum('aj,ij->ai', x_ab, vkmom[:noccb,:nocca]) * 2

    w = -numpy.hstack((wvoa.ravel(),wvob.ravel()))
    vresp = mf.gen_response(hermi=1)    
    def fvind(x):
        dm1 = numpy.empty((2,nao,nao))
        xa = x[:nvira*nocca].reshape(nvira,nocca)
        xb = x[nvira*nocca:].reshape(nvirb,noccb)
        dma = reduce(numpy.dot, (orbva, xa, orboa.T))
        dmb = reduce(numpy.dot, (orbvb, xb, orbob.T))
        dm1[0] = (dma + dma.T)/2
        dm1[1] = (dmb + dmb.T)/2
        v1 = vresp(dm1) * 2
        #Z_bi*F_ba - Z_aj*F_ij + 2*G_ia[Z^S]
        v1a = reduce(numpy.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(numpy.dot, (orbvb.T, v1[1], orbob))
        v1a+= numpy.einsum('bi,ba->ai', xa,fockamo[nocca:,nocca:])
        v1a-= numpy.einsum('aj,ij->ai', xa,fockamo[:nocca,:nocca])
        v1b+= numpy.einsum('bi,ba->ai', xb,fockbmo[noccb:,noccb:])
        v1b-= numpy.einsum('aj,ij->ai', xb,fockbmo[:noccb,:noccb])
        return numpy.hstack((v1a.ravel(), v1b.ravel()))
    z = lib.solve(
        fvind, w, tol=1e-10, max_cycle=500, dot=numpy.dot,
        lindep=DSOLVE_LINDEP, verbose=0, tol_residual=None
        )
    z1a = z[:nocca*nvira].reshape(nvira,nocca)
    z1b = z[nocca*nvira:].reshape(nvirb,noccb)
    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    z1ao = numpy.empty((2,nao,nao))
    z1ao[0] = reduce(numpy.dot, (orbva, z1a, orboa.T))
    z1ao[1] = reduce(numpy.dot, (orbvb, z1b, orbob.T))
    veff = vresp((z1ao+z1ao.transpose(0,2,1)) * .5)

    im0a = numpy.zeros((nmoa,nmoa))
    im0b = numpy.zeros((nmob,nmob))
    #Ground state: W_ij = F_ij
    im0a[:nocca,:nocca]+= fockamo[:nocca,:nocca]
    im0b[:noccb,:noccb]+= fockbmo[:noccb,:noccb]
    #W^alpha_ij = G_ij[T+Z^S] + T_ik*F_jk - X_pi*K_pj
    im0a[:nocca,:nocca]+= numpy.einsum('ik,kl,lj->ij', orboa.T, veff0dooa+veff[0], orboa)
    im0a[:nocca,:nocca]+= numpy.einsum('ik,kj->ij', ta,fockamo[:nocca,:nocca])
    im0a[:nocca,:nocca]-= numpy.einsum('ai,aj->ij', x_ab, vkmop[noccb:,:nocca])
    im0a[:nocca,:nocca]-= numpy.einsum('ai,aj->ij', x_ab, vkmom[noccb:,:nocca])
    #W^beta_ij = G_ij[T+Z^S]
    im0b[:noccb,:noccb]+= numpy.einsum('ik,kl,lj->ij', orbob.T, veff0doob+veff[1], orbob)
    #W^beta_ab = T_ac*F^beta_bc - X_aq*K_bq
    im0b[noccb:,noccb:] = numpy.einsum('ac,bc->ab', tb, fockbmo[noccb:,noccb:])
    im0b[noccb:,noccb:]-= numpy.einsum('ai,bi->ab', x_ab, vkmop[noccb:,:nocca])
    im0b[noccb:,noccb:]-= numpy.einsum('ai,bi->ab', x_ab, vkmom[noccb:,:nocca])
    #W_ai和W_ia合并
    #W^alpha_ai = Z^alpha_aj*F^alpha_ij / 2
    im0a[nocca:,:nocca] = numpy.einsum('aj,ij->ai', z1a, fockamo[:nocca,:nocca])
    #W^beta_ai = Z^beta_aj*F^beta_ij / 2 - X_aq*K_iq
    im0b[noccb:,:noccb] = numpy.einsum('aj,ij->ai', z1b, fockbmo[:noccb,:noccb]) 
    im0b[noccb:,:noccb]-= numpy.einsum('aj,ij->ai', x_ab, vkmop[:noccb, :nocca]) * 2
    im0b[noccb:,:noccb]-= numpy.einsum('aj,ij->ai', x_ab, vkmom[:noccb, :nocca]) * 2

    im0 = numpy.einsum('kp,pq,ql->kl',mo_coeff[0], im0a ,mo_coeff[0].T)
    im0+= numpy.einsum('kp,pq,ql->kl',mo_coeff[1], im0b ,mo_coeff[1].T)
    

    

    #(T+Z^S)(ao)
    dmz1dooa = (z1ao[0] + z1ao[0].T)/2 + dmta
    dmz1doob = (z1ao[1] + z1ao[1].T)/2 + dmtb


    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)                     #S(ao)

    #Ground state density matrix
    oo0a = orboa@orboa.T
    oo0b = orbob@orbob.T
    oo0T = oo0a + oo0b

    vj, vk = td_grad.get_jk(mol, (oo0a, dmz1dooa,
                                  oo0b, dmz1doob))       
    vj = vj.reshape(2,2,3,nao,nao)
    vk = vk.reshape(2,2,3,nao,nao)
    vhf1a,vhf1b = vj[0] + vj[1] - vk
    time1 = log.timer('2e AO integral derivatives', *time1)

    dmxvk = td_grad.get_k(mol,(dmSx,dmAx))
    dmxvk = dmxvk.reshape(2,3,nao,nao)
    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        #Ground state: D_ij*h1ao + D_ij*G_(\nabla i)j[D] + D_ij*G_(i\nabla j)[D]
        h1ao = hcore_deriv(ia)
        de[k] = numpy.einsum('xpq,pq->x', h1ao, oo0T)  
        de[k] += numpy.einsum('xpq,pq->x', vhf1a[0,:,p0:p1],  oo0a[p0:p1]) 
        de[k] += numpy.einsum('xpq,qp->x', vhf1a[0,:,p0:p1],  oo0a[:,p0:p1])  
        de[k] += numpy.einsum('xpq,pq->x', vhf1b[0,:,p0:p1],  oo0b[p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', vhf1b[0,:,p0:p1],  oo0b[:,p0:p1])
        
        #(T+Z^S)\nabla F: (T + Z^S)*h1ao + (T_ij + Z^S_ij)*G_(\nabla i)j[D] + (T_ij + Z^S_ij)*G_i(\nabla j)[D] + D_ij*G_(\nabla i)j[T+Z^S] + D_ij*G_i(\nabla j)[T+Z^S]
        de[k] += numpy.einsum('xpq,pq->x', h1ao, dmz1dooa) 
        de[k] += numpy.einsum('xpq,pq->x', h1ao, dmz1doob)
        de[k] += numpy.einsum('xpq,pq->x', vhf1a[0,:,p0:p1], dmz1dooa[p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', vhf1a[0,:,p0:p1], dmz1dooa[:,p0:p1])
        de[k] += numpy.einsum('xpq,pq->x', vhf1b[0,:,p0:p1], dmz1doob[p0:p1]) 
        de[k] += numpy.einsum('xpq,qp->x', vhf1b[0,:,p0:p1], dmz1doob[:,p0:p1]) 
        de[k] += numpy.einsum('xij,ij->x', vhf1a[1,:,p0:p1], oo0a[p0:p1]) 
        de[k] += numpy.einsum('xij,ji->x', vhf1a[1,:,p0:p1], oo0a[:,p0:p1]) 
        de[k] += numpy.einsum('xij,ij->x', vhf1b[1,:,p0:p1], oo0b[p0:p1]) 
        de[k] += numpy.einsum('xij,ji->x', vhf1b[1,:,p0:p1], oo0b[:,p0:p1])

        #(R^S*R^S + L^A*L^A)G: 2*R^S_ij*G_(\nabla i)j[R^S] + 2*R^S_ij*G_i(\nabla j)[R^S] + 2*L^A_ij*G_(\nabla i)j[L^A] + 2*L^A_ij*G_i(\nabla j)[L^A]
        de[k] -= numpy.einsum('xij,ij->x', dmxvk[0,:,p0:p1], dmSx[p0:p1]) * 2  
        de[k] -= numpy.einsum('xij,ji->x', dmxvk[0,:,p0:p1], dmSx[:,p0:p1]) * 2 
        de[k] -= numpy.einsum('xij,ij->x', dmxvk[1,:,p0:p1], dmAx[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ji->x', dmxvk[1,:,p0:p1], dmAx[:,p0:p1]) * 2 
        #S*W: -W_ij*S_(\nabla i)j - W_ij*S_i(\nabla j)
        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xpq,qp->x', s1[:,p0:p1], im0[:,p0:p1])
  


  

        

        de[k] += td_grad.extra_force(ia, locals())

    log.timer('TDHF nuclear gradients', *time0)
    return de




class MyGradients(uhf_grad.Gradients):

    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20)
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)
    
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
    
    def grad_elec(self, singlet, atmlst=None, with_nlc=None):
        if with_nlc is None:
            with_nlc = self.with_nlc
        return grad_elec(td_grad=self,          # 对应内层td_grad
            singlet=singlet,       # 对应内层singlet
            atmlst=atmlst,         # 对应内层atmlst
            max_memory=self.max_memory,  # 对应内层max_memory
            verbose=self.verbose,  # 对应内层verbose
            with_nlc=with_nlc,     # 对应内层with_nlc
            state_idx=self.state-1)

    def kernel(self, state=None, singlet=None, atmlst=None, with_nlc=None):
        '''
        Args:
        state : int
            Excited state ID.  state = 1 means the first excited state.
        with_nlc : bool or None
            Whether to include Nuclear-Lagrangian contributions.
            If None, use self.with_nlc
        '''
        
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

        de = self.grad_elec(singlet, atmlst, with_nlc=with_nlc)
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

from pyscf import gto, scf


class USFTDA(SF_TDA_down):
    def __init__(self, mf, method=0, davidson=False):
        super().__init__(mf, method, davidson)
        self.frozen = None
        self.mol = mf.mol


    
    def nuc_grad_method(self):
        return MyGradients(self)


from pyscf import scf
mol = gto.M(atom=''' N 0. 0. 0.; 
            N 0. 0. 0.97
            ''',charge = 1 , spin=1 ,basis = '6-31g')
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.max_cycle = 1000
mf.kernel()


td = USFTDA(mf, method=0 , davidson=False)
td.kernel()


my_grad = td.nuc_grad_method().kernel()


