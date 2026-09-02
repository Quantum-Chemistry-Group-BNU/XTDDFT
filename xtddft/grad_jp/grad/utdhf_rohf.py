#!/usr/bin/env python

from functools import reduce
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from TDA.UCIS import UCIS
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rohf as rohf_grad
from pyscf import __config__

def grad_elec(td_grad, x_y, singlet=True, atmlst = None,
              max_memory=2000, verbose=logger.INFO,with_nlc=None):
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
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]  #C_oa
    orbob = mo_coeff[1][:,occidxb]  #C_ob
    orbva = mo_coeff[0][:,viridxa]  #C_va
    orbvb = mo_coeff[1][:,viridxb]  #C_vb
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb
    nc = min(nocca, noccb)
    no = abs(nocca - noccb)
    nv = min(nvira, nvirb)

    #只测TDA，不调用y
    
    x_a = x_y[:nc*nv + no*nv].reshape(-1)  
    x_b = x_y[nc*nv + no*nv:].reshape(-1)
    x_a = numpy.hstack((x_a[:nc*nv].reshape(nc,nv).T,x_a[nc*nv:].reshape(no,nv).T))
    x_b = numpy.vstack((x_b[:nc*no].reshape(nc,no).T,x_b[nc*no:].reshape(nc,nv).T))

    fock_ao = mf.get_fock()  #fock(ao)
    focka = fock_ao[0]  #focka(ao)
    fockb = fock_ao[1]  #fockb(ao)
    
    fockamo = reduce(numpy.dot, (mo_coeff[0].T, focka, mo_coeff[0]))  #focka(mo)
    fockbmo = reduce(numpy.dot, (mo_coeff[1].T, fockb, mo_coeff[1]))  #fockb(mo)

    #所有fock矩阵块均做对称化处理
    #CC block
    fockacc = (fockamo[:nc,:nc] + fockamo[:nc,:nc].T) / 2
    fockbcc = (fockbmo[:nc,:nc] + fockbmo[:nc,:nc].T) / 2
 
    #focka OC-CO block
    fockaoc = (fockamo[nc:(nc+no),:nc] + fockamo[:nc,nc:(nc+no)].T) / 2  
    #VC-CV block
    fockavc = (fockamo[(nc+no):,:nc] + fockamo[:nc,(nc+no):].T) / 2
    fockbvc = (fockbmo[(nc+no):,:nc] + fockbmo[:nc,(nc+no):].T) / 2
    #VV block
    fockavv = (fockamo[(nc+no):,(nc+no):]+fockamo[(nc+no):,(nc+no):].T) / 2
    fockbvv = (fockbmo[(nc+no):,(nc+no):] + fockbmo[(nc+no):,(nc+no):].T) / 2
    #fockb VO-OV block  
    fockbvo = (fockbmo[(nc+no):,nc:(nc+no)] + fockbmo[nc:(nc+no),(nc+no):].T) / 2
    #OO block
    fockaoo = (fockamo[nc:(nc+no),nc:(nc+no)] + fockamo[nc:(nc+no),nc:(nc+no)].T) / 2  
    fockboo = (fockbmo[nc:(nc+no),nc:(nc+no)] + fockbmo[nc:(nc+no),nc:(nc+no)].T) / 2  

    #T_ab
    dvva = numpy.einsum('ai,bi->ab', x_a, x_a)            
    dvvb = numpy.einsum('ai,bi->ab', x_b, x_b)
    #T_ij
    dooa = -numpy.einsum('ai,aj->ij', x_a, x_a)           
    doob = -numpy.einsum('ai,aj->ij', x_b, x_b)
    #X(ao)
    dmxa = numpy.einsum('ka,ai,il->kl',orbva, x_a, orboa.T)    
    dmxb = numpy.einsum('ka,ai,il->kl',orbvb, x_b, orbob.T)
    #R^S
    dmSa = (dmxa + dmxa.T)/2                              
    dmSb = (dmxb + dmxb.T)/2
    #L^A                               
    dmAa = (dmxa - dmxa.T)/2                              
    dmAb = (dmxb - dmxb.T)/2
    #T(ao)
    dmzooa = numpy.einsum('ki,ij,jl->kl', orboa, dooa, orboa.T) 
    dmzoob = numpy.einsum('ki,ij,jl->kl', orbob, doob, orbob.T)  
    dmzooa+= numpy.einsum('ka,ab,bl->kl', orbva, dvva, orbva.T)
    dmzoob+= numpy.einsum('ka,ab,bl->kl', orbvb, dvvb, orbvb.T)

    Q_a = numpy.zeros((nmoa,nmoa))
    Q_b = numpy.zeros((nmob,nmob))
    #wvo = -(Q_ia-Q_ai)
    #    = -(2*G_ia[T] + 2*R_ib*G_ab[R^S] + 2*L_ib*G_ab[L^A] - 2*R_ja*G_ji[R^S] - 2*L_ja*G_ji[L^A] + 2*T_ij*F_aj - 2*T_ba*F_bi)
    vj, vk = mf.get_jk(mol, (dmzooa, dmSa, dmAa,
                             dmzoob, dmSb, dmAb), hermi=0)
    vj = vj.reshape(2,3,nao,nao)
    vk = vk.reshape(2,3,nao,nao)
    veff0doo = vj[0,0]+vj[1,0] - vk[:,0]  #G[T](ao)
   #G_ia[T]
    Q_a[:nocca,nocca:]+= numpy.einsum('ik,kl,la->ia', orboa.T, veff0doo[0], orbva) * 2  
    Q_b[:noccb,noccb:]+= numpy.einsum('ik,kl,la->ia', orbob.T, veff0doo[1], orbvb) * 2
    veff = vj[0,1]+vj[1,1] - vk[:,1]  #G[R^S](ao)
    #G[R^S](mo)
    veff0mopa = numpy.einsum('pk,kl,lq->pq', mo_coeff[0].T, veff[0], mo_coeff[0]) 
    veff0mopb = numpy.einsum('pk,kl,lq->pq', mo_coeff[1].T, veff[1], mo_coeff[1])  
    veff = - vk[:,2]  #G[L^A](ao)
    #G[L^A](mo)
    veff0moma = numpy.einsum('pk,kl,lq->pq', mo_coeff[0].T, veff[0], mo_coeff[0])
    veff0momb = numpy.einsum('pk,kl,lq->pq', mo_coeff[1].T, veff[1], mo_coeff[1])
    #R_biG_ba[R^S]
    Q_a[:nocca,nocca:] += numpy.einsum('bi,ba->ia', x_a, veff0mopa[nocca:,nocca:]) * 2  
    Q_b[:noccb,noccb:] += numpy.einsum('bi,ba->ia', x_b, veff0mopb[noccb:,noccb:]) * 2
    #L_biG_ba[L^A]
    Q_a[:nocca,nocca:] += numpy.einsum('bi,ba->ia', x_a, veff0moma[nocca:,nocca:]) * 2  
    Q_b[:noccb,noccb:] += numpy.einsum('bi,ba->ia', x_b, veff0momb[noccb:,noccb:]) * 2
    #T_ij*F_aj
    Q_a[:nocca,nocca:] += numpy.einsum('ij,aj->ia', dooa, fockamo[nocca:,:nocca]) * 2
    Q_b[:noccb,noccb:] += numpy.einsum('ij,aj->ia', doob, fockbmo[noccb:,:noccb]) * 2
    #R_ajG_ij[R^S]
    Q_a[nocca:,:nocca] += numpy.einsum('ai,ki->ak', x_a, veff0mopa[:nocca,:nocca]) * 2  
    Q_b[noccb:,:noccb] += numpy.einsum('ai,ki->ak', x_b, veff0mopb[:noccb,:noccb]) * 2
    #L_ajG_ij[L^A]
    Q_a[nocca:,:nocca] += numpy.einsum('ai,ki->ak', x_a, veff0moma[:nocca,:nocca]) * 2  
    Q_b[noccb:,:noccb] += numpy.einsum('ai,ki->ak', x_b, veff0momb[:noccb,:noccb]) * 2
    #T_ba*F_bi
    Q_a[nocca:,:nocca] += numpy.einsum('ba,bi->ai', dvva, fockamo[nocca:,:nocca]) * 2
    Q_b[noccb:,:noccb] += numpy.einsum('ba,bi->ai', dvvb, fockbmo[noccb:,:noccb]) * 2

    Q_a[:nocca,:nocca] += numpy.einsum('ik,kl,lj->ij', orboa.T, veff0doo[0], orboa) * 2
    Q_b[:noccb,:noccb] += numpy.einsum('ik,kl,lj->ij', orbob.T, veff0doo[1], orbob) * 2
    Q_a[:nocca,:nocca] += numpy.einsum('bi,bj->ij', x_a, veff0mopa[nocca:,:nocca]) * 2
    Q_b[:noccb,:noccb] += numpy.einsum('bi,bj->ij', x_b, veff0mopb[noccb:,:noccb]) * 2
    Q_a[:nocca,:nocca] += numpy.einsum('bi,bj->ij', x_a, veff0moma[nocca:,:nocca]) * 2
    Q_b[:noccb,:noccb] += numpy.einsum('bi,bj->ij', x_b, veff0momb[noccb:,:noccb]) * 2
    Q_a[:nocca,:nocca] += numpy.einsum('ik,jk->ij', dooa, fockamo[:nocca,:nocca]) * 2
    Q_b[:noccb,:noccb] += numpy.einsum('ik,jk->ij', doob, fockbmo[:noccb,:noccb]) * 2

    Q_a[nocca:,nocca:] += numpy.einsum('ai,bi->ab', x_a, veff0mopa[nocca:,:nocca]) * 2
    Q_b[noccb:,noccb:] += numpy.einsum('ai,bi->ab', x_b, veff0mopb[noccb:,:noccb]) * 2
    Q_a[nocca:,nocca:] += numpy.einsum('ai,bi->ab', x_a, veff0moma[nocca:,:nocca]) * 2
    Q_b[noccb:,noccb:] += numpy.einsum('ai,bi->ab', x_b, veff0momb[noccb:,:noccb]) * 2
    Q_a[nocca:,nocca:] += numpy.einsum('ac,bc->ab', dvva,fockamo[nocca:,nocca:]) * 2
    Q_b[noccb:,noccb:] += numpy.einsum('ac,bc->ab', dvvb,fockbmo[noccb:,noccb:]) * 2
    Q_t = Q_a + Q_b
    qt = Q_t - Q_t.T

    wvc = qt[(nc+no):,:nc]
    wvo = qt[(nc+no):,nc:(nc+no)]
    woc = qt[nc:(nc+no),:nc]
    
    w = numpy.hstack((wvc.ravel(),wvo.ravel(),woc.ravel()))

    vresp = mf.gen_response(hermi=1)
    def matvec(x):
        xvc = x[:nv*nc].reshape(nv,nc)                               #Z_ai
        xvo = x[nv*nc:(nv*nc + nv*no)].reshape(nv,no)                #Z_at(a) 
        xoc = x[(nv*nc + nv*no):].reshape(no,nc)                     #Z_ti(b)
        #Z(mo)
        xa = numpy.hstack((xvc,xvo))                                 
        xb = numpy.vstack((xoc,xvc))
        #Z(ao)
        dm1 = numpy.zeros((2,nao,nao))
        dma = numpy.einsum('ka,ai,il->kl', orbva, xa, orboa.T)               
        dmb = numpy.einsum('ka,ai,il->kl', orbvb, xb, orbob.T)
        #Z^S
        dm1[0] += (dma + dma.T)/2                                     
        dm1[1] += (dmb + dmb.T)/2                                     
        v1 = vresp(dm1)                                               #G[Z^S](ao)
        #G[Z^S](mo)
        v1a = numpy.einsum('ak,kl,li->ai', orbva.T, v1[0], orboa) 
        v1b = numpy.einsum('ak,kl,li->ai', orbvb.T, v1[1], orbob) 
        vvc = (v1a[:,:nc] + v1b[no:,:])  #G^T_ai[Z^S]
        voc = v1b[:no,:]                 #G^beta_ti[Z^S]
        vvo = v1a[:,nc:]                 #G^alpha_at[Z^S]

        #VC block: Z_bi*F^alpha_ab + Z_bi*F^beta_ab - Z_aj*F^alpha_ji - Z_aj*F^beta_ji + Z_ti*F^beta_at - Z_at*F^alpha_ti + 2G^T_ai[Z^S]
        Fxvc = numpy.zeros((nv,nc))
        Fxvc += numpy.einsum('bi,ab->ai',xvc,fockavv)
        Fxvc += numpy.einsum('bi,ab->ai',xvc,fockbvv)
        Fxvc -= numpy.einsum('aj,ji->ai',xvc,fockacc)
        Fxvc -= numpy.einsum('aj,ji->ai',xvc,fockbcc)
        Fxvc += numpy.einsum('ti,at->ai',xoc,fockbvo)
        Fxvc -= numpy.einsum('at,ti->ai',xvo,fockaoc)
        Fxvc += vvc * 2
        #VO block: Z_ti*F^beta_ai - Z_ai*F^alpha_ti + Z_bt*F^alpha_ab - Z_au*F^alpha_ut + 2G^alpha_at[Z^S]
        Fxvo = numpy.zeros((nv,no))
        Fxvo += numpy.einsum('ti,ai->at',xoc,fockbvc)
        Fxvo -= numpy.einsum('ai,ti->at',xvc,fockaoc)
        Fxvo += numpy.einsum('bt,ba->at',xvo,fockavv)
        Fxvo -= numpy.einsum('au,tu->at',xvo,fockaoo)
        Fxvo += vvo * 2
        #OC block: Z_ui*F^beta_tu - Z_tj*F^beta_ji + Z_ai*F^beta_at - Z_at*F^alpha_ai + 2G^beta_ti[Z^S]
        Fxoc = numpy.zeros((no,nc))
        Fxoc += numpy.einsum('ui,tu->ti',xoc,fockboo)
        Fxoc -= numpy.einsum('tj,ij->ti',xoc,fockbcc)
        Fxoc += numpy.einsum('ai,at->ti',xvc,fockbvo)
        Fxoc -= numpy.einsum('at,ai->ti',xvo,fockavc)
        Fxoc += voc * 2
        return numpy.hstack((Fxvc.ravel(),Fxvo.ravel(),Fxoc.ravel()))

    #get_z
    z = lib.solve(
        matvec, w, tol=1e-20, max_cycle=500, dot=numpy.dot,
        lindep=DSOLVE_LINDEP, verbose=0, tol_residual=None
        )
    zvc = z[:nv*nc].reshape(nv,nc)
    zvo = z[nv*nc:(nv*nc + nv*no)].reshape(nv,no)
    zoc = z[(nv*nc + nv*no):].reshape(no,nc)

    z1a = numpy.hstack((zvc,zvo))
    z1b = numpy.vstack((zoc,zvc))
    
    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    z1ao = numpy.zeros((2,nao,nao))
    z1ao[0] += numpy.einsum('ka,ai,il->kl', orbva, z1a, orboa.T)
    z1ao[1] += numpy.einsum('ka,ai,il->kl', orbvb, z1b, orbob.T)

    #G[Z^S]
    veff = vresp((z1ao+z1ao.transpose(0,2,1)) * .5)


    im0a = numpy.zeros((nmoa,nmoa))
    im0b = numpy.zeros((nmob,nmob))
    #Ground state: W_ij = F_ij
    im0a[:nocca,:nocca]+= fockamo[:nocca,:nocca]
    im0b[:noccb,:noccb]+= fockbmo[:noccb,:noccb]
    
    
    #W_ij = G_ij[T+Z^S] + R_ai*G_aj[R^S] + L_ai*G_ai[L^A] + T_ik*F_kj
  
    im0a[:nocca,:nocca]+= numpy.einsum('ik,kl,lj->ij', orboa.T, veff0doo[0]+veff[0], orboa)
    im0b[:noccb,:noccb]+= numpy.einsum('ik,kl,lj->ij', orbob.T, veff0doo[1]+veff[1], orbob)
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', x_a, veff0mopa[nocca:,:nocca])
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', x_b, veff0mopb[noccb:,:noccb])
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', x_a, veff0moma[nocca:,:nocca])
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', x_b, veff0momb[noccb:,:noccb])
    im0a[:nocca,:nocca]+= numpy.einsum('ik,jk->ij', dooa,fockamo[:nocca,:nocca])
    im0b[:noccb,:noccb]+= numpy.einsum('ik,jk->ij', doob,fockbmo[:noccb,:noccb])

    #W_ab = R_ai*G_bi[R^S] + L_ai*G_bi[L^A] + T_ac*F_cb
    im0a[nocca:,nocca:] = numpy.einsum('ai,ci->ac', x_a, veff0mopa[nocca:,:nocca])
    im0b[noccb:,noccb:] = numpy.einsum('ai,ci->ac', x_b, veff0mopb[noccb:,:noccb])
    im0a[nocca:,nocca:]+= numpy.einsum('ai,ci->ac', x_a, veff0moma[nocca:,:nocca])
    im0b[noccb:,noccb:]+= numpy.einsum('ai,ci->ac', x_b, veff0momb[noccb:,:noccb])
    im0a[nocca:,nocca:]+= numpy.einsum('ac,bc->ab', dvva,fockamo[nocca:,nocca:])
    im0b[noccb:,noccb:]+= numpy.einsum('ac,bc->ab', dvvb,fockbmo[noccb:,noccb:])
    #W_ai和W_ia合并
    #W_ai = R_ja*G_ji[R^S] + L_ja*G_ji[L^A] + T_ba*F_bi + Z_aj*F^_ji / 2
    im0a[nocca:,:nocca] = numpy.einsum('ai,ki->ak', x_a, veff0mopa[:nocca,:nocca]) 
    im0b[noccb:,:noccb] = numpy.einsum('ai,ki->ak', x_b, veff0mopb[:noccb,:noccb]) 
    im0a[nocca:,:nocca]+= numpy.einsum('ai,ki->ak', x_a, veff0moma[:nocca,:nocca]) 
    im0b[noccb:,:noccb]+= numpy.einsum('ai,ki->ak', x_b, veff0momb[:noccb,:noccb]) 
    im0a[nocca:,:nocca]+= numpy.einsum('ba,bi->ai', dvva, fockamo[nocca:,:nocca]) 
    im0b[noccb:,:noccb]+= numpy.einsum('ba,bi->ai', dvvb, fockbmo[noccb:,:noccb]) 
    im0a[nocca:,:nocca]+= numpy.einsum('aj,ij->ai', z1a, fockamo[:nocca,:nocca]) / 2
    im0b[noccb:,:noccb]+= numpy.einsum('aj,ij->ai', z1b, fockbmo[:noccb,:noccb]) / 2
    im0b[nc:(nc+no),:nc]+= numpy.einsum('bt,bi->ti', zvo, fockavc) / 2

    #W_ia = G_ia[T+Z^S] + R_bi*G_ba[R^S] + L_bi*G_ba[L^A] + T_ij*F_ja + Z_bi*F^_ba / 2
    im0a[:nocca,nocca:]+= numpy.einsum('ik,kl,la->ia', orboa.T, veff0doo[0]+veff[0], orbva)
    im0b[:noccb,noccb:]+= numpy.einsum('ik,kl,la->ia', orbob.T, veff0doo[1]+veff[1], orbvb)
    im0a[:nocca,nocca:]+= numpy.einsum('bi,ba->ia', x_a, veff0mopa[nocca:,nocca:])
    im0b[:noccb,noccb:]+= numpy.einsum('bi,ba->ia', x_b, veff0mopb[noccb:,noccb:])
    im0a[:nocca,nocca:]+= numpy.einsum('bi,ba->ia', x_a, veff0moma[nocca:,nocca:])
    im0b[:noccb,noccb:]+= numpy.einsum('bi,ba->ia', x_b, veff0momb[noccb:,noccb:])
    im0a[:nocca,nocca:]+= numpy.einsum('ij,ja->ia', dooa, fockamo[:nocca,nocca:])
    im0b[:noccb,noccb:]+= numpy.einsum('ij,ja->ia', doob, fockbmo[:noccb,noccb:])
    im0a[:nocca,nocca:]+= numpy.einsum('bi,ba->ia', z1a, fockamo[nocca:,nocca:]) / 2
    im0b[:noccb,noccb:]+= numpy.einsum('bi,ba->ia', z1b, fockbmo[noccb:,noccb:]) / 2
    im0a[nc:(nc+no),(nc+no):]+= numpy.einsum('tj,aj->ta', zoc, fockbvc) / 2


    im0 = im0a + im0b
    print(nc,no,nv)
    print(im0-im0.T)
    im0 = numpy.einsum('kp,pq,ql->kl',mo_coeff[0], im0 ,mo_coeff[0].T)

    

    #(T+Z^S)(ao)
    dmz1dooa = (z1ao[0] + z1ao[0].T)/2 + dmzooa
    dmz1doob = (z1ao[1] + z1ao[1].T)/2 + dmzoob

    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)                     #S(ao)

    #Ground state density matrix
    oo0a = orboa@orboa.T
    oo0b = orbob@orbob.T
    oo0T = oo0a + oo0b

    vj, vk = td_grad.get_jk(mol, (oo0a, dmz1dooa, dmSa,dmAa,
                                  oo0b, dmz1doob, dmSb,dmAb))       
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
        #Ground state: D_ij*h1ao + D_ij*G_(\nabla i)j[D] + D_ij*G_(i\nabla j)[D]
        h1ao = hcore_deriv(ia)
        de[k] = numpy.einsum('xpq,pq->x', h1ao, oo0T)  
        de[k] += numpy.einsum('xpq,pq->x', vhf1a[0,:,p0:p1],  oo0a[p0:p1])
        de[k] += numpy.einsum('xqp,pq->x', vhf1a[0,:,p0:p1],  oo0a[:,p0:p1])
        de[k] += numpy.einsum('xpq,pq->x', vhf1b[0,:,p0:p1],  oo0b[p0:p1])
        de[k] += numpy.einsum('xqp,pq->x', vhf1b[0,:,p0:p1],  oo0b[:,p0:p1])
        #(T+Z^S)\nabla F: (T + Z^S)*h1ao + (T_ij + Z^S_ij)*G_(\nabla i)j[D] + (T_ij + Z^S_ij)*G_i(\nabla j)[D] + D_ij*G_(\nabla i)j[T+Z^S] + D_ij*G_i(\nabla j)[T+Z^S]
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
        #(R^S*R^S + L^A*L^A)G: 2*R^S_ij*G_(\nabla i)j[R^S] + 2*R^S_ij*G_i(\nabla j)[R^S] + 2*L^A_ij*G_(\nabla i)j[L^A] + 2*L^A_ij*G_i(\nabla j)[L^A]
        de[k] += numpy.einsum('xij,ij->x', vhf1a[2,:,p0:p1], dmSa[p0:p1]) * 2
        de[k] += numpy.einsum('xji,ij->x', vhf1a[2,:,p0:p1], dmSa[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1b[2,:,p0:p1], dmSb[p0:p1]) * 2
        de[k] += numpy.einsum('xji,ij->x', vhf1b[2,:,p0:p1], dmSb[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1a[3,:,p0:p1], dmAa[p0:p1]) * 2
        de[k] -= numpy.einsum('xji,ij->x', vhf1a[3,:,p0:p1], dmAa[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1b[3,:,p0:p1], dmAb[p0:p1]) * 2
        de[k] -= numpy.einsum('xji,ij->x', vhf1b[3,:,p0:p1], dmAb[:,p0:p1]) * 2  
        #S*W: -W_ij*S_(\nabla i)j - W_ij*S_i(\nabla j)
        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])
  

        

        de[k] += td_grad.extra_force(ia, locals())

    log.timer('TDHF nuclear gradients', *time0)
    return de




class MyGradients(rohf_grad.Gradients):

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
    
    def grad_elec(self, xy, singlet, atmlst=None, with_nlc=None):
        if with_nlc is None:
            with_nlc = self.with_nlc
        return grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)

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

            xy = self.base.v[:,state-1]


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

from pyscf import gto, scf


class UTDHF_ROHF(UCIS):
    def __init__(self, mf):
        if isinstance(mf, scf.rohf.ROHF):
            mf = mf.to_uhf()
        super().__init__(mf.mol, mf,  
            savedata=False)
        self.frozen = None

    
    def nuc_grad_method(self):
        return MyGradients(self)
    

from pyscf import scf
mol = gto.M(atom ='''O 0.000000 0.000000 1.207800;
            C 0.000000 0.000000 0.000000;
            H 0.000000 1.092733 -0.356967;
            H 0.000000 -1.092733 -0.356967''',charge = 1 , spin= 1 ,basis = '6-31g**')
mf = scf.ROHF(mol)
mf.conv_tol = 1e-12
mf.max_cycle = 1000
mf.kernel()


td = UTDHF_ROHF(mf)  
td.kernel()
my_grad = td.nuc_grad_method().kernel()
