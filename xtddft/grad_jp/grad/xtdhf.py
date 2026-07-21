from functools import reduce
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from TDA.XCIS import XCIS
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rohf as rohf_grad
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
    DSOLVE_LINDEP = getattr(__config__, 'lib_linalg_helper_dsolve_lindep', 1e-20)

    mol = td_grad.mol
    mf = td_grad.base._scf
    xtda = td_grad.base
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    occidxa = xtda.occidx_a
    viridxa = xtda.viridx_a 
    occidxb = xtda.occidx_b 
    viridxb = xtda.viridx_b
    nc = xtda.nc  #double occupied number
    no = xtda.no  #single occupied number
    nv = xtda.nv  #virtual occupied number
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[:,occidxa]  #C_oa
    orbob = mo_coeff[:,occidxb]  #C_ob
    orbva = mo_coeff[:,viridxa]  #C_va
    orbvb = mo_coeff[:,viridxb]  #C_vb
    doccidx = numpy.where(mo_occ == 2)[0]
    soccidx = numpy.where(mo_occ == 1)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    orbc = mo_coeff[:,doccidx]
    orbo = mo_coeff[:,soccidx]
    orbv = mo_coeff[:,viridx]
    nao = mo_coeff.shape[0]
    nmo = nocca + nvira
    

    s = (mol.spin) / 2
    
    x_vc_a = xtda.xycv_a[state_idx].reshape(nc,nv).T
    x_vo_a = xtda.xyov_a[state_idx].reshape(no,nv).T
    x_oc_b = xtda.xyco_b[state_idx].reshape(nc,no).T
    x_vc_b = xtda.xycv_b[state_idx].reshape(nc,nv).T
    x_a = numpy.hstack((x_vc_a, x_vo_a))
    x_b = numpy.vstack((x_oc_b, x_vc_b))

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    fock_ao = mf.get_fock(dm = dm)  #fock(ao)
    focka = fock_ao.focka  #focka(ao)
    fockb = fock_ao.fockb  #fockb(ao)
    
    fockamo = reduce(numpy.dot, (mo_coeff.T, focka, mo_coeff))  #focka(mo)
    fockbmo = reduce(numpy.dot, (mo_coeff.T, fockb, mo_coeff))  #fockb(mo)

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

    #Ground state density matrix
    oo0a = orboa@orboa.T
    oo0b = orbob@orbob.T
    oo0T = oo0a + oo0b
    oo0S = (oo0a - oo0b) / 2



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
    #additional coefficients
    c1 = -1 + numpy.sqrt((s+1)/s) + 1/(2*s)
    c2 = 1 - numpy.sqrt((s+1)/s) + 1/(2*s)
    c3 = -1/(2*s)
    #xtcc 
    xtcc = numpy.einsum('ia,ja->ij', x_vc_a.T,x_vc_a.T) * c1
    xtcc+= numpy.einsum('ia,ja->ij', x_vc_b.T,x_vc_b.T) * c2
    xtcc+= (numpy.einsum('ia,ja->ij', x_vc_a.T,x_vc_b.T) + numpy.einsum('ia,ja->ij', x_vc_b.T,x_vc_a.T)) * c3
    #xdvv
    xtvv = numpy.einsum('ai,bi->ab', x_vc_a,x_vc_a) * c2
    xtvv+= numpy.einsum('ai,bi->ab', x_vc_b,x_vc_b) * c1
    xtvv+= (numpy.einsum('ai,bi->ab', x_vc_a,x_vc_b) + numpy.einsum('ai,bi->ab', x_vc_b,x_vc_a)) * c3

    #xd(ao)
    xtao = numpy.einsum('ki,ij,jl->kl', orbc, xtcc, orbc.T) + numpy.einsum('ka,ab,bl->kl', orbv, xtvv, orbv.T)

    
    vj, vk = mf.get_jk(mol, (dmzooa, dmSa, dmAa,
                             dmzoob, dmSb, dmAb), hermi=0)
    vj = vj.reshape(2,3,nao,nao)
    vk = vk.reshape(2,3,nao,nao)
    veff0doo = vj[0,0]+vj[1,0] - vk[:,0]  #G[T](ao)
    
    veff = vj[0,1]+vj[1,1] - vk[:,1]  #G[R^S](ao)
    #G[R^S](mo)
    veff0mopa = numpy.einsum('pk,kl,lq->pq', mo_coeff.T, veff[0], mo_coeff) 
    veff0mopb = numpy.einsum('pk,kl,lq->pq', mo_coeff.T, veff[1], mo_coeff)  
                
    veff = -vk[:,2]  #G[L^A](ao)
    #G[L^A](mo)
    veff0moma = numpy.einsum('pk,kl,lq->pq', mo_coeff.T, veff[0], mo_coeff)
    veff0momb = numpy.einsum('pk,kl,lq->pq', mo_coeff.T, veff[1], mo_coeff)
    
    Q_a = numpy.zeros((nmo,nmo))
    Q_b = numpy.zeros((nmo,nmo))
    #Qia
    Q_a[:nocca,nocca:]+= numpy.einsum('ik,kl,la->ia', orboa.T, veff0doo[0], orbva) * 2  
    Q_b[:noccb,noccb:]+= numpy.einsum('ik,kl,la->ia', orbob.T, veff0doo[1], orbvb) * 2   
    Q_a[:nocca,nocca:] += numpy.einsum('bi,ba->ia', x_a, veff0mopa[nocca:,nocca:]) * 2  
    Q_b[:noccb,noccb:] += numpy.einsum('bi,ba->ia', x_b, veff0mopb[noccb:,noccb:]) * 2
    Q_a[:nocca,nocca:] += numpy.einsum('bi,ba->ia', x_a, veff0moma[nocca:,nocca:]) * 2  
    Q_b[:noccb,noccb:] += numpy.einsum('bi,ba->ia', x_b, veff0momb[noccb:,noccb:]) * 2
    Q_a[:nocca,nocca:] += numpy.einsum('ij,aj->ia', dooa, fockamo[nocca:,:nocca]) * 2
    Q_b[:noccb,noccb:] += numpy.einsum('ij,aj->ia', doob, fockbmo[noccb:,:noccb]) * 2
    #Qai
    Q_a[nocca:,:nocca] += numpy.einsum('ai,ki->ak', x_a, veff0mopa[:nocca,:nocca]) * 2  
    Q_b[noccb:,:noccb] += numpy.einsum('ai,ki->ak', x_b, veff0mopb[:noccb,:noccb]) * 2
    Q_a[nocca:,:nocca] += numpy.einsum('ai,ki->ak', x_a, veff0moma[:nocca,:nocca]) * 2  
    Q_b[noccb:,:noccb] += numpy.einsum('ai,ki->ak', x_b, veff0momb[:noccb,:noccb]) * 2
    Q_a[nocca:,:nocca] += numpy.einsum('ba,bi->ai', dvva, fockamo[nocca:,:nocca]) * 2
    Q_b[noccb:,:noccb] += numpy.einsum('ba,bi->ai', dvvb, fockbmo[noccb:,:noccb]) * 2
    #Qij
    Q_a[:nocca,:nocca] += numpy.einsum('ik,kl,lj->ij', orboa.T, veff0doo[0], orboa) * 2
    Q_b[:noccb,:noccb] += numpy.einsum('ik,kl,lj->ij', orbob.T, veff0doo[1], orbob) * 2
    Q_a[:nocca,:nocca] += numpy.einsum('bi,bj->ij', x_a, veff0mopa[nocca:,:nocca]) * 2
    Q_b[:noccb,:noccb] += numpy.einsum('bi,bj->ij', x_b, veff0mopb[noccb:,:noccb]) * 2
    Q_a[:nocca,:nocca] += numpy.einsum('bi,bj->ij', x_a, veff0moma[nocca:,:nocca]) * 2
    Q_b[:noccb,:noccb] += numpy.einsum('bi,bj->ij', x_b, veff0momb[noccb:,:noccb]) * 2
    Q_a[:nocca,:nocca] += numpy.einsum('ik,jk->ij', dooa, fockamo[:nocca,:nocca]) * 2
    Q_b[:noccb,:noccb] += numpy.einsum('ik,jk->ij', doob, fockbmo[:noccb,:noccb]) * 2
    #Qab
    Q_a[nocca:,nocca:] += numpy.einsum('ai,bi->ab', x_a, veff0mopa[nocca:,:nocca]) * 2
    Q_b[noccb:,noccb:] += numpy.einsum('ai,bi->ab', x_b, veff0mopb[noccb:,:noccb]) * 2
    Q_a[nocca:,nocca:] += numpy.einsum('ai,bi->ab', x_a, veff0moma[nocca:,:nocca]) * 2
    Q_b[noccb:,noccb:] += numpy.einsum('ai,bi->ab', x_b, veff0momb[noccb:,:noccb]) * 2
    Q_a[nocca:,nocca:] += numpy.einsum('ac,bc->ab', dvva,fockamo[nocca:,nocca:]) * 2
    Q_b[noccb:,noccb:] += numpy.einsum('ac,bc->ab', dvvb,fockbmo[noccb:,noccb:]) * 2
                
    Q_t = Q_a + Q_b

    dvk = mf.get_k(mol, (oo0S,xtao), hermi=0)
    dvk = dvk.reshape(2,nao,nao)
    dvkmo = numpy.zeros((2,nmo,nmo))
    dvkmo[0] = numpy.einsum('pk,kl,lq',mo_coeff.T, dvk[0], mo_coeff)
    dvkmo[1] = numpy.einsum('pk,kl,lq',mo_coeff.T, dvk[1], mo_coeff)
    doo_mo = numpy.ones((no,no)) / 2

    Q_t[:nc,(nc+no):] += numpy.einsum('ij,aj->ia', xtcc, dvkmo[0][(nc+no):,:nc]) * 2
    Q_t[(nc+no):,:nc] += numpy.einsum('ab,ib->ai', xtvv, dvkmo[0][:nc,(nc+no):]) * 2
    Q_t[nc:(nc+no),(nc+no):] += numpy.einsum('tu,au->ta',doo_mo, dvkmo[1][(nc+no):, nc:(nc+no)]) * 2
    Q_t[(nc+no):,nc:(nc+no)] += numpy.einsum('ab,tb->at',xtvv, dvkmo[0][nc:(nc+no),(nc+no):]) * 2
    Q_t[:nc,nc:(nc+no)] += numpy.einsum('ij,tj->it', xtcc, dvkmo[0][nc:(nc+no),:nc]) * 2
    Q_t[nc:(nc+no),:nc] += numpy.einsum('ut,ui->ti', doo_mo, dvkmo[1][nc:(nc+no),:nc]) * 2   
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
        matvec, w, tol=1e-20, max_cycle=1000, dot=numpy.dot,
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


    im0a = numpy.zeros((nmo,nmo))
    im0b = numpy.zeros((nmo,nmo))
    #Ground state: W_ij = F_ij
    im0a[:nocca,:nocca]+= fockamo[:nocca,:nocca]
    im0b[:noccb,:noccb]+= fockbmo[:noccb,:noccb]
    
    
    #W_ij = G_ij[T+Z^S] + R_ai*G_aj[R^S] + L_ai*G_ai[L^A] + T_ik*F_kj
  
    im0a[:nocca,:nocca]+= numpy.einsum('ik,kl,lj->ij', orboa.T, veff0doo[0]+veff[0], orboa)
    im0b[:noccb,:noccb]+= numpy.einsum('ik,kl,lj->ij', orbob.T, veff0doo[1]+veff[1], orbob)
    im0a[:nocca,:nocca]+= numpy.einsum('ai,ak->ki', x_a, veff0mopa[nocca:,:nocca])
    im0b[:noccb,:noccb]+= numpy.einsum('ai,ak->ki', x_b, veff0mopb[noccb:,:noccb])
    im0a[:nocca,:nocca]+= numpy.einsum('ai,ak->ki', x_a, veff0moma[nocca:,:nocca])
    im0b[:noccb,:noccb]+= numpy.einsum('ai,ak->ki', x_b, veff0momb[noccb:,:noccb])
    im0a[:nocca,:nocca]+= numpy.einsum('ik,kj->ij', dooa,fockamo[:nocca,:nocca])
    im0b[:noccb,:noccb]+= numpy.einsum('ik,kj->ij', doob,fockbmo[:noccb,:noccb])
    #im0a[:nocca,:nocca]+= numpy.einsum('ai,aj->ij', z1a, fockamo[nocca:,:nocca]) / 2
    #im0b[:noccb,:noccb]+= numpy.einsum('ai,aj->ij', z1b, fockbmo[noccb:,:noccb]) / 2

    #W_ab = R_ai*G_bi[R^S] + L_ai*G_bi[L^A] + T_ac*F_cb
    im0a[nocca:,nocca:]+= numpy.einsum('ci,ai->ac', veff0mopa[nocca:,:nocca], x_a)
    im0b[noccb:,noccb:]+= numpy.einsum('ci,ai->ac', veff0mopb[noccb:,:noccb], x_b)
    im0a[nocca:,nocca:]+= numpy.einsum('ci,ai->ac', veff0moma[nocca:,:nocca], x_a)
    im0b[noccb:,noccb:]+= numpy.einsum('ci,ai->ac', veff0momb[noccb:,:noccb], x_b)
    im0a[nocca:,nocca:]+= numpy.einsum('ac,cb->ab', dvva,fockamo[nocca:,nocca:])
    im0b[noccb:,noccb:]+= numpy.einsum('ac,cb->ab', dvvb,fockbmo[noccb:,noccb:])
    #im0a[nocca:,nocca:]+= numpy.einsum('ai,bi->ab', z1a, fockamo[nocca:,:nocca]) / 2
    #im0b[noccb:,noccb:]+= numpy.einsum('ai,bi->ab', z1b, fockbmo[noccb:,:noccb]) / 2

    #W_ai和W_ia合并
    #W_ai = R_ja*G_ji[R^S] + L_ja*G_ji[L^A] + T_ba*F_bi + Z_aj*F^_ji / 2
    im0a[nocca:,:nocca]+= numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], x_a) * 2
    im0b[noccb:,:noccb]+= numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], x_b) * 2
    im0a[nocca:,:nocca]+= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], x_a) * 2
    im0b[noccb:,:noccb]+= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], x_b) * 2
    im0a[nocca:,:nocca]+= numpy.einsum('ba,bi->ai', dvva, fockamo[nocca:,:nocca]) * 2
    im0b[noccb:,:noccb]+= numpy.einsum('ba,bi->ai', dvvb, fockbmo[noccb:,:noccb]) * 2
    im0a[nocca:,:nocca]+= numpy.einsum('aj,ji->ai', z1a, fockamo[:nocca,:nocca]) 
    im0b[noccb:,:noccb]+= numpy.einsum('aj,ji->ai', z1b, fockbmo[:noccb,:noccb]) 
    im0b[nc:(nc+no),:nc]+= numpy.einsum('bt,bi->ti', zvo, fockavc) 
    
    im0 = im0a + im0b
    #additional terms
    #CC block
    im0[:nc,:nc]+= numpy.einsum('ik,jk->ij',xtcc,dvkmo[:nc,:nc])

    #VV block
    im0[(nc+no):,(nc+no):]+= numpy.einsum('ac,bc->ab',xtvv,dvkmo[(nc+no):,(nc+no):])
    #OO block
    im0[nc:(nc+no),nc:(nc+no)]+= numpy.einsum('tk,kl,lu->tu',orbo.T, dvk[1],orbo) / 2
    #OC block
    #im0[nc:(nc+no),:nc]+= numpy.einsum('tk,kl,li->ti',orbo.T,dvk[1],orbc)
    #im0[:nc,nc:(nc+no)]+= (numpy.einsum('tk,kl,li->ti',orbo.T,dvk[1],orbc) / 2).T
    im0[:nc,nc:(nc+no)]+= numpy.einsum('ik,tk->it',xtcc,dvkmo[nc:(nc+no),:nc]) * 2
    #VO block
    im0[(nc+no):,nc:(nc+no)]+= numpy.einsum('ab,tb->at',xtvv,dvkmo[nc:(nc+no),(nc+no):]) * 2
    #im0[nc:(nc+no),(nc+no):]+= numpy.einsum('ab,tb->at',xtvv,dvkmo[nc:(nc+no),(nc+no):]).T 
    #im0[nc:(nc+no),(nc+no):]+= numpy.einsum('tk,kl,la->ta',orbo.T,dvk[1],orbv)
    #VC block
    #im0[(nc+no):,:nc]+= numpy.einsum('ab,ib->ai', xtvv, dvkmo[:nc,(nc+no):]) * 2
    #im0[:nc,(nc+no):]+= numpy.einsum('ab,ib->ai', xtvv, dvkmo[:nc,(nc+no):]).T
    im0[:nc,(nc+no):]+= numpy.einsum('ik,ak->ia',xtcc,dvkmo[(nc+no):,:nc]) * 2

    

    im0 = numpy.einsum('kp,pq,ql->kl',mo_coeff, im0 ,mo_coeff.T)

    

    #(T+Z^S)(ao)
    dmz1dooa = (z1ao[0] + z1ao[0].T)/2 + dmzooa
    dmz1doob = (z1ao[1] + z1ao[1].T)/2 + dmzoob

    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)                     #S(ao)


    vj, vk = td_grad.get_jk(mol, (oo0a, dmz1dooa, dmSa,dmAa,
                                  oo0b, dmz1doob, dmSb,dmAb))       
    vj = vj.reshape(2,4,3,nao,nao)
    vk = vk.reshape(2,4,3,nao,nao)
    vhf1a,vhf1b = vj[0] + vj[1] - vk
    time1 = log.timer('2e AO integral derivatives', *time1)

    d1vk = td_grad.get_k(mol,(xtao,oo0S))
    d1vk = d1vk.reshape(2,3,nao,nao)

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
        de[k] += numpy.einsum('xij,ij->x', vhf1a[2,:,p0:p1], dmSa[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ji->x', vhf1a[2,:,p0:p1], dmSa[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1b[2,:,p0:p1], dmSb[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ji->x', vhf1b[2,:,p0:p1], dmSb[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1a[3,:,p0:p1], dmAa[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ji->x', vhf1a[3,:,p0:p1], dmAa[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1b[3,:,p0:p1], dmAb[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ji->x', vhf1b[3,:,p0:p1], dmAb[:,p0:p1]) * 2
        #S*W: -W_ij*S_(\nabla i)j - W_ij*S_i(\nabla j)
        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1]) 
        de[k] -= numpy.einsum('xpq,qp->x', s1[:,p0:p1], im0[:,p0:p1]) 
        #additonal terms
        de[k] += numpy.einsum('xpq,pq->x', d1vk[0,:,p0:p1], oo0S[p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', d1vk[0,:,p0:p1], oo0S[:,p0:p1])
        de[k] += numpy.einsum('xpq,pq->x', d1vk[1,:,p0:p1], xtao[p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', d1vk[1,:,p0:p1], xtao[:,p0:p1])


  

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


class XTDHF(XCIS):
    def __init__(self, mf):
        super().__init__(mf.mol,mf,             
            nstates=1,    
            savedata=False,
            basis = 'orbital',
            so2st = True)
        self.frozen = None

    
    def nuc_grad_method(self):
        return MyGradients(self)


from pyscf import scf
mol = gto.M(atom=''' N 0. 0. 0.; 
            N 0. 0. 1.10''',charge = 1 , spin=1 ,basis = '6-31g**')
mf = scf.ROHF(mol)
mf.conv_tol = 1e-12
mf.max_cycle = 1000
mf.kernel()


td = XTDHF(mf)  
td.conv_tol = 1e-10
td.max_cycle = 200
td.kernel()


my_grad = td.nuc_grad_method().kernel()






from functools import reduce
import numpy as np
from pyscf import gto, scf, lib
from pyscf.lib import logger

# ======================== 核心：不换算激发能的ROHF-TDA总能量计算 ========================
def compute_rohf_tda_total_energy_no_convert(mol_coords, mol_kwargs, state=1):
    atom = []
    for i, (sym, _) in enumerate(mol_kwargs['atom']):
        x, y, z = mol_coords[i]
        atom.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    mol = gto.M(
        atom='\n'.join(atom),
        basis=mol_kwargs['basis'],
        charge=mol_kwargs['charge'],
        spin=mol_kwargs['spin'],
        symmetry=False,
        verbose=0,
        max_memory=mol_kwargs.get('max_memory', 4000)
    )
    
    
    # 纯ROHF基态
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 1000
    mf.kernel()
    
    # 纯ROHF-TDA激发态
    td = XTDHF(mf)
    td.conv_tol = 1e-10
    td.max_cycle = 200
    td.frozen = None
    td.kernel()
    
  
    excitation_energy = td.e[state-1]  
    E_excited = mf.e_tot + excitation_energy
    
    return E_excited

# ======================== 有限差分梯度主函数（无换算） ========================
def fd_rohf_tda_gradient_no_convert(mol_kwargs, state=1, h=1e-4, verbose=logger.INFO):
    log = logger.new_logger(None, verbose)
    log.info("="*60)
    log.info("ROHF-TDA 有限差分梯度（激发能直接用td.e，无换算）")
    log.info(f"差分步长 h = {h:.1e} Å")
    log.info(f"目标激发态 = {state}")
    log.info("="*60)
    
    # 初始坐标
    natm = len(mol_kwargs['atom'])
    coords_ref = []
    for sym, (x, y, z) in mol_kwargs['atom']:
        coords_ref.append([x, y, z])
    coords_ref = np.array(coords_ref, dtype=np.float64)
    
    # 参考能量
    log.info("计算参考结构的激发态总能量...")
    E_ref = compute_rohf_tda_total_energy_no_convert(coords_ref, mol_kwargs, state)
    log.info(f"参考激发态总能量 = {E_ref:.12f} a.u.")
    
    # 梯度初始化
    grad_fd = np.zeros((natm, 3), dtype=np.float64)
    
    # 单位转换：Å → a.u.（仅步长转换，能量无换算）
    au2ang = 1.8897261246
    h_au = h * au2ang
    
    # 逐原子/逐方向差分
    log.info("开始逐原子/逐方向计算差分...")
    for i in range(natm):
        for α in range(3):
            # +h 扰动
            coords_plus = coords_ref.copy()
            coords_plus[i, α] += h
            E_plus = compute_rohf_tda_total_energy_no_convert(coords_plus, mol_kwargs, state)
            
            # -h 扰动
            coords_minus = coords_ref.copy()
            coords_minus[i, α] -= h
            E_minus = compute_rohf_tda_total_energy_no_convert(coords_minus, mol_kwargs, state)
            
            # 中心差分（无额外换算）
            grad_fd[i, α] = (E_plus - E_minus) / (2 * h_au)
            
            log.info(f"原子 {i} ({mol_kwargs['atom'][i][0]}) - 方向 {['x','y','z'][α]}:")
            log.info(f"  E(+h) = {E_plus:.12f} a.u., E(-h) = {E_minus:.12f} a.u.")
            log.info(f"  梯度 = {grad_fd[i, α]:.10f} a.u.")
    
    log.info("="*60)
    log.info("有限差分梯度计算完成！")
    log.info("="*60)
    return grad_fd, E_ref

# ======================== 对比函数（无修改） ========================
def compare_gradient_no_convert(grad_analytic, grad_fd, mol_kwargs, verbose=logger.INFO):
    log = logger.new_logger(None, verbose)
    log.info("\n" + "="*60)
    log.info("对比：解析梯度 vs 有限差分梯度（激发能无换算）")
    log.info("="*60)
    
    log.info("梯度结果对比（单位：a.u.）：")
    headers = ["原子", "方向", "解析梯度", "有限差分梯度", "绝对误差"]
    log.info(f"{headers[0]:<4}  {headers[1]:<4}  {headers[2]:<12}  {headers[3]:<12}  {headers[4]:<12}")
    log.info("-"*60)
    
    errors = []
    for i in range(len(mol_kwargs['atom'])):
        sym = mol_kwargs['atom'][i][0]
        for α, dir_sym in enumerate(['x','y','z']):
            g_a = grad_analytic[i, α]
            g_f = grad_fd[i, α]
            err = abs(g_a - g_f)
            errors.append(err)
            log.info(f"{sym:<4}  {dir_sym:<4}  {g_a:.10f}    {g_f:.10f}    {err:.10f}")
    
    errors = np.array(errors)
    max_err = np.max(errors)
    rms_err = np.sqrt(np.mean(errors**2))
    
    log.info("-"*60)
    log.info(f"最大绝对误差 = {max_err:.2e} a.u.")
    log.info(f"均方根误差（RMS） = {rms_err:.2e} a.u.")
    log.info("="*60)
    
    if max_err < 1e-4:
        log.info("✅ 完美匹配！误差<1e-4 a.u.，解析梯度定义与有限差分一致！")
    else:
        log.info(f"✅ 对比完成！当前最大误差 = {max_err:.2e} a.u.")
    
    return {
        'max_abs_error': max_err,
        'rms_error': rms_err,
        'errors': errors
    }

# ======================== 测试用例（直接运行） ========================
if __name__ == "__main__":
    # 1. 分子参数（和你的计算完全一致）
    mol_kwargs = {
        'atom': [
            ('N', (0.000000, 0.000000, 0.000000)),
            ('N', (0.000000, 0.000000, 1.100000)), 
            ],
        'basis': '6-31g**',
        'charge': 1,        
        'spin': 1
              
    }
    
    # 2. 计算有限差分梯度（无换算激发能）
    grad_fd, E_ref = fd_rohf_tda_gradient_no_convert(
        mol_kwargs=mol_kwargs,
        state=1,
        h=1e-04,
        verbose=logger.INFO
    )
    
    # 3. 你的解析梯度结果（替换为实际值）
    grad_analytic = np.array([
        [0.0000000000, 0.0000000000,  0.1244841462],
        [0.0000000000, 0.0000000000, -0.1244841462],
    ])
    
    # 4. 对比
    error_stats = compare_gradient_no_convert(grad_analytic, grad_fd, mol_kwargs)
