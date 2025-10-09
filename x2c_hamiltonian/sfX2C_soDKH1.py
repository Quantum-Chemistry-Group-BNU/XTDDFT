#
# Description: The sfX2C_soDKH1 Hamiltonian
#
# OriginalTheory: JCP 137, 154114 (2012)
#                 JCP 141, 054111 (2014)
#
# Implementation: Mol. Phys. 111 (24), 3741-3755
#.                see Eqs(34)-(53) for details of formulae.
#
# Implemented by Zhendong Li (zhendongli2008@gmail.com)
#
# Functions:
#
# def inv12(s):
# def sfx2c1e(t, v, w, s, c):
# def get_p(dm,x,rp):
# def get_wso(mol):
# def get_kint(mol):
# def get_hso1e(wso,x,rp):
# def get_fso2e(kint,x,rp,pLL,pLS,pSS):
# def get_soDKH1_somf(myhf,mol,c,iop='x2c',debug=False):
# 
import numpy, time
import scipy.linalg
from functools import reduce

def inv12(s):
    '''
    Return s^{-1/2}
    '''
    e,v = scipy.linalg.eigh(s)
    return reduce(numpy.dot,(v,numpy.diag(1/numpy.sqrt(e)),v.T))

def sfx2c1e(t, v, w, s, c):
    '''
    Generate hso1e
    here
        t: T kinetic energy term
        v: V Nuclear attraction term
        w: Wsf p·Vne·p
        s: S Overlap (Metric)
        c: speed of light
    '''
    nao = s.shape[0]
    n2 = nao * 2
    # costruct h and m
    h = numpy.zeros((n2,n2), dtype=v.dtype)
    m = numpy.zeros((n2,n2), dtype=v.dtype)
    # ref.(42) of JCP 137, 154114 (2012)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)
    e, a = scipy.linalg.eigh(h, m)
    cl = a[:nao,nao:]
    cs = a[nao:,nao:]
    x = cs.dot(cl.T.dot(numpy.linalg.inv(cl.dot(cl.T)))) # ref.(40)
    stilde = s + x.T.dot(m[nao:,nao:].dot(x)) # ref.(39) and here m[nao:,nao:] = t * (.5/c**2) is t·a^2/2
    sih = inv12(s) # s^{-1/2}
    sh = numpy.linalg.inv(inv12(s)) # s^{1/2}
    rp = sih.dot(inv12(sih.dot(stilde.dot(sih))).dot(sh)) # ref.(38)
    # ref.(48) of JCP 137, 154114 (2012) 
    l1e = h[:nao,:nao] + h[:nao,nao:].dot(x) + x.T.dot(h[nao:,:nao]) + x.T.dot(h[nao:,nao:].dot(x))
    h1e = rp.T.dot(l1e.dot(rp))
    return x,rp,h1e

def get_p(dm,x,rp):
    '''
    Generate spin-averaged dm P
    ref.(50):
        PLL = R+0@P@(R+0)^dagger
        PLS = PLL@X0^dagger
        PSS = X0@PLL@X0^dagger
    here
        dm: P
        x: X0
        rp: R+0
    '''
    pLL = rp.dot(dm.dot(rp.T))
    pLS = pLL.dot(x.T)
    pSS = x.dot(pLL.dot(x.T))
    return pLL,pLS,pSS

def get_wso(mol):
    '''
    Generate Wso
    ref.(43)
        Wso = sigma_l w^l
        w^l = varepslion(lmn) <mu_m|Vne|nu_n>
    '''
    nb = mol.nao_nr()
    wso = numpy.zeros((3,nb,nb))
    for iatom in range(mol.natm):
        zA  = mol.atom_charge(iatom)
        xyz = mol.atom_coord(iatom)
        mol.set_rinv_orig(xyz)
        wso += -zA*mol.intor('cint1e_prinvxp_sph', 3) # sign due to integration by part
    return wso

def get_kint(mol,debug=0):
    '''
    Generate two-electron integrals K(l)mu,nu;kappa,lambda
    where mu, nu, kappa, lambda index orbitals
    ref.(49)
        K(l)mu,nu;lappa,lambda = varepsilon(lmn)(mu_m,nu|kappa_n,lambda)
        mu_m = \partial_m mu
    where (mu_m,nu|kappa_n,lambda) come from `int2e_ip1ip2_sph` libcint
    '''
    nb = mol.nao_nr()
    np = nb*nb # number of excitations
    nq = np*np # two single excitation
    ddint = mol.intor('int2e_ip1ip2_sph').reshape(3,3,nq)
    kint = numpy.zeros((3,nq))
    kint[0] = ddint[1,2] - ddint[2,1] # Lx = y p_z - z p_y
    kint[1] = ddint[2,0] - ddint[0,2] # Ly = z p_x - x p_z
    kint[2] = ddint[0,1] - ddint[1,0] # Lz = x p_y - y p_x
    kint = kint.reshape(3,nb,nb,nb,nb)
    if debug:
        for n in range(0,kint.shape[0]):
            print(f"axis-{n}, kint+kint.T={numpy.linalg.norm(kint[n]+kint[n].transpose(3,4,1,2))}")
    return kint

def get_hso1e(wso,x,rp):
    '''
    Generate hso1e
    ref.(41)
        hso1e = a4 * (R+0)^dagger @ X0^dagger @ Wso @ X0 @ R+0
    here
        wso: Wso = ref.(43)
        x: X0
        rp: R+0
    '''
    nb = x.shape[0]
    hso1e = numpy.zeros((3,nb,nb))
    for ic in range(3):
        hso1e[ic] = reduce(numpy.dot,(rp.T,x.T,wso[ic],x,rp))
    return hso1e

def get_fso2e(kint,x,rp,pLL,pLS,pSS):
    '''
    Generate fso2e
    ref.(42)
        fso2e = a4*(R+0)^dagger [GsoLL + GsoLS + X0^dagger@GsoSL + X0^dagger@GsoSS@X0] R+0
    here
        kint: K(l)mu,nu,kappa,lambda
        x: X0
        rp: R+0
        pLL, pLS, pSS: spin-averaged density matrices
    '''
    nb = x.shape[0]
    fso2e = numpy.zeros((3,nb,nb))
    for ic in range(3):
        gsoLL  = -numpy.einsum('lmkn,lk->mn',kint[ic],pSS)*2.

        gsoLS  = -numpy.einsum('mlkn,lk->mn',kint[ic],pLS)
        gsoLS -=  numpy.einsum('lmkn,lk->mn',kint[ic],pLS)

        gsoSS  = -numpy.einsum('mnkl,lk->mn',kint[ic],pLL)*2.
        gsoSS -=  numpy.einsum('mnlk,lk->mn',kint[ic],pLL)*2.
        gsoSS +=  numpy.einsum('mlnk,lk->mn',kint[ic],pLL)*2.

        fso2e[ic] = gsoLL + gsoLS.dot(x) + x.T.dot(-gsoLS.T) + x.T.dot(gsoSS.dot(x))
        fso2e[ic] = reduce(numpy.dot,(rp.T,fso2e[ic],rp))
    return fso2e

def get_soDKH1_somf(myhf,mol,c,iop='x2c',include_mf2e=True,debug=False):
    '''
    Generate Hso
    here
        myhf: scf
        mol: mole
        c: speed of light
        iop: x2c or bp
    return
        Vso in ao basis
    '''
    print(f"Begin to generate Vso")
    time0 = time.time()
    xmol,contr_coeff = myhf.with_x2c.get_xmol(mol) # change basis
    print(f"get_soDKH1_somf with iop={iop}, (np,nc)={contr_coeff.shape}")
    nb = contr_coeff.shape[0]
    nc = contr_coeff.shape[1]
    if iop == 'x2c':
        t = xmol.intor_symmetric('int1e_kin') # kinetic energy term
        v = xmol.intor_symmetric('int1e_nuc') # Nuclear attraction term
        s = xmol.intor_symmetric('int1e_ovlp') # Overlap (Metric)
        w = xmol.intor_symmetric('int1e_pnucp') # p·Vne·p
        x,rp,h1e = sfx2c1e(t, v, w, s, c)
    elif iop == 'bp':
        x = numpy.identity(nb)
        rp = numpy.identity(nb)
    else:
        raise ValueError(f"iop={iop} not in {'x2c','bp'}")
    dm = myhf.make_rdm1()/2. # Here the DM matrix element is 1 or 1/2 or 0, not 2, 1, 0
    # Spin-Averaged for ROHF or UHF
    if len(dm.shape)==3: dm = (dm[0]+dm[1])/2.0
    dm = reduce(numpy.dot,(contr_coeff,dm,contr_coeff.T))
    pLL,pLS,pSS = get_p(dm,x,rp) # ref.(50)
    wso = get_wso(xmol)
    kint = get_kint(xmol)
    if debug:
        for ic in range(3):
            print(f"ic={ic, numpy.linalg.norm(kint[ic]+kint[ic].transpose(2,3,0,1))}")
    # Make combination Hsd = hso1e + fso2e ref.(36)
    a4 = 0.25/c**2 # alpha^2/4
    hso1e = get_hso1e(wso,x,rp)
    VsoDKH1 = a4*(hso1e)
    if include_mf2e:
        fso2e = get_fso2e(kint,x,rp,pLL,pLS,pSS)
        VsoDKH1 += a4*(fso2e)

    if debug:
        for ic in range(3):
            tmp = hso1e[ic]
            print(f"{ic} hso1e  norm={numpy.linalg.norm(tmp):.12e}, "
                f"Norm(tmp+tmp.T)={numpy.linalg.norm(tmp + tmp.T):.12e}")
            if include_mf2e:
                tmp = fso2e[ic]
                print(f"{ic} fso2e  norm={numpy.linalg.norm(tmp):.12e}, "
                    f"Norm(tmp+tmp.T)={numpy.linalg.norm(tmp + tmp.T):.12e}")
                tmp = hso1e[ic] + fso2e[ic]
                print(f"{ic} vso2e  norm={numpy.linalg.norm(tmp):.12e}, "
                    f"Norm(tmp+tmp.T)={numpy.linalg.norm(tmp + tmp.T):.12e}")

    VsoDKH1contr = numpy.zeros((3,nc,nc))
    for ic in range(3):
        VsoDKH1contr[ic] = reduce(numpy.dot,(contr_coeff.T,VsoDKH1[ic],contr_coeff))
    if debug:
        for ic in range(3):
            tmp = VsoDKH1contr[ic]
            print(f"{ic} VsoDKH1contr  norm={numpy.linalg.norm(tmp):.12e}, "
                f"Norm(tmp+tmp.T)={numpy.linalg.norm(tmp + tmp.T):.12e}")
    time1 = time.time()
    print(f"End of generating Vso, cost time={time1-time0:.2f}s")
    return VsoDKH1contr