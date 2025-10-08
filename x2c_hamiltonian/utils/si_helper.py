import math, numpy

from .utils import read_ns

def w(S,M,Sprime,Mprime):
    from sympy.physics.wigner import wigner_3j
    if abs(wigner_3j(S,1,Sprime,-S,S-Sprime,Sprime).doit().evalf())<1e-3:
        return 0.
    else:
        return (-1)**(S-M) * float((wigner_3j(S,1,Sprime,-M,M-Mprime,Mprime) / wigner_3j(S,1,Sprime,-S,S-Sprime,Sprime)).evalf())

def generate_hm(mol,S,Vso):
    '''
    here
        mol: mole
        S: Si
        Vso: Hsd
    return
        norb: #spatial orbitals
        nc: #Core spatial orbitals
        no: #Open-shell spatial orbitals
        nv: #Virtual spatial orbitals
        and corresponding slice (slc,slo,slv)
    '''
    print(f"Generate hm matrix elements --")
    norb, nc,no,nv, slc,slo,slv = read_ns(mol,S)
    ns = nc*nv
    h = {
        -1: numpy.zeros((norb,norb), dtype=numpy.complex128),
         0: numpy.zeros((norb,norb), dtype=numpy.complex128),
         1: numpy.zeros((norb,norb), dtype=numpy.complex128),
    }
    h[1]  +=  1j*Vso[0,:,:] - Vso[1,:,:]
    h[0]  +=  1j*Vso[2,:,:]*math.sqrt(2)
    h[-1] += -1j*Vso[0,:,:] - Vso[1,:,:]
    
    Scond = abs(S) > 1e-3

    hm = {i: {m: None for m in (-1, 0, 1)} for i in range(1, 22)}
    for m in (-1, 0, 1):
        # --- 6
        hm[1][m]  =  numpy.zeros((1,1), dtype=numpy.complex128)
        hm[2][m]  =  numpy.zeros((nv,nc), dtype=numpy.complex128)
        if Scond:
            hm[3][m]  =  numpy.zeros((no,nc), dtype=numpy.complex128)
            hm[3][m] += -numpy.einsum('jv->vj',h[m][slc,slo])/math.sqrt(2)
            hm[4][m]  =  numpy.zeros((nv,no), dtype=numpy.complex128)
            hm[4][m] +=  numpy.einsum('vb->bv',h[m][slo,slv])/math.sqrt(2)
            hm[5][m]  =  numpy.zeros((nv,nc), dtype=numpy.complex128)
            hm[5][m] += -numpy.einsum('jb->bj',h[m][slc,slv])*math.sqrt(S/(1+S))
        hm[6][m]  =  numpy.zeros((nv,nc), dtype=numpy.complex128)
        hm[6][m] += -numpy.einsum('jb->bj',h[m][slc,slv])
        # --- 5
        hm[7][m]  =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
        if Scond:
            hm[8][m]  =  numpy.zeros((nv,nc,no,nc), dtype=numpy.complex128)
            hm[8][m] += -numpy.einsum('av,ij->aivj',h[m][slv,slo],numpy.eye(nc,nc))/2.
            hm[9][m]  =  numpy.zeros((nv,nc,nv,no), dtype=numpy.complex128)
            hm[9][m] += -numpy.einsum('vi,ab->aibv',h[m][slo,slc],numpy.eye(nv,nv))/2.
            hm[10][m] =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
            hm[10][m]+= -numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))*math.sqrt(S/(2*(1+S)))
            hm[10][m]-= -numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))*math.sqrt(S/(2*(1+S)))
        hm[11][m] =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
        hm[11][m]+=  numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))/math.sqrt(2)
        hm[11][m]+= -numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))/math.sqrt(2)
        # --- 4
        if Scond:
            hm[12][m] =  numpy.zeros((no,nc,no,nc), dtype=numpy.complex128)
            hm[12][m]+= -numpy.einsum('uv,ij->uivj',h[m][slo,slo],numpy.eye(nc,nc))/math.sqrt(2)
            hm[12][m]-= -numpy.einsum('ji,uv->uivj',h[m][slc,slc],numpy.eye(no,no))/math.sqrt(2)
            hm[13][m] =  numpy.zeros((no,nc,nv,no), dtype=numpy.complex128)
            hm[14][m] =  numpy.zeros((no,nc,nv,nc), dtype=numpy.complex128)
            hm[14][m]+=  numpy.einsum('ub,ij->uibj',h[m][slo,slv],numpy.eye(nc,nc))*0.5*(1-S)/math.sqrt(S*(S+1))
            hm[15][m] =  numpy.zeros((no,nc,nv,nc), dtype=numpy.complex128)
            hm[15][m]+= -numpy.einsum('ub,ij->uibj',h[m][slo,slv],numpy.eye(nc,nc))
        # --- 3
            hm[16][m] =  numpy.zeros((nv,no,nv,no), dtype=numpy.complex128)
            hm[16][m]+=  numpy.einsum('ab,uv->aubv',h[m][slv,slv],numpy.eye(no,no))
            hm[17][m] =  numpy.zeros((nv,no,nv,nc), dtype=numpy.complex128)
            hm[17][m]+=  numpy.einsum('ju,ab->aubj',h[m][slc,slo],numpy.eye(nv,nv))*0.5*(S-1)/math.sqrt(S*(S+1))
            hm[18][m] =  numpy.zeros((nv,no,nv,nc), dtype=numpy.complex128)
            hm[18][m]+=  numpy.einsum('ju,ab->aubj',h[m][slc,slo],numpy.eye(nv,nv))
        # --- 2
            hm[19][m] =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
            hm[19][m]+=  numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))/((1+S)*math.sqrt(2))
            hm[19][m]+=  numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))/((1+S)*math.sqrt(2))
            hm[20][m] =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
            hm[20][m]+= -numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))*math.sqrt(S/(2*(S+1)))
            hm[20][m]+= -numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))*math.sqrt(S/(2*(S+1)))
        # --- 1
        hm[21][m] =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
        hm[21][m]+=  numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))/math.sqrt(2)
        hm[21][m]+=  numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))/math.sqrt(2)
    print(f"End of generating hm matrix elements --")
    return hm

def sidriver(mf,S,Vso,gs=1,nstates=None):
    si0driver(mf,S,Vso,gs,nstates)

def si0driver(mf,S,Vso,gs=1,nstates=None):
    mol = mf.mol
    hm = generate_hm(mol,S,Vso)
    norb, nc,no,nv, slc,slo,slv = read_ns(mol,S)
    ns = nc*nv
    if nstates == None:
        nstates = (ns,ns)
    from TDA import TDA

    tda_s = TDA(mol, mf, singlet=1, nstates=nstates[0])
    e_s, _, _, xs = tda_s.kernel()
    # tda_s.analyze()

    tda_t = TDA(mol, mf, singlet=0, nstates=nstates[1])
    e_t, _, _, xt0 = tda_t.kernel()
    # tda_t.analyze()
    xgs = numpy.ones((1,1))
    xt_1 = xt1 = xt0.copy()

    dimGS = gs
    dimSS = nstates[0]
    dimTT = nstates[1]
    dimSO = dimGS+dimSS+dimTT*3

    slGS = slice(0,dimGS,1)
    slS = slice(dimGS,dimGS+dimSS,1)
    slT_1 = slice(dimGS+dimSS,dimGS+dimSS+dimTT,1)
    slT0 = slice(dimGS+dimSS+dimTT,dimGS+dimSS+dimTT*2,1)
    slT1 = slice(dimGS+dimSS+dimTT*2,dimGS+dimSS+dimTT*3,1)
    
    # Blockwise construct hso
    hso = numpy.zeros((dimSO,dimSO,),dtype=numpy.complex128)
    # line1
    hso[slGS,slGS]  = xgs.T.conjugate()@hm[1][0]*w(0,0,0,0)@xgs
    hso[slGS,slS]   = xgs.T.conjugate()@numpy.einsum('bj->jb',hm[2][0]).reshape(1,ns)*w(0,0,0,0)@xs
    hso[slGS,slT_1] = xgs.T.conjugate()@numpy.einsum('bj->jb',hm[6][-1]).reshape(1,ns)*w(0,0,1,-1)@xt_1
    hso[slGS,slT0]  = xgs.T.conjugate()@numpy.einsum('bj->jb',hm[6][0]).reshape(1,ns)*w(0,0,1,0)@xt0
    hso[slGS,slT1]  = xgs.T.conjugate()@numpy.einsum('bj->jb',hm[6][1]).reshape(1,ns)*w(0,0,1,1)@xt1

    # line2
    hso[slS,slS]   = xs.T.conjugate()@numpy.einsum('aibj->iajb',hm[7][0]).reshape(ns,ns)*w(0,0,0,0)@xs
    hso[slS,slT_1] = xs.T.conjugate()@numpy.einsum('aibj->iajb',hm[11][-1]).reshape(ns,ns)*w(0,0,1,-1)@xt_1
    hso[slS,slT0]  = xs.T.conjugate()@numpy.einsum('aibj->iajb',hm[11][0]).reshape(ns,ns)*w(0,0,1,0)@xt0
    hso[slS,slT1]  = xs.T.conjugate()@numpy.einsum('aibj->iajb',hm[11][1]).reshape(ns,ns)*w(0,0,1,1)@xt1

    # line3
    hso[slT_1,slT_1] = xt_1.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][0]).reshape(ns,ns)*w(1,-1,1,-1)@xt_1
    hso[slT_1,slT0]  = xt_1.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][1]).reshape(ns,ns)*w(1,-1,1,0)@xt0
    # hso[slT_1,slT1]

    # line4
    hso[slT0,slT0]  = xt0.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][0]).reshape(ns,ns)*w(1,0,1,0)@xt0
    hso[slT0,slT1]  = xt0.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][1]).reshape(ns,ns)*w(1,0,1,1)@xt1

    # line5
    hso[slT1,slT1] = xt1.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][0]).reshape(ns,ns)*w(1,1,1,1)@xt1

    # Symmetrized
    hso = numpy.triu(hso) + numpy.triu(hso, 1).T.conjugate()

    Omega = numpy.zeros((dimSO,))
    Omega[slGS] = 0.0
    Omega[slS] = e_s.copy()
    Omega[slT_1] = Omega[slT0] = Omega[slT1] = e_t.copy()

    ha2eV = 27.2113834  # orca transformation
    Omega = numpy.diag(Omega)/ha2eV

    print(f"Begin to diag Omega+hso --{hso.shape}--")
    import scipy
    e_so, v_so = scipy.linalg.eigh(hso + Omega, driver='evd')
    print(f"RKS E (in Hatree) = {mf.e_tot}")
    print(f"The E (in eV), shape={e_so.shape} --")
    print(numpy.sort(numpy.diag(Omega))[:20] * ha2eV)
    print(f"The E_soc (in eV) --")
    print(e_so[:20] * ha2eV)
    breakpoint()

def si1driver(mf,S,Vso):
    print(f"Perform SI calculation with Si={S}")