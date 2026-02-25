import math, numpy, sys, time
sys.path.append('./')
sys.path.append('./../')
sys.path.append('./../../')

from .fso_utils import read_ns
from utils import unit

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
            hm[10][m]+=  numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))*math.sqrt(S/(2*(1+S)))
        hm[11][m] =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
        hm[11][m]+=  numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))/math.sqrt(2)
        hm[11][m]+= -numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))/math.sqrt(2)
        # --- 4
        if Scond:
            hm[12][m] =  numpy.zeros((no,nc,no,nc), dtype=numpy.complex128)
            hm[12][m]+= -numpy.einsum('uv,ij->uivj',h[m][slo,slo],numpy.eye(nc,nc))/math.sqrt(2)
            hm[12][m]+=  numpy.einsum('ji,uv->uivj',h[m][slc,slc],numpy.eye(no,no))/math.sqrt(2)
            hm[13][m] =  numpy.zeros((no,nc,nv,no), dtype=numpy.complex128)
            hm[14][m] =  numpy.zeros((no,nc,nv,nc), dtype=numpy.complex128)
            hm[14][m]+=  numpy.einsum('ub,ij->uibj',h[m][slo,slv],numpy.eye(nc,nc))*0.5*(1-S)/math.sqrt(S*(S+1))
            hm[15][m] =  numpy.zeros((no,nc,nv,nc), dtype=numpy.complex128)
            hm[15][m]+= -numpy.einsum('ub,ij->uibj',h[m][slo,slv],numpy.eye(nc,nc))
        # --- 3
            hm[16][m] =  numpy.zeros((nv,no,nv,no), dtype=numpy.complex128)
            hm[16][m]+=  numpy.einsum('ab,uv->aubv',h[m][slv,slv],numpy.eye(no,no))/math.sqrt(2)
            hm[16][m]+= -numpy.einsum('vu,ab->aubv',h[m][slo,slo],numpy.eye(nv,nv))/math.sqrt(2)
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

def gmap_hm():
    '''
    Generate the map from <bra|hm|ket> to the number of hm 
    (ref. Mol. Phys. 111 (24), 3741-3755)
    the key of map_hm is [bra] [ket]
    '''
    # 6
    map_hm = {}
    map_hm['GS'] = {}
    map_hm['GS']['GS'] = 1
    map_hm['GS']['S_CV(0)'] = 2
    map_hm['GS']['S_CO(0)'] = 3
    map_hm['GS']['S_OV(0)'] = 4
    map_hm['GS']['S_CV(1)'] = 5
    map_hm['GS']['S1_CV(1)'] = 6
    # 5
    map_hm['S_CV(0)'] = {}
    map_hm['S_CV(0)']['S_CV(0)'] = 7
    map_hm['S_CV(0)']['S_CO(0)'] = 8
    map_hm['S_CV(0)']['S_OV(0)'] = 9
    map_hm['S_CV(0)']['S_CV(1)'] = 10
    map_hm['S_CV(0)']['S1_CV(1)'] = 11
    # 4
    map_hm['S_CO(0)'] = {}
    map_hm['S_CO(0)']['S_CO(0)'] = 12
    map_hm['S_CO(0)']['S_OV(0)'] = 13
    map_hm['S_CO(0)']['S_CV(1)'] = 14
    map_hm['S_CO(0)']['S1_CV(1)'] = 15
    # 3
    map_hm['S_OV(0)'] = {}
    map_hm['S_OV(0)']['S_OV(0)'] = 16
    map_hm['S_OV(0)']['S_CV(1)'] = 17
    map_hm['S_OV(0)']['S1_CV(1)'] = 18
    # 2
    map_hm['S_CV(1)'] = {}
    map_hm['S_CV(1)']['S_CV(1)'] = 19
    map_hm['S_CV(1)']['S1_CV(1)'] = 20
    # 1
    map_hm['S1_CV(1)'] = {}
    map_hm['S1_CV(1)']['S1_CV(1)'] = 21
    return map_hm

def symm_matrix(h):
    return numpy.triu(h) + numpy.triu(h, 1).T.conjugate()

def sidriver(mf,S,Vso,gs=1,nstates=None,analyze=0):
    print(f"Perform SI calculation with Si={S}")
    time0 = time.time()
    if abs(S) < 1e-3:
        si1driver(mf,S,Vso,gs,nstates,analyze)
    elif abs(S-1/2) < 1e-3:
        si2driver(mf,S,Vso,gs,nstates,analyze)
    else:
        print(f"Please check out spin Si={S}, this helper could only support Si=0,1/2.")
        print(f"The general method is the corresponding class.")
    time1 = time.time()
    print(f"End SI calculation, cost time {time1-time0:.2f}s")

def si1driver(mf,S,Vso,gs=1,nstates=None,analyze=0):
    mol = mf.mol
    hm = generate_hm(mol,S,Vso)
    norb, nc,no,nv, slc,slo,slv = read_ns(mol,S)
    ns = nc*nv
    if nstates == None:
        nstates = (ns,ns)
    else:
        assert len(nstates) == 2
    from xtddft.TDA import TDA

    tda_s = TDA(mol, mf, singlet=1, nstates=nstates[0])
    e_s, _, _, xs = tda_s.kernel()
    if analyze:
        tda_s.analyze()

    tda_t = TDA(mol, mf, singlet=0, nstates=nstates[1])
    e_t, _, _, xt0 = tda_t.kernel()
    if analyze:
        tda_t.analyze()
    
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
    hso = symm_matrix(hso)

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

def si2driver(mf,S,Vso,gs=1,nstates=[None,None],analyze=0):
    mol = mf.mol
    norb, nc,no,nv, slc,slo,slv = read_ns(mol,S)
    if nstates[0] == None:
        nstates[0] = (nc+no)*nv + nc*(nv+no)
    if nstates[1] == None:
        nstates[1] = nc*nv
    if gs == 1:
        print(f"The ground state is included in SI calculation")

    print(f"{'='*15} State interaction setting {'='*15}")
    # print(f"GS {gs}, state-({S:.2f}) {nstates[0]}, state-({S+1:.2f}) {nstates[1]} are selected")
    print(f"GS {gs}, state-|Si> {nstates[0]}, state-|Si+1> {nstates[1]} are selected")

    from xtddft.XTDA import XTDA
    print(f"{'='*15} Perform XTDA calculation {'='*15}")
    xtda= XTDA(mol,mf,basis='tensor')
    xtda.nstates = nstates[0]
    xtda.kernel()
    ed = xtda.e
    xd = xtda.v # CV(0), CO(0), OV(0), CV(1) by X_TDA
    # breakpoint()
    # use whb's XTDA
    # xtda1= XTDA(mol,mf,so2st=1)
    # xtda1.nstates = nstates[0]
    # xtda1.kernel()
    # xtda1.analyze() # this should be performed trans. so2st basis
    # ed = xtda1.e
    # xd1 = xd1_ = xtda1.v # CV(0), OV(0), CO(0), CV(1) by XTDA
    # # transform to CV(0), CO(0), OV(0), CV(1)
    # xd1[nc*nv:nc*(nv+no)] = xd1_[nc*nv+no*nv:nc*nv+no*nv+nc*no]
    # xd1[nc*(nv+no):nc*(nv+no)+no*nv] = xd1_[nc*nv:nc*nv+no*nv]
    # xd = xd1

    from xtddft.SF_TDA import SF_TDA
    print(f"{'='*15} Perform SF-up-TDA calculation {'='*15}")
    sf_tda = SF_TDA(mf,isf=1,davidson=0)
    sf_tda.nstates = nstates[1]
    eq, xq = sf_tda.kernel()
    sf_tda.analyse()

    print(f"{'='*15} Construst hso matrix {'='*15}")
    dimGS = int(gs)
    dimDD = xd.shape[-1]
    dimQQ = xq.shape[-1]
    dimSO = dimGS*2+dimDD*2+dimQQ*4

    xgs = numpy.ones((1,dimGS))

    slGS_12 = slice(0,dimGS,1)
    slGS12 = slice(dimGS,dimGS*2,1)
    slD_12 = slice(dimGS*2,dimGS*2+dimDD,1)
    slD12 = slice(dimGS*2+dimDD,dimGS*2+dimDD*2,1)

    slQ_32 = slice(dimGS*2+dimDD*2,dimGS*2+dimDD*2+dimQQ*1,1)
    slQ_12 = slice(dimGS*2+dimDD*2+dimQQ*1,dimGS*2+dimDD*2+dimQQ*2,1)
    slQ12 = slice(dimGS*2+dimDD*2+dimQQ*2,dimGS*2+dimDD*2+dimQQ*3,1)
    slQ32 = slice(dimGS*2+dimDD*2+dimQQ*3,dimGS*2+dimDD*2+dimQQ*4,1)
    print(f"slice is {slGS_12,slGS12,slD_12,slD12,slQ_32,slQ_12,slQ12,slQ32}")
    
    # Matrix between GS, D and Q i.e. hm ==> hm block
    dimCV = nc*nv
    dimOV = no*nv
    dimCO = nc*no

    slS_CV0 = slice(0,nc*nv,1)
    slS_CO0 = slice(nc*nv,(nv+no)*nc,1)
    slS_OV0 = slice((nv+no)*nc,(nv+no)*nc+nv*no,1)
    slS_CV1 = slice((nv+no)*nc+nv*no,(nc+no)*nv+nc*(nv+no),1)

    hm = generate_hm(mol,S,Vso)
    map_hm = gmap_hm()

    hmGSGS = {}
    hmGSD = {}
    hmGSQ = {}
    hmDD = {}
    hmDQ = {}
    hmQQ = {}

    for m in (-1, 0, 1):
        # GS-GS
        hmGSGS[m] = hm[map_hm['GS']['GS']][m]
        # GS-D
        hmGSD[m] = numpy.zeros((1,(nc+no)*nv + nc*(nv+no)),dtype=numpy.complex128)
        hmGSD[m][0,slS_CV0] = numpy.einsum('bj->jb',hm[map_hm['GS']['S_CV(0)']][m]).reshape(1,dimCV)
        hmGSD[m][0,slS_CO0] = numpy.einsum('bj->jb',hm[map_hm['GS']['S_CO(0)']][m]).reshape(1,dimCO)
        hmGSD[m][0,slS_OV0] = numpy.einsum('bj->jb',hm[map_hm['GS']['S_OV(0)']][m]).reshape(1,dimOV)
        hmGSD[m][0,slS_CV1] = numpy.einsum('bj->jb',hm[map_hm['GS']['S_CV(1)']][m]).reshape(1,dimCV)
        # GS-Q
        hmGSQ[m] = numpy.einsum('bj->jb',hm[map_hm['GS']['S1_CV(1)']][m]).reshape(1,dimCV)
        # D-D CV(0), CO(0), OV(0), CV(1)
        hmDD[m] = numpy.zeros(((nc+no)*nv + nc*(nv+no),(nc+no)*nv + nc*(nv+no)),dtype=numpy.complex128)
        hmDD[m][slS_CV0,slS_CV0] = numpy.einsum('aibj->iajb',hm[map_hm['S_CV(0)']['S_CV(0)']][m]).reshape(dimCV,dimCV)
        hmDD[m][slS_CV0,slS_CO0] = numpy.einsum('aibj->iajb',hm[map_hm['S_CV(0)']['S_CO(0)']][m]).reshape(dimCV,dimCO)
        hmDD[m][slS_CV0,slS_OV0] = numpy.einsum('aibj->iajb',hm[map_hm['S_CV(0)']['S_OV(0)']][m]).reshape(dimCV,dimOV)
        hmDD[m][slS_CV0,slS_CV1] = numpy.einsum('aibj->iajb',hm[map_hm['S_CV(0)']['S_CV(1)']][m]).reshape(dimCV,dimCV)

        hmDD[m][slS_CO0,slS_CV0] = -numpy.einsum('bjai->iajb',hm[map_hm['S_CV(0)']['S_CO(0)']][m]).reshape(dimCO,dimCV)
        hmDD[m][slS_CO0,slS_CO0] = numpy.einsum('aibj->iajb',hm[map_hm['S_CO(0)']['S_CO(0)']][m]).reshape(dimCO,dimCO)
        hmDD[m][slS_CO0,slS_OV0] = numpy.einsum('aibj->iajb',hm[map_hm['S_CO(0)']['S_OV(0)']][m]).reshape(dimCO,dimOV)
        hmDD[m][slS_CO0,slS_CV1] = numpy.einsum('aibj->iajb',hm[map_hm['S_CO(0)']['S_CV(1)']][m]).reshape(dimCO,dimCV)

        hmDD[m][slS_OV0,slS_CV0] = -numpy.einsum('bjai->iajb',hm[map_hm['S_CV(0)']['S_OV(0)']][m]).reshape(dimOV,dimCV)
        hmDD[m][slS_OV0,slS_CO0] = -numpy.einsum('bjai->iajb',hm[map_hm['S_CO(0)']['S_OV(0)']][m]).reshape(dimOV,dimCO)
        hmDD[m][slS_OV0,slS_OV0] = numpy.einsum('aibj->iajb',hm[map_hm['S_OV(0)']['S_OV(0)']][m]).reshape(dimOV,dimOV)
        hmDD[m][slS_OV0,slS_CV1] = numpy.einsum('aibj->iajb',hm[map_hm['S_OV(0)']['S_CV(1)']][m]).reshape(dimOV,dimCV)

        hmDD[m][slS_CV1,slS_CV0] = -numpy.einsum('bjai->iajb',hm[map_hm['S_CV(0)']['S_CV(1)']][m]).reshape(dimCV,dimCV)
        hmDD[m][slS_CV1,slS_CO0] = -numpy.einsum('bjai->iajb',hm[map_hm['S_CO(0)']['S_CV(1)']][m]).reshape(dimCV,dimCO)
        hmDD[m][slS_CV1,slS_OV0] = -numpy.einsum('bjai->iajb',hm[map_hm['S_OV(0)']['S_CV(1)']][m]).reshape(dimCV,dimOV)
        hmDD[m][slS_CV1,slS_CV1] = numpy.einsum('aibj->iajb',hm[map_hm['S_CV(1)']['S_CV(1)']][m]).reshape(dimCV,dimCV)
        # D-Q
        hmDQ[m] = numpy.zeros(((nc+no)*nv + nc*(nv+no),nc*nv),dtype=numpy.complex128)
        hmDQ[m][slS_CV0,:] = numpy.einsum('aibj->iajb',hm[map_hm['S_CV(0)']['S1_CV(1)']][m]).reshape(dimCV,dimCV)
        hmDQ[m][slS_CO0,:] = numpy.einsum('aibj->iajb',hm[map_hm['S_CO(0)']['S1_CV(1)']][m]).reshape(dimCO,dimCV)
        hmDQ[m][slS_OV0,:] = numpy.einsum('aibj->iajb',hm[map_hm['S_OV(0)']['S1_CV(1)']][m]).reshape(dimOV,dimCV)
        hmDQ[m][slS_CV1,:] = numpy.einsum('aibj->iajb',hm[map_hm['S_CV(1)']['S1_CV(1)']][m]).reshape(dimCV,dimCV)
        # Q-Q
        hmQQ[m] = numpy.einsum('aibj->iajb',hm[map_hm['S1_CV(1)']['S1_CV(1)']][m]).reshape(dimCV,dimCV)

    # Blockwise construct hso
    hso = numpy.zeros((dimSO,dimSO,),dtype=numpy.complex128) # blockwise (8x8)
    # 8
    hso[slGS_12,slGS_12]  = xgs.T.conjugate()@hmGSGS[0]*w(1/2,-1/2,1/2,-1/2)@xgs
    hso[slGS_12,slGS12]  = xgs.T.conjugate()@hmGSGS[1]*w(1/2,-1/2,1/2,1/2)@xgs
    hso[slGS_12,slD_12]  = xgs.T.conjugate()@hmGSD[0]*w(1/2,-1/2,1/2,-1/2)@xd
    hso[slGS_12,slD12]  = xgs.T.conjugate()@hmGSD[1]*w(1/2,-1/2,1/2,1/2)@xd
    hso[slGS_12,slQ_32]  = xgs.T.conjugate()@hmGSQ[-1]*w(1/2,-1/2,3/2,-3/2)@xq
    hso[slGS_12,slQ_12]  = xgs.T.conjugate()@hmGSQ[0]*w(1/2,-1/2,3/2,-1/2)@xq
    hso[slGS_12,slQ12]  = xgs.T.conjugate()@hmGSQ[1]*w(1/2,-1/2,3/2,1/2)@xq
    # hso[slGS_12,slQ32]  = xgs_12.T.conjugate()@hmGSQ[1]*w(1/2,-1/2,3/2,3/2)@xq32
    # 7
    hso[slGS12,slGS12]  = xgs.T.conjugate()@hmGSGS[0]*w(1/2,1/2,1/2,1/2)@xgs
    hso[slGS12,slD_12]  = xgs.T.conjugate()@hmGSD[-1]*w(1/2,1/2,1/2,-1/2)@xd
    hso[slGS12,slD12]  = xgs.T.conjugate()@hmGSD[0]*w(1/2,1/2,1/2,1/2)@xd
    # hso[slGS_12,slQ_32]  = xgs12.T.conjugate()@hmGSQ[0]*w(1/2,1/2,3/2,-3/2)@xq_32
    hso[slGS12,slQ_12]  = xgs.T.conjugate()@hmGSQ[-1]*w(1/2,1/2,3/2,-1/2)@xq
    hso[slGS12,slQ12]  = xgs.T.conjugate()@hmGSQ[0]*w(1/2,1/2,3/2,1/2)@xq
    hso[slGS12,slQ32]  = xgs.T.conjugate()@hmGSQ[1]*w(1/2,1/2,3/2,3/2)@xq
    # 6
    hso[slD_12,slD_12]  = xd.T.conjugate()@hmDD[0]*w(1/2,-1/2,1/2,-1/2)@xd
    hso[slD_12,slD12]  = xd.T.conjugate()@hmDD[1]*w(1/2,-1/2,1/2,1/2)@xd
    hso[slD_12,slQ_32]  = xd.T.conjugate()@hmDQ[-1]*w(1/2,-1/2,3/2,-3/2)@xq
    hso[slD_12,slQ_12]  = xd.T.conjugate()@hmDQ[0]*w(1/2,-1/2,3/2,-1/2)@xq
    hso[slD_12,slQ12]  = xd.T.conjugate()@hmDQ[1]*w(1/2,-1/2,3/2,1/2)@xq
    # hso[slD_12,slQ32]  = xd_12.T.conjugate()@hmDQ[1]*w(1/2,-1/2,3/2,3/2)@xq32
    # 5
    hso[slD12,slD12]  = xd.T.conjugate()@hmDD[0]*w(1/2,1/2,1/2,1/2)@xd
    # hso[slD12,slQ_32]  = xd12.T.conjugate()@hmDQ[0]*w(1/2,1/2,3/2,-3/2)@xq_32
    hso[slD12,slQ_12]  = xd.T.conjugate()@hmDQ[-1]*w(1/2,1/2,3/2,-1/2)@xq
    hso[slD12,slQ12]  = xd.T.conjugate()@hmDQ[0]*w(1/2,1/2,3/2,1/2)@xq
    hso[slD12,slQ32]  = xd.T.conjugate()@hmDQ[1]*w(1/2,1/2,3/2,3/2)@xq
    # 4
    hso[slQ_32,slQ_32]  = xq.T.conjugate()@hmQQ[0]*w(3/2,-3/2,3/2,-3/2)@xq
    hso[slQ_32,slQ_12]  = xq.T.conjugate()@hmQQ[1]*w(3/2,-3/2,3/2,-1/2)@xq
    # hso[slQ_32,slQ12]  = xq_32.T.conjugate()@hmQQ[0]*w(3/2,-3/2,3/2,1/2)@xq12
    # hso[slQ_32,slQ32]  = xq_32.T.conjugate()@hmQQ[1]*w(3/2,-3/2,3/2,3/2)@xq32
    # 3
    hso[slQ_12,slQ_12]  = xq.T.conjugate()@hmQQ[0]*w(3/2,-1/2,3/2,-1/2)@xq
    hso[slQ_12,slQ12]  = xq.T.conjugate()@hmQQ[1]*w(3/2,-1/2,3/2,1/2)@xq
    # hso[slQ_12,slQ32]  = xq_12.T.conjugate()@hmQQ[1]*w(3/2,-1/2,3/2,3/2)@xq32
    # 2
    hso[slQ12,slQ12]  = xq.T.conjugate()@hmQQ[0]*w(3/2,1/2,3/2,1/2)@xq
    hso[slQ12,slQ32]  = xq.T.conjugate()@hmQQ[1]*w(3/2,1/2,3/2,3/2)@xq
    # 1
    hso[slQ32,slQ32]  = xq.T.conjugate()@hmQQ[0]*w(3/2,3/2,3/2,3/2)@xq

    # Symmetrized
    hso = symm_matrix(hso)

    Omega = numpy.zeros((dimSO,))
    Omega[slGS_12] = Omega[slGS12] = 0.0
    Omega[slD12] = Omega[slD_12] = ed.copy()
    Omega[slQ12] = Omega[slQ32] = Omega[slQ_12] = Omega[slQ_32] = eq.copy()

    Omega = numpy.diag(Omega) / unit.ha2eV

    print(f"Begin to diag Omega+hso --{hso.shape}--")
    import scipy
    eso, vso = scipy.linalg.eigh(hso + Omega, driver='evd')
    print(f"RKS E (in Hatree) = {mf.e_tot}")
    print(f"The E (in eV), shape={eso.shape} --")
    print(numpy.sort(numpy.diag(Omega))[:20] * unit.ha2eV)
    print(f"The E_soc (in eV) --")
    print(eso[:20] * unit.ha2eV)
    print(f"The E_soc (in cm-1) --")
    print(eso[:20] * unit.ha2eV * unit.eV2cm_1)
    hso_cm_1 = hso*unit.ha2eV*unit.eV2cm_1
    return hso, eso

# The code can be optmized:
# 1. construct hmDD, ... directly
# 2. calculate (Xhm_ab)X