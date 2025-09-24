import os, sys
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"


import scipy.linalg
import numpy
import math
from pyscf import gto, scf

mol = gto.M(
    atom = 'Zn 0 0 0',
    basis = 'aug-cc-pVTZ-DK',
    verbose=4,
    symmetry='C1',
    spin = 0)

from pyscf import lib
lib.param.LIGHT_SPEED = 137.0359895000 # From BDF

from pyscf import dft
mf = scf.sfx2c(dft.RKS(mol))
mf.xc = 'SVWN'
# mf.xc = 'b3lyp'
mf.conv_tol = 1e-12
mf = mf.run()

c = lib.param.LIGHT_SPEED

import sfX2C_soDKH1
VsoDKH1_a = sfX2C_soDKH1.get_soDKH1_somf(mf,mol,c,iop='bp',include_mf2e=0,debug=1)
VsoDKH1_a = numpy.einsum('nij,ik,jl->nkl',VsoDKH1_a,mf.mo_coeff,mf.mo_coeff)

def w(S,M,Sprime,Mprime):
    from sympy.physics.wigner import wigner_3j
    if abs(wigner_3j(S,1,Sprime,-S,S-Sprime,Sprime).doit().evalf())<1e-3:
        return 0.
    else:
        return (-1)**(S-M) * float((wigner_3j(S,1,Sprime,-M,M-Mprime,Mprime) / wigner_3j(S,1,Sprime,-S,S-Sprime,Sprime)).evalf())

def read_ns(mol,S):
    norb = mol.nao
    Smax = int(S*2)
    nc,no,nv = (mol.nelectron-Smax)//2, Smax, mol.nao-(mol.nelectron-Smax)//2-Smax
    slc = slice(0,nc,1)
    slo = slice(nc,nc+no,1)
    slv = slice(nc+no,norb,1)
    assert nc+no+nv==norb
    return norb, nc,no,nv, slc,slo,slv

def Generate_hm(mol,S,Vso):
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
        # if Scond:
        #     hm[3][m]  = -numpy.einsum('jv->vj',h[m][slc,slo])/math.sqrt(2)
        #     hm[4][m]  =  numpy.einsum('vb->bv',h[m][slo,slv])/math.sqrt(2)
        #     hm[5][m]  = -numpy.einsum('jb->bj',h[m][slc,slv])*math.sqrt(S/(1+S))
        hm[6][m]   =  numpy.zeros((nv,nc), dtype=numpy.complex128)
        hm[6][m]  += -numpy.einsum('jb->bj',h[m][slc,slv])
        # --- 5
        hm[7][m]  =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
        # if Scond:
        #     hm[8][m]  = -numpy.einsum('av,ij->aivj',h[m][slv,slo],numpy.eye(nc,nc))/2.
        #     hm[9][m]  = -numpy.einsum('vi,ab->aibv',h[m][slo,slc],numpy.eye(nv,nv))/2.
        #     hm[10][m] = -numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))*math.sqrt(S/(2*(1+S)))
        #     hm[10][m]-= -numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))*math.sqrt(S/(2*(1+S)))
        hm[11][m] =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
        hm[11][m]+=  numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))/math.sqrt(2)
        hm[11][m]+= -numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))/math.sqrt(2)
        # --- 4
        # if Scond:
        #     hm[12][m] = -numpy.einsum('uv,ij->uivj',h[m][slo,slo],numpy.eye(nc,nc))/math.sqrt(2)
        #     hm[12][m]-= -numpy.einsum('ji,uv->uivj',h[m][slc,slc],numpy.eye(no,no))/math.sqrt(2)
        #     hm[13][m] =  numpy.zeros((no,nc,nv,no))
        #     hm[14][m] =  numpy.einsum('ub,ij->uibj',h[m][slo,slv],numpy.eye(nc,nc))*0.5*(1-S)/math.sqrt(S*(S+1))
        #     hm[15][m] = -numpy.einsum('ub,ij->uibj',h[m][slo,slv],numpy.eye(nc,nc))
        # # --- 3
        #     hm[16][m] =  numpy.einsum('ab,uv->aubv',h[m][slv,slv],numpy.eye(no,no))
        #     hm[17][m] =  numpy.einsum('ju,ab->aubj',h[m][slc,slo],numpy.eye(nv,nv))*0.5*(S-1)/math.sqrt(S*(S+1))
        #     hm[18][m] =  numpy.einsum('ju,ab->aubj',h[m][slc,slo],numpy.eye(nv,nv))
        # # --- 2
        #     hm[19][m] =  numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))/((1+S)*math.sqrt(2))
        #     hm[19][m]+=  numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))/((1+S)*math.sqrt(2))
        #     hm[20][m] = -numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))*math.sqrt(S/(2*(S+1)))
        #     hm[20][m]+= -numpy.einsum('ab,ji->aibj',h[m][slv,slv],numpy.eye(nc,nc))*math.sqrt(S/(2*(S+1)))
        # --- 1
        hm[21][m] =  numpy.zeros((nv,nc,nv,nc), dtype=numpy.complex128)
        hm[21][m]+=  numpy.einsum('ji,ab->aibj',h[m][slc,slc],numpy.eye(nv,nv))/math.sqrt(2)
        hm[21][m]+=  numpy.einsum('ab,ij->aibj',h[m][slv,slv],numpy.eye(nc,nc))/math.sqrt(2)
    print(f"End of generating hm matrix elements --")
    return hm

S = 0
hm = Generate_hm(mol,S,VsoDKH1_a)

# from mokit.lib import py2bdf
# py2bdf(mf, 'x2c_hamiltonian/BDF/C/C.inp')
norb, nc,no,nv, slc,slo,slv = read_ns(mol,S)
ns = nc*nv
from TDA import TDA

tda = TDA(mol, mf, singlet=1, nstates=ns)
# e_s, _, xs = tda.pyscf_tda(conv_tol=1e-12, is_analyze=False)
e_s, _, _, xs = tda.kernel()

tda = TDA(mol, mf, singlet=0, nstates=ns)
# e_t, _, xt0 = tda.pyscf_tda(conv_tol=1e-12, is_analyze=False)
e_t, _, _, xt0 = tda.kernel()

xgs = numpy.ones((1,1))
xt_1 = xt1 = xt0.copy()

hso = numpy.zeros((4*ns+1,4*ns+1,),dtype=numpy.complex128)
dimGS = 1
dimSS = xs.shape[-1]
dimTT = xt0.shape[-1]

slGS = slice(0,dimGS,1)
slS = slice(dimGS,dimGS+dimSS,1)
slT_1 = slice(dimGS+dimSS,dimGS+dimSS+dimTT,1)
slT0 = slice(dimGS+dimSS+dimTT,dimGS+dimSS+dimTT*2,1)
slT1 = slice(dimGS+dimSS+dimTT*2,dimGS+dimSS+dimTT*3,1)

# line1
hso[slGS,slGS]  = xgs.T.conjugate()@hm[1][0]*w(0,0,0,0)@xgs
hso[slGS,slS]   = xgs.T.conjugate()@numpy.einsum('bj->jb',hm[2][0]).reshape(1,ns)*w(0,0,0,0)@xs
hso[slGS,slT_1] = xgs.T.conjugate()@numpy.einsum('bj->jb',hm[6][-1]).reshape(1,ns)*w(0,0,1,-1)@xt_1
hso[slGS,slT0]  = xgs.T.conjugate()@numpy.einsum('bj->jb',hm[6][0]).reshape(1,ns)*w(0,0,1,0)@xt0
hso[slGS,slT1]  = xgs.T.conjugate()@numpy.einsum('bj->jb',hm[6][1]).reshape(1,ns)*w(0,0,1,1)@xt1

# line2
# hso[slS,slGS]  = hso[slGS,slS].T.conjugate()
hso[slS,slS]   = xs.T.conjugate()@numpy.einsum('aibj->iajb',hm[7][0]).reshape(ns,ns)*w(0,0,0,0)@xs
hso[slS,slT_1] = xs.T.conjugate()@numpy.einsum('aibj->iajb',hm[11][-1]).reshape(ns,ns)*w(0,0,1,-1)@xt_1
hso[slS,slT0]  = xs.T.conjugate()@numpy.einsum('aibj->iajb',hm[11][0]).reshape(ns,ns)*w(0,0,1,0)@xt0
hso[slS,slT1]  = xs.T.conjugate()@numpy.einsum('aibj->iajb',hm[11][1]).reshape(ns,ns)*w(0,0,1,1)@xt1

# line3
# hso[slT_1,slGS]  = hso[slGS,slT_1].T.conjugate()
# hso[slT_1,slS]   = hso[slS,slT_1].T.conjugate()
hso[slT_1,slT_1] = xt_1.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][0]).reshape(ns,ns)*w(1,-1,1,-1)@xt_1
hso[slT_1,slT0]  = xt_1.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][1]).reshape(ns,ns)*w(1,-1,1,0)@xt0
# hso[slT_1,slT1]

# line4
# hso[slT0,slGS]  = hso[slGS,slT0].T.conjugate()
# hso[slT0,slS]   = hso[slS,slT0].T.conjugate()
# hso[slT0,slT_1] = hso[slT_1,slT0].T.conjugate()
hso[slT0,slT0]  = xt0.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][0]).reshape(ns,ns)*w(1,0,1,0)@xt0
hso[slT0,slT1]  = xt0.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][1]).reshape(ns,ns)*w(1,0,1,1)@xt1

# line5
# hso[slT1,slGS] = hso[slGS,slT1].T.conjugate()
# hso[slT1,slS]  = hso[slS,slT1].T.conjugate()
# # hso[slT1,slT_1]= hso[slT_1,slT1].T.conjugate()
# hso[slT1,slT0] = hso[slT0,slT1].T.conjugate()
hso[slT1,slT1] = xt1.T.conjugate()@numpy.einsum('aibj->iajb',hm[21][0]).reshape(ns,ns)*w(1,1,1,1)@xt1

# Symmetrized
hso = numpy.triu(hso) + numpy.triu(hso, 1).T.conjugate()

Omega = numpy.zeros((4*ns+1))
Omega[slGS] = 0.0
Omega[slS] = e_s.copy()
Omega[slT_1] = Omega[slT0] = Omega[slT1] = e_t.copy()

from utils import unit
Omega = numpy.diag(Omega)/unit.ha2eV

print(f"Begin to diag Omega+hso --{hso.shape}--")
import scipy
e_so, v_so = scipy.linalg.eigh(Omega+hso, driver='evd')
print(f"RKS E (in Hatree) = {mf.e_tot}")
print(f"The E (in eV), shape={e_so.shape} --")
print(numpy.sort(numpy.diag(Omega))[:20] * unit.ha2eV)
print(f"The E_soc (in eV) --")
print(e_so[:20] * unit.ha2eV)

breakpoint()