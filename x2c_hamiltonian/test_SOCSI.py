import os, sys
os.environ["OMP_NUM_THREADS"] = '2'
os.environ["MKL_NUM_THREADS"] = '2'
os.environ["OPENBLAS_NUM_THREADS"] = '2'
os.environ["NUMEXPR_NUM_THREADS"] = '2'
sys.path.append('./')
sys.path.append('./x2c_hamiltonian')

import numpy
from pyscf import gto, scf

import logging
logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def make_scf_stable(mf):
    mo,occ,stable,_ = mf.stability(return_status=True)

    while (not stable):
        dm = mf.make_rdm1(mo,occ)
        mf.kernel(dm)
        mo,occ,stable,_ = mf.stability(return_status=True)
    return mf

from pyscf import lib, dft
lib.param.LIGHT_SPEED = 137.0359895000 # From BDF

c = lib.param.LIGHT_SPEED
ha2eV = 27.2113834

def soc_mf(mf):
    # ================ Construct a state dict. ====================
    state_dict = {}
    state_dict['|So>'] = []
    state_dict['|S+>'] = []
    state_dict['|S->'] = []

    # |S->
    from xtddft.XSF_TDA import SA_SF_TDA
    print(f"{'='*15} Perform SF-down-TDA calculation {'='*15}")
    xsf_tda = SA_SF_TDA(mf,davidson=1)
    xsf_tda.nstates = 20
    xsf_tda.kernel(nstates = xsf_tda.nstates, remove = 1)
    em = xsf_tda.e
    xm_ = xsf_tda.v
    # trans the formulation of X vector
    dim = xsf_tda.nc*xsf_tda.nv + xsf_tda.nc*xsf_tda.no + xsf_tda.no*xsf_tda.nv
    xm = numpy.zeros((dim+xsf_tda.no**2+xsf_tda.no,xsf_tda.nstates,))
    xm[:dim,:] = xm_[:dim,:]
    Xo = (xsf_tda.vects @ xm_[dim:,:]).reshape(xsf_tda.no,xsf_tda.no,xsf_tda.nstates)
    Xo_diag = numpy.einsum('iiN->iN',Xo)
    Xo_non = Xo - numpy.einsum('iN,ij->ijN',Xo_diag,numpy.eye(Xo_diag.shape[0]))
    # O1O2
    xm[dim:dim+xsf_tda.no**2,:] = Xo_non.reshape(xsf_tda.no**2,xsf_tda.nstates)
    # O1O1
    xm[dim+xsf_tda.no**2:,:] = Xo_diag
    # analyse
    xsf_tda.re=1
    xsf_tda.analyse()
    print(f"Excited states |S-⟩:")
    for i in range(0,em.shape[0],1):
        print(f"No.{i:3d} Esf={(em[i]*ha2eV):.5f} eV, Eex={(em[i]-em[0])*ha2eV:.5f}")
    breakpoint()

    for i in range(0,len(em),1):
        state_dict['|S->'].append((em[i], xm[:,i]))

    # |So>
    from xtddft.XTDA import XTDA
    print(f"{'='*15} Perform XTDA calculation {'='*15}")
    xtda= XTDA(mol,mf,basis='tensor')
    xtda.nstates = 40
    xtda.kernel()
    eo = xtda.e / ha2eV # whb's code gives energy in eV
    xo = xtda.v
    print(f"Excited states |So⟩:")
    for i in range(0,xtda.nstates,1):
        print(f"No.{i:3d} Esf={xtda.e[i]:.5f} eV, Eex={(xtda.e[i]-xtda.e[0]):.5f}")
    breakpoint()

    for i in range(0,len(eo),1):
        state_dict['|So>'].append((eo[i], xo[:,i]))

    # |S+>
    from xtddft.SF_TDA import SF_TDA
    print(f"{'='*15} Perform SF-up-TDA calculation {'='*15}")
    sf_tda = SF_TDA(mf,isf=1,davidson=0)
    sf_tda.nstates = 20
    sf_tda.kernel()
    ep = sf_tda.e[:sf_tda.nstates]
    xp = sf_tda.v[:,:sf_tda.nstates]
    sf_tda.analyse()
    print(f"Excited states |S+⟩:")
    for i in range(0,ep.shape[0],1):
        print(f"No.{i:3d} Esf={(ep[i]*ha2eV):.5f} eV, Eex={(ep[i]-ep[0])*ha2eV:.5f}")
    breakpoint()

    for i in range(0,len(ep),1):
        state_dict['|S+>'].append((ep[i], xp[:,i]))

    # ================ X2C(BP) sf+somf Hamilonian ====================
    import sfX2C_soDKH1
    VsoDKH1_a = sfX2C_soDKH1.get_soDKH1_somf(mf,mol,c,iop='x2c',include_mf2e=1,debug=1)
    VsoDKH1_a = numpy.einsum('nij,ik,jl->nkl',VsoDKH1_a,mf.mo_coeff,mf.mo_coeff)

    # ================ State interaction ====================
    from driver.si_driver import SI_driver
    mysi = SI_driver(mf=mf,
                    S=mf.mol.spin/2,
                    Vso=VsoDKH1_a,
                    ngs=0,
                    states=state_dict,
                    )
    mysi.kernel(print=40)
    # mysi.print_hso()

mol = gto.M(
    atom = 'As 0 0 0',
    basis = 'cc-pVTZ-DK',
    verbose=6,
    symmetry='D2h',
    charge=0,
    spin = 3)

mol.build()
print(f"mol.nao={mol.nao}")

mf = scf.sfx2c(dft.ROKS(mol))
mf.xc = 'bhandhlyp'
mf.max_cycle = 200
mf = mf.run()
mf = make_scf_stable(mf)
if not mf.converged: breakpoint()
soc_mf(mf)