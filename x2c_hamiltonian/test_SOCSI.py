import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
sys.path.append('./')
sys.path.append('./../')
sys.path.append('./../../')

import scipy.linalg
import numpy
import math
from pyscf import gto, scf

import logging
logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from pyscf import lib, dft
lib.param.LIGHT_SPEED = 137.0359895000 # From BDF

mol = gto.M(
    atom = 'N 0 0 0',
    basis = 'aug-cc-pVTZ-DK',
    verbose=6,
    symmetry='C1',
    spin = 3)

mf = scf.sfx2c(dft.ROKS(mol))
mf.xc = 'SVWN'
mf.conv_tol = 1e-12
mf = mf.run()

c = lib.param.LIGHT_SPEED
ha2eV = 27.2113834
# ================ X2C(BP) sf+somf Hamilonian ====================
import sfX2C_soDKH1
VsoDKH1_a = sfX2C_soDKH1.get_soDKH1_somf(mf,mol,c,iop='x2c',include_mf2e=1,debug=1)
VsoDKH1_a = numpy.einsum('nij,ik,jl->nkl',VsoDKH1_a,mf.mo_coeff,mf.mo_coeff)

# ================ Construct a state dict. ====================
state_dict = {}
state_dict['|So>'] = []
state_dict['|S+>'] = []
state_dict['|S->'] = []

# |S->
from SF_TDA.SF_XTDA import SA_SF_TDA
print(f"{'='*15} Perform SF-down-TDA calculation {'='*15}")
xsf_tda = SA_SF_TDA(mf,davidson=0)
xsf_tda.nstates = 20
xsf_tda.kernel(nstates = xsf_tda.nstates, remove = 1)
em = xsf_tda.e
xm_ = xsf_tda.v
dim = xsf_tda.nc*xsf_tda.nv + xsf_tda.nc*xsf_tda.no + xsf_tda.no*xsf_tda.nv
xm = numpy.zeros((dim+xsf_tda.no**2,xsf_tda.nstates,))
xm[:dim,:] = xm_[:dim,:]
xm[dim:,:] = xsf_tda.vects @ xm_[dim:,:]
xsf_tda.re=1
xsf_tda.analyse()

for i in range(0,len(em),1):
    state_dict['|S->'].append((em[i], xm[:,i]))

# |So>
from XTDA import XTDA
print(f"{'='*15} Perform XTDA calculation {'='*15}")
xtda= XTDA(mol,mf,basis='tensor')
xtda.nstates = 20
xtda.kernel()
eo = xtda.e
xo = xtda.v

for i in range(0,len(eo),1):
    state_dict['|So>'].append((eo[i], xo[:,i]))

# |S+>
from SF_TDA.SF_TDA import SF_TDA
print(f"{'='*15} Perform SF-up-TDA calculation {'='*15}")
sf_tda = SF_TDA(mf,isf=1,davidson=0)
sf_tda.nstates = 20
sf_tda.kernel()
ep = sf_tda.e[:sf_tda.nstates]
xp = sf_tda.v[:,:sf_tda.nstates]
sf_tda.analyse()

for i in range(0,len(ep),1):
    state_dict['|S+>'].append((ep[i], xp[:,i]))

# ================ State interaction ====================
from driver.si_driver import SI_driver
mysi = SI_driver(mf=mf,
                 S=1.5,
                 Vso=VsoDKH1_a,
                 ngs=1,
                 states=state_dict,
                 )
mysi.kernel()

# which gives
# ================== Summary of S I calculation ==================
#   No   i-th   Excited state    v**2        Eso(eV)       Esf(eV)
#    0   0-th   |GS, +1.5⟩       99.1%     -0.000010      0.000000
#    1   0-th   |GS, -1.5⟩       99.1%     -0.000010      0.000000
#    2   0-th   |GS, +0.5⟩       99.2%     -0.000010      0.000000
#    3   0-th   |GS, -0.5⟩       99.2%     -0.000010      0.000000
#    4   4-th   |S-, -0.5⟩       36.4%      2.326570      2.326849
#    5   4-th   |S-, -1.5⟩       36.4%      2.326570      2.326849
#    6   3-th   |S-, -1.5⟩       21.0%      2.326583      2.326849
#    7   3-th   |S-, -0.5⟩       21.0%      2.326583      2.326849
#    8   0-th   |S-, -1.5⟩       48.4%      2.326839      2.326849
#    9   0-th   |S-, -0.5⟩       48.4%      2.326839      2.326849
#   10   1-th   |S-, -0.5⟩       21.5%      2.327049      2.326849
#   11   1-th   |S-, -1.5⟩       21.5%      2.327049      2.326849
#   12   4-th   |S-, -0.5⟩       32.5%      2.327067      2.326849
#   13   4-th   |S-, -1.5⟩       32.5%      2.327067      2.326849
#   14   5-th   |S-, -1.5⟩       47.9%      3.965525      3.965721
#   15   5-th   |S-, -0.5⟩       47.9%      3.965525      3.965721
#   16   7-th   |S-, -0.5⟩       48.7%      3.965776      3.965721
#   17   7-th   |S-, -1.5⟩       48.7%      3.965776      3.965721
#   18   5-th   |S-, -0.5⟩       48.7%      3.966019      3.965721
#   19   5-th   |S-, -1.5⟩       48.7%      3.966019      3.965721