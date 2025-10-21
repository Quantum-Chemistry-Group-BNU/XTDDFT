import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import scipy.linalg
import numpy
import math
from pyscf import gto, scf


from pyscf import lib, dft
lib.param.LIGHT_SPEED = 137.0359895000 # From BDF

mol = gto.M(
    atom = 'Cu 0 0 0',
    basis = 'aug-cc-pVTZ-DK',
    verbose=6,
    symmetry='C1',
    spin = 1)

mf = scf.sfx2c(dft.ROKS(mol))
mf.xc = 'SVWN'
mf.conv_tol = 1e-12
mf = mf.run()

c = lib.param.LIGHT_SPEED

import sfX2C_soDKH1
VsoDKH1_a = sfX2C_soDKH1.get_soDKH1_somf(mf,mol,c,iop='x2c',include_mf2e=1,debug=1)
VsoDKH1_a = numpy.einsum('nij,ik,jl->nkl',VsoDKH1_a,mf.mo_coeff,mf.mo_coeff)

from utils.si_helper import sidriver
sidriver(mf,0.5,VsoDKH1_a,1,[None,None])
# [-0.1212, -0.1212, 0.6474, 0.6474, 0.6474]