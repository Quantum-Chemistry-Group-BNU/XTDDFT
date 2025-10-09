import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"


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
VsoDKH1_a = sfX2C_soDKH1.get_soDKH1_somf(mf,mol,c,iop='bp',include_mf2e=1,debug=1)
VsoDKH1_a = numpy.einsum('nij,ik,jl->nkl',VsoDKH1_a,mf.mo_coeff,mf.mo_coeff)

from utils.si_helper import sidriver
sidriver(mf,0.0,VsoDKH1_a)