from pyscf import gto, dft
import SF_XTDA
from pyscf import sftda
import numpy as np

mol = gto.M(
        atom = """ Be """,
        basis = '631g',
        charge = 0,
        spin = 2,
        verbose = 3,
    )
  
mf = dft.UKS(mol)
mf.xc = 'b3lyp'
mf.kernel()
sf_tda = SF_XTDA.SA_SF_TDA(mf, SA=0, davidson=True)
e0, values = sf_tda.kernel(nstates=10, remove=False)
print(e0)
mftd1 = sftda.TDA_SF(mf)
mftd1.max_space=500
mftd1.nstates = 10 # the number of excited states
mftd1.extype = 1  # 1 for spin flip down excited energies
mftd1.collinear_samples=200
mftd1.kernel()
print(mftd1.e*27.21138505)

assert np.allclose(e0, mftd1.e*27.21138505)