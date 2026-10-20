#!/usr/bin/env python
"""在 CPU 上直接运行 UKS → USF-TDA-down，不需要预先生成 chk 文件。"""
import os
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np
from pyscf import dft, gto, lib

from XTDDFT.utils.backend import backend_info, set_backend

set_backend("cpu")
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down


lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))


# ===== Manually edit these parameters =====
# atom = """
# C   0.000000   0.000000   0.000000
# H   0.000000   0.950000   0.150000
# H   0.100000  -0.750000   0.650000
# """
# # 2,7-NQ triplet, state = 1
# atom = '''
#  C                -0.0009921647   1.2149703900  -1.1753951761
#  C                -0.0005862116   2.4722095967  -0.4788410917
#  C                 0.0011188769   2.4121475512   0.9757040351
#  C                 0.0017102937   1.2327779142   1.6273937099
#  C                 0.0016566668  -1.2327483797   1.6275034908
#  C                 0.0012537948  -2.4121799234   0.9759224856
#  C                -0.0003990493  -2.4723722595  -0.4786151055
#  C                -0.0008222621  -1.2151884205  -1.1752790656
#  C                -0.0004585087  -0.0000865530  -0.4993691910
#  C                 0.0009512513  -0.0000134420   0.9354541543
#  H                -0.0015714324   1.2476595640  -2.2485022910
#  O                -0.0012701703   3.5385648210  -1.0778638474
#  H                 0.0018134990   3.3501766464   1.4971313454
#  H                 0.0028090921   1.2101797480   2.7023613678
#  H                 0.0026024636  -1.2100742864   2.7024683033
#  H                 0.0020208687  -3.3501731810   1.4974087918
#  O                -0.0015079723  -3.5387398769  -1.0776095116
#  H                -0.0015752574  -1.2479444099  -2.2483886071
# '''
# 2,7-NQ triplet, XSF-TDA optimized 
atom = '''
    C      -0.0009544154     1.2331503849    -1.1556948940
    C      -0.0003076207     2.4458800751    -0.4733764392
    C       0.0010697820     2.4113842873     0.9635579712
    C       0.0017118155     1.2393330551     1.6259736749
    C       0.0017275307    -1.2393119426     1.6260760839
    C       0.0011003546    -2.4114177982     0.9637553823
    C      -0.0002766447    -2.4460312556    -0.4731747098
    C      -0.0009387866    -1.2333584425    -1.1555932749
    C      -0.0002858638    -0.0000764696    -0.4661676715
    C       0.0010511377    -0.0000189516     0.9286513379
    H      -0.0019840076     1.2588888134    -2.2296092541
    O      -0.0009207283     3.5432518055    -1.1063652430
    H       0.0015582482     3.3529783657     1.4791565872
    H       0.0027417351     1.2208174527     2.7003760934
    H       0.0027572178    -1.2207083323     2.7004769833
    H       0.0016007683    -3.3529695184     1.4794312884
    O      -0.0008758100    -3.5434563238    -1.1060741714
    H      -0.0019680434    -1.2591853761    -2.2295055732
'''
# # m-xylene triplet, state = 1
# atom = '''
#  C                 1.2298817410   0.3357535834   0.0024882433
#  C                -0.0005696256   1.0142836035   0.0014798303
#  C                -1.2311709942   0.3360573356   0.0007075015
#  C                -1.1986290367  -1.0852230443   0.0005344519
#  C                -0.0009214993  -1.7612324059   0.0014477325
#  C                 1.1969762855  -1.0855065481   0.0024711369
#  C                 2.4310210780   1.0228036672   0.0034540372
#  C                -2.4321415499   1.0234533785   0.0004161832
#  H                -0.0004472577   2.0891785241   0.0016239612
#  H                -2.1265507130  -1.6257162133  -0.0006782018
#  H                -0.0009280580  -2.8359022622   0.0012535553
#  H                 2.1246159332  -1.6264840572   0.0032974485
#  H                 2.4559292123   2.0949282327   0.0034205026
#  H                 3.3675184379   0.5004400104   0.0041407698
#  H                -2.4563350960   2.0956062295   0.0003283158
#  H                -3.3689059598   0.5015949974   0.0000445155
# '''
basis = "cc-pvtz"
charge = 0
spin = 2  # N_alpha - N_beta = 2S；2 对应三重态参考态
unit = "Angstrom"
xc = "bhandhlyp"
nstates = 5

use_density_fit = False
auxbasis = None
grid_level = 5
max_memory = 4000  # MB
max_cycle = 200
conv_tol = 1e-10
conv_tol_grad = 1e-6
verbose = 4

method = 1  # 0: ALDA0; 1: multicollinear; 2: collinear
collinear_samples = 20
SA = 0  # UKS 下的普通 USF-TDA，不加自旋匹配修正
remove = False
foo = 1.0
d_lda = 0.3
fglobal = 0.0
fit = False
davidson_backend = "cpu"
davidson_matvec_batch_size = 15
run_analyse = True
analyse_threshold = 0.05
save_results = True
save_file = "USF_mol"
# ==========================================


def main():
    mol = gto.M(
        atom=atom, basis=basis, charge=charge, spin=spin,
        unit=unit, verbose=verbose, max_memory=max_memory,
    )
    mf = dft.UKS(mol)
    mf.xc = xc
    if use_density_fit:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.grids.level = grid_level
    mf.max_cycle = max_cycle
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("UKS reference did not converge.")

    print("backend:", backend_info())
    print("mf class:", type(mf))
    print("reference: UKS")
    print("with_df:", type(getattr(mf, "with_df", None)))
    print("e_tot:", mf.e_tot)
    print("<S^2>, 2S+1:", mf.spin_square())
    print("mo_coeff shape:", mf.mo_coeff.shape)
    print("mo_occ shape:", mf.mo_occ.shape)
    print("SA:", SA)
    print("remove:", remove)

    usf_method = XSF_TDA_down(
        mf,
        method=method,
        SA=SA,
        davidson=True,
        davidson_backend=davidson_backend,
        davidson_matvec_batch_size=davidson_matvec_batch_size,
        collinear_samples=collinear_samples,
    )
    ee, vv = usf_method.kernel(
        nstates=nstates,
        remove=remove,
        foo=foo,
        d_lda=d_lda,
        fglobal=fglobal,
        fit=fit,
        save=save_results,
        save_file=save_file,
    )

    print("converged:", usf_method.converged)
    print("energies / eV:")
    print(np.asarray(ee))
    print("vectors shape:", vv.shape)

    if run_analyse:
        usf_method.analyse(threshold=analyse_threshold)
    return usf_method, ee, vv


if __name__ == "__main__":
    main()
