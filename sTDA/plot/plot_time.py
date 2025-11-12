import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# plt Settings
plt.rc('font', family='Times New Roman', size=14)
plt.rcParams['mathtext.fontset'] = 'custom'

plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.rm'] = 'STIXGeneral:bold'
plt.rcParams['mathtext.tt'] = 'STIXGeneral'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:bold:italic'
plt.rcParams['mathtext.cal'] = 'STIXGeneral'
plt.rcParams['mathtext.sf'] = 'STIXGeneral'

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.linewidth'] = 1

# each part use time
def time(cFm, sP, sS, cAm, dAm, total, nroot, title='time'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nroot, cFm, marker='o', lw=2, label='construct Fock matrix')
    ax.plot(nroot, sP, marker='o', lw=2, label='select PCSF')
    ax.plot(nroot, sS, marker='o', lw=2, label='select SCSF')
    ax.plot(nroot, cAm, marker='o', lw=2, label='construct A matrix')
    ax.plot(nroot, dAm, marker='o', lw=2, label='diagonal A matrix')
    ax.plot(nroot, total, marker='o', lw=2, label='total')
    ax.set_xlabel('number of root')
    ax.set_ylabel('time (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    fig.tight_layout()
    plt.show()
    fig.savefig(title + '.eps', dpi=1200, bbox_inches='tight')
    fig.savefig(title + '.pdf', dpi=1200, bbox_inches='tight')


cFm_ttm = np.array([35.3965, 35.3266, 35.3335, 35.4938, 35.3327])
sP_ttm = np.array([0.0596, 0.0633, 0.0622, 0.0623, 0.0634])
sS_ttm = np.array([0.0541, 0.1200, 0.7204, 5.4631, 27.0665])
cAm_ttm = np.array([0.0009, 0.0042, 0.2642, 3.1767, 39.7718])
dAm_ttm = np.array([0.0002, 0.0003, 0.0044, 0.2206, 39.7716])
total_ttm = np.array([35.6190, 35.6247, 36.4997, 44.5234, 143.1760])
nroot_ttm = np.array([1, 10, 100, 1000, 9715])
time(cFm_ttm, sP_ttm, sS_ttm, cAm_ttm, dAm_ttm, total_ttm, nroot_ttm, 'ttm-part-time')

cFm_mttm2 = np.array([175.2972, 175.5340, 175.1589, 175.6795, 175.6613, 175.4707])
sP_mttm2 = np.array([1.5150, 1.5369, 1.5315, 1.5022, 1.5485, 1.5115])
sS_mttm2 = np.array([0.6077, 0.9632, 7.4614, 52.3502, 383.5761, 128.5612])
cAm_mttm2 = np.array([0.0010, 0.0067, 0.2136, 3.0975, 149.5075, 2227.7476])
dAm_mttm2 = np.array([0.0002, 0.0003, 0.0027, 0.2197, 42.9262, 2059.5060])
total_mttm2 = np.array([178.3171, 178.9396, 185.1384, 233.7626, 755.2129, 4612.3005])
nroot_mttm2 = np.array([1, 12, 104, 1043, 10134, 42928])
time(cFm_mttm2, sP_mttm2, sS_mttm2, cAm_mttm2, dAm_mttm2, total_mttm2, nroot_mttm2, 'mttm2-part-time')

