import numpy as np
import pandas as pd
from utils import unit


def overlap(vec2, pscsf, vec1=None):
    """
    vec1 is refer eigen vectors
    vec2 is calculated eigen vectors
    """
    ol = []
    if vec1 is not None:
        for vec, ps in zip(vec2, pscsf):
            # ol.append(vec.T @ vec1[ps, :] / np.sum(np.abs(vec.T @ vec1[ps, :]), axis=1, keepdims=True))
            ol.append(
                (vec.T @ vec1[ps] / np.expand_dims(
                    np.sqrt(np.einsum('ij, ij->j', vec1, vec1) * np.einsum('ij, ij->j', vec, vec)),
                    axis=1
                ))**2
            )
    else:
        assert len(vec2) >= 2
        vec1 = vec2[0]
        pscsf1 = pscsf[0]
        pscsf1ind = np.where(pscsf1)[0]
        for vec, ps in zip(vec2[1:], pscsf[1:]):
            ps = pscsf1[pscsf1ind] & ps[pscsf1ind]
            # ol.append(vec.T @ vec1[ps, :] / np.sum(np.abs(vec.T @ vec1[ps, :]), axis=1, keepdims=True))
            ol.append(
                (vec.T @ vec1[ps] / np.expand_dims(
                    np.sqrt(np.einsum('ij, ij->j', vec1, vec1) * np.einsum('ij, ij->j', vec, vec)),
                    axis=1
                ))**2
            )
    return ol


# # have error
# def order_pyscf2my(nc, no, nv):
#     # nc and nv is the value before selecting activate space
#     order = np.indices(((nc+no)*nv+nc*(no+nv), )).squeeze()
#     # # my order to pyscf order
#     # for oi in range(nc*no):
#     #     order = np.insert(order, (nc_old+no)*nv_old+oi*nv_old+nc_old*no, (nc_old+no)*nv_old+oi)
#     #     order = np.delete(order, (nc_old+no)*nv_old)
#     # # pyscf order to my order
#     for oi in range(nc):
#         for noi in range(no):
#             order = np.insert(order, (nc+no)*nv+no*oi+noi, (nc+no)*nv+oi*nv+noi)
#             order = np.delete(order, (nc+no)*nv+oi*nv+noi+1)
#     return order


def get_cov(mf):
    # get nc, nv, no(if open shell)
    mo_occ = mf.mo_occ
    if len(mo_occ.shape) == 1 and (1 not in mo_occ):
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        nc = len(occidx)
        no = 0
        nv = len(viridx)
    elif len(mo_occ.shape) == 1 and (1 in mo_occ):
        occidx_a = np.where(mo_occ >= 1)[0]
        viridx_a = np.where(mo_occ == 0)[0]
        occidx_b = np.where(mo_occ >= 2)[0]
        viridx_b = np.where(mo_occ != 2)[0]
        nocc_a = len(occidx_a)
        nvir_a = len(viridx_a)
        nocc_b = len(occidx_b)
        nvir_b = len(viridx_b)
        nc = min(nocc_a, nocc_b)
        no = abs(nocc_a - nocc_b)
        nv = min(nvir_a, nvir_b)
    elif len(mo_occ.shape) == 2:
        occidx_a = np.where(mo_occ[0] == 1)[0]
        viridx_a = np.where(mo_occ[0] == 0)[0]
        occidx_b = np.where(mo_occ[1] == 1)[0]
        viridx_b = np.where(mo_occ[1] == 0)[0]
        nocc_a = len(occidx_a)
        nvir_a = len(viridx_a)
        nocc_b = len(occidx_b)
        nvir_b = len(viridx_b)
        nc = min(nocc_a, nocc_b)
        no = abs(nocc_a - nocc_b)
        nv = min(nvir_a, nvir_b)
    else:
        raise ValueError("mo_occ is error")
    return nc, no, nv


def save_appro_result(xc, result, tda=None, overlap=None, **kwargs):
    """overlap can input maxol or maxolind or maxol_olind"""
    e = []
    for approi in result.keys():
        e.append(result[approi]["energy"])
    if "method" in kwargs and "path" in kwargs:
        path = kwargs["path"]
        method = kwargs["method"]
    elif "method" in kwargs and "path" not in kwargs:
        path = './'
        method = kwargs['method']
    else:
        path = './'
        method = ''
    print("=" * 50)
    print("save " + method + "sTDA and " + method + "TDA result")
    print("=" * 50)
    filename = path + xc + method
    pd.DataFrame(np.array(e).T).to_csv(filename + 'sTDA.csv')
    if tda is not None:
        pd.DataFrame(tda * unit.ha2eV).to_csv(filename + 'TDA.csv')
    if overlap is not None:
        for ol, i in zip(overlap, range(len(overlap))):
            pd.DataFrame(ol).to_csv(filename + 'overlap{}.csv'.format(i))

