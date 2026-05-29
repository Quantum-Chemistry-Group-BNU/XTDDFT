import numpy as np

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


def order_pyscf2my(nc, no, nv):
    # nc and nv is the value before selecting activate space
    order = np.indices(((nc+no)*nv+nc*(no+nv), )).squeeze()
    # # my order to pyscf order
    # for oi in range(nc*no):
    #     order = np.insert(order, (nc_old+no)*nv_old+oi*nv_old+nc_old*no, (nc_old+no)*nv_old+oi)
    #     order = np.delete(order, (nc_old+no)*nv_old)
    # # pyscf order to my order
    for oi in range(nc):
        for noi in range(no):
            order = np.insert(order, (nc+no)*nv+no*oi+noi, (nc+no)*nv+oi*(no+nv)+noi)
            order = np.delete(order, (nc+no)*nv+oi*(no+nv)+noi+1)
    return order


def so2st(eigvec, nc=None, no=None, nv=None, lcva=None, lova=None, lcob=None, lcvb=None):
    # transform eigen vector in spin orbital basis to spin tensor basis
    # eigen value is same in spin orbital basis and spin tensor orbital, so just transform eigen vector
    # Note: this function only transform eigen vector that from top to bottom is cv(aa) ov(aa) co(bb) cv(bb)
    if nc is not None:
        assert no is not None and nv is not None, "no and nv cannot be None"
        cva = eigvec[:nc*nv, :]
        ova = eigvec[nc*nv:(nc+no)*nv, :]
        cob = eigvec[(nc+no)*nv:(nc+no)*nv+no*nc, :]
        cvb = eigvec[(nc+no)*nv+no*nc:, :]
        cv0 = np.sqrt(2)/2*(cva+cvb)
        cv1 = np.sqrt(2)/2*(cva-cvb)
        ov0 = ova
        co0 = cob
    elif lcva is not None:
        assert lova is not None and lcob is not None and lcvb is not None, "lova, lcob and lcvb cannot be None"
        cva = eigvec[:lcva, :]
        ova = eigvec[lcva:lcva+lova, :]
        cob = eigvec[lcva+lova:lcva+lova+lcob, :]
        cvb = eigvec[lcva+lova+lcob:, :]
        cv0 = np.sqrt(2) / 2 * (cva + cvb)
        cv1 = np.sqrt(2) / 2 * (cva - cvb)
        ov0 = ova
        co0 = cob
    else:
        raise ValueError("please input nc no nv or lcva lova lcob lcvb")
    eigvec_st = np.concatenate((cv0, ov0, co0, cv1), axis=0)
    return eigvec_st


def st2so(eigvec, nc=None, no=None, nv=None, lcva=None, lova=None, lcob=None, lcvb=None):
    # Note: this function only transform eigen vector that from top to bottom is cv(0) ov(0) co(0) cv(1)
    if nc is not None:
        assert no is not None and nv is not None, "no and nv cannot be None"
        cv0 = eigvec[:nc*nv, :]
        ov0 = eigvec[nc*nv:(nc+no)*nv, :]
        co0 = eigvec[(nc+no)*nv:(nc+no)*nv+no*nc, :]
        cv1 = eigvec[(nc+no)*nv+no*nc:, :]
        cva = (cv0+cv1)/np.sqrt(2)
        cvb = (cv0-cv1)/np.sqrt(2)
        ova = ov0
        cob = co0
    elif lcva is not None:
        assert lova is not None and lcob is not None and lcvb is not None, "lova, lcob and lcvb cannot be None"
        cv0 = eigvec[:lcva, :]
        ov0 = eigvec[lcva:lcva+lova, :]
        co0 = eigvec[lcva+lova:lcva+lova+lcob, :]
        cv1 = eigvec[lcva+lova+lcob:, :]
        cva = (cv0+cv1)/np.sqrt(2)
        cvb = (cv0-cv1)/np.sqrt(2)
        ova = ov0
        cob = co0
    else:
        raise ValueError("please input nc no nv or lcva lova lcob lcvb")
    eigvec_so = np.concatenate((cva, ova, cob, cvb), axis=0)
    return eigvec_so
