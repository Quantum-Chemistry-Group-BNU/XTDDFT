#
# Description: The sfX2C_soDKH1 Hamiltonian
#
# OriginalTheory: JCP 137, 154114 (2012)
#                 JCP 141, 054111 (2014)
#
# Implementation: Mol. Phys. 111 (24), 3741-3755
#.                see Eqs(34)-(53) for details of formulae.
#
# Implemented by Zhendong Li (zhendongli2008@gmail.com)
#
# Functions:
#
# def inv12(s):
# def sfx2c1e(t, v, w, s, c):
# def get_p(dm,x,rp):
# def get_wso(mol):
# def get_kint(mol):
# def get_hso1e(wso,x,rp):
# def get_fso2e(kint,x,rp,pLL,pLS,pSS):
# def get_fso2e_direct(mol,x,rp,pLL,pLS,pSS):
# def get_fso2e_direct_par(mol,x,rp,pLL,pLS,pSS,nproc):
# def get_fso2e_block_par(mol,x,rp,pLL,pLS,pSS,nproc,max_block_ao):
# def get_soDKH1_somf(myhf,mol,c,iop='x2c',mf2e_impl='auto',nproc=1,debug=False):
# 
import multiprocessing
import os
import numpy, time
import scipy.linalg
from functools import reduce
from typing import Literal

_GSO_DIRECT_PAR_STATE = {}
_DEFAULT_BLOCK_AO = 24

Mf2eImpl = Literal['auto', 'full', 'direct', 'direct-par', 'block-par']

def _set_pyscf_num_threads(num_threads):
    try:
        from pyscf import lib
    except Exception:
        return None
    try:
        old_threads = lib.num_threads()
        lib.num_threads(int(num_threads))
        return old_threads
    except Exception:
        return None

def _restore_pyscf_num_threads(old_threads):
    if old_threads is None:
        return
    try:
        from pyscf import lib
        lib.num_threads(int(old_threads))
    except Exception:
        pass

def _format_memory(num_bytes):
    if num_bytes is None:
        return 'unknown'
    return f'{num_bytes/1024.0**3:.2f} GiB'

def _get_available_memory_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        pass
    try:
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, OSError, ValueError):
        return None

def _estimate_full_mf2e_memory_bytes(nb, dtype=numpy.float64):
    '''
    Estimate the dominant peak-memory cost of the original full-K build.
    ddint uses 9*nb^4 elements, kint uses 3*nb^4 elements. A modest safety
    factor is applied to leave room for temporaries and Python overhead.
    '''
    itemsize = numpy.dtype(dtype).itemsize
    nq = nb**4
    dominant = (9 + 3) * nq * itemsize
    overhead = (12 * nb * nb + 3 * nb * nb * nb) * itemsize
    return int((dominant + overhead) * 1.25)

def _estimate_block_mf2e_memory_bytes(nb,
                                      max_block_ao=_DEFAULT_BLOCK_AO,
                                      nproc=1,
                                      dtype=numpy.float64):
    '''
    Estimate the peak memory for block-par. The dominant cost is the temporary
    block integral tensor plus one set of accumulated Gso matrices per worker.
    '''
    itemsize = numpy.dtype(dtype).itemsize
    block_ao = min(int(max_block_ao), int(nb))
    block_dominant = (9 + 3) * (block_ao ** 4) * itemsize
    worker_gso = 9 * nb * nb * itemsize
    shared = (12 * nb * nb + 3 * nb * nb * nb) * itemsize
    workers = max(1, int(nproc))
    return int((block_dominant + shared + (workers + 1) * worker_gso) * 1.35)

def _estimate_direct_mf2e_memory_bytes(nb,
                                       max_shell_ao,
                                       dtype=numpy.float64):
    '''
    Estimate the peak memory for the serial direct contraction. The largest
    shell-quartet temporary scales with the maximum shell size, while the
    dominant persistent arrays are the accumulated Gso/P/X/R matrices.
    '''
    itemsize = numpy.dtype(dtype).itemsize
    shell_ao = max(1, int(max_shell_ao))
    block_dominant = (9 + 3) * (shell_ao ** 4) * itemsize
    shared = (21 * nb * nb + 3 * nb * nb * nb) * itemsize
    return int((block_dominant + shared) * 1.35)

def _resolve_mf2e_impl(mf2e_impl, mol, nb, nproc=1, max_block_ao=_DEFAULT_BLOCK_AO):
    if mf2e_impl != 'auto':
        return mf2e_impl, None, None, None, None
    ao_loc = mol.ao_loc_nr()
    max_shell_ao = int((ao_loc[1:] - ao_loc[:-1]).max())
    estimated = _estimate_full_mf2e_memory_bytes(nb)
    block_estimated = _estimate_block_mf2e_memory_bytes(nb, max_block_ao, nproc)
    direct_estimated = _estimate_direct_mf2e_memory_bytes(nb, max_shell_ao)
    available = _get_available_memory_bytes()
    if available is None:
        return 'block-par', estimated, block_estimated, direct_estimated, available
    if estimated < 0.7 * available:
        return 'full', estimated, block_estimated, direct_estimated, available
    if block_estimated < 0.7 * available:
        return 'block-par', estimated, block_estimated, direct_estimated, available
    if direct_estimated < 0.7 * available:
        return 'direct', estimated, block_estimated, direct_estimated, available
    raise MemoryError(
        'Insufficient memory for all mf2e implementations: '
        f'full={_format_memory(estimated)}, '
        f'block-par={_format_memory(block_estimated)}, '
        f'direct={_format_memory(direct_estimated)}, '
        f'available={_format_memory(available)}'
    )

def inv12(s):
    '''
    Return s^{-1/2}
    '''
    e,v = scipy.linalg.eigh(s)
    return reduce(numpy.dot,(v,numpy.diag(1/numpy.sqrt(e)),v.T))

def sfx2c1e(t, v, w, s, c):
    '''
    Generate hso1e
    here
        t: T kinetic energy term
        v: V Nuclear attraction term
        w: Wsf p·Vne·p
        s: S Overlap (Metric)
        c: speed of light
    '''
    nao = s.shape[0]
    n2 = nao * 2
    # costruct h and m
    h = numpy.zeros((n2,n2), dtype=v.dtype)
    m = numpy.zeros((n2,n2), dtype=v.dtype)
    # ref.(42) of JCP 137, 154114 (2012)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)
    e, a = scipy.linalg.eigh(h, m)
    cl = a[:nao,nao:]
    cs = a[nao:,nao:]
    x = cs.dot(cl.T.dot(numpy.linalg.inv(cl.dot(cl.T)))) # ref.(40)
    stilde = s + x.T.dot(m[nao:,nao:].dot(x)) # ref.(39) and here m[nao:,nao:] = t * (.5/c**2) is t·a^2/2
    sih = inv12(s) # s^{-1/2}
    sh = numpy.linalg.inv(inv12(s)) # s^{1/2}
    rp = sih.dot(inv12(sih.dot(stilde.dot(sih))).dot(sh)) # ref.(38)
    # ref.(48) of JCP 137, 154114 (2012) 
    l1e = h[:nao,:nao] + h[:nao,nao:].dot(x) + x.T.dot(h[nao:,:nao]) + x.T.dot(h[nao:,nao:].dot(x))
    h1e = rp.T.dot(l1e.dot(rp))
    return x,rp,h1e

def get_p(dm,x,rp):
    '''
    Generate spin-averaged dm P
    ref.(50):
        PLL = R+0@P@(R+0)^dagger
        PLS = PLL@X0^dagger
        PSS = X0@PLL@X0^dagger
    here
        dm: P
        x: X0
        rp: R+0
    '''
    pLL = rp.dot(dm.dot(rp.T))
    pLS = pLL.dot(x.T)
    pSS = x.dot(pLL.dot(x.T))
    return pLL,pLS,pSS

def get_wso(mol):
    '''
    Generate Wso
    ref.(43)
        Wso = sigma_l w^l
        w^l = varepslion(lmn) <mu_m|Vne|nu_n>
    '''
    nb = mol.nao_nr()
    wso = numpy.zeros((3,nb,nb))
    for iatom in range(mol.natm):
        zA  = mol.atom_charge(iatom)
        xyz = mol.atom_coord(iatom)
        mol.set_rinv_orig(xyz)
        wso += -zA*mol.intor('cint1e_prinvxp_sph', 3) # sign due to integration by part
    return wso

def get_kint(mol,debug=0):
    '''
    Generate two-electron integrals K(l)mu,nu;kappa,lambda
    where mu, nu, kappa, lambda index orbitals
    ref.(49)
        K(l)mu,nu;lappa,lambda = varepsilon(lmn)(mu_m,nu|kappa_n,lambda)
        mu_m = \partial_m mu
    where (mu_m,nu|kappa_n,lambda) come from `int2e_ip1ip2_sph` libcint
    '''
    nb = mol.nao_nr()
    np = nb*nb # number of excitations
    nq = np*np # two single excitation
    ddint = mol.intor('int2e_ip1ip2_sph').reshape(3,3,nq)
    kint = numpy.zeros((3,nq))
    kint[0] = ddint[1,2] - ddint[2,1] # Lx = y p_z - z p_y
    kint[1] = ddint[2,0] - ddint[0,2] # Ly = z p_x - x p_z
    kint[2] = ddint[0,1] - ddint[1,0] # Lz = x p_y - y p_x
    kint = kint.reshape(3,nb,nb,nb,nb)
    if debug:
        for n in range(0,kint.shape[0]):
            print(f"axis-{n}, kint+kint.T={numpy.linalg.norm(kint[n]+kint[n].transpose(3,4,1,2))}")
    return kint

def get_hso1e(wso,x,rp):
    '''
    Generate hso1e
    ref.(41)
        hso1e = a4 * (R+0)^dagger @ X0^dagger @ Wso @ X0 @ R+0
    here
        wso: Wso = ref.(43)
        x: X0
        rp: R+0
    '''
    nb = x.shape[0]
    hso1e = numpy.zeros((3,nb,nb))
    for ic in range(3):
        hso1e[ic] = reduce(numpy.dot,(rp.T,x.T,wso[ic],x,rp))
    return hso1e

def get_fso2e(kint,x,rp,pLL,pLS,pSS):
    '''
    Generate fso2e (cost memory step)
    ref.(42)
        fso2e = a4*(R+0)^dagger [GsoLL + GsoLS + X0^dagger@GsoSL + X0^dagger@GsoSS@X0] R+0
    here
        kint: K(l)mu,nu,kappa,lambda
        x: X0
        rp: R+0
        pLL, pLS, pSS: spin-averaged density matrices
    '''
    nb = x.shape[0]
    fso2e = numpy.zeros((3,nb,nb))
    for ic in range(3):
        gsoLL  = -numpy.einsum('lmkn,lk->mn',kint[ic],pSS)*2.

        gsoLS  = -numpy.einsum('mlkn,lk->mn',kint[ic],pLS)
        gsoLS -=  numpy.einsum('lmkn,lk->mn',kint[ic],pLS)

        gsoSS  = -numpy.einsum('mnkl,lk->mn',kint[ic],pLL)*2.
        gsoSS -=  numpy.einsum('mnlk,lk->mn',kint[ic],pLL)*2.
        gsoSS +=  numpy.einsum('mlnk,lk->mn',kint[ic],pLL)*2.

        fso2e[ic] = gsoLL + gsoLS.dot(x) + x.T.dot(-gsoLS.T) + x.T.dot(gsoSS.dot(x))
        fso2e[ic] = reduce(numpy.dot,(rp.T,fso2e[ic],rp))
    return fso2e

def _accumulate_gso_shell_range(mol,pLL,pLS,pSS,ish0,ish1):
    nb = pLL.shape[0]
    gsoLL = numpy.zeros((3,nb,nb))
    gsoLS = numpy.zeros((3,nb,nb))
    gsoSS = numpy.zeros((3,nb,nb))
    ao_loc = mol.ao_loc_nr()
    for ish in range(ish0, ish1):
        i0, i1 = ao_loc[ish], ao_loc[ish+1]
        di = i1 - i0
        for jsh in range(mol.nbas):
            j0, j1 = ao_loc[jsh], ao_loc[jsh+1]
            dj = j1 - j0
            for ksh in range(mol.nbas):
                k0, k1 = ao_loc[ksh], ao_loc[ksh+1]
                dk = k1 - k0
                for lsh in range(mol.nbas):
                    l0, l1 = ao_loc[lsh], ao_loc[lsh+1]
                    dl = l1 - l0
                    ddblk = mol.intor_by_shell(
                        'int2e_ip1ip2_sph', (ish, jsh, ksh, lsh)
                    ).reshape(3,3,di,dj,dk,dl)
                    kblk = numpy.empty((3,di,dj,dk,dl), dtype=ddblk.dtype)
                    kblk[0] = ddblk[1,2] - ddblk[2,1]
                    kblk[1] = ddblk[2,0] - ddblk[0,2]
                    kblk[2] = ddblk[0,1] - ddblk[1,0]
                    for ic in range(3):
                        gsoLL[ic, j0:j1, l0:l1] += -2. * numpy.einsum(
                            'lmkn,lk->mn', kblk[ic], pSS[i0:i1,k0:k1]
                        )
                        gsoLS[ic, i0:i1, l0:l1] += -numpy.einsum(
                            'mlkn,lk->mn', kblk[ic], pLS[j0:j1,k0:k1]
                        )
                        gsoLS[ic, j0:j1, l0:l1] += -numpy.einsum(
                            'lmkn,lk->mn', kblk[ic], pLS[i0:i1,k0:k1]
                        )
                        gsoSS[ic, i0:i1, j0:j1] += -2. * numpy.einsum(
                            'mnkl,lk->mn', kblk[ic], pLL[l0:l1,k0:k1]
                        )
                        gsoSS[ic, i0:i1, j0:j1] += -2. * numpy.einsum(
                            'mnlk,lk->mn', kblk[ic], pLL[k0:k1,l0:l1]
                        )
                        gsoSS[ic, i0:i1, k0:k1] += 2. * numpy.einsum(
                            'mlnk,lk->mn', kblk[ic], pLL[j0:j1,l0:l1]
                        )
    return gsoLL,gsoLS,gsoSS

def _finalize_fso2e(gsoLL,gsoLS,gsoSS,x,rp):
    '''
    Apply the final X and R transformations to the accumulated Gso blocks.
    '''
    nb = x.shape[0]
    fso2e = numpy.zeros((3,nb,nb))
    for ic in range(3):
        fso2e[ic] = gsoLL[ic] + gsoLS[ic].dot(x) + x.T.dot(-gsoLS[ic].T) + x.T.dot(gsoSS[ic].dot(x))
        fso2e[ic] = reduce(numpy.dot,(rp.T,fso2e[ic],rp))
    return fso2e

def get_fso2e_direct(mol,x,rp,pLL,pLS,pSS):
    '''
    Generate fso2e without explicitly storing the full K tensor.
    The algebra is identical to get_fso2e(); only the contraction order
    is changed to accumulate shell blocks directly into the final matrices.
    '''
    gsoLL,gsoLS,gsoSS = _accumulate_gso_shell_range(
        mol,pLL,pLS,pSS,0,mol.nbas
    )
    return _finalize_fso2e(gsoLL,gsoLS,gsoSS,x,rp)

def _partition_shell_blocks_by_ao(mol,max_block_ao):
    ao_loc = mol.ao_loc_nr()
    blocks = []
    sh0 = 0
    ao0 = ao_loc[0]
    for sh1 in range(1, mol.nbas + 1):
        aocur = ao_loc[sh1] - ao0
        if aocur > max_block_ao and sh1 - 1 > sh0:
            blocks.append((sh0, sh1 - 1))
            sh0 = sh1 - 1
            ao0 = ao_loc[sh0]
    blocks.append((sh0, mol.nbas))
    return blocks

def _accumulate_gso_block_range(mol,pLL,pLS,pSS,ib0,ib1,max_block_ao):
    nb = pLL.shape[0]
    gsoLL = numpy.zeros((3,nb,nb))
    gsoLS = numpy.zeros((3,nb,nb))
    gsoSS = numpy.zeros((3,nb,nb))
    ao_loc = mol.ao_loc_nr()
    blocks = _partition_shell_blocks_by_ao(mol, max_block_ao)

    for ib in range(ib0, ib1):
        ish0, ish1 = blocks[ib]
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        di = i1 - i0
        for jsh0, jsh1 in blocks:
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            dj = j1 - j0
            for ksh0, ksh1 in blocks:
                k0, k1 = ao_loc[ksh0], ao_loc[ksh1]
                dk = k1 - k0
                for lsh0, lsh1 in blocks:
                    l0, l1 = ao_loc[lsh0], ao_loc[lsh1]
                    dl = l1 - l0
                    ddblk = mol.intor(
                        'int2e_ip1ip2_sph',
                        aosym='s1',
                        shls_slice=(ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1),
                    ).reshape(3,3,di,dj,dk,dl)
                    kblk = numpy.empty((3,di,dj,dk,dl), dtype=ddblk.dtype)
                    kblk[0] = ddblk[1,2] - ddblk[2,1]
                    kblk[1] = ddblk[2,0] - ddblk[0,2]
                    kblk[2] = ddblk[0,1] - ddblk[1,0]
                    for ic in range(3):
                        gsoLL[ic, j0:j1, l0:l1] += -2. * numpy.einsum(
                            'lmkn,lk->mn', kblk[ic], pSS[i0:i1,k0:k1]
                        )
                        gsoLS[ic, i0:i1, l0:l1] += -numpy.einsum(
                            'mlkn,lk->mn', kblk[ic], pLS[j0:j1,k0:k1]
                        )
                        gsoLS[ic, j0:j1, l0:l1] += -numpy.einsum(
                            'lmkn,lk->mn', kblk[ic], pLS[i0:i1,k0:k1]
                        )
                        gsoSS[ic, i0:i1, j0:j1] += -2. * numpy.einsum(
                            'mnkl,lk->mn', kblk[ic], pLL[l0:l1,k0:k1]
                        )
                        gsoSS[ic, i0:i1, j0:j1] += -2. * numpy.einsum(
                            'mnlk,lk->mn', kblk[ic], pLL[k0:k1,l0:l1]
                        )
                        gsoSS[ic, i0:i1, k0:k1] += 2. * numpy.einsum(
                            'mlnk,lk->mn', kblk[ic], pLL[j0:j1,l0:l1]
                        )
    return gsoLL,gsoLS,gsoSS

def _init_gso_direct_par_worker(mol,pLL,pLS,pSS,num_threads=1):
    _set_pyscf_num_threads(num_threads)
    global _GSO_DIRECT_PAR_STATE
    _GSO_DIRECT_PAR_STATE = {
        'mol': mol,
        'pLL': pLL,
        'pLS': pLS,
        'pSS': pSS,
    }

def _gso_direct_par_worker(task):
    ish0, ish1 = task
    return _accumulate_gso_shell_range(
        _GSO_DIRECT_PAR_STATE['mol'],
        _GSO_DIRECT_PAR_STATE['pLL'],
        _GSO_DIRECT_PAR_STATE['pLS'],
        _GSO_DIRECT_PAR_STATE['pSS'],
        ish0,
        ish1,
    )

def _gso_block_par_worker(task):
    ib0, ib1, max_block_ao = task
    return _accumulate_gso_block_range(
        _GSO_DIRECT_PAR_STATE['mol'],
        _GSO_DIRECT_PAR_STATE['pLL'],
        _GSO_DIRECT_PAR_STATE['pLS'],
        _GSO_DIRECT_PAR_STATE['pSS'],
        ib0,
        ib1,
        max_block_ao,
    )

def _partition_shell_ranges(mol,nparts):
    '''
    Split shells into a small number of contiguous ranges. The shell AO sizes
    are used as cheap weights so each worker gets a comparable amount of work.
    '''
    nparts = max(1, min(int(nparts), mol.nbas))
    if nparts == 1:
        return [(0, mol.nbas)]

    ao_loc = mol.ao_loc_nr()
    shell_weights = ao_loc[1:] - ao_loc[:-1]
    total_weight = int(shell_weights.sum())
    target_weight = max(1, total_weight / nparts)

    tasks = []
    ish0 = 0
    acc = 0
    remaining_parts = nparts
    for ish, weight in enumerate(shell_weights):
        acc += int(weight)
        remaining_shells = mol.nbas - (ish + 1)
        if remaining_parts > 1 and (
            acc >= target_weight or remaining_shells < (remaining_parts - 1)
        ):
            tasks.append((ish0, ish + 1))
            ish0 = ish + 1
            acc = 0
            remaining_parts -= 1
    if ish0 < mol.nbas:
        tasks.append((ish0, mol.nbas))
    return tasks

def get_fso2e_direct_par(mol,x,rp,pLL,pLS,pSS,nproc):
    '''
    Parallel exact direct SOMF contraction. Each worker accumulates a shell
    range and returns partial Gso blocks that are summed in the parent.
    '''
    if nproc is None or nproc <= 1 or mol.nbas <= 1:
        return get_fso2e_direct(mol,x,rp,pLL,pLS,pSS)

    methods = multiprocessing.get_all_start_methods()
    ctx = multiprocessing.get_context('fork' if 'fork' in methods else methods[0])
    if mol.nbas <= 3 * nproc:
        tasks = [(ish, ish+1) for ish in range(mol.nbas)]
    else:
        tasks = _partition_shell_ranges(mol, 2 * nproc)
    gsoLL = numpy.zeros((3,pLL.shape[0],pLL.shape[1]))
    gsoLS = numpy.zeros((3,pLL.shape[0],pLL.shape[1]))
    gsoSS = numpy.zeros((3,pLL.shape[0],pLL.shape[1]))
    print(
        f"direct-par setup: processes={nproc}, tasks={len(tasks)}, "
        f"shells={mol.nbas}"
    )
    progress_stride = max(1, len(tasks) // 10)
    done = 0
    par_time0 = time.time()
    with ctx.Pool(
        processes=nproc,
        initializer=_init_gso_direct_par_worker,
        initargs=(mol,pLL,pLS,pSS,1),
    ) as pool:
        for partLL, partLS, partSS in pool.imap_unordered(_gso_direct_par_worker, tasks):
            gsoLL += partLL
            gsoLS += partLS
            gsoSS += partSS
            done += 1
            if done == 1 or done == len(tasks) or done % progress_stride == 0:
                print(
                    f"direct-par progress: {done}/{len(tasks)} tasks "
                    f"finished in {time.time()-par_time0:.2f}s"
                )
    print("direct-par finalize: assembling fso2e from accumulated Gso blocks")
    return _finalize_fso2e(gsoLL,gsoLS,gsoSS,x,rp)

def get_fso2e_block_par(mol,x,rp,pLL,pLS,pSS,nproc,max_block_ao=_DEFAULT_BLOCK_AO):
    '''
    Parallel exact semi-direct contraction on larger AO shell blocks. This
    trades some memory for fewer integral calls and larger contraction kernels.
    '''
    blocks = _partition_shell_blocks_by_ao(mol, max_block_ao)
    if nproc is None or nproc <= 1:
        gsoLL,gsoLS,gsoSS = _accumulate_gso_block_range(
            mol,pLL,pLS,pSS,0,len(blocks),max_block_ao
        )
        return _finalize_fso2e(gsoLL,gsoLS,gsoSS,x,rp)

    methods = multiprocessing.get_all_start_methods()
    ctx = multiprocessing.get_context('fork' if 'fork' in methods else methods[0])
    ntask = max(1, min(len(blocks), int(nproc)))
    tasks = []
    start = 0
    for idx in range(ntask):
        stop = start + (len(blocks) - start + (ntask - idx) - 1) // (ntask - idx)
        tasks.append((start, stop, max_block_ao))
        start = stop
    gsoLL = numpy.zeros((3,pLL.shape[0],pLL.shape[1]))
    gsoLS = numpy.zeros((3,pLL.shape[0],pLL.shape[1]))
    gsoSS = numpy.zeros((3,pLL.shape[0],pLL.shape[1]))
    print(
        f"block-par setup: processes={min(int(nproc), len(tasks))}, "
        f"tasks={len(tasks)}, blocks={len(blocks)}, max_block_ao={max_block_ao}"
    )
    progress_stride = max(1, len(tasks) // 10)
    done = 0
    par_time0 = time.time()
    with ctx.Pool(
        processes=min(int(nproc), len(tasks)),
        initializer=_init_gso_direct_par_worker,
        initargs=(mol,pLL,pLS,pSS,1),
    ) as pool:
        for partLL, partLS, partSS in pool.imap_unordered(_gso_block_par_worker, tasks):
            gsoLL += partLL
            gsoLS += partLS
            gsoSS += partSS
            done += 1
            if done == 1 or done == len(tasks) or done % progress_stride == 0:
                print(
                    f"block-par progress: {done}/{len(tasks)} tasks "
                    f"finished in {time.time()-par_time0:.2f}s"
                )
    print("block-par finalize: assembling fso2e from accumulated Gso blocks")
    return _finalize_fso2e(gsoLL,gsoLS,gsoSS,x,rp)

def get_soDKH1_somf(myhf,
                    mol,
                    c,
                    iop='x2c',
                    include_mf2e=True,
                    mf2e_impl:Mf2eImpl='auto',
                    nproc=1,
                    debug=False
    ):
    '''
    Generate Hso
    here
        myhf: scf
        mol: mole
        c: speed of light
        iop: x2c or bp
        include_mf2e: include fso2e or not
        mf2e_impl: implementation of the SOMF 2e contraction
            auto   : choose full, else block-par, else direct
            full   : build the full K tensor explicitly (original implementation)
            direct : direct shell-block contraction with lower peak memory
            direct-par : parallel exact direct shell-block contraction
            block-par : parallel exact semi-direct shell-block contraction
        nproc: number of worker processes for parallel modes
        debug: print extra matrix info
    return
        Vso in ao basis
    '''
    print(f"Begin to generate Vso")
    time0 = time.time()
    xmol,contr_coeff = myhf.with_x2c.get_xmol(mol) # change basis
    nb = contr_coeff.shape[0]
    nc = contr_coeff.shape[1]
    resolved_impl = mf2e_impl
    est_full_mem = None
    est_block_mem = None
    est_direct_mem = None
    avail_mem = None
    if include_mf2e:
        resolved_impl, est_full_mem, est_block_mem, est_direct_mem, avail_mem = _resolve_mf2e_impl(
            mf2e_impl, xmol, nb, nproc
        )
    print(
        f"get_soDKH1_somf with iop={iop}, SOMF={include_mf2e}, "
        f"mf2e_impl={mf2e_impl}->{resolved_impl}, nproc={nproc}, "
        f"(nb,nc)={contr_coeff.shape}"
    )
    if include_mf2e and est_full_mem is not None:
        print(
            f"memory estimate full={_format_memory(est_full_mem)}, "
            f"block-par={_format_memory(est_block_mem)}, "
            f"direct={_format_memory(est_direct_mem)}, "
            f"available={_format_memory(avail_mem)}"
        )
    if iop == 'x2c':
        t = xmol.intor_symmetric('int1e_kin') # kinetic energy term
        v = xmol.intor_symmetric('int1e_nuc') # Nuclear attraction term
        s = xmol.intor_symmetric('int1e_ovlp') # Overlap (Metric)
        w = xmol.intor_symmetric('int1e_pnucp') # p·Vne·p
        x,rp,h1e = sfx2c1e(t, v, w, s, c)
    elif iop == 'bp':
        x = numpy.identity(nb)
        rp = numpy.identity(nb)
    else:
        raise ValueError(f"iop={iop} not in {'x2c','bp'}")
    dm = myhf.make_rdm1()
    # Spin-Averaged for ROHF or UHF
    if len(dm.shape)==3: 
        dm = (dm[0]+dm[1])/2.
    else: # Here the DM matrix element is 1 or 1/2 or 0, not 2, 1, 0
        dm = dm/2.
    dm = reduce(numpy.dot,(contr_coeff,dm,contr_coeff.T))
    pLL,pLS,pSS = get_p(dm,x,rp) # ref.(50)
    wso = get_wso(xmol)
    # Make combination Hsd = hso1e + fso2e ref.(36)
    a4 = 0.25/c**2 # alpha^2/4
    hso1e = get_hso1e(wso,x,rp)
    VsoDKH1 = a4*(hso1e)
    if include_mf2e:
        mf2e_time0 = time.time()
        limited_threads = None
        try:
            if resolved_impl != 'full':
                limited_threads = _set_pyscf_num_threads(1)
                if limited_threads is not None and limited_threads != 1:
                    print(
                        f"mf2e thread control ({resolved_impl}): "
                        f"temporarily set pyscf/lib threads {limited_threads} -> 1"
                    )
            if resolved_impl == 'full':
                kint_time0 = time.time()
                kint = get_kint(xmol)
                kint_time1 = time.time()
                if debug:
                    for ic in range(3):
                        print(f"ic={ic, numpy.linalg.norm(kint[ic]+kint[ic].transpose(2,3,0,1))}")
                fso2e_time0 = time.time()
                fso2e = get_fso2e(kint,x,rp,pLL,pLS,pSS)
                fso2e_time1 = time.time()
                print(
                    f"mf2e timing ({resolved_impl}): "
                    f"get_kint={kint_time1-kint_time0:.2f}s, "
                    f"get_fso2e={fso2e_time1-fso2e_time0:.2f}s, "
                    f"total={fso2e_time1-mf2e_time0:.2f}s"
                )
            elif resolved_impl == 'direct':
                fso2e = get_fso2e_direct(xmol,x,rp,pLL,pLS,pSS)
                mf2e_time1 = time.time()
                print(f"mf2e timing ({resolved_impl}): total={mf2e_time1-mf2e_time0:.2f}s")
            elif resolved_impl == 'direct-par':
                fso2e = get_fso2e_direct_par(xmol,x,rp,pLL,pLS,pSS,nproc)
                mf2e_time1 = time.time()
                print(f"mf2e timing ({resolved_impl}, nproc={nproc}): total={mf2e_time1-mf2e_time0:.2f}s")
            elif resolved_impl == 'block-par':
                fso2e = get_fso2e_block_par(xmol,x,rp,pLL,pLS,pSS,nproc)
                mf2e_time1 = time.time()
                print(f"mf2e timing ({resolved_impl}, nproc={nproc}): total={mf2e_time1-mf2e_time0:.2f}s")
            else:
                raise ValueError(f"mf2e_impl={mf2e_impl} not in {'auto','full','direct','direct-par','block-par'}")
        finally:
            if resolved_impl != 'full':
                _restore_pyscf_num_threads(limited_threads)
        VsoDKH1 += a4*(fso2e)

    if debug:
        for ic in range(3):
            tmp = hso1e[ic]
            print(f"{ic} hso1e  norm={numpy.linalg.norm(tmp):.12e}, "
                f"Norm(tmp+tmp.T)={numpy.linalg.norm(tmp + tmp.T):.12e}")
            if include_mf2e:
                tmp = fso2e[ic]
                print(f"{ic} fso2e  norm={numpy.linalg.norm(tmp):.12e}, "
                    f"Norm(tmp+tmp.T)={numpy.linalg.norm(tmp + tmp.T):.12e}")
                tmp = hso1e[ic] + fso2e[ic]
                print(f"{ic} vso2e  norm={numpy.linalg.norm(tmp):.12e}, "
                    f"Norm(tmp+tmp.T)={numpy.linalg.norm(tmp + tmp.T):.12e}")

    VsoDKH1contr = numpy.zeros((3,nc,nc))
    for ic in range(3):
        VsoDKH1contr[ic] = reduce(numpy.dot,(contr_coeff.T,VsoDKH1[ic],contr_coeff))
    if debug:
        for ic in range(3):
            tmp = VsoDKH1contr[ic]
            print(f"{ic} VsoDKH1contr  norm={numpy.linalg.norm(tmp):.12e}, "
                f"Norm(tmp+tmp.T)={numpy.linalg.norm(tmp + tmp.T):.12e}")
    time1 = time.time()
    print(f"End of generating Vso, cost time={time1-time0:.2f}s")
    return VsoDKH1contr
