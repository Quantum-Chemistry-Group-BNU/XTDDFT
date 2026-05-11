#!/usr/bin/env python
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append('/')
sys.path.append('../../')
import numpy as np
import cupy as cp
from pyscf.tdscf import rhf as tdhf_cpu
from pyscf import gto, lib
from pyscf.lib import logger
from gpu4pyscf.scf import hf, rohf, uhf
from gpu4pyscf import dft, scf
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from utils import unit, atom, utils, Davidson


class TimeCounter:
    def __init__(self):
        # init a time counter class, when count consuming time, instantiate corresponding variables
        pass


class XTDA:
    def __init__(self, mol, mf, nstates=10, so2st=True):
        self.mol = mol
        self.mf = mf
        self.nstates = nstates
        self.so2st = so2st
        self.conv_tol = tdhf_cpu.TDBase.conv_tol
        self.lindep = tdhf_cpu.TDBase.lindep
        self.max_cycle = tdhf_cpu.TDBase.max_cycle
        self.deg_eia_thresh = tdhf_cpu.TDBase.deg_eia_thresh
        self.positive_eig_threshold = tdhf_cpu.TDBase.positive_eig_threshold
        self.tc = TimeCounter()
        if self.mf.level_shift is None:
            self.level_shift = 0
        else:
            self.level_shift = self.mf.level_shift
        if isinstance(mf, rohf.ROHF):
            self.X = True
            self.mo_energy = cp.stack((mf.mo_energy, mf.mo_energy), axis=0)
            self.mo_coeff = cp.stack((mf.mo_coeff, mf.mo_coeff), axis=0)
            self.mo_occ = cp.zeros((2, len(mf.mo_coeff)))
            self.mo_occ[0][cp.where(mf.mo_occ >= 1)[0]] = 1
            self.mo_occ[1][cp.where(mf.mo_occ >= 2)[0]] = 1
        elif isinstance(mf, uhf.UHF):
            self.X = False
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = mf.mo_occ
        else:
            raise ValueError

    def gen_response(self, mo_coeff=None, mo_occ=None,
                     with_j=True, hermi=0, max_memory=None, with_nlc=True):
        '''Generate a function to compute the product of UHF response function and
        UHF density matrices.
        '''
        mf = self.mf
        assert isinstance(mf, (uhf.UHF, rohf.ROHF))
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        if not isinstance(mo_coeff, cp.ndarray):
            mo_coeff = cp.asarray(mo_coeff)
        if not isinstance(mo_occ, cp.ndarray):
            mo_occ = cp.asarray(mo_occ)
        mol = mf.mol
        self.tc.A_vxc = 0.0
        self.tc.A_gk = 0.0
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
            hybrid = ni.libxc.is_hybrid_xc(mf.xc)

            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)
            dm0 = None
            print('Omega, alpha, hyb', omega, alpha, hyb)

            if not with_nlc and mf.do_nlc():
                logger.warn(mf, "NLC contribution in gen_response is NOT included")

            def vind(dm1):
                tAvxc0 = time.perf_counter()
                if hermi == 2:
                    v1 = cp.zeros_like(dm1)
                else:
                    v1 = ni.nr_uks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi, rho0, vxc, fxc, max_memory=max_memory)
                    if with_nlc and mf.do_nlc():
                        from pyscf.hessian.rks import get_vnlc_resp  # Cannot import at top due to circular dependency
                        v1 += get_vnlc_resp(mf, mol, mo_coeff, mo_occ, dm1[0] + dm1[1], max_memory)
                cp.cuda.Stream.null.synchronize()
                tAvxc1 = time.perf_counter()
                self.tc.A_vxc += tAvxc1 - tAvxc0

                tAgjk0 = time.perf_counter()
                if not hybrid:
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi=hermi)
                        v1 += vj[0] + vj[1]
                else:
                    # # cpu code
                    # if omega == 0:
                    #     vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    #     vk *= hyb
                    # elif alpha == 0:  # LR=0, only SR exchange
                    #     if with_j:
                    #         vj = mf.get_j(mol, dm1, hermi)
                    #     vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                    #     vk *= hyb
                    # elif hyb == 0:  # SR=0, only LR exchange
                    #     if with_j:
                    #         vj = mf.get_j(mol, dm1, hermi)
                    #     vk = mf.get_k(mol, dm1, hermi, omega=omega)
                    #     vk *= alpha
                    # else:  # SR and LR exchange with different ratios
                    #     vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    #     vk *= hyb
                    #     vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha - hyb)
                    # if with_j:
                    #     v1 += vj[0] + vj[1] - vk
                    # else:
                    #     v1 -= vk

                    # gpu code, gpu version only support some functionals
                    if with_j:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if omega > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha - hyb)
                        v1 += vj[0] + vj[1] - vk
                    else:
                        vk = mf.get_k(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if omega > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha - hyb)
                        v1 -= vk
                cp.cuda.Stream.null.synchronize()
                tAgjk1 = time.perf_counter()
                self.tc.A_gk += tAgjk1 - tAgjk0
                return v1

        elif with_j:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                v1 = vj[0] + vj[1] - vk
                return v1

        else:
            def vind(dm1):
                return -mf.get_k(mol, dm1, hermi=hermi)

        return vind

    def _gen_tda_operation(self, fock_ao=None, wfnsym=None):
        tAp0 = time.perf_counter()
        assert fock_ao is None
        mf = self.mf
        mo_coeff = self.mo_coeff
        mo_energy = self.mo_energy
        mo_occ = self.mo_occ
        mol = mf.mol
        # assert (mf.mo_coeff[0].dtype == cp.double)
        occidxa = cp.where(mo_occ[0] > 0)[0]
        occidxb = cp.where(mo_occ[1] > 0)[0]
        viridxa = cp.where(mo_occ[0] == 0)[0]
        viridxb = cp.where(mo_occ[1] == 0)[0]
        nocca = len(occidxa)
        noccb = len(occidxb)
        nvira = len(viridxa)
        nvirb = len(viridxb)
        orboa = mo_coeff[0][:, occidxa]
        orbob = mo_coeff[1][:, occidxb]
        orbva = mo_coeff[0][:, viridxa]
        orbvb = mo_coeff[1][:, viridxb]

        tAf0 = time.perf_counter()
        if self.X:
            dm = mf.make_rdm1()
            vhf = mf.get_veff(mf.mol, dm)
            h1e = mf.get_hcore()
            focka_ao = h1e + vhf[0]
            fockb_ao = h1e + vhf[1]
            focka_mo = mo_coeff[0].T @ focka_ao @ mo_coeff[0]
            fockb_mo = mo_coeff[1].T @ fockb_ao @ mo_coeff[1]
            e_ia_a = focka_mo.diagonal()[viridxa] - focka_mo.diagonal()[occidxa, None]
            e_ia_b = fockb_mo.diagonal()[viridxb] - fockb_mo.diagonal()[occidxb, None]
        else:
            e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa, None]
            e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb, None]
        e_ia = cp.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
        hdiag = e_ia
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * .8 - mem_now)
        cp.cuda.Stream.null.synchronize()
        tAf1 = time.perf_counter()
        self.tc.Ap_f = tAf1 - tAf0

        tAk0 = time.perf_counter()
        vresp = self.gen_response(hermi=0, max_memory=max_memory)
        cp.cuda.Stream.null.synchronize()
        tAk1 = time.perf_counter()
        self.tc.Ap_k = tAk1 - tAk0

        tdAf0 = time.perf_counter()
        if self.X:
            hf = scf.ROHF(mol)
            veff = hf.get_veff(mol, dm)
            focka_mo_hf = mo_coeff[0].T @ (h1e + veff[0]) @ mo_coeff[0]
            fockb_mo_hf = mo_coeff[1].T @ (h1e + veff[1]) @ mo_coeff[1]
            si = 0.5 * self.mol.spin
        cp.cuda.Stream.null.synchronize()
        tdAf1 = time.perf_counter()
        self.tc.dAp = tdAf1 - tdAf0

        self.tc.Adv = 0.0  # A davidson time consuming
        self.tc.dAdv = 0.0  # Delta A davidson time consuming

        def vind(zs):
            nz = len(zs)
            zs = cp.asarray(zs)

            tAdv0 = time.perf_counter()
            za = zs[:, :nocca * nvira].reshape(nz, nocca, nvira)
            zb = zs[:, nocca * nvira:].reshape(nz, noccb, nvirb)
            mo1a = contract('xov,pv->xpo', za, orbva)
            dmsa = contract('xpo,qo->xpq', mo1a, orboa.conj())
            mo1b = contract('xov,pv->xpo', zb, orbvb)
            dmsb = contract('xpo,qo->xpq', mo1b, orbob.conj())
            dms = cp.asarray((dmsa, dmsb))
            dms = tag_array(dms, mo1=[mo1a, mo1b], occ_coeff=[orboa, orbob])
            v1ao = vresp(dms)
            v1a = contract('xpq,qo->xpo', v1ao[0], orboa)
            v1a = contract('xpo,pv->xov', v1a, orbva.conj())
            v1b = contract('xpq,qo->xpo', v1ao[1], orbob)
            v1b = contract('xpo,pv->xov', v1b, orbvb.conj())
            cp.cuda.Stream.null.synchronize()
            tAdv1 = time.perf_counter()
            self.tc.Adv += tAdv1 - tAdv0

            if self.X:
                tdAdv0 = time.perf_counter()
                v1a += (
                        contract('xib,ab->xia', za, focka_mo[len(occidxa):, len(occidxa):])
                        - contract('xja,ij->xia', za, focka_mo[:len(occidxa), :len(occidxa)])
                )

                # for alpha block, my order is same with pyscf
                # CV(aa)-CV(aa)
                v1a[:, :noccb, :] += (
                    0.5 * (1 - cp.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                        contract('xib,ab->xia', za[:, :noccb, :], fockb_mo_hf[nocca:, nocca:])
                        - contract('xib,ab->xia ', za[:, :noccb, :], focka_mo_hf[nocca:, nocca:])
                    )
                    + 0.5 * (-1 + cp.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                        contract('xja,ij->xia', za[:, :noccb, :], fockb_mo_hf[:noccb, :noccb])
                        - contract('xja,ij->xia ', za[:, :noccb, :], focka_mo_hf[:noccb, :noccb])
                    )
                )

                # for beta block, my order is different with pyscf, so need transfer
                # CV(aa)-CV(bb)
                v1a[:, :noccb, :] -= (
                    0.5 * 1 / (2 * si) * (
                        contract('xib,ab->xia', zb[:, :, -nvira:], fockb_mo_hf[nocca:, nocca:])
                        - contract('xib,ab->xia', zb[:, :, -nvira:], focka_mo_hf[nocca:, nocca:])
                        + contract('xja,ij->xia', zb[:, :, -nvira:], fockb_mo_hf[:noccb, :noccb])
                        - contract('xja,ij->xia', zb[:, :, -nvira:], focka_mo_hf[:noccb, :noccb])
                    )
                )

                v1b += (
                        contract('xib,ab->xia', zb, fockb_mo[len(occidxb):, len(occidxb):])
                        - contract('xja,ij->xia', zb, fockb_mo[:len(occidxb), :len(occidxb)])
                )

                # CV(bb)-CV(aa)
                v1b[:, :, -nvira:] -= (
                    0.5 * 1 / (2 * si) * (
                        contract('xib,ab->xia', za[:, :noccb, :], fockb_mo_hf[nocca:, nocca:])
                        - contract('xib,ab->xia', za[:, :noccb, :], focka_mo_hf[nocca:, nocca:])
                        + contract('xja,ij->xia', za[:, :noccb, :], fockb_mo_hf[:noccb, :noccb])
                        - contract('xja,ij->xia', za[:, :noccb, :], focka_mo_hf[:noccb, :noccb])
                    )
                )

                # for beta block, my order is different with pyscf, so need transfer
                # CV(bb)-CV(bb)
                v1b[:, :, -nvira:] += (
                    0.5 * (-1 + cp.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                        contract('xib,ab->xia', zb[:, :, -nvira:], fockb_mo_hf[nocca:, nocca:])
                        - contract('xib,ab->xia ', zb[:, :, -nvira:], focka_mo_hf[nocca:, nocca:])
                    )
                    + 0.5 * (1 - cp.sqrt((si + 1) / si) + 1 / (2 * si)) * (
                        contract('xja,ij->xia', zb[:, :, -nvira:], fockb_mo_hf[:noccb, :noccb])
                        - contract('xja,ij->xia ', zb[:, :, -nvira:], focka_mo_hf[:noccb, :noccb])
                    )
                )
                cp.cuda.Stream.null.synchronize()
                tdAdv1 = time.perf_counter()
                self.tc.dAdv += tdAdv1 - tdAdv0
            else:
                v1a += contract('xia,ia->xia', za, e_ia_a)
                v1b += contract('xia,ia->xia', zb, e_ia_b)

            hx = cp.hstack((v1a.reshape(nz, -1), v1b.reshape(nz, -1)))
            return hx

        cp.cuda.Stream.null.synchronize()
        tAp1 = time.perf_counter()
        self.tc.Ap = tAp1 - tAp0 - self.tc.dAp
        return vind, hdiag

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        # assert mf is None or mf is self._scf
        assert mf is None or mf is self.mf
        return self._gen_tda_operation()

    def init_guess(self, mf=None, nstates=None, wfnsym=None, return_symmetry=False):
        if mf is None: mf = self.mf
        if nstates is None: nstates = self.nstates
        assert wfnsym is None
        assert not return_symmetry
        mo_energy_a = self.mo_energy[0]
        mo_energy_b = self.mo_energy[1]
        mo_occ_a = self.mo_occ[0]
        mo_occ_b = self.mo_occ[1]

        if isinstance(mo_energy_a, cp.ndarray):
            mo_energy_a = mo_energy_a.get()
            mo_energy_b = mo_energy_b.get()
        if isinstance(mo_occ_a, cp.ndarray):
            mo_occ_a = mo_occ_a.get()
            mo_occ_b = mo_occ_b.get()
        occidxa = mo_occ_a >  0
        occidxb = mo_occ_b >  0
        viridxa = mo_occ_a == 0
        viridxb = mo_occ_b == 0
        e_ia_a = mo_energy_a[viridxa] - mo_energy_a[occidxa,None]
        e_ia_b = mo_energy_b[viridxb] - mo_energy_b[occidxb,None]
        nov = e_ia_a.size + e_ia_b.size
        nstates = min(nstates, nov)

        e_ia = np.append(e_ia_a.ravel(), e_ia_b.ravel())
        # Find the nstates-th lowest energy gap
        e_threshold = np.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = np.where(e_ia <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1
        return x0

    def get_precond(self, hdiag):
        threshold_t=1.0e-4
        def precond(x, e, *args):
            e = e.reshape(-1,1)
            diagd = hdiag - (e-self.level_shift)
            diagd = cp.where(abs(diagd) < threshold_t, cp.sign(diagd)*threshold_t, diagd)
            a_size = x.shape[1]//2
            diagd[:,a_size:] = diagd[:,a_size:]*(-1)
            return x/diagd
        return precond

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        t0 = time.perf_counter()
        print('use cpu Davidson')
        mf = self.mf
        mo_occ = self.mo_occ
        if mf.mo_energy is None:
            mf.run()

        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        vind, hdiag = self.gen_vind(self.mf)

        tih0 = time.perf_counter()  # init guess
        if x0 is None:
            x0 = self.init_guess()
        cp.cuda.Stream.null.synchronize()
        tih1 = time.perf_counter()
        self.tc.ih = tih1 - tih0

        tdv0 = time.perf_counter()
        self.converged, self.e, x1, Davidcyc = Davidson.davidson1(
            vind, x0, hdiag, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, max_cycle=self.max_cycle)
        print('Davidson each state Converged ', self.converged)
        print('Davidson iteration {:2} times'.format(Davidcyc[0]))
        cp.cuda.Stream.null.synchronize()
        tdv1 = time.perf_counter()
        self.tc.dv = tdv1 - tdv0

        ttv0 = time.perf_counter()
        nmo = mo_occ[0].size
        nocca = int((mo_occ[0] > 0).sum())
        noccb = int((mo_occ[1] > 0).sum())
        nvira = nmo - nocca
        nvirb = nmo - noccb
        xy = [((xi[:nocca * nvira].reshape(nocca, nvira),  # X_alpha
                xi[nocca * nvira:].reshape(noccb, nvirb)),  # X_beta
                (0, 0))  # (Y_alpha, Y_beta)
                for xi in x1]
        v = np.empty((xy[0][0][0].reshape(-1).shape[0] + xy[0][0][1].reshape(-1).shape[0], 0))
        for vec in xy:
            x, y = vec
            v = np.concatenate(
                (v, np.expand_dims(np.concatenate((x[0].reshape(-1), x[1].reshape(-1))), axis=1)),
                axis=1
            )
        if isinstance(v, np.ndarray):
            v = cp.asarray(v)
        self.nc = noccb
        self.no = nocca - noccb
        self.nv = nvira
        self.order = utils.order_pyscf2my(noccb, nocca - noccb, nvira)
        self.v = v[self.order, :]  # transform pyscf order to my order
        self.xy_a = self.v.T[:, :nocca * nvira]  # V_alpha
        self.xy_b = self.v.T[:, nocca * nvira:]  # V_beta
        self.xycv_a = self.v.T[:, :noccb * nvira]  # V_cv_a
        self.xyov_a = self.v.T[:, noccb * nvira:nocca * nvira]  # V_ov_a
        self.xyco_b = self.v.T[:, nocca * nvira:nocca * nvira + noccb * (nocca - noccb)]  # V_co_b
        self.xycv_b = self.v.T[:, nocca * nvira + noccb * (nocca - noccb):]  # V_cv_b
        self.occidx_a = cp.where(self.mo_occ[0] >= 1)[0]
        self.viridx_a = cp.where(self.mo_occ[0] == 0)[0]
        self.occidx_b = cp.where(self.mo_occ[1] >= 1)[0]
        self.viridx_b = cp.where(self.mo_occ[1] == 0)[0]
        cp.cuda.Stream.null.synchronize()
        ttv1 = time.perf_counter()
        self.tc.tv = ttv1 - ttv0

        tds20 = time.perf_counter()
        self.dS2 = self.deltaS2()
        cp.cuda.Stream.null.synchronize()
        tds21 = time.perf_counter()
        self.tc.ds2 = tds21 - tds20

        tosc0 = time.perf_counter()
        self.os = self.osc_str()  # oscillator strength
        cp.cuda.Stream.null.synchronize()
        tosc1 = time.perf_counter()
        self.tc.osc = tosc1 - tosc0

        trot0 = time.perf_counter()
        # before calculate rot_str, check whether mol is chiral mol
        if gto.mole.chiral_mol(self.mol):
            self.rs = self.rot_str()
        else:
            self.rs = cp.zeros(self.nstates)
        cp.cuda.Stream.null.synchronize()
        trot1 = time.perf_counter()
        self.tc.rot = trot1 - trot0

        tana0 = time.perf_counter()
        self.analyze()
        cp.cuda.Stream.null.synchronize()
        tana1 = time.perf_counter()
        self.tc.ana = tana1 - tana0
        print('='*50)

        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        self.tc.total = t1 - t0

        print('XTDA result is')
        print(f'{"num":>4} {"energy":>8} {"wav_len":>10} {"osc_str":>8} {"rot_str":>8} {"deltaS2":>8}')
        for ni, ei, wli, osi, rsi, ds2i in zip(
            range(self.nstates), self.e * unit.ha2eV, unit.eVxnm / (self.e * unit.ha2eV), self.os, self.rs, self.dS2
        ):
            print(f'{ni:4d} {ei:8.4f} {wli:10.4f} {osi:8.4f} {rsi:8.4f} {ds2i:8.4f}')
        print('='*50)

        print('initial time consuming')
        print('    perpare A calculation  use              {:12.3f} s'.format(self.tc.Ap))
        print('        Fock (UKS is mo energy)  use        {:12.3f} s'.format(self.tc.Ap_f))
        print('        vxc kernel use                      {:12.3f} s'.format(self.tc.Ap_k))
        print('    perpare delta A calculation (Fock)  use {:12.3f} s'.format(self.tc.dAp))
        # print('    init guess and hdiag use                {:12.3f} s'.format(self.tc.ih))
        print('Davidson process time consuming')
        print('    calculate Ax use                        {:12.3f} s'.format(self.tc.Adv))
        print('        calculate A vxc use                 {:12.3f} s'.format(self.tc.A_vxc))
        print('        calculate A get_k use               {:12.3f} s'.format(self.tc.A_gk))
        print('    calculate delta Ax use                  {:12.3f} s'.format(self.tc.dAdv))
        print('    other davidson process use              {:12.3f} s'.format(self.tc.dv - self.tc.Adv - self.tc.dAdv))
        print('each Davidson iteration use                 {:12.3f} s'.format(self.tc.dv / Davidcyc[0]))
        print('property calculation')
        print('    transform pyscf order to my order use   {:12.3f} s'.format(self.tc.tv))
        print('    oscillator strength use                 {:12.3f} s'.format(self.tc.osc))
        print('    rotator strength use                    {:12.3f} s'.format(self.tc.rot))
        print('    analyze use                             {:12.3f} s'.format(self.tc.ana))
        print('X-TDA total use                             {:12.3f} s'.format(self.tc.total))
        return self.e, v

    def deltaS2(self):
        # refer to J. Chem. Phys. 134, 134101 (2011), ignore some term and some term cancel each other out
        if self.X:
            dS2 = (cp.einsum('ij,ij->i', self.xycv_a, self.xycv_a)
                   + cp.einsum('ij,ij->i', self.xycv_b, self.xycv_b)
                   - 2 * cp.einsum('ij,ij->i', self.xycv_a, self.xycv_b))
        else:
            orbo_a = self.mo_coeff[0][:, self.occidx_a]
            orbv_a = self.mo_coeff[0][:, self.viridx_a]
            orbo_b = self.mo_coeff[1][:, self.occidx_b]
            orbv_b = self.mo_coeff[1][:, self.viridx_b]
            S = self.mol.intor('int1e_ovlp')
            Sccba = cp.einsum('pq,pi,qj->ij', S, orbo_b, orbo_a)
            Sccab = cp.einsum('pq,pi,qj->ij', S, orbo_a, orbo_b)
            Svcab = cp.einsum('pq,pi,qj->ij', S, orbv_a, orbo_b)
            Svcba = cp.einsum('pq,pi,qj->ij', S, orbv_b, orbo_a)
            Svvab = cp.einsum('pq,pi,qj->ij', S, orbv_a, orbv_b)
            xycv_a = self.xycv_a.reshape(self.nstates, self.nc, self.nv)
            xyov_a = self.xyov_a.reshape(self.nstates, self.no, self.nv)
            xyco_b = self.xyco_b.reshape(self.nstates, self.nc, self.no)
            xycv_b = self.xycv_b.reshape(self.nstates, self.nc, self.nv)
            dS2 = (
                cp.einsum('nia,nja,ki,jk->n', xycv_a, xycv_a, Sccba[:, :self.nc], Sccba.T[:self.nc, :])  # first term cvacva
                + cp.einsum('nia,nja,ki,jk->n', xyov_a, xyov_a, Sccba[:, self.nc:], Sccba.T[self.nc:, :])  # first term ovaova
                + cp.einsum('nia,nja,ki,jk->n', xyov_a, xycv_a, Sccba[:, self.nc:], Sccba.T[:self.nc, :])  # first term ovacva
                + cp.einsum('nia,nja,ki,jk->n', xycv_a, xyov_a, Sccba[:, :self.nc], Sccba.T[self.nc:, :])  # first term cvaova
                - cp.einsum('nia,nib,ak,kb->n', xycv_a, xycv_a, Svcab, Svcab.T)  # second term  cva
                - cp.einsum('nia,nib,ak,kb->n', xyov_a, xyov_a, Svcab, Svcab.T)  # second term ova
                + cp.einsum('nia,nja,ki,jk->n', xycv_b, xycv_b, Sccab, Sccab.T)  # third term  cvb
                + cp.einsum('nia,nja,ki,jk->n', xyco_b, xyco_b, Sccab, Sccab.T)  # third term cob
                - cp.einsum('nia,nib,ak,kb->n', xyco_b, xyco_b, Svcba[:self.no, :], Svcba.T[:, :self.no])  # forth term cobcob
                - cp.einsum('nia,nib,ak,kb->n', xycv_b, xycv_b, Svcba[self.no:, :], Svcba.T[:, self.no:])  # forth term cvbcvb
                - cp.einsum('nia,nib,ak,kb->n', xyco_b, xycv_b, Svcba[:self.no, :], Svcba.T[:, self.no:])  # forth term cobcvb
                - cp.einsum('nia,nib,ak,kb->n', xycv_b, xyco_b, Svcba[self.no:, :], Svcba.T[:, :self.no])  # forth term cvbcob
                - 2 * cp.einsum('nia,njb,ji,ab->n', xycv_a, xycv_b, Sccba[:, :self.nc], Svvab[:, self.no:])  # fifth term cvacvb
                - 2 * cp.einsum('nia,njb,ji,ab->n', xycv_a, xyco_b, Sccba[:, :self.nc], Svvab[:, :self.no])  # fifth term cvacob
                - 2 * cp.einsum('nia,njb,ji,ab->n', xyov_a, xycv_b, Sccba[:, self.nc:], Svvab[:, self.no:])  # fifth term ovacvb
                - 2 * cp.einsum('nia,njb,ji,ab->n', xyov_a, xyco_b, Sccba[:, self.nc:], Svvab[:, :self.no])  # fifth term ovacob
            )
        return dS2

    def osc_str(self):
        # oscillator strength
        # f = 2/3 \omega_{0\nu} | \langle 0|\sum_s r_s |\nu\rangle |^2  # length form
        # f = 2/3 \omega_{0\nu}^{-1} | \langle 0|\sum_s \nabla_s |\nu\rangle |^2  # velocity form
        orbo_a = self.mo_coeff[0][:, self.occidx_a]
        orbv_a = self.mo_coeff[0][:, self.viridx_a]
        orbo_b = self.mo_coeff[1][:, self.occidx_b]
        orbv_b = self.mo_coeff[1][:, self.viridx_b]
        omega = self.e[:self.nstates]
        # length form oscillator strength
        dipole_ao = self.mol.intor_symmetric("int1e_r", comp=3)  # dipole moment, comp=3 is 3 axis
        # # these two line use too many VRAM
        # dipole_mo_a = cp.einsum('xpq,pi,qj->xij', dipole_ao, orbo_a, orbv_a)
        # dipole_mo_b = cp.einsum('xpq,pi,qj->xij', dipole_ao, orbo_b, orbv_b)
        dipole_mo_a = contract('xpq,pi->xqi', dipole_ao, orbo_a)
        dipole_mo_a = contract('xqi,qj->xij', dipole_mo_a, orbv_a)
        dipole_mo_b = contract('xpq,pi->xqi', dipole_ao, orbo_b)
        dipole_mo_b = contract('xqi,qj->xij', dipole_mo_b, orbv_b)
        dipole_mo_a = dipole_mo_a.reshape(3, -1)
        dipole_mo_b = dipole_mo_b.reshape(3, -1)[:, self.order[(self.nc+self.no)*self.nv:]-(self.nc+self.no)*self.nv]
        trans_dip_a = cp.einsum('xi,yi->yx', dipole_mo_a, self.xy_a)
        trans_dip_b = cp.einsum('xi,yi->yx', dipole_mo_b, self.xy_b)
        trans_dip = trans_dip_a + trans_dip_b
        f = 2. / 3. * cp.einsum('s,sx,sx->s', omega, trans_dip, trans_dip)
        # cp.save("osc_str_stda.npy", f)
        return f

    def rot_str(self):
        # rotator strength
        # f = 2/3 \omega_{0\nu} | \langle 0|\sum_s r_s |\nu\rangle |^2  # length form
        # f = 2/3 \omega_{0\nu}^{-1} | \langle 0|\sum_s \nabla_s |\nu\rangle |^2  # velocity form
        orbo_a = self.mo_coeff[0][:, self.occidx_a]
        orbv_a = self.mo_coeff[0][:, self.viridx_a]
        orbo_b = self.mo_coeff[1][:, self.occidx_b]
        orbv_b = self.mo_coeff[1][:, self.viridx_b]
        omega = self.e[:self.nstates]
        dip_ele_ao = self.mol.intor('int1e_ipovlp', comp=3, hermi=2)  # transition electric dipole moment
        # dip_ele_mo_a = cp.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo_a, orbv_a)
        # dip_ele_mo_b = cp.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo_b, orbv_b)
        dip_ele_mo_a = contract('xpq,pi->xqi', dip_ele_ao, orbo_a)
        dip_ele_mo_a = contract('xqi,qj->xij', dip_ele_mo_a, orbv_a)
        dip_ele_mo_b = contract('xpq,pi->xqi', dip_ele_ao, orbo_b)
        dip_ele_mo_b = contract('xqi,qj->xij', dip_ele_mo_b, orbv_b)
        dip_meg_ao = self.mol.intor('int1e_cg_irxp', comp=3, hermi=2)  # transition magnetic dipole moment
        # dip_meg_mo_a = cp.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo_a, orbv_a)
        # dip_meg_mo_b = cp.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo_b, orbv_b)
        dip_meg_mo_a = contract('xpq,pi->xqi', dip_meg_ao, orbo_a)
        dip_meg_mo_a = contract('xqi,qj->xij', dip_meg_mo_a, orbv_a)
        dip_meg_mo_b = contract('xpq,pi->xqi', dip_meg_ao, orbo_b)
        dip_meg_mo_b = contract('xqi,qj->xij', dip_meg_mo_b, orbv_b)

        dip_ele_mo_a = dip_ele_mo_a.reshape(3, -1)
        dip_ele_mo_b = dip_ele_mo_b.reshape(3, -1)[:,self.order[(self.nc + self.no) * self.nv:] - (self.nc + self.no) * self.nv]
        trans_ele_dip_a = cp.einsum('xi,yi->yx', dip_ele_mo_a, self.xy_a)
        trans_ele_dip_b = cp.einsum('xi,yi->yx', dip_ele_mo_b, self.xy_b)
        trans_ele_dip = -(trans_ele_dip_a + trans_ele_dip_b)
        dip_meg_mo_a = dip_meg_mo_a.reshape(3, -1)
        dip_meg_mo_b = dip_meg_mo_b.reshape(3, -1)[:,self.order[(self.nc + self.no) * self.nv:] - (self.nc + self.no) * self.nv]
        trans_meg_dip_a = cp.einsum('xi,yi->yx', dip_meg_mo_a, self.xy_a)
        trans_meg_dip_b = cp.einsum('xi,yi->yx', dip_meg_mo_b, self.xy_b)
        trans_meg_dip = 0.5 * (trans_meg_dip_a + trans_meg_dip_b)
        # in Gaussian and ORCA, do not multiply constant
        # f = 1./unit.c * cp.einsum('s,sx,sx->s', 1./omega, trans_ele_dip, trans_meg_dip)
        f = cp.einsum('s,sx,sx->s', 1. / omega, trans_ele_dip, trans_meg_dip)
        f = f / unit.cgs2au  # transform atom unit to cgs unit
        # cp.save("rot_str_stda.npy", f)
        return f

    def analyze(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        if self.so2st:
            self.v = utils.so2st(self.v, nc, no, nv)
        else:
            pass
        for nstate in range(self.nstates):
            print('-' * 50)
            value = self.v[:, nstate]
            x_cv_aa = value[:nc * nv].reshape(nc, nv)
            x_ov_aa = value[nc * nv:(nc + no) * nv].reshape(no, nv)
            x_co_bb = value[(nc + no) * nv:(nc + no) * nv + nc * no].reshape(nc, no)
            x_cv_bb = value[(nc + no) * nv + nc * no:].reshape(nc, nv)
            print(
                f'D{nstate + 1}' + r"    w:" + f'{self.e[nstate] * unit.ha2eV:10.4f} eV'
                + r"    d<S^2>:" + f'{self.dS2[nstate]:8.4f}'
                + r"    f:" + f'{self.os[nstate]:8.4f}'
            )
            if self.so2st:
                for o, v in zip(*cp.where(abs(x_cv_aa) > 0.1)):
                    print(
                        f'    CV(0) {o + 1:3d} -> {v + 1 + nc + no:3d}    c_i: {x_cv_aa[o, v]:8.5f}    Per: {100 * x_cv_aa[o, v] ** 2:5.2f}%')
                for o, v in zip(*cp.where(abs(x_ov_aa) > 0.1)):
                    print(
                        f'    OV(0) {nc + o + 1:3d} -> {v + 1 + nc + no:3d}    c_i: {x_ov_aa[o, v]:8.5f}    Per: {100 * x_ov_aa[o, v] ** 2:5.2f}%')
                for o, v in zip(*cp.where(abs(x_co_bb) > 0.1)):
                    print(
                        f'    CO(0) {o + 1:3d} -> {v + 1 + nc:3d}    c_i: {x_co_bb[o, v]:8.5f}    Per: {100 * x_co_bb[o, v] ** 2:5.2f}%')
                for o, v in zip(*cp.where(abs(x_cv_bb) > 0.1)):
                    print(
                        f'    CV(1) {o + 1:3d} -> {v + 1 + nc + no:3d}    c_i: {x_cv_bb[o, v]:8.5f}    Per: {100 * x_cv_bb[o, v] ** 2:5.2f}%')
            else:
                for o, v in zip(*cp.where(abs(x_cv_aa) > 0.1)):
                    print(
                        f'    CV(aa) {o + 1:3d} -> {v + 1 + nc + no:3d}    c_i: {x_cv_aa[o, v]:8.5f}    Per: {100 * x_cv_aa[o, v] ** 2:5.2f}%')
                for o, v in zip(*cp.where(abs(x_ov_aa) > 0.1)):
                    print(
                        f'    OV(aa) {nc + o + 1:3d} -> {v + 1 + nc + no:3d}    c_i: {x_ov_aa[o, v]:8.5f}    Per: {100 * x_ov_aa[o, v] ** 2:5.2f}%')
                for o, v in zip(*cp.where(abs(x_co_bb) > 0.1)):
                    print(
                        f'    CO(bb) {o + 1:3d} -> {v + 1 + nc:3d}    c_i: {x_co_bb[o, v]:8.5f}    Per: {100 * x_co_bb[o, v] ** 2:5.2f}%')
                for o, v in zip(*cp.where(abs(x_cv_bb) > 0.1)):
                    print(
                        f'    CV(bb) {o + 1:3d} -> {v + 1 + nc + no:3d}    c_i: {x_cv_bb[o, v]:8.5f}    Per: {100 * x_cv_bb[o, v] ** 2:5.2f}%')


if __name__ == "__main__":
    mol = gto.M(
        atom='H 0 0 0; F 0 0 1.1',
        # atom = atom.ttm_vacuum,
        # atom = atom.hhcrqpp2,
        # basis='631g',
        # basis='def2-tzvp',
        # basis='aug-cc-pvtz',
        # basis='6-31g',
        # basis='cc-pvdz',
        unit='A',
        # unit='B',
        charge=0,
        spin=0,
        verbose=4
    )
    # path = '/home/whb/Documents/TDDFT/orcabasis/'
    # # bse.convert_formatted_basis_file(path+'sto-3g.bas', path+'sto-3g.nw')
    # # with open(path + "sto-3g.nw", "r") as f:
    # # bse.convert_formatted_basis_file(path+'tzvp.bas', path+'tzvp.nw')
    # with open(path + "tzvp.nw", "r") as f:
    #     basis = f.read()
    # mol.basis = basis
    # mol.build()

    # # add solvents
    # # t_dft0 = time.time()
    # mf = dft.ROKS(mol).SMD()
    # # mf.with_solvent.method = 'COSMO'  # C-PCM, SS(V)PE, COSMO, IEF-PCM
    # # in https://gaussian.com/scrf/ solvents entry, give different eps for different solvents
    # mf.with_solvent.eps = 2.0165  # for Cyclohexane 环己烷
    # # mf.with_solvent.eps = 4.2400  # for DiethylEther 乙醚
    # # mf.with_solvent.eps = 7.4257  # for TetraHydroFuran 四氢呋喃

    tscf0 = time.perf_counter()
    # use ROKS is for use ROKS orbital, in paper assume ROKS orbital same with UKS orbital
    mf = dft.ROKS(mol)
    # xc = 'svwn'
    # xc = 'blyp'
    xc = 'b3lyp'
    # xc = 'cam-b3lyp'
    # xc = 'wb97xd'
    # xc = '0.50*HF + 0.50*B88 + GGA_C_LYP'  # BHHLYP
    # xc = 'pbe0'
    # xc = 'pbe38'
    # xc = 'hf'
    mf.xc = xc
    # xc = 'bhhlyp'
    # mf.conv_tol = 1e-8
    # mf.conv_tol_grad = 1e-5
    mf.max_cycle = 200
    mf.level_shift = 0.6
    mf.kernel()
    cp.cuda.Stream.null.synchronize()
    tscf1 = time.perf_counter()
    print('scf use      {:8.4f} s'.format(tscf1 - tscf0))
    print('=' * 50)

    if isinstance(mf, rohf.ROHF):
        print('num.orb    mo_energy     mo_occ')
        for me, o in zip(enumerate(mf.mo_energy), mf.mo_occ):
            ind, moei = me
            print(f'{ind + 1:5d}    {moei:10.6f}    {o:8.3f}')
        print('=' * 50)
        print('there are {:8} orbitals (basis)'.format(len(mf.mo_occ)))
    elif isinstance(mf, uhf.UHF):
        print('num.orb    mo_energy_a   mo_energy_b    mo_occ_a    mo_occ_b')
        for me, o in zip(enumerate(mf.mo_energy.T), mf.mo_occ.T):
            ind, moei = me
            print(f'{ind + 1:5d}    {moei[0]:10.6f}    {moei[1]:10.6f}    {o[0]:8.3f}    {o[1]:8.3f}')
        print('=' * 50)
        print('there are {:8} orbitals (basis)'.format(mf.mo_occ.shape[1]))
    else:
        raise ValueError

    xtda = XTDA(mol, mf)
    xtda.nstates = 7
    xtda.kernel()

