#!/usr/bin/env python
import sys
sys.path.append('/')
import time
import scipy
import numpy as np
import pandas as pd
import basis_set_exchange as bse
from pyscf import dft, gto, scf, tddft, ao2mo, lib
from pyscf.lib import logger
from utils import atom, unit
'''
most coda refer to pyscf code
'''


class TDA:
    def __init__(self, mol, mf, nstates=5, savedata=False, singlet=True):
        self.mol = mol
        self.mf = mf
        self.singlet = singlet
        self.nstates = nstates
        self.savedata = savedata

    def pyscf_tda(self, conv_tol=1e-6, is_analyze=False):
        td = tddft.TDA(self.mf)
        td.singlet = self.singlet
        td.nstates = self.nstates  # calculate nstates excited states
        td.conv_tol = conv_tol  # pyscf default is 1e-5, not 1e-9
        # td.deg_eia_thresh = 1e-9
        e, xy = td.kernel()
        # transform tuple xy to array xy
        v = np.empty((xy[0][0].reshape(-1).shape[0], 0))
        for vec in xy:
            x, y = vec
            v = np.concatenate((v, np.expand_dims(x.reshape(-1), axis=1)), axis=1)
        if is_analyze:
            td.analyze()
        # os = td.oscillator_strength(gauge="velocity")
        os = td.oscillator_strength(gauge="length")
        logger.info(self.mf, "oscillator strength \n{}".format(os))
        # np.save("energy_tda.npy", e)
        # np.save("osc_str_tda.npy", os)
        return e, os, v
        # # use this file can input into multiwfn to check molecular orbital
        # from pyscf.tools import cubegen
        # cubegen.orbital(mol, 'mo11.cube', mf.mo_coeff[:, 10])
        # cubegen.orbital(mol, 'mo6.cube', mf.mo_coeff[:, 5])

    def kernel(self):
        # most code copy from pyscf/tdscf/rhf.py
        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        assert (mo_coeff.dtype == np.double)
        mo_energy = self.mf.mo_energy
        mo_occ = self.mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:, viridx]
        orbo = mo_coeff[:, occidx]
        mo = np.hstack((orbo, orbv))

        e_ia = mo_energy[viridx] - mo_energy[occidx, None]
        a = np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)

        # a = np.zeros((nocc, nvir, nocc, nvir))  # two-electron integral and E_{XC}
        # dm = self.mf.make_rdm1()
        # vhf = self.mf.get_veff(mf.mol, dm)
        # h1e = self.mf.get_hcore()
        # fock = h1e + vhf
        # fock = mo_coeff.T @ fock @ mo_coeff
        # delta_ij = np.eye(nocc)
        # delta_ab = np.eye(nvir)

        def add_hf_(a, hyb=1):
            # hyb=1 is CIS
            eri_mo = ao2mo.general(mol, [mo, mo, mo, mo], compact=False)
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            if self.singlet:
                a += np.einsum('iabj->iajb', eri_mo[:nocc, nocc:, nocc:, :nocc]) * 2
            a -= np.einsum('ijba->iajb', eri_mo[:nocc, :nocc, nocc:, nocc:]) * hyb

        ni = self.mf._numint
        ni.libxc.test_deriv_order(self.mf.xc, 2, raise_error=True)
        if self.mf.do_nlc():
            logger.warn(self.mf, 'NLC functional found in DFT object.  Its second '
                            'derivative is not available. Its contribution is '
                            'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, mol.spin)
        # print('omega alpha hyb', omega, alpha, hyb)
        add_hf_(a, hyb)

        xctype = ni._xc_type(self.mf.xc)
        dm0 = self.mf.make_rdm1(mo_coeff, mo_occ)
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.mf.max_memory * .8 - mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, self.mf.grids, nao, ao_deriv, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                rho *= .5
                rho = np.repeat(rho[np.newaxis], 2, axis=0)
                fxc = ni.eval_xc_eff(self.mf.xc, rho, deriv=2, xctype=xctype)[2]
                # below 'if' is inspired by pyscf/dft/numint.py 'nr_rks_fxc_st'
                if self.singlet:
                    fxc = 0.5 * (fxc[0, :, 0] + fxc[0, :, 1])
                else:
                    fxc = 0.5 * (fxc[0, :, 0] - fxc[0, :, 1])
                wfxc = fxc[0, 0] * weight
                rho_o = lib.einsum('rp,pi->ri', ao, orbo)
                rho_v = lib.einsum('rp,pi->ri', ao, orbv)
                rho_ov = np.einsum('ri,ra->ria', rho_o, rho_v)
                w_ov = np.einsum('ria,r->ria', rho_ov, wfxc)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov, w_ov) * 2
                a += iajb
                # b += iajb

        elif xctype == 'GGA':
            ao_deriv = 1
            # self.mf.grids.level = 3  # set level to reduce memory, default is 3
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, self.mf.grids, nao, ao_deriv, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                rho *= .5
                rho = np.repeat(rho[np.newaxis], 2, axis=0)
                fxc = ni.eval_xc_eff(self.mf.xc, rho, deriv=2, xctype=xctype)[2]
                # below 'if' is inspired by pyscf/dft/numint.py 'nr_rks_fxc_st'
                if self.singlet:
                    fxc = 0.5 * (fxc[0, :, 0] + fxc[0, :, 1])
                else:
                    fxc = 0.5 * (fxc[0, :, 0] - fxc[0, :, 1])
                wfxc = fxc * weight
                rho_o = lib.einsum('xrp,pi->xri', ao, orbo)
                rho_v = lib.einsum('xrp,pi->xri', ao, orbv)
                rho_ov = np.einsum('xri,ra->xria', rho_o, rho_v[0])
                rho_ov[1:4] += np.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                w_ov = np.einsum('xyr,xria->yria', wfxc, rho_ov)
                iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov) * 2
                a += iajb
                # b += iajb

        self.nc = nocc
        self.nv = nvir
        dim = self.nc * self.nv
        A = np.zeros((dim, dim))

        # # Construct A matrix
        # # CV-CV
        # A += (
        #     (np.einsum('ij,ab -> iajb', delta_ij, fock[self.nc:, self.nc:])
        #     - np.einsum('ab,ij->iajb', delta_ab, fock[:self.nc, :self.nc])
        #     + a)
        # ).reshape(self.nc * self.nv, self.nc * self.nv)
        A += a.reshape(self.nc * self.nv, self.nc * self.nv)

        # # upper code to get A is same as below code, but below code inconvenient to modify
        # A, B = tddft.rhf.get_ab(mf)
        # A = A.reshape(nocc * nvir, nocc * nvir)

        self.e, self.v = scipy.linalg.eigh(A)
        # self.e = self.e[np.where(self.e>0)[0]]  # avoid appear negetive value
        # self.v = self.v[:, np.where(self.e>0)[0]]
        self.e_eV = self.e[:self.nstates] * unit.ha2eV
        # self.e_eV = unit.eVxnm/self.e_eV
        # logger.info(self.mf, "my tda result is \n{}".format(self.e_eV))
        self.v = self.v[:, :self.nstates]
        if self.singlet:
            os = self.osc_str()  # oscillator strength
            # # before calculate rot_str, check whether mol is chiral mol
            if gto.mole.chiral_mol(self.mol):
                rs = self.rot_str()
            else:
                # molecule do not have chiral
                rs = np.zeros(self.nstates)
        else:
            # transitions forbidden
            os = np.zeros(self.nstates)
            rs = np.zeros(self.nstates)
        print('TDA result is')
        print(f'{"num":>4} {"energy":>8} {"wav_len":>8} {"osc_str":>8} {"rot_str":>8}')
        for ni, ei, wli, osi, rsi in zip(range(self.nstates), self.e_eV, unit.eVxnm / self.e_eV, os, rs):
            print(f'{ni:4d} {ei:8.4f} {wli:8.4f} {osi:8.4f} {rsi:8.4f}')
        if self.savedata:
            pd.DataFrame(
                np.concatenate((unit.eVxnm / np.expand_dims(self.e_eV, axis=1), np.expand_dims(os, axis=1)), axis=1)
            ).to_csv('uvspec_data.csv', index=False, header=None)
        # logger.info(self.mf, 'oscillator strength \n{}'.format(self.os))
        # logger.info(self.mf, 'rotatory strength (cgs unit) \n{}'.format(self.rs))
        return self.e_eV, os, rs, self.v

    def osc_str(self):
        # oscillator strength, only implement length form
        # f = 2/3 \omega_{0\nu} | \langle 0|\sum_s r_s |\nu\rangle |^2  # length form
        # f = 2/3 \omega_{0\nu}^{-1} | \langle 0|\sum_s \nabla_s |\nu\rangle |^2  # velocity form
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        orbv = mo_coeff[:, viridx]
        orbo = mo_coeff[:, occidx]
        omega = self.e[:self.nstates]
        # pyscf X^{+}X=1/2, but here X^{+}X=1, so to make result same, here multiple a constant
        xy = self.v.T.reshape(self.nstates, self.nc, self.nv) * np.sqrt(2)  # '.T' make 'reshape' is right

        # length form oscillator strength
        dipole_ao = self.mol.intor_symmetric("int1e_r", comp=3)  # dipole moment, comp=3 is 3 axis
        dipole_mo = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo, orbv)
        trans_dip = np.einsum('xij,yij->yx', dipole_mo, xy)
        f = 2. / 3. * np.einsum('s,sx,sx->s', omega, trans_dip, trans_dip)
        # np.save("osc_str_tda.npy", f)

        # # velocity form oscillator strength
        # dipole_ao = self.mol.intor('int1e_ipovlp', comp=3, hermi=2)  # dipole moment, (nabla \|\)
        # dipole_mo = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo, orbv)
        # trans_dip = -np.einsum('xij,yij->yx', dipole_mo, xy)  # Note: dipole is (-i nabla \|\), so here have a '-'
        # f = 2. / 3. * np.einsum('s,sx,sx->s', 1. / omega, trans_dip, trans_dip)
        return f

    def rot_str(self):
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        orbv = mo_coeff[:, viridx]
        orbo = mo_coeff[:, occidx]
        omega = self.e[:self.nstates]
        # pyscf X^{+}X=1/2, but here X^{+}X=1, so to make result same, here multiple a constant
        xy = self.v.T.reshape(self.nstates, self.nc, self.nv) * np.sqrt(2)  # '.T' make 'reshape' is right
        # dipole_ao = self.mol.intor_symmetric("int1e_r", comp=3)  # dipole moment, comp=3 is 3 axis
        # dipole_mo = np.einsum('xpq,pi,qj->xij', dipole_ao, orbo, orbv)
        # trans_ele_dip = np.einsum('xij,yij->yx', dipole_mo, xy)  # length form
        dip_ele_ao = self.mol.intor('int1e_ipovlp', comp=3, hermi=2)  # transition electric dipole moment
        dip_ele_mo = np.einsum('xpq,pi,qj->xij', dip_ele_ao, orbo, orbv)
        trans_ele_dip = -np.einsum('xij,yij->yx', dip_ele_mo, xy)  # velocity form
        dip_meg_ao = self.mol.intor('int1e_cg_irxp', comp=3, hermi=2)  # transition magnetic dipole moment
        dip_meg_mo = np.einsum('xpq,pi,qj->xij', dip_meg_ao, orbo, orbv)
        trans_meg_dip = 0.5 * np.einsum('xij,yij->yx', dip_meg_mo, xy)  # remove '-' to keep same with Gaussian and ORCA

        # in Gaussian and ORCA, do not multiply constant
        # f = 1./unit.c * np.einsum('s,sx,sx->s', 1./omega, trans_ele_dip, trans_meg_dip)
        f = np.einsum('s,sx,sx->s', 1. / omega, trans_ele_dip, trans_meg_dip)
        f = f / unit.cgs2au  # transform atom unit to cgs unit
        # np.save("rot_str_tda.npy", f)
        return f

    def analyze(self):
        nc = self.nc
        nv = self.nv
        for nstate in range(self.nstates):
            value = self.v[:,nstate]
            x_cv_ab = value[:nc*nv].reshape(nc,nv)
            print(f'Excited state {nstate + 1} {self.e[nstate] * unit.ha2eV:10.5f} eV')
            for o, v in zip(*np.where(abs(x_cv_ab) > 0.1)):
                print(f'CV {o + 1}a -> {v + 1 + nc}a {x_cv_ab[o, v]:10.5f}')


if __name__ == "__main__":
    mol = gto.M(
        # atom=atom.ch2o,
        atom=atom.n2,  # unit is Angstrom
        basis='cc-pvdz',
        # basis='sto3g',
        # unit='B',
        unit='A',
        spin=0,
        max_memory=5000,
        verbose=4,
    )
    # # # bse.convert_formatted_basis_file('../orcabasis/sto-3g.bas', '../orcabasis/sto-3g.nw')
    # # with open("../orcabasis/sto-3g.nw", "r") as f:  # 打开文件
    # with open("../orcabasis/tzvp.nw", "r") as f:  # 打开文件
    #     basis = f.read()  # 读取文件
    # mol.basis = basis
    # mol.build()

    mf = dft.RKS(mol)
    # xc = 'svwn'
    # xc = 'blyp'
    # xc = 'b3lyp'
    xc = 'pbe0'
    # xc = 'pbe38'
    # xc = 'hf'
    mf.xc = xc
    mf.conv_tol = 1e-12  # set small to make pyscf result same with my result
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    # mf.grids.level = 9
    mf.conv_check = False
    mf.kernel()

    singlet = True
    is_analyze = False

    tda = TDA(mol, mf, singlet=singlet, nstates=10)
    tda.nstates = 12
    tda.pyscf_tda(conv_tol=1e-6, is_analyze=True)

    print("="*50)
    t0 = time.time()
    e_eV, os, rs, v = tda.kernel(is_analyze=is_analyze)
    t1 = time.time()
    print('tda use {} s'.format(t1 - t0))

    # import pandas as pd
    # pd.DataFrame(e_eV).to_csv(xc + 'TDA.csv')
    # tda.spec(os)
    # tda.spec(rs)
