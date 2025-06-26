# spin-conserving spin-adapted x-tda

from pyscf import gto, dft, scf, ao2mo, lib, tddft
import numpy as np
import scipy


def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return np.einsum('z,zr->r', charges, coords)/charges.sum()

class X_TDA():
    def __init__(self,mf,s_tda=False):
        
        if np.array(mf.mo_coeff).ndim==3:# UKS
            mo_energy = mf.mo_energy
            mo_coeff = mf.mo_coeff
            mo_occ = mf.mo_occ
        else: # ROKS
            mo_energy = np.array([mf.mo_energy, mf.mo_energy])
            mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            mo_occ = np.zeros((2,len(mf.mo_coeff)))
            mo_occ[0][np.where(mf.mo_occ>=1)[0]]=1
            mo_occ[1][np.where(mf.mo_occ>=2)[0]]=1
            
        self.s_tda = s_tda
        self.mol = mf.mol
        self.mf = mf
        nao = self.mol.nao_nr()
        occidx_a = np.where(mo_occ[0]==1)[0]
        viridx_a = np.where(mo_occ[0]==0)[0]
        occidx_b = np.where(mo_occ[1]==1)[0]
        viridx_b = np.where(mo_occ[1]==0)[0]
        orbo_a = mo_coeff[0][:,occidx_a]
        orbv_a = mo_coeff[0][:,viridx_a]
        orbo_b = mo_coeff[1][:,occidx_b]
        orbv_b = mo_coeff[1][:,viridx_b]
        nocc_a = orbo_a.shape[1]
        nvir_a = orbv_a.shape[1]
        nocc_b = orbo_b.shape[1]
        nvir_b = orbv_b.shape[1]
        mo_a = np.hstack((orbo_a,orbv_a))
        mo_b = np.hstack((orbo_b,orbv_b))
        nmo_a = nocc_a + nvir_a
        nmo_b = nocc_b + nvir_b
        
        self.eri_mo = ao2mo.general(self.mol, [mo_a,mo_a,mo_a,mo_a], compact=False) # iajb
        self.eri_mo = self.eri_mo.reshape(nmo_a,nmo_a,nmo_a,nmo_a)
        self.nc = nocc_b
        self.nv = nvir_a
        self.no = abs(nvir_a-nvir_b)
        print('nc ',self.nc)
        print('no ',self.no)
        print('nv ',self.nv)
        
        dm = mf.make_rdm1()
        vhf = mf.get_veff(mf.mol, dm)
        h1e = mf.get_hcore()
        focka = mo_coeff[0].T @ (h1e+vhf[0]) @ mo_coeff[0]
        fockb = mo_coeff[1].T @ (h1e+vhf[1]) @ mo_coeff[1]

        fab_a = focka[nocc_a:, nocc_a:]
        fab_b = fockb[nocc_b:, nocc_b:]
        fij_a = focka[:nocc_a, :nocc_a]
        fij_b = fockb[:nocc_b, :nocc_b]
        
        if not s_tda:
            hf = scf.ROHF(self.mol)
            veff = hf.get_veff(self.mol, dm)
            focka2 = mo_coeff[0].T @ (h1e+veff[0]) @ mo_coeff[0]
            fockb2 = mo_coeff[1].T @ (h1e+veff[1]) @ mo_coeff[1]
            fab_a2 = focka2[nocc_a:, nocc_a:]
            fab_b2 = fockb2[nocc_b:, nocc_b:]
            fij_a2 = focka2[:nocc_a, :nocc_a]
            fij_b2 = fockb2[:nocc_b, :nocc_b]
        
        def get_kxc():
            occ = np.zeros((2, len(mo_occ)))
            occ[0][:len(occidx_a)] = 1
            occ[1][:len(occidx_b)] = 1

            
            from pyscf.dft import xc_deriv
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            if mf.nlc or ni.libxc.is_nlc(mf.xc):
                logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                            'deriviative is not available. Its contribution is '
                            'not included in the response function.')
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
            print('omega alpha hyb',omega, alpha, hyb)

            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1(mf.mo_coeff, mo_occ)
            #dm0 = mf.make_rdm1(mo_coeff0, mo_occ)
            if np.array(mf.mo_coeff).ndim==2:
                dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
                #dm0.mo_coeff = (mo_coeff0, mo_coeff0)
            make_rho = ni._gen_rho_evaluator(mf.mol, dm0, hermi=1, with_lapl=False)[0]
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)
            print('xctype',xctype)
            
            if xctype == 'LDA':
                ao_deriv = 0
                for ao, mask, weight, coords \
                        in ni.block_loop(self.mol, mf.grids, nao, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, xctype)
                    rho0b = make_rho(1, ao, mask, xctype)
                    rho = (rho0a, rho0b)
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]#(2,1,2,1,N)
                    wfxc = fxc[:,0,:,0] * weight #(2,2,N)

                    rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                    rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                    rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                    rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                    rho_ov_a = np.einsum('ri,ra->ria', rho_o_a, rho_v_a)
                    rho_ov_b = np.einsum('ri,ra->ria', rho_o_b, rho_v_b)

                    w_ov = np.einsum('ria,r->ria', rho_ov_a, wfxc[0,0])
                    iajb = lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov)
                    a_aa = iajb
                    #b_aa = iajb

                    w_ov = np.einsum('ria,r->ria', rho_ov_b, wfxc[0,1])
                    iajb = lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov)
                    a_ab = iajb
                    #b_ab = iajb

                    w_ov = np.einsum('ria,r->ria', rho_ov_b, wfxc[1,1])
                    iajb = lib.einsum('ria,rjb->iajb', rho_ov_b, w_ov)
                    a_bb = iajb
                    #b_bb = iajb

            elif xctype == 'GGA':
                ao_deriv = 1
                for ao, mask, weight, coords \
                        in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv, max_memory):#ao(4,N,nao): AO values and x,y,z compoents in grids
                    rho0a = make_rho(0, ao, mask, xctype)#(4,N): density and "density derivatives" for x,y,z components in grids
                    rho0b = make_rho(1, ao, mask, xctype)
                    rho = (rho0a, rho0b)
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, omega=omega, xctype=xctype)[2]#second order derivatives about\rho
                    wfxc = fxc * weight
                    rho_o_a = lib.einsum('xrp,pi->xri', ao, orbo_a)# AO values to MO values in grids
                    rho_v_a = lib.einsum('xrp,pi->xri', ao, orbv_a)
                    rho_o_b = lib.einsum('xrp,pi->xri', ao, orbo_b)
                    rho_v_b = lib.einsum('xrp,pi->xri', ao, orbv_b)
                    rho_ov_a = np.einsum('xri,ra->xria', rho_o_a, rho_v_a[0])#\rho for \alpha 
                    rho_ov_b = np.einsum('xri,ra->xria', rho_o_b, rho_v_b[0])
                    rho_ov_a[1:4] += np.einsum('ri,xra->xria', rho_o_a[0], rho_v_a[1:4])#(4,N,i,a): \rho values for occupied and virtual orbitals
                    rho_ov_b[1:4] += np.einsum('ri,xra->xria', rho_o_b[0], rho_v_b[1:4])
                    w_ov_aa = np.einsum('xyr,xria->yria', wfxc[0,:,0], rho_ov_a)#(4,N,i,a) for \alpha\alpha
                    w_ov_ab = np.einsum('xyr,xria->yria', wfxc[0,:,1], rho_ov_a)
                    w_ov_bb = np.einsum('xyr,xria->yria', wfxc[1,:,1], rho_ov_b)

                    iajb = lib.einsum('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                    a_aa = iajb
                    #b_aa = iajb

                    iajb = lib.einsum('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                    a_bb = iajb
                    #b_bb = iajb

                    iajb = lib.einsum('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                    a_ab = iajb
                    #b_ab = iajb
            return a_aa,a_ab,a_bb,hyb
        
        try:
            xctype = mf.xc
        except:
            xctype = None
        
        if xctype is not None:
            a_aa,a_ab,a_bb,hyb = get_kxc()
        else:
            hyb = 1

        aa = np.zeros((nocc_a, nvir_a, nocc_a, nvir_a))
        ab = np.zeros((nocc_a, nvir_a, nocc_b, nvir_b))
        ba = np.zeros((nocc_b, nvir_b, nocc_a, nvir_a))
        bb = np.zeros((nocc_b, nvir_b, nocc_b, nvir_b))
        aa += np.einsum('iabj->iajb', self.eri_mo[:nocc_a,nocc_a:,nocc_a:,:nocc_a])
        aa -= hyb*np.einsum('ijba->iajb', self.eri_mo[:nocc_a,:nocc_a,nocc_a:,nocc_a:])
        bb += np.einsum('iabj->iajb', self.eri_mo[:nocc_b,nocc_b:,nocc_b:,:nocc_b])
        bb -= hyb*np.einsum('ijba->iajb', self.eri_mo[:nocc_b,:nocc_b,nocc_b:,nocc_b:])
        ab += np.einsum('iabj->iajb', self.eri_mo[:nocc_a,nocc_a:,nocc_b:,:nocc_b])
        ba += np.einsum('iabj->iajb', self.eri_mo[:nocc_b,nocc_b:,nocc_a:,:nocc_a])
        
        if xctype is not None:
            aa += a_aa
            ab += a_ab
            bb += a_bb
            ba += a_ab.transpose(2,3,0,1)
        if not s_tda:
            info_eris = (aa,ab,ba,bb,fij_a,fij_b,fab_a,fab_b,fij_a2,fij_b2,fab_a2,fab_b2)
        else:
            info_eris = (aa,ab,ba,bb,fij_a,fij_b,fab_a,fab_b)
            
        self.A = self.get_matA(info_eris)
        self.nocc_a = nocc_a
        self.nocc_b = nocc_b
        self.nvir_a = nvir_a
        self.nvir_b = nvir_b
        
    def get_matA(self,info_eris):
        dim = 2*self.nc*self.nv+self.nc*self.no+self.nv*self.no
        print('Dimension of A matrix ',dim)
        A = np.zeros((dim,dim))
        no = self.no
        nv = self.nv
        nc = self.nc
        dim1 = nc*nv
        dim2 = dim1 + nc*no
        dim3 = dim2 + no*nv
        iden_C = np.identity(nc)
        iden_O = np.identity(no)
        iden_V = np.identity(nv)
        
        if not self.s_tda:
            aa,ab,ba,bb,fij_a,fij_b,fab_a,fab_b,fij_a2,fij_b2,fab_a2,fab_b2 = info_eris
        else:
            aa,ab,ba,bb,fij_a,fij_b,fab_a,fab_b = info_eris

        # CV(0)-CV(0) iajb
        A[:dim1, :dim1] += 0.5*(aa[:-no,:,:-no,:] + bb[:,no:,:,no:] + ab[:-no,:,:,no:] + ba[:,no:,:-no,:] + \
                        np.einsum('ij,ab -> iajb',iden_C, fab_a) - np.einsum('ji,ab->iajb',fij_a[:-no,:-no],iden_V)+\
                        np.einsum('ij,ab-> iajb',iden_C,fab_b[no:,no:]) - np.einsum('ji,ab->iajb',fij_b,iden_V)).reshape(nc*nv, nc*nv)
        #CV(0)-CO(0) iajv
        A_cv0co0 = (1/np.sqrt(2))*(ab[:-no,:,:,:no] + bb[:,no:,:,:no] + np.einsum('ij,av->iajv',iden_C,fab_b[no:,:no])).reshape(nc*nv, nc*no)
        A[:dim1, dim1:dim2] += A_cv0co0
        A[dim1:dim2, :dim1] += A_cv0co0.T

        # CV(0)-OV(0) iavb
        A_cv0ov0 = (1/np.sqrt(2))*(aa[:-no,:,nc:,:]+ba[:,no:,nc:,:]-np.einsum('vi,ab->iavb',fij_a[nc:, :nc],iden_V)).reshape(nc*nv, nv*no)
        A[:dim1, dim2:dim3] += A_cv0ov0
        A[dim2:dim3, :dim1] += A_cv0ov0.T
        
        # CV(0)-CV(1)
        si = -np.sqrt((0.5*self.mol.spin+1)/(0.5*self.mol.spin))
        if self.s_tda:
            A_cv0cv1 = si*0.5*(aa[:-no,:,:-no,:]-bb[:,no:,:,no:] -ab[:-no,:,:,no:] + ba[:,no:,:-no,:]+\
                 (np.einsum('ij,ab -> iajb',iden_C, fab_a) - np.einsum('ab,ij->iajb', iden_V, fij_a[:-no,:-no]))+\
                 (-np.einsum('ij,ab-> iajb',iden_C,fab_b[no:,no:]) + np.einsum('ab,ij->iajb',iden_V, fij_b))).reshape((nc*nv, nc*nv))
        else:
            A_cv0cv1 = 0.5*(aa[:nc,:,:nc,:]-bb[:,no:,:,no:] -ab[:nc,:,:,no:] + ba[:,no:,:nc,:]+\
                +np.einsum('ij,ab -> iajb',iden_C, fab_a) - np.einsum('ab,ij->iajb', iden_V, fij_a[:nc,:nc])+\
                -np.einsum('ij,ab-> iajb',iden_C,fab_b[no:,no:]) + np.einsum('ab,ij->iajb',iden_V, fij_b)).reshape((nc*nv, nc*nv))+\
                0.5*(1+si)*(np.einsum('ij,ab->iajb',iden_C, fab_b2[no:,no:])-np.einsum('ij,ab->iajb',iden_C,fab_a2)-\
                np.einsum('ab,ij->iajb',iden_V,fij_b2)+np.einsum('ab,ij->iajb',iden_V,fij_a2[:-no,:-no])).reshape(nc*nv,nc*nv)
        A[:dim1, dim3:] += A_cv0cv1
        A[dim3:, :dim1] += A_cv0cv1.T
        
        # CO(0)-CO(0)
        A[dim1:dim2, dim1:dim2] += bb[:,:no,:,:no].reshape(nc*no,nc*no)+(np.einsum('ij, ab->iajb',iden_C,                     fab_b[:no,:no].reshape(no,no))-np.einsum('ab,ji->iajb',iden_O.reshape(no,no), fij_b)).reshape(nc*no, nc*no)
        
        # CO(0)-OV(0)
        A_co0ov0 = ba[:,:no,nc:,:].reshape(nc*no,nv*no)
        A[dim1:dim2,dim2:dim3] += A_co0ov0
        A[dim2:dim3,dim1:dim2] += A_co0ov0.T
        
        #CO(0)-CV(1) 
        if self.s_tda:
            si = -np.sqrt((0.5*self.mol.spin+1)/(0.5*self.mol.spin))
        else:
            si = 1
        A_co0cv1 = (1/np.sqrt(2))*si*(ba[:,:no,:-no,:]-bb[:,:no,:,no:]-np.einsum('ij,ub->iujb',iden_C, fab_b[:no,no:])).reshape(nc*no,nc*nv)
        A[dim1:dim2,dim3:] += A_co0cv1
        A[dim3:,dim1:dim2] += A_co0cv1.T
        
        # OV(0)-OV(0)
        A[dim2:dim3,dim2:dim3] += aa[nc:,:,nc:,:].reshape(nv*no,nv*no) + \
                    (np.einsum('ji,ab->iajb',iden_O.reshape(no,no), fab_a) - \
                    np.einsum('ab,ji->iajb',iden_V, fij_a[nc:,nc:].reshape(no,no))).reshape(nv*no,nv*no)
        
        # OV(0)-CV(1)
        A_ov0cv1 = (1/np.sqrt(2))*si*(aa[nc:,:,:-no,:]-ab[nc:,:,:,no:]-np.einsum('ab,ju->uajb',iden_V, fij_a[:-no,nc:])).reshape(nv*no,nc*nv)
        A[dim2:dim3, dim3:] += A_ov0cv1
        A[dim3:, dim2:dim3] += A_ov0cv1.T
        
        #CV(1)-CV(1) iajb
        if self.s_tda:
            A[dim3:, dim3:] += 0.5*(aa[:-no,:,:-no,:]+bb[:,no:,:,no:]-ab[:-no,:,:,no:]-ba[:,no:,:-no,:]+\
               (1-1/(0.5*mpl.spin))*np.einsum('ij,ab->iajb',iden_C, fab_a) - \
               (1+1/(0.5*mpl.spin))*np.einsum('ab,ij->iajb',iden_V,fij_a[:-no,:-no])+\
               (1+1/(0.5*mpl.spin))*np.einsum('ij,ab->iajb',iden_C,fab_b[no:,no:]) - \
               (1-1/(0.5*mpl.spin))*np.einsum('ab,ij->iajb',iden_V,fij_b)).reshape(nc*nv,nc*nv)
        else:
            A[dim3:, dim3:] += 0.5*(aa[:-no,:,:-no,:]+bb[:,no:,:,no:]-ab[:-no,:,:,no:]-ba[:,no:,:-no,:]+\
            np.einsum('ij,ab->iajb',iden_C, fab_a) - np.einsum('ab,ij->iajb',iden_V,fij_a[:-no,:-no])+\
            np.einsum('ij,ab->iajb',iden_C,fab_b[no:,no:]) - np.einsum('ab,ij->iajb',iden_V,fij_b)).reshape(nc*nv,nc*nv)+\
            0.5*(1/(0.5*self.mol.spin))*(np.einsum('ij,ab->iajb',iden_C, fab_b2[no:,no:])-np.einsum('ij,ab->iajb',iden_C,fab_a2)+\
             np.einsum('ab,ij->iajb',iden_V,fij_b2)-np.einsum('ab,ij->iajb',iden_V,fij_a2[:-no,:-no])).reshape(nc*nv,nc*nv)
        return A
    
    def analyze_TDM(self):
        with mol.with_common_orig(_charge_center(mol)):
            ints = mol.intor_symmetric('int1e_r', comp=3) # (3,nao,nao)
        ints_mo = np.einsum('xpq,pi,qj->xij', ints, mf.mo_coeff, mf.mo_coeff)
        print("Ground state to Excited state transition dipole moments(Au)")
        print('X    Y    Z    OSC.')
        for i in range(len(self.e)):
            cv0 = v[:,i][:dim1].reshape(nc,nv)
            co0 = v[:,i][dim1:dim2].reshape(nc,no)
            ov0 = v[:,i][dim2:dim3].reshape(no,nv)
            cv1 = v[:,i][dim3:].reshape(nc,nv)
            hcv0 = np.einsum('...ia,ia->...',ints_mo[:,:nc,nc+no:],cv0)
            hco0 = np.einsum('...iv,iv->...',ints_mo[:,:nc,nc:nc+no],co0)
            hov0 = np.einsum('...va,va->...',ints_mo[:,nc:nc+no,nc+no:],ov0)
            tdm = np.sqrt(2)*hcv0+hco0+hov0
            osc = (2/3)*e[i]*(tdm[0]**2+tdm[1]**2+tdm[2]**2)
            print(i+1,tdm,osc)
            
        dip_elec = -self.mf.dip_moment(unit='au')
        nuc_charges = mol.atom_charges()
        nuc_coords = mol.atom_coords()
        dip_nuc = np.einsum('i,ix->x', nuc_charges, nuc_coords)
        gs = dip_elec + dip_nuc
        print('Ground state dipole moment (in Debye) ', gs)
        print(' ')
        print("Excited state to Excited state transition dipole moments(Au)")
        print('State   State   X    Y    Z    OSC.')
        si = self.mol.spin/2
        iden_C = np.identity(self.nc)
        iden_O = np.identity(self.no)
        iden_V = np.identity(self.nv)
        dim1 = self.nc*self.nv
        dim2 = dim1+self.nc*self.no
        dim3 = dim2+self.no*self.nv
        for i in range(len(self.e)):
            s0_cv0 = v[:,i][:dim1].reshape(self.nc,self.nv)
            s0_co0 = v[:,i][dim1:dim2].reshape(self.nc,self.no)
            s0_ov0 = v[:,i][dim2:dim3].reshape(self.no,self.nv)
            s0_cv1 = v[:,i][dim3:].reshape(self.nc,self.nv)
            for j in range(len(self.e)):
                s1_cv0 = v[:,j][:dim1].reshape(self.nc,self.nv)
                s1_co0 = v[:,j][dim1:dim2].reshape(self.nc,self.no)
                s1_ov0 = v[:,j][dim2:dim3].reshape(self.no,self.nv)
                s1_cv1 = v[:,j][dim3:].reshape(self.nc,self.nv)

                # diagonal
                # cv0-cv0 delta_ij*r_ab - delta_ab*r_ij
                h_cv0_cv0 = np.einsum('ia,xba,jb,ij->x',s0_cv0,ints_mo[:,self.nc+self.no:,self.nc+self.no:],s1_cv0,iden_C)-\
                            np.einsum('ia,xji,jb,ab->x',s0_cv0,ints_mo[:,:self.nc,:self.nc],s1_cv0,iden_V)
                #print(h_cv0_cv0)
                # co0-co0 delta_ij*r_uv - delta_uv*r_ij
                h_co0_co0 = np.einsum('iu,xvu,jv,ij->x',s0_co0,ints_mo[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_co0,iden_C)-\
                            np.einsum('iu,xji,jv,uv->x',s0_co0,ints_mo[:,:self.nc,:self.nc],s1_co0,iden_O)
                #print(h_co0_co0)
                # ov0-ov0 delta_uv*r_ab - delta_ab*r_uv
                h_ov0_ov0 = np.einsum('ua,xba,vb,uv->x',s0_ov0,ints_mo[:,self.nc+self.no:,self.nc+self.no:],s1_ov0,iden_O)-\
                            np.einsum('ua,xuv,vb,ab->x',s0_ov0,ints_mo[:,self.nc:self.nc+self.no,self.nc:self.nc+self.no],s1_ov0,iden_V)
                #print(h_ov0_ov0)
                # cv1-cv1 delta_ij*r_ab - delta_ab*r_ij
                h_cv1_cv1 = np.einsum('ia,xba,jb,ij->x',s0_cv1,ints_mo[:,self.nc+self.no:,self.nc+self.no:],s1_cv1,iden_C)-\
                            np.einsum('ia,xji,jb,ab->x',s0_cv1,ints_mo[:,:self.nc,:self.nc],s1_cv1,iden_V)
                #print(h_cv1_cv1)

                # off-diagonal
                factor = np.sqrt((si+1)/(2*si))
                #factor = 0.5
                # cv0-co0 1/sqrt{2} * delta_ij*r_av
                h_cv0_co0 = np.einsum('ia,xva,jv,ij->x',s0_cv0,ints_mo[:,self.nc:self.nc+self.no,self.nc+self.no:],s1_co0,iden_C)/np.sqrt(2)
                #print(h_cv0_co0)
                # cv0-ov0 -1/sqrt{2} * delta_ab*r_iv
                h_cv0_ov0 = -np.einsum('ia,xvi,vb,ab->x',s0_cv0,ints_mo[:,self.nc:self.nc+self.no,:self.nc],s1_ov0,iden_V)/np.sqrt(2)
                #print(h_cv0_ov0)

                # co0 -cv0
                h_co0_cv0 = np.einsum('iu,xbu,jb,ij->x',s0_co0,ints_mo[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_cv0,iden_C)/np.sqrt(2)
                # co0-cv1 -1/sqrt{2} * delta_ij*r_ub
                h_co0_cv1 = -factor*np.einsum('iu,xbu,jb,ij->x',s0_co0,ints_mo[:,self.nc+self.no:,self.nc:self.nc+self.no],s1_cv1,iden_C)
                #print(h_co0_cv1)

                #ov0 - cv0
                h_ov0_cv0 = -np.einsum('ua,xju,jb,ab->x',s0_ov0,ints_mo[:,:self.nc,self.nc:self.nc+self.no],s1_cv0,iden_V)/np.sqrt(2)
                # ov0-cv1 -1/sqrt{2} * delta_ab*r_ju
                h_ov0_cv1 = -factor*np.einsum('ua,xju,jb,ab->x',s0_ov0,ints_mo[:,:self.nc,self.nc:self.nc+self.no],s1_cv1,iden_V)
                #print(h_ov0_cv1)

                #cv1-co0
                h_cv1_co0 = -factor*np.einsum('ia,xva,jv,ij->x',s0_cv1,ints_mo[:,self.nc:self.nc+self.no,self.nc+self.no:],s1_co0,iden_C)
                #cv1-ov1
                h_cv1_ov0 = -factor*np.einsum('ia,xvi,vb,ab->x',s0_cv1,ints_mo[:,self.nc:self.nc+self.no,:self.nc],s1_ov0,iden_V)

                tdm=h_cv0_cv0+h_co0_co0+h_ov0_ov0+h_cv1_cv1+h_cv0_co0+h_cv0_ov0+h_co0_cv0+h_co0_cv1+h_ov0_cv0+h_ov0_cv1+h_cv1_co0+h_cv1_ov0
                #tdm=h_cv0_cv0+h_co0_co0+h_ov0_ov0+h_cv1_cv1+h_ov0_cv1
                if i==j:
                    tdm += gs
                osc = (2/3)*(e[i]-e[j])*(tdm[0]**2 + tdm[1]**2 + tdm[2]**2)
                print(f'{i+1:2d} {j+1:2d} {tdm[0]:>8.4f} {tdm[1]:>8.4f} {tdm[2]:>8.4f} {osc:>8.4f}')
        
    
    def analyze(self):
        nc = self.nc
        nv = self.nv
        no = self.no
        dim1 = nc*nv
        dim2 = dim1+nc*no
        dim3 = dim2+no*nv
        for i in range(len(self.e)):
            value = self.values[:, i]
            #print(lib.norm(value)) # 1
            x_cv0 = value[:dim1].reshape(nc,nv)
            x_co0 = value[dim1:dim2].reshape(nc,no)
            x_ov0 = value[dim2:dim3].reshape(no,nv)
            x_cv1 = value[dim3:].reshape(nc,nv)
            Dp_ab = 0.
            print(f'Excited state {i+1} {self.e[i]*27.21138505:12.5f} eV')
            
            for o,v in zip(* np.where(abs(x_cv0)>0.1)):
                #print(f'CV(0) {o+1}b -> {v+1+self.nc+self.no}b {x_cv_bb[o,v]:10.5f} {100*x_cv_bb[o,v]**2:2.2f}%')
                print(f'CV(0) {o+1}b -> {v+1+self.nc+self.no}b {x_cv0[o,v]:10.5f} {100*x_cv0[o,v]**2:2.2f}%')
            for o,v in zip(* np.where(abs(x_co0)>0.1)):
                print(f'CO(0) {o+1}b -> {v+1+self.nc}b {x_co0[o,v]:10.5f} {100*x_co0[o,v]**2:5.2f}%')
            for o,v in zip(* np.where(abs(x_ov0)>0.1)):
                print(f'OV(0) {o+self.nc+1}a -> {v+1+self.nc+self.no}a {x_ov0[o,v]:10.5f} {100*x_ov0[o,v]**2:5.2f}%')
            for o,v in zip(* np.where(abs(x_cv1)>0.1)):
                #print(f'CV(1) {o+1}a -> {v+1+self.nc+self.no}a {x_cv_aa[o,v]:10.5f} {100*x_cv_aa[o,v]**2:5.2f}%')
                print(f'CV(1) {o+1}a -> {v+1+self.nc+self.no}a {x_cv1[o,v]:10.5f} {100*x_cv1[o,v]**2:5.2f}%')
            print(' ')
                
    def kernel(self,nstates=1):
        e,v = scipy.linalg.eigh(self.A)
        self.e = e[:nstates]
        self.values = v[:,:nstates]
        self.nstates = nstates
        return self.e*27.21138505, self.values