# A general SOC-SI code
# |GS> |So> |S+> |S-> interaction
# Implemented by Bohan Zhang@BNU2025
import numpy, sys, pyscf, pickle, logging, subprocess, os, time
from math import sqrt as sqrt
sqrt2 = sqrt(2)
sys.path.append('./')
sys.path.append('./../')
sys.path.append('./../../')

from typing import Optional, Union
from pyscf import scf, dft
from opt_einsum import contract as einsum

from sympy.physics.wigner import wigner_3j
from utils import unit

def progress_bar(current, total, start_time, bar_length=30):
    fraction = current / total
    arrow = int(fraction * bar_length) * '█'
    padding = (bar_length - len(arrow)) * '-'
    percent = int(fraction * 100)
    
    elapsed = time.time() - start_time
    eta = (elapsed / fraction - elapsed) if fraction > 0 else 0

    sys.stdout.write(
        f'\rProgress: |{arrow}{padding}| {percent}% = {current}/{total} '
        f'Elapsed: {elapsed:.1f}s ETA: {eta:.1f}s'
    )
    sys.stdout.flush()

def w(S,M,Sprime,Mprime):
    '''
    Calculate w(S,M,S',M') geo-factor by using Sympy
    '''
    if abs(wigner_3j(S,1,Sprime,-S,S-Sprime,Sprime).doit().evalf())<1e-3:
        return 0.
    else:
        return (-1)**(S-M) * float((wigner_3j(S,1,Sprime,-M,M-Mprime,Mprime) \
                                    / wigner_3j(S,1,Sprime,-S,S-Sprime,Sprime)).evalf())

def symm_matrix(h):
    '''
    Fill the lower triangle of the matrix using the upper triangle
    '''
    return numpy.triu(h) + numpy.triu(h, 1).T.conjugate()

States_list = ['|GS>', '|S->', '|So>', '|S+>']

class SI_driver():
    def __init__(
            self,
            mf: Optional[Union[
                scf.hf.RHF,
                scf.rohf.ROHF,
                dft.rks.RKS,
                dft.roks.ROKS
            ]] = None,
            S: float = None,
            Vso: numpy.ndarray = None,
            ngs: Optional[Union[int, bool]] = 1,
            states:dict = {},
    ) -> None:
        super().__init__()
        self.mf = mf
        self.mol = mf.mol
        # Spin and corresponding relationship
        self.S = S
        assert self.S == self.mol.spin/2.
        self.str2S = {'|GS>': self.S, '|So>': self.S, '|S+>': self.S+1, '|S->': self.S-1}
        self.S2str = {(self.S, 1):'|GS>', (self.S, 0):'|So>', (self.S+1, 0):'|S+>', (self.S-1, 0):'|S->'}
        # Vso to hm
        self.hm = self.cal_hm(Vso)
        # check state label
        self.ngs = ngs
        self.states = states
        if int(ngs) == 0:
            assert len(self.states['|GS>']) == 0 or '|GS>' not in self.states.keys()
        elif int(ngs) == 1:
            self.states['|GS>'] = [(0.0, numpy.array([1.0,]))]
        if '|So>' not in self.states.keys(): self.states['|So>'] = []
        if '|S+>' not in self.states.keys(): self.states['|S+>'] = []
        if '|S->' not in self.states.keys(): self.states['|S->'] = []
        # calculate nc, no, nv and corresponding slice
        self.cal_dims()

    def cal_hm(self, Vso):
        '''
        Transfer Vso to hm
        '''
        hm = numpy.zeros((self.mol.nao,self.mol.nao,3,), dtype=numpy.complex128)
        hm[...,0]  =  1j*Vso[0,:,:] - Vso[1,:,:] # h^1
        hm[...,1]  =  1j*Vso[2,:,:]*sqrt2 # h^0
        hm[...,2]  = -1j*Vso[0,:,:] - Vso[1,:,:] # h^-1
        return hm[...,::-1]
    
    def cal_dims(self) -> None:
        '''
        Calculate some dimesions
        '''
        self.norb = self.mol.nao
        self.nelec = self.mol.nelectron
        Smax = int(self.S*2)
        self.nc,self.no,self.nv = (self.nelec-Smax)//2, Smax, self.norb-(self.nelec-Smax)//2-Smax
        # delta
        self.delta_c = numpy.eye(self.nc)
        self.delta_o = numpy.eye(self.no)
        self.delta_v = numpy.eye(self.nv)
        self.slc = slice(0,self.nc,1)
        self.slo = slice(self.nc,self.nc+self.no,1)
        self.slv = slice(self.nc+self.no,self.norb,1)
        self.sl = (self.slc, self.slo, self.slv)
        logging.info(f"Slice of orbitals:")
        logging.info(f"slc {self.slc} length: {self.nc}")
        logging.info(f"slo {self.slo} length: {self.no}")
        logging.info(f"slv {self.slv} length: {self.nv}")
        assert self.nc+self.no+self.nv == self.norb
        self.cv = self.nc * self.nv
        self.co = self.nc * self.no
        self.ov = self.no * self.nv
        self.oo = self.no * self.no
        self.o1o2 = self.no * (self.no-1)
        self.o1o1 = self.no-1
        # make slice
        ## |S->
        self.sl_S_1_dim0 = slice(0, self.cv, 1)
        self.sl_S_1_dim1 = slice(self.cv, self.cv+self.co, 1)
        self.sl_S_1_dim2 = slice(self.cv+self.co, self.cv+self.co+self.ov, 1)
        self.sl_S_1_dim3 = slice(self.cv+self.co+self.ov, self.cv+self.co+self.ov+self.oo, 1)
        # self.sl_S_1_dim4 = slice(self.cv+self.co+self.ov+self.oo, self.cv+self.co+self.ov+self.o1o2+self.o1o1, 1)
        logging.debug(f"Slice of |S-> is")
        logging.debug(f"|S-> CV(1):   Slice {self.sl_S_1_dim0}, length: {self.cv}")
        logging.debug(f"|S-> CO(1):   Slice {self.sl_S_1_dim1}, length: {self.co}")
        logging.debug(f"|S-> OV(1):   Slice {self.sl_S_1_dim2}, length: {self.ov}")
        logging.debug(f"|S-> OO(1): Slice {self.sl_S_1_dim3}, length: {self.oo}")
        # logging.debug(f"|S-> O1O1(1): Slice {self.sl_S_1_dim4}, length: {self.o1o1}")
        ## |S>
        self.sl_S_dim0 = slice(0, self.cv, 1)
        self.sl_S_dim1 = slice(self.cv, self.cv+self.co, 1)
        self.sl_S_dim2 = slice(self.cv+self.co, self.cv+self.co+self.ov, 1)
        self.sl_S_dim3 = slice(self.cv+self.co+self.ov, self.cv+self.co+self.ov+self.cv, 1)
        logging.debug(f"Slice of |So> is")
        logging.debug(f"|So> CV(0):   Slice {self.sl_S_dim0}, length: {self.cv}")
        logging.debug(f"|So> CO(0):   Slice {self.sl_S_dim1}, length: {self.co}")
        logging.debug(f"|So> OV(0):   Slice {self.sl_S_dim2}, length: {self.ov}")
        logging.debug(f"|So> CV(1):   Slice {self.sl_S_dim3}, length: {self.cv}")
        ## |S+>
        logging.debug(f"Slice of |S+> is")
        logging.debug(f"|S+> CV(1):                           length: {self.cv}")
        # hso pos
        self.nGS = self.ngs
        self.nSo = len(self.states['|So>'])
        self.nSp = len(self.states['|S+>'])
        self.nSm = len(self.states['|S->'])
        self.dim0 = int((2*self.S-1)*self.nSm)
        self.dim1 = int((2*self.S-1)*self.nSm + (2*self.S+1)*self.nGS)
        self.dim2 = int((2*self.S-1)*self.nSm + (2*self.S+1)*(self.nGS+self.nSo))
        self.dim_hso = int((2*self.S-1)*self.nSm + (2*self.S+1)*(self.nGS+self.nSo) + (2*self.S+3)*self.nSp)
        self.str2dim = {'|S->': 0,'|GS>': self.dim0,'|So>': self.dim1,'|S+>': self.dim2}
        self.str2nS = {'|S->': self.nSm,'|GS>': self.nGS,'|So>': self.nSo,'|S+>': self.nSp}

    def kernel(self,):
        logging.info(f"Perform SI calculation with Si={self.S}")
        time0 = time.time()
        self.heff = self.make_heff()
        logging.info(f"Begin to diag Omega+hso --{self.heff.shape}--")
        import scipy
        logging.info(f"{'='*20} Heff diagonalization {'='*20}")
        time1 = time.time()
        self.eso, self.vso = scipy.linalg.eigh(self.heff, driver='evd')
        self.esf = numpy.diag(self.vso.T.conjugate()@self.Omega@self.vso).real
        time2 = time.time()
        logging.info(f"End diagonalization, cost time {time2-time1:.2f}s")
        logging.info(f"Ref. E (in Hatree) = {self.mf.e_tot}")
        logging.info(f"The  E (in eV), shape={self.eso.shape} --")
        logging.info(f"{numpy.diag(self.Omega)[:20] * unit.ha2eV}")
        # logging.info(f"{self.esf[:20] * unit.ha2eV}")
        logging.info(f"The  E_soc (in eV) --")
        logging.info(f"{self.eso[:20] * unit.ha2eV}")
        logging.info(f"End SI calculation, cost time {time2-time0:.2f}s")
        self.summary()
        return self.eso, self.vso
    
    def summary(self):
        logging.info(f"{'='*10} Summary of SI calculation {'='*10}")
        for i, e in enumerate(self.eso[:20]):
            logging.info(f"Excited state [{i:4d}]: Eso {self.eso[i]*unit.ha2eV:>12.6f} eV  --  Esf {self.esf[i]*unit.ha2eV:>12.6f} eV")

    def cal_hso_pos(self,SL,ML,Li,igsL,SR,MR,Ri,igsR,debug=0):
        L_index = self.S2str[(SL,igsL)] # e.g. |So>
        R_index = self.S2str[(SR,igsR)]
        L_dim = self.str2dim[L_index] # begin of this kind of state
        R_dim = self.str2dim[R_index]
        L_Mnum = ML+SL # the (L_Mnum)-th M
        R_Mnum = MR+SR
        L_length = self.str2nS[L_index]
        R_length = self.str2nS[R_index]
        L_pos = int(L_dim + L_Mnum*L_length + Li)
        R_pos = int(R_dim + R_Mnum*R_length + Ri)
        if debug:
            logging.debug(f"(SL,ML,Li,igsL,SR,MR,Ri,igsR) = {(SL,ML,Li,igsL,SR,MR,Ri,igsR)}")
            logging.debug(f"state={L_index}, begin={L_dim}, {int(L_Mnum)}-M, foot-length={L_length}")
        return (L_pos, R_pos)

    def make_heff(self):
        logging.info(f"{'='*20} Begin to make heff {'='*20}")
        time0 = time.time()
        hso = numpy.zeros((self.dim_hso, self.dim_hso,),dtype=numpy.complex128)
        Omega = numpy.zeros((self.dim_hso, self.dim_hso,))
        self.hso_pos = {}
        count = 0
        n_sum = int((self.dim_hso**2+self.dim_hso)//2)
        for L_index in States_list: # [L_index]: |S->, |GS>, |So>, |S+> Different [SL]
            SL = self.str2S[L_index]
            for Li, L_value in enumerate(self.states[L_index]): # [L_value]: Different eigenvecter for fixed SL
                L = (L_index, L_value)
                for R_index in States_list: # [R_index]: |S->, |GS>, |So>, |S+> Different [SR]
                    SR = self.str2S[R_index]
                    for Ri, R_value in enumerate(self.states[R_index]): # [R_value]: Different eigenvecter for fixed SR
                        R = (R_index, R_value)
                        # ============ construct matrix element ============
                        h, exitcode = self.make_hso_local(SL,SR,L,R) # <Phi|hm|Phi> without geo-factor w(SL,ML,SR,MR)
                        for ML in numpy.arange(-SL,SL+1,1): # Different [ML] for fixed SL
                            for MR in numpy.arange(-SR,SR+1,1): # Different [MR] for fixed SR
                                igsL, igsR = 0, 0
                                if L[0] == '|GS>': igsL = 1
                                if R[0] == '|GS>': igsR = 1
                                # Obtain posision of hm (L,R)
                                Lpos, Rpos = self.cal_hso_pos(SL,ML,Li,igsL,SR,MR,Ri,igsR)
                                if Lpos <= Rpos: # For only upper triangle
                                    # Debug
                                    # logging.debug(f"(SL,ML,Li,igsL,SR,MR,Ri,igsR) = {(SL,ML,Li,igsL,SR,MR,Ri,igsR)}")
                                    # logging.debug(f"Lpos={Lpos}({L_index}, {float(ML)}, {Li}-th),\
                                    #                 Rpos={Rpos}({R_index}, {float(MR)}, {Ri}-th)")
                                    # -----------------------------------------------------
                                    # make sure that hso_so is calculated only once.
                                    if not exitcode == 1:
                                        logging.warning(f"exitcode={exitcode}, check hso!")
                                    # check out |M'-M|<=1 condition
                                    if abs(MR-ML)<=1:
                                        # Calculate h*w
                                        hso[Lpos, Rpos] = h[int(MR-ML)+1]*w(SL,ML,SR,MR)
                                    else:
                                        hso[Lpos, Rpos] = 0.0
                                    count += 1
                                    if count%10000 == 0:
                                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                                            progress_bar(count, n_sum, time0)
                                        else:
                                            logging.info(f"finish {count}/{n_sum}")
                                if Lpos == Rpos: # E(0)
                                    assert abs(L_value[0]-R_value[0])<1e-6 # share the same energy
                                    assert abs(hso[Lpos, Rpos].imag) < 1e-6 # real if diagonal
                                    Omega[Lpos, Rpos] = L[1][0]
        self.hso = symm_matrix(hso)
        self.Omega = Omega / unit.ha2eV # XTDA, XSF-TDA mathod give energy in eV
        self.heff = self.hso + self.Omega
        time1 = time.time()
        logging.info(f"End of contructing heff, cost time {time1-time0:.2f}s")
        return self.heff
    
    def make_hso_local(self, SL, SR, L, R):
        '''
        SL, XL, SR, XR given
        calculate matirx element XL<Phi(SL)|hm|Phi(SR)>XR => hm, m=0,1,2 of those four quantum numbers.
        '''
        igsL, igsR = 0, 0
        if L[0] == '|GS>': igsL = 1
        if R[0] == '|GS>': igsR = 1
        exitcode = 0
        hso_local = 0.0

        # S-1 S-1
        if abs(SL-(self.S-1))<1e-6 and abs(SR-(self.S-1))<1e-6 and int(igsL+igsR) == 0:
            hso_local = self.interact_S_1S_1(L,R); exitcode += 1
        # S-1 GS
        if abs(SL-(self.S-1))<1e-6 and abs(SR-(self.S))<1e-6 and int(igsR-igsL) == 1:
            hso_local = self.interact_S_1GS(L,R); exitcode += 1
        # S-1 S
        if abs(SL-(self.S-1))<1e-6 and abs(SR-(self.S))<1e-6 and int(igsL+igsR) == 0:
            hso_local = self.interact_S_1S(L,R); exitcode += 1

        # GS GS
        if abs(SL-(self.S))<1e-6 and abs(SR-(self.S))<1e-6 and int(igsL+igsR) == 2:
            hso_local = self.interact_GSGS(L,R); exitcode += 1
        # GS S
        if abs(SL-(self.S))<1e-6 and abs(SR-(self.S))<1e-6 and int(igsL-igsR) == 1:
            hso_local = self.interact_GSS(L,R); exitcode += 1
        # GS S+1
        if abs(SL-(self.S))<1e-6 and abs(SR-(self.S+1))<1e-6 and int(igsL-igsR) == 1:
            hso_local = self.interact_GSS1(L,R); exitcode += 1

        # S S
        if abs(SL-(self.S))<1e-6 and abs(SR-(self.S))<1e-6 and int(igsL+igsR) == 0:
            hso_local = self.interact_SS(L,R); exitcode += 1
        # S S+1
        if abs(SL-(self.S))<1e-6 and abs(SR-(self.S+1))<1e-6 and int(igsL+igsR) == 0:
            hso_local = self.interact_SS1(L,R); exitcode += 1

        # S+1 S+1
        if abs(SL-(self.S+1))<1e-6 and abs(SR-(self.S+1))<1e-6 and int(igsL+igsR) == 0:
            hso_local = self.interact_S1S1(L,R); exitcode += 1

        return hso_local, exitcode
    
    def interact_S_1S_1(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # =============== line0 ===============
        XL = L[1][1][self.sl_S_1_dim0].reshape(self.nc,self.nv)
        # Case (1) (S-1)(CV1) (S-1)(CV1)
        XR = R[1][1][self.sl_S_1_dim0].reshape(self.nc,self.nv)
        factor = (1-self.S)/(self.S*sqrt2)
        XhX += einsum('ia,abm,ij,jb->m',XL, self.hm[self.slv,self.slv,:], self.delta_c, XR)*factor
        XhX += einsum('ia,jim,ab,jb->m',XL, self.hm[self.slc,self.slc,:], self.delta_v, XR)*factor
        # Case (2) (S-1)(CV1) (S-1)(CO1)
        XR = R[1][1][self.sl_S_1_dim1].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S+1)/self.S)*(1-self.S)/(self.S*2)
        XhX += einsum('ia,atm,ij,jt->m',XL, self.hm[self.slv,self.slo,:], self.delta_c, XR)*factor
        # Case (3) (S-1)(CV1) (S-1)(OV1)
        XR = R[1][1][self.sl_S_1_dim2].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S+1)/self.S)*(1-self.S)/(self.S*2)
        XhX += einsum('ia,tim,ab,tb->m',XL, self.hm[self.slo,self.slc,:], self.delta_v, XR)*factor
        # Case (4) (S-1)(CV1) (S-1)(O1O2) = 0
        # Case (5) (S-1)(CV1) (S-1)(O1O1) = 0 #
        # =============== line1 ===============
        XL = L[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        # Case (2) (S-1)(CO1) (S-1)(CV1)
        XR = R[1][1][self.sl_S_1_dim0].reshape(self.nc,self.nv)
        # Case (11) (S-1)(CO1) (S-1)(CO1)
        XR = R[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        factor = -(self.S-1)/(self.S*sqrt2)
        XhX += einsum('iu,jim,ut,jt->m',XL, self.hm[self.slc,self.slc,:], self.delta_o, XR)*factor
        XhX += einsum('iu,utm,ij,jt->m',XL, ((2*self.S+1)/(2*self.S-1))*self.hm[self.slo,self.slo,:], self.delta_c, XR)*factor
        # Case (12) (S-1)(CO1) (S-1)(OV1) = 0
        # Case (13) (S-1)(CO1) (S-1)(O1O2)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = XR - numpy.diag(numpy.diag(XR))
        factor = -(self.S-1)/(self.S*(2*self.S-1))
        XhX += einsum('iu,wim,ut,wt->m',XL, self.hm[self.slo,self.slc,:], self.delta_o, XR)*factor
        # Case (14) (S-1)(CO1) (S-1)(O1O1)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = numpy.diag(XR)
        factor = -1.0/(2*sqrt(self.S*(2*self.S-1)))
        XhX += einsum('iu,uim,ut,t->m',XL, self.hm[self.slo,self.slc,:],\
                       ((1-self.S)/self.S) + 2*(self.S-1)*self.delta_o, XR)*factor
        # =============== line2 ===============
        XL = L[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        # Case (3) (S-1)(OV1) (S-1)(CV1)
        XR = R[1][1][self.sl_S_1_dim0].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S+1)/self.S)*(1-self.S)/(self.S*2)
        XhX += einsum('ia,tim,ab,tb->m',XR, -self.hm[self.slo,self.slc,:], self.delta_v, XL)*factor
        # Case (12) (S-1)(OV1) (S-1)(CO1) = 0
        # Case (20) (S-1)(OV1) (S-1)(OV1)
        XR = R[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        factor = (self.S-1)/(self.S*sqrt2)
        XhX += einsum('ua,abm,ut,tb->m',XL, self.hm[self.slv,self.slv,:], self.delta_o, XR)*factor
        XhX += einsum('ua,tum,ab,tb->m',XL, self.hm[self.slo,self.slo,:], self.delta_v, XR)*factor
        # Case (21) (S-1)(OV1) (S-1)(O1O2)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = XR - numpy.diag(numpy.diag(XR))
        factor = (1-self.S)/sqrt(self.S*(2*self.S-1))
        XhX += einsum('ua,atm,uw,wt->m',XL, self.hm[self.slv,self.slo,:], self.delta_o, XR)*factor
        # Case (22) (S-1)(OV1) (S-1)(O1O1)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = numpy.diag(XR)
        factor = -1.0/(2*sqrt(self.S*(2*self.S-1)))
        XhX += einsum('ua,aum,ut,t->m',XL, self.hm[self.slv,self.slo,:],\
                      ((1-self.S)/self.S + 2*(self.S-1)*self.delta_o), XR)*factor
        # =============== line3 ===============
        XL = L[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XL = XL - numpy.diag(numpy.diag(XL))
        # Case (4) (S-1)(O1O2) (S-1)(CV1) = 0
        # Case (13) (S-1)(O1O2) (S-1)(CO1)
        XR = R[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        factor = -(self.S-1)/(self.S*(2*self.S-1))
        XhX += einsum('iu,wim,ut,wt->m',XR, -self.hm[self.slo,self.slc,:], self.delta_o, XL)*factor
        # Case (21) (S-1)(O1O2) (S-1)(OV1)
        XR = R[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        factor = (1-self.S)/sqrt(self.S*(2*self.S-1))
        XhX += einsum('ua,atm,uw,wt->m',XR, -self.hm[self.slv,self.slo,:], self.delta_o, XL)*factor
        # Case (28) (S-1)(O1O2) (S-1)(O1O2)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = XR - numpy.diag(numpy.diag(XR))
        factor = -1/sqrt2
        XhX += einsum('vu,wvm,ut,wt->m',XL, self.hm[self.slo,self.slo,:], self.delta_o, XR)*factor
        XhX += einsum('vu,utm,vw,wt->m',XL, self.hm[self.slo,self.slo,:], self.delta_o, XR)*factor
        # Case (29) (S-1)(O1O2) (S-1)(O1O1)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        factor = -1/sqrt2
        XhX += einsum('vu,uvm,ut,t->m',XL, self.hm[self.slo,self.slo,:], self.delta_o, XR)*factor
        XhX += einsum('vu,uvm,vt,t->m',XL, self.hm[self.slo,self.slo,:], self.delta_o-1/self.S, XR)*factor
        XR = numpy.diag(XR)
        # =============== line4 ===============
        XL = L[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XL = numpy.diag(XL)
        # Case (5) (S-1)(O1O1) (S-1)(CV1) = 0 #
        # Case (14) (S-1)(O1O1) (S-1)(CO1)
        XR = R[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        factor = -1.0/(2*sqrt(self.S*(2*self.S-1)))
        XhX += einsum('iu,uim,ut,t->m',XR, -self.hm[self.slo,self.slc,:],\
                       ((1-self.S)/self.S) + 2*(self.S-1)*self.delta_o, XL)*factor
        # Case (22) (S-1)(O1O1) (S-1)(OV1)
        XR = R[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        factor = -1.0/(2*sqrt(self.S*(2*self.S-1)))
        XhX += einsum('ua,aum,ut,t->m',XR, -self.hm[self.slv,self.slo,:],\
                      ((1-self.S)/self.S + 2*(self.S-1)*self.delta_o), XL)*factor
        # Case (29) (S-1)(O1O1) (S-1)(O1O2)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = XR - numpy.diag(numpy.diag(XR))
        XhX += einsum('vu,uvm,ut,t->m',XR, -self.hm[self.slo,self.slo,:], self.delta_o, XL)*factor
        XhX += einsum('vu,uvm,vt,t->m',XR, -self.hm[self.slo,self.slo,:], self.delta_o-1/self.S, XL)*factor
        # Case (35) (S-1)(O1O1) (S-1)(O1O1) = 0
        return XhX
    
    def interact_S_1GS(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # =============== line0 ===============
        # Case (6) (S-1)(CV1) GS
        XL = L[1][1][self.sl_S_1_dim0].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S-1)/(2*self.S+1))
        XhX += einsum('ia,aim->m',XL, self.hm[self.slv,self.slc,:])*factor
        # =============== line1 ===============
        # Case (15) (S-1)(CO1) GS
        XL = L[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        factor = sqrt((2*self.S-1)/(2*self.S))
        XhX += einsum('iu,uim->m',XL, self.hm[self.slv,self.slo,:])*factor
        # =============== line2 ===============
        # Case (23) (S-1)(OV1) GS
        XL = L[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        factor = sqrt((2*self.S-1)/(2*self.S))
        XhX += einsum('ua,aum->m',XL, self.hm[self.slo,self.slv,:])*factor
        # =============== line3 ===============
        # Case (30) (S-1)(O1O2) GS
        XL = L[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XL = XL - numpy.diag(numpy.diag(XL))
        factor = sqrt((2*self.S-1)/(2*self.S+1))
        XhX += einsum('vu,uvm->m',XL, self.hm[self.slo,self.slo,:])*factor
        # =============== line4 ===============
        # Case (36) (S-1)(O1O1) GS = 0
        return XhX
    
    def interact_S_1S(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # =============== line0 ===============
        XL = L[1][1][self.sl_S_1_dim0].reshape(self.nc,self.nv)
        # Case (7) (S-1)(CV1) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S-1)/(2*self.S+1))/sqrt2
        XhX += einsum('ia,abm,ij,jb->m',XL, self.hm[self.slv,self.slv,:],self.delta_c, XR)*factor
        XhX += einsum('ia,jim,ab,jb->m',XL,-self.hm[self.slc,self.slc,:],self.delta_v, XR)*factor
        # Case (8) (S-1)(CV1) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -sqrt((2*self.S-1)/(2*self.S+1))/(2*self.S)
        XhX += einsum('ia,atm,ij,jt->m',XL, self.hm[self.slv,self.slvo,:],self.delta_c, XR)*factor
        # Case (9) (S-1)(CV1) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = sqrt((2*self.S-1)/(2*self.S+1))/(2*self.S)
        XhX += einsum('ia,tim,ab,tb->m',XL, self.hm[self.slo,self.slc,:],self.delta_v, XR)*factor
        # Case (10) (S-1)(CV1) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(((1+self.S)*(2*self.S-1))/(2*self.S*(2*self.S+1)))
        XhX += einsum('ia,abm,ij,jb->m',XL, self.hm[self.slv,self.slv,:],self.delta_c, XR)*factor
        XhX += einsum('ia,jim,ab,jb->m',XL, self.hm[self.slc,self.slc,:],self.delta_v, XR)*factor
        # =============== line1 ===============
        XL = L[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        # Case (16) (S-1)(CO1) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S-1)/(self.S))/2.
        XhX += einsum('iu,ubm,ij,jb->m',XL, self.hm[self.slo,self.slv,:],self.delta_c, XR)*factor
        # Case (17) (S-1)(CO1) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.nO)
        factor = -sqrt((2*self.S-1)/(2*self.S))
        XhX += einsum('iu,jim,ut,jt->m',XL, self.hm[self.slc,self.slc,:],self.delta_o, XR)*factor
        XhX += einsum('iu,utm,ij,jt->m',XL, self.hm[self.slo,self.slo,:]/(2*self.S-1),self.delta_c, XR)*factor
        # Case (18) (S-1)(CO1) S(OV0) = 0
        # Case (19) (S-1)(CO1) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(((1+self.S)*(2*self.S-1)))/(2*self.S)
        XhX += einsum('iu,ubm,ij,jb->m',XL, self.hm[self.slo,self.slv,:],self.delta_c, XR)*factor
        # =============== line2 ===============
        XL = L[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        # Case (24) (S-1)(OV1) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = -sqrt((2*self.S-1)/(self.S))/2.
        XhX += einsum('ua,jum,ab,jb->m',XL, self.hm[self.slc,self.slo,:],self.delta_v, XR)*factor
        # Case (25) (S-1)(OV1) S(CO0) = 0
        # Case (26) (S-1)(OV1) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = sqrt((2*self.S-1)/(2*self.S))
        XhX += einsum('ua,abm,ut,tb->m',XL, self.hm[self.slv,self.slv,:],self.delta_o, XR)*factor
        XhX += einsum('ua,tum,ab,tb->m',XL, self.hm[self.slo,self.slo,:],self.delta_v, XR)*factor
        # Case (27) (S-1)(OV1) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(((1+self.S)*(2*self.S-1)))/(2*self.S)
        XhX += einsum('ua,jum,ab,jb->m',XL, self.hm[self.slc,self.slo,:],self.delta_v, XR)*factor
        # =============== line3 ===============
        XL = L[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XL = XL - numpy.diag(numpy.diag(XL))
        # Case (31) (S-1)(O1O2) S(CV0) = 0
        # Case (32) (S-1)(O1O2) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1.0
        XhX += einsum('vu,jum,tu,jt->m',XL, self.hm[self.slc,self.slo,:],self.delta_o, XR)*factor
        # Case (33) (S-1)(O1O2) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = 1.0
        XhX += einsum('vu,ubm,vt,tb->m',XL, self.hm[self.slo,self.slv,:],self.delta_o, XR)*factor
        # Case (34) (S-1)(O1O2) S(CV1) = 0
        # =============== line4 ===============
        XL = L[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XL = numpy.diag(XL)
        # Case (37) (S-1)(O1O1) S(CV0) = 0
        # Case (38) (S-1)(O1O1) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1.0
        XhX += einsum('u,jtm,ut,jt->m',XL, self.hm[self.slc,self.slo,:],(self.delta_o-1/(2*self.S)), XR)*factor
        # Case (39) (S-1)(O1O1) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = -1.0
        XhX += einsum('u,tbm,ut,tb->m',XL, self.hm[self.slo,self.slv,:],(self.delta_o-1/(2*self.S)), XR)*factor
        # Case (40) (S-1)(O1O1) S(CV1) = 0
        return XhX
    
    def interact_GSGS(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # Case (41) GS GS = 0
        return XhX

    def interact_GSS(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # Case (42) GS S(CV0) = 0
        # Case (43) GS S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1/sqrt2
        XhX += einsum('jvm,jv->m', self.hm[self.slc,self.slo,:], XR)*factor
        # Case (44) GS S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = 1/sqrt2
        XhX += einsum('vbm,vb->m', self.hm[self.slo,self.slv,:], XR)*factor
        # Case (45) GS S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(self.S/(1+self.S))
        XhX += einsum('jbm,jb->m',self.hm[self.slc,self.slv,:], XR)*factor
        return XhX

    def interact_GSS1(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # Case (46) GS (S+1)(CV1)
        XR = R[1][1].reshape(self.nc,self.nv)
        factor = -1.0
        XhX += einsum('jbm,jb->m', self.hm[self.slc,self.slv,:], XR)*factor
        return XhX
    
    def interact_SS(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # ====== line0 ==========
        XL = L[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        # Case (47) S(CV0) S(CV0) = 0
        # Case (48) S(CV0) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1/2
        XhX += einsum('ia,avm,ij,jv->m',XL, self.hm[self.slv,self.slo,:], self.delta_c, XR)*factor
        # Case (49) S(CV0) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = -1/2
        XhX += einsum('ia,vim,ab,vb->m',XL, self.hm[self.slo,self.slc,:], self.delta_v, XR)*factor
        # Case (50) S(CV0) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(self.S/(2*(1+self.S)))
        XhX += einsum('ia,abm,ij,jb->m',XL, self.hm[self.slv,self.slv,:], self.delta_c, XR)*factor
        XhX += einsum('ia,jim,ab,jb->m',XL,-self.hm[self.slc,self.slc,:], self.delta_v, XR)*factor
        # ====== line1 ==========
        XL = L[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        # Case (48) S(CO0) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = -1/2
        XhX += einsum('ia,avm,ij,jv->m',XR, -self.hm[self.slv,self.slo,::-1].conjugate(), self.delta_c, XL)*factor
        # Case (52) S(CO0) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1/sqrt2
        XhX += einsum('iu,uvm,ij,jv->m',XL, self.hm[self.slo,self.slo,:], self.delta_c, XR)*factor
        XhX += einsum('iu,jim,uv,jv->m',XL,-self.hm[self.slc,self.slc,:], self.delta_o, XR)*factor
        # Case (53) S(CO0) S(OV0) = 0
        # Case (54) S(CO0) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = (1-self.S)/(2*sqrt(self.S*(self.S+1)))
        XhX += einsum('iu,ubm,ij,jb->m',XL,self.hm[self.slo,self.slv,:], self.delta_c, XR)*factor
        # ====== line2 ==========
        XL = L[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        # Case (49) S(OV0) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = -1/2
        XhX += einsum('ia,vim,ab,vb->m',XR, -self.hm[self.slo,self.slc,::-1].conjugate(), self.delta_v, XL)*factor
        # Case (53) S(OV0) S(CO0) = 0
        # Case (56) S(OV0) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = 1/sqrt2
        XhX += einsum('ua,abm,uv,vb->m',XL, self.hm[self.slv,self.slv,:], self.delta_o, XR)*factor
        XhX += einsum('ua,vum,ab,vb->m',XL,-self.hm[self.slo,self.slo,:], self.delta_v, XR)*factor
        # Case (57) S(OV0) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = (self.S-1)/(2*sqrt(self.S*(self.S+1)))
        XhX += einsum('ua,jum,ab,jb->m',XL, self.hm[self.slc,self.slo,:], self.delta_v, XR)*factor
        # ====== line3 ==========
        XL = L[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        # Case (50) S(CV1) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = -sqrt(self.S/(2*(1+self.S)))
        XhX += einsum('ia,abm,ij,jb->m',XR,-self.hm[self.slv,self.slv,::-1].conjugate(), self.delta_c, XL)*factor
        XhX += einsum('ia,jim,ab,jb->m',XR, self.hm[self.slc,self.slc,::-1].conjugate(), self.delta_v, XL)*factor
        # Case (54) S(CV1) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = (1-self.S)/(2*sqrt(self.S*(self.S+1)))
        XhX += einsum('iu,ubm,ij,jb->m',XR,-self.hm[self.slo,self.slv,::-1].conjugate(), self.delta_c, XL)*factor
        # Case (57) S(CV1) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = (self.S-1)/(2*sqrt(self.S*(self.S+1)))
        XhX += einsum('ua,jum,ab,jb->m',XR,-self.hm[self.slc,self.slo,::-1].conjugate(), self.delta_v, XL)*factor
        # Case (59) S(CV1) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = 1/(sqrt2*(1+self.S))
        XhX += einsum('ia,abm,ij,jb->m',XL, self.hm[self.slv,self.slv,:], self.delta_c, XR)*factor
        XhX += einsum('ia,jim,ab,jb->m',XL, self.hm[self.slc,self.slc,:], self.delta_v, XR)*factor
        return XhX

    def interact_SS1(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        XR = R[1][1].reshape(self.nc,self.nv)
        # ====== line0 ==========
        # Case (51) S(CV0) (S+1)(CV1)
        XL = L[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = 1/sqrt2
        XhX += einsum('ia,jim,ab,jb->m',XL, self.hm[self.slc,self.slc,:], self.delta_v, XR)*factor
        XhX += einsum('ia,abm,ij,jb->m',XL,-self.hm[self.slv,self.slv,:], self.delta_c, XR)*factor
        # ====== line1 ==========
        # Case (55) S(CO0) (S+1)(CV1)
        XL = L[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1.
        XhX += einsum('iu,ubm,ij,jb->m',XL, self.hm[self.slo,self.slv,:], self.delta_c, XR)*factor
        # ====== line2 ==========
        # Case (58) S(OV0) (S+1)(CV1)
        XL = L[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = 1.
        XhX += einsum('ua,jum,ab,jb->m',XL, self.hm[self.slc,self.slo,:], self.delta_v, XR)*factor
        # ====== line3 ==========
        # Case (60) (S+1)(CV1) (S+1)(CV1)
        XL = L[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(self.S/(2*(self.S+1)))
        XhX += einsum('ia,jim,ab,jb->m',XL, self.hm[self.slc,self.slc,:], self.delta_v, XR)*factor
        XhX += einsum('ia,abm,ij,jb->m',XL, self.hm[self.slv,self.slv,:], self.delta_c, XR)*factor
        return XhX
    
    def interact_S1S1(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # Case (61) S(CO0) (S+1)(CV1)
        XR = R[1][1].reshape(self.nc,self.nv)
        XL = L[1][1].reshape(self.nc,self.nv)
        factor = 1/sqrt2
        XhX += einsum('ia,abm,ij,jb->m',XL, self.hm[self.slv,self.slv,:], self.delta_c, XR)*factor
        XhX += einsum('ia,jim,ab,jb->m',XL, self.hm[self.slc,self.slc,:], self.delta_v, XR)*factor
        return XhX