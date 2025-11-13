# A general SOC-SI code
# |GS> |So> |S+> |S-> interaction
# Implemented by Bohan Zhang @BNU 2025.11.07
import numpy, sys, pyscf, pickle, logging, subprocess, os, time
from math import sqrt as sqrt
sqrt2 = sqrt(2)
sys.path.append('./')
sys.path.append('./../')
sys.path.append('./../../')

from typing import Optional, Union
from pyscf import scf, dft
from opt_einsum import contract as einsum

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

from sympy.physics.wigner import wigner_3j
def w(S,M,Sprime,Mprime):
    '''
    Calculate w(S,M,S',M') geo-factor by using Sympy
    '''
    if abs(wigner_3j(S,1,Sprime,-S,S-Sprime,Sprime).doit().evalf())<1e-9:
        return 0.
    else:
        return (-1)**(S-M) * (wigner_3j(S,1,Sprime,-M,M-Mprime,Mprime) \
                                    / wigner_3j(S,1,Sprime,-S,S-Sprime,Sprime)).evalf()

def symm_matrix(h):
    '''
    Fill the lower triangle of the matrix using the upper triangle
    '''
    return numpy.triu(h) + numpy.triu(h, 1).T.conjugate()

States_list = ['|GS>', '|S->', '|So>', '|S+>']

class SI_driver():
    '''
    Input states: dict
    should be key in '|GS>', '|S->', '|So>', '|S+>'
    and value is list of tuple (e(in Hartree), X(CI coefficient))
    '''
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
        # deal with Ground state
        if int(ngs) == 0:
            self.states['|GS>'] = []
        elif int(ngs) == 1:
            self.states['|GS>'] = [(0.0, numpy.array([1.0,]))]
        if '|So>' not in self.states.keys(): self.states['|So>'] = []
        if '|S+>' not in self.states.keys(): self.states['|S+>'] = []
        if '|S->' not in self.states.keys(): self.states['|S->'] = []
        # calculate nc, no, nv and corresponding slice
        self.cal_dims()

    def kernel(self,print=100):
        '''
        State interaction
        0. Do a SI calculation, return eso, vso and hso.
           one can use `self.cal_hso_pos` function
           to determine the posision of matrix element of your interesting state.
        1. Summary of eso, esf and corresponding state.
        '''
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
        self.codetime = time2-time0
        logging.info(f"End diagonalization, cost time {time2-time1:.2f}s")
        logging.info(f"Ref. E (in Hatree) = {self.mf.e_tot}")
        # logging.info(f"The  E (in eV), shape={self.eso.shape} --")
        # logging.info(f"{numpy.diag(self.Omega)[:20] * unit.ha2eV}")
        # logging.info(f"{self.esf[:20] * unit.ha2eV}")
        # logging.info(f"The  E_soc (in eV) --")
        # logging.info(f"{self.eso[:20] * unit.ha2eV}")
        logging.info(f"End SI calculation, cost time {self.codetime:.2f}s")
        self.summary(print)
        return self.eso, self.vso
    
    def summary(self,printnum=100):
        logging.info(f"{'='*16} Basic setting {'='*16}")
        logging.info(f"Time cost {self.codetime:.2f}s")
        logging.info(f"|Ref⟩ with S = Sz = {self.S}")
        logging.info(f"|Ref⟩ with E (in Hatree) = {self.mf.e_tot}")
        logging.info(f"Interaction among {self.nSm} |S-⟩, {self.ngs} |GS⟩, {self.nSo} |So⟩, {self.nSp} |S+⟩.")
        logging.info(f"{'='*18} Summary of S I calculation {'='*18}")
        print(f"  No   i-th   Excited state    v**2        Esf(eV)       Eso(eV)     Eex(so)(cm-1)")
        for i, e in enumerate(self.eso[:printnum]):
            str_state, e_sf = self.v2state(self.vso[:,i])
            print(f"{i:4d} {str_state}  {e_sf*unit.ha2eV:>12.8f}  {self.eso[i]*unit.ha2eV:>12.8f}     {((self.eso[i]-self.eso[0])*unit.ha2eV*unit.eV2cm_1):>10.2f}")
    
    def print_hso_local(self,SML,SMR):
        SL, ML, iL, igsL = SML
        SR, MR, iR, igsR = SMR
        strL = self.S2str[(SL, igsL)]
        strR = self.S2str[(SR, igsR)]
        marker = '⟨' + strL[1:-1] + f" {ML:.1f} {iL}|hso|" + strR[1:-1] + f" {MR:.1f} {iR}⟩"
        # hso = self.hso[L_pos,R_pos]
        hso = self.hso[self.cal_hso_pos(SL, ML, iL, igsL,SR, MR, iR, igsR)]
        print(f"{marker} = {hso:.8f} Ha. = {hso*unit.ha2eV:.8f} eV = {hso*unit.ha2eV*unit.eV2cm_1:.8f} cm-1")
        return None
    
    def print_hso(self,):
        for L_pos in range(0,self.dim_hso,1):
            breakpoint()
            for R_pos in range(L_pos,self.dim_hso,1):
                SML = self.cal_pos(L_pos)
                SMR = self.cal_pos(R_pos)
                self.print_hso_local(SML,SMR)
        return None
    
    def v2state(self,v):
        '''
        Given v (a eigenvector of Heff)
        return 0. state information with length 27.
               1. corresponding Esf.
        '''
        v2 = (v.T.conjugate()*v).real
        pos = numpy.argmax(v2)
        S, M, ith, igs = self.cal_pos(pos)
        strS = self.S2str[(S, igs)] # e.g. |So>
        state_str = f'{ith:>3d}-th   ' + strS[:-1] + f', {M:+.1f}⟩      {100*v2[pos]:>5.1f}%'
        # e.g. ' 99-th    |So, +0.5⟩  95.3%'
        # Esf
        esf = self.states[strS][ith][0]
        return state_str, esf
    
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
        self.sl_S_1_dim4 = slice(self.cv+self.co+self.ov+self.oo, self.cv+self.co+self.ov+self.oo+self.no, 1)
        # self.sl_S_1_dim4 = slice(self.cv+self.co+self.ov+self.oo, self.cv+self.co+self.ov+self.o1o2+self.o1o1, 1)
        logging.info(f"Slice of |S-> is")
        logging.info(f"|S-> CV(1):   Slice {self.sl_S_1_dim0}, length: {self.cv}")
        logging.info(f"|S-> CO(1):   Slice {self.sl_S_1_dim1}, length: {self.co}")
        logging.info(f"|S-> OV(1):   Slice {self.sl_S_1_dim2}, length: {self.ov}")
        logging.info(f"|S-> O1O2(1): Slice {self.sl_S_1_dim3}, length: {self.oo}")
        logging.info(f"|S-> O1O1(1): Slice {self.sl_S_1_dim4}, length: {self.no}")
        ## |S>
        self.sl_S_dim0 = slice(0, self.cv, 1)
        self.sl_S_dim1 = slice(self.cv, self.cv+self.co, 1)
        self.sl_S_dim2 = slice(self.cv+self.co, self.cv+self.co+self.ov, 1)
        self.sl_S_dim3 = slice(self.cv+self.co+self.ov, self.cv+self.co+self.ov+self.cv, 1)
        logging.info(f"Slice of |So> is")
        logging.info(f"|So> CV(0):   Slice {self.sl_S_dim0}, length: {self.cv}")
        logging.info(f"|So> CO(0):   Slice {self.sl_S_dim1}, length: {self.co}")
        logging.info(f"|So> OV(0):   Slice {self.sl_S_dim2}, length: {self.ov}")
        logging.info(f"|So> CV(1):   Slice {self.sl_S_dim3}, length: {self.cv}")
        ## |S+>
        logging.info(f"Slice of |S+> is")
        logging.info(f"|S+> CV(1):                           length: {self.cv}")
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
        logging.info(f"Interaction among {self.nSm} |S-⟩, {self.ngs} |GS⟩, {self.nSo} |So⟩, {self.nSp} |S+⟩.")

    def cal_hso_pos(self,SL,ML,Li,igsL,SR,MR,Ri,igsR,debug=0):
        '''
        Given four quantum numbers (S,M,i,igs) of one state |Phi⟩ which means
        S, M: spin of state |Phi⟩
        i: i-th excited of state |Phi⟩ (degenerate without considering of SOC)
        igs: |Phi⟩ == reference or not.
        
        Given four quantum numbers of bra(L) and ket(R).
        the hso matrix like
                          ⟨S-|hso|S-⟩ ⟨S-|hso|GS⟩ ⟨S-|hso|So⟩ ⟨S-|hso|S+⟩
                          ⟨GS|hso|S-⟩ ⟨GS|hso|GS⟩ ⟨GS|hso|So⟩ ⟨GS|hso|S+⟩
                          ⟨So|hso|S-⟩ ⟨So|hso|GS⟩ ⟨So|hso|So⟩ ⟨So|hso|S+⟩
                          ⟨S+|hso|S-⟩ ⟨S+|hso|GS⟩ ⟨S+|hso|So⟩ ⟨S+|hso|S+⟩
        can be construct, this function gives the position of matrix element with 
        Bra(SL,ML,Li,igsL) and (SR,MR,Ri,igsR).
        '''
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

    def cal_pso_hso(self,L_pos,R_pos):
        '''
        Given position in hso (L_pos,R_pos)
        Return the quantum number (S,M,i,igs) of bra and ket,
        which is correspondig to matrix in func `cal_hso_pos`
        '''
        return (*self.cal_pso(L_pos), *self.cal_pso(R_pos))
    
    def cal_pos(self,pos):
        igs = 0
        if 0<=pos<self.dim0: # |S->
            S = self.S-1
            M, ith = divmod(pos, self.nSm)
        elif self.dim0<=pos<self.dim1: # |GS>
            pos = pos-self.dim0
            S = self.S
            M = pos
            ith = 0
            igs = 1
        elif self.dim1<=pos<self.dim2: # |So>
            pos = pos-self.dim1
            S = self.S
            M, ith = divmod(pos, self.nSo)
        elif self.dim2<=pos<self.dim_hso: # |S+>
            pos = pos-self.dim2
            S = self.S+1
            M, ith = divmod(pos, self.nSp)
        else:
            logging.warning(f"pos {pos} missing!")
            breakpoint()
        M = -self.S + M
        return (S,M,ith,igs)
    
    def make_heff(self):
        '''
        Construct Heff = hso + Omega
        '''
        logging.info(f"{'='*20} Begin to make heff {'='*20}")
        logging.info(f"dim of Heff = {self.dim_hso}")
        time0 = time.time()
        hso = numpy.zeros((self.dim_hso, self.dim_hso,),dtype=numpy.complex128)
        Omega = numpy.zeros((self.dim_hso, self.dim_hso,))
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
                                    if abs(MR-ML)<=1 and abs(SR-SL)<=1:
                                        # Calculate h*w
                                        hso[Lpos, Rpos] = h[int(MR-ML)+1]*w(SL,ML,SR,MR)
                                    else:
                                        hso[Lpos, Rpos] = 0.0
                                    count += 1
                                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                                        if count%n_sum == 0: progress_bar(count, n_sum, time0)
                                    else:
                                        if count%int(n_sum/10) == 0:
                                            logging.info(f"{count//int(n_sum/10)}0% -- finish {count}/{n_sum}")
                                if Lpos == Rpos: # E^{(0)}_I
                                    assert abs(L_value[0]-R_value[0])<1e-6 # share the same energy
                                    if abs(hso[Lpos, Rpos].imag) > 1e-6: # real if diagonal
                                        logging.warning(f"{(Lpos, Rpos), (SL,ML,Li,igsL,SR,MR,Ri,igsR)},abs(hso[Lpos, Rpos].imag) = {abs(hso[Lpos, Rpos].imag):.2e}")
                                    Omega[Lpos, Rpos] = L[1][0]
        self.hso = symm_matrix(hso)
        self.hso = self.hso - numpy.diag(numpy.diag(self.hso))
        self.Omega = Omega
        self.heff = self.hso + self.Omega
        time1 = time.time()
        ndiff_hso = numpy.linalg.norm(self.hso-self.hso.T.conjugate())
        ndiff_heff = numpy.linalg.norm(self.heff-self.heff.T.conjugate())
        logging.info(f"Norm(Hso-Hso.dagger)={ndiff_hso:.2e}")
        logging.info(f"Norm(Heff-Heff.dagger)={ndiff_heff:.2e}")
        # if ndiff_heff > 1e-6: breakpoint()
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
        hso_local = 0

        # S-1 S-1
        if abs(SL-(self.S-1))<1e-6 and abs(SR-(self.S-1))<1e-6 and int(igsL+igsR) == 0:
            hso_local = self.interact_S_1S_1(L,R); exitcode += 1
        # S-1 GS
        if abs(SL-(self.S-1))<1e-6 and abs(SR-(self.S))<1e-6 and int(igsR-igsL) == 1:
            hso_local = self.interact_S_1GS(L,R); exitcode += 1
        # S-1 S
        if abs(SL-(self.S-1))<1e-6 and abs(SR-(self.S))<1e-6 and int(igsL+igsR) == 0:
            hso_local = self.interact_S_1S(L,R); exitcode += 1
        # S-1 S+1
        if abs(SL-(self.S-1))<1e-6 and abs(SR-(self.S+1))<1e-6 and int(igsL+igsR) == 0:
            hso_local = self.interact_S_1S1(L,R); exitcode += 1

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
    
    # Interaction part
    def interact_S_1S_1(self, L, R):
        dim = self.cv+self.co+self.ov+self.oo+self.no
        hX = numpy.zeros((dim,3,), dtype=numpy.complex128)
        # XL-- S-1 [CV CO OV OO O]
        XL = numpy.zeros((dim,))
        XL[self.sl_S_1_dim0] = L[1][1][self.sl_S_1_dim0]
        XL[self.sl_S_1_dim1] = L[1][1][self.sl_S_1_dim1]
        XL[self.sl_S_1_dim2] = L[1][1][self.sl_S_1_dim2]
        X_ = L[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XL[self.sl_S_1_dim3] = (X_ - numpy.diag(numpy.diag(X_))).reshape(self.oo)
        XL[self.sl_S_1_dim4] = numpy.diag(X_)
        # =============== line0 ===============
        # 0. Case (1) (S-1)(CV1) (S-1)(CV1)
        XR = R[1][1][self.sl_S_1_dim0].reshape(self.nc,self.nv)
        factor = (1-self.S)/(self.S*sqrt2)
        hX[self.sl_S_1_dim0] += factor*einsum('abm,ij,jb->iam',self.hm[self.slv,self.slv,:], self.delta_c, XR).reshape(self.cv,3)
        hX[self.sl_S_1_dim0] += factor*einsum('jim,ab,jb->iam',self.hm[self.slc,self.slc,:], self.delta_v, XR).reshape(self.cv,3)
        # 1. Case (2) (S-1)(CV1) (S-1)(CO1)
        XR = R[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        factor = sqrt((2*self.S+1)/self.S)*(1-self.S)/(self.S*2)
        hX[self.sl_S_1_dim0] += factor*einsum('atm,ij,jt->iam', self.hm[self.slv,self.slo,:], self.delta_c, XR).reshape(self.cv,3)
        # 2. Case (3) (S-1)(CV1) (S-1)(OV1)
        XR = R[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        factor = sqrt((2*self.S+1)/self.S)*(1-self.S)/(self.S*2)
        hX[self.sl_S_1_dim0] += factor*einsum('tim,ab,tb->iam', self.hm[self.slo,self.slc,:], self.delta_v, XR).reshape(self.cv,3)
        # 3. Case (4) (S-1)(CV1) (S-1)(O1O2) = 0
        # 4. Case (5) (S-1)(CV1) (S-1)(O1O1) = 0 #
        # =============== line1 ===============
        # 0. Case (2) (S-1)(CO1) (S-1)(CV1)
        XR = R[1][1][self.sl_S_1_dim0].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S+1)/self.S)*(1-self.S)/(self.S*2)
        hX[self.sl_S_1_dim1] += factor*einsum('ia,atm,ij->jtm',XR, -self.hm[self.slv,self.slo,:], self.delta_c).reshape(self.co,3)
        # 1. Case (11) (S-1)(CO1) (S-1)(CO1)
        XR = R[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        factor = -(self.S-1)/(self.S*sqrt2)
        hX[self.sl_S_1_dim1] += factor*einsum('jim,ut,jt->ium', self.hm[self.slc,self.slc,:], self.delta_o, XR).reshape(self.co,3)
        hX[self.sl_S_1_dim1] += factor*einsum('utm,ij,jt->ium', ((2*self.S+1)/(2*self.S-1))*self.hm[self.slo,self.slo,:], self.delta_c, XR).reshape(self.co,3)
        # 2. Case (12) (S-1)(CO1) (S-1)(OV1) = 0
        # 3. Case (13) (S-1)(CO1) (S-1)(O1O2)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = XR - numpy.diag(numpy.diag(XR))
        factor = -(self.S-1)/(self.S*(2*self.S-1))
        hX[self.sl_S_1_dim1] += factor*einsum('wim,ut,wt->ium', self.hm[self.slo,self.slc,:], self.delta_o, XR).reshape(self.co,3)
        # 4. Case (14) (S-1)(CO1) (S-1)(O1O1)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = numpy.diag(XR)
        factor = -1.0/(2*sqrt(self.S*(2*self.S-1)))
        hX[self.sl_S_1_dim1] += factor*einsum('uim,ut,t->ium', self.hm[self.slo,self.slc,:],\
                       ((1-self.S)/self.S) + 2*(self.S-1)*self.delta_o, XR).reshape(self.co,3)
        # =============== line2 ===============
        # 0. Case (3) (S-1)(OV1) (S-1)(CV1)
        XR = R[1][1][self.sl_S_1_dim0].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S+1)/self.S)*(1-self.S)/(self.S*2)
        hX[self.sl_S_1_dim2] += factor*einsum('ia,tim,ab->tbm',XR, -self.hm[self.slo,self.slc,:], self.delta_v).reshape(self.ov,3)
        # 1. Case (12) (S-1)(OV1) (S-1)(CO1) = 0
        # 2. Case (20) (S-1)(OV1) (S-1)(OV1)
        XR = R[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        factor = -(self.S-1)/(self.S*sqrt2)
        hX[self.sl_S_1_dim2] += factor*einsum('abm,ut,tb->uam', self.hm[self.slv,self.slv,:], self.delta_o, XR).reshape(self.ov,3)
        hX[self.sl_S_1_dim2] += factor*einsum('tum,ab,tb->uam', ((2*self.S+1)/(2*self.S-1))*self.hm[self.slo,self.slo,:], self.delta_v, XR).reshape(self.ov,3)
        # 3. Case (21) (S-1)(OV1) (S-1)(O1O2)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = XR - numpy.diag(numpy.diag(XR))
        factor = (1-self.S)/sqrt(self.S*(2*self.S-1))
        hX[self.sl_S_1_dim2] += factor*einsum('atm,uw,wt->uam', self.hm[self.slv,self.slo,:], self.delta_o, XR).reshape(self.ov,3)
        # 4. Case (22) (S-1)(OV1) (S-1)(O1O1)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = numpy.diag(XR)
        factor = -1.0/(2*sqrt(self.S*(2*self.S-1)))
        hX[self.sl_S_1_dim2] += factor*einsum('aum,ut,t->uam', self.hm[self.slv,self.slo,:],\
                      ((1-self.S)/self.S + 2*(self.S-1)*self.delta_o), XR).reshape(self.ov,3)
        # =============== line3 ===============
        # 0. Case (4) (S-1)(O1O2) (S-1)(CV1) = 0
        # 1. Case (13) (S-1)(O1O2) (S-1)(CO1)
        XR = R[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        factor = -(self.S-1)/(self.S*(2*self.S-1))
        hX[self.sl_S_1_dim3] += factor*einsum('iu,wim,ut->wtm',XR, -self.hm[self.slo,self.slc,:], self.delta_o).reshape(self.oo,3)
        # 2. Case (21) (S-1)(O1O2) (S-1)(OV1)
        XR = R[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        factor = (1-self.S)/sqrt(self.S*(2*self.S-1))
        hX[self.sl_S_1_dim3] += factor*einsum('ua,atm,uw->wtm',XR, -self.hm[self.slv,self.slo,:], self.delta_o).reshape(self.oo,3)
        # 3. Case (28) (S-1)(O1O2) (S-1)(O1O2)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = XR - numpy.diag(numpy.diag(XR))
        factor = -1/sqrt2
        hX[self.sl_S_1_dim3] += factor*einsum('wvm,ut,wt->vum', self.hm[self.slo,self.slo,:], self.delta_o, XR).reshape(self.oo,3)
        hX[self.sl_S_1_dim3] += factor*einsum('utm,vw,wt->vum', self.hm[self.slo,self.slo,:], self.delta_o, XR).reshape(self.oo,3)
        # 4. Case (29) (S-1)(O1O2) (S-1)(O1O1)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = numpy.diag(XR)
        factor = -1/sqrt2
        hX[self.sl_S_1_dim3] += factor*einsum('uvm,ut,t->vum', self.hm[self.slo,self.slo,:], self.delta_o, XR).reshape(self.oo,3)
        hX[self.sl_S_1_dim3] += factor*einsum('uvm,vt,t->vum', self.hm[self.slo,self.slo,:], self.delta_o-1/self.S, XR).reshape(self.oo,3)
        # =============== line4 ===============
        # 0. Case (5) (S-1)(O1O1) (S-1)(CV1) = 0 #
        # 1. Case (14) (S-1)(O1O1) (S-1)(CO1)
        XR = R[1][1][self.sl_S_1_dim1].reshape(self.nc,self.no)
        factor = -1.0/(2*sqrt(self.S*(2*self.S-1)))
        hX[self.sl_S_1_dim4] += factor*einsum('iu,uim,ut->tm',XR, -self.hm[self.slo,self.slc,:],\
                       ((1-self.S)/self.S) + 2*(self.S-1)*self.delta_o)
        # 2. Case (22) (S-1)(O1O1) (S-1)(OV1)
        XR = R[1][1][self.sl_S_1_dim2].reshape(self.no,self.nv)
        factor = -1.0/(2*sqrt(self.S*(2*self.S-1)))
        hX[self.sl_S_1_dim4] += factor*einsum('ua,aum,ut->tm',XR, -self.hm[self.slv,self.slo,:],\
                      ((1-self.S)/self.S + 2*(self.S-1)*self.delta_o))
        # 3. Case (29) (S-1)(O1O1) (S-1)(O1O2)
        XR = R[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XR = XR - numpy.diag(numpy.diag(XR))
        factor = -1/sqrt2
        hX[self.sl_S_1_dim4] += factor*einsum('vu,uvm,ut->tm',XR, -self.hm[self.slo,self.slo,:], self.delta_o)
        hX[self.sl_S_1_dim4] += factor*einsum('vu,uvm,vt->tm',XR, -self.hm[self.slo,self.slo,:], self.delta_o-1/self.S)
        # 4. Case (35) (S-1)(O1O1) (S-1)(O1O1) = 0
        return XL@hX
    
    def interact_S_1GS(self, L, R):
        dim = self.cv+self.co+self.ov+self.oo+self.no
        hX = numpy.zeros((dim,3,), dtype=numpy.complex128)
        # XL-- S-1 [CV CO OV OO O]
        XL = numpy.zeros((dim,))
        XL[self.sl_S_1_dim0] = L[1][1][self.sl_S_1_dim0]
        XL[self.sl_S_1_dim1] = L[1][1][self.sl_S_1_dim1]
        XL[self.sl_S_1_dim2] = L[1][1][self.sl_S_1_dim2]
        X_ = L[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XL[self.sl_S_1_dim3] = (X_ - numpy.diag(numpy.diag(X_))).reshape(self.oo)
        XL[self.sl_S_1_dim4] = numpy.diag(X_)
        # =============== line0 ===============
        # Case (6) (S-1)(CV1) GS
        factor = sqrt((2*self.S-1)/(2*self.S+1))
        hX[self.sl_S_1_dim0] += factor*einsum('aim->iam', self.hm[self.slv,self.slc,:]).reshape(self.cv,3)
        # =============== line1 ===============
        # Case (15) (S-1)(CO1) GS
        factor = sqrt((2*self.S-1)/(2*self.S))
        hX[self.sl_S_1_dim1] += factor*einsum('uim->ium', self.hm[self.slo,self.slc,:]).reshape(self.co,3)
        # =============== line2 ===============
        # Case (23) (S-1)(OV1) GS
        factor = sqrt((2*self.S-1)/(2*self.S))
        hX[self.sl_S_1_dim2] += factor*einsum('aum->uam', self.hm[self.slv,self.slo,:]).reshape(self.ov,3)
        # =============== line3 ===============
        # Case (30) (S-1)(O1O2) GS
        factor = sqrt((2*self.S-1)/(2*self.S+1))
        hX[self.sl_S_1_dim3] += factor*einsum('uvm->vum', self.hm[self.slo,self.slo,:]).reshape(self.oo,3)
        # =============== line4 ===============
        # Case (36) (S-1)(O1O1) GS = 0
        return XL@hX
    
    def interact_S_1S(self, L, R):
        dim = self.cv+self.co+self.ov+self.oo+self.no
        hX = numpy.zeros((dim,3,), dtype=numpy.complex128)
        # XL-- S-1 [CV CO OV OO O]
        XL = numpy.zeros((dim,))
        XL[self.sl_S_1_dim0] = L[1][1][self.sl_S_1_dim0]
        XL[self.sl_S_1_dim1] = L[1][1][self.sl_S_1_dim1]
        XL[self.sl_S_1_dim2] = L[1][1][self.sl_S_1_dim2]
        X_ = L[1][1][self.sl_S_1_dim3].reshape(self.no,self.no)
        XL[self.sl_S_1_dim3] = (X_ - numpy.diag(numpy.diag(X_))).reshape(self.oo)
        XL[self.sl_S_1_dim4] = numpy.diag(X_)
        # =============== line0 ===============
        # Case (7) (S-1)(CV1) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S-1)/(2*self.S+1))/sqrt2
        hX[self.sl_S_1_dim0] += factor*einsum('abm,ij,jb->iam', self.hm[self.slv,self.slv,:],self.delta_c, XR).reshape(self.cv,3)
        hX[self.sl_S_1_dim0] += factor*einsum('jim,ab,jb->iam',-self.hm[self.slc,self.slc,:],self.delta_v, XR).reshape(self.cv,3)
        # Case (8) (S-1)(CV1) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -sqrt((2*self.S-1)/(2*self.S+1))/(2*self.S)
        hX[self.sl_S_1_dim0] += factor*einsum('atm,ij,jt->iam', self.hm[self.slv,self.slo,:],self.delta_c, XR).reshape(self.cv,3)
        # Case (9) (S-1)(CV1) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = sqrt((2*self.S-1)/(2*self.S+1))/(2*self.S)
        hX[self.sl_S_1_dim0] += factor*einsum('tim,ab,tb->iam', self.hm[self.slo,self.slc,:],self.delta_v, XR).reshape(self.cv,3)
        # Case (10) (S-1)(CV1) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(((1+self.S)*(2*self.S-1))/(2*self.S*(2*self.S+1)))
        hX[self.sl_S_1_dim0] += factor*einsum('abm,ij,jb->iam', self.hm[self.slv,self.slv,:],self.delta_c, XR).reshape(self.cv,3)
        hX[self.sl_S_1_dim0] += factor*einsum('jim,ab,jb->iam', self.hm[self.slc,self.slc,:],self.delta_v, XR).reshape(self.cv,3)
        # =============== line1 ===============
        # Case (16) (S-1)(CO1) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = sqrt((2*self.S-1)/(self.S))/2.
        hX[self.sl_S_1_dim1] += factor*einsum('ubm,ij,jb->ium', self.hm[self.slo,self.slv,:],self.delta_c, XR).reshape(self.co,3)
        # Case (17) (S-1)(CO1) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -sqrt((2*self.S-1)/(2*self.S))
        hX[self.sl_S_1_dim1] += factor*einsum('jim,ut,jt->ium', self.hm[self.slc,self.slc,:],self.delta_o, XR).reshape(self.co,3)
        hX[self.sl_S_1_dim1] += factor*einsum('utm,ij,jt->ium', self.hm[self.slo,self.slo,:]/(2*self.S-1),self.delta_c, XR).reshape(self.co,3)
        # Case (18) (S-1)(CO1) S(OV0) = 0
        # Case (19) (S-1)(CO1) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(((1+self.S)*(2*self.S-1)))/(2*self.S)
        hX[self.sl_S_1_dim1] += factor*einsum('ubm,ij,jb->ium', self.hm[self.slo,self.slv,:],self.delta_c, XR).reshape(self.co,3)
        # =============== line2 ===============
        # Case (24) (S-1)(OV1) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = -sqrt((2*self.S-1)/(self.S))/2.
        hX[self.sl_S_1_dim2] += factor*einsum('jum,ab,jb->uam', self.hm[self.slc,self.slo,:],self.delta_v, XR).reshape(self.ov,3)
        # Case (25) (S-1)(OV1) S(CO0) = 0
        # Case (26) (S-1)(OV1) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = sqrt((2*self.S-1)/(2*self.S))
        hX[self.sl_S_1_dim2] += factor*einsum('abm,ut,tb->uam', self.hm[self.slv,self.slv,:],self.delta_o, XR).reshape(self.ov,3)
        hX[self.sl_S_1_dim2] += factor*einsum('tum,ab,tb->uam', self.hm[self.slo,self.slo,:],self.delta_v, XR).reshape(self.ov,3)
        # Case (27) (S-1)(OV1) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(((1+self.S)*(2*self.S-1)))/(2*self.S)
        hX[self.sl_S_1_dim2] += factor*einsum('jum,ab,jb->uam', self.hm[self.slc,self.slo,:],self.delta_v, XR).reshape(self.ov,3)
        # =============== line3 ===============
        # Case (31) (S-1)(O1O2) S(CV0) = 0
        # Case (32) (S-1)(O1O2) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1.0
        hX[self.sl_S_1_dim3] += factor*einsum('jvm,tu,jt->vum', self.hm[self.slc,self.slo,:],self.delta_o, XR).reshape(self.oo,3)
        # Case (33) (S-1)(O1O2) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = 1.0
        hX[self.sl_S_1_dim3] += factor*einsum('ubm,vt,tb->vum', self.hm[self.slo,self.slv,:],self.delta_o, XR).reshape(self.oo,3)
        # Case (34) (S-1)(O1O2) S(CV1) = 0
        # =============== line4 ===============
        # Case (37) (S-1)(O1O1) S(CV0) = 0
        # Case (38) (S-1)(O1O1) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1.0
        hX[self.sl_S_1_dim4] += factor*einsum('jtm,ut,jt->um', self.hm[self.slc,self.slo,:],(self.delta_o-1/(2*self.S)), XR)
        # Case (39) (S-1)(O1O1) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = -1.0
        hX[self.sl_S_1_dim4] += factor*einsum('tbm,ut,tb->um', self.hm[self.slo,self.slv,:],(self.delta_o-1/(2*self.S)), XR)
        # Case (40) (S-1)(O1O1) S(CV1) = 0
        return XL@hX
    
    def interact_S_1S1(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
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
        XhX += factor*einsum('jvm,jv->m', self.hm[self.slc,self.slo,:], XR)
        # Case (44) GS S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = 1/sqrt2
        XhX += factor*einsum('vbm,vb->m', self.hm[self.slo,self.slv,:], XR)
        # Case (45) GS S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(self.S/(1+self.S))
        XhX += factor*einsum('jbm,jb->m',self.hm[self.slc,self.slv,:], XR)
        return XhX

    def interact_GSS1(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # Case (46) GS (S+1)(CV1)
        XR = R[1][1].reshape(self.nc,self.nv)
        factor = -1.0
        XhX += factor*einsum('jbm,jb->m', self.hm[self.slc,self.slv,:], XR)
        return XhX
    
    def interact_SS(self, L, R):
        # XL-- S [CV CO OV CV]
        XL = L[1][1]
        hX = numpy.zeros((XL.shape[0],3,), dtype=numpy.complex128)
        # ====== line0 ==========
        # Case (47) S(CV0) S(CV0) = 0
        # Case (48) S(CV0) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1/2
        hX[self.sl_S_dim0] += factor*einsum('avm,ij,jv->iam', self.hm[self.slv,self.slo,:], self.delta_c, XR).reshape(self.cv,3)
        # Case (49) S(CV0) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = -1/2
        hX[self.sl_S_dim0] += factor*einsum('vim,ab,vb->iam', self.hm[self.slo,self.slc,:], self.delta_v, XR).reshape(self.cv,3)
        # Case (50) S(CV0) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = -sqrt(self.S/(2*(1+self.S)))
        hX[self.sl_S_dim0] += factor*einsum('abm,ij,jb->iam', self.hm[self.slv,self.slv,:], self.delta_c, XR).reshape(self.cv,3)
        hX[self.sl_S_dim0] += factor*einsum('jim,ab,jb->iam',-self.hm[self.slc,self.slc,:], self.delta_v, XR).reshape(self.cv,3)
        # ====== line1 ==========
        # Case (48) S(CO0) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = -1/2
        hX[self.sl_S_dim1] += factor*einsum('ia,avm,ij->jvm',XR, -self.hm[self.slv,self.slo,:], self.delta_c).reshape(self.co,3)
        # Case (52) S(CO0) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = -1/sqrt2
        hX[self.sl_S_dim1] += factor*einsum('uvm,ij,jv->ium', self.hm[self.slo,self.slo,:], self.delta_c, XR).reshape(self.co,3)
        hX[self.sl_S_dim1] += factor*einsum('jim,uv,jv->ium',-self.hm[self.slc,self.slc,:], self.delta_o, XR).reshape(self.co,3)
        # Case (53) S(CO0) S(OV0) = 0
        # Case (54) S(CO0) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = (1-self.S)/(2*sqrt(self.S*(self.S+1)))
        hX[self.sl_S_1_dim1] += factor*einsum('ubm,ij,jb->ium',self.hm[self.slo,self.slv,:], self.delta_c, XR).reshape(self.co,3)
        # ====== line2 ==========
        # Case (49) S(OV0) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = -1/2
        hX[self.sl_S_dim2] += factor*einsum('ia,vim,ab->vbm',XR, -self.hm[self.slo,self.slc,:], self.delta_v).reshape(self.ov,3)
        # Case (53) S(OV0) S(CO0) = 0
        # Case (56) S(OV0) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = 1/sqrt2
        hX[self.sl_S_dim2] += factor*einsum('abm,uv,vb->uam', self.hm[self.slv,self.slv,:], self.delta_o, XR).reshape(self.ov,3)
        hX[self.sl_S_dim2] += factor*einsum('vum,ab,vb->uam',-self.hm[self.slo,self.slo,:], self.delta_v, XR).reshape(self.ov,3)
        # Case (57) S(OV0) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = (self.S-1)/(2*sqrt(self.S*(self.S+1)))
        hX[self.sl_S_dim2] += factor*einsum('jum,ab,jb->uam', self.hm[self.slc,self.slo,:], self.delta_v, XR).reshape(self.ov,3)
        # ====== line3 ==========
        # Case (50) S(CV1) S(CV0)
        XR = R[1][1][self.sl_S_dim0].reshape(self.nc,self.nv)
        factor = -sqrt(self.S/(2*(1+self.S)))
        hX[self.sl_S_dim3] += factor*einsum('ia,abm,ij->jbm',XR,-self.hm[self.slv,self.slv,:], self.delta_c).reshape(self.cv,3)
        hX[self.sl_S_dim3] += factor*einsum('ia,jim,ab->jbm',XR, self.hm[self.slc,self.slc,:], self.delta_v).reshape(self.cv,3)
        # Case (54) S(CV1) S(CO0)
        XR = R[1][1][self.sl_S_dim1].reshape(self.nc,self.no)
        factor = (1-self.S)/(2*sqrt(self.S*(self.S+1)))
        hX[self.sl_S_dim3] += factor*einsum('iu,ubm,ij->jbm',XR,-self.hm[self.slo,self.slv,:], self.delta_c).reshape(self.cv,3)
        # Case (57) S(CV1) S(OV0)
        XR = R[1][1][self.sl_S_dim2].reshape(self.no,self.nv)
        factor = (self.S-1)/(2*sqrt(self.S*(self.S+1)))
        hX[self.sl_S_dim3] += factor*einsum('ua,jum,ab->jbm',XR,-self.hm[self.slc,self.slo,:], self.delta_v).reshape(self.cv,3)
        # Case (59) S(CV1) S(CV1)
        XR = R[1][1][self.sl_S_dim3].reshape(self.nc,self.nv)
        factor = 1/(sqrt2*(1+self.S))
        hX[self.sl_S_dim3] += factor*einsum('abm,ij,jb->iam', self.hm[self.slv,self.slv,:], self.delta_c, XR).reshape(self.cv,3)
        hX[self.sl_S_dim3] += factor*einsum('jim,ab,jb->iam', self.hm[self.slc,self.slc,:], self.delta_v, XR).reshape(self.cv,3)
        return XL@hX

    def interact_SS1(self, L, R):
        # XL-- S [CV CO OV CV]
        XL = L[1][1]
        hX = numpy.zeros((XL.shape[0],3,), dtype=numpy.complex128)
        XR = R[1][1].reshape(self.nc,self.nv)
        # ====== line0 ==========
        # Case (51) S(CV0) (S+1)(CV1)
        factor = 1/sqrt2
        hX[self.sl_S_dim0] += factor*einsum('jim,ab,jb->iam', self.hm[self.slc,self.slc,:], self.delta_v, XR).reshape(self.cv,3)
        hX[self.sl_S_dim0] += factor*einsum('abm,ij,jb->iam',-self.hm[self.slv,self.slv,:], self.delta_c, XR).reshape(self.cv,3)
        # ====== line1 ==========
        # Case (55) S(CO0) (S+1)(CV1)
        factor = -1.
        hX[self.sl_S_dim1] += factor*einsum('ubm,ij,jb->ium', self.hm[self.slo,self.slv,:], self.delta_c, XR).reshape(self.co,3)
        # ====== line2 ==========
        # Case (58) S(OV0) (S+1)(CV1)
        factor = 1.
        hX[self.sl_S_dim2] += factor*einsum('jum,ab,jb->uam', self.hm[self.slc,self.slo,:], self.delta_v, XR).reshape(self.ov,3)
        # ====== line3 ==========
        # Case (60) (S+1)(CV1) (S+1)(CV1)
        factor = -sqrt(self.S/(2*(self.S+1)))
        hX[self.sl_S_dim3] += factor*einsum('jim,ab,jb->iam', self.hm[self.slc,self.slc,:], self.delta_v, XR).reshape(self.cv,3)
        hX[self.sl_S_dim3] += factor*einsum('abm,ij,jb->iam', self.hm[self.slv,self.slv,:], self.delta_c, XR).reshape(self.cv,3)
        return XL@hX
    
    def interact_S1S1(self, L, R):
        XhX = numpy.zeros((3,), dtype=numpy.complex128)
        # Case (61) S(CO0) (S+1)(CV1)
        XR = R[1][1].reshape(self.nc,self.nv)
        XL = L[1][1].reshape(self.nc,self.nv)
        factor = 1/sqrt2
        XhX += factor*einsum('ia,abm,ij,jb->m',XL, self.hm[self.slv,self.slv,:], self.delta_c, XR)
        XhX += factor*einsum('ia,jim,ab,jb->m',XL, self.hm[self.slc,self.slc,:], self.delta_v, XR)
        return XhX