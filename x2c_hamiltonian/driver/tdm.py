# A general code to calculate TDM for TDA
# Implemented by Bohan Zhang @BNU 2025.11.28
import numpy as np
from opt_einsum import contract as einsum

def TDM_GSGS(S,XL,XR,ints_mo,n,sl):
    '''
    TDM_GSGS: Calculate
        ⟨ref.|O|ref.⟩ = \sum_pq r_{pq}^{l} ⟨ref.|E_{pq}|ref.⟩ == 0 Case(16)
    Case(16)
    '''
    return np.zeros((3,))

def TDM_GSS(S,XL,XR,ints_mo,n,sl):
    '''
    TDM_GSS: Calculate
        XL⟨Psi_L|O|Psi_R⟩XR = \sum_pq r_{pq}^{l} (XL ⟨Psi_L|E_{pq}|Psi_R⟩ XR)
    Case(17) -- Case(20)
    here
        XL⟨Psi_L|E_{pq}|Psi_R⟩XR is TDM1;
        S: Spin projection Sz of |ref.⟩
        XL: 1 of |ref.⟩
        XR: CI coefficient of |Psi_R⟩
        and here |ref.⟩ and |Psi_R⟩ spin SS
        ints_mo: r_{pq}^{l} (Note that the other dimension (l) should be placed at the foremost index.)
        n: nc,no,nv: number of spatial orbitals (core, open-shell, vitural)
        sl: slc,slo,slv: slice of spatial orbitals (core, open-shell, vitural)
    '''
    slc,slo,slv=sl
    assert np.allclose(XL,np.ones((1,)))
    assert len(XR) == 4
    assert len(XR[0].shape) == 2
    tdm = np.zeros((3,))
    # 0 Case(17) GS-CV0
    tdm += np.sqrt(2)*einsum('xbj,jb->x', ints_mo[:,slv,slc], XR[0])
    # 1 Case(18) GS-CO0
    tdm += einsum('xjt,jt->x', ints_mo[:,slc,slo], XR[1])
    # 2 Case(19) GS-OV0
    tdm += einsum('xtb,tb->x', ints_mo[:,slo,slv], XR[2])
    # 3 Case(20) GS-CV1 0
    return tdm

def TDM_SGS(S,XL,XR,ints_mo,n,sl):
    '''
    TDM_SGS: Calculate
        XL⟨Psi_L|O|Psi_R⟩XR = \sum_pq r_{pq}^{l} (XL ⟨Psi_L|E_{pq}|Psi_R⟩ XR)
    Case(17) -- Case(20)
    here
        XL⟨Psi_L|E_{pq}|Psi_R⟩XR is TDM1;
        S: Spin projection Sz of |ref.⟩
        XL: CI coefficient of |Psi_L⟩
        XR: 1 of |ref.⟩
        and here |Psi_L⟩ and |ref.⟩ spin SS
        ints_mo: r_{pq}^{l} (Note that the other dimension (l) should be placed at the foremost index.)
        n: nc,no,nv: number of spatial orbitals (core, open-shell, vitural)
        sl: slc,slo,slv: slice of spatial orbitals (core, open-shell, vitural)
    '''
    return TDM_GSS(S,XR,XL,ints_mo,n,sl)

def TDM_S(S,XL,XR,ints_mo,n,sl):
    '''
    TDM_S: Calculate
        XL⟨Psi_L|O|Psi_R⟩XR = \sum_pq r_{pq}^{l} (XL ⟨Psi_L|E_{pq}|Psi_R⟩ XR)
    Case(21) -- Case(30)
    here
        XL⟨Psi_L|E_{pq}|Psi_R⟩XR is TDM1;
        S: Spin projection Sz of |ref.⟩
        XL: CI coefficient of |Psi_L⟩
        XR: CI coefficient of |Psi_R⟩
        and here |Psi_L⟩ and |Psi_R⟩ spin SS
        ints_mo: r_{pq}^{l} (Note that the other dimension (l) should be placed at the foremost index.)
        n: nc,no,nv: number of spatial orbitals (core, open-shell, vitural)
        sl: slc,slo,slv: slice of spatial orbitals (core, open-shell, vitural)
    '''
    nc,no,nv=n
    slc,slo,slv=sl
    delta_c = np.eye(nc)
    delta_o = np.eye(no)
    delta_v = np.eye(nv)
    assert len(XL) == 4 and len(XR) == 4
    assert len(XL[0].shape) == 2 and len(XR[0].shape) == 2
    tdm = np.zeros((3,))

    # Diagonal
    # (0,0) Case(21) CV(0)-CV(0) r_ab*delta_ij - r_ij*delta_ab
    tdm +=  einsum('ia,xba,ij,jb->x', XL[0], ints_mo[:,slv,slv], delta_c, XR[0])
    tdm += -einsum('ia,xji,ab,jb->x', XL[0], ints_mo[:,slc,slc], delta_v, XR[0])
    # (1,1) Case(25) CO(0)-CO(0) r_uv*delta_ij - r_ij*delta_uv
    tdm +=  einsum('iu,xvu,ij,jv->x', XL[1], ints_mo[:,slo,slo], delta_c, XR[1])
    tdm += -einsum('iu,xji,uv,jv->x', XL[1], ints_mo[:,slc,slc], delta_o, XR[1])
    # (2,2) Case(28) OV(0)-OV(0) r_ab*delta_uv - r_uv*delta_ab
    tdm +=  einsum('ua,xab,uv,vb->x', XL[2], ints_mo[:,slv,slv], delta_o, XR[2])
    tdm += -einsum('ua,xuv,ab,vb->x', XL[2], ints_mo[:,slo,slo], delta_v, XR[2])
    # (3,3) Case(30) CV(1)-CV(1) r_ab*delta_ij - r_ij*delta_ab
    tdm +=  einsum('ia,xab,ij,jb->x', XL[3], ints_mo[:,slv,slv], delta_c, XR[3])
    tdm += -einsum('ia,xji,ab,jb->x', XL[3], ints_mo[:,slc,slc], delta_v, XR[3])

    # Off-diagonal
    # (0,1) Case(22) CV(0)-CO(0) 1/sqrt{2} * r_at*delta_ij
    factor = 1/np.sqrt(2)
    tdm += factor*einsum('ia,xat,ij,jt->x', XL[0], ints_mo[:,slv,slo], delta_c, XR[1])
    # (1,0) Case(22) CO(0)-CV(0) 1/sqrt{2} * r_at*delta_ij
    tdm += factor*einsum('jt,xat,ij,ia->x', XL[1], ints_mo[:,slv,slo], delta_c, XR[0])
    # (0,2) Case(23) CV(0)-OV(0) -1/sqrt{2} * r_ti*delta_ab
    factor = -1/np.sqrt(2)
    tdm += factor*einsum('ia,xti,ab,tb->x', XL[0], ints_mo[:,slo,slc], delta_v, XR[2])
    # (2,0) Case(23) OV(0)-CV(0) -1/sqrt{2} * r_ti*delta_ab
    tdm += factor*einsum('tb,xti,ab,ia->x', XL[2], ints_mo[:,slo,slc], delta_v, XR[0])
    # (0,3) Case(24) CV(0)-CV(1) 0
    # (3,0) Case(24) CV(1)-CV(0) 0
    # (1,2) Case(24) CO(0)-OV(0) 0
    # (2,1) Case(24) OV(0)-CO(0) 0
    # (1,3) Case(24) CO(0)-CV(1) sqrt((1+S)/(2*S)) * (1+1/S)r_ub*delta_ij
    factor = np.sqrt((1+S)/(2*S))
    tdm += factor*einsum('iu,xub,ij,jb->x', XL[1], ints_mo[:,slo,slv], delta_c, XR[3])
    # (3,1) Case(24) CO(0)-CV(1) sqrt(S/(2*(1+S))) * (1+1/S)r_ub*delta_ij
    tdm += factor*einsum('jb,xub,ij,iu->x', XL[3], ints_mo[:,slo,slv], delta_c, XR[1])
    # (2,3) Case(24) OV(0)-CV(1) sqrt((1+S)/(2*S)) * r_ju*delta_ab
    factor = np.sqrt((1+S)/(2*S))
    tdm += factor*einsum('ua,xju,ab,jb->x', XL[2], ints_mo[:,slc,slo], delta_v, XR[3])
    # (3,2) Case(24) CV(1)-OV(0) sqrt((1+S)/(2*S)) * r_ju*delta_ab
    tdm += factor*einsum('jb,xju,ab,ua->x', XL[3], ints_mo[:,slc,slo], delta_v, XR[2])
    return tdm

def TDM_S1(S,XL,XR,ints_mo,n,sl):
    '''
    TDM_S1: Calculate
        XL⟨Psi_L|O|Psi_R⟩XR = \sum_pq r_{pq}^{l} (XL ⟨Psi_L|E_{pq}|Psi_R⟩ XR)
    Case(31)
    here
        XL⟨Psi_L|E_{pq}|Psi_R⟩XR is TDM1;
        S: Spin projection Sz of |ref.⟩
        XL: CI coefficient of |Psi_L⟩
        XR: CI coefficient of |Psi_R⟩
        and here |Psi_L⟩ and |Psi_R⟩ spin (S+1)(S+1)
        ints_mo: r_{pq}^{l} (Note that the other dimension (l) should be placed at the foremost index.)
        n: nc,no,nv: number of spatial orbitals (core, open-shell, vitural)
        sl: slc,slo,slv: slice of spatial orbitals (core, open-shell, vitural)
    '''
    nc,no,nv=n
    slc,slo,slv=sl
    delta_c = np.eye(nc)
    delta_o = np.eye(no)
    delta_v = np.eye(nv)
    assert len(XL) == 1 and len(XR) == 1
    assert len(XL[0].shape) == 2 and len(XR[0].shape) == 2
    tdm = np.zeros((3,))
    # Case(31) CV1-CV1
    tdm +=  einsum('ia,xab,ij,jb->x', XL[0], ints_mo[:,slv,slv], delta_c, XR[0])
    tdm += -einsum('ia,xji,ab,jb->x', XL[0], ints_mo[:,slc,slc], delta_v, XR[0])
    return tdm

def TDM_S_1(S,XL,XR,ints_mo,n,sl):
    '''
    TDM_S_1: Calculate
        XL⟨Psi_L|O|Psi_R⟩XR = \sum_pq r_{pq}^{l} (XL ⟨Psi_L|E_{pq}|Psi_R⟩ XR)
    Case(1) -- Case(15)
    here
        XL⟨Psi_L|E_{pq}|Psi_R⟩XR is TDM1;
        S: Spin projection Sz of |ref.⟩
        XL: CI coefficient of |Psi_L⟩
        XR: CI coefficient of |Psi_R⟩
        and here |Psi_L⟩ and |Psi_R⟩ spin (S-1)(S-1)
        ints_mo: r_{pq}^{l} (Note that the other dimension (l) should be placed at the foremost index.)
        n: nc,no,nv: number of spatial orbitals (core, open-shell, vitural)
        sl: slc,slo,slv: slice of spatial orbitals (core, open-shell, vitural)
    '''
    nc,no,nv=n
    slc,slo,slv=sl
    delta_c = np.eye(nc)
    delta_o = np.eye(no)
    delta_v = np.eye(nv)
    assert len(XL) == 5 and len(XR) == 5
    assert len(XL[0].shape) == 2 and len(XR[0].shape) == 2
    tdm = np.zeros((3,))

    # 1. (0,0) Case(1) CV1
    factor = 1.0
    tdm +=  factor*einsum('ia,xab,ij,jb->x', XL[0], ints_mo[:,slv,slv], delta_c, XR[0])
    tdm += -factor*einsum('ia,xji,ab,jb->x', XL[0], ints_mo[:,slc,slc], delta_v, XR[0])
    # 2. (1,1) Case(6) CO1
    tdm +=  factor*einsum('iu,xut,ij,jt->x', XL[1], ints_mo[:,slo,slo], delta_c, XR[1])
    tdm += -factor*einsum('iu,xji,ut,jt->x', XL[1], ints_mo[:,slc,slc], delta_o, XR[1])
    # 3. (2,2) Case(10) OV1
    tdm +=  factor*einsum('ua,xab,ut,tb->x', XL[2], ints_mo[:,slv,slv], delta_o, XR[2])
    tdm += -factor*einsum('ua,xtu,ab,tb->x', XL[2], ints_mo[:,slo,slo], delta_v, XR[2])
    # 4. (3,3) Case(13) O1O2
    tdm +=  factor*einsum('vu,xut,vw,wt->x', XL[3], ints_mo[:,slo,slo], delta_o, XR[3])
    tdm += -factor*einsum('vu,xwv,ut,wt->x', XL[3], ints_mo[:,slo,slo], delta_o, XR[3])
    # 5. (3,3) Case(15) O1O1 = 0

    # 6. (0,1) Case(2) CV1-CO1
    factor = np.sqrt((2*S+1)/(2*S))
    tdm +=  factor*einsum('ia,xat,ij,jt->x', XL[0], ints_mo[:,slv,slo], delta_c, XR[1])
    # 7. (1,0) Case(2) CO1-CV1
    tdm +=  factor*einsum('jt,xat,ij,ia->x', XL[1], ints_mo[:,slv,slo], delta_c, XR[0])
    # 8. (0,2) Case(3) CV1-OV1
    factor = -np.sqrt((2*S+1)/(2*S))
    tdm +=  factor*einsum('ia,xti,ab,tb->x', XL[0], ints_mo[:,slo,slc], delta_v, XR[2])
    # 9. (2,0) Case(3) OV1-CV1
    tdm +=  factor*einsum('tb,xti,ab,ia->x', XL[2], ints_mo[:,slo,slc], delta_v, XR[0])
    # 10. (0,3) Case(4) CV1-O1O2 = 0
    # 11. (3,0) Case(4) O1O2-CV1 = 0
    # 12. (0,4) Case(5) CV1-O1O1 = 0
    # 13. (4,0) Case(5) O1O1-CV1 = 0
    # 14. (1,2) Case(7) CO1-OV1 = 0
    # 15. (2,1) Case(7) CO1-OV1 = 0
    # 16. (1,3) Case(8) CO1-O1O2
    factor = -np.sqrt((2*S)/(2*S-1))
    tdm +=  factor*einsum('iu,xwi,ut,wt->x', XL[1], ints_mo[:,slo,slc], delta_o, XR[3])
    # 17. (3,1) Case(8) O1O2-CO1
    tdm +=  factor*einsum('wt,xwi,ut,iu->x', XL[3], ints_mo[:,slo,slc], delta_o, XR[1])
    # 18. (1,4) Case(8) CO1-O1O1
    factor = 1/np.sqrt(2*S*(2*S-1))
    tdm +=  factor*einsum('iu,xui,ut,t->x', XL[1], -2*S*ints_mo[:,slo,slc], delta_o, XR[4])
    # 19. (4,1) Case(8) O1O1-CO1
    tdm +=  factor*einsum('t,xui,ut,iu->x', XL[4], -2*S*ints_mo[:,slo,slc], delta_o, XR[1])
    # 20. (2,3) Case(11) OV1-O1O2
    factor = np.sqrt((2*S)/(2*S-1))
    tdm +=  factor*einsum('ua,xat,uw,wt->x', XL[2], ints_mo[:,slv,slo], delta_o, XR[3])
    # 21. (3,2) Case(11) O1O2-OV1
    tdm +=  factor*einsum('wt,xat,uw,ua->x', XL[3], ints_mo[:,slv,slo], delta_o, XR[2])
    # 22. (2,4) Case(12) OV1-O1O1
    factor = -1/np.sqrt(2*S*(2*S-1))
    tdm +=  factor*einsum('ua,xat,ut,t->x', XL[2], -2*S*ints_mo[:,slv,slo], delta_o, XR[4])
    # 23. (4,2) Case(12) O1O1-OV1
    tdm +=  factor*einsum('t,xat,ut,ua->x', XL[4], -2*S*ints_mo[:,slv,slo], delta_o, XR[2])
    # 24. (3,4) Case(14) O1O2-O1O1
    factor = 1.0
    tdm +=  factor*einsum('vu,xut,vt,t->x', XL[3], ints_mo[:,slo,slo], delta_o, XR[4])
    tdm += -factor*einsum('vu,xtv,ut,t->x', XL[3], ints_mo[:,slo,slo], delta_o, XR[4])
    # 25. (4,3) Case(14) O1O1-O1O2
    tdm +=  factor*einsum('t,xut,vt,vu->x', XL[4], ints_mo[:,slo,slo], delta_o, XR[3])
    tdm += -factor*einsum('t,xtv,ut,vu->x', XL[4], ints_mo[:,slo,slo], delta_o, XR[3])
    return tdm