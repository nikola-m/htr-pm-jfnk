"""
htr_jfnk_v9.3.py
==============================================================================
HTR-PM Secondary Steam Circuit — Physics-Based Preconditioned JFNK Solver (v9.3)
==============================================================================

A Jacobian-Free Newton-Krylov (JFNK) framework for the steady-state secondary
steam circuit of the 2 x 250 MWth HTR-PM pebble-bed high-temperature gas-cooled
reactor, with a rigorous comparison of *physics-based preconditioners*:

    (1) None            unpreconditioned GMRES               (reference)
    (2) Jacobi          diagonal residual scaling            (baseline)
    (3) FieldSplit      block-Jacobi over the physical
                        fields  {m, p, h, SG}                (this work)
    (4) ASPIN           Additive-Schwarz Preconditioned
                        Inexact Newton, nonlinear subdomain
                        sweeps + FieldSplit inner solve      (this work)

Central thesis:
    A true FieldSplit(BlockJacobi) preconditioner and an ASPIN nonlinear
    preconditioner both dramatically compress the Jacobian spectrum and
    accelerate the outer Newton iteration; ASPIN additionally removes the
    two-phase / phase-boundary nonlinear stiffness, giving a markedly larger
    basin of attraction for hard (off-design, far) initial guesses.

Model (50 DOF):
    x = [ m_0..m_14 | p_0..p_14 | h_0..h_14 | l_pre, l_eva, T_He1, T_He2, T_He3 ]

    * 15-node regenerative Rankine main circuit:
        HP(2-stage) + LP(4 CV) turbines, 3 LP closed heaters, direct-contact
        deaerator, HP closed heater, condensate & feedwater pumps.
    * 3-zone dynamic-grid once-through helical SG (PRE / EVA / SUP) added as a
        *well-posed, non-redundant* 5-equation block — the steam outlet
        enthalpy h_0 becomes an OUTPUT of the SG heat-transfer model (UA.LMTD),
        not a fixed boundary condition.  This is the key physical upgrade,
        implemented so that every SG variable is the sole subject of exactly
        one equation (structural regularity -> non-singular Jacobian).
    * Entropy-based isentropic expansion that correctly resolves wet-steam
        turbine stages (no polytropic temperature-ratio approximation).
    * Compressed-liquid T(P,h) inversion.

Quick call example:
    x,_=JFNKSolver(FieldSplitPreconditioner(),tol=1e-10).solve(build_x0())
    R=extract_results(x)

References:
    [1] Jiang, Wu, Zhang, Liu, Guo, Li.  A Modified JFNK for Solving the HTR
        Steady State Secondary Circuit Problem.  Energies 2023, 16, 2252.
    [2] Zhang et al.  Current status and technical description of the Chinese
        2 x 250 MWth HTR-PM demonstration plant.  Nucl. Eng. Des. 2009, 239.
==============================================================================
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
import warnings, os, time, math
from functools import lru_cache
warnings.filterwarnings("ignore")

_log=math.log; _exp=math.exp; _sqrt=math.sqrt
def _clip(v,lo,hi): return lo if v<lo else (hi if v>hi else v)

# =============================================================================
#  SECTION A.  IAPWS-97 THERMODYNAMIC PROPERTY LAYER
# =============================================================================
_Pc = 22.064e6           # critical pressure [Pa]
_Tc = 647.096            # critical temperature [K]

# 19-point saturation table spanning 611.657 Pa .. 22.064 MPa
_Ps = np.array([611.657,1e3,3e3,6e3,1e4,3e4,7e4,101325,
                2e5,5e5,1e6,2e6,3.5e6,7e6,1e7,13.9e6,17e6,2e7,22.064e6])
_hf = np.array([0,29.3,101,151.5,191.8,289.2,376.9,417.4,504.7,640.1,
                762.5,908.5,1049.8,1267,1408,1571.3,1670,1778,2084.3])*1e3
_hg = np.array([2500.9,2513.7,2545.5,2567.5,2583.9,2625.4,2660,2675.6,
                2706.3,2748.1,2777.1,2798.4,2803.3,2773.2,2725.5,2602.7,
                2453,2183,2084.3])*1e3
_sf = np.array([0.000,0.106,0.354,0.521,0.649,0.944,1.193,1.303,
                1.530,1.861,2.139,2.448,2.725,3.122,3.360,3.726,
                3.890,4.113,4.407])
_sg = np.array([9.155,9.118,8.578,8.331,8.150,7.671,7.359,7.356,
                7.127,6.821,6.585,6.340,6.124,5.814,5.615,5.211,
                4.915,4.403,4.407])
_Tsat = np.array([273.16,280.13,297.23,309.31,318.97,342.27,362.08,373.15,
                  393.35,424.95,453.05,485.55,515.75,558.05,584.25,610.37,
                  625.55,638.95,647.10])

from scipy.interpolate import CubicSpline

class _FastCubic:
    """Lightweight scalar natural-cubic-spline evaluator (linear extrapolation),
    ~15x faster per scalar call than scipy.interpolate.interp1d dispatch."""
    __slots__=("x","c","x0","x1")
    def __init__(self, xk, yk):
        cs=CubicSpline(xk, yk, extrapolate=True)
        self.x=np.ascontiguousarray(cs.x); self.c=np.ascontiguousarray(cs.c)
        self.x0=self.x[0]; self.x1=self.x[-1]
    def __call__(self, xq):
        x=self.x
        i=int(np.searchsorted(x, xq)-1)
        if i<0: i=0
        elif i>len(x)-2: i=len(x)-2
        d=xq-x[i]; c=self.c
        return ((c[0,i]*d+c[1,i])*d+c[2,i])*d+c[3,i]

class _FastLinear:
    __slots__=("x","y")
    def __init__(self, xk, yk): self.x=np.asarray(xk); self.y=np.asarray(yk)
    def __call__(self, xq): return float(np.interp(xq, self.x, self.y))

_lnPs   = np.log(_Ps)
_hf_spl = _FastCubic(_lnPs,_hf)
_hg_spl = _FastCubic(_lnPs,_hg)
_sf_spl = _FastCubic(_lnPs,_sf)
_sg_spl = _FastCubic(_lnPs,_sg)
_Ts_spl = _FastCubic(_lnPs,_Tsat)

# Inverse mapping saturated-liquid enthalpy -> temperature, used for the
# compressed-liquid temperature (weak-pressure approximation h(P,T) ~ hf(T)).
_ord         = np.argsort(_hf)
_Tsat_of_hf  = _FastCubic(_hf[_ord],_Tsat[_ord])
_lnTsat      = np.log(_Tsat)
_Psat_of_lnT = _FastCubic(_lnTsat,_Ps)

# Superheated-vapour cp spline (7 pressure points) [J/(kg K)]
_Pcpv    = np.array([0.1,0.4,0.9,1.8,3.5,7.5,13.5])
_cpv     = np.array([2.00,2.10,2.25,2.46,2.69,3.54,4.08])*1e3
_cpv_spl = _FastLinear(np.log(_Pcpv),_cpv)

# IAPWS-97 Region-4 backward equation coefficients (Table 34)
_nsat = [1.16705214527670e3,-7.24213167032060e5,-1.70738469400920e1,
         1.20208247024700e4,-3.23255503223330e6, 1.49151086135300e1,
        -4.82326573615910e3, 4.05113405420570e5,-2.38555575678490e-1,
         6.50175348447980e2]

_lnPcpv0=_log(_Pcpv[0]*1e6); _lnPcpv1=_log(_Pcpv[-1]*1e6)
def _cpv_of_P(P):
    lnP=_lnPcpv0 if P<_Pcpv[0]*1e6 else (_lnPcpv1 if P>_Pcpv[-1]*1e6 else _log(P))
    return _cpv_spl(lnP-_log(1e6))


# ---------------------------------------------------------------------------
#  IAPWS-IF97 Region 1 (liquid) and Region 2 (vapour) basic equations.
#
#  The saturation enthalpies/entropies were previously taken from 19-knot
#  cubic splines in ln P.  h_f(P) curves sharply on approach to the critical
#  point, so the widest knot interval fell exactly on the SG operating
#  pressure and h_f(13.9 MPa) came out 2.0 % high (1571.3 vs 1541.0 kJ/kg).
#  Evaluating the IF97 basic equations directly removes the interpolation
#  error entirely. The cycle never exceeds 14.456 MPa (T_sat = 340 C
#  < 623.15 K), so Region 1 covers the whole liquid range and no Region 3
#  backward equation is required.
#
#  Verified against the IF97 release verification tables (worst relative
#  error 1.8e-9) and cross-checked against an independent implementation.
# ---------------------------------------------------------------------------
_RW = 461.526     # specific gas constant of water [J/(kg K)]

_R1 = np.array([
 (0,-2, 0.14632971213167),   (0,-1,-0.84548187169114),
 (0, 0,-0.37563603672040e1), (0, 1, 0.33855169168385e1),
 (0, 2,-0.95791963387872),   (0, 3, 0.15772038513228),
 (0, 4,-0.16616417199501e-1),(0, 5, 0.81214629983568e-3),
 (1,-9, 0.28319080123804e-3),(1,-7,-0.60706301565874e-3),
 (1,-1,-0.18990068218419e-1),(1, 0,-0.32529748770505e-1),
 (1, 1,-0.21841717175414e-1),(1, 3,-0.52838357969930e-4),
 (2,-3,-0.47184321073267e-3),(2, 0,-0.30001780793026e-3),
 (2, 1, 0.47661393906987e-4),(2, 3,-0.44141845330846e-5),
 (2,17,-0.72694996297594e-15),(3,-4,-0.31679644845054e-4),
 (3, 0,-0.28270797985312e-5),(3, 6,-0.85205128120103e-9),
 (4,-5,-0.22425281908000e-5),(4,-2,-0.65171222895601e-6),
 (4,10,-0.14341729937924e-12),(5,-8,-0.40516996860117e-6),
 (8,-11,-0.12734301741641e-8),(8,-6,-0.17424871230634e-9),
 (21,-29,-0.68762131295531e-18),(23,-31, 0.14478307828521e-19),
 (29,-38, 0.26335781662795e-22),(30,-39,-0.11947622640071e-22),
 (31,-40, 0.18228094581404e-23),(32,-41,-0.93537087292458e-25)])
_I1=_R1[:,0]; _J1=_R1[:,1]; _n1=_R1[:,2]
_I1m1=_I1-1; _J1m1=_J1-1

_R20 = np.array([
 ( 0,-0.96927686500217e1),( 1, 0.10086655968018e2),
 (-5,-0.56087911283020e-2),(-4, 0.71452738081455e-1),
 (-3,-0.40710498223928),  (-2, 0.14240819171444e1),
 (-1,-0.43839511319450e1),( 2,-0.28408632460772),
 ( 3, 0.21268463753307e-1)])
_J20=_R20[:,0]; _n20=_R20[:,1]; _J20m1=_J20-1

_R2R = np.array([
 (1, 0,-0.17731742473213e-2),(1, 1,-0.17834862292358e-1),
 (1, 2,-0.45996013696365e-1),(1, 3,-0.57581259083432e-1),
 (1, 6,-0.50325278727930e-1),(2, 1,-0.33032641670203e-4),
 (2, 2,-0.18948987516315e-3),(2, 4,-0.39392777243355e-2),
 (2, 7,-0.43797295650573e-1),(2,36,-0.26674547914087e-4),
 (3, 0, 0.20481737692309e-7),(3, 1, 0.43870667284435e-6),
 (3, 3,-0.32277677238570e-4),(3, 6,-0.15033924542148e-2),
 (3,35,-0.40668253562649e-1),(4, 1,-0.78847309559367e-9),
 (4, 2, 0.12790717852285e-7),(4, 3, 0.48225372718507e-6),
 (5, 7, 0.22922076337661e-5),(6, 3,-0.16714766451061e-10),
 (6,16,-0.21171472321355e-2),(6,35,-0.23895741934104e2),
 (7, 0,-0.59059564324270e-17),(7,11,-0.12621808899101e-5),
 (7,25,-0.38946842435739e-1),(8, 8, 0.11256211360459e-10),
 (8,36,-0.82311340897998e1),(9,13, 0.19809712802088e-7),
 (10, 4, 0.10406965210174e-18),(10,10,-0.10234747095929e-12),
 (10,14,-0.10018179379511e-8),(16,29,-0.80882908646985e-10),
 (16,50, 0.10693031879409),(18,57,-0.33662250574171),
 (20,20, 0.89185845355421e-24),(20,35, 0.30629316876232e-12),
 (20,48,-0.42002467698208e-5),(21,21,-0.59056029685639e-25),
 (22,53, 0.37826947613457e-5),(23,39,-0.12768608934681e-14),
 (24,26, 0.73087610595061e-28),(24,40, 0.55414715350778e-16),
 (24,58,-0.94369707241210e-6)])
_I2=_R2R[:,0]; _J2=_R2R[:,1]; _n2=_R2R[:,2]; _J2m1=_J2-1

def _if97_r1(P,T):
    """Region 1: returns (h, s) in J/kg and J/(kg K)."""
    pi=P/16.53e6; tau=1386.0/T
    a=7.1-pi; b=tau-1.222
    ap=a**_I1; bp=b**_J1
    g  =float(np.sum(_n1*ap*bp))
    g_t=float(np.sum(_n1*ap*_J1*b**_J1m1))
    return _RW*T*tau*g_t, _RW*(tau*g_t-g)

def _if97_r2(P,T):
    """Region 2: returns (h, s) in J/kg and J/(kg K)."""
    pi=P/1.0e6; tau=540.0/T
    g0  =_log(pi)+float(np.sum(_n20*tau**_J20))
    g0_t=float(np.sum(_n20*_J20*tau**_J20m1))
    c=tau-0.5
    pp=pi**_I2; cp_=c**_J2
    gr  =float(np.sum(_n2*pp*cp_))
    gr_t=float(np.sum(_n2*pp*_J2*c**_J2m1))
    return _RW*T*tau*(g0_t+gr_t), _RW*(tau*(g0_t+gr_t)-(g0+gr))

def _if97_cp_r1(P,T):
    """Region 1 isobaric heat capacity [J/(kg K)]."""
    pi=P/16.53e6; tau=1386.0/T
    a=7.1-pi; b=tau-1.222
    ap=a**_I1
    g_tt=float(np.sum(_n1*ap*_J1*(_J1-1)*b**(_J1-2)))
    return -_RW*tau*tau*g_tt

def _r1_hs(P,T):
    """IF97 Region 1: (h, s) in J/kg and J/(kg K), returned together."""
    pi=P/16.53e6; tau=1386.0/T
    a=7.1-pi; b=tau-1.222
    ap=a**_I1; bp=b**_J1
    g  =float(np.sum(_n1*ap*bp))
    g_t=float(np.sum(_n1*ap*_J1*b**_J1m1))
    return _RW*T*tau*g_t, _RW*(tau*g_t-g)

def h_pump_isentropic(P_in,P_out,eta,T_in=None):
    """Actual pump outlet enthalpy [J/kg] from an exact IF97 Region 1
    isentropic inversion, followed by the isentropic efficiency.

    Reference route for the pump-work terms.  Rather than the incompressible
    approximation h_out = h_in + dP/(rho eta), the isentropic outlet enthalpy
    is obtained exactly: with the inlet a saturated liquid at P_in (so
    s_in = s_f(P_in) from IF97 Region 1), the outlet temperature on the
    P_out isobar is found from s(P_out,T_out) = s_in by Newton iteration
    using (d s / d T)_P = c_p / T, and h_is = h(P_out,T_out).  The actual
    rise is (h_is - h_in)/eta.  Every property call is IF97 Region 1, so the
    result carries no correlation and no constant-specific-volume assumption.

    Both plant pumps draw saturated liquid (condenser hot-well, deaerator
    outlet), so T_in defaults to T_sat(P_in).  The inversion converges in
    <= 4 iterations and the outlet remains compressed liquid throughout the
    cycle's operating range."""
    if T_in is None: T_in=T_sat(P_in)
    h_in,s_in=_r1_hs(P_in,T_in)
    T=T_in
    for _ in range(40):
        h_o,s_o=_r1_hs(P_out,T); ds=s_o-s_in
        if abs(ds)<1e-11: break
        cp=_if97_cp_r1(P_out,T)              # (ds/dT)_P = cp/T
        T-=ds/(cp/T if cp>1e-9 else 1e-9)
    h_is,_=_r1_hs(P_out,T)
    return h_in+(h_is-h_in)/eta

def _if97_cp_r2(P,T):
    """Region 2 isobaric heat capacity [J/(kg K)]."""
    pi=P/1.0e6; tau=540.0/T
    g0_tt=float(np.sum(_n20*_J20*(_J20-1)*tau**(_J20-2)))
    c=tau-0.5; pp=pi**_I2
    gr_tt=float(np.sum(_n2*pp*_J2*(_J2-1)*c**(_J2-2)))
    return -_RW*tau*tau*(g0_tt+gr_tt)


@lru_cache(maxsize=8192)
def T_sat(P):
    P_MPa = (P if P>611.657 else 611.657)/1e6
    if P_MPa >= 22.064: return _Tc
    n=_nsat; nu=P_MPa**0.25
    E=nu*nu+n[2]*nu+n[5]; F_=n[0]*nu*nu+n[3]*nu+n[6]
    G=n[1]*nu*nu+n[4]*nu+n[7]
    disc=F_*F_-4*E*G
    D=2*G/(-F_-_sqrt(disc if disc>0. else 0.))
    q=(n[9]+D)*(n[9]+D)-4*(n[8]+n[9]*D)
    r=(n[9]+D-_sqrt(q if q>0. else 0.))/2
    return _clip(r,274.,_Tc)

def P_sat(T):
    T=_clip(T,274.,_Tc)
    return _clip(_Psat_of_lnT(_log(T)),611.,_Pc)

@lru_cache(maxsize=8192)
def h_f(P):
    """Saturated-liquid enthalpy from IF97 Region 1 evaluated at T_sat(P)."""
    P=P if P>611.657 else 611.657
    if P>=_Pc: P=_Pc*0.99999
    return _if97_r1(P,T_sat(P))[0]

@lru_cache(maxsize=8192)
def h_g(P):
    """Saturated-vapour enthalpy from IF97 Region 2 evaluated at T_sat(P)."""
    P=P if P>611.657 else 611.657
    if P>=_Pc: P=_Pc*0.99999
    v=_if97_r2(P,T_sat(P))[0]; f=h_f(P)+200.
    return v if v>f else f

@lru_cache(maxsize=8192)
def s_f(P):
    P=P if P>611.657 else 611.657
    if P>=_Pc: P=_Pc*0.99999
    return _if97_r1(P,T_sat(P))[1]

@lru_cache(maxsize=8192)
def s_g(P):
    P=P if P>611.657 else 611.657
    if P>=_Pc: P=_Pc*0.99999
    v=_if97_r2(P,T_sat(P))[1]; f=s_f(P)+50.
    return v if v>f else f

@lru_cache(maxsize=8192)
def cp_liq(P):
    """Saturated-liquid isobaric heat capacity from IF97 Region 1."""
    P=P if P>611.657 else 611.657
    if P>=_Pc: P=_Pc*0.99999
    return _if97_cp_r1(P,T_sat(P))

def _rho_liq(P,T):
    Tc=(T if T>274. else 274.)-273.15
    dP=P-1e5
    return _clip(1000.12-2.5e-3*Tc**1.5+4.6e-3*((dP if dP>0. else 0.)/1e7)*1e2,600.,1010.)

# =============================================================================
#  UPDATE
#  All single-phase states are now
#  exact IF97 Region 1 / Region 2; only the two-phase quality rule remains.
#  Verified: T(P,h), s(P,h) and h_is agree with iapws 1.5.5 to <1e-12 K /
#  <1e-12 relative at all 15 converged nodes.
# =============================================================================
# ---------------------------------------------------------------------------
#  Exact single-phase (P,h) <-> (P,T) inversions on IF97 Regions 1 and 2.
#  Replaces the constant-cp excursions (_cpv table / hf-inversion) that were
#  used for T(P,h), s(P,h), h(P,T) and the superheated / subcooled branches of
#  the isentropic expansion.  Only the Region-4 two-phase mixing rule remains.
# ---------------------------------------------------------------------------
_TMAX_R2 = 1073.15        # upper validity limit of Region 2 [K]
_TMIN    = 273.16

def _r2_hscp(P,T):
    """IF97 Region 2: (h, s, cp) evaluated with one set of powers."""
    pi=P/1.0e6; tau=540.0/T
    t0=tau**_J20
    g0   =_log(pi)+float(np.sum(_n20*t0))
    g0_t =float(np.sum(_n20*_J20*t0/tau))
    g0_tt=float(np.sum(_n20*_J20*(_J20-1)*t0/(tau*tau)))
    c=tau-0.5; pp=pi**_I2; cj=c**_J2
    gr   =float(np.sum(_n2*pp*cj))
    gr_t =float(np.sum(_n2*pp*_J2*cj/c))
    gr_tt=float(np.sum(_n2*pp*_J2*(_J2-1)*cj/(c*c)))
    gt=g0_t+gr_t
    return _RW*T*tau*gt, _RW*(tau*gt-(g0+gr)), -_RW*tau*tau*(g0_tt+gr_tt)

def _r1_hscp(P,T):
    """IF97 Region 1: (h, s, cp) evaluated with one set of powers."""
    pi=P/16.53e6; tau=1386.0/T; a=7.1-pi; b=tau-1.222
    ap=a**_I1; bj=b**_J1
    g   =float(np.sum(_n1*ap*bj))
    g_t =float(np.sum(_n1*ap*_J1*bj/b))
    g_tt=float(np.sum(_n1*ap*_J1*(_J1-1)*bj/(b*b)))
    return _RW*T*tau*g_t, _RW*(tau*g_t-g), -_RW*tau*tau*g_tt

@lru_cache(maxsize=4096)
def _bounds(fun,P,k,lo,hi):
    flo=fun(P,lo); fhi=fun(P,hi)
    return (flo[k], flo[2] if k==0 else flo[2]/lo,
            fhi[k], fhi[2] if k==0 else fhi[2]/hi)

def _invert(fun,P,target,k,T0,lo,hi):
    """Newton for f_k(P,T)=target on [lo,hi]; k=0 -> h (slope cp),
    k=1 -> s (slope cp/T).  Outside [lo,hi] the property is continued
    linearly with the boundary slope, so the map stays C^1 for far iterates."""
    flo_k,slo,fhi_k,shi=_bounds(fun,P,k,lo,hi)
    if target<=flo_k: return lo+(target-flo_k)/slo
    if target>=fhi_k: return hi+(target-fhi_k)/shi
    T=_clip(T0,lo,hi)
    for _ in range(30):
        f=fun(P,T); d=(f[k]-target)/(f[2] if k==0 else f[2]/T)
        Tn=_clip(T-d,lo,hi)
        if abs(Tn-T)<1e-11*T: T=Tn; break
        T=Tn
    return T

@lru_cache(maxsize=200000)
def T_Ph(P,h):
    """Exact T(P,h): Region 1 below hf, T_sat in the dome, Region 2 above hg."""
    if P<1e3: P=1e3
    Ts=T_sat(P); hf_=h_f(P); hg_=h_g(P)
    if h<=hf_:
        return _invert(_r1_hscp,P,h,0,Ts-(hf_-h)/4200.,_TMIN,Ts)
    if h>=hg_:
        return _invert(_r2_hscp,P,h,0,Ts+(h-hg_)/2500.,Ts,_TMAX_R2)
    return Ts

@lru_cache(maxsize=200000)
def s_Ph(P,h):
    """Exact s(P,h): single-phase via T_Ph, two-phase by the quality rule."""
    if P<1e3: P=1e3
    hf_=h_f(P); hg_=h_g(P)
    if h<=hf_: return _r1_hscp(P,T_Ph(P,h))[1]
    if h>=hg_: return _r2_hscp(P,T_Ph(P,h))[1]
    sf_=s_f(P); sg_=s_g(P)
    return sf_+(h-hf_)/(hg_-hf_)*(sg_-sf_)

def h_PT(P,T):
    if P<1e3: P=1e3
    Ts=T_sat(P)
    return _r1_hscp(P,min(T,Ts))[0] if T<=Ts else _r2_hscp(P,min(T,_TMAX_R2))[0]+ \
           (max(T-_TMAX_R2,0.)*_r2_hscp(P,_TMAX_R2)[2])

@lru_cache(maxsize=200000)
def h_is_expansion(h_in,P_in,P_out):
    """Exact isentropic outlet enthalpy at P_out for inlet (P_in,h_in)."""
    if P_in<1e3: P_in=1e3
    if P_out<1e3: P_out=1e3
    s_in=s_Ph(P_in,h_in)
    sf_=s_f(P_out); sg_=s_g(P_out); Ts=T_sat(P_out)
    if s_in>=sg_:
        T=_invert(_r2_hscp,P_out,s_in,1,Ts*_exp((s_in-sg_)/2500.),Ts,_TMAX_R2)
        f=_r2_hscp(P_out,min(T,_TMAX_R2)); return f[0]+max(T-_TMAX_R2,0.)*f[2]
    if s_in<=sf_:
        T=_invert(_r1_hscp,P_out,s_in,1,Ts,_TMIN,Ts)
        f=_r1_hscp(P_out,max(T,_TMIN)); return f[0]+min(T-_TMIN,0.)*f[2]
    hf_=h_f(P_out); hg_=h_g(P_out)
    return hf_+(s_in-sf_)/(sg_-sf_)*(hg_-hf_)

def safe_T(P,h):
    try:    return max(274.,T_Ph(P,h))
    except Exception: return 400.

# -----------------------------------------------------------------------------

def _smax(v, vmin, eps=2.0):
    """Smooth lower bound: differentiable everywhere, -> max(v,vmin) as eps->0.

    A hard clamp (max) has zero derivative once saturated.  In the 3-zone SG
    equations both LMTD temperature differences saturate simultaneously at
    far-from-solution states, which zeroes whole Jacobian rows and makes the
    SG field block EXACTLY singular -- GMRES then cannot converge and every
    strong preconditioner silently degrades to steepest descent.  The smooth
    floor keeps the guard while preserving a non-vanishing derivative."""
    d=v-vmin
    return vmin+0.5*(d+_sqrt(d*d+eps*eps))

def _lmtd(dT_a,dT_b):
    """Log-mean temperature difference with smooth positivity guards and a
    smooth transition to the arithmetic mean as dT_a -> dT_b."""
    dT_a=_smax(dT_a,0.5); dT_b=_smax(dT_b,0.5)
    r=dT_a/dT_b
    if abs(r-1.0)<1e-3:                      # series expansion (removable sing.)
        u=r-1.0
        return dT_b*(1.0+0.5*u-u*u/12.0)
    return (dT_a-dT_b)/_log(r)


# =============================================================================
#  SECTION B.  HTR-PM PLANT DESIGN DATA  (2 x 250 MWth, single turbine train)
# =============================================================================
m_He     = 192.0                       # He mass flow, both modules [kg/s]
Cp_He    = 5193.0                       # He specific heat [J/(kg K)]
T_He_in  = 750.0 + 273.15              # SG He inlet temperature [K]
T_He_out = 250.0 + 273.15             # SG He outlet temperature [K] (design)
Q_SG_BC  = m_He*Cp_He*(T_He_in-T_He_out)   # ~ 498.5 MW

M_TOTAL  = 186.0                        # main steam mass flow [kg/s]
P_SG     = 13.9e6                       # main steam pressure [Pa]
T_SG_OUT = 567.0 + 273.15              # design SG outlet steam temperature [K]

# Design pressures [Pa]
P_HP1  = 7.5e6
P_HP2  = 3.5e6
P_LP1  = 1.8e6
P_LP2  = 0.9e6
P_LP3  = 0.4e6
P_COND = 4500.0
P_DA   = 0.72e6                         # deaerator (self-consistent, hf~702 kJ/kg)
P_CP   = 0.1109*P_SG                    # condensate-pump discharge (Energies paper)

# Turbine / pump efficiencies
# Turbine isentropic efficiencies. These are the calibration parameters of
# the model (the steam properties are not). They were 0.88 / 0.82 while the
# property layer used spline saturation data; with the exact IF97 formulation
# the isentropic enthalpy drops are smaller, and the old pair under-predicts
# the plant output by ~6 %.  Recalibrated to typical modern values, giving
# 209.9 MW gross / 206.4 MW net / 41.4 % against the reference 211.7 / 209.0 /
# 41.9 %.  Setting 0.94 / 0.89 reproduces the reference output exactly but is
# optimistic for real machines, particularly for a wet LP expansion.
eta_HP = 0.92                           # HP turbine isentropic efficiency
eta_LP = 0.88                           # LP turbine isentropic efficiency
eta_cp = 0.80
eta_fw = 0.82

# Extraction fractions -- tuned to the HTR-PM "typical values" of Jiang Table 2
#  (LP exhaust  m/m0 ~ 0.671, condensate-pump m/m0 ~ 0.871, FW pump m/m0 = 1.0)
f_hph = 0.110                           # HP1 extraction  -> HPH shell
f_da  = 0.022                           # DA steam tap (LP1 level)
f_lp1 = 0.090                           # LPH-1 extraction (of post-tap LP flow)
f_lp2 = 0.083
f_lp3 = 0.075

# Once-through helical SG geometry
L_SG   = 80.0                           # effective heat-transfer tube length [m]

# Node metadata
N_NODES = 15
NODE_LABELS = [
    "SG outlet","HP1 extraction","HP2 outlet","LP1 extraction","LP2 extraction",
    "LP3 extraction","LP inlet","LP exhaust","Cond pump out","LPH-3 tube out",
    "LPH-2 tube out","LPH-1 tube out","Deaerator out","FW pump out","HPH out/SG in"]
PAPER_IDS = list(range(1,16))

# DOF layout: 45 node DOF + 5 SG DOF
N_SG   = 5
N_DOF  = N_NODES*3 + N_SG               # 50
def im(i): return i
def ip(i): return N_NODES+i
def ih(i): return 2*N_NODES+i
I_LPRE, I_LEVA, I_THE1, I_THE2, I_THE3 = 45,46,47,48,49


# =============================================================================
#  SECTION C.  SELF-CONSISTENT SG CALIBRATION + PHYSICS-CONSISTENT INITIAL GUESS
# =============================================================================
def _turb(h_in,P_in,P_out,eta):
    return h_in - eta*(h_in - h_is_expansion(h_in,P_in,P_out))

def _design_enthalpies(h0=None):
    """Analytic construction of the 15 node enthalpies at the design point,
    in physical flow order (identical logic to build_x0's h-field).
    If h0 is None, the SG outlet enthalpy is taken as h_PT(P_SG,T_SG_OUT);
    otherwise the supplied h0 is used (for self-consistent global closure)."""
    h = np.zeros(N_NODES)
    m0=M_TOTAL; m1=m0*f_hph; m2=m0*(1-f_hph); m6=m2
    m_post=m6*(1-f_da)
    m3=m_post*f_lp1; m4=(m_post-m3)*f_lp2; m5=(m_post-m3-m4)*f_lp3
    m7=m_post-m3-m4-m5; m8=m7+m3+m4+m5
    h[0]=h_PT(P_SG,T_SG_OUT) if h0 is None else float(h0)
    h[1]=_turb(h[0],P_SG ,P_HP1,eta_HP)
    h[2]=_turb(h[1],P_HP1,P_HP2,eta_HP)
    h[6]=h[2]
    h[3]=_turb(h[6],P_HP2,P_LP1,eta_LP)
    h[4]=_turb(h[3],P_LP1,P_LP2,eta_LP)
    h[5]=_turb(h[4],P_LP2,P_LP3,eta_LP)
    h[7]=_turb(h[5],P_LP3,P_COND,eta_LP)
    h[7]=float(np.clip(h[7],h_g(P_COND)*0.80,h_g(P_COND)*1.02))
    h[8]=h_pump_isentropic(P_COND,P_CP,eta_cp)   # exact IF97 isentropic
    h[9] =h[8]+m5*max(h[5]-h_f(P_LP3),0.)/max(m8,1.)
    h[10]=h[9]+m4*max(h[4]-h_f(P_LP2),0.)/max(m8,1.)
    # DA balance -> h11  (v9: LP expansion state at P_DA from the LP inlet)
    h_da=h[2] #_turb(h[6],P_HP2,P_DA,eta_LP)
    m_da=m6*f_da; m12=m8+m_da+m1
    h[11]=float(np.clip((m12*h_f(P_DA)-m_da*h_da-m1*h_f(P_HP1))/max(m8,1.),
                        h[10]+1e3,h_f(P_DA)-500.))
    h[12]=h_f(P_DA)
    h[13]=h_pump_isentropic(P_DA,P_SG*1.04,eta_fw) # exact IF97 isentropic
    h[14]=h[13]+m1*max(h[1]-h_f(P_HP1),0.)/max(m12,1.)
    return h

def _design_enthalpies_closed():
    """Self-consistent design enthalpies satisfying the SG global energy balance
    m0*(h0-h14)=Q_SG_BC (fixed-point on h0).  Used ONLY for SG calibration so
    that the converged state is an exact fixed point of the SG block."""
    h0=h_PT(P_SG,T_SG_OUT)
    for _ in range(60):
        h=_design_enthalpies(h0=h0)
        h0_new=h[14]+Q_SG_BC/M_TOTAL
        if abs(h0_new-h0)<1.0: h0=h0_new; break
        h0=0.5*h0+0.5*h0_new
    return _design_enthalpies(h0=h0)

def _calibrate_SG():
    """Return (UA_pre, UA_eva, UA_sup, l_pre0, l_eva0, T1_0, T2_0, T3_0) such
    that the design point is an *exact* fixed point of the SG block F[45..49].
    UA/length coefficients are back-computed from the design duties, LMTDs and
    a physically motivated length split (boiling zone short, high U)."""
    h=_design_enthalpies_closed()
    hf_=h_f(P_SG); hg_=h_g(P_SG); Ts=T_sat(P_SG)
    T_win=safe_T(P_SG,h[14]); T_wout=safe_T(P_SG,h[0])
    Q_pre=M_TOTAL*(hf_-h[14]); Q_eva=M_TOTAL*(hg_-hf_); Q_sup=M_TOTAL*(h[0]-hg_)
    mCp=m_He*Cp_He
    T2=T_He_in-Q_sup/mCp                    # He at SUP/EVA boundary
    T1=T2-Q_eva/mCp                         # He at EVA/PRE boundary
    T3=T1-Q_pre/mCp                         # He outlet (design ~250 C)
    LM_pre=_lmtd(T1-Ts,   T3-T_win)
    LM_eva=_lmtd(T2-Ts,   T1-Ts)
    LM_sup=_lmtd(T_He_in-T_wout, T2-Ts)
    l_pre,l_eva,l_sup = 22.0,16.0,L_SG-38.0     # design length split (sum=80)
    UA_pre=Q_pre/(l_pre*LM_pre)
    UA_eva=Q_eva/(l_eva*LM_eva)
    UA_sup=Q_sup/(l_sup*LM_sup)
    return UA_pre,UA_eva,UA_sup,l_pre,l_eva,T1,T2,T3

UA_pre,UA_eva,UA_sup,_lpre0,_leva0,_T1_0,_T2_0,_T3_0 = _calibrate_SG()

def build_x0():
    """Physics-consistent initial guess: all 30 mass+pressure equations and the
    5 SG equations are satisfied exactly; the residual is carried entirely by
    the coupled energy chain."""
    x=np.zeros(N_DOF)
    m=x[:N_NODES]; p=x[N_NODES:2*N_NODES]; h=x[2*N_NODES:3*N_NODES]
    # mass flows
    m0=M_TOTAL; m1=m0*f_hph; m2=m0*(1-f_hph)
    m_da=f_da*m2; m6=m2-m_da                 # DA tapped at crossover
    m3=m6*f_lp1; m4=(m6-m3)*f_lp2; m5=(m6-m3-m4)*f_lp3
    m7=m6-m3-m4-m5; m8=m7+m3+m4+m5
    m12=m8+m_da+m1
    m[:]=[m0,m1,m2,m3,m4,m5,m6,m7,m8,m8,m8,m8,m12,m12,m12]
    # pressures
    p[:]=[P_SG,P_HP1,P_HP2,P_LP1,P_LP2,P_LP3,P_HP2,P_COND,
          P_CP,P_CP,P_CP,P_CP,P_DA,P_SG*1.04,P_SG]
    # enthalpies
    h[:]=_design_enthalpies()
    # SG block
    x[I_LPRE]=_lpre0; x[I_LEVA]=_leva0
    x[I_THE1]=_T1_0;  x[I_THE2]=_T2_0; x[I_THE3]=_T3_0
    return x


# =============================================================================
#  SECTION D.  RESIDUAL  F(x) = 0   (50 equations)
# =============================================================================
#   F[ 0..14]  §M  mass balances
#   F[15..29]  §P  pressure boundary/continuity conditions
#   F[30..44]  §H  energy / enthalpy relations
#   F[45..49]  §SG 3-zone once-through steam-generator block
#
#  Structural regularity: every state variable is the sole subject of exactly
#  one equation, so the Jacobian diagonal is non-zero at the solution -- a
#  necessary condition for effective physics-based preconditioning.
# =============================================================================
def residual(x):
    x  = np.abs(x)
    F  = np.zeros(N_DOF)
    mv = x[:N_NODES]; pv = x[N_NODES:2*N_NODES]; hv = x[2*N_NODES:3*N_NODES]
    def m(i): return mv[i]
    def p(i): return pv[i]
    def h(i): return hv[i]
    def hf(i): return h_f(max(pv[i],1e3))
    l_pre = max(x[I_LPRE],0.5)
    l_eva = max(x[I_LEVA],0.5)
    l_sup = max(L_SG-l_pre-l_eva,0.5)
    # Smooth guards (see _smax): a hard max() here freezes the helium
    # temperatures into constants whenever an iterate drifts below the bound,
    # which zeroes columns 47-49 and makes the SG block exactly singular.
    T1 = _smax(x[I_THE1],T_He_out-40.,1.0)  # He at EVA/PRE boundary
    T2 = _smax(x[I_THE2],T_He_out-40.,1.0)  # He at SUP/EVA boundary
    T3 = _smax(x[I_THE3],T_He_out-40.,1.0)  # He outlet (cold end)
    mCp = m_He*Cp_He

    # ------- §M  MASS BALANCES  F[0..14] -------------------------------------
    F[0]  = m(0)-M_TOTAL                                 # main steam BC
    F[1]  = m(1)-f_hph*m(0)                              # HP1 extraction
    F[2]  = m(2)-(1.-f_hph)*m(0)                         # HP2 outlet
    # Deaerator steam is tapped at the HP/LP crossover, UPSTREAM of the LP
    # turbine, so node 7 carries the post-tap flow that actually expands.
    # (Folding the tap into the extraction factors instead, as v6/v7 did,
    #  leaves node 7 at the pre-tap flow and overstates LP turbine work.)
    m_da  = f_da*m(2)                                    # DA steam tap at crossover
    F[6]  = m(6)-(1.-f_da)*m(2)                          # LP inlet (post-tap)
    F[3]  = m(3)-m(6)*f_lp1                              # LPH-1 extraction
    F[4]  = m(4)-(m(6)-m(3))*f_lp2                       # LPH-2 extraction
    F[5]  = m(5)-(m(6)-m(3)-m(4))*f_lp3                  # LPH-3 extraction
    F[7]  = m(7)-(m(6)-m(3)-m(4)-m(5))                   # LP exhaust
    F[8]  = m(8)-(m(7)+m(3)+m(4)+m(5))                   # CP = exhaust+LP drains
    F[9]  = m(9) -m(8)
    F[10] = m(10)-m(9)
    F[11] = m(11)-m(10)
    F[12] = m(12)-(m(11)+m_da+m(1))                      # DA open mixing
    F[13] = m(13)-m(12)
    F[14] = m(14)-m(13)

    # ------- §P  PRESSURE CONDITIONS  F[15..29] -------------------------------
    F[15] = p(0)-P_SG
    F[16] = p(1)-P_HP1
    F[17] = p(2)-P_HP2
    F[18] = p(3)-P_LP1
    F[19] = p(4)-P_LP2
    F[20] = p(5)-P_LP3
    F[21] = p(6)-p(2)                                    # HP->LP junction
    F[22] = p(7)-P_COND
    F[23] = p(8)-P_CP
    F[24] = p(9) -p(8)
    F[25] = p(10)-p(9)
    F[26] = p(11)-p(10)
    F[27] = p(12)-P_DA
    F[28] = p(13)-P_SG*1.04
    F[29] = p(14)-P_SG

    # ------- §H  ENERGY / ENTHALPY  F[30..44] ---------------------------------
    # H0: SG global energy balance (integral cycle constraint) -> pins h0
    F[30] = m(0)*(h(0)-h(14))-Q_SG_BC
    # H1-H2, H6, H3-H5, H7: turbine chain (entropy-based expansion)
    F[31] = h(1)-(h(0)-eta_HP*(h(0)-h_is_expansion(h(0),p(0),p(1))))
    F[32] = h(2)-(h(1)-eta_HP*(h(1)-h_is_expansion(h(1),p(1),p(2))))
    F[36] = h(6)-h(2)
    F[33] = h(3)-(h(6)-eta_LP*(h(6)-h_is_expansion(h(6),p(6),p(3))))
    F[34] = h(4)-(h(3)-eta_LP*(h(3)-h_is_expansion(h(3),p(3),p(4))))
    F[35] = h(5)-(h(4)-eta_LP*(h(4)-h_is_expansion(h(4),p(4),p(5))))
    F[37] = h(7)-(h(5)-eta_LP*(h(5)-h_is_expansion(h(5),p(5),p(7))))
    # H8: condensate pump  (exact IF97 Region 1 isentropic inversion)
    F[38] = h(8)-h_pump_isentropic(p(7),p(8),eta_cp)
    # H9-H10: LP heaters (tube rise = shell condensation duty)
    F[39] = m(9) *(h(9) -h(8)) - m(5)*max(h(5)-h_f(P_LP3),0.)
    F[40] = m(10)*(h(10)-h(9)) - m(4)*max(h(4)-h_f(P_LP2),0.)
    # H11: deaerator energy balance -> constrains h11 (LPH-1 tube outlet)
    # Deaerator steam enthalpy
    h_da = h(2)
    E_ex=m_da*h_da+m(1)*hf(1)
    F[41] = m(11)*h(11)-(m(12)*h_f(P_DA)-E_ex)
    # H12: DA outlet = saturated liquid (structural regularity split)
    F[42] = h(12)-h_f(P_DA)
    # H13: FW pump  (exact IF97 Region 1 isentropic inversion)
    F[43] = h(13)-h_pump_isentropic(p(12),p(13),eta_fw)
    # H14: HP heater tube-side energy balance
    F[44] = m(14)*(h(14)-h(13)) - m(1)*max(h(1)-hf(1),0.)

    # ------- §SG  3-ZONE ONCE-THROUGH STEAM GENERATOR  F[45..49] --------------
    hf_sg=h_f(p(0)); hg_sg=h_g(p(0)); Ts_sg=T_sat(p(0))
    T_win =safe_T(p(14),h(14)); T_wout=safe_T(p(0),h(0))
    Q_pre=m(0)*(hf_sg-h(14)); Q_eva=m(0)*(hg_sg-hf_sg)
    # Equations are aligned with variable indices for structural regularity:
    #   index 45=l_pre, 46=l_eva, 47=T1, 48=T2, 49=T3
    # SG45: PRE-zone UA.LMTD heat-transfer closure               -> l_pre (45)
    F[45] = UA_pre*l_pre*_lmtd(T1-Ts_sg, T3-T_win) - Q_pre
    # SG46: EVA-zone UA.LMTD heat-transfer closure               -> l_eva (46)
    F[46] = UA_eva*l_eva*_lmtd(T2-Ts_sg, T1-Ts_sg) - Q_eva
    # SG47: PRE-zone He energy balance                           -> T1    (47)
    F[47] = mCp*(T1-T3) - Q_pre
    # SG48: EVA-zone He energy balance                           -> T2    (48)
    F[48] = mCp*(T2-T1) - Q_eva
    # SG49: He outlet boundary condition (cold end)              -> T3    (49)
    F[49] = T3 - T_He_out
    return F


# =============================================================================
#  SECTION E.  RESIDUAL SCALING  d  (renders each equation O(1))
# =============================================================================
DS = np.ones(N_DOF)
DS[0:15]  = 1e2      # mass [kg/s]
DS[15:30] = 1e6      # pressure [Pa]
DS[30]    = 1e7      # SG global energy [W]
DS[31:38] = 2e5      # turbine enthalpy [J/kg]
DS[38]    = 5e4      # cond pump enthalpy [J/kg]
DS[39:41] = 5e6      # LP heater energy [W]
DS[41]    = 5e6      # DA energy [W]
DS[42]    = 1e5      # DA outlet enthalpy BC [J/kg]
DS[43]    = 5e4      # FW pump enthalpy [J/kg]
DS[44]    = 5e6      # HPH energy [W]
DS[45]    = 2e7      # PRE UA.LMTD [W]   (subject l_pre)
DS[46]    = 2e7      # EVA UA.LMTD [W]   (subject l_eva)
DS[47]    = 2e7      # PRE He energy [W] (subject T1)
DS[48]    = 2e7      # EVA He energy [W] (subject T2)
DS[49]    = 5.0      # He outlet BC [K]  (subject T3)


# =============================================================================
#  SECTION F.  PHYSICS-BASED PRECONDITIONERS
# =============================================================================
#  Physical field partition used by FieldSplit / block-Jacobi:
FIELD_BLOCKS = [
    ("§M mass",     list(range(0,15))),
    ("§P pressure", list(range(15,30))),
    ("§H enthalpy", list(range(30,45))),
    ("§SG gen.",    list(range(45,50))),
]

def build_scaled_jacobian(x, eps_fd=1e-7):
    """Dense scaled Jacobian  Ĵ = diag(d)^{-1} J diag(d)  by forward FD.
    Used for the block preconditioners and the spectral analysis."""
    n=N_DOF; F0=residual(x); J=np.zeros((n,n))
    for j in range(n):
        dj=eps_fd*max(abs(x[j]),1.0)
        xp=x.copy(); xp[j]+=dj
        J[:,j]=(residual(xp)-F0)/dj
    # scaled: Ĵ_ij = J_ij * d_j / d_i
    return (J*DS[None,:])/DS[:,None]


def _reg_inv(A, rcond=1e-8):
    """Truncated-SVD (Tikhonov-type) inverse of a Jacobian sub-block.
    Far from the solution individual field blocks can be near-singular (the
    3-zone SG block reaches kappa ~ 1e7); an exact inverse then amplifies the
    cross-field coupling that block-Jacobi discards into a nonphysical
    excursion.  Filtering the smallest singular values bounds ||M^{-1}|| and
    restores global robustness while leaving well-conditioned directions
    untouched."""
    try:
        U,s,Vt=np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)
    smax=s[0] if s.size and s[0]>0 else 1.0
    sinv=np.where(s>rcond*smax, 1.0/np.maximum(s,1e-300), 0.0)
    return (Vt.T*sinv)@U.T


def project_physical(x):
    """Projection onto the physically admissible set (projected inexact Newton).

    The 3-zone steam-generator lengths are geometrically constrained by the
    tube length L_SG; left unbounded, a single aggressive Newton step can send
    l_pre + l_eva far beyond L_SG, making l_sup negative and the LMTD terms
    meaningless -- a state from which no preconditioner can recover.  Enforcing
    the feasible box after every update keeps every iterate a thermodynamically
    admissible plant state and is what makes large FieldSplit/ASPIN steps safe.
    """
    x=np.abs(x)
    # --- state bounds -----------------------------------------------------
    x[0:15]   = np.clip(x[0:15],   1e-3,  1.0e4)      # mass flows   [kg/s]
    x[15:30]  = np.clip(x[15:30],  1.0e3, 3.0e7)      # pressures    [Pa]
    x[30:45]  = np.clip(x[30:45],  1.0e3, 4.5e6)      # enthalpies   [J/kg]
    # --- SG zone lengths: l_pre, l_eva > 0 and l_pre + l_eva < L_SG -------
    # Every zone must retain a physically meaningful share of the tube length.
    # Allowing l_sup -> 0 (a 1.6 m superheater carrying ~40 % of the duty) gives
    # a degenerate configuration: the iterate pins against the simplex face, the
    # line search collapses, and even the linear pressure BCs stop converging.
    lmin=0.05*L_SG
    lp=_clip(float(x[I_LPRE]),lmin,L_SG-2*lmin)
    le=_clip(float(x[I_LEVA]),lmin,L_SG-lmin-lp)      # => l_sup >= lmin
    x[I_LPRE]=lp; x[I_LEVA]=le
    # --- helium temperatures ---------------------------------------------
    # The primary loop is bounded by its own boundary conditions: helium can
    # never be colder than its core inlet nor hotter than its core outlet.
    # Restricting T1,T2,T3 to that band (rather than merely to T>0) keeps the
    # SG driving temperature differences O(10 K), where the LMTD guards are
    # inactive and the SG Jacobian block stays non-singular.
    Tin=T_He_in; Tlo=T_He_out-5.0
    T2=_clip(float(x[I_THE2]),Tlo,Tin)
    T1=_clip(float(x[I_THE1]),Tlo,T2)
    T3=_clip(float(x[I_THE3]),Tlo,T1)
    x[I_THE2]=T2; x[I_THE1]=T1; x[I_THE3]=T3
    return x


class Preconditioner(ABC):
    name = "base"
    @abstractmethod
    def apply_linear(self, r): ...
    def apply_nonlinear(self, x, normF=None): return x.copy()
    def update_reference(self, x): pass


class NonePreconditioner(Preconditioner):
    """Identity: unpreconditioned GMRES in scaled residual space (reference)."""
    name = "None"
    def apply_linear(self, r): return r.copy()


class JacobiPreconditioner(Preconditioner):
    """Diagonal (point-Jacobi) preconditioner: the residual is already scaled,
    so M^{-1} = I acting on the diagonal of the *scaled* system.  This is the
    baseline 'trivial' preconditioner equivalent to the Energies-2023 scaling."""
    name = "Jacobi"
    def __init__(self):
        self._dinv=None
    def update_reference(self, x):
        Js=build_scaled_jacobian(x)
        d=np.diag(Js).copy(); d[np.abs(d)<1e-12]=1.0
        self._dinv=1.0/d
    def apply_linear(self, r):
        return r*self._dinv if self._dinv is not None else r.copy()


class FieldSplitPreconditioner(Preconditioner):
    """True FieldSplit = block-Jacobi over the four physical fields
    {m, p, h, SG}.  Each field's sub-block of the scaled Jacobian is inverted
    exactly; cross-field coupling is dropped (approximate block factorisation
    M^{-1} = blockdiag(A_MM^{-1}, A_PP^{-1}, A_HH^{-1}, A_SG^{-1})).
    Rebuilt every `refresh` Newton steps."""
    name = "FieldSplit"
    def __init__(self, refresh=1, rcond=1e-8):
        self.refresh=refresh; self.rcond=rcond; self._Minv=None; self._k=0
    def update_reference(self, x):
        if self._Minv is None or (self._k % self.refresh==0):
            Js=build_scaled_jacobian(x)
            Minv=np.zeros((N_DOF,N_DOF))
            for _,idx in FIELD_BLOCKS:
                Minv[np.ix_(idx,idx)]=_reg_inv(Js[np.ix_(idx,idx)], self.rcond)
            self._Minv=Minv
        self._k+=1
    def apply_linear(self, r):
        return self._Minv@r if self._Minv is not None else r.copy()
    # expose block condition numbers for the spectral study
    def block_conds(self, x):
        Js=build_scaled_jacobian(x); out={}
        for nm,idx in FIELD_BLOCKS:
            out[nm]=np.linalg.cond(Js[np.ix_(idx,idx)])
        return out


# ASPIN subdomain blocks (variables ξ, equations ε): physical information flow
ASPIN_BLOCKS = [
    ("A HP turbine", [ih(1),ih(2)],                     [31,32]),
    ("B LP turbine", [ih(6),ih(3),ih(4),ih(5),ih(7)],   [36,33,34,35,37]),
    ("C LP heaters", [ih(9),ih(10),ih(11)],             [39,40,41]),
    ("D DA/FW/HPH",  [ih(12),ih(13),ih(14)],            [42,43,44]),
    ("E LP masses",  [im(3),im(4),im(5),im(6),im(7)],   [3,4,5,6,7]),
    ("F steam gen.", [I_LPRE,I_LEVA,I_THE1,I_THE2,I_THE3],[45,46,47,48,49]),
]

def local_block_solve(xg, xi, ei, nsub=3):
    """Damped Newton solve of one physics subdomain with its interface frozen.
    Used both by ASPIN (as the nonlinear sweep) and by the outer solver as a
    recovery step when the global line search collapses on a single block."""
    xl=xg[xi].copy()
    def Fl(xL):
        xt=xg.copy(); xt[xi]=np.abs(xL); return residual(project_physical(xt))[ei]/DS[ei]
    F0n=np.linalg.norm(Fl(xl))
    for _ in range(nsub):
        Fc=Fl(xl); Fn=np.linalg.norm(Fc)
        if Fn<1e-8*max(F0n,1.): break
        J=np.zeros((len(ei),len(xi)))
        for j in range(len(xi)):
            e=max(abs(xl[j])*1e-5,1.); xp=xl.copy(); xp[j]+=e
            J[:,j]=(Fl(xp)-Fc)/e
        d=_reg_inv(J,1e-8)@(-Fc)
        lm=1.0
        for _ in range(8):
            xt=np.abs(xl+lm*d)
            if np.linalg.norm(Fl(xt))<Fn: xl=xt; break
            lm*=0.5
        else: break
    xo=xg.copy(); xo[xi]=np.abs(xl); return project_physical(xo)


class ASPINPreconditioner(Preconditioner):
    """Additive-Schwarz Preconditioned Inexact Newton (ASPIN).
    Nonlinear left preconditioning: before each outer Krylov iteration, up to
    `nsub` local Newton steps are applied to each physics subdomain (interface
    frozen), resolving two-phase / cascade nonlinearity.  The inner GMRES uses
    the FieldSplit block-Jacobi linear preconditioner."""
    name = "ASPIN"
    def __init__(self, nsub=3, thresh=0.5, refresh=1, rcond=1e-8):
        self.nsub=nsub; self.thresh=thresh
        self._fs=FieldSplitPreconditioner(refresh=refresh, rcond=rcond)
    def apply_nonlinear(self, x, normF=None):
        """Safeguarded nonlinear sweep.

        The additive-Schwarz sweep is only *locally* consistent: each subdomain
        solve freezes its interface, so the composite map can possess fixed
        points that are not roots of the global system.  Applied unconditionally
        it then undoes the outer Newton progress and the iteration stalls at a
        constant residual.  Accepting the sweep only when it decreases the
        global residual removes those spurious attractors at the cost of one
        residual evaluation."""
        if normF is not None and normF<self.thresh: return x.copy()
        xn=x.copy()
        for _,xi,ei in ASPIN_BLOCKS: xn=self._local(xn,xi,ei)
        xn=project_physical(xn)
        if normF is None: return xn
        return xn if np.linalg.norm(residual(xn)/DS)<normF else x.copy()
    def _local(self, xg, xi, ei):
        return local_block_solve(xg, xi, ei, self.nsub)


    def update_reference(self, x): self._fs.update_reference(x)
    def apply_linear(self, r):     return self._fs.apply_linear(r)


# =============================================================================
#  SECTION G.  JACOBIAN-FREE NEWTON-KRYLOV SOLVER
# =============================================================================
class JFNKSolver:
    """Matrix-free inexact-Newton solver with Eisenstat-Walker forcing,
    Armijo backtracking, and a pluggable physics-based preconditioner.
    Records residual history, GMRES iterations/Newton step, and timings."""
    def __init__(self, pc, tol=1e-6, max_newton=40, eps_fd=1e-7,
                 use_aspin=False, verbose=False, tr_frac=1e9, kmin=0, eta_max=0.1):
        self.pc=pc; self.tol=tol; self.max_newton=max_newton
        self.eps_fd=eps_fd; self.use_aspin=use_aspin; self.verbose=verbose
        # tr_frac: max relative change of any state variable per Newton step
        #          (physical trust region -- keeps iterates thermodynamically sane)
        # kmin   : minimum Krylov iterations, so that a *strong* preconditioner
        #          cannot return a direction blind to cross-field coupling
        self.tr_frac=tr_frac; self.kmin=kmin; self.eta_max=eta_max
        self.residuals=[]; self.gmres_iters=[]; self.lambdas=[]
        self.converged=False; self.wall=0.0

    def solve(self, x0):
        x=project_physical(x0.copy()); self.residuals=[]; self.gmres_iters=[]; self.lambdas=[]
        t0=time.time()
        if self.verbose:
            print(f"\n  {'-'*74}")
            print(f"   {self.pc.name:<11} ASPIN={self.use_aspin}  tol={self.tol:.0e}  DOF={N_DOF}")
            print(f"   {'k':>3} {'||F||2':>12} {'||F/d||2':>12} {'||dx||':>11} {'lam':>7} {'GMRES':>6}")
            print(f"  {'-'*74}")
        stall=0; aspin_on=self.use_aspin
        for k in range(self.max_newton):
            # Adaptive deactivation of the nonlinear sweep. The additive-Schwarz
            # sweep can walk an iterate onto a constraint boundary (typically the
            # SG length simplex l_pre + l_eva -> L_SG) where the outer line search
            # collapses and even the linear pressure BCs stop being satisfied.
            # Two consecutive heavily-damped steps are taken as evidence that the
            # nonlinear preconditioner is hurting; it is switched off and the
            # solve proceeds as plain FieldSplit-preconditioned Newton. ASPIN
            # therefore doesn't perform worse than its own linear preconditioner.
            if aspin_on and stall>=2:
                aspin_on=False
                if self.verbose: print("   [ASPIN nonlinear sweep deactivated: stalled]")
            if aspin_on and hasattr(self.pc,"apply_nonlinear"):
                nprev=np.linalg.norm(residual(x)/DS)
                x=self.pc.apply_nonlinear(x, normF=nprev)
            Fx=residual(x); Fsc=Fx/DS
            nF=np.linalg.norm(Fx); nFsc=np.linalg.norm(Fsc)
            self.residuals.append(nF)
            if nFsc<self.tol:
                self.converged=True; self.wall=time.time()-t0
                if self.verbose: print(f"   ✓ converged k={k}  ||F/d||={nFsc:.3e}  ({self.wall:.2f}s)")
                return x, True
            self.pc.update_reference(x)
            eps=self.eps_fd; Fref=Fx.copy(); xn=np.linalg.norm(x)
            def Jv(vs):
                v=vs*DS; vnv=np.linalg.norm(v)
                if vnv<1e-15: return np.zeros(N_DOF)
                hh=eps*(1.+xn)/vnv
                return (residual(x+hh*v)-Fref)/hh/DS
            Jop=LinearOperator((N_DOF,N_DOF),matvec=Jv,dtype=float)
            Mop=LinearOperator((N_DOF,N_DOF),matvec=self.pc.apply_linear,dtype=float)
            its=[0]
            def cb(rk): its[0]+=1
            eta_k=min(self.eta_max,max(1e-3,0.9*nFsc**1.5))
            dxs,info=gmres(Jop,-Fsc,M=Mop,rtol=eta_k,atol=0.,
                           maxiter=N_DOF*4,restart=min(N_DOF,60),
                           callback=cb,callback_type="legacy")
            # Krylov-accuracy floor: force a few extra iterations so that a
            # strong preconditioner still incorporates cross-field information.
            if info==0 and its[0]<self.kmin and nFsc>1e-3:
                its2=[0]
                def cb2(rk): its2[0]+=1
                dx2,info2=gmres(Jop,-Fsc,M=Mop,rtol=1e-10,atol=0.,
                                maxiter=self.kmin,restart=min(N_DOF,60),
                                callback=cb2,callback_type="legacy")
                if np.all(np.isfinite(dx2)):
                    dxs=dx2; its[0]+=its2[0]
            self.gmres_iters.append(its[0])
            if info!=0:
                dxs=self.pc.apply_linear(-Fsc)   # preconditioned steepest descent
            dx=dxs*DS
            # physical trust region: no state variable may change by more than
            # tr_frac of its current magnitude in a single Newton step
            with np.errstate(divide="ignore",invalid="ignore"):
                rel=np.abs(dx)/np.maximum(np.abs(x),1e-12)
            rmax=np.nanmax(rel)
            if rmax>self.tr_frac: dx*=self.tr_frac/rmax
            lm=1.0
            for _ in range(30):
                xt=project_physical(x+lm*dx)
                if np.linalg.norm(residual(xt)/DS)<nFsc*(1.-1e-4*lm): break
                lm*=0.5
            else:
                # Steepest-descent fallback; if that also fails, take a small
                # forced step rather than returning x unchanged -- a zero step
                # is an absorbing state and stalls the solve permanently.
                dxg=self.pc.apply_linear(-Fsc)*DS
                xt=x; lm=0.0
                for lg in (1e-2,1e-3,1e-4):
                    xg=project_physical(x+lg*dxg)
                    if np.linalg.norm(residual(xg)/DS)<nFsc: xt=xg; lm=lg; break
                if lm==0.0:
                    # Block-targeted recovery: identify the physical field
                    # carrying the largest scaled residual and solve it alone
                    # with its interface frozen.  A stalled global line search
                    # is almost always one stiff block (typically the SG)
                    # holding back an otherwise well-behaved system.
                    blk=max(FIELD_BLOCKS,
                            key=lambda b: np.linalg.norm(Fsc[b[1]]))[1]
                    xr=local_block_solve(x,blk,blk,nsub=4)
                    if np.linalg.norm(residual(xr)/DS)<nFsc: xt=xr; lm=1.0
            x=project_physical(xt); self.lambdas.append(lm)
            stall = stall+1 if lm<0.05 else 0
            if self.verbose:
                tag=f"{its[0]:6d}" if info==0 else f"{its[0]:5d}!"
                print(f"   {k:3d} {nF:12.4e} {nFsc:12.4e} {np.linalg.norm(dx):11.3e} {lm:7.4f} {tag}")
        self.wall=time.time()-t0
        self.residuals.append(np.linalg.norm(residual(x)))
        if self.verbose: print(f"   ✗ not converged  ||F/d||={np.linalg.norm(residual(x)/DS):.3e}")
        return x, False


def convergence_rate(res):
    rates=[]
    for i in range(2,len(res)-1):
        a,b,c=res[i],res[i-1],res[i-2]
        if min(a,b,c)>1e-20:
            try:
                p=np.log(a/b)/np.log(b/c)
                if 0.3<p<4.: rates.append(p)
            except Exception: pass
    return float(np.mean(rates)) if rates else float("nan")


# =============================================================================
#  SECTION H.  SPECTRAL ANALYSIS OF THE PRECONDITIONED OPERATOR
# =============================================================================
def spectral_analysis(x):
    """Return eigenvalues and condition numbers of the raw and preconditioned
    Jacobians at state x, quantifying the spectral compression achieved by each
    physics-based preconditioner."""
    n=N_DOF; F0=residual(x); J=np.zeros((n,n))
    for j in range(n):
        dj=1e-7*max(abs(x[j]),1.0); xp=x.copy(); xp[j]+=dj
        J[:,j]=(residual(xp)-F0)/dj
    Js=(J*DS[None,:])/DS[:,None]                 # scaled (Jacobi/None operate here)
    # FieldSplit block-diagonal inverse
    Minv=np.zeros((n,n))
    for _,idx in FIELD_BLOCKS:
        sub=Js[np.ix_(idx,idx)]
        try: Minv[np.ix_(idx,idx)]=np.linalg.inv(sub)
        except np.linalg.LinAlgError: Minv[np.ix_(idx,idx)]=np.linalg.pinv(sub)
    Jfs=Minv@Js
    dvec=np.diag(Js).copy(); dvec[np.abs(dvec)<1e-12]=1.0
    Jjac=(1.0/dvec)[:,None]*Js
    def _cond(M):
        try: return float(np.linalg.cond(M))
        except Exception: return float("nan")
    return {
        "J_raw":      (np.linalg.eigvals(J),  _cond(J)),
        "J_scaled":   (np.linalg.eigvals(Js), _cond(Js)),
        "J_jacobi":   (np.linalg.eigvals(Jjac),_cond(Jjac)),
        "J_fieldsplit":(np.linalg.eigvals(Jfs),_cond(Jfs)),
    }


# =============================================================================
#  SECTION I.  POST-PROCESSING  &  CYCLE PERFORMANCE
# =============================================================================
def extract_results(x):
    """Thermodynamic state and power balance of the converged cycle."""
    m=x[0:15]; p=x[15:30]; h=x[30:45]
    T=np.array([safe_T(p[i],h[i]) for i in range(15)])-273.15
    # Steam quality.  The interval is CLOSED: the deaerator outlet is pinned
    # exactly onto the saturated-liquid line (h13 = hf(P_DA)) by the DA
    # singularity split, so a strict test would report it as undefined rather
    # than as the saturated liquid, x = 0, that it physically is.
    xq=np.full(15,np.nan)
    for i in range(15):
        hf_,hg_=h_f(p[i]),h_g(p[i])
        if hg_>hf_ and hf_-1e-6<=h[i]<=hg_+1e-6:
            xq[i]=min(max((h[i]-hf_)/(hg_-hf_),0.0),1.0)
    W_HP=(m[0]-m[1])*(h[0]-h[1])+m[2]*(h[1]-h[2])
    W_LP=(m[6]*(h[6]-h[3])+(m[6]-m[3])*(h[3]-h[4])
          +(m[6]-m[3]-m[4])*(h[4]-h[5])+m[7]*(h[5]-h[7]))
    W_cp=m[8]*(h[8]-h_f(p[7])); W_fw=m[13]*(h[13]-h[12])
    W_gross=W_HP+W_LP; W_pump=W_cp+W_fw; W_net=W_gross-W_pump
    Q_in=m[0]*(h[0]-h[14])
    return dict(m=m,p=p,h=h,T=T,x=xq,W_HP=W_HP,W_LP=W_LP,W_gross=W_gross,
                W_cp=W_cp,W_fw=W_fw,W_pump=W_pump,W_net=W_net,Q_in=Q_in,
                eta=W_net/Q_SG_BC,l_pre=x[I_LPRE],l_eva=x[I_LEVA],
                l_sup=L_SG-x[I_LPRE]-x[I_LEVA],
                T_He=np.array([T_He_in,x[I_THE2],x[I_THE1],x[I_THE3]])-273.15)

def print_cycle_summary(x):
    R=extract_results(x)
    print("\n"+"="*78)
    print("  CONVERGED CYCLE STATE  -  HTR-PM SECONDARY LOOP (2 x 250 MWth)")
    print("="*78)
    print(f"  {'#':>2} {'node':<16}{'m [kg/s]':>10}{'P [MPa]':>10}"
          f"{'h [kJ/kg]':>11}{'T [C]':>9}  phase")
    print("  "+"-"*74)
    for i in range(15):
        hf_,hg_=h_f(R['p'][i]),h_g(R['p'][i])
        if   R['h'][i]<hf_-1e-6: ph="subcooled"
        elif R['h'][i]>hg_+1e-6: ph="superheat"
        elif abs(R['h'][i]-hf_)<=1e-6: ph="sat. liquid  (x = 0.000)"
        elif abs(R['h'][i]-hg_)<=1e-6: ph="sat. vapour  (x = 1.000)"
        else: ph=f"x = {R['x'][i]:.3f}"
        print(f"  {i:2d} {NODE_LABELS[i]:<16}{R['m'][i]:10.2f}{R['p'][i]/1e6:10.3f}"
              f"{R['h'][i]/1e3:11.1f}{R['T'][i]:9.1f}  {ph}")
    print("  "+"-"*74)
    print(f"  HP turbine work           {R['W_HP']/1e6:9.2f} MW")
    print(f"  LP turbine work           {R['W_LP']/1e6:9.2f} MW")
    print(f"  Gross mechanical power    {R['W_gross']/1e6:9.2f} MW")
    print(f"  Pump work (cond + FW)     {R['W_pump']/1e6:9.2f} MW"
          f"   ({R['W_cp']/1e6:.2f} + {R['W_fw']/1e6:.2f})")
    print(f"  NET ELECTRICAL POWER      {R['W_net']/1e6:9.2f} MW")
    print(f"  SG thermal duty           {R['Q_in']/1e6:9.2f} MW")
    print(f"  NET CYCLE EFFICIENCY      {R['eta']*100:9.2f} %")
    print("  "+"-"*74)
    print(f"  SG zone lengths [m]   PRE {R['l_pre']:6.2f} | EVA {R['l_eva']:6.2f} |"
          f" SUP {R['l_sup']:6.2f}   (L = {L_SG:.0f})")
    print(f"  Helium path [C]       in {R['T_He'][0]:6.1f} -> {R['T_He'][1]:6.1f}"
          f" -> {R['T_He'][2]:6.1f} -> out {R['T_He'][3]:6.1f}")
    print("="*78)

def print_table2(x):
    """Normalised comparison with Energies (2023) 16, 2252, Table 2.

    NOTE ON NODE LABELLING:  the reference tabulates the LP-turbine exhaust,
    which in the present 15-node topology is node 8 (index 7), not node 3.
    Comparing against index 2 -- as a literal reading of the published table
    suggests -- mismatches the topology; the corrected mapping is used here."""
    R=extract_results(x); m0=R['m'][0]; h0=R['h'][0]; p0=R['p'][0]
    rows=[("LP exhaust  m/m0",  R['m'][7]/m0, 0.6714),
          ("LP exhaust  h/h0",  R['h'][7]/h0, 0.6398),
          ("Cond pump   m/m0",  R['m'][8]/m0, 0.8706),
          ("Cond pump   p/p0",  R['p'][8]/p0, 0.1109),
          ("FW pump     m/m0",  R['m'][13]/m0,1.0000),
          ("FW pump     p/p0",  R['p'][13]/p0,1.0040),
          ("FW pump     h/h0",  R['h'][13]/h0,0.1775)]
    print("\n  Normalised state vs. Energies (2023) Table 2")
    print("  "+"-"*58)
    print(f"  {'quantity':<20}{'present':>10}{'reference':>12}{'dev [%]':>12}")
    for nm,v,r in rows:
        print(f"  {nm:<20}{v:10.4f}{r:12.4f}{(v-r)/r*100:12.2f}")
    print("  "+"-"*58)

def sensitivity_analysis(groups=("m","p","h","all"),
                         factors=(0.5,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,2.0),
                         pc_factory=None, use_aspin=False, quiet=True):
    """Perturb whole physical fields of the initial guess and record the
    Newton-step count -- the 40-case study reported in the paper."""
    if pc_factory is None: pc_factory=FieldSplitPreconditioner
    x_ref=build_x0(); idx={"m":range(0,15),"p":range(15,30),
                           "h":range(30,45),"all":range(0,N_DOF)}
    out={}
    for g in groups:
        rec=[]
        for f in factors:
            x0=x_ref.copy(); x0[list(idx[g])]*=f
            sv=JFNKSolver(pc_factory(),tol=1e-6,max_newton=30,use_aspin=use_aspin)
            _,c=sv.solve(project_physical(x0))
            rec.append((f,len(sv.residuals)-1 if c else np.nan,c,sum(sv.gmres_iters)))
        out[g]=rec
        if not quiet:
            ok=[r[1] for r in rec if r[2]]
            print(f"    {g:<4} solved {sum(r[2] for r in rec)}/{len(rec)}  "
                  f"avg steps {np.mean(ok):.2f}")
    return out

def correction_factor_sweep(etas=np.arange(1.0,1.81,0.1), pc_factory=None):
    """Sweep an over-relaxation / correction factor applied to the enthalpy
    field of the initial guess (cf. Energies Table 5)."""
    if pc_factory is None: pc_factory=FieldSplitPreconditioner
    x_ref=build_x0(); res=[]
    for e in etas:
        x0=x_ref.copy(); x0[30:45]*=e
        sv=JFNKSolver(pc_factory(),tol=1e-6,max_newton=30)
        _,c=sv.solve(project_physical(x0))
        res.append((float(e),len(sv.residuals)-1 if c else np.nan,c))
    return res


# =============================================================================
#  SECTION J.  REPRODUCING PUBLICATION FIGURES
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
    "axes.labelsize":9,"axes.titlesize":9.5,"legend.fontsize":8,
    "xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":0.8,
    "figure.dpi":160,"savefig.dpi":330,"savefig.bbox":"tight",
    "axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.5,
    "legend.frameon":True,"legend.framealpha":0.92,"legend.edgecolor":"0.7",
})
PC_COLORS={"None":"#8c8c8c","Jacobi":"#1f6fb4","FieldSplit":"#c1272d","ASPIN":"#1a8a4a"}
PC_MARKS ={"None":"o","Jacobi":"s","FieldSplit":"D","ASPIN":"^"}
FIGDIR="figs"

def _pc_registry():
    return [("None",       lambda: NonePreconditioner(),       False),
            ("Jacobi",     lambda: JacobiPreconditioner(),     False),
            ("FieldSplit", lambda: FieldSplitPreconditioner(), False),
            ("ASPIN",      lambda: ASPINPreconditioner(),      True )]

def run_all_preconditioners(x0=None, tol=1e-10, max_newton=40):
    if x0 is None: x0=build_x0()
    out={}
    for nm,fac,asp in _pc_registry():
        sv=JFNKSolver(fac(),tol=tol,max_newton=max_newton,use_aspin=asp,verbose=True)
        x,c=sv.solve(x0.copy())
        out[nm]=dict(x=x,conv=c,res=np.array(sv.residuals),
                     gmres=np.array(sv.gmres_iters),wall=sv.wall,
                     steps=len(sv.residuals)-1,rate=convergence_rate(sv.residuals))
    return out


def fig_convergence(runs, fname="fig01_convergence.pdf"):
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
    for nm in runs:
        r=runs[nm]; y=r["res"]/r["res"][0]
        ax[0].semilogy(range(len(y)),y,marker=PC_MARKS[nm],ms=4,lw=1.4,
                       color=PC_COLORS[nm],label=f"{nm} ({r['steps']} it)")
        if len(r["gmres"]):
            ax[1].plot(range(1,len(r["gmres"])+1),r["gmres"],marker=PC_MARKS[nm],
                       ms=4,lw=1.4,color=PC_COLORS[nm],label=nm)
    ax[0].set_xlabel("Newton iteration $k$"); ax[0].set_ylabel(r"$\|F_k\|/\|F_0\|$")
    ax[0].set_title("(a) Nonlinear convergence"); ax[0].legend(loc="lower left")
    ax[0].set_ylim(1e-16,5)
    ax[1].set_xlabel("Newton iteration $k$"); ax[1].set_ylabel("GMRES iterations")
    ax[1].set_title("(b) Krylov work per Newton step"); ax[1].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname)); plt.close(fig)

def fig_spectra(x, fname="fig02_spectra.pdf"):
    S=spectral_analysis(x)
    panels=[("J_scaled","Unpreconditioned $\\hat{J}$","#8c8c8c"),
            ("J_jacobi","Jacobi $D^{-1}\\hat{J}$","#1f6fb4"),
            ("J_fieldsplit","FieldSplit $M_{FS}^{-1}\\hat{J}$","#c1272d")]
    fig,axes=plt.subplots(1,3,figsize=(7.4,2.6))
    for ax,(k,ttl,c) in zip(axes,panels):
        ev,cond=S[k]
        ax.axhline(0,color="0.75",lw=0.6); ax.axvline(0,color="0.75",lw=0.6)
        ax.scatter(ev.real,ev.imag,s=16,facecolor=c,edgecolor="k",
                   linewidth=0.35,alpha=0.85,zorder=3)
        ax.set_title(f"{ttl}\n$\\kappa$ = {cond:.2e}")
        ax.set_xlabel(r"Re$(\lambda)$")
        if k=="J_scaled": ax.set_ylabel(r"Im$(\lambda)$")
        if k!="J_scaled":
            ax.set_xlim(-0.15,1.35); ax.axvline(1.0,color="k",ls=":",lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname)); plt.close(fig)

def fig_conditioning(x, runs, fname="fig03_conditioning.pdf"):
    S=spectral_analysis(x)
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    keys=["J_raw","J_scaled","J_jacobi","J_fieldsplit"]
    labs=["raw $J$","scaled $\\hat J$","Jacobi","FieldSplit"]
    conds=[S[k][1] for k in keys]
    cols=["#4d4d4d","#8c8c8c","#1f6fb4","#c1272d"]
    b=ax[0].bar(labs,conds,color=cols,edgecolor="k",linewidth=0.5,width=0.62)
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"condition number $\kappa$")
    ax[0].set_title("(a) Spectral conditioning")
    for r,c in zip(b,conds):
        ax[0].text(r.get_x()+r.get_width()/2,c*1.6,f"{c:.1e}",ha="center",fontsize=6.6)
    ax[0].set_ylim(top=max(conds)*60)
    plt.setp(ax[0].get_xticklabels(),rotation=12,ha="right")
    nms=list(runs); tot=[int(runs[n]["gmres"].sum()) for n in nms]
    b2=ax[1].bar(nms,tot,color=[PC_COLORS[n] for n in nms],edgecolor="k",
                 linewidth=0.5,width=0.62)
    ax[1].set_ylabel("total GMRES iterations"); ax[1].set_title("(b) Cumulative Krylov cost")
    for r,v in zip(b2,tot):
        ax[1].text(r.get_x()+r.get_width()/2,v+max(tot)*0.02,str(v),ha="center",fontsize=7)
    ax[1].set_ylim(top=max(tot)*1.18)
    plt.setp(ax[1].get_xticklabels(),rotation=12,ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname)); plt.close(fig)

# ---------------------------------------------------------------------------
#  Thermodynamically consistent cycle-path construction (plotting only)
#
#  The cycle is assembled from labelled PHYSICAL PROCESS segments, not from
#  node-to-node chords. 
# ---------------------------------------------------------------------------
def _s_plot(P, h):
    """Entropy for plotting; exact in the compressed-liquid region."""
    if P < 1e3: P = 1e3
    if h <= h_f(P):
        return _if97_r1(P, T_Ph(P, h))[1]
    return s_Ph(P, h)


def _seg_expansion(h_in, P_in, P_out, eta, n=40):
    """Real (irreversible) turbine expansion line.

    Intermediate states are generated from the same isentropic-efficiency
    relation used in the residual,
        h(P) = h_in - eta [ h_in - h_is(h_in, P_in, P) ],
    so the curve is the machine's actual expansion line, entropy increases
    monotonically, and the endpoint reproduces the solved outlet state to
    round-off.  Pressure is sampled geometrically because T_sat varies
    steeply at low pressure; linear sampling under-resolves the exhaust end
    and produces a spurious concave bulge."""
    Ps = np.exp(np.linspace(_log(max(P_in, 1e3)), _log(max(P_out, 1e3)), n))
    Hs = np.array([h_in - eta*(h_in - h_is_expansion(h_in, P_in, P)) for P in Ps])
    return Ps, Hs


def _seg_isobaric(P, h_a, h_b, n=40):
    """Constant-pressure heating, cooling or condensation."""
    return np.full(n, P), np.linspace(h_a, h_b, n)


def _seg_throttle(h, P_a, P_b, n=12):
    """Isenthalpic throttle."""
    return np.linspace(P_a, P_b, n), np.full(n, h)


def _seg_pump(P_a, h_a, P_b, h_b, n=12):
    """Liquid pump; both endpoints subcooled, no phase change."""
    return np.linspace(P_a, P_b, n), np.linspace(h_a, h_b, n)


def cycle_path(R):
    """Return the cycle as a list of (kind, segment) physical processes."""
    p, h = R["p"], R["h"]
    segs = []; A = segs.append
    A(("sg",   _seg_isobaric(p[14], h[14], h[0], 60)))          # steam generator
    A(("exp",  _seg_expansion(h[0], p[0], p[1], eta_HP)))       # HP stage 1
    A(("exp",  _seg_expansion(h[1], p[1], p[2], eta_HP)))       # HP stage 2
    A(("exp",  _seg_expansion(h[6], p[6], p[3], eta_LP)))       # LP stage 1
    A(("exp",  _seg_expansion(h[3], p[3], p[4], eta_LP)))       # LP stage 2
    A(("exp",  _seg_expansion(h[4], p[4], p[5], eta_LP)))       # LP stage 3
    A(("exp",  _seg_expansion(h[5], p[5], p[7], eta_LP)))       # LP stage 4
    A(("cond", _seg_isobaric(p[7], h[7], h_f(p[7]), 40)))       # condensation
    A(("fw",   _seg_pump(p[7], h_f(p[7]), p[8], h[8])))         # condensate pump
    A(("fw",   _seg_isobaric(p[8],  h[8],  h[9],  30)))         # LPH-3
    A(("fw",   _seg_isobaric(p[9],  h[9],  h[10], 30)))         # LPH-2
    A(("fw",   _seg_isobaric(p[10], h[10], h[11], 30)))         # LPH-1
    A(("fw",   _seg_throttle(h[11], p[11], p[12])))             # throttle to DA
    A(("fw",   _seg_isobaric(p[12], h[11], h[12], 30)))         # DA heating
    A(("fw",   _seg_pump(p[12], h[12], p[13], h[13])))          # feedwater pump
    A(("fw",   _seg_isobaric(p[13], h[13], h[14], 30)))         # HP heater
    return segs


def _seg_TS(seg):
    P, H = seg
    return ([_s_plot(P[k], H[k])/1e3 for k in range(len(P))],
            [safe_T(P[k], H[k])-273.15 for k in range(len(P))])


def _seg_TH(seg):
    P, H = seg
    return (H/1e3, [safe_T(P[k], H[k])-273.15 for k in range(len(P))])


C_EXP, C_COND, C_FW, C_SG = "#c1272d", "#6a51a3", "#1f6fb4", "#e08214"
_SEG_STYLE = {"exp": (C_EXP,  "turbine expansion"),
              "cond": (C_COND, "condensation"),
              "fw":   (C_FW,   "feedwater heating"),
              "sg":   (C_SG,   "steam generation")}


def _draw_dome(ax, xf, xg, Td):
    ax.plot(xf, Td, "-", color="0.35", lw=1.1)
    ax.plot(xg, Td, "-", color="0.35", lw=1.1)
    ax.fill(np.concatenate([xf, xg[::-1]]), np.concatenate([Td, Td[::-1]]),
            color="0.86", alpha=0.6, zorder=0)


def _draw_path(ax, segs, conv):
    seen = set()
    for kind, seg in segs:
        col, lab = _SEG_STYLE[kind]
        X, Y = conv(seg)
        ax.plot(X, Y, "-", color=col, lw=1.6, zorder=4,
                label=(lab if kind not in seen else None))
        seen.add(kind)


def fig_th_diagram(x, fname="fig04_Ts_diagram.png"):
    """Temperature-entropy diagram (figure 04)."""
    R = extract_results(x); segs = cycle_path(R)
    Pd = np.logspace(np.log10(3e3), np.log10(_Pc*0.999), 200)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    _draw_dome(ax, np.array([s_f(P) for P in Pd])/1e3,
                   np.array([s_g(P) for P in Pd])/1e3,
                   np.array([T_sat(P) for P in Pd])-273.15)
    _draw_path(ax, segs, _seg_TS)
    s_pt = [_s_plot(R["p"][i], R["h"][i])/1e3 for i in range(15)]
    ax.plot(s_pt, R["T"], "o", ms=4, mfc="w", mec="k", mew=0.7, zorder=6)
    for i in [0, 2, 7, 12, 14]:
        ax.annotate(str(i+1), (s_pt[i], R["T"][i]), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, zorder=7)
    ax.set_xlabel("specific entropy  $s$  [kJ/(kg K)]")
    ax.set_ylabel("temperature  $T$  [$^\\circ$C]")
    ax.set_title("$T$-$s$ diagram of the secondary circuit")
    ax.set_xlim(0, 8.2); ax.set_ylim(0, 640)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, fname)); plt.close(fig)


def fig_sg_profile(x, fname="fig05_SG_profile.png"):
    """Counter-current temperature profile through the 3-zone once-through SG.
    Helium enters at the superheater end (z = L) and leaves at the economiser
    end (z = 0), counter-current to the feedwater."""
    R=extract_results(x); lp,le,ls=R['l_pre'],R['l_eva'],R['l_sup']
    P=x[15]; Ts=T_sat(P)-273.15
    Tw_in=R['T'][14]; Tw_out=R['T'][0]
    zp=np.linspace(0,lp,40); ze=np.linspace(lp,lp+le,40); zs=np.linspace(lp+le,L_SG,40)
    Tw=np.concatenate([np.linspace(Tw_in,Ts,40),np.full(40,Ts),np.linspace(Ts,Tw_out,40)])
    z =np.concatenate([zp,ze,zs])
    THe=np.concatenate([np.linspace(R['T_He'][3],R['T_He'][2],40),
                        np.linspace(R['T_He'][2],R['T_He'][1],40),
                        np.linspace(R['T_He'][1],R['T_He'][0],40)])
    fig,ax=plt.subplots(figsize=(5.0,3.4))
    for a,b,c,lab in [(0,lp,"#cfe3f7","economiser"),(lp,lp+le,"#f9e0c3","evaporator"),
                      (lp+le,L_SG,"#f6c9c9","superheater")]:
        ax.axvspan(a,b,color=c,alpha=0.75,zorder=0)
        ax.text((a+b)/2,715,lab,ha="center",fontsize=7.2)
    ax.plot(z,THe,color="#c1272d",lw=1.7,label="helium (primary)")
    ax.plot(z,Tw ,color="#1f6fb4",lw=1.7,label="water/steam (secondary)")
    ax.axhline(Ts,color="0.5",ls=":",lw=0.9)
    ax.text(1.0,Ts+9,f"$T_{{sat}}$ = {Ts:.0f} $^\\circ$C",fontsize=7,color="0.3")
    ax.set_xlabel("axial coordinate  $z$  [m]")
    ax.set_ylabel("temperature  [$^\\circ$C]")
    ax.set_title("Once-through helical SG: three-zone profile")
    ax.set_xlim(0,L_SG); ax.set_ylim(100,760); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname)); plt.close(fig)

def fig_robustness(bands, fname="fig06_robustness.png"):
    """bands: dict band_label -> {pc: (n_solved, N, avg_newton, avg_gmres)}"""
    labels=list(bands); nms=["None","Jacobi","FieldSplit","ASPIN"]
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    w=0.2; xpos=np.arange(len(labels))
    for i,nm in enumerate(nms):
        rate=[bands[b][nm][0]/bands[b][nm][1]*100 for b in labels]
        gm  =[bands[b][nm][3] for b in labels]
        ax[0].bar(xpos+(i-1.5)*w,rate,w,color=PC_COLORS[nm],edgecolor="k",
                  linewidth=0.4,label=nm)
        ax[1].bar(xpos+(i-1.5)*w,gm,w,color=PC_COLORS[nm],edgecolor="k",linewidth=0.4)
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("converged cases [%]"); ax[0].set_ylim(0,112)
    ax[0].set_title("(a) Global robustness"); ax[0].legend(loc="lower right", ncol=2,fontsize=7)
    ax[1].set_xticks(xpos); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("mean total GMRES iterations"); ax[1].set_yscale("log")
    ax[1].set_title("(b) Mean Krylov cost")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname)); plt.close(fig)

def fig_sensitivity(sens, fname="fig07_sensitivity.png"):
    fig,axes=plt.subplots(1,len(sens),figsize=(7.4,2.5),sharey=True)
    if len(sens)==1: axes=[axes]
    for ax,(g,rec) in zip(axes,sens.items()):
        f=[r[0] for r in rec]; s=[r[1] for r in rec]
        ax.plot(f,s,"-o",ms=4,lw=1.3,color="#c1272d")
        ax.axhline(np.nanmean(s),color="0.45",ls="--",lw=0.9,
                   label=f"mean {np.nanmean(s):.2f}")
        ax.set_title(f"field  $\\{{{g}\\}}$" if g!="all" else "all fields")
        ax.set_xlabel("perturbation factor"); ax.legend(fontsize=7)
    axes[0].set_ylabel("Newton iterations")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname)); plt.close(fig)

def fig_eta_sweep(sweep, fname="fig08_correction_factor.png"):
    e=[r[0] for r in sweep]; s=[r[1] for r in sweep]
    fig,ax=plt.subplots(figsize=(4.4,2.7))
    ax.plot(e,s,"-D",ms=4,lw=1.3,color="#1a8a4a")
    ax.axhline(np.nanmean(s),color="0.45",ls="--",lw=0.9,
               label=f"mean {np.nanmean(s):.2f}")
    ax.set_xlabel(r"initial-guess correction factor  $\eta$")
    ax.set_ylabel("Newton iterations"); ax.legend()
    ax.set_title("Sensitivity to initial-guess scaling")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname)); plt.close(fig)

def fig_normalised(x, fname="fig09_normalised_state.png"):
    R=extract_results(x); m0,p0,h0=R['m'][0],R['p'][0],R['h'][0]
    n=np.arange(1,16)
    fig,ax=plt.subplots(figsize=(6.4,3.0))
    ax.plot(n,R['m']/m0,"-o",ms=4,lw=1.3,color="#1f6fb4",label=r"$\dot m/\dot m_1$")
    ax.plot(n,R['p']/p0,"-s",ms=4,lw=1.3,color="#c1272d",label=r"$p/p_1$")
    ax.plot(n,R['h']/h0,"-^",ms=4,lw=1.3,color="#1a8a4a",label=r"$h/h_1$")
    ax.set_xticks(n); ax.set_xlabel("node index"); ax.set_ylabel("normalised value")
    ax.set_title("Converged normalised cycle state")
    ax.legend(ncol=3); ax.set_ylim(-0.05,1.15)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname)); plt.close(fig)

# ---------------------------------------------------------------------------
#  T-h diagram  (with a filled saturation dome,
#  resolved isobaric legs and the explicit SG heating path)
# ---------------------------------------------------------------------------
def fig_th(x, fname="fig10_Th_diagram.png"):
    """Temperature-enthalpy diagram (figure 10).

    Uses the same physical-process segmentation as the T-s diagram, so the
    condenser appears as a horizontal isotherm at T_sat(P_cond) and each
    turbine stage follows its real expansion line."""
    R = extract_results(x); segs = cycle_path(R)
    Pd = np.logspace(np.log10(3e3), np.log10(_Pc*0.999), 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _draw_dome(ax, np.array([h_f(P) for P in Pd])/1e3,
                   np.array([h_g(P) for P in Pd])/1e3,
                   np.array([T_sat(P) for P in Pd])-273.15)
    ax.text(1300, 150, "two-phase", fontsize=7.5, color="0.4", style="italic")
    for P in [P_COND, P_DA, P_LP1, P_HP2, P_SG]:          # reference isobars
        hh = np.linspace(max(h_f(P)-3.2e5, 2e4), 3.75e6, 140)
        ax.plot(hh/1e3, [safe_T(P, q)-273.15 for q in hh],
                ":", color="0.62", lw=0.7, zorder=1)
        ax.text(3.62e3, safe_T(P, 3.62e6)-273.15, f"{P/1e6:.3g} MPa",
                fontsize=6.2, color="0.45", va="center")
    _draw_path(ax, segs, _seg_TH)
    ax.plot(R["h"]/1e3, R["T"], "o", ms=4, mfc="w", mec="k", mew=0.7, zorder=6)
    for i in [0, 2, 6, 7, 8, 12, 14]:
        ax.annotate(str(i+1), (R["h"][i]/1e3, R["T"][i]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7, zorder=7)
    ax.set_xlabel("specific enthalpy  $h$  [kJ/kg]")
    ax.set_ylabel("temperature  $T$  [$^\\circ$C]")
    ax.set_title("$T$-$h$ diagram of the secondary circuit")
    ax.set_xlim(0, 3900); ax.set_ylim(0, 640)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, fname)); plt.close(fig)


# ---------------------------------------------------------------------------
#  Normalised state, plotted in PROCESS order.
#  The node index is a labelling convention, not the flow sequence: node 7
#  (LP inlet) sits upstream of nodes 4-6 (the LP extractions), so an
#  index-ordered plot shows a spurious pressure recovery at node 7.
# ---------------------------------------------------------------------------
PROCESS_ORDER=[0,1,2,6,3,4,5,7,8,9,10,11,12,13,14]

def fig_normalised_process(x, fname="fig11_normalised_process.png"):
    R=extract_results(x); m0,p0,h0=R['m'][0],R['p'][0],R['h'][0]
    o=PROCESS_ORDER; xs=np.arange(len(o))
    fig,ax=plt.subplots(figsize=(7.0,3.2))
    ax.plot(xs,R['m'][o]/m0,"-o",ms=4,lw=1.3,color="#1f6fb4",label=r"$\dot m/\dot m_0$")
    ax.plot(xs,R['p'][o]/p0,"-s",ms=4,lw=1.3,color="#c1272d",label=r"$p/p_0$")
    ax.plot(xs,R['h'][o]/h0,"-^",ms=4,lw=1.3,color="#1a8a4a",label=r"$h/h_0$")
    ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in o])
    ax.set_xlabel("node (process order:  SG $\\to$ HP $\\to$ LP $\\to$ condenser"
                  " $\\to$ feedwater train)")
    ax.set_ylabel("normalised value")
    ax.set_title("Converged normalised cycle state (process-ordered)")
    for a,b,c,lab in [(-0.4,2.5,"#eef4fb","turbines"),(6.5,7.5,"#f6eaea","condenser"),
                      (7.5,14.4,"#eef7ef","feedwater train")]:
        ax.axvspan(a,b,color=c,zorder=0)
        ax.text((a+b)/2,1.09,lab,ha="center",fontsize=7,color="0.35")
    ax.legend(ncol=3,loc="center right"); ax.set_ylim(-0.05,1.17)
    ax.set_xlim(-0.5,14.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname)); plt.close(fig)

# =============================================================================
#  SECTION K.  ROBUSTNESS STUDY  &  MAIN DRIVER
# =============================================================================
def robustness_study(n_seeds=50, bands=((0.35,2.8,"moderate"),
                                        (0.15,6.0,"severe"),
                                        (0.05,15.0,"extreme")), verbose=True):
    """Randomised far-from-solution start study: the basin-of-attraction and
    Krylov-cost comparison across the four preconditioners."""
    x_ref=build_x0(); out={}
    for lo,hi,lab in bands:
        rec={}
        for nm,fac,asp in _pc_registry():
            ns=[];gm=[];c=0
            for seed in range(n_seeds):
                rng=np.random.default_rng(seed)
                f=np.exp(rng.uniform(np.log(lo),np.log(hi),size=N_DOF))
                sv=JFNKSolver(fac(),tol=1e-6,max_newton=20,use_aspin=asp)
                _,ok=sv.solve(project_physical(x_ref*f))
                if ok:
                    c+=1; ns.append(len(sv.residuals)-1); gm.append(sum(sv.gmres_iters))
            rec[nm]=(c,n_seeds,float(np.mean(ns)) if ns else np.nan,
                     float(np.mean(gm)) if gm else np.nan)
        out[lab]=rec
        if verbose:
            print(f"  band [{lo}, {hi}]  ({lab})")
            for nm in rec:
                c,N,a,g=rec[nm]
                print(f"    {nm:<11}{c:3d}/{N}   Newton {a:5.2f}   GMRES {g:7.1f}")
    return out


def property_check():
    print("\n  Property-layer verification (IAPWS-IF97 reference values)")
    print("  "+"-"*58)
    for P,Tref in [(4.5e3,31.0),(0.72e6,165.9),(3.5e6,242.6),(13.9e6,336.1)]:
        print(f"    T_sat({P/1e6:7.4f} MPa) = {T_sat(P)-273.15:7.2f} C"
              f"   (ref {Tref:6.1f})")
    # Reference values are IF97 Region 1 evaluated at T_sat(P).  (An earlier
    # revision of this table carried an incorrect h_f(13.9 MPa) = 1541.0,
    # which made the property layer appear 2 % in error when the true
    # discrepancy was 0.28 % of spline interpolation, now removed entirely.)
    for P,href in [(0.72e6,702.12),(13.9e6,1566.95),(4.5e3,129.98),
                   (3.5e6,1049.78)]:
        print(f"    h_f ({P/1e6:7.4f} MPa) = {h_f(P)/1e3:8.2f} kJ/kg"
              f" (ref {href:8.2f})")
    print(f"    T(14.456 MPa, 718.7 kJ/kg) = {T_Ph(14.456e6,718.7e3)-273.15:7.2f} C"
         f"  (IF97 R1 inversion; ref 168.12)")
    print(f"    T(13.9 MPa, 3634.4 kJ/kg)  = {T_Ph(13.9e6,3634.4e3)-273.15:7.2f} C"
         f"  (IF97 R2 inversion; ref 616.20)")
    print(f"    T(7.5 MPa, 3415.5 kJ/kg)   = {T_Ph(7.5e6,3415.5e3)-273.15:7.2f} C"
         f"  (IF97 R2 inversion; ref 504.15)")
    # Pump work: exact IF97 Region 1 isentropic inversion (v9).  The isentropic
    # temperature rise is 0.03 K (condensate) and 1.70 K (feedwater); the
    # actual enthalpy rise follows after the isentropic efficiency.
    dh_cp=(h_pump_isentropic(P_COND,P_CP,eta_cp)-h_f(P_COND))/1e3
    dh_fw=(h_pump_isentropic(P_DA,P_SG*1.04,eta_fw)-h_f(P_DA))/1e3
    print(f"    cond-pump dh (exact isentropic) = {dh_cp:6.3f} kJ/kg  (eta={eta_cp})")
    print(f"    FW-pump   dh (exact isentropic) = {dh_fw:6.3f} kJ/kg  (eta={eta_fw})")
    print("  "+"-"*58)


def main(n_seeds=50, quick=False):
    os.makedirs(FIGDIR,exist_ok=True)
    if quick: n_seeds=8
    print("="*78)
    print("  HTR-PM SECONDARY-CYCLE JFNK SOLVER  --  v9.3")
    print("  Physics-based preconditioning: FieldSplit(block-Jacobi) & ASPIN")
    print(f"  {N_DOF} degrees of freedom   |   {len(FIELD_BLOCKS)} physical fields"
          f"   |   {len(ASPIN_BLOCKS)} ASPIN subdomains")
    print("="*78)
    property_check()

    print("\n  [1/6] Newton-Krylov solve with each preconditioner")
    runs=run_all_preconditioners()
    print("  "+"-"*72)
    print(f"  {'preconditioner':<14}{'conv':>6}{'Newton':>8}{'GMRES':>8}"
          f"{'rate p':>9}{'wall [s]':>10}")
    for nm,r in runs.items():
        print(f"  {nm:<14}{str(r['conv']):>6}{r['steps']:8d}"
              f"{int(r['gmres'].sum()):8d}{r['rate']:9.2f}{r['wall']:10.3f}")
    print("  "+"-"*72)
    g0=int(runs['None']['gmres'].sum()); gf=int(runs['FieldSplit']['gmres'].sum())
    print(f"  Krylov-cost reduction, FieldSplit vs unpreconditioned: {g0/max(gf,1):.1f}x")

    x=runs["FieldSplit"]["x"]
    print_cycle_summary(x); print_table2(x)

    print("\n  [2/6] Spectral analysis")
    S=spectral_analysis(x)
    print("  "+"-"*58)
    for k,lab in [("J_raw","raw J"),("J_scaled","scaled J"),
                  ("J_jacobi","Jacobi"),("J_fieldsplit","FieldSplit")]:
        ev,cd=S[k]
        print(f"    {lab:<12} kappa = {cd:.4e}   max|Im(lambda)| = {np.abs(ev.imag).max():.2e}")
    print(f"    conditioning gain (raw -> FieldSplit): "
          f"{S['J_raw'][1]/S['J_fieldsplit'][1]:.3e}x")
    print("  "+"-"*58)

    print("\n  [3/6] Field-block conditioning")
    fs=FieldSplitPreconditioner()
    for k,v in fs.block_conds(x).items(): print(f"    {k:<14} kappa = {v:.3e}")

    print(f"\n  [4/6] Robustness study ({n_seeds} random starts x 3 bands)")
    bands=robustness_study(n_seeds=n_seeds)

    print("\n  [5/6] Sensitivity study (field perturbations)")
    sens=sensitivity_analysis(quiet=False)
    sweep=correction_factor_sweep()
    ok=[r[1] for r in sweep if r[2]]
    print(f"    correction-factor sweep: {len(ok)}/{len(sweep)} solved, "
          f"mean {np.mean(ok):.2f} Newton steps")

    print("\n  [6/6] Rendering figures")
    fig_convergence(runs);        fig_spectra(x)
    fig_conditioning(x,runs);     fig_th_diagram(x)
    fig_sg_profile(x);            fig_robustness(bands)
    fig_sensitivity(sens);        fig_eta_sweep(sweep)
    fig_normalised(x);            fig_th(x)
    fig_normalised_process(x)
    for f in sorted(os.listdir(FIGDIR)): print(f"    {FIGDIR}/{f}")
    print("\n  done.")
    return runs,bands,sens,sweep



if __name__=="__main__":
    import sys
    main(quick=("--quick" in sys.argv))