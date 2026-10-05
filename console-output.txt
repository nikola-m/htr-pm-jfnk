==============================================================================
  HTR-PM SECONDARY-CYCLE JFNK SOLVER  --  v9
  Physics-based preconditioning: FieldSplit(block-Jacobi) & ASPIN
  50 degrees of freedom   |   4 physical fields   |   6 ASPIN subdomains
==============================================================================

  Property-layer verification (IAPWS-IF97 reference values)
  ----------------------------------------------------------
    T_sat( 0.0045 MPa) =   31.01 C   (ref   31.0)
    T_sat( 0.7200 MPa) =  166.09 C   (ref  165.9)
    T_sat( 3.5000 MPa) =  242.56 C   (ref  242.6)
    T_sat(13.9000 MPa) =  336.10 C   (ref  336.1)
    h_f ( 0.7200 MPa) =   702.12 kJ/kg (ref   702.12)
    h_f (13.9000 MPa) =  1566.95 kJ/kg (ref  1566.95)
    h_f ( 0.0045 MPa) =   129.98 kJ/kg (ref   129.98)
    h_f ( 3.5000 MPa) =  1049.78 kJ/kg (ref  1049.78)
    T(14.456 MPa, 718.7 kJ/kg) =  170.0 C   [compressed liquid; v7 gave 8.6 C]
    cond-pump dh (exact isentropic) =  1.930 kJ/kg  (eta=0.8)
    FW-pump   dh (exact isentropic) = 18.515 kJ/kg  (eta=0.82)
  ----------------------------------------------------------

  [1/6] Newton-Krylov solve with each preconditioner

  --------------------------------------------------------------------------
   None        ASPIN=False  tol=1e-10  DOF=50
     k       ||F||2     ||F/d||2      ||dx||     lam  GMRES
  --------------------------------------------------------------------------
     0   8.9074e+06   9.0454e-01   1.065e+05  1.0000     42
     1   1.4817e+04   1.8617e-03   4.111e+02  1.0000     55
     2   7.4706e-01   7.7034e-08   1.813e-02  1.0000     50
   ✓ converged k=3  ||F/d||=3.161e-14  (0.13s)

  --------------------------------------------------------------------------
   Jacobi      ASPIN=False  tol=1e-10  DOF=50
     k       ||F||2     ||F/d||2      ||dx||     lam  GMRES
  --------------------------------------------------------------------------
     0   8.9074e+06   9.0454e-01   1.060e+05  1.0000      8
     1   2.7029e+05   5.3680e-02   1.823e+03  1.0000      8
     2   1.5831e+02   1.5903e-05   1.741e+00  1.0000      9
   ✓ converged k=3  ||F/d||=5.027e-11  (0.11s)

  --------------------------------------------------------------------------
   FieldSplit  ASPIN=False  tol=1e-10  DOF=50
     k       ||F||2     ||F/d||2      ||dx||     lam  GMRES
  --------------------------------------------------------------------------
     0   8.9074e+06   9.0454e-01   1.065e+05  1.0000      1
     1   1.5991e+06   7.9972e-02   4.111e+02  1.0000      1
     2   7.8796e+04   3.9398e-03   2.242e-02  1.0000      2
     3   1.8303e+02   9.1513e-06   3.017e-05  1.0000      1
   ✓ converged k=4  ||F/d||=1.504e-13  (0.13s)

  --------------------------------------------------------------------------
   ASPIN       ASPIN=True  tol=1e-10  DOF=50
     k       ||F||2     ||F/d||2      ||dx||     lam  GMRES
  --------------------------------------------------------------------------
     0   8.6795e+06   8.6795e-01   1.062e+05  1.0000      1
     1   1.5492e+06   7.7478e-02   4.111e+02  1.0000      1
     2   7.7626e+04   3.8813e-03   2.231e-02  1.0000      2
     3   1.8134e+02   9.0671e-06   2.987e-05  1.0000      1
   ✓ converged k=4  ||F/d||=1.386e-11  (0.13s)
  ------------------------------------------------------------------------
  preconditioner  conv  Newton   GMRES   rate p  wall [s]
  None            True       3     147     1.55     0.128
  Jacobi          True       3      25     2.13     0.105
  FieldSplit      True       4       5     1.88     0.126
  ASPIN           True       4       5     1.88     0.128
  ------------------------------------------------------------------------
  Krylov-cost reduction, FieldSplit vs unpreconditioned: 29.4x

==============================================================================
  CONVERGED CYCLE STATE  -  HTR-PM SECONDARY LOOP (2 x 250 MWth)
==============================================================================
   # node              m [kg/s]   P [MPa]  h [kJ/kg]    T [C]  phase
  --------------------------------------------------------------------------
   1 SG outlet           186.00    13.900     3634.4    579.7  superheat
   2 HP1 extraction       20.46     7.500     3415.5    474.1  superheat
   3 HP2 outlet          165.54     3.500     3202.0    391.0  superheat
   4 LP1 extraction       14.57     1.800     3042.0    307.1  superheat
   5 LP2 extraction       12.23     0.900     2897.2    230.6  superheat
   6 LP3 extraction       10.13     0.400     2752.4    150.4  superheat
   7 LP inlet            161.90     3.500     3202.0    391.0  superheat
   8 LP exhaust          124.97     0.004     2178.8     31.0  x = 0.844
   9 Cond pump out       161.90     1.542      131.9     31.5  subcooled
  10 LPH-3 tube out      161.90     1.542      266.3     63.9  subcooled
  11 LPH-2 tube out      161.90     1.542      429.1    103.0  subcooled
  12 LPH-1 tube out      161.90     1.542      571.2    135.5  subcooled
  13 Deaerator out       186.00     0.720      702.1    166.1  sat. liquid  (x = 0.000)
  14 FW pump out         186.00    14.456      720.6    170.4  subcooled
  15 HPH out/SG in       186.00    13.900      954.1    222.4  subcooled
  --------------------------------------------------------------------------
  HP turbine work               71.57 MW
  LP turbine work              138.49 MW
  Gross mechanical power       210.06 MW
  Pump work (cond + FW)          3.76 MW   (0.31 + 3.44)
  NET ELECTRICAL POWER         206.30 MW
  SG thermal duty              498.53 MW
  NET CYCLE EFFICIENCY          41.38 %
  --------------------------------------------------------------------------
  SG zone lengths [m]   PRE  22.00 | EVA  16.00 | SUP  42.00   (L = 80)
  Helium path [C]       in  750.0 ->  564.6 ->  364.3 -> out  250.0
==============================================================================

  Normalised state vs. Energies (2023) Table 2
  ----------------------------------------------------------
  quantity                    v8   reference     dev [%]
  LP exhaust  m/m0        0.6719      0.6714        0.07
  LP exhaust  h/h0        0.5995      0.6398       -6.30
  Cond pump   m/m0        0.8704      0.8706       -0.02
  Cond pump   p/p0        0.1109      0.1109        0.00
  FW pump     m/m0        1.0000      1.0000       -0.00
  FW pump     p/p0        1.0400      1.0040        3.59
  FW pump     h/h0        0.1983      0.1775       11.71
  ----------------------------------------------------------

  [2/6] Spectral analysis
  ----------------------------------------------------------
    raw J        kappa = 5.4018e+11   max|Im(lambda)| = 4.25e+00
    scaled J     kappa = 4.3869e+09   max|Im(lambda)| = 4.25e+00
    Jacobi       kappa = 1.1878e+05   max|Im(lambda)| = 3.98e-01
    FieldSplit   kappa = 1.1876e+05   max|Im(lambda)| = 8.13e-17
    conditioning gain (raw -> FieldSplit): 4.548e+06x
  ----------------------------------------------------------

  [3/6] Field-block conditioning
    §M mass        kappa = 2.305e+01
    §P pressure    kappa = 5.411e+00
    §H enthalpy    kappa = 9.948e+02
    §SG gen.       kappa = 1.273e+07

  [4/6] Robustness study (50 random starts x 3 bands)
  band [0.35, 2.8]  (moderate)
    None        50/50   Newton  5.48   GMRES   228.8
    Jacobi      50/50   Newton  6.16   GMRES    67.7
    FieldSplit  50/50   Newton  5.26   GMRES     8.0
    ASPIN       49/50   Newton  4.96   GMRES     9.4
  band [0.15, 6.0]  (severe)
    None        50/50   Newton  5.92   GMRES   248.1
    Jacobi      50/50   Newton  6.30   GMRES    79.4
    FieldSplit  50/50   Newton  5.50   GMRES    12.7
    ASPIN       49/50   Newton  5.61   GMRES    11.3
  band [0.05, 15.0]  (extreme)
    None        49/50   Newton  6.41   GMRES   290.3
    Jacobi      49/50   Newton  6.55   GMRES    81.7
    FieldSplit  50/50   Newton  5.98   GMRES    21.1
    ASPIN       50/50   Newton  6.36   GMRES    14.3

  [5/6] Sensitivity study (field perturbations)
    m    solved 10/10  avg steps 4.70
    p    solved 10/10  avg steps 3.90
    h    solved 10/10  avg steps 4.40
    all  solved 10/10  avg steps 4.80
    correction-factor sweep: 9/9 solved, mean 4.44 Newton steps

  [6/6] Rendering figures
    figs/fig01_convergence.png
    figs/fig02_spectra.png
    figs/fig03_conditioning.png
    figs/fig04_Ts_diagram.png
    figs/fig05_SG_profile.png
    figs/fig06_robustness.png
    figs/fig07_sensitivity.png
    figs/fig08_correction_factor.png
    figs/fig09_normalised_state.png
    figs/fig10_Th_diagram.png
    figs/fig11_normalised_process.png

  done.
