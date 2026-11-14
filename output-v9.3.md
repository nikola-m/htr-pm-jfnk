==============================================================================
  HTR-PM SECONDARY-CYCLE JFNK SOLVER  --  v9.3
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
    T(14.456 MPa, 718.7 kJ/kg) =  168.12 C  (IF97 R1 inversion; ref 168.12)
    T(13.9 MPa, 3634.4 kJ/kg)  =  616.20 C  (IF97 R2 inversion; ref 616.20)
    T(7.5 MPa, 3415.5 kJ/kg)   =  504.15 C  (IF97 R2 inversion; ref 504.15)
    cond-pump dh (exact isentropic) =  1.930 kJ/kg  (eta=0.8)
    FW-pump   dh (exact isentropic) = 18.515 kJ/kg  (eta=0.82)
  ----------------------------------------------------------

  [1/6] Newton-Krylov solve with each preconditioner

  --------------------------------------------------------------------------
   None        ASPIN=False  tol=1e-10  DOF=50
     k       ||F||2     ||F/d||2      ||dx||     lam  GMRES
  --------------------------------------------------------------------------
     0   2.2049e+07   2.1781e+00   2.611e+05  1.0000     42
     1   7.1352e+04   9.9693e-03   2.301e+03  1.0000     76
     2   8.3898e+00   4.8036e-06   9.571e-01  1.0000     38
   ✓ converged k=3  ||F/d||=1.634e-11  (0.43s)

  --------------------------------------------------------------------------
   Jacobi      ASPIN=False  tol=1e-10  DOF=50
     k       ||F||2     ||F/d||2      ||dx||     lam  GMRES
  --------------------------------------------------------------------------
     0   2.2049e+07   2.1781e+00   2.597e+05  1.0000      8
     1   6.7684e+05   1.3399e-01   4.746e+03  1.0000     14
     2   6.8678e+01   7.5034e-06   1.087e+00  1.0000     16
     3   5.9579e-03   6.0315e-10   6.872e-05  1.0000      9
   ✓ converged k=4  ||F/d||=3.864e-14  (0.21s)

  --------------------------------------------------------------------------
   FieldSplit  ASPIN=False  tol=1e-10  DOF=50
     k       ||F||2     ||F/d||2      ||dx||     lam  GMRES
  --------------------------------------------------------------------------
     0   2.2049e+07   2.1781e+00   2.611e+05  1.0000      1
     1   4.0721e+06   2.0382e-01   2.301e+03  1.0000      1
     2   4.7124e+05   2.3562e-02   9.609e-01  1.0000      1
     3   2.3577e+01   1.1789e-06   3.877e-06  1.0000      1
   ✓ converged k=4  ||F/d||=1.492e-11  (0.20s)

  --------------------------------------------------------------------------
   ASPIN       ASPIN=True  tol=1e-10  DOF=50
     k       ||F||2     ||F/d||2      ||dx||     lam  GMRES
  --------------------------------------------------------------------------
     0   2.1692e+07   2.1692e+00   2.611e+05  1.0000      1
     1   3.7590e+06   1.8818e-01   2.301e+03  1.0000      1
     2   4.5397e+05   2.2698e-02   9.607e-01  1.0000      1
     3   2.2726e+01   1.1363e-06   3.731e-06  1.0000      1
   ✓ converged k=4  ||F/d||=1.452e-11  (0.10s)
  ------------------------------------------------------------------------
  preconditioner  conv  Newton   GMRES   rate p  wall [s]
  None            True       3     156     1.58     0.429
  Jacobi          True       4      47     1.83     0.211
  FieldSplit      True       4       4     1.28     0.197
  ASPIN           True       4       4     1.21     0.103
  ------------------------------------------------------------------------
  Krylov-cost reduction, FieldSplit vs unpreconditioned: 39.0x

==============================================================================
  CONVERGED CYCLE STATE  -  HTR-PM SECONDARY LOOP (2 x 250 MWth)
==============================================================================
   # node              m [kg/s]   P [MPa]  h [kJ/kg]    T [C]  phase
  --------------------------------------------------------------------------
   0 SG outlet           186.00    13.900     3636.1    616.9  superheat
   1 HP1 extraction       20.46     7.500     3431.2    510.6  superheat
   2 HP2 outlet          165.54     3.500     3210.7    394.7  superheat
   3 LP1 extraction       14.57     1.800     3051.4    309.4  superheat
   4 LP2 extraction       12.23     0.900     2906.2    231.5  superheat
   5 LP3 extraction       10.13     0.400     2760.4    153.4  superheat
   6 LP inlet            161.90     3.500     3210.7    394.7  superheat
   7 LP exhaust          124.97     0.004     2184.8     31.0  x = 0.847
   8 Cond pump out       161.90     1.542      131.9     31.1  subcooled
   9 LPH-3 tube out      161.90     1.542      266.8     63.4  subcooled
  10 LPH-2 tube out      161.90     1.542      430.2    102.4  subcooled
  11 LPH-1 tube out      161.90     1.542      571.1    135.6  subcooled
  12 Deaerator out       186.00     0.720      702.1    166.1  sat. liquid  (x = 0.000)
  13 FW pump out         186.00    14.456      720.6    168.6  subcooled
  14 HPH out/SG in       186.00    13.900      955.9    221.9  subcooled
  --------------------------------------------------------------------------
  HP turbine work               70.42 MW
  LP turbine work              138.81 MW
  Gross mechanical power       209.23 MW
  Pump work (cond + FW)          3.76 MW   (0.31 + 3.44)
  NET ELECTRICAL POWER         205.48 MW
  SG thermal duty              498.53 MW
  NET CYCLE EFFICIENCY          41.22 %
  --------------------------------------------------------------------------
  SG zone lengths [m]   PRE  22.00 | EVA  16.00 | SUP  42.00   (L = 80)
  Helium path [C]       in  750.0 ->  564.3 ->  364.0 -> out  250.0
==============================================================================

  Normalised state vs. Energies (2023) Table 2
  ----------------------------------------------------------
  quantity               present   reference     dev [%]
  LP exhaust  m/m0        0.6719      0.6714        0.07
  LP exhaust  h/h0        0.6009      0.6398       -6.09
  Cond pump   m/m0        0.8704      0.8706       -0.02
  Cond pump   p/p0        0.1109      0.1109        0.00
  FW pump     m/m0        1.0000      1.0000       -0.00
  FW pump     p/p0        1.0400      1.0040        3.59
  FW pump     h/h0        0.1982      0.1775       11.65
  ----------------------------------------------------------


  [2/6] Spectral analysis
  ----------------------------------------------------------
    raw J        kappa = 5.3097e+11   max|Im(lambda)| = 4.26e+00
    scaled J     kappa = 4.3898e+09   max|Im(lambda)| = 4.26e+00
    Jacobi       kappa = 1.1885e+05   max|Im(lambda)| = 3.98e-01
    FieldSplit   kappa = 1.1883e+05   max|Im(lambda)| = 1.48e-16
    conditioning gain (raw -> FieldSplit): 4.468e+06x
  ----------------------------------------------------------

  [3/6] Field-block conditioning
    §M mass        kappa = 2.305e+01
    §P pressure    kappa = 5.411e+00
    §H enthalpy    kappa = 9.813e+02
    §SG gen.       kappa = 1.274e+07

  [4/6] Robustness study (50 random starts x 3 bands)
  band [0.35, 2.8]  (moderate)
    None        50/50   Newton  5.34   GMRES   213.9
    Jacobi      50/50   Newton  6.10   GMRES    69.3
    FieldSplit  50/50   Newton  5.26   GMRES     8.1
    ASPIN       50/50   Newton  4.86   GMRES    10.2
  band [0.15, 6.0]  (severe)
    None        50/50   Newton  5.82   GMRES   239.9
    Jacobi      50/50   Newton  6.38   GMRES    73.6
    FieldSplit  50/50   Newton  5.42   GMRES    12.7
    ASPIN       49/50   Newton  5.20   GMRES    11.4
  band [0.05, 15.0]  (extreme)
    None        49/50   Newton  6.22   GMRES   275.1
    Jacobi      49/50   Newton  6.51   GMRES    79.2
    FieldSplit  49/50   Newton  5.80   GMRES    13.0
    ASPIN       49/50   Newton  5.55   GMRES    11.9

  [5/6] Sensitivity study (field perturbations)
    m    solved 10/10  avg steps 4.40
    p    solved 10/10  avg steps 4.20
    h    solved 10/10  avg steps 4.10
    all  solved 10/10  avg steps 4.80
    correction-factor sweep: 9/9 solved, mean 4.33 Newton steps

  [6/6] Rendering figures
    figs/fig01_convergence.pdf
    figs/fig02_spectra.pdf
    figs/fig03_conditioning.pdf
    figs/fig04_Ts_diagram.png
    figs/fig05_SG_profile.png
    figs/fig06_robustness.png
    figs/fig07_sensitivity.png
    figs/fig08_correction_factor.png
    figs/fig09_normalised_state.png
    figs/fig10_Th_diagram.png
    figs/fig11_normalised_process.png

  done.

