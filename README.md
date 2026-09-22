# HTR-PM-JFNK
Jacobian-free Newton-Krylov solver for the secondary cycle of the HTR-PM power plant

We present a mathematical framework and a novel numerical solution architecture for the steady-state secondary circuit of the HTR-PM power plant, and use it to evaluate physics-based preconditioning for strongly coupled thermo-hydraulic networks.

The system is solved by a Jacobian-free Newton-Krylov (JFNK) method with an Eisenstat-Walker forcing term and Armijo backtracking. 
Two preconditioners are developed and compared: a Field-Split preconditioner formed by inverting four physical field blocks exactly and an Additive Schwarz Preconditioned Inexact Newton (ASPIN) strategy that performs sub-domain Newton sweeps over six physics-motivated subdomains.
Both preconditioners reduce the global Krylov cost compared to unpreconditioned and point Jacobi (diagonal) precodnitioner.
