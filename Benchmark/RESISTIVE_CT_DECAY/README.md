# Resistive CT magnetic decay

This periodic end-to-end test initializes the force-free Beltrami field

```text
B = (0, A sin(kx), A cos(kx)),  k = 2 pi / Lx.
```

The exact resistive solution is `A(t) = A(0) exp(-eta k^2 t)`.  The test
checks the projected amplitude, conversion of magnetic energy into internal
energy, global total-energy conservation, and face-flux divergence.

Run from the project root:

```powershell
julia --startup-file=no --project=. Benchmark/RESISTIVE_CT_DECAY/gen_mesh.jl
julia --startup-file=no --project=. Benchmark/RESISTIVE_CT_DECAY/run.jl
```

To exercise the damped Chebyshev STS path above the explicit resistive time
step, keep the same physical final time while selecting `sts`, for example:

```powershell
$env:RESISTIVE_CT_DECAY_INTEGRATOR = "sts"
$env:RESISTIVE_CT_DECAY_DT = "0.02"
$env:RESISTIVE_CT_DECAY_STEPS = "5"
julia --startup-file=no --project=. Benchmark/RESISTIVE_CT_DECAY/run.jl
```

The STS run applies the same amplitude, total-energy, heating-closure, and
face-divergence acceptance gates as the explicit run.

The second-order RKL2-Strang path is selected with
`RESISTIVE_CT_DECAY_INTEGRATOR=rkl2_strang`. Its end-to-end temporal
self-convergence test uses 10, 20, and 40 macro steps against a 160-step
reference on the same mesh:

```powershell
julia --startup-file=no --project=. Benchmark/RESISTIVE_CT_DECAY/run_rkl2_temporal_convergence.jl
```

The resistive RKL2 stages reuse the same high-order face-flux operator as the
coupled explicit path while suppressing fluid viscosity and heat conduction
inside the split resistive half-steps.

The same `dt=0.02`, 5-step setup shown above for STS also exercises RKL2
above the explicit resistive limit when the integrator value is changed to
`rkl2_strang`.
