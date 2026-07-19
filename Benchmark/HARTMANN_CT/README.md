# Resistive CT Hartmann validation

This benchmark exercises the full explicit resistive-MHD CT path on CUDA.
The channel is periodic in x and z and has no-slip, electrically insulating
walls at y = +/-H. The default parameters are H = 1, mu = eta = 0.1, and
B0 = 0.3, giving Ha = B0 H / sqrt(mu eta) = 3.

Generate the compact 4 x 64 x 4 mesh and run 5000 RK3 steps from the project
root:

```powershell
julia --startup-file=no --project=. Benchmark/HARTMANN_CT/gen_mesh.jl
julia --startup-file=no --project=. Benchmark/HARTMANN_CT/run.jl
```

The measured bulk velocity is used in the analytic solution, so the test
compares profile shape independently of the forcing controller's small
compressible offset. The run fails unless the default gates are met:

- relative velocity L2 error <= 0.08;
- relative induced-Bx L2 error <= 0.15;
- normalized face-divB Linf <= 1e-11.

`hartmann_ct_profile.dat` contains y, numerical/analytic u, and
numerical/analytic Bx. `hartmann_ct_summary.dat` contains all error norms.
The step count, time step, and gates can be overridden with
`HARTMANN_CT_STEPS`, `HARTMANN_CT_DT`, `HARTMANN_CT_MAX_U_L2`,
`HARTMANN_CT_MAX_B_L2`, and `HARTMANN_CT_MAX_DIVB`.
Set `HARTMANN_CT_INTEGRATOR=rkl2_strang` to run the same wall-bounded
acceptance case with the second-order RKL2-Strang resistive integrator.

The three-grid acceptance run uses `Ny = 32, 64, 128` and requires monotone
errors with an observed order of at least 1.2:

```powershell
julia --startup-file=no --project=. Benchmark/HARTMANN_CT/run_convergence.jl
```
