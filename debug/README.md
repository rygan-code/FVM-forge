# Debug Diagnostics

This directory contains diagnostic scripts for investigating multi-block numerical artifacts.

## Quick Start

### Step 1: Convert y_mesh to HDF5 format

```bash
julia debug/convert_y_mesh_to_h5.jl
```

This creates:
- `debug/output/mesh_b0.h5`, `mesh_b1.h5`, `mesh_b2.h5` — node coordinates
- `debug/output/block_connectivity.h5` — connectivity and BC info

### Step 2: Run TGV verification on y_mesh

```bash
julia debug/run_y_mesh_tgv.jl 100
```

This runs 100 steps of Taylor-Green Vortex on the 3-block y_mesh and compares with analytical solution.

### Step 3: Check results

The verification script outputs:
- Max relative error in u, v, p, ρ at each time step
- PASS/FAIL verdict (threshold: 0.5%)

## Diagnostic Scripts

### Phase 1: Ghost Cell Exchange
- `dump_ghost_interface.jl` — Dumps Q values at block interfaces to verify ghost cell exchange integrity
  - `dump_ghost_interface(blocks, connectivity, world_rank, tt)` — Compare ghost vs neighbor interior
  - `dump_interface_flux(blocks, connectivity, shared_Fx, shared_Fy, shared_Fz, world_rank, tt)` — Check flux conservation

### Phase 3: Metric Consistency
- `check_metrics.jl` — Verifies face normals and areas are consistent across block interfaces
  - `check_metric_consistency(blocks, connectivity, world_rank)` — Check Area and normal vectors
  - `check_gcl(blocks, world_rank)` — Check Geometric Conservation Law (Σ n·Area = 0)

### Phase 2: Mesh Conversion
- `convert_y_mesh_to_h5.jl` — Converts y_mesh VTS files to HDF5 format for solver

### Phase 4: TGV Verification
- `run_y_mesh_tgv.jl` — Multi-block TGV verification on y_mesh with analytical solution comparison

## Classical Test Cases with Analytical Solutions

### Taylor-Green Vortex (TGV)
**Analytical Solution (incompressible):**
```
u(x,y,z) =  sin(x)cos(y)cos(z)
v(x,y,z) = -cos(x)sin(y)cos(z)
p(x,y,z) = p₀ + (1/16)(cos(2x) + cos(2y))(cos(2z) + 2)
ρ(x,y,z) = ρ₀ (constant)
```

**Verification:** Compare numerical solution with analytical at t=0. Any artifacts at block interfaces will appear as deviations from the analytical solution.

**Expected Accuracy:** For low Ma (0.1) and Re=100, errors should be < 0.5% after short time (t=0.1).

### Acoustic Pulse
**Analytical Solution:**
A pressure pulse propagates radially. The solution is known and any artifacts at block interfaces would be visible as distortions in the circular wave front.

### Sod Shock Tube
**Analytical Solution:**
Exact Riemann problem solution. Good for testing shock-capturing at block interfaces.

## Output Files

All diagnostic output goes to `debug/` directory as `.txt` files. Each file contains:
- Header with metadata (block IDs, face IDs, rank, time step)
- Variable-by-variable comparison data
- Error metrics and warnings

## Cleanup

This directory can be safely deleted after debugging is complete.

## Notes

- The y_mesh is a 3-block Y-junction (16×16×16 cells per block)
- Block 0: center block
- Block 1: right branch
- Block 2: left branch
- Ghost cell layers: NG=4
