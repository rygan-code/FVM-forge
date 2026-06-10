# Pipe Flow Artifact Detection Design

## Overview

Create a 128×44×44 pipe mesh with **full-field artifact diagnostics**, run locally and on cluster in parallel. Detect numerical artifacts (>0.5%) in velocity, pressure, temperature, and density across the entire flow field after at least 2000 steps.

## Requirements

- **Grid**: 128×44×44 (128 streamwise, 44×44 cross-section), 5-block butterfly topology
- **Pipe length**: Lx = 2.0 (shorter than production 7.5)
- **Artifact threshold**: 0.5% for all flow variables
- **Detection scope**: Full-field (every cell), not just interfaces
- **Execution**: Parallel (local debugging + cluster production)
- **Steps**: At least 2000 steps for artifact assessment

## Architecture

### Components

1. **Mesh Generator** (`debug/gen_pipe_test_mesh.jl`)
   - 128×44×44 grid with 5-block butterfly topology
   - Shorter pipe length (Lx = 2.0)
   - Outputs to `debug/MESH_PIPE_TEST/`
   - Reuses existing `Utils/gen_butterfly_fvm.jl` infrastructure

2. **Test Runner** (`debug/run_pipe_artifact_test.jl`)
   - Based on existing `debug/run_pipe_debug.jl`
   - Full-field artifact detection (not just interfaces)
   - Checkerboard diagnostic at configurable intervals
   - Comprehensive artifact reporting

3. **Diagnostic Framework** (enhanced)
   - Full-field checkerboard detection: Scan entire domain for 2Δx oscillations
   - Interface-specific checks: E_2Δx at block interfaces
   - Solution boundedness: Velocity, pressure, temperature across entire field
   - Gradient smoothness: Detect non-physical jumps anywhere
   - NaN/Inf detection: Global check

### Data Flow

```
Local Path:
gen_pipe_test_mesh.jl → MESH_PIPE_TEST/ → run_pipe_artifact_test.jl → results

Cluster Path:
Upload mesh + code → sbatch sub.sh → Monitor → Download PLT → Analyze
```

## Full-Field Artifact Detection

### Detection Methods

1. **Full-Field Checkerboard Energy**:
   ```julia
   For each cell (i,j,k):
     E_2Δx = |Q(i,j,k) - 2*Q(i+1,j,k) + Q(i+2,j,k)| / |Q(i,j,k)|
   ```
   - Scan all interior cells
   - Report maximum E_2Δx and its location
   - Flag if > 0.5% anywhere

2. **Gradient Smoothness Check**:
   ```julia
   For each cell:
     ΔQ_x = |Q(i+1) - Q(i)|
     ΔQ_y = |Q(j+1) - Q(j)|
     ΔQ_z = |Q(k+1) - Q(k)|
   ```
   - Detect non-physical jumps
   - Compare to local magnitude

3. **Solution Boundedness**:
   - Check min/max of ρ, u, v, w, p, T across entire domain
   - Flag if outside physical bounds

4. **Interface Amplification Factor**:
   - Compare artifact magnitude at interfaces vs interior
   - Determine if interfaces amplify artifacts

### Reporting Format

```
Full-Field Artifact Detection Report
=====================================
Step: 2000, Time: 0.012345
Mesh: debug/MESH_PIPE_TEST (128×44×44, 5 blocks)

── Full-Field Checkerboard Energy ──
  Max E_2Δx: 2.34e-05 at (Block 1, i=45, j=12, k=8)
  Field average: 1.23e-06
  Interface average: 1.45e-05
  Interior average: 8.90e-07
  Amplification factor: 16.3× (interface vs interior)

── Solution Bounds ──
  rho: [0.987, 1.013] (PASS)
  u:   [-1.23, 1.45] (PASS)
  v:   [-0.89, 0.92] (PASS)
  w:   [-0.78, 0.81] (PASS)
  p:   [98.5, 101.5] (PASS)
  T:   [297.0, 303.0] (PASS)

── Interface-Specific Checks ──
  B0_f3↔B1_f5: E_2Δx = 1.23e-05 (PASS)
  B1_f5↔B0_f3: E_2Δx = 1.34e-05 (PASS)
  ...

── Gradient Smoothness ──
  Max |Δρ/ρ|: 0.12% at (Block 2, i=67, j=22, k=15)
  Max |Δp/p|: 0.08% at (Block 0, i=12, j=5, k=30)

✓ PASS: All artifacts < 0.5%
```

## Implementation Plan

### Phase 1: Mesh Generation (Local)
- Create `debug/gen_pipe_test_mesh.jl`
- Generate 128×44×44 mesh
- Validate mesh quality

### Phase 2: Test Runner (Local + Cluster Parallel)
- Create `debug/run_pipe_artifact_test.jl`
- Implement full-field diagnostics
- Run locally for validation (100-500 steps)

### Phase 3: Cluster Deployment
- Upload mesh and code to cancon
- Run 2000+ steps
- Download results

### Phase 4: Analysis and Fix (if needed)
- Analyze artifact patterns
- Identify root cause
- Implement fixes
- Re-test

## Success Criteria

- All E_2Δx < 0.5% at block interfaces
- Solution remains bounded (physical values)
- No NaN/Inf through 2000 steps
- Full-field artifacts < 0.5%

## Failure Handling

If artifacts are detected:
1. Report location, magnitude, interface
2. Save diagnostic data for post-processing
3. Propose detailed fix plan for user review
4. Implement approved fixes
5. Re-test

## Dependencies

- Existing butterfly mesh generator (`Utils/gen_butterfly_fvm.jl`)
- Existing debug infrastructure (`debug/run_pipe_debug.jl`)
- Checkerboard diagnostic framework
- Remote cluster access (cancon.hpccube.com)
