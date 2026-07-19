# =============================================================================
# newton_krylov_solver.py — Matrix-Free Newton-Krylov-Hookstep RPO Solver
# Coordinates GPU FVM time-stepper and performs Newton-GMRES iterations
# =============================================================================

import os
import sys
import subprocess
import numpy as np
import h5py
from scipy.sparse.linalg import LinearOperator, gmres
from group_actions import ButterflyRotator, apply_group_action

class CompressibleRPOSolver:
    def __init__(self, run_script=None, mesh_dir=None, num_ranks=4, Lx=7.5, NG=4, temp_dir=None):
        # Dynamically determine project root relative to this script's location
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(os.path.dirname(self.script_dir))
        
        self.run_script = run_script if run_script is not None else os.path.join(self.project_root, "run_pipe.jl")
        self.mesh_dir = mesh_dir if mesh_dir is not None else os.path.join(self.project_root, "MESH")
        self.temp_dir = temp_dir if temp_dir is not None else os.path.join(self.project_root, "tmp_rpo")
        
        self.num_ranks = num_ranks
        self.Lx = Lx
        self.NG = NG
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        # Get connectivity and mesh file paths
        mesh_paths = []
        bid = 0
        while True:
            path = os.path.join(self.mesh_dir, f"mesh_b{bid}.h5")
            if os.path.exists(path):
                mesh_paths.append(path)
                bid += 1
            else:
                break
        self.Nblocks = len(mesh_paths)
        print(f"[RPO Solver] Detected {self.Nblocks} mesh blocks in {self.mesh_dir}.")
        
        # Initialize the butterfly rotator
        self.rotator = ButterflyRotator(mesh_paths, NG=NG)
        
        self.in_prefix = os.path.join(self.temp_dir, "rpo_in")
        self.out_prefix = os.path.join(self.temp_dir, "rpo_out")
        
    def load_state(self, path_prefix):
        """
        Load 5 blocks of primitives Q[Nx, Ny, Nz, Nprim] from HDF5
        """
        Q_blocks = []
        for bid in range(self.Nblocks):
            filename = f"{path_prefix}-b{bid}.h5"
            with h5py.File(filename, 'r') as f:
                Q_blocks.append(f['Q'][:])
        return Q_blocks

    def save_state(self, path_prefix, Q_blocks):
        """
        Save Q_blocks back to HDF5 files
        """
        for bid in range(self.Nblocks):
            filename = f"{path_prefix}-b{bid}.h5"
            # Overwrite if exists
            with h5py.File(filename, 'w') as f:
                f.create_dataset('Q', data=Q_blocks[bid])

    def run_fvm_solver(self, T):
        """
        Call FVM solver as a subprocess to integrate for time T.
        Runs inside project_root to ensure Julia includes solver files correctly.
        """
        cmd = [
            "mpirun", "-np", str(self.num_ranks),
            "julia", self.run_script,
            f"--init_rpo={self.in_prefix}",
            f"--save_rpo={self.out_prefix}",
            f"--run_time={T}",
            f"--mesh_dir={self.mesh_dir}"
        ]
        # Run and check output, setting cwd to project_root
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=self.project_root)
        if res.returncode != 0:
            print("[RPO Solver] Error: Bottom solver failed!")
            print("=== Solver STDOUT ===")
            print(res.stdout)
            print("=== Solver STDERR ===")
            print(res.stderr)
            sys.exit(1)
            
    def compute_residual(self, Q_blocks, T, phi, s):
        """
        Residual F(u0) = g(-phi, -s)*u(T) - u0
        """
        # 1. Save state as input to the FVM solver
        self.save_state(self.in_prefix, Q_blocks)
        
        # 2. Run solver for time T
        self.run_fvm_solver(T)
        
        # 3. Load final integrated state
        Q_integrated = self.load_state(self.out_prefix)
        
        # 4. Apply shift back: g(-phi, -s)
        Q_shifted = apply_group_action(Q_integrated, -phi, -s, self.Lx, self.rotator)
        
        # 5. Compute residual: Q_shifted - Q_blocks
        residual = [Q_shifted[bid] - Q_blocks[bid] for bid in range(self.Nblocks)]
        return residual

    def flatten_state(self, Q_blocks):
        return np.concatenate([Q.flatten() for Q in Q_blocks])

    def reconstruct_state(self, Q_flat, shapes):
        Q_blocks = []
        idx = 0
        for shape in shapes:
            sz = np.prod(shape)
            Q_blocks.append(Q_flat[idx:idx+sz].reshape(shape))
            idx += sz
        return Q_blocks

    def solve(self, initial_path_prefix, T0, phi0, s0, max_iter=20, trust_radius=0.1, tol=1e-5):
        """
        Main Newton-Krylov-Hookstep solver loop
        """
        Q_blocks = self.load_state(initial_path_prefix)
        shapes = [Q.shape for Q in Q_blocks]
        
        T, phi, s = T0, phi0, s0
        
        print("\n" + "="*70)
        print(f"  Starting Newton-Krylov ECS search (ranks={self.num_ranks}, Lx={self.Lx})")
        print("="*70)
        
        for k in range(max_iter):
            # Compute current residual
            res_blocks = self.compute_residual(Q_blocks, T, phi, s)
            res_flat = self.flatten_state(res_blocks)
            res_norm = np.linalg.norm(res_flat)
            
            print(f"\nNewton Step {k} | T={T:.6f} | phi={phi:.6f} | s={s:.6f} | L2 Residual={res_norm:.6e}")
            if res_norm < tol:
                print(">>> Success: Newton convergence achieved!")
                self.save_state("ecs_converged", Q_blocks)
                with h5py.File("ecs_converged_params.h5", 'w') as f:
                    f['T'] = T
                    f['phi'] = phi
                    f['s'] = s
                break
                
            # Define Matrix-free Jacobian-Vector Product: J * dX
            # dX is a combined perturbation: [dQ_flat, dT, dphi, ds]
            Q_flat = self.flatten_state(Q_blocks)
            
            def jacobian_vector_product(dX):
                # Unpack perturbation
                dQ_flat = dX[:-3]
                dT = dX[-3]
                dphi = dX[-2]
                ds = dX[-1]
                
                # Finite difference step size
                epsilon = 1e-6
                
                # Perturb state
                Q_pert_flat = Q_flat + epsilon * dQ_flat
                Q_pert = self.reconstruct_state(Q_pert_flat, shapes)
                T_pert = T + epsilon * dT
                phi_pert = phi + epsilon * dphi
                s_pert = s + epsilon * ds
                
                # Evaluate perturbed residual
                res_pert_blocks = self.compute_residual(Q_pert, T_pert, phi_pert, s_pert)
                res_pert_flat = self.flatten_state(res_pert_blocks)
                
                # (F(X + eps*dX) - F(X)) / eps
                jvp = (res_pert_flat - res_flat) / epsilon
                return jvp

            # Solve J * dX = -res using GMRES
            n_state = len(Q_flat) + 3
            # Linear operator
            J_op = LinearOperator((len(res_flat), n_state), matvec=jacobian_vector_product)
            
            print(f"  [GMRES] Starting Krylov solver...")
            # Target RHS is -res_flat
            rhs = -res_flat
            
            # Since the matrix J is not square (n_state = N_Q + 3, but rhs is N_Q), we solve the least squares
            # system. Or we append 3 phase conditions (constraints) to make it square!
            # Phase conditions:
            # 1. Time phase condition: dot(dQ, dQ_dt) = 0
            # 2. Theta phase condition: dot(dQ, dQ_dtheta) = 0
            # 3. Z phase condition: dot(dQ, dQ_dz) = 0
            # Let's approximate by adding these constraints to GMRES by setting up a square system or solving
            # the underdetermined system directly via GMRES on the normal equations or least squares.
            # In GMRES, we can simply solve the least-squares system J * dX = -res_flat.
            # We can define a square system by fixing T, phi, s in the inner Krylov step, and then doing
            # a separate update for them, or we can solve the full system by appending 3 dummy rows to J_op.
            # For simplicity and robust convergence, we can solve the system J_Q * dQ = -res_flat for the
            # velocity perturbation first, and then update (T, phi, s) using the phase alignment.
            # This is called the "decoupled Newton step" and is highly stable.
            # Let's solve J_Q * dQ = -res_flat (where J_Q is the Jacobian w.r.t Q only, keeping T, phi, s fixed).
            
            def jacobian_Q_product(dQ_flat):
                epsilon = 1e-6
                Q_pert_flat = Q_flat + epsilon * dQ_flat
                Q_pert = self.reconstruct_state(Q_pert_flat, shapes)
                res_pert_blocks = self.compute_residual(Q_pert, T, phi, s)
                res_pert_flat = self.flatten_state(res_pert_blocks)
                return (res_pert_flat - res_flat) / epsilon

            J_Q_op = LinearOperator((len(res_flat), len(Q_flat)), matvec=jacobian_Q_product)
            
            # GMRES with restart=20, maxiter=3
            dQ_flat, info = gmres(J_Q_op, -res_flat, tol=1e-3, restart=20, maxiter=3)
            
            # Hookstep trust-region scaling
            dx_norm = np.linalg.norm(dQ_flat)
            if dx_norm > trust_radius:
                print(f"  [Hookstep] Trust region limit exceeded! Scaling step: {dx_norm:.4f} -> {trust_radius:.4f}")
                dQ_flat = (trust_radius / dx_norm) * dQ_flat
                
            # Update state Q
            Q_flat_new = Q_flat + dQ_flat
            Q_blocks = self.reconstruct_state(Q_flat_new, shapes)
            
            # Update RPO parameters (T, phi, s) using a line-search or phase minimization
            # We can perform a mini-optimization to find the best (T, phi, s) that minimizes the new residual
            print(f"  [Newton] Optimizing period T and drift parameters (phi, s)...")
            best_res = Inf = 1e10
            best_T, best_phi, best_s = T, phi, s
            
            # Search grid for line-search
            dT_grid = [-0.01 * T, 0.0, 0.01 * T]
            dphi_grid = [-0.02, 0.0, 0.02]
            ds_grid = [-0.02 * self.Lx, 0.0, 0.02 * self.Lx]
            
            for dt_val in dT_grid:
                for dp_val in dphi_grid:
                    for ds_val in ds_grid:
                        test_T = T + dt_val
                        test_phi = phi + dp_val
                        test_s = s + ds_val
                        
                        test_res_blocks = self.compute_residual(Q_blocks, test_T, test_phi, test_s)
                        test_res_norm = np.linalg.norm(self.flatten_state(test_res_blocks))
                        
                        if test_res_norm < best_res:
                            best_res = test_res_norm
                            best_T, best_phi, best_s = test_T, test_phi, test_s
                            
            T, phi, s = best_T, best_phi, best_s
            print(f"  [Newton] Parameter update completed: T={T:.6f}, phi={phi:.6f}, s={s:.6f}")

if __name__ == "__main__":
    # Example usage:
    # python newton_krylov_solver.py <initial_checkpoint_prefix> <T0_factor> <phi0> <s0> [mesh_dir_name]
    if len(sys.argv) < 5:
        print("Usage: python newton_krylov_solver.py <init_prefix> <T0_factor> <phi0> <s0> [mesh_dir_name]")
        sys.exit(1)
        
    init_prefix = sys.argv[1]
    T0_factor = float(sys.argv[2])
    phi0 = float(sys.argv[3])
    s0 = float(sys.argv[4])
    mesh_dir_name = sys.argv[5] if len(sys.argv) >= 6 else "MESH_LEN15"
    
    # Calculate physical flow-through time (T_pass) automatically
    Ma_target = 0.8
    Tw = 307.0
    gamma = 1.4
    Rg = 287.0
    Lx = 15.0
    c_sound = np.sqrt(gamma * Rg * Tw)
    U_bulk = Ma_target * c_sound
    T_pass = Lx / U_bulk
    T0 = T0_factor * T_pass
    
    print("=" * 70)
    print(f"[RPO Solver] Flow-through parameters:")
    print(f"  Main domain length Lx = {Lx} m")
    print(f"  Bulk velocity Ub      = {U_bulk:.2f} m/s")
    print(f"  Flow-through time Tp  = {T_pass:.6f} s")
    print(f"  Using T0              = {T0_factor} * Tp = {T0:.6f} s")
    print(f"  Using Mesh Directory  = {mesh_dir_name}")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    solver = CompressibleRPOSolver(
        run_script=os.path.join(project_root, "run_pipe_cebl_diffrot.jl"),
        mesh_dir=os.path.join(project_root, mesh_dir_name),
        num_ranks=4,
        Lx=Lx,
        NG=4
    )
    
    solver.solve(init_prefix, T0, phi0, s0, max_iter=15, trust_radius=0.05, tol=1e-5)
