import os
import sys
import numpy as np
import h5py

# Add current directory to path to import local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from group_actions import ButterflyRotator, apply_group_action, translate_fields

def run_verification():
    print("=" * 60)
    print("  Flame3D Compressible ECS Group Action Verification")
    print("=" * 60)
    
    # 1. Paths and CLI Arguments
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    chk_prefix_arg = sys.argv[1] if len(sys.argv) > 1 else "CHK/chk-0"
    mesh_dir_arg = sys.argv[2] if len(sys.argv) > 2 else "MESH"
    Lx_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 7.5
    
    # Resolve relative to project root if they are relative
    chk_prefix = chk_prefix_arg if os.path.isabs(chk_prefix_arg) else os.path.join(project_root, chk_prefix_arg)
    mesh_dir = mesh_dir_arg if os.path.isabs(mesh_dir_arg) else os.path.join(project_root, mesh_dir_arg)
    Lx = Lx_arg
    
    if not os.path.exists(mesh_dir):
        print(f"[Error] Mesh directory not found: {mesh_dir}")
        return
        
    mesh_paths = []
    bid = 0
    while True:
        path = os.path.join(mesh_dir, f"mesh_b{bid}.h5")
        if os.path.exists(path):
            mesh_paths.append(path)
            bid += 1
        else:
            break
    Nblocks = len(mesh_paths)
    print(f"[*] Detected {Nblocks} mesh blocks in {mesh_dir}.")
    print(f"[*] Lx domain length set to: {Lx}")
    
    # Check if checkpoint exists
    checkpoint_exists = True
    for bid in range(Nblocks):
        if not os.path.exists(f"{chk_prefix}-b{bid}.h5"):
            checkpoint_exists = False
            break
            
    if not checkpoint_exists:
        print(f"[Error] Initial checkpoint not found: {chk_prefix}-b*.h5")
        return
        
    print(f"[*] Loading initial state from checkpoint: {chk_prefix}...")
    Q_blocks = []
    for bid in range(Nblocks):
        filename = f"{chk_prefix}-b{bid}.h5"
        with h5py.File(filename, 'r') as f:
            Q_blocks.append(f['Q'][:]) # Shape is (Nprim, Nz, Ny, Nx) in Python/row-major
            
    # Print loaded shape
    # Wait, our group_actions.py expects Q_blocks to have shape (Nx, Ny, Nz, Nprim)
    # let's transpose the loaded state from row-major to column-major (Nx, Ny, Nz, Nprim)
    Q_blocks_transposed = []
    for bid in range(Nblocks):
        # Transpose (Nprim, Nz, Ny, Nx) -> (Nx, Ny, Nz, Nprim)
        # in numpy transpose: axes=(3, 2, 1, 0)
        q_trans = np.transpose(Q_blocks[bid], (3, 2, 1, 0))
        Q_blocks_transposed.append(q_trans)
        print(f"    Block {bid}: shape={q_trans.shape}")
        
    # 2. Initialize Rotator
    print("\n[*] Initializing ButterflyRotator...")
    rotator = ButterflyRotator(mesh_paths, NG=4)
    
    # 3. Test Identity Rotation (phi = 2 * pi)
    print("\n[*] Test 1: Identity Rotation (phi = 2*pi)")
    phi_test = 2.0 * np.pi
    Q_rot = rotator.rotate_fields(Q_blocks_transposed, phi_test)
    
    l2_errors_rot = []
    for bid in range(Nblocks):
        err = np.linalg.norm(Q_rot[bid] - Q_blocks_transposed[bid]) / np.linalg.norm(Q_blocks_transposed[bid])
        l2_errors_rot.append(err)
        print(f"    Block {bid} L2 relative error: {err:.4e}")
    
    # 4. Test Velocity Vector Rotation consistency (phi, then -phi)
    print("\n[*] Test 2: Double Rotation Consistency (phi = pi/4, then -pi/4)")
    phi_1 = np.pi / 4.0
    Q_rot_1 = rotator.rotate_fields(Q_blocks_transposed, phi_1)
    Q_rot_inv = rotator.rotate_fields(Q_rot_1, -phi_1)
    
    l2_errors_double = []
    for bid in range(Nblocks):
        err = np.linalg.norm(Q_rot_inv[bid] - Q_blocks_transposed[bid]) / np.linalg.norm(Q_blocks_transposed[bid])
        l2_errors_double.append(err)
        print(f"    Block {bid} L2 reconstruction error: {err:.4e}")
        
    # 5. Test Streamwise Translation (s = 0.5 * Lx, then -0.5 * Lx)
    print(f"\n[*] Test 3: Streamwise Translation Consistency (s = {0.5 * Lx}, then {-0.5 * Lx} with Lx = {Lx})")
    s_test = 0.5 * Lx
    Q_trans_1 = translate_fields(Q_blocks_transposed, s_test, Lx)
    Q_trans_inv = translate_fields(Q_trans_1, -s_test, Lx)
    
    l2_errors_trans = []
    for bid in range(Nblocks):
        err = np.linalg.norm(Q_trans_inv[bid] - Q_blocks_transposed[bid]) / np.linalg.norm(Q_blocks_transposed[bid])
        l2_errors_trans.append(err)
        print(f"    Block {bid} L2 translation reconstruction error: {err:.4e}")
        
    # Summary of verification
    print("\n" + "=" * 60)
    print("  Verification Summary:")
    print("=" * 60)
    max_rot_err = max(l2_errors_double)
    max_trans_err = max(l2_errors_trans)
    
    print(f"  Max Rotation Reconstruction Error (pi/4): {max_rot_err:.4e}")
    print(f"  Max Translation Reconstruction Error (FFT): {max_trans_err:.4e}")
    
    if max_rot_err < 5e-2 and max_trans_err < 1e-6:
        print("\n  >>> STATUS: VERIFICATION SUCCESSFUL! <<<")
        print("  - Streamwise FFT translation is highly accurate (accounting for Nyquist limits).")
        print("  - Azimuthal rotation has acceptable interpolation errors on butterfly grid.")
    else:
        print("\n  >>> STATUS: VERIFICATION FAILED! Please check grid interpolation metrics. <<<")

if __name__ == "__main__":
    run_verification()
