# =============================================================================
# group_actions.py — Symmetry group actions for relative periodic orbits (RPO)
# Implements:
# 1. Streamwise translation (T_z) along x-axis using 1D FFT phase shift
# 2. Azimuthal rotation (R_theta) around x-axis using 2D Delaunay interpolation
# =============================================================================

import numpy as np
import h5py
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

class ButterflyRotator:
    """
    Handles 2D cross-section rotations on multi-block butterfly grids.
    Pre-computes the Delaunay triangulation of the cross-section to allow fast queries.
    """
    def __init__(self, mesh_paths, NG=4):
        self.mesh_paths = mesh_paths
        self.NG = NG
        self.Nblocks = len(mesh_paths)
        
        # Load cross-section coordinate slices at the first interior x-plane
        self.Y_list = []
        self.Z_list = []
        self.block_dims = []
        
        for bid in range(self.Nblocks):
            with h5py.File(mesh_paths[bid], 'r') as f:
                # Read node coordinates
                y_coord = f['y'][:]
                z_coord = f['z'][:]
                Nx = int(f['Nx'][()])
                Ny = int(f['Ny'][()])
                Nz = int(f['Nz'][()])
                
                # y_coord/z_coord have shape (Nz+1, Ny+1, Nx+1)
                # Compute cell-centered coordinates in the y-z cross-section
                # Take x-slice at 0, shape (Nz+1, Ny+1)
                y_node_2d = y_coord[:, :, 0]
                z_node_2d = z_coord[:, :, 0]
                
                # Average 4 cell corner nodes to get cell center, shape (Nz, Ny)
                y_cell = 0.25 * (
                    y_node_2d[:-1, :-1] + y_node_2d[:-1, 1:] +
                    y_node_2d[1:, :-1] + y_node_2d[1:, 1:]
                )
                z_cell = 0.25 * (
                    z_node_2d[:-1, :-1] + z_node_2d[:-1, 1:] +
                    z_node_2d[1:, :-1] + z_node_2d[1:, 1:]
                )
                
                # Transpose to shape (Ny, Nz) to match Q_blocks slice
                self.Y_list.append(y_cell.T)
                self.Z_list.append(z_cell.T)
                self.block_dims.append((Nx, Ny, Nz))
                
        # Flatten and concatenate to form a global 2D point list
        self.Y_global = np.concatenate([y.flatten() for y in self.Y_list])
        self.Z_global = np.concatenate([z.flatten() for z in self.Z_list])
        self.coords_global = np.vstack((self.Y_global, self.Z_global)).T
        
        # Build global Delaunay triangulation for fast interpolation
        self.tri = Delaunay(self.coords_global)
        
    def rotate_fields(self, Q_blocks, phi):
        """
        Rotate the multi-block primitives Q_blocks by angle phi.
        Q_blocks is a list of 4D arrays: Q[Nx, Ny, Nz, Nprim].
        phi is the rotation angle in radians.
        """
        if abs(phi) < 1e-12:
            return [np.copy(Q) for Q in Q_blocks]
            
        # 1. Compute query points (pre-rotated coordinates)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        
        Y_query =  self.Y_global * cos_p + self.Z_global * sin_p
        Z_query = -self.Y_global * sin_p + self.Z_global * cos_p
        query_points = np.vstack((Y_query, Z_query)).T
        
        # 2. Precompute Delaunay interpolation weights for all query points
        simplex = self.tri.find_simplex(query_points)
        valid = simplex >= 0
        
        # We need to construct weights only for valid points
        ndim = 2
        T = self.tri.transform[simplex[valid], :ndim, :ndim]
        r = query_points[valid] - self.tri.transform[simplex[valid], ndim]
        
        c = np.einsum('ijk,ik->ij', T, r)
        c3 = 1.0 - c.sum(axis=1)
        weights = np.column_stack((c, c3))
        v = self.tri.simplices[simplex[valid]]
        
        # Extrapolation precomputations for invalid points
        any_invalid = np.any(~valid)
        if any_invalid:
            from scipy.spatial import KDTree
            kdtree = KDTree(self.coords_global)
            _, nearest_idx = kdtree.query(query_points[~valid])
            
        # Output blocks initialized to copy
        Q_rot = [np.copy(Q) for Q in Q_blocks]
        Nprim = Q_blocks[0].shape[-1]
        Nx_total = Q_blocks[0].shape[0]
        
        for i_x in range(Nx_total):
            # Gather global 2D slice values for all variables at this plane
            slice_vals = []
            for bid in range(self.Nblocks):
                slice_vals.append(Q_blocks[bid][i_x, :, :, :]) # shape (Ny, Nz, Nprim)
            # Flatten to global cross-section, shape (Npts, Nprim)
            slice_global = np.concatenate([s.reshape(-1, Nprim) for s in slice_vals], axis=0)
            
            # Interpolate all variables simultaneously
            interpolated_global = np.empty((query_points.shape[0], Nprim))
            
            # Valid query points interpolation (vectorized across slices and variables)
            interpolated_global[valid] = np.einsum('ij,ijk->ik', weights, slice_global[v])
            
            # Invalid points extrapolation
            if any_invalid:
                interpolated_global[~valid] = slice_global[nearest_idx]
                
            # Scatter back to blocks
            idx_start = 0
            for bid in range(self.Nblocks):
                Nx, Ny, Nz = self.block_dims[bid]
                sz = Ny * Nz
                # Reshape to (Ny, Nz, Nprim)
                Q_rot[bid][i_x, :, :, :] = interpolated_global[idx_start:idx_start+sz].reshape((Ny, Nz, Nprim))
                idx_start += sz
                
            # Velocity vector transformation for (v, w) in Cartesian coordinates:
            # v (component index 2) and w (component index 3) in Q
            for bid in range(self.Nblocks):
                v_orig = np.copy(Q_rot[bid][i_x, :, :, 2])
                w_orig = np.copy(Q_rot[bid][i_x, :, :, 3])
                Q_rot[bid][i_x, :, :, 2] = v_orig * cos_p + w_orig * sin_p
                Q_rot[bid][i_x, :, :, 3] = -v_orig * sin_p + w_orig * cos_p
                
        return Q_rot

def translate_fields(Q_blocks, s, Lx):
    """
    Translate the multi-block primitives Q_blocks streamwise by distance s.
    Uses 1D FFT along the first dimension (x-direction) for spectral accuracy.
    Lx is the streamwise length of the domain.
    """
    if abs(s) < 1e-12:
        return [np.copy(Q) for Q in Q_blocks]
        
    Q_trans = []
    for Q in Q_blocks:
        Nx, Ny, Nz, Nprim = Q.shape
        # 1D FFT along axis 0
        Q_hat = np.fft.fft(Q, axis=0)
        
        # Phase shift factor: e^(-i * k * s * 2pi / Lx)
        # k wavenumbers
        k = np.fft.fftfreq(Nx).reshape((Nx, 1, 1, 1)) * Nx # actual integer wavenumbers
        phase = np.exp(-1j * k * s * (2.0 * np.pi / Lx))
        
        Q_hat_shifted = Q_hat * phase
        Q_shifted = np.real(np.fft.ifft(Q_hat_shifted, axis=0))
        Q_trans.append(Q_shifted)
        
    return Q_trans

def apply_group_action(Q_blocks, phi, s, Lx, rotator):
    """
    Apply combined rotation by phi and translation by s.
    """
    # 1. Streamwise translation (exact FFT)
    Q_trans = translate_fields(Q_blocks, s, Lx)
    # 2. Cross-section rotation (Delaunay interpolation)
    Q_rot = rotator.rotate_fields(Q_trans, phi)
    return Q_rot
