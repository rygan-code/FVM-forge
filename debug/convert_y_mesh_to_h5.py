#!/usr/bin/env python3
"""
convert_y_mesh_to_h5.py — Convert y_mesh VTS files to HDF5 format

Reads the 3-block y_mesh from VTS files and creates:
  - mesh_b0.h5, mesh_b1.h5, mesh_b2.h5 (node coordinates)
  - block_connectivity.h5 (connectivity and BC info)

The y_mesh is a simple 3-block Y-junction:
  Block 0: center block
  Block 1: right branch
  Block 2: left branch
"""

import re
import numpy as np
import h5py
import os

NG = 4  # Ghost cell layers

def read_vts_coords(filename):
    """Read coordinates from VTS file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find extent
    extent_line = [l for l in lines if 'WholeExtent' in l][0]
    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', extent_line)
    nx_nodes = int(m.group(2)) + 1
    ny_nodes = int(m.group(4)) + 1
    nz_nodes = int(m.group(6)) + 1

    # Find coordinate data
    data_start = None
    for i, line in enumerate(lines):
        if 'format="ascii"' in line:
            data_start = i + 1
            break

    x = np.zeros((nx_nodes, ny_nodes, nz_nodes))
    y = np.zeros((nx_nodes, ny_nodes, nz_nodes))
    z = np.zeros((nx_nodes, ny_nodes, nz_nodes))

    idx = data_start
    for k in range(nz_nodes):
        for j in range(ny_nodes):
            for i in range(nx_nodes):
                vals = lines[idx].strip().split()
                x[i, j, k] = float(vals[0])
                y[i, j, k] = float(vals[1])
                z[i, j, k] = float(vals[2])
                idx += 1

    return x, y, z, nx_nodes - 1, ny_nodes - 1, nz_nodes - 1

def convert_y_mesh():
    input_dir = "debug/output"
    output_dir = "debug/output"

    print("Converting y_mesh VTS to HDF5...")
    print(f"  Input:  {input_dir}/y_mesh_b{{0,1,2}}.vts")
    print(f"  Output: {output_dir}/")

    # Read all blocks
    blocks = []
    for bid in range(3):
        vts_file = os.path.join(input_dir, f"y_mesh_b{bid}.vts")
        print(f"  Reading {vts_file}...")
        x, y, z, nx, ny, nz = read_vts_coords(vts_file)
        blocks.append({'x': x, 'y': y, 'z': z, 'nx': nx, 'ny': ny, 'nz': nz})
        print(f"    Block {bid}: {nx}×{ny}×{nz} cells")

    # Write mesh files
    for bid, blk in enumerate(blocks):
        mesh_file = os.path.join(output_dir, f"mesh_b{bid}.h5")
        print(f"  Writing {mesh_file}...")
        with h5py.File(mesh_file, 'w') as f:
            f.create_dataset('x', data=blk['x'])
            f.create_dataset('y', data=blk['y'])
            f.create_dataset('z', data=blk['z'])
            f.create_dataset('NG', data=NG)
            f.create_dataset('Nx', data=blk['nx'])
            f.create_dataset('Ny', data=blk['ny'])
            f.create_dataset('Nz', data=blk['nz'])

    # Create connectivity file
    # For the y_mesh, we need to define:
    # - Which faces are connected to which
    # - BC types for each face
    #
    # Assuming Y-junction topology:
    #   Block 0 face 2 (ξ+) connects to block 1 face 1 (ξ-)
    #   Block 0 face 4 (η+) connects to block 2 face 3 (η-)
    #   Other faces are walls

    Nblocks = 3
    Nx_b = [blk['nx'] for blk in blocks]
    Ny_b = [blk['ny'] for blk in blocks]
    Nz_b = [blk['nz'] for blk in blocks]

    # BC types: 0=interblock, 1=isothermal_wall, 2=periodic, etc.
    # Initialize all faces as isothermal wall
    face_bc = np.ones((Nblocks, 6), dtype=np.int32)

    # Define inter-block connections
    # Block 0 face 2 (ξ+) <-> Block 1 face 1 (ξ-)
    face_bc[0, 1] = 0  # BC_INTERBLOCK (0-indexed: face 2 = index 1)
    face_bc[1, 0] = 0  # BC_INTERBLOCK

    # Block 0 face 4 (η+) <-> Block 2 face 3 (η-)
    face_bc[0, 3] = 0  # BC_INTERBLOCK
    face_bc[2, 2] = 0  # BC_INTERBLOCK

    # BC parameters (wall temperature, etc.)
    bc_params = np.zeros((Nblocks, 6, 20), dtype=np.float32)
    for bid in range(Nblocks):
        for fid in range(6):
            if face_bc[bid, fid] == 1:  # isothermal_wall
                bc_params[bid, fid, 0] = 300.0  # Tw = 300K

    # Connectivity array format: [dst_block, dst_face, src_block, src_face]
    # Each row is one connection
    connectivity = np.array([
        [1, 2, 2, 1],  # Block 0 face 2 <-> Block 1 face 1
        [2, 1, 1, 2],  # Reverse connection
        [1, 4, 3, 3],  # Block 0 face 4 <-> Block 2 face 3
        [3, 3, 1, 4],  # Reverse connection
    ], dtype=np.int64)

    # reverse_tan: 0 = no reversal, 1 = reverse tangential direction
    reverse_tan = np.zeros(len(connectivity), dtype=np.int64)

    # flip_normal: 0 = no flip, 1 = flip normal direction
    flip_normal = np.zeros(len(connectivity), dtype=np.int64)

    # Write connectivity file
    conn_file = os.path.join(output_dir, "block_connectivity.h5")
    print(f"  Writing {conn_file}...")
    with h5py.File(conn_file, 'w') as f:
        f.create_dataset('Nblocks', data=Nblocks)
        f.create_dataset('Nx_b', data=Nx_b)
        f.create_dataset('Ny_b', data=Ny_b)
        f.create_dataset('Nz_b', data=Nz_b)

        # Write connectivity
        f.create_dataset('connectivity', data=connectivity)

        # Write reverse_tan and flip_normal
        f.create_dataset('reverse_tan', data=reverse_tan)
        f.create_dataset('flip_normal', data=flip_normal)

        # Write face_bc
        f.create_dataset('face_bc', data=face_bc)

        # Write bc_params
        f.create_dataset('bc_params', data=bc_params)

    print("\nConversion complete!")
    print("  Files created:")
    for bid in range(3):
        print(f"    mesh_b{bid}.h5")
    print("    block_connectivity.h5")

if __name__ == "__main__":
    convert_y_mesh()
