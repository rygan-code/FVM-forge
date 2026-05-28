include("solver.jl")

# LES
const LES_smag::Bool = false       # if use Smagorinsky model
const LES_wale::Bool = false        # if use WALE model

# thermal state
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.72

# flow control
const test_case::String = "FlatPlate" # "TGV" or "ObliqueShock" or "Sod" or "FlatPlate"
const mesh::String = "Benchmark/BL/bl_mesh.h5"
const metrics::String = "Benchmark/BL/bl_metrics.h5"
const Nprocs::SVector{3, Int64} = [1,1,1] # number of GPUs
const Iperiodic = (false, false, true)   # periodic direction

const adaptive_dt::Bool = true          # Master switch: compute dt from CFL
const CFL::FT = FT(0.8e0)             #   CFL number
const LTS::Bool = false                 #   Sub-option: per-cell dt (true) or global min (false)
const dt::FT = 1e-4             # Fixed dt (used when adaptive_dt=false)
const Time::FT = 1.2             # Total simulation time
const maxStep::Int64 = 12000          # Max steps for benchmark

const plt_xdmf::Bool = true         # if use HDF5+XDMF for plt output
const plt_out::Bool = true           # if output plt file
const step_plt::Int64 = 1000           # how many steps to save plt
const plt_shuffle::Bool = true       # shuffle to make compress more efficient
const plt_compress_level::Int64 = 1  # output file compression level 0-9, 0 for no compression

const chk_out::Bool = false           # if checkpoint is made on save
const step_chk::Int64 = 1000          # how many steps to save chk
const chk_shuffle::Bool = true       # shuffle to make compress more efficient
const chk_compress_level::Int64 = 1  # checkpoint file compression level 0-9, 0 for no compression
const restart::String = "none"     # restart use checkpoint, file name "*.h5" or "none"

const average::Bool = false                 # if do average
const avg_step::Int64 = 10                  # average interval
const avg_total::Int64 = 2000               # total number of samples
const avg_shuffle::Bool = true       # shuffle to make compress more efficient
const avg_compress_level::Int64 = 1  # output file compression level 0-9, 0 for no compression

const sample::Bool = false                             # if do sampling (slice)
const sample_step::Int64 = 10                          # sampling interval
const sample_index::SVector{3, Int64} = [-1, -1, -1]  # slice index in 3 directions, -1 for no slicing

# filtering
const filtering::Bool = false              # if do filtering
const filtering_nonlinear::Bool = false    # if filtering is shock capturing
const filtering_interval::Int64 = 100      # filtering step interval
const filtering_rth::FT = FT(1e-5)        # filtering threshold for nonlinear
const filtering_s0::FT = FT(1.e0)         # filtering strength

# do not change 
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 6 # ρ u v w p T

# is viscous
const viscous::Bool = true
const viscous_order::Int64 = 6 # 2, 4 or 6

# Finite Volume Config
const eigen_reconstruction::Bool = false # if use eigen reconstruction
const character::Bool = true            # Characteristic-wise reconstruction or not
const splitMethod::String = "HLLC"       # options are: SW, LF, VL, AUSM, HLLC
const hybrid_ϕ1::FT = zero(FT)          # < ϕ1: Linear
const hybrid_ϕ2::FT = FT(1.e0)          # < ϕ2: WENO7
const hybrid_ϕ3::FT = FT(10.e0)         # < ϕ3: WENO5, else NND2
const Linear_ϕ::FT = FT(1.e0)           # dissipation control for linear scheme
const UP7::SVector{7, FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, FT} = UP7 * Linear_ϕ + CD6 * (FT(1.e0) - Linear_ϕ)

# load mesh info
const NG::Int64 = h5read(mesh, "NG")
const Nx::Int64 = h5read(mesh, "Nx")
const Ny::Int64 = h5read(mesh, "Ny")
const Nz::Int64 = h5read(mesh, "Nz")
const Nxp::Int64 = Nx ÷ Nprocs[1] # make sure it is integer
const Nyp::Int64 = Ny ÷ Nprocs[2] # make sure it is integer
const Nzp::Int64 = Nz ÷ Nprocs[3] # make sure it is integer

# GPU Kernel Config
const maxreg::Int64 = 256
const nthreads::Tuple{Int32, Int32, Int32} = (8, 4, 8)
const nblock::Tuple{Int32, Int32, Int32} = (cld((Nxp+2*NG), 8), 
                                            cld((Nyp+2*NG), 4),
                                            cld((Nzp+2*NG), 8))

# For simple kernel without register limit
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 8, 8)
const nblock2::Tuple{Int32, Int32, Int32} = (cld((Nxp+2*NG), 16), 
                                             cld((Nyp+2*NG), 8),
                                             cld((Nzp+2*NG), 8))
# Run the simulation
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nGPU = MPI.Comm_size(comm)
shmcomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
local_rank = MPI.Comm_rank(shmcomm)
comm_cart = MPI.Cart_create(comm, Nprocs; periodic=Iperiodic)
if nGPU != prod(Nprocs) && rank == 0
    error("Oops, nGPU �?$Nprocs\n")
end
# set device on each MPI rank
# device!(local_rank)

time_step(rank, comm_cart)

MPI.Finalize()

