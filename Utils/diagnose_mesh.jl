using HDF5, Printf

mesh_dir = "MESH"
nblocks = 5

println("="^72)
println("  Butterfly Grid Cell Size Diagnostic")
println("="^72)

γ=1.4; Rg=287.0; Tw=307.0; Ma=0.8; Re=17000.0; Pr=0.71
C_s=1.458e-6; T_s=110.4; R0=0.5

c_wall = sqrt(γ*Rg*Tw)
u_bulk = Ma*c_wall
μ_w = C_s*Tw*sqrt(Tw)/(Tw+T_s)
ρ_ref = Re*μ_w/(u_bulk*2.0*R0)

@printf("  c_wall=%.2f  u_bulk=%.2f  μ_w=%.4e  ρ_ref=%.6f\n\n", c_wall, u_bulk, μ_w, ρ_ref)

global_min_dh = Inf
global_min_info = nothing

for bid in 0:nblocks-1
    fname = joinpath(mesh_dir, "mesh_b$(bid).h5")
    if !isfile(fname)
        println("  Warning: $fname not found, skipping")
        continue
    end
    
    try
        h5open(fname, "r") do file
            # Inspect structure first
            ks = keys(file)
            
            local coords
            if "coords" in ks
                coords = read(file["coords"])  # (3, Nx+1, Ny+1, Nz+1)
            elseif "x" in ks
                x = read(file["x"]); y = read(file["y"]); z = read(file["z"])
                # Stack into (3, ...)
                coords = cat(reshape(x, 1, size(x)...), 
                            reshape(y, 1, size(y)...), 
                            reshape(z, 1, size(z)...), dims=1)
            else
                println("  Block $bid: unknown format, keys=$ks")
                return
            end
            
            ndim = ndims(coords)
            sz = size(coords)
            @printf("  Block %d: coords shape = %s\n", bid, string(sz))
            
            # coords is (3, Nx+1, Ny+1, Nz+1)
            nx = sz[2]-1; ny = sz[3]-1; nz = sz[4]-1
            @printf("  Block %d: cells = %d×%d×%d\n", bid, nx, ny, nz)
            
            min_dh = Inf; min_dx_b=Inf; min_dy_b=Inf; min_dz_b=Inf
            min_loc = (1,1,1)
            
            # Track top-10 smallest cells
            worst_cells = Tuple{Float64, Int, Int, Int, Float64, Float64, Float64}[]
            
            for k in 1:nz, j in 1:ny, i in 1:nx
                # ξ-direction average edge length
                dx = 0.0
                for dj in 0:1, dk in 0:1
                    dx += sqrt((coords[1,i+1,j+dj,k+dk]-coords[1,i,j+dj,k+dk])^2 +
                               (coords[2,i+1,j+dj,k+dk]-coords[2,i,j+dj,k+dk])^2 +
                               (coords[3,i+1,j+dj,k+dk]-coords[3,i,j+dj,k+dk])^2)
                end
                dx *= 0.25
                
                # η-direction
                dy = 0.0
                for di in 0:1, dk in 0:1
                    dy += sqrt((coords[1,i+di,j+1,k+dk]-coords[1,i+di,j,k+dk])^2 +
                               (coords[2,i+di,j+1,k+dk]-coords[2,i+di,j,k+dk])^2 +
                               (coords[3,i+di,j+1,k+dk]-coords[3,i+di,j,k+dk])^2)
                end
                dy *= 0.25
                
                # ζ-direction
                dz = 0.0
                for di in 0:1, dj in 0:1
                    dz += sqrt((coords[1,i+di,j+dj,k+1]-coords[1,i+di,j+dj,k])^2 +
                               (coords[2,i+di,j+dj,k+1]-coords[2,i+di,j+dj,k])^2 +
                               (coords[3,i+di,j+dj,k+1]-coords[3,i+di,j+dj,k])^2)
                end
                dz *= 0.25
                
                dh = min(dx, dy, dz)
                if dh < min_dh
                    min_dh = dh; min_dx_b=dx; min_dy_b=dy; min_dz_b=dz
                    min_loc = (i,j,k)
                end
                
                # Track smallest cells
                if length(worst_cells) < 5 || dh < worst_cells[end][1]
                    push!(worst_cells, (dh, i, j, k, dx, dy, dz))
                    sort!(worst_cells, by=x->x[1])
                    if length(worst_cells) > 5
                        pop!(worst_cells)
                    end
                end
            end
            
            i,j,k = min_loc
            xc=0.0; yc=0.0; zc=0.0
            for di in 0:1, dj in 0:1, dk in 0:1
                xc += coords[1,i+di,j+dj,k+dk]
                yc += coords[2,i+di,j+dj,k+dk]
                zc += coords[3,i+di,j+dj,k+dk]
            end
            xc/=8; yc/=8; zc/=8
            rc = sqrt(yc^2+zc^2)
            
            a_max = u_bulk+c_wall
            dt_conv = 0.5*min_dh/a_max
            dt_diff = 0.25*ρ_ref*min_dh^2*Pr/μ_w
            
            @printf("    Min at (%d,%d,%d) -> (y=%.5f, z=%.5f, r=%.5f, r/R0=%.4f)\n", i,j,k,yc,zc,rc,rc/R0)
            @printf("    dx=%.4e  dy=%.4e  dz=%.4e  dh=%.4e\n", min_dx_b, min_dy_b, min_dz_b, min_dh)
            @printf("    dt_conv=%.4e  dt_diff=%.4e  dt_eff=%.4e  (%s limited)\n",
                    dt_conv, dt_diff, min(dt_conv,dt_diff)*0.5,
                    dt_diff < dt_conv ? "VISCOUS" : "convective")
            
            # Print top-5 smallest cells
            println("    Top-5 smallest cells:")
            for (idx, (dh_v, ci, cj, ck, cdx, cdy, cdz)) in enumerate(worst_cells)
                cx=0.0; cy=0.0; cz=0.0
                for di in 0:1, dj in 0:1, dk in 0:1
                    cx += coords[1,ci+di,cj+dj,ck+dk]
                    cy += coords[2,ci+di,cj+dj,ck+dk]
                    cz += coords[3,ci+di,cj+dj,ck+dk]
                end
                cx/=8; cy/=8; cz/=8
                cr = sqrt(cy^2+cz^2)
                @printf("      %d. (%3d,%3d,%3d) dh=%.4e r/R0=%.4f dx=%.2e dy=%.2e dz=%.2e\n",
                        idx, ci, cj, ck, dh_v, cr/R0, cdx, cdy, cdz)
            end
            println()
            
            if min_dh < global_min_dh
                global_min_dh = min_dh
                global_min_info = (bid, min_loc, min_dx_b, min_dy_b, min_dz_b, dt_conv, dt_diff, rc)
            end
        end
    catch e
        @printf("  Block %d: ERROR - %s\n\n", bid, string(e))
    end
end

if global_min_info !== nothing
    bid, loc, gdx, gdy, gdz, gdt_c, gdt_d, grc = global_min_info
    println("="^72)
    println("  GLOBAL MINIMUM")
    println("="^72)
    @printf("  Block %d  Cell (%d,%d,%d)  r/R0=%.4f\n", bid, loc..., grc/R0)
    @printf("  dh_min  = %.4e\n", global_min_dh)
    @printf("  dx/dy/dz= %.4e / %.4e / %.4e\n", gdx, gdy, gdz)
    @printf("  dt_conv = %.4e\n", gdt_c)
    @printf("  dt_diff = %.4e  (%s)\n", gdt_d, gdt_d < gdt_c ? "VISCOUS limited" : "convective limited")
    @printf("  dt_eff  = %.4e  (CFL=0.5)\n\n", min(gdt_c, gdt_d)*0.5)
    
    dx_x = 10.0/768
    dt_ref = 0.5*dx_x/(u_bulk+c_wall)
    @printf("  Streamwise ref: dx=%.4e  dt=%.4e\n", dx_x, dt_ref)
    @printf("  Slowdown: %.0fx smaller than streamwise CFL\n", dt_ref/(min(gdt_c,gdt_d)*0.5))
    @printf("\n  Actual simulation dt ≈ 4.76e-8, ratio = %.1fx\n", min(gdt_c,gdt_d)*0.5 / 4.76e-8)
end
println("="^72)
