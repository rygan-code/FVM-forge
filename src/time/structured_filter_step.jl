function _structured_filter_face_is_physical(face_bc,block_id,face_id)
    boundary_type=get(face_bc,(block_id,face_id),BC_INTERBLOCK)
    return boundary_type!=BC_INTERBLOCK && boundary_type!=BC_PERIODIC
end

function structured_filter_limits(direction::Integer,block,face_bc,
                                  block_processes,rank_coordinate)
    extent=(block.Nx,block.Ny,block.Nz)[direction]
    lower=Int32(NG+1)
    upper=Int32(extent+NG)
    lower_face=2*direction-1
    upper_face=2*direction
    if rank_coordinate==0 &&
       _structured_filter_face_is_physical(face_bc,block.id,lower_face)
        lower+=Int32(4)
    end
    if rank_coordinate==block_processes[direction]-1 &&
       _structured_filter_face_is_physical(face_bc,block.id,upper_face)
        upper-=Int32(4)
    end
    return lower,upper
end
