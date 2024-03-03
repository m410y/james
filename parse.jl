using PyCall
using StaticArrays
using Rotations
using CrystalInfoFramework, FilePaths

fabio = pyimport("fabio")

function sfrm_image(filename::String)::Matrix{Int32}
    return fabio.open(filename).data[end:-1:1, :]
end

function sfrm_meta(filename::String)::Dict
    meta = Dict{String, Float64}()
    header = fabio.openheader(filename).header

    angles = rem2pi.(deg2rad.(parse.(Float64, split(header["ANGLES"]))), RoundNearest)
    meta["tth"] = angles[1]
    meta["omega"] = angles[2] + 0.5*deg2rad(parse(Float64, split(header["RANGE"])[1]))
    meta["phi"] = angles[3]
    meta["chi"] = angles[4]
    meta["d"] = 10*parse(Float64, split(header["DISTANC"])[2])
    meta["cols"] = parse(Float64, split(header["NCOLS"])[1])
    meta["rows"] = parse(Float64, split(header["NROWS"])[1])
    meta["xc"] = parse(Float64, split(header["CENTER"])[1])
    meta["yc"] = parse(Float64, split(header["CENTER"])[2])

    return meta
end

function p4p_orient(filename::String)::Tuple{Float64, RotMatrix3{Float64}}
    a = 0.0
    ort = zeros(3, 3)
    row = 1
    for line in eachline(open(filename))
        if startswith(line, "CELL ")
           a = parse(Float64, split(line)[2]) 
        elseif startswith(line, "ORT")
            ort[row, :] += parse.(Float64, split(line)[2:end])
            row += 1
            row > 3 && break
        end
    end

    return (a, nearest_rotation(ort))
end

function parse_sym(str::String)::SVector{3, Tuple{Float64, Int, Float64}}
    re = r"(-)?([xyz])([+-])?(\d)?\/?(\d)?"
    sym = Vector{Tuple{Float64, Int, Float64}}()
    for ex in split(str, ", ")
        m = match(re, ex)
        sign = isnothing(m[1]) ? 1.0 : -1.0
        perm = Dict("x"=>1, "y"=>2, "z"=>3)[m[2]]
        if isnothing(m[3])
            push!(sym, (sign, perm, 0.0))
        else
            shift_sign = first(m[3]) != '-' ? 1.0 : -1.0
            shift_num = isnothing(m[4]) ? 0.0 : parse(Float64, m[4])
            shift_den = isnothing(m[5]) ? 1.0 : parse(Float64, m[5])
            push!(sym, (sign, perm, shift_sign*shift_num/shift_den))
        end
    end
    return SVector{3, Tuple{Float64, Int, Float64}}(sym)
end

function apply_sym(sym::SVector{3, Tuple{Float64, Int, Float64}}, vec::SVector{3, Float64})::SVector{3, Float64}
    return SVector{3, Float64}([op[1]*vec[op[2]]+op[3] for op in sym])
end

function cif_cell(filename::String)::Vector{Tuple{Tuple{String, Float64}, SVector{3, Float64}}}
    data = first(Cif(Path(filename))).second
    
    symops = parse_sym.(data["_space_group_symop_operation_xyz"])
    atoms = data["_atom_site_type_symbol"]

    numparse(str::String)::Float64 = parse(Float64, first(split(str, "(")))
    occupancy = numparse.(data["_atom_site_occupancy"])
    xyz = numparse.(hcat(data["_atom_site_fract_x"], data["_atom_site_fract_y"], data["_atom_site_fract_z"]))

    cell = Vector{Tuple{Tuple{String, Float64}, SVector{3, Float64}}}()
    for (atom, occup, coord) in zip(atoms, occupancy, eachrow(xyz))
        positions = Set{SVector{3, Float64}}()
        for sym in symops; push!(positions, apply_sym(sym, SVector{3, Float64}(coord))); end
        for pos in positions; push!(cell, ((atom, occup), pos)); end
    end
    return cell
end