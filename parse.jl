module Parse

using PyCall
using StaticArrays
using LinearAlgebra
using Rotations
using CrystalInfoFramework
using FilePaths


fabio = pyimport("fabio")

export sfrm_image
function sfrm_image(filename::String)::Matrix{Int32}
    return fabio.open(filename).data[end:-1:1, :]
end

export sfrm_meta
function sfrm_meta(filename::String)::Dict{String, Any}
    meta = Dict{String, Any}()
    header = fabio.openheader(filename).header

    angles = deg2rad.(parse.(Float64, split(header["ANGLES"])))
    meta["tth"] = angles[1]
    meta["omega"] = angles[2] + 0.5*deg2rad(parse(Float64, split(header["RANGE"])[1]))
    meta["phi"] = angles[3]
    meta["chi"] = angles[4]
    meta["d"] = 10*parse(Float64, split(header["DISTANC"])[2])
    meta["cols"] = parse(Float64, split(header["NCOLS"])[1])
    meta["rows"] = parse(Float64, split(header["NROWS"])[1])
    meta["xc"] = parse(Float64, split(header["CENTER"])[1])
    meta["yc"] = parse(Float64, split(header["CENTER"])[2])
    meta["sample"] = split(header["FILENAM"], r"_+")[2]
    
    return meta
end

function p4p_meta(filename::String)::Dict
    file = open(filename)
    meta = Dict{String, Vector{String}}()
    for line in eachline(file)
        data = split(line)
        startswith(data[1], "REF") && return meta
        meta[data[1]] = data[2:end]
    end
    return meta
end

function lattice_eval(a::Float64, b::Float64, c::Float64, α::Float64, β::Float64, γ::Float64)::UpperTriangular{Float64}
    ec_y = (cos(α) - cos(β)*cos(γ))/sin(γ)
    ec_z = sqrt(sin(β)^2 - ec_y^2)
    return UpperTriangular(
        [a   b*cos(γ) c*cos(β);
         0.0 b*sin(γ)   c*ec_y;
         0.0      0.0   c*ec_z])
end

export p4p_orient
function p4p_orient(filename::String)::Tuple{RotMatrix3{Float64}, UpperTriangular{Float64}}
    meta = p4p_meta(filename)
    a, b, c = parse.(Float64, meta["CELL"][1:3])
    α, β, γ = deg2rad.(parse.(Float64, meta["CELL"][4:6]))
    bravais = lowercase(meta["BRAVAIS"][1])

    if bravais == "cubic"
        A = Diagonal([a, a, a])
    elseif bravais == "tetragonal"
        A = Diagonal([a, a, c])
    elseif bravais == "orthorhombic"
        A = Diagonal([a, b, c])
    elseif bravais == "rhombohedral"
        A = lattice_eval(a, a, a, α, α, α)
    elseif bravais == "hexagonal"
        A = lattice_eval(a, a, c, pi/2, pi/2, 2pi/3)
    else
        A = lattice_eval(a, b, c, α, β, γ)
    end

    UB = zeros(3, 3)
    UB[1, :] = parse.(Float64, meta["ORT1"])
    UB[2, :] = parse.(Float64, meta["ORT2"])
    UB[3, :] = parse.(Float64, meta["ORT3"])

    return nearest_rotation(UB*A), A
end

function parse_sym(str::String)::SVector{3, Tuple{Float64, Int, Float64}}
    re_x = r"([+-]?)([xyz])"
    re_a = r"([+-]?\d)\/?(\d)?"
    sym = Vector{Tuple{Float64, Int, Float64}}()
    for ex in split(str, ",")
        m_x = match(re_x, ex)
        m_a = match(re_a, ex)
        sign = isnothing(m_x[1]) ? 1.0 : -1.0
        perm = Dict("x"=>1, "y"=>2, "z"=>3)[m_x[2]]
        if isnothing(m_a)
            shift_num = 0.0
            shift_den = 1.0
        else
            shift_num =  parse(Float64, m_a[1])
            shift_den = isnothing(m_a[2]) ? 1.0 : parse(Float64, m_a[2])
        end
        push!(sym, (sign, perm, shift_num/shift_den))

    end
    return SVector{3, Tuple{Float64, Int, Float64}}(sym)
end

function apply_sym(sym::SVector{3, Tuple{Float64, Int, Float64}}, vec::SVector{3, Float64})::SVector{3, Float64}
    return SVector{3, Float64}([op[1]*vec[op[2]]+op[3] for op in sym])
end

export cif_cell
function cif_cell(filename::String)::Vector{Tuple{Tuple{String, Float64}, SVector{3, Float64}}}
    data = first(Cif(Path(filename))).second
    
    if "_space_group_symop_operation_xyz" in keys(data)
        symops = parse_sym.(data["_space_group_symop_operation_xyz"])
    elseif "_symmetry_equiv_pos_as_xyz" in keys(data)
        symops = parse_sym.(data["_symmetry_equiv_pos_as_xyz"])
    else
        print("Cant parse symmetry\n")
        symops = nothing
    end
    
    atoms = data["_atom_site_type_symbl"]
    numparse(str::String)::Float64 = parse(Float64, first(split(str, "(")))
    occupancy = numparse.(data["_atom_site_occupancy"])
    xyz = numparse.(hcat(data["_atom_site_fract_x"], data["_atom_site_fract_y"], data["_atom_site_fract_z"]))

    cell = Vector{Tuple{Tuple{String, Float64}, SVector{3, Float64}}}()
    for (atom, occup, coord) in zip(atoms, occupancy, eachrow(xyz))
        positions = Set{SVector{3, Float64}}()
        isnothing(symops) || for sym in symops; push!(positions, apply_sym(sym, SVector{3, Float64}(coord))); end
        for pos in positions; push!(cell, ((atom, occup), pos)); end
    end
    return cell
end

end