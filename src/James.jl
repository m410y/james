module James
export experiment_from_sfrm, experiments_from_folder, state_from_init, state_from_exp_update
export hkl_at_xy_near, xy_at_hkl_near

include("Parse.jl")

using .Parse
using TOML
import Glob: glob
using LinearAlgebra
using Rotations
using StaticArrays

struct Axis
    angle::Float64
    pos::SVector{3, Float64}
    dir::SVector{3, Float64}
end

struct BeamSpectrum
    material::String
    mean::Float64
    deviation::Float64
    wl::Vector{Vector{Float64}}
end

struct Beam
    spec::BeamSpectrum
    pos::SVector{3, Float64}
    dir::SVector{3, Float64}
end

struct Crystal
    omega::Axis
    chi::Axis
    phi::Axis
    pos::SVector{3, Float64}
    orient::SMatrix{3, 3, Float64}
end

struct Detector
    theta::Axis
    pos::SVector{3, Float64}
    dir::SMatrix{3, 2, Float64}
    shape::SVector{2, Int16}
end

struct State
    beam::Beam
    crystal::Crystal
    detector::Detector
end

struct Experiment
    time::Float64
    range::Float64
    angles::Vector{Float64}
    distance::Float64
    temperature::Float64
    image::Matrix{Int32}
end

parse_number(str::AbstractString)::Float64 =
    parse(Float64, str)
parse_angle(str::AbstractString)::Float64 =
    rem2pi(deg2rad(parse(Float64, str)), RoundNearest)

    # TODO: remove this shit
lazy_iterate(state::State) =
    state.beam, state.crystal, state.detector
axis_rotation(axis::Axis) =
    AngleAxis(axis.angle, axis.dir ...)
inv_axis_rotation(axis::Axis) =
    AngleAxis(-axis.angle, axis.dir ...)

function experiment_from_sfrm(filename::AbstractString)::Experiment
    header, image = sfrm_full(filename)
    time = parse_number(header["CUMULAT"][1])
    range = parse_angle(header["RANGE"][1])
    angles = parse_angle.(header["ANGLES"])
    distance = 10 * parse_number(header["DISTANC"][2]) # converting to mm
    temperature = parse_number(header["LOWTEMP"][5])
    return Experiment(time, range, angles, distance, temperature, image)
end

function experiment_sum(ed_vec::Vector{Experiment})::Experiment
    time = sum(ed -> ed.time, ed_vec)
    image = sum(ed -> ed.image, ed_vec)
    ed = first(ed_vec) # any of these
    return Experiment(time, ed.range, ed.angles, ed.distance, ed.temperature, image)
end

function experiments_from_folder(folder::AbstractString)::Vector{Experiment}
    expers_files = experiment_from_sfrm.(glob("*.sfrm", folder))
    expers_to_sum = Dict{Tuple, Vector{Experiment}}()
    for ed in expers_files
        key = (ed.angles, ed.distance, ed.range, ed.temperature)
        haskey(expers_to_sum, key) || (expers_to_sum[key] = Experiment[])
        push!(expers_to_sum[key], ed)
    end
    return experiment_sum.(values(expers_to_sum))
end

function beam_spectrum_from_file(filename::AbstractString)::BeamSpectrum
    init = TOML.parsefile(filename)
    material = init["material"]
    wl = [init["Ka1"], init["Ka2"]]
    wl_mean = sum(I -> I[1]*I[2], wl) / sum(I -> I[2], wl)
    wl_dev = 3 * abs(wl[2][1] - wl[1][1]) # 3 ~ 3σ, idk TODO: take it from config mb
    return BeamSpectrum(material, wl_mean, wl_dev, wl)
end

function crystal_orient_from_p4p(filename::AbstractString)::SMatrix{3, 3, Float64}
    header = p4p_header(filename)
    ort1 = parse_number.(header["ORT1"])
    ort2 =  parse_number.(header["ORT2"])
    ort3 = parse_number.(header["ORT3"])
    return transpose([ort1 ort2 ort3])
end

function state_from_init(filename::AbstractString)::State
    init = TOML.parsefile(filename)
    beam_init, crystal_init, detector_init = init["beam"], init["crystal"], init["detector"]
    beam = Beam(
        beam_spectrum_from_file(beam_init["spectrum"]),
        beam_init["pos"],
        beam_init["dir"]
    )
    crystal = Crystal(
        Axis(crystal_init["omega"], crystal_init["omega_pos"], crystal_init["omega_dir"]),
        Axis(crystal_init["chi"], crystal_init["chi_pos"], crystal_init["chi_dir"]),
        Axis(crystal_init["phi"], crystal_init["phi_pos"], crystal_init["phi_dir"]),
        crystal_init["pos"],
        crystal_orient_from_p4p(crystal_init["orient"])
    )
    detector = Detector(
        Axis(detector_init["theta"], detector_init["theta_pos"], detector_init["theta_dir"]),
        detector_init["pos"],
        [detector_init["dirX"] detector_init["dirY"]],
        detector_init["shape"]
    )
    return State(beam, crystal, detector)
end

function axis_from_angle_update(axis::Axis, angle::Real)::Axis
    return Axis(angle, axis.pos, axis.dir)
end

function state_from_exp_update(state_0::State, exp::Experiment)::State
    beam_0, crystal_0, detector_0 = lazy_iterate(state_0)
    crystal = Crystal(
        axis_from_angle_update(crystal_0.omega, exp.angles[2]),
        axis_from_angle_update(crystal_0.chi, exp.angles[4]),
        axis_from_angle_update(crystal_0.phi, exp.angles[3]),
        crystal_0.pos,
        crystal_0.orient
    )
    detector = Detector(
        axis_from_angle_update(detector_0.theta, exp.angles[1]),
        [exp.distance, detector_0.pos[2], detector_0.pos[3]],
        detector_0.dir,
        detector_0.shape
    )
    return State(beam_0, crystal, detector)
end

function two_vec_basis(a::AbstractVector, b::AbstractVector)::RotMatrix3
    n_a, n_b = normalize.((a, b))
    e_1 = n_a
    e_2 = normalize(n_b - n_a * dot(n_b, n_a))
    e_3 = normalize(cross(n_a, n_b))
    return [e_1 e_2 e_3]
end

function omega_at_hkl_reflect(state::State, hkl::AbstractVector)::Union{NTuple{2, Float64}, Nothing}
    beam, crystal, _ = lazy_iterate(state)
    U = two_vec_basis(crystal.omega.dir, beam.dir)
    v_0 = axis_rotation(crystal.phi) * crystal.orient * hkl * beam.spec.mean
    n = U'beam.dir
    v = U'v_0
    omega_0 = -atan(v[3], v[2])
    omega_cos = -(v'v + 2 * n[1]*v[1])/(2 * sqrt(v[2]^2 + v[3]^2) * n[2])
    abs(omega_cos) > 1 && return nothing
    omega = rem2pi.((omega_0 + acos(omega_cos), omega_0 - acos(omega_cos)), RoundNearest)
    return abs(omega[1]) < abs(omega[2]) ? omega : (omega[2], omega[1])
end

# function hkl_diff_at_xy_omega(state::State, xy::AbstractVector)::SMatrix{3, 3, Float64}
#     beam, crystal, detector = lazy_iterate(state)
#     d_0 = detector.pos + detector.dir * xy
#     R_d = axis_rotation(detector.theta)
#     d = R_d * d_0 - crystal.pos
#     n = normalize(d)
#     N = I - n*n'
#     v_xy = inv_axis_rotation(crystal.omega) * N * R_d * detector.dir / norm(d)
#     v_omega = cross(n - beam.dir, crystal.omega.dir)
#     R_inv = inv(crystal.orient) * inv_axis_rotation(crystal.phi) * inv_axis_rotation(crystal.chi)
#     diff = R_inv * [v_xy v_omega] / beam.spec.mean
#     return diff
# end

function hkl_at_xy_near(state::State, xy::AbstractVector)::SVector{3, Float64}
    beam, crystal, detector = lazy_iterate(state)
    d_0 = detector.pos + detector.dir * xy
    d = axis_rotation(detector.theta) * d_0
    n = normalize(d - crystal.pos)
    q = (n - beam.dir) / beam.spec.mean
    s = inv_axis_rotation(crystal.phi) * inv_axis_rotation(crystal.chi) * inv_axis_rotation(crystal.omega) * q
    hkl = inv(state.crystal.orient) * s
    return round.(hkl, RoundNearest)
end

function xy_at_hkl_near(state::State, hkl::AbstractVector)::Union{SVector{2, Float64}, Nothing}
    beam, crystal, detector = lazy_iterate(state)
    R = axis_rotation(crystal.omega) * axis_rotation(crystal.chi) * axis_rotation(crystal.phi)
    s = R * crystal.orient * hkl
    n = normalize(s + beam.dir / beam.spec.mean)
    n_d = inv_axis_rotation(detector.theta) * n
    m = cross(detector.dir[:, 1], detector.dir[:, 2])
    d_0 = detector.pos - crystal.pos
    d_coord = n_d * dot(d_0, m) / dot(n_d, m) - d_0
    return detector.dir \ d_coord
end

end # module
