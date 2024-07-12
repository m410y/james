module James
export experiment_collect_from_folder, facility_state_from_init

include("Parse.jl")

using .Parse
using TOML
import Glob: glob
using LinearAlgebra
using Rotations
using StaticArrays

struct RotationAxis
    angle::Real
    pos::Vector{Real} # 3
    dir::Vector{Real} # 3
end

struct BeamSpectrum
    material::AbstractString
    mean::Real
    deviation::Real
    wl::Vector{Vector{Real}}
end

struct BeamGeometry
    pos::Vector{Real} # 3
    dir::Vector{Real} # 3
end

struct CrystalGeometry
    axes::Vector{RotationAxis} # 2 for D8 VENTURE
    pos::Vector{Real} # 3
    orient::Matrix{Real} # 3x3
end

struct DetectorGeometry
    axis::RotationAxis
    pos::Vector{Real} # 3
    dir::Matrix{Real} # 3x2
end

struct FacilityState
    spectrum::BeamSpectrum
    beam::BeamGeometry
    crystal::CrystalGeometry
    detector::DetectorGeometry
end

struct ExperimentData
    time::Real
    range::Real
    angles::Vector{Real}
    distance::Real
    temperature::Real
    image::Matrix{Real}
end

struct ReflexArea
    center::Vector{Real} # 2
    diag::Vector{Real} # 2
end


parse_number(str::AbstractString)::Float64 =
    parse(Float64, str)
parse_angle(str::AbstractString)::Float64 =
    rem2pi(deg2rad(parse(Float64, str)), RoundNearest)
lazy_iterate(state::FacilityState) =
    state.spectrum, state.beam, state.crystal, state.detector
axis_rotation(axis::RotationAxis) =
    AngleAxis(axis.angle, axis.dir ...)

function experiment_from_sfrm(filename::AbstractString)::ExperimentData
    header, image = sfrm_full(filename)
    time = parse_number(header["CUMULAT"][1])
    range = parse_angle(header["RANGE"][1])
    angles = parse_angle.(header["ANGLES"])
    distance = 10 * parse_number(header["DISTANC"][2]) # converting to mm
    temperature = parse_number(header["LOWTEMP"][5])
    return ExperimentData(time, range, angles, distance, temperature, image)
end

function experiment_sum(ed_vec::Vector{ExperimentData})::ExperimentData
    time = sum(ed -> ed.time, ed_vec)
    image = sum(ed -> ed.image, ed_vec)
    ed = first(ed_vec) # any of these
    return ExperimentData(time, ed.range, ed.angles, ed.distance, ed.temperature, image)
end

function experiment_collect_from_folder(folder::AbstractString)::Vector{ExperimentData}
    expers_files = experiment_from_sfrm.(glob("*.sfrm", folder))
    expers_to_sum = Dict{Tuple, Vector{ExperimentData}}()
    for ed in expers_files
        key = (ed.angles, ed.distance, ed.range, ed.temperature)
        haskey(expers_to_sum, key) || (expers_to_sum[key] = ExperimentData[])
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

function crystal_orient_from_p4p(filename::AbstractString)::Matrix{Real}
    header = p4p_header(filename)
    ort1 = parse_number.(header["ORT1"])
    ort2 =  parse_number.(header["ORT2"])
    ort3 = parse_number.(header["ORT3"])
    return transpose([ort1 ort2 ort3])
end

function facility_state_from_init(filename::AbstractString)::FacilityState
    init = TOML.parsefile(filename)
    beam_init, crystal_init, detector_init = init["beam"], init["crystal"], init["detector"]
    spec = beam_spectrum_from_file(beam_init["spectrum"])
    beam = BeamGeometry(
        beam_init["pos"],
        beam_init["dir"]
    )
    crystal = CrystalGeometry(
        [RotationAxis(crystal_init["phi"], crystal_init["phi_pos"], crystal_init["phi_dir"]),
        RotationAxis(crystal_init["omega"], crystal_init["omega_pos"], crystal_init["omega_dir"])],
        crystal_init["pos"],
        crystal_orient_from_p4p(crystal_init["orient"])
    )
    detector = DetectorGeometry(
        RotationAxis(detector_init["theta"], detector_init["theta_pos"], detector_init["theta_dir"]),
        detector_init["pos"],
        [detector_init["dirX"] detector_init["dirY"]]
    )
    return FacilityState(spec, beam, crystal, detector)
end

# TODO: more complex linear detector area movement analysis
function hkl_from_area(state::FacilityState, area::ReflexArea)::Union{Vector{Int64}, Nothing}
    spectrum, beam, crystal, detector = lazy_iterate(state)
    d_0 = detector.pos + detector.dir * area.center
    d = axis_rotation(detector.axis) * d_0
    n = normalize(d - crystal.pos)
    q = (n - beam.dir) / spectrum.mean
    s = inv(axis_rotation(crystal.axes[2]) * axis_rotation(crystal.axes[1])) * q
    hkl = inv(crystal.orient) * s
    hkl_int = round.(hkl, RoundNearest)
    return hkl_int
end

function area_from_hkl(state::FacilityState, hkl::Vector{Int64})::Union{ReflexArea, Nothing}
    spectrum, beam, crystal, detector = lazy_iterate(state)
    k = beam.dir / spectrum.mean
    s = crystal.orient * hkl
    q = axis_rotation(crystal.axes[2]) * axis_rotation(crystal.axes[1]) * s
    n = inv(axis_rotation(detector.axis)) * normalize(k + q)
    c = inv(axis_rotation(detector.axis)) * crystal.pos
    # diffracted beam intersection with detector
    d_diff = detector.pos - c
    d_inc = d_diff - n * det([d_diff detector.dir]) / det([n detector.dir])
    center = (d_inc' / detector.dir')' # julia linalg magic
    # TODO: use linear approximation to eval reflex area
    return ReflexArea(center, [50, 50]) # diag is placeholder
end

end # module
