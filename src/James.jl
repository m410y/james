module James

include("Parse.jl")
using .Parse
import Glob: glob
using TOML
using LinearAlgebra
using Rotations
using StaticArrays

struct CrystalOrient
    UB::Matrix{Real} # 3x3
end

struct BeamSpectrum
    mean::Real
    deviation::Real
    wl::Vector{Vector{Real}}
end

struct GoniometerAngles
    θ::Real
    ω::Real
    ϕ::Real
end

struct RotationAxis
    pos::Vector{Real} # 3
    dir::Vector{Real} # 3
end

struct GoniometerAxes
    θ::RotationAxis
    ω::RotationAxis
    ϕ::RotationAxis
end

struct DetectorGeometry
    pos::Vector{Real} # 3
    dir::Matrix{Real} # 3x2
end

struct BeamDirection
    pos::Vector{Real} # 3
    dir::Vector{Real} # 3
end

struct SamplePosition
    pos::Vector{Real} # 3
end

struct FacilityGeometry
    angles::GoniometerAngles
    axes::GoniometerAxes
    detector::DetectorGeometry
    beam::BeamDirection
    sample::SamplePosition
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

function experiment_collect(folder::AbstractString)::Vector{ExperimentData}
    expers_files = experiment_from_sfrm.(glob("*.sfrm", folder))
    expers_to_sum = Dict{Tuple, Vector{ExperimentData}}()
    for ed in expers_files
        key = (ed.angles, ed.distance, ed.range, ed.temperature)
        haskey(expers_to_sum, key) || (expers_to_sum[key] = ExperimentData[])
        push!(expers_to_sum[key], ed)
    end
    return experiment_sum.(values(expers_to_sum))
end

function crystal_orient_from_p4p(filename::AbstractString)::CrystalOrient
    header = p4p_header(filename)
    ort1 = parse_number.(header["ORT1"])
    ort2 =  parse_number.(header["ORT2"])
    ort3 = parse_number.(header["ORT3"])
    return CrystalOrient(transpose([ort1 ort2 ort3]))
end

function facility_geometry_from_init(filename::AbstractString)::FacilityGeometry
    init = TOML.parsefile(filename)
    angles = GoniometerAngles(
        init["angles"]["theta"],
        init["angles"]["omega"],
        init["angles"]["phi"]
    )
    axes = GoniometerAxes(
        RotationAxis(init["axes"]["theta_pos"], init["axes"]["theta_dir"]),
        RotationAxis(init["axes"]["omega_pos"], init["axes"]["omega_dir"]),
        RotationAxis(init["axes"]["phi_pos"], init["axes"]["phi_dir"])
    )
    detector = DetectorGeometry(
        init["detector"]["pos"],
        [init["detector"]["dirX"] init["detector"]["dirY"]]
    )
    beam = BeamDirection(
        init["beam"]["pos"],
        init["beam"]["dir"]
    )
    sample = SamplePosition(
        init["sample"]["pos"]
    )
    return FacilityGeometry(angles, axes, detector, beam, sample)
end

function beam_spectrum_from_file(filename::AbstractString)::BeamSpectrum
    init = TOML.parsefile(filename)
    # TODO: make different spectra
    wl = [init["Mo"]["Ka1"] Init["Mo"]["Ka2"]]
    wl_mean = sum(I -> I[1]*I[2], wl) / sum(I -> I[2], wl)
    wl_dev = 3 * abs(wl[2] - wl[1]) # 3 ~ 3σ, idk
    return BeamSpectrum(wl_mean, wl_dev, wl)
end

# TODO: more complex linear detector area movement analysis
function hkl_from_area(fg::FacilityGeometry, area::ReflexArea, orient::CrystalOrient, spec::BeamSpectrum)::Union{Vector{Integer}, Nothing}
    d_0 = fg.detector.pos + fg.detector.dir * area.center
    d = AngleAxis(2 * fg.angles.θ, fg.axes.θ ...) * d_0
    n = normalize(d - fg.sample.pos)
    q = (n - fg.beam.dir) / spec.mean
    s = AngleAxis(-fg.angles.ϕ, fg.axes.ϕ ...) * AngleAxis(-fg.angles.ω, fg.axes.ω ...) * q
    hkl = inv(orient.UB) * s
    hkl_int = round.(hkl, RoundNearest)
    s_int = orient.UB * hkl_int
    iszero(hkl) && return nothing
    return norm(s - s_int)/norm(s_int) > spec.deviation / spec.mean ? nothing : hlk_int
end

function area_from_hkl(fg::FacilityGeometry, hkl::Vector{Integer}, orient::CrystalOrient, spec::BeamSpectrum)::ReflexArea
    k = fg.beam.dir / spec.mean
    s = orient.UB * hkl
    q = AngleAxis(fg.angles.ω, fg.axes.ω ...) * AngleAxis(fg.angles.ϕ, fg.axes.ϕ ...) * s
    n = AngleAxis(-2 * fg.angles.θ, fg.axes.θ ...) * normalize(k + q)
    c = AngleAxis(-2 * fg.angles.θ, fg.axes.θ ...) * fg.sample.pos
    # diffracted beam intersection with detector
    d_diff = fg.detector.pos - c
    d_inc = d_diff - n * det([d_diff, fg.detector.dir]) / det([n, fg.detector.dir])
    center = d_inc / fg.detector.dir # julia linalg magic
    # TODO: use linear approximation to eval reflex area
    
end

end # module
