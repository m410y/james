using StaticArrays
using LinearAlgebra
using Rotations
using TOML
using Printf
using FilePaths
using Glob
using LsqFit
using DSP
using Statistics

include("parse.jl")
using .Parse


init = TOML.parsefile("init.toml")

init = TOML.parsefile("init.toml")
U, A = p4p_orient(init["p4p_file"])
init["correct_lattice"] && (A = Diagonal([init["a"], init["a"], init["a"]]))
UB = U*inv(A)

sample_name = split(split(init["p4p_file"], "\\")[end], ".")[1]
@printf "Sample: %s\n" sample_name

wl = init["wl"]
χ = deg2rad(init["chi"])
ray = [1, 0, 0]
cols = init["cols"]
rows = init["rows"]
xc = init["xc"]
yc = init["yc"]
px_size = init["px_size"]
x_diap = init["x_lims"][1]:init["x_lims"][2]
y_diap = init["y_lims"][1]:init["y_lims"][2]
x0 = (init["x_lims"][1] + init["x_lims"][2])/2
y0 = (init["y_lims"][1] + init["y_lims"][2])/2
xy_mesh = hcat(vec([x for y=y_diap, x=x_diap]), vec([y for y=y_diap, x=x_diap]))
σ0 = init["sigma"]
threshold = init["threshold"]

hank(ang) = rad2deg(rem2pi(ang, RoundNearest))

experiments = Dict{Tuple{SVector{3, Int}, NTuple{3, Float64}}, Vector{String}}()
experiment_name = split(init["sfrm_folder"], "\\")[end]
for path in glob("*.sfrm", init["sfrm_folder"])
    meta = sfrm_meta(path)
    d, θ, ω, ϕ = meta["d"], meta["tth"]/2, meta["omega"], meta["phi"]
    hkl = round.(Int, A*U'RotZXZ(ϕ, χ, -ω)*(RotZ(2θ)*ray - ray)/wl[1])
    ex_params = hkl, (d, θ, ϕ)
    ex_params in keys(experiments) || (experiments[ex_params] = [])
    push!(experiments[ex_params], path)
end

function coords(v::SVector{3, Float64}, d::Float64, θ::Float64)::SVector{2, Float64}
    v = RotZ(-2θ)*LinearAlgebra.normalize(v)*d/px_size
    return xc - v[2], yc + v[3]
end

function gauss_2D(t::Matrix, p::Vector{Float64})::Vector{Float64}
    A, x0, y0, σx, σy = p
    x = @. (t[:, 1] - x0)/σx
    y = @. (t[:, 2] - y0)/σy
    return @. A*exp(-(x^2+y^2)/2)
end

function fit_func(t::Matrix, p::Vector{Float64})::Vector{Float64}
    return p[1] .+ gauss_2D(t, p[2:6]) + gauss_2D(t, p[7:11])
end

function xcorr_correction(image::Matrix{Float64}, param::Vector{Float64})::SVector{2, Float64}
    xcorr_range = range(round(Int, length(image)/2)+1, step=1, length = length(image))
    xcorr_data = xcorr(vec(image), fit_func(xy_mesh, param))[xcorr_range]
    xcorr_data ./= sum(xcorr_data)
    return 2(xy_mesh'xcorr_data - [x0, y0])
end

function max_correction(image::Matrix{Float64}, param::Vector{Float64})::SVector{2, Float64}
    max_ij = findmax(image)[2]
    return x_diap[max_ij[2]] - param[3], y_diap[max_ij[1]] - param[4]
end

experiment_counter = 0
file = open(joinpath(init["res_folder"], "$(experiment_name)_eval.txt"), "w")
bond_experiments = Dict{SVector{3, Int}, SVector{2, Vector{Float64}}}()
for (meta, paths) in experiments
    hkl, (d, θ, ϕ) = meta

    image = zeros(rows, cols)
    foreach(path -> image += sfrm_image(path), paths)
    image = image[y_diap, x_diap]

    noise0 = median(image)
    image[findall(image.>threshold)] .= noise0
    A0 = first(findmax(image))

    s = RotXZ(-χ, -ϕ)*UB*hkl
    true_ω(s::SVector{3, Float64}, λ::Float64, θ::Float64)::Float64 = return atan(s[2], s[1]) + sign(θ)*acos(-λ*norm(s)/2)
    peaks = [coords(ray/λ + RotZ(true_ω(s, λ, θ))*s, d, θ) for λ in wl]

    gen_param(noise, A, xy) = [noise, A, xy[1][1], xy[1][2], σ0, σ0, A/2, xy[2][1], xy[2][2], σ0, σ0]
    p0 = gen_param(noise0, A0, peaks)

    #xy_shift = xcorr_correction(image, p0)
    xy_shift = max_correction(image, p0)
    map!(peak -> peak + xy_shift, peaks, peaks)
    p0 = gen_param(noise0, A0, peaks)

    fit = try
        curve_fit(fit_func, xy_mesh, vec(image), p0)
    catch
        curve_fit(fit_func, xy_mesh, fit_func(xy_mesh, p0), p0)
    end

    param = fit.param
    stdev = try
        stderror(fit)
    catch
        zeros(size(param))
    end

    if stdev[3] > 0.1 || stdev[3] == 0.0 || param[2] < 1e3
        continue
    end

    write(file, @sprintf "h k l: %d %d %d\n" hkl[1] hkl[2] hkl[3])
    write(file, @sprintf "d      2θ       ω       ϕ\n")
    write(file, @sprintf "%03.0lf%8.2lf%8.2lf%8.2lf\n" d hank(2θ) hank(true_ω(s, wl[1], θ)) hank(ϕ))
    write(file, @sprintf "Pred. Ka1: %.2lf  %.2lf\n" peaks[1][1] peaks[1][2])
    write(file, @sprintf "Pred. Ka2: %.2lf  %.2lf\n" peaks[2][1] peaks[2][2])
    write(file, @sprintf "Noise: %.1f(%2.0f)\n" param[1] 10stdev[1])
    write(file, @sprintf "Params:      Int            x0           y0         σx         σy\n")
    write(file, @sprintf "Fit. Ka1: %6.0f(%.0f)  %7.3f(%.0f)  %7.3f(%.0f)  %5.3f(%.0f)  %5.3f(%.0f)\n" param[2] stdev[2] param[3] 1e3stdev[3] param[4] 1e3stdev[4] param[5] 1e3stdev[5]  param[6] 1e3stdev[6])
    write(file, @sprintf "Fit. Ka2: %6.0f(%.0f)  %7.3f(%.0f)  %7.3f(%.0f)  %5.3f(%.0f)  %5.3f(%.0f)\n" param[7] stdev[7] param[8] 1e3stdev[8] param[9] 1e3stdev[9] param[10] 1e3stdev[10]  param[11] 1e3stdev[11])
    write(file, @sprintf "\n")

    @printf "written%4d%4d%4d%8.2f\n" hkl[1] hkl[2] hkl[3] hank(2θ)
    global experiment_counter += 1
    
    hkl in keys(bond_experiments) || (bond_experiments[hkl] = ([], []))
    append!(bond_experiments[hkl][θ < 0 ? 1 : 2], d, θ, ϕ, param)
end
@printf "Evaluated %d experiments\n" experiment_counter

experiment_counter = 0
write(file, "="^16*" Bond experiments "*"="^16*"\n")
for (hkl, (neg, pos)) in bond_experiments
    isempty(neg) | isempty(pos) && continue
    single_θ(ex::Vector{Float64})::Float64 = ex[2] - px_size*ex[6]/2ex[1]
    h, k, l = hkl
    θ = (single_θ(pos) - single_θ(neg))/2
    a_bond = wl[1]*norm(hkl)/2sin(θ)
    write(file, @sprintf "%4d%4d%4d => %10.5f,%10.5f mul %d %d\n" h k l hank(2θ) a_bond length(neg)/14 length(pos)/14)
    global experiment_counter += 1
end
@printf "Evaluated %d bond schemes\n" experiment_counter