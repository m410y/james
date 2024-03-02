using StaticArrays
using LinearAlgebra
using Rotations
using TOML
using Printf

include("parse.jl")


init = TOML.parsefile("init.toml")

a, U = p4p_orient(joinpath(init["p4p_folder"], init["p4p_file"]))
init["correct_lattice"] && (a = init["a"])
@printf "Sample: %s\n" split(init["p4p_file"], ".")[1]

meta = sfrm_meta(joinpath(init["sfrm_folder"], init["sfrm_files"][1]))

drawing = init["drawing"]
check = init["check"]
shift_corr = init["shift_corr"]
logging= init["logging"]

wl = init["wl"]
d = meta["d"]
χ = meta["chi"]
ϕ = meta["phi"]
ω = meta["omega"]
θ = meta["tth"]/2
cols = meta["x_max"]
rows = meta["y_max"]
xc = init["xc"]
yc = init["yc"]
px_size = init["px_size"]

hank(ang) = rad2deg(rem2pi(ang, RoundNearest))
@printf "   d    2θ   ω    ϕ\n"
@printf "%4.0lf %6.2lf  %6.2lf %6.2lf\n" d hank(2*θ) hank(ω) hank(ϕ)

ray = [1, 0, 0]
hkl = round.(Int, U'RotZXZ(ϕ, χ, -ω)*(RotZ(2*θ)*ray - ray)*a/wl[1])
@printf "h k l: %d %d %d\n" hkl[1] hkl[2] hkl[3] 

s = RotXZ(-χ, -ϕ)*U*hkl/a

function true_ω(λ::Float64)::Float64
    _ω = atan(s[2], s[1])*[1, 1] - acos(-0.5*λ*norm(s))*[1, -1]
    Δω = @. abs(rem2pi(_ω - ω, RoundNearest))
    return Δω[1] < Δω[2] ? _ω[1] : _ω[2]
end

function coords(v::SVector{3, Float64})::SVector{2, Float64}
    v = RotZ(-2*θ)*normalize(v)*d/px_size
    return xc - v[2], yc + v[3]
end

peaks = "peaks" in keys(init) ? init["peaks"] : [coords(ray/λ + RotZ(true_ω(λ))*s) for λ in wl]

x_diap = init["x_lims"][1]:init["x_lims"][2]
y_diap = init["y_lims"][1]:init["y_lims"][2]
image = sfrm_image(joinpath.(init["sfrm_folder"], init["sfrm_files"]))
fit_data = image[y_diap, x_diap]

@printf "Pred. Ka1: %.2lf  %.2lf\n" peaks[1][1] peaks[1][2]
@printf "Pred. Ka2: %.2lf  %.2lf\n" peaks[2][1] peaks[2][2]

if drawing
    using Plots

    heatmap(x_diap, y_diap, fit_data, aspect_ratio=:equal)
    savefig("plot.png")
end

check && exit()

function gauss_2D(t::Matrix, p::Vector{Float64})::Vector{Float64}
    A, x0, y0, σx, σy = p
    x = @. (t[:, 1] - x0)/σx
    y = @. (t[:, 2] - y0)/σy
    return @. A*exp(-0.5*(x^2+y^2))
end

function fit_func(t::Matrix, p::Vector{Float64})::Vector{Float64}
    return p[1] .+ gauss_2D(t, p[2:6]) + gauss_2D(t, p[7:11])
end

using LsqFit

xy_mesh = hcat(vec([x for y=y_diap, x=x_diap]), vec([y for y=y_diap, x=x_diap]))

A0 = findmax(fit_data)[1]
σ0 = init["sigma"]
noise0 = init["noise"]

g1 = [A0, peaks[1][1], peaks[1][2], σ0, σ0]
g2 = [A0/2, peaks[2][1], peaks[2][2], σ0, σ0]
p0 = [noise0; g1; g2]

if shift_corr
    using DSP
    using Statistics

    xcorr_data = xcorr(fit_func(xy_mesh, p0), vec(fit_data))[round(Int, length(fit_data)/2)+1 : round(Int, 3*length(fit_data)/2)+1]
    xy_shift = 2*(xy_mesh'xcorr_data./sum(xcorr_data) - [mean(x_diap), mean(y_diap)])
    for i in eachindex(peaks); peaks[i] -= xy_shift; end

    @printf "Corr. Ka1: %.2lf  %.2lf\n" peaks[1][1] peaks[1][2]
    @printf "Corr. Ka2: %.2lf  %.2lf\n" peaks[2][1] peaks[2][2]

    g1 = [A0, peaks[1][1], peaks[1][2], σ0, σ0]
    g2 = [A0/2, peaks[2][1], peaks[2][2], σ0, σ0]
    p0 = [noise0; g1; g2]
end

fit = curve_fit(fit_func, xy_mesh, vec(fit_data), p0)
param = fit.param
stdev = stderror(fit)

@printf "Params:      Int       x0       y0     σx     σy\n"
@printf "Fit. Ka1: %6.0lf  %7.3f  %7.3f  %5.3f  %5.3f\n" param[2] param[3] param[4] param[5]  param[6]
@printf "Fit. Ka2: %6.0lf  %7.3f  %7.3f  %5.3f  %5.3f\n" param[7] param[8] param[9] param[10] param[11]
@printf "Err. Ka1: %6.0lf  %7.3f  %7.3f  %5.3f  %5.3f\n" stdev[2] stdev[3] stdev[4] stdev[5]  stdev[6]
@printf "Err. Ka2: %6.0lf  %7.3f  %7.3f  %5.3f  %5.3f\n" stdev[7] stdev[8] stdev[9] stdev[10] stdev[11]

if logging
    file = open("fit_log.txt", "a")
    write(file, "\n")
    write(file, "$(split(init["p4p_file"], ".")[1])\n")
    for idx in hkl; write(file, "$idx  "); end
    write(file, "\n")
    θ, ϕ, ω = @. rad2deg(rem2pi([θ, ϕ, ω], RoundNearest))
    write(file, "$d\t$(2*θ)\t$ϕ\t$ω\n")
    write(file, "$(fit.param[1])\n")
    for g_diap in [2:6, 7:11]
        for par in param[g_diap]; write(file, "$par\t"); end
        write(file, "\n")  
    end
    for g_diap in [2:6, 7:11]
        for err in stdev[g_diap]; write(file, "$err\t"); end
        write(file, "\n")
    end
    close(file)
end

if drawing
    using Plots

    heatmap(x_diap, y_diap, reshape(fit.resid, size(fit_data)), aspect_ratio=:equal)
    savefig("resid.png")
end