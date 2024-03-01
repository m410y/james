using StaticArrays
using LinearAlgebra
using Rotations
using TOML
using DSP

include("parse.jl")


init = TOML.parsefile("init.toml")

a, U = p4p_orient("$(init["p4p_folder"])/$(init["p4p_file"])")
"a" in keys(init) && (a = init["a"])

meta = sfrm_meta("$(init["sfrm_folder"])/$(init["sfrm_files"][1])")
image = sfrm_image("$(init["sfrm_folder"])/".*init["sfrm_files"])

drawing = init["drawing"]
check = init["check"]
wl = init["wl"]
d = meta["d"]
chi = meta["chi"]
phi = meta["phi"]
omega = meta["omega"]
tth = meta["tth"]
cols = meta["x_max"]
rows = meta["y_max"]
xc = init["xc"]
yc = init["yc"]
px_size = init["px_size"]

print("d tth omega phi\n")
print("$d $(rad2deg(tth)) $(rad2deg(omega)) $(rad2deg(phi))\n")

ray = [1, 0, 0]
hkl = round.(Int, U'RotZXZ(phi, chi, -omega)*(RotZ(tth)*ray - ray)*a/wl[1])
print("h k l: $hkl\n")

s = RotXZ(-chi, -phi)*U*hkl/a

function true_omega(λ::Float64)::Float64
    t_om = atan(s[2], s[1])*[1, 1] - acos(-0.5*λ*norm(s))*[1, -1]
    diff = @. abs(rem2pi(t_om - omega, RoundNearest))
    return diff[1] < diff[2] ? t_om[1] : t_om[2]
end

function coords(v::SVector{3, Float64})::SVector{2, Float64}
    v = RotZ(-tth)*normalize(v)*d/px_size
    return xc - v[2], yc + v[3]
end

peaks = "peaks" in keys(init) ? init["peaks"] : [coords(ray/λ + RotZ(true_omega(λ))*s) for λ in wl]

x_diap = init["x_lims"][1]:init["x_lims"][2]
y_diap = init["y_lims"][1]:init["y_lims"][2]
fit_data = image[y_diap, x_diap]

print("Ka1: $(peaks[1])\n")
print("Ka2: $(peaks[2])\n")

if drawing & check
    using Plots

    heatmap(x_diap, y_diap, fit_data, aspect_ratio=:equal)
    savefig("plot.png")

    exit()
end

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
fit = curve_fit(fit_func, xy_mesh, vec(fit_data), p0)

file = open("fit_log.txt", "a")
write(file, "$(split(init["p4p_file"], ".")[1])\n")
for idx in hkl; write(file, "$idx  "); end
write(file, "\n")
tth, phi, omega = @. rad2deg(rem2pi([tth, phi, omega], RoundNearest))
write(file, "$d\t$tth\t$phi\t$omega\n")
write(file, "$(fit.param[1])\n")
for g in [fit.param[2:6], fit.param[7:11]]
    for par in g; write(file, "$par\t"); end
    write(file, "\n")
end
write(file, "\n")

if drawing
    using Plots

    heatmap(x_diap, y_diap, reshape(fit.resid, size(fit_data)), aspect_ratio=:equal)
    savefig("resid.png")
end