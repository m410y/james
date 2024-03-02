using StaticArrays
using LinearAlgebra
using Rotations
using TOML
using Printf
using Combinatorics

include("parse.jl")
include("form_factor.jl")


init = TOML.parsefile("init.toml")

a, U = p4p_orient(joinpath(init["p4p_folder"], init["p4p_file"]))
init["correct_lattice"] && (a = init["a"])
@printf "Sample: %s\n" split(init["p4p_file"], ".")[1]

cell = cif_cell(joinpath(init["cif_folder"], init["cif_file"]))

λ = init["wl"][1]
χ = deg2rad(init["chi"])

hkl_max = init["hkl_max"]
θ_min = deg2rad(init["tth_min"])/2
θ_max = deg2rad(min(init["tth_max"], init["tth_lim"]))/2
Δ_θω = deg2rad(init["tth_om_delta"])
Δω = deg2rad(init["om_scan_range"])/2

hank(ang::Float64)::Float64 = rad2deg(rem2pi(ang, RoundNearest))

function valid_bond(ω_m::Float64, ω_p::Float64, θ::Float64)::Bool
    valid(_ω, _θ) = abs(rem2pi(_ω - pi/2 - 2*_θ, RoundNearest)) > Δ_θω
    return valid(ω_m-Δω, -θ) & valid(ω_m+Δω, -θ) & valid(ω_p-Δω, θ) & valid(ω_p+Δω, θ)
end

function process_hkl(hkl::SVector{3, Int})
    hkl != [0, 0, 0] || return

    v = LinearAlgebra.normalize(U*hkl)
    abs(v[3]/sin(χ)) < 1 || return

    d = a/norm(hkl)
    (0.5*λ/d < 1) ? (θ = asin(0.5*λ/d)) : return
    θ_min < θ < θ_max || return

    ff = ff_eval(1/d, hkl, cell)
    ff > 1e-3 || return

    result = []
    for ω_0 in [asin(v[3]/sin(χ)), pi - asin(v[3]/sin(χ))]
        ray = RotXZ(χ, -ω_0)*[-1, 0, 0]
        ϕ = atan(v[2], v[1])-atan(ray[2], ray[1])
        
        ω_m  = ω_0 + pi/2 - θ
        ω_p  = ω_0 - pi/2 + θ
        ω_mf = ω_0 - pi/2 - θ
        ω_pf = ω_0 + pi/2 + θ
        
        v_n = valid_bond(ω_m,  ω_p,  θ)
        v_f = valid_bond(ω_mf, ω_pf, θ)

        append!(result, (v_n, v_f, ff, θ, ϕ, ω_0, ω_m, ω_p, ω_mf, ω_pf))
    end

    return result
end

function signperm(h, k, l)
    pm = [1, -1]
    return [perm.*signs for perm in permutations([h, k, l]), signs in [[i, j, k] for i=pm, j=pm, k=pm]]
end

file = open(joinpath(init["res_folder"], "suppose_$(split(init["p4p_file"], ".")[1]).txt"), "w")
write(file, "h\tk\tl\tv_n\tv_f\tff\t2θ\tϕ\tω_0\tω_m\tω_p\tω_mf\tω_pf\n")

experiment_counter = 0
for hkl in signperm(20, 10, 4)
    hkl[1] < 0 && continue
    result = process_hkl(SVector{3, Int}(hkl))
    isnothing(result) && continue
    for res in [result[1:10], result[11:20]]
        v_n, v_f, ff, θ, ϕ, ω_0, ω_m, ω_p, ω_mf, ω_pf = res
        for idx in hkl; write(file, "$idx\t"); end
        write(file, "$v_n\t$v_f\t$ff")
        for ang in hank.([2*θ, ϕ, ω_0, ω_m, ω_p, ω_mf, ω_pf]); write(file, "\t$ang"); end
        write(file, "\n")
        global experiment_counter += 1
    end
end

@printf "Supposed %d experiments\n" experiment_counter