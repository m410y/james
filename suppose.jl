using StaticArrays
using LinearAlgebra
using Rotations
using TOML
using Printf
using Combinatorics

include("parse.jl")
using .Parse

include("form_factor.jl")
using .FormFactor


init = TOML.parsefile("init.toml")
U, A = p4p_orient(init["p4p_file"])
init["correct_lattice"] && (A = Diagonal([init["a"], init["a"], init["a"]]))
UB = U*inv(A)

sample_name = split(split(init["p4p_file"], "\\")[end], ".")[1]
@printf "Sample: %s\n" sample_name

cell = cif_cell(init["cif_file"])

λ = init["wl"][1]
χ = deg2rad(init["chi"])

hkl_max = init["hkl_max"]
θ_min = deg2rad(init["tth_min"])/2
θ_max = deg2rad(min(init["tth_max"], init["tth_lim"]))/2
Δ_θω = deg2rad(init["tth_om_delta"])
Δω = deg2rad(init["om_scan_range"])/2

hank(ang::Float64)::Float64 = rad2deg(rem2pi(ang, RoundNearest))

function valid_bond(ω_m::Float64, ω_p::Float64, θ::Float64)::Bool
    valid(ω, θ_d) = abs(rem2pi(ω - pi/2 - 2θ_d, RoundNearest)) > Δ_θω
    return valid(ω_m-Δω, -θ) & valid(ω_m+Δω, -θ) & valid(ω_p-Δω, θ) & valid(ω_p+Δω, θ)
end

function process_hkl(h::Int, k::Int, l::Int)::Vector{Tuple{Bool, Bool, Float64, NTuple{7, Float64}}}
    hkl = SVector(h, k, l)
    hkl != [0, 0, 0] || return []

    s = UB*hkl
    v = normalize(s)
    d = 1/norm(s)

    abs(v[3]/sin(χ)) < 1 || return []

    (λ/2d < 1) ? (θ = asin(λ/2d)) : return []
    θ_min < θ < θ_max || return []

    ff = ff_eval(1/d, hkl, cell)
    ff > 1e-3 || return []

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

        push!(result, (v_n, v_f, ff, (θ, ϕ, ω_0, ω_m, ω_p, ω_mf, ω_pf)))
    end

    return result
end

function gen_hkl(θ_min::Float64, θ_max::Float64)::Vector{SVector{3, Int}}
    res = Vector{SVector{3, Int}}()
    
    s_eval(θ::Float64)::Float64 = 2sin(θ)/λ
    s_min = s_eval(θ_min)
    s_max = s_eval(θ_max)
    
    S = UB'UB
    Q = A*A'
    M = Q/det(A)^2
    N = S*det(A)^2

    k_min(h::Int)::Int = ceil(Int, (Q[1,2]*h - sqrt((Q[1,1]*s_max^2 - h^2)*N[3,3]))/Q[1,1])
    k_max(h::Int)::Int = floor(Int, (Q[1,2]*h + sqrt((Q[1,1]*s_max^2 - h^2)*N[3,3]))/Q[1,1])
    l_min(h::Int, k::Int)::Int = ceil(Int, (-S[1,3]*h - S[2,3]*k - sqrt(S[3,3]*s_max^2 - M[2,2]*h^2 - M[1,1]*k^2 + 2M[1,2]*h*k))/S[3,3])
    l_max(h::Int, k::Int)::Int = floor(Int, (-S[1,3]*h - S[2,3]*k + sqrt(S[3,3]*s_max^2 - M[2,2]*h^2 - M[1,1]*k^2 + 2M[1,2]*h*k))/S[3,3])
    l_min_in(h::Int, k::Int)::Int = floor(Int, (-S[1,3]*h - S[2,3]*k - sqrt(S[3,3]*s_min^2 - M[2,2]*h^2 - M[1,1]*k^2 + 2M[1,2]*h*k))/S[3,3])
    l_max_in(h::Int, k::Int)::Int = ceil(Int, (-S[1,3]*h - S[2,3]*k + sqrt(S[3,3]*s_min^2 - M[2,2]*h^2 - M[1,1]*k^2 + 2M[1,2]*h*k))/S[3,3])

    h_max = floor(Int, s_max*sqrt(Q[1,1]))
    for h in -h_max:h_max, k in k_min(h):k_max(h)
        if S[3,3]*s_min^2 > M[2,2]*h^2 + M[1,1]*k^2 - 2M[1,2]*h*k
            foreach(l -> push!(res, (h, k, l)), l_min(h,k):l_min_in(h,k))
            foreach(l -> push!(res, (h, k, l)), l_max_in(h,k):l_max(h,k))
        else
            foreach(l -> push!(res, (h, k, l)), l_min(h,k):l_max(h,k))
        end
    end

    return res
end

function signperm(h, k, l)
    pm = [1, -1]
    return [perm.*signs for perm in permutations([h, k, l]), signs in [[i, j, k] for i=pm, j=pm, k=pm]]
end

file = open(joinpath(init["res_folder"], sample_name*"_suppose.txt"), "w")
write(file, "N      h   k   l   v_n   v_f    ff     2θ       ϕ     ω_0     ω_m     ω_p    ω_mf    ω_pf\n")

counter = 0
for (h, k, l) in gen_hkl(θ_min, θ_max)
    h < 0 && continue
    for (v_n, v_f, ff, angles) in process_hkl(h, k, l)
        θ, ϕ, ω_0, ω_m, ω_p, ω_mf, ω_pf = hank.(angles)
        global counter += 1
        write(file, @sprintf "%04d%4d%4d%4d%6s%6s%6.0f%7.2f%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f\n" counter h k l v_n v_f ff 2θ ϕ ω_0 ω_m ω_p ω_mf ω_pf)
    end
end

@printf "Supposed %d experiments\n" counter