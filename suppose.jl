using StaticArrays
using LinearAlgebra
using Rotations
using TOML
using Combinatorics
using IterTools

include("parse.jl")


init = TOML.parsefile("init.toml")

a, U = p4p_orient("$(init["p4p_folder"])/$(init["p4p_file"])")
"a" in keys(init) && (a = init["a"])

wl = init["wl"]
hkl_max = init["hkl_max"]
tth_min = deg2rad(init["tth_min"])
tth_max = deg2rad(min(init["tth_max"], init["tth_lim"]))
tth_om_delta = deg2rad(init["tth_om_delta"])
chi = deg2rad(init["chi"])
om_range = deg2rad(init["om_scan_range"])/2

file = open("results.txt", "w")
write(file, "h k l norm frid F tth phi om_0 om_m om_p om_mf om_pf\n")
#for hkl in CartesianIndices((0:hkl_max, -hkl_max:hkl_max, -hkl_max:hkl_max))
for hkl in [p.*s for p in permutations([20, 10, 4]), s in product([1, -1], [1, -1], [1, -1])]
    hkl = SVector(Tuple(hkl))
    hkl != [0,0,0] || continue
    
    s = normalize(SVector{3, Float64}(U*hkl))
    d = a/norm(hkl)
    
    (0.5*wl[1]/d < 1) ? (tth = 2*asin(0.5*wl[1]/d)) : continue
    tth_min < tth < tth_max || continue

    ff = 1.0
    ff > 1e-3 || continue

    abs(s[3]/sin(chi)) < 1 || continue
    for omega_0 in [-asin(s[3]/sin(chi)), pi+asin(s[3]/sin(chi))]
        omega_0 -= om_range/2
        v = RotXZ(chi, -omega_0)*[-1, 0, 0]
        phi = atan(s[2], s[1])-atan(v[2], v[1])
        omega_m = omega_0 + (pi - tth)/2
        omega_p = omega_0 - (pi - tth)/2
        omega_mf = omega_m + pi
        omega_pf = omega_p + pi
        valid(_om, _tth) = abs(rem2pi(_om - pi/2 - _tth, RoundNearest)) > tth_om_delta
        valid_bond(_om_m, _om_p, _tth) = valid(_om_m, -_tth) & valid(_om_p, _tth)
        val_norm = valid_bond(omega_m, omega_p, tth) & valid_bond(omega_m+om_range, omega_p+om_range, tth)
        val_frid = valid_bond(omega_mf, omega_pf, tth) & valid_bond(omega_mf+om_range, omega_pf+om_range, tth)
        
        for idx in hkl; write(file, "$idx\t"); end
        write(file, "$val_norm\t$val_frid\t$ff")
        angles = rad2deg.(rem2pi.((tth, phi, omega_0, omega_m, omega_p, omega_mf, omega_pf), RoundNearest))
        for ang in angles; write(file, "\t$ang"); end
        write(file, "\n")
    end
end