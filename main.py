from parse import *
from scripts import *
from crystal import *
from fitting import *
from utils import *

import toml as tomllib
import os
import matplotlib.pyplot as plt


def read_init():
    return tomllib.load("init.toml")


def reflex_in_center(init: dict):
    m = p4p_orient(os.path.join(init["p4p_folder"], init["p4p_file"]))
    m = fix_orient(m, init["a"]) if "a" in init.keys() else m

    meta = sfrm_meta(os.path.join(init["sfrm_folder"],  init["sfrm_files"][0]))
    image = sfrm_image([os.path.join(init["sfrm_folder"],  file) for file in init["sfrm_files"]])

    dx = init["dx"] if "dx" in init.keys() else 32
    dy = init["dy"] if "dy" in init.keys() else 32
    check = init["check"] if "check" in init.keys() else False

    print("sample: {}".format(init["p4p_file"].split("_")[1].split(".")[0]))

    G = Goniometer(meta["d"], chi=meta["chi"], tth=meta["tth"], omega=meta["omega"], phi=meta["phi"])
    D = Detector()
    S = Xsource()
    
    print("d, mm   2th, deg  omega, deg  phi, deg")
    print(f"{G.d:8.2f}{np.rad2deg(norm_sym(G.tth)):10.2f}{np.rad2deg(G.omega):12.2f}{np.rad2deg(G.phi):8.2f}")

    hkl = assume_hkl(m, G, S)
    print(f"reflex: {hkl}")
    
    plane = plane_eval(m, hkl)
    peaks = init["peaks"] if "peaks" in init.keys() else predict_coords(plane, G, D, S)
    
    if check:
        print( "WL    X, px   Y, px")
        print(f"Ka1:  {peaks[0][0]:8.2f}{peaks[0][1]:8.2f}")
        print(f"Ka2:  {peaks[1][0]:8.2f}{peaks[1][1]:8.2f}")

        plt.imshow(image, norm="log")
        plt.show()
        return
    
    result = fit_image(image, peaks, dx=dx, dy=dy, verbose=0)

    popt = result.x
    perr = jac_to_stdev(result.jac)
    print(f"noise: {popt[0]:.2f}({perr[0]:.2f})")
    print("WL    Int   X0, px  Wx, px  Y0, px  Wy, px")
    print(f"Ka1:  {popt[1]:6.0f}{popt[2]:8.3f}{popt[3]:8.3f}{popt[4]:8.3f}{popt[5]:8.3f}")
    #print(f"err:\t{perr[1]:5.0f}\t{perr[2]:5.3f}\t{perr[3]:5.3f}\t{perr[4]:5.3f}\t{perr[5]:5.3f}")
    print(f"Ka2:  {popt[7]:6.0f}{popt[8]:8.3f}{popt[9]:8.3f}{popt[10]:8.3f}{popt[11]:8.3f}")
    #print(f"err:\t{perr[7]:5.0f}\t{perr[8]:5.3f}\t{perr[9]:5.3f}\t{perr[10]:5.3f}\t{perr[11]:5.3f}")
    
    plt.imshow(result.fun.reshape((2*dx,2*dy)))
    plt.show()


def suppose_bond(init: dict, weight=lambda p: 10/(1+(10*fabs(p[6]-180))**2)+p[3]+0.1*p[2]):
    m = p4p_orient(os.path.join(init["p4p_folder"], init["p4p_file"]))
    m = fix_orient(m, init["a"]) if "a" in init.keys() else m

    atoms = init["atoms"]
    coords = init["coords"]
    cell = combine_cell(atoms, coords)

    hkl_min = int(init["hkl_min"]) if "hkl_min" in init.keys() else 0
    hkl_max = int(init["hkl_max"]) if "hkl_max" in init.keys() else 12
    tth_min = init["tth_min"] if "tth_min" in init.keys() else 90.0
    tth_max = init["tth_max"] if "tth_max" in init.keys() else 105.0

    G = Goniometer(init["d"])
    S = Xsource()

    poss = []
    for hkl0 in gen_hkl(hkl_min, hkl_max):
        tth = bragg_angle(plane_eval(m, hkl0), S.wl["Ka1"])
        if tth > np.deg2rad(tth_max):
            continue

        for hkl in gen_sym_hkl(hkl0):
            if hkl[0] < 0:
                continue

            ff = struct_factor(hkl, m, cell)
            if fabs(ff) < 1e-3:
                continue
            
            angles = to_reflection(plane_eval(m, hkl), G, S)
            for tth, omega_m, omega_p, phi in angles:
                G.tth = tth

                phi = norm_pos(phi)
                omega_0 = norm_pos(0.5*(omega_m + omega_p))
                omega_m = norm_pos(omega_m)
                omega_p = norm_pos(omega_p)
                omega_mf = norm_pos(omega_m + pi)
                omega_pf = norm_pos(omega_p + pi)
                normal_valid = G.valid(omega_m, -tth) and G.valid(omega_p, tth)
                friedel_valid = G.valid(omega_mf, -tth) and G.valid(omega_pf, tth)

                angles_deg = np.rad2deg(array((tth, phi, omega_0, omega_m, omega_mf, omega_p, omega_pf)))
                res = (hkl, normal_valid, friedel_valid, ff, *angles_deg)
                if res not in poss:
                    poss.append(res)

    with open(init["hkl_output"] if "hkl_output" in init.keys() else "results.txt", "w") as file:
        file.write("hkl         n_valid  f_valid  F    2theta  phi     om_0    om_m    om_mf   om_p    om_pf\n")
        poss.sort(key=weight, reverse=True)
        for p in poss:
            hkl, normal_valid, friedel_valid, ff, tth, phi, omega_0, omega_m, omega_mf, omega_p, omega_pf = p
            h, k, l = hkl
            ok = True
            ok *= normal_valid
            ok *= friedel_valid
            ok *= tth > tth_min
            if ok:
                file.write(f"{h:4d}{k:4d}{l:4d}{normal_valid:8d}{friedel_valid:8d}{ff:8.2f}{tth:8.2f}{phi:8.2f}{omega_0:8.2f}{omega_m:8.2f}{omega_mf:8.2f}{omega_p:8.2f}{omega_pf:8.2f}\n")


def main():
    init = read_init()
    mode = init["mode"]
    if mode == "suppose":
        suppose_bond(init)
    elif mode =="eval": 
        reflex_in_center(init)
    

if __name__ == "__main__":
    main()