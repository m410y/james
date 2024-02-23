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


def reflex_in_center(init):
    m = p4p_orient(os.path.join(init["p4p_folder"], init["p4p_file"]))
    meta = sfrm_meta(os.path.join(init["sfrm_folder"],  init["sfrm_files"][0]))
    image = sfrm_image([os.path.join(init["sfrm_folder"],  file) for file in init["sfrm_files"]])

    print("sample: {}".format(init["p4p_file"].split("_")[1].split(".")[0]))

    G = Goniometer(meta["d"], chi=meta["chi"], tth=meta["tth"], omega=meta["omega"], phi=meta["phi"])
    D = Detector()
    S = Xsource()
    
    print("d, mm\t2th, deg\tomega, deg\tphi, deg")
    print(f"{G.d:.2f}\t{np.rad2deg(norm_sym(G.tth)):.2f}\t{np.rad2deg(G.omega):.2f}\t{np.rad2deg(G.phi):.2f}")

    hkl = assume_hkl(m, G, S)
    print(f"reflex: {hkl}")
    
    plane = plane_eval(m, hkl)
    peaks = predict_coords(plane, G, D, S)
    #peaks = [(389, 506), (402, 506)]
    print(f"Ka1: {peaks[0][0]:.2f}, {peaks[0][1]:.2f}")
    print(f"Ka2: {peaks[1][0]:.2f}, {peaks[1][1]:.2f}")
    #return

    plt.imshow(image, norm="log")
    plt.show()
    result = fit_image(image, peaks, verbose=0)

    popt = result.x
    perr = jac_to_stdev(result.jac)
    print(f"noise: {popt[0]:.2f}({perr[0]:.2f})")
    print("Int, count\tX0, px\tWx, px\tY0, px\tWy, px")
    print(f"Ka1:\t{popt[1]:5.0f}\t{popt[2]:5.3f}\t{popt[3]:5.3f}\t{popt[4]:5.3f}\t{popt[5]:5.3f}")
    print(f"err:\t{perr[1]:5.0f}\t{perr[2]:5.3f}\t{perr[3]:5.3f}\t{perr[4]:5.3f}\t{perr[5]:5.3f}")
    print(f"Ka2:\t{popt[7]:5.0f}\t{popt[8]:5.3f}\t{popt[9]:5.3f}\t{popt[10]:5.3f}\t{popt[11]:5.3f}")
    print(f"err:\t{perr[7]:5.0f}\t{perr[8]:5.3f}\t{perr[9]:5.3f}\t{perr[10]:5.3f}\t{perr[11]:5.3f}")
    
    plt.imshow(result.fun.reshape((64,64)))
    plt.show()


def suppose_bond(init):
    m = p4p_orient(os.path.join(init["p4p_folder"], init["p4p_file"]))
    m = fix_orient(m, init["a"])
    G = Goniometer(init["d"])
    S = Xsource()
    poss = []
    atoms = atoms_primitive
    for hkl0 in gen_hkl(10, atoms):
        ff = struct_factor(hkl0, atoms)
        tth = bragg_angle(plane_eval(m, hkl0), S.wl["Ka1"])
        if ff == 0.0 or tth > np.deg2rad(97.0):
            continue

        for hkl in gen_sym_hkl(hkl0):
            if hkl[0] < 0:
                continue

            angles = to_reflection(plane_eval(m, hkl), G, S)
            for tth, omega_m, omega_p, phi in angles:
                G.tth = tth
                omega_0 = norm_pos(0.5*(omega_m + omega_p))
                phi = norm_pos(phi)
                omega_m = norm_pos(omega_m)
                omega_p = norm_pos(omega_p)
                omega_mf = norm_pos(omega_m + pi)
                omega_pf = norm_pos(omega_p + pi)
                normal_valid = G.valid(omega_m, -tth) and G.valid(omega_p, tth)
                friedel_valid = G.valid(omega_mf, -tth) and G.valid(omega_pf, tth)
                angles_res = np.rad2deg(array((tth, phi, omega_0, omega_m, omega_mf, omega_p, omega_pf)))
                res = (hkl, normal_valid, friedel_valid, ff, *angles_res)
                if poss.count(res) == 0:
                    poss.append(res)

    with open("results.txt", "w") as file:
        file.write("hkl         n_valid  f_valid  F^2  2theta  phi     om_0    om_m    om_mf   om_p    om_pf\n")
        weight = lambda p: 10/(1+(10*fabs(p[6]-180))**2)+p[3]+0.1*p[2]
        poss.sort(key=weight, reverse=True)
        for p in poss:
            hkl, normal_valid, friedel_valid, ff, tth, phi, omega_0, omega_m, omega_mf, omega_p, omega_pf = p
            h, k, l = hkl
            ok = True
            ok *= tth > 96.5
            if ok:
                file.write(f"{h:4d}{k:4d}{l:4d}{normal_valid:8d}{friedel_valid:8d}{ff:4.0f}{tth:8.2f}{phi:8.2f}{omega_0:8.2f}{omega_m:8.2f}{omega_mf:8.2f}{omega_p:8.2f}{omega_pf:8.2f}\n")


def main():
    init = read_init()
    suppose_bond(init)
    #reflex_in_center(init)
    

if __name__ == "__main__":
    main()