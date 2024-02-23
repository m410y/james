print("sample: {}".format(p4p_filename.split("_")[1].split(".")[0]))
print(f"reflex: {hkl}")
print("d, mm\t2th, deg\tphi, deg")
print(f"{d:.2f}\t{np.rad2deg(tth):.2f}\t{np.rad2deg(phi):.2f}")

s0 = plane_eval(m, hkl)
s = rotate_plane(s0, omega=0.0, phi=phi, chi=chi)

omega_m, omega_p = omega_to_reflection(s, wl=MoKa1)
omega = omega_p if tth > 0 else omega_m

print(f"Predicted omega: {np.rad2deg(norm_pos(omega)):.2f} deg")


popt = optimize_res.x
jac = optimize_res.jac
cov = np.linalg.inv(jac.T.dot(jac))
perr = np.sqrt(np.diag(cov))

print(f"noise: {popt[0]:.2f}({perr[0]:.2f})")
print("Int, count\tX0, px\tWx, px\tY0, px\tWy, px")
print(f"Ka1:\t{popt[1]:5.0f}\t{popt[2]:5.3f}\t{popt[3]:5.3f}\t{popt[4]:5.3f}\t{popt[5]:5.3f}")
print(f"err:\t{perr[1]:5.0f}\t{perr[2]:5.3f}\t{perr[3]:5.3f}\t{perr[4]:5.3f}\t{perr[5]:5.3f}")
print(f"Ka2:\t{popt[7]:5.0f}\t{popt[8]:5.3f}\t{popt[9]:5.3f}\t{popt[10]:5.3f}\t{popt[11]:5.3f}")
print(f"err:\t{perr[7]:5.0f}\t{perr[8]:5.3f}\t{perr[9]:5.3f}\t{perr[10]:5.3f}\t{perr[11]:5.3f}")

plt.imshow(optimize_res.fun.reshape(fit_data.shape))
plt.show()