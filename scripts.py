from goniometer import *
from crystal import *

import numpy as np
from numpy import arctan2, arccos, arcsin, sqrt, fabs, array, pi
from numpy.linalg import norm, inv


def omega_to_reflection(plane, wl):
    s = norm(plane)
    v = plane / s
    arg = arctan2(v[1], v[0])
    acos = arccos(-0.5*wl*s/sqrt(v[0]**2+v[1]**2))
    omega_m = -arg - acos
    omega_p = -arg + acos
    return omega_p, omega_m


def to_reflection(plane, G: Goniometer, S: Xsource, wl="Ka1") -> list:
    if fabs(0.5*S.wl[wl]*norm(plane)) > 1.0:
        return []
    
    tth = 2*arcsin(0.5*S.wl[wl]*norm(plane))
    
    angles0 = G.to_beam(plane)
    if not angles0:
        return []

    angles = []
    for omega, phi in angles0:
        omega_m = omega + (pi - tth)/2
        omega_p = omega - (pi - tth)/2
        angles.append((tth, omega_m, omega_p, phi))

    return angles


def predict_coords(plane, G: Goniometer, D: Detector, S: Xsource) -> list:
    peaks = []
    for wl in S.wl:
        omega_m, omega_p = omega_to_reflection(plane, S.wl[wl])
        s = G.omega_rot(plane, omega_p if G.tth % 2*pi < pi else omega_m)
        peaks.append(D.coords(S.ray + s*S.wl[wl], G))

    return peaks


def assume_hkl(orient, G: Goniometer, S: Xsource):
    s =  G.inv_rot(G.omega_rot(S.ray, G.tth) - S.ray)/S.wl["Ka1"]
    return np.round(inv(orient).dot(s))


def fix_orient(orient, a):
    old_s = np.sqrt(np.sum(np.diag(np.dot(orient.T, orient)))/3)
    return orient / (a*old_s)


def combine_cell(atoms, coords):
    return (atoms_to_form_factor(atoms), array(coords))