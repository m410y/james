from utils import *

import fabio
import numpy as np


def sfrm_image(filenames):
    sum = None
    for filename in filenames:
        image = fabio.open(filename).data[::-1]
        sum = sum + image if sum is not None else image
        
    return sum


def sfrm_meta(filename):
    meta = {}
    file = fabio.openheader(filename)
    angles = np.deg2rad(np.array(file.header["ANGLES"].split(), dtype=float))
    meta["tth"] = norm_sym(angles[0])
    meta["omega"] = norm_pos(angles[1] + 0.5*np.deg2rad(float(file.header["RANGE"].split()[0])))
    meta["phi"] = angles[2]
    meta["chi"] = angles[3]
    meta["d"] = 10*float(file.header["DISTANC"].split()[1])
    meta["x_max"] = float(file.header["NCOLS"].split()[0])
    meta["y_max"] = float(file.header["NROWS"].split()[0])
    meta["x_c"] = float(file.header["CENTER"].split()[0])
    meta["y_c"] = float(file.header["CENTER"].split()[1])

    return meta


def p4p_orient(filename):
    with open(filename, "r") as file:
        m = np.zeros((3, 3), dtype=float)
        row = 0
        for line in file:
            if line.startswith("ORT"):
                m[row] = np.array(tuple(map(float, line.split()[1:4])))
                row += 1
                if row == 3:
                    break
        
        return m
    

def form_factor_read(filename="form_factor.txt"):
    with open(filename, "r") as file:
        f = {}
        for line in file:
            data = line[:-1].split(' ')
            if len(data) != 10:
                break

            f[data[0]] = tuple(float(a) for a in data[1:])

        return f