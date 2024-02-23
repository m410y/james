from utils import *
from scipy.spatial.transform import Rotation

import numpy as np
from numpy import array, vstack, deg2rad, fabs, pi, sin, cos, arcsin, arctan2
from numpy.linalg import norm


class Xsource:
    def __init__(self):
        self.name = "Incoatec 3.0Ims"
        self.material = "Mo"
        self.wl = {"Ka1": 0.70931715,
                   "Ka2": 0.713607}
        self.ray = array((1.0, 0.0, 0.0)).reshape((3,))
        self.width = 0.11
        self.vdiv = 0.0005
        self.hdiv = 0.0005
    

class Goniometer:
    def __init__(self, d, chi=0.9548905777171216, tth=0.0, omega=0.0, phi=0.0):
        self.d = d
        self.chi = chi
        self.tth = tth
        self.omega = omega
        self.phi = phi
        self.tth_max = 1.6920967098085025
        #self.delta = 0.17453292519943295
        self.delta = 0.7543313027119493 + 0.035 # +2deg

    def rot(self, vec):
        return Rotation.from_euler("zxz", (self.phi, -self.chi, self.omega)).apply(vec)


    def inv_rot(self, vec):
        return Rotation.from_euler("zxz", (-self.omega, self.chi, self.phi)).apply(vec)
    

    def omega_rot(self, vec, omega):
        return Rotation.from_euler("z", omega).apply(vec)
    

    def valid(self, omega, tth=None):
        if tth:
            self.tth = tth
        tth_valid = self.tth < self.tth_max
        omega_valid = fabs(norm_sym(omega - pi/2 - self.tth)) > self.delta
        return tth_valid and omega_valid


    def to_beam(self, vec) -> list: 
        vec = vec / norm(vec)
        
        if fabs(vec[2]) > fabs(sin(self.chi)):
            return []
        
        angles = []
        omega0 = arcsin(vec[2]/sin(self.chi)) 
        for omega in (omega0, pi-omega0):
            a = -cos(omega)
            b = sin(omega)*cos(self.chi)
            phi = arctan2(a*vec[1]-b*vec[0],a*vec[0]+b*vec[1])
            angles.append((omega, phi))

        return angles


class Detector:
    def __init__(self):
        self.name = "Photon III"
        self.cols = 786
        self.rows = 1024
        self.xc = 388.7
        self.yc = 504.0
        self.px = 0.135

    def coords(self, vec, G: Goniometer):
        vec = G.omega_rot(vec, -G.tth)
        gamma = self.px/G.d
        x = self.xc - vec[1]/gamma
        y = self.yc + vec[2]/gamma
        return x, y