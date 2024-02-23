from itertools import product, permutations
from numpy import array, dot, exp, pi, sum, arcsin
from numpy.linalg import norm


atoms_diamond = [(0, 0, 0),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.5, 0.5, 0),
            (0.25, 0.25, 0.25),
            (-0.25, 0.25, 0.25),
            (0.25, -0.25, 0.25),
            (0.25, 0.25, -0.25)]



atoms_primitive = [(0, 0, 0)]


def bragg_angle(s, wl):
    return 2*arcsin(0.5*wl*norm(s))


def struct_factor(hkl, atoms):
    f = sum(exp(2j*pi*dot(atoms, hkl)))
    return round((f.conjugate()*f).real, 5)


def gen_hkl(radius, atoms):
    for h in range(1, radius):
        for k in range(h + 1):
            for l in range(k + 1):
                if h**2 + k**2 + l**2 > radius**2:
                    continue

                yield (h, k, l)


def gen_sym_hkl(hkl_0):
    for p in permutations(hkl_0, 3):
        for sign in product((1, -1), repeat=3):
            hkl = tuple(array(p)*array(sign))
            yield hkl


def plane_eval(orient, indices):
    return dot(orient, indices)