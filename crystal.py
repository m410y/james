from parse import form_factor_read
from itertools import product, permutations
from numpy import array, dot, exp, pi, sum, arcsin, sqrt, ceil
from numpy.linalg import norm


def bragg_angle(s, wl):
    return 2*arcsin(0.5*wl*norm(s))


def atoms_to_form_factor(atoms):
    factors = form_factor_read()
    res = []
    for atom in atoms:
        res.append(factors[atom])

    return array(res).T


def form_factor_eval(aff, s):
    res = aff[8].copy()
    for i in range(4):
        res += aff[i]*exp(-aff[i+1]*(s/4/pi)**2)
    
    return res


def struct_factor(hkl, orient, cell):
    aff, coords = cell
    factors = form_factor_eval(aff, norm(plane_eval(orient, hkl)))
    f = sum(factors*exp(2j*pi*dot(coords, hkl)))
    return sqrt((f.conjugate()*f).real)


def gen_hkl(radius_min, radius_max):
    for h in range(1, int(ceil(radius_max))):
        for k in range(h + 1):
            for l in range(k + 1):
                sqr = h**2 + k**2 + l**2
                if sqr < radius_min**2 or radius_max**2 < sqr:
                    continue

                yield (h, k, l)


def gen_sym_hkl(hkl_0):
    for p in permutations(hkl_0, 3):
        for sign in product((1, -1), repeat=3):
            hkl = tuple(array(p)*array(sign))
            yield hkl


def plane_eval(orient, indices):
    return dot(orient, indices)