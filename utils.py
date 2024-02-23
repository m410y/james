from numpy import pi


def norm_pos(angle):
    return angle % (2*pi)


def norm_sym(angle):
    return ((angle + pi) % (2*pi)) - pi