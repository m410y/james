from scipy.optimize import least_squares
from scipy.special import voigt_profile
import numpy as np
from numpy import sqrt, newaxis, full, fabs, arange, pi, exp
import matplotlib.pyplot as plt


def voigt2d(x, y, A, xc, wx, yc, wy, mu):
    dist_x = ((x - xc)/wx)**2
    dist_y = ((y - yc)/wy)**2
    dist = sqrt(dist_y[:, newaxis] + dist_x)
    return A * voigt_profile(dist, 1 - mu, mu)


def gauss2d(x, y, A, xc, wx, yc, wy):
    dist_x = ((x - xc)/wx)**2
    dist_y = ((y - yc)/wy)**2
    dist2 = dist_y[:, newaxis] + dist_x
    return A*exp(-0.5*dist2)/sqrt(2*pi)


def fit_func(x, y, params):
    noise, A1, x1, wx1, y1, wy1, mu1, A2, x2, wx2, y2, wy2, mu2 = params
    res = full((len(y), len(x)), noise, dtype=float)
    res += voigt2d(x, y, A1, x1, wx1, y1, wy1, mu1)
    res += voigt2d(x, y, A2, x2, wx2, y2, wy2, mu2)
    return res


def fit_image(image, peaks, verbose=1):
    x1, y1 = peaks[0]
    x2, y2 = peaks[1]

    x0 = int(2/3*x1+1/3*x2)
    y0 = int(2/3*y1+1/3*y2)
    dx = 32
    dy = 32
    fit_data = image[y0-dy:y0+dy, x0-dx:x0+dx]

    if verbose > 0:
        plt.imshow(fit_data, norm="log")
        plt.show()

    x_diff = fabs(x2 - x1)
    noise0 = 64
    A0 = np.max(fit_data)
    w0 = x_diff / 4
    mu0 = 0.1
    
    guess = (noise0, 2*A0, x1, w0, y1, w0, mu0, A0, x2, w0, y2, w0, mu0)
    left_bounds = (0, 0, x0-dx, 0, y0-dy, 0, 0, 0, x0-dx, 0, y0-dy, 0, 0)
    right_bounds = (A0, 2*A0, x0+dx, x_diff, y0+dy, x_diff, 1, A0, x0+dx, x_diff, y0+dy, x_diff, 1)
    x_grid = arange(x0-dx, x0 + dx, dtype=float)
    y_grid = arange(y0-dy, y0 + dy, dtype=float)

    residuals = lambda a: (fit_data - fit_func(x_grid, y_grid, a)).ravel()
    optimize_res = least_squares(residuals, guess, bounds=(left_bounds, right_bounds), verbose=verbose, ftol=1e-5)
    optimize_res = least_squares(residuals, optimize_res.x, method="lm", verbose=1 if verbose > 0 else 0)

    if verbose > 0:
        plt.imshow(optimize_res.fun.reshape(fit_data.shape))
        plt.show()
    
    return optimize_res


def jac_to_stdev(jac):
    corr = np.linalg.inv(jac.T.dot(jac))
    return np.sqrt(np.diag(corr))
    