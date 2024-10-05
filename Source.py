import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from labellines import labelLine, labelLines
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.figure import figaspect
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft, fftfreq, rfft, ifft
from scipy.stats import chi2, norm, t, poisson
from scipy.odr import *
import uncertainties as unc
import uncertainties.unumpy as unp
import seaborn as sns
import warnings
import astropy.constants as const
import astropy.units as u


mycolor1 = "#F5B7B1"
mycolor2 = "#F49292"

# define plot params
plt.style.use("Source.mplstyle")
plt.rc('text', usetex=True)  # enable use of LaTeX in matplotlib
plt.rc('font', family="serif", size=15)  # font settings

def const(x, *p):
    return p[0]


def parabola(x, *p):
    return p[0]*x**2 + p[1]*x + p[2]


def affineline(x, *p):
    return p[0]*x


def line(x, *p):
    return p[0]*x + p[1]


def exponential(x, *p):
    """
    f(x) = p0 exp(p1 (x - p2)) + p3
    """
    return p[0]*np.exp(p[1]*(x - p[2])) + p[3]


def sine(x, *p):
    return p[0]*np.sin(p[1]*x + p[2]) + p[3]


def chisq(obs, exp, sigma=None, dof=0, fitparams=0):
    """
    Calculate the chi squared value
    :param obs: measured y values
    :param exp: expected y values
    :param sigma: error of the measured y values (optional)
    :param dof: degrees of freedom (optional)
    :param fitparams: number of fit parameters (optional)
    :return: chi squared
    """
    exp = np.asarray(exp)
    obs = np.asarray(obs)
    if sigma is None:
        sigma = np.ones_like(exp)
    else:
        sigma = np.asarray(sigma)
    if fitparams > 0:
        dof = len(obs) - fitparams
    if dof == 0:
        return np.sum(((obs - exp) / sigma)**2)
    elif dof < 0:
        raise ValueError("dof must be positive")
    else:
        return np.sum(((obs - exp) / sigma)**2) / dof



def sine_fit(x, y, err=None, min=0, p0=None, verbose=False):
    if err is None:
        err = np.ones(len(x))
    if p0 is None:
        p0 = [1000, 1100]
    start, end = p0[0], p0[1]
    popt, pcov = curve_fit(sine, x.iloc[start:end], y.iloc[start:end], sigma=err.iloc[start:end], absolute_sigma=True, p0=[1, 5, 1, 1])
    chi = chisq(sine(x.iloc[start:end], *popt), y.iloc[start:end], dof=len(x.iloc[start:end]) - 4)
    if verbose:
        print(f"start: {start}, end: {end}, chi: {chi}")
    # increase start and end by 100 as long as chi is smaller than 1
    while chi < 1:
        end += len(x)//30
        if start > min:
            start -= 100
        try:
            popt, pcov = curve_fit(sine, x.iloc[start:end], y.iloc[start:end], sigma=err.iloc[start:end], absolute_sigma=True, p0=[popt[0], popt[1], popt[2], popt[3]])
        except RuntimeError:
            print("RuntimeError")
            break
        if end > 4*len(x)/5:
            if verbose:
                print("end too large")
            break
        chi = chisq(sine(x.iloc[start:end], *popt), y.iloc[start:end], dof=len(x.iloc[start:end]) - 4)
        if verbose:
            print(f"start: {start}, end: {end}, chi: {chi}")
    end -= len(x)//30
    start += 100
    popt, pcov = curve_fit(sine, x.iloc[start:end], y.iloc[start:end], sigma=err.iloc[start:end], absolute_sigma=True, p0=[popt[0], popt[1], popt[2], popt[3]])
    return popt, pcov


def get_fun_params(fobj, max=100):
    """
    Get the number of parameters of a function, that takes parameters as array (used to for fitting)
    :param fobj: function to get the number of parameters from
    :param max: maximum number of parameters to check
    :return: number of parameters
    """
    for i in range(max):
        try:
            a = (1,) * i
            # print(a)
            fobj(1, *a)
        except IndexError:
            continue
        else:
            return i


def swap(fobj):
    """
    function that swaps the order of the parameters
    :param fobj: function to swap the parameters of
    :return: function with swapped parameters
    """
    def wrapper(*args):
        return fobj(args[1], *args[0])

    return wrapper


# implement differential evolution
def de(fit, xdata, ydata, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000, fobj=chisq, seed=None, sigma=None):
    """
    rand/1/bin differential evolution algorithm used for fitting
    :param fit: function to fit
    :param xdata: xdata to use for fitting
    :param ydata: ydata to use for fitting
    :param bounds: bounds for the parameters
    :param mut: mutation factor between 0 and 1 (default 0.8)
    :param crossp: crossover probability between 0 and 1 (default 0.7)
    :param popsize: population size of fitparameter vectors (default 20)
    :param its: number of iterations to run the evolution (default 1000)
    :param fobj: objective function to use for fitness (default chisq)
    :param seed: seed for random number generator, use for troubleshooting or reproducibility (default None)
    :param sigma: error of the measured y values (optional)
    :return: list of the best fitparameter and fitness for each iteration
    """
    # initial checks
    if not 0 <= mut <= 1:
        raise ValueError("mut must be between 0 and 1")
    if not 0 <= crossp <= 1:
        raise ValueError("crossp must be between 0 and 1")
    if xdata.size != ydata.size:
        raise ValueError("xdata and ydata must have the same size")
    # if len(bounds) != get_fun_params(fit):
    #     raise ValueError("bounds must have the same size as the number of parameters of fit")
    # set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    dimensions = len(bounds)
    # create population with random parameters (between 0 and 1)
    pop = np.random.rand(popsize, dimensions)
    # scale parameters to the given bounds
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    # calculate fitness (higher is worse)
    fitness = np.asarray([fobj(fit(xdata, *ind), ydata, sigma=sigma) for ind in pop_denorm])
    # sort by fitness and get best (lowest) one
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    # start evolution
    for i in range(its):
        for j in range(popsize):
            # select three random vector index positions (not equal to j)
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            # create a mutant by adding random scaled difference vectors
            mutant = np.clip(a + mut * (b - c), 0, 1)
            # randomly create a crossover mask
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            # construct trial vector by mixing the mutant and the current vector
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            # calculate fitness
            f = fobj(fit(xdata, *trial_denorm), ydata, sigma=sigma)
            # replace the current vector if the trial vector is better
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]
        # yield min_b + pop * diff, [fitness for fitness in fitness]


# 2d fitting
def myfit(fobj, xdata, ydata, xerr=None, yerr=None, its=1000, p0=None, **kwargs):
    """ Fit a function fobj(xdata, *p) to (xdata, ydata)
    :param fobj: function to be fitted of the form fobj(xdata, *p) eg. line(x, *p) = p[0]*x + p[1]
    :param xdata: x data to be fitted
    :param ydata: y data to be fitted
    :param xerr: optional x error (default None)
    :param yerr: optional y error (default None)
    :param its: number of iterations for ODR to run (default 1000)
    :param p0: initial parameters; if none given, using 1 as starting value (default None)
    :param kwargs:
    :return: popt, pcov
    """
    # initial checks
    if p0 is None:
        num = get_fun_params(fobj)
        p0 = np.ones(num)
        warnings.warn("p0 not specified, using 1 for all parameters")
    if xdata.size != ydata.size:
        raise ValueError("x and y must be of the same size")

    if xerr is None:  # use curve_fit
        if yerr is None:
            return curve_fit(fobj, xdata, ydata, p0=p0, **kwargs)
        else:
            return curve_fit(fobj, xdata, ydata, sigma=yerr, absolute_sigma=True, p0=p0, **kwargs)
    else:  # use ODR
        print("using ODR")
        model = Model(swap(fobj))
        data = RealData(xdata, ydata, sx=xerr, sy=yerr)
        odr = ODR(data, model, beta0=p0, maxit=its, **kwargs)
        out = odr.run()

        return out.beta, out.sd_beta


def bootstrap(fobj, xdata, ydata, xerr=None, yerr=None, p=0.95, its=1000, p0=None, ax=None, **kwargs):
    """ Bootstrap fit including confidence bands to a function fobj(xdata, *p)
        :param fobj: function to be fitted of the form fobj(xdata, *p) eg. line(x, *p) = p[0]*x + p[1]
        :param xdata: x data to be fitted
        :param ydata: y data to be fitted
        :param xerr: optional x error (default None)
        :param yerr: optional y error (default None)
        :param p: confidence interval (default 0.95)
        :param its: number of iterations (default 1000)
        :param p0: initial parameters; if none given, using 1 as starting value (default None)
        :param ax: axis to plot population of each generation on (default None)
        :param kwargs:
        :return: optimal parameters (array), 1 sigma error of parameters (array), x and y values for confidence band (dataframe)
        """
    if xdata.size != ydata.size:
        raise ValueError("x and y must be of the same size")
    if p < 0:
        raise ValueError("p must be positive")
    elif p > 100:
        raise ValueError("p is way too large (should be a percentage)")
    elif p > 1:
        warnings.warn("p > 1, assuming p is a percentage")
        p = p / 100
    if p0 is None:
        num = get_fun_params(fobj)
        p0 = np.ones(num)
        warnings.warn("p0 not specified, using 1 for all parameters")
    if xerr is None:
        xerr = np.zeros(len(xdata))
    if yerr is None:
        yerr = np.zeros(len(ydata))
    arr = []
    for i in range(its):
        # ind = range(len(xdata))
        ind = np.random.randint(0, len(xdata), len(xdata))
        newx, newy = np.random.normal(xdata[ind], xerr[ind]), np.random.normal(ydata[ind], yerr[ind])
        try:
            popt, pcov = curve_fit(fobj, newx, newy, p0=p0, **kwargs)
        except:
            results = list(
                de(fobj, newx, newy, [-1*p0, 3 * p0],
                   **kwargs))
            popt = results[-1][0]
        arr.append(popt)
        if ax is not None:
            ax.plot(xdata, fobj(xdata, *popt), alpha=0.1, c="gray")
            ax.scatter(newx, newy, alpha=0.1, c="gray")

    arr = np.array(arr)
    tempx = np.linspace(xdata.min(), xdata.max(), 1000)
    arr2 = np.array([fobj(tempx, *arr[i]) for i in range(its)])

    pmean = np.mean(arr, axis=0)
    cov = np.cov(arr.T)
    perr = np.sqrt(np.diag(cov))

    ci = pd.DataFrame(arr2).quantile([0.5 * (1 - p), 1 - 0.5 * (1 - p)]).T
    ci.insert(0, "x", tempx)
    ci["mean"] = arr2.mean(axis=0)
    ci.columns = ["x", "c0", "c1", "mean"]
    popt, pcov = curve_fit(fobj, ci.x, ci["mean"], p0=pmean, **kwargs)
    # popt2, pcov2 = curve_fit(fobj, ci.x, ci.c0, p0=pmean, **kwargs)
    # popt3, pcov3 = curve_fit(fobj, ci.x, ci.c1, p0=pmean, **kwargs)
    # print(pmean, popt)
    # print(popt, np.abs((popt - popt2)), np.abs((popt - popt3)))

    return popt, perr, ci


def contributions(var, rel=True, precision=2):
    if rel:
        for (name, error) in var.error_components().items():
            print("{}: {} %".format(name.tag, round(error ** 2 / var.s ** 2 * 100, precision)))
    else:
        for (name, error) in var.error_components().items():
            print("{}: {}".format(name.tag, round(error, precision)))


# test
def autokorrelation(t, y):
    mean = np.mean(y)
    y = y-mean
    print(len(t), t)
    w = np.linspace(0,  (t[len(t)-1]-t[0]),  len(t))
    Psi = np.zeros(len(t))
    divi = np.sum(y*y)
    for j in range(len(t)):
        for n in range(len(t)-j):
            Psi[j] += y[n]*y[n+j]
    return w, Psi/divi


def chi2_confidence_plot(ax, p=None, xmax=50, xvals=None):
    """
    Plot the reduced chi2 against degrees of freedom (dof) and draw confidence intervals for a given set of percentiles
    :param ax: ax object to plot on
    :param p: list of percentiles to plot (default: 1, 2, 3 sigma = [0.667, 0.95, 0.998])
    :param xmax: maximum x value (dof) to plot (default: 50)
    :param xvals: relative x position to plot the contour labels at (default: 0.4, 0.55, 0.7)
    :return: None
    """
    if p is None:
        p = [0.667, 0.95, 0.998]
    if xvals is None:
        xvals = [0.4, 0.55, 0.7]
    if len(p) != len(xvals):
        raise ValueError("p and xvals must be of the same length")

    cmap = sns.color_palette("rocket", as_cmap=True)
    dof = np.linspace(1, xmax, 250)
    for i, percentile in enumerate(p):
        ax.plot(dof, chi2.ppf((1+percentile)/2, dof)/dof, c=cmap(2/9 + i/5))
        ax.plot(dof, chi2.ppf((1-percentile)/2, dof)/dof, c=cmap(2/9 + i/5))

    lines = plt.gca().get_lines()
    for (i, line) in enumerate(lines):
        labelLine(line, xvals[i // 2]*xmax, label=r'${}\%$'.format(p[i//2]), align=True)


def get_polling_rate(df):
    # get time difference between two rows
    t = df["t"][1] - df["t"][0]
    # calculate polling rate and round
    return round(1/t)
