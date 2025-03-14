using CSV, Roots, LaTeXStrings, PythonCall, Bessels
using PhysicalConstants.CODATA2018: R

include(string(@__DIR__, "/../Source.jl"))
# include(string(@__DIR__, "/../SourceStatistics.jl"))
@py import matplotlib as mpl
@py import scipy.signal as ss

mpl.use("pgf")
mpl.use("TkAgg")

# parabola such that p[1] is the maximum
parabola(x, p) = @. p[1] * (x - p[2])^2 + p[3]
func(x, p) = @. p[1] * exp(-p[2] * x)

# get reference intensity from water measurement
begin
    ax = plt.subplots(1, 2, figsize=(7, 2.5), sharey=true)[1]
    data = CSV.read(joinpath(@__DIR__, "data/SP_water.txt"), DataFrame, skipto=500, header=79, delim=';')
    # only keep Wavelength and Raw data #1
    data = data[:, ["Wavelength", "Raw data #1"]]
    # rename columns
    rename!(data, Dict("Wavelength" => "x", "Raw data #1" => "y"))
    ax[0].plot(data.x, data.y)

    data = data[200:300, :]

    maxima = argmax(data.y)
    # fit parabola to maxima
    popt, ci = bootstrap(parabola, data.x[maxima-5:maxima+5], data.y[maxima-5:maxima+5], p0=[-10., 520., maximum(data.y)], redraw=false, unc=true)

    # calculate maximum of parabola
    I_0 = parabola(nom(popt[2]), popt)

    ax[0].set_xlabel(L"\lambda\ (\mathrm{nm})")
    ax[0].set_ylabel(L"I\ (\mathrm{a.u.})")
    ax[1].set_xlabel(L"\lambda\ (\mathrm{nm})")

    ax[0].set_xlim(400, 900)
    ax[1].set_xlim(500, 540)

    ax[1].scatter(data.x, data.y, s=5)
    # plot fit
    ax[1].plot(ci.x, parabola(ci.x, nom.(popt)), c="C1")


    plt.tight_layout()
    # plt.savefig(joinpath(@__DIR__, "bilder/plot0.pdf"), bbox_inches="tight")
    plt.show()
end

# load data
c = ["C0", "C1", "C4"]
begin
    ax = plt.subplots(figsize=(7, 3.5))[1]
    ks = []
    stops = [44, 39, 22]
    for j in 1:3
        peaks = []
        for i in 0:stops[j]

        # open file and replace comma by dots
        data = CSV.read(joinpath(@__DIR__, "data/Messung$j/SP_$i.txt"), DataFrame, skipto=500, header=79)
        # only keep Wavelength and Raw data #1
        data = data[:, ["Wavelength", "Raw data #1"]]
        # rename columns
        rename!(data, Dict("Wavelength" => "x", "Raw data #1" => "y"))
        data = data[200:300, :]

        maxima = argmax(data.y)
        # fit parabola to maxima
        popt, ci = bootstrap(parabola, data.x[maxima-5:maxima+5], data.y[maxima-5:maxima+5], p0=[-10., 520., maximum(data.y)], redraw=false, unc=true)
        # println(popt)

        
        # calculate maximum of parabola
        y_max = parabola(nom(popt[2]), popt)
        push!(peaks, log10(I_0 / y_max))

        # scatter(data.x, data.y)
        # plot fit
        # plot(ci.x, parabola(ci.x, nom.(popt)))
        end
        # create array of datapoints spaced by 30
        x = 30:30:30*length(peaks)
        # plot peaks
        plt.errorbar(x, nom.(peaks), xerr=ones(length(x)), yerr=err.(peaks), fmt=".", capsize=3, mfc=c[j], mec="k", ms=10, ecolor="k")
        xlim, ylim = plt.xlim(), plt.ylim()
        # fit to data
        if j != 3
            popt, ci = bootstrap(func, x[7:end], nom.(peaks[7:end]), xerr=ones(length(x)-6), yerr=err.(peaks[7:end]), p0=[.5, 0.001], redraw=false, unc=true)
        else
            popt, ci = bootstrap(func, x[5:end], nom.(peaks[5:end]), xerr=ones(length(x)-4), yerr=err.(peaks[5:end]), p0=[.5, 0.005], redraw=false, unc=true)
        end
        println(popt)
        push!(ks, popt[2])
        # plot fit
        plot(ci.x, func(ci.x, nom.(popt)), c=c[j], lw=2)
        plt.xlim(xlim)
        plt.ylim(ylim)
    end
    # create dummy plot for legend
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.plot(0, 0, c="gray", lw=2, label=L"\mathrm{Fit}")
    plt.plot(0, 0, c="crimson", lw=2, ls="--", label=L"\mathrm{Excluded}")
    plt.errorbar(0, 0, xerr=0, yerr=0, fmt=".", capsize=3, mfc="gray", mec="k", ms=10, ecolor="k", label=L"\mathrm{Data}")
    plt.fill_between(0, [0, 0], color=c[1], label=L"23.2(5)\ ^{\circ}\mathrm{C}", ec="k")
    plt.fill_between(0, [0, 0], color=c[2], label=L"26.3(5)\ ^{\circ}\mathrm{C}", ec="k")
    plt.fill_between(0, [0, 0], color=c[3], label=L"35.1(5)\ ^{\circ}\mathrm{C}", ec="k")

    plt.xlim(xlim)  
    plt.ylim(ylim)

    # draw box around excluded datapoints
    rect = mpl.patches.FancyBboxPatch((0.05, 0.6), 0.11, 0.35, lw=2, ls="--", ec="crimson", fc="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
    ax.add_patch(rect)

    # add fitfunction and parameters as text
    plt.text(0.38, 0.875, L"f(t) = A \exp(-k t)", fontsize=14, transform=ax.transAxes)
    plt.text(0.38, 0.775, L"k = 1.486(5) \times 10^{-3}\, \mathrm{s}^{-1}", fontsize=14, transform=ax.transAxes, c=c[1])
    plt.text(0.38, 0.675, L"k = 1.814(6) \times 10^{-3}\, \mathrm{s}^{-1}", fontsize=14, transform=ax.transAxes, c=c[2])
    plt.text(0.38, 0.575, L"k = 6.31(4) \times 10^{-3}\, \mathrm{s}^{-1}", fontsize=14, transform=ax.transAxes, c=c[3])


    plt.xlabel(L"t\ (\mathrm{s})")
    plt.ylabel(L"\log_{10}\left(\frac{I_0}{I}\right)")

    # handles, labels = ax.get_legend_handles_labels()
    # handles, labels = pyconvert(Array, handles), pyconvert(Array, labels)
    # println(length(handles))
    # order = [1, 2, 3, 4, 5, 6]
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], borderaxespad=0.5)
    plt.legend(borderaxespad=0.5)
    plt.tight_layout()
    # plt.savefig(joinpath(@__DIR__, "bilder/plot1.pdf"), bbox_inches="tight")
    plt.show()
end

# plot ks against temperature and fit exponential
func2(T, p) = @. p[1] * exp(-p[2]/(ustrip(R)*T))
Tlit = [10, 20, 30, 40] .+ 273.15
klit = [2.39e-3, 0.0085, 0.03, 0.085] ./ 60
begin
    ax = plt.subplots(figsize=(7, 3.5))[1]
    T = measurement.([23.2, 26.3, 35.1], [0.5, 0.5, 0.5]) .+ 273.15

    ax.errorbar(nom.(T), nom.(ks), xerr=err.(T), yerr=err.(ks), fmt=".", capsize=3, mfc="C0", mec="k", ms=12, ecolor="k")
    ax.scatter(Tlit, klit, c="C1", s=35, edgecolors="k")

    popt, ci = bootstrap(func2, nom.(T), nom.(ks), xerr=err.(T), yerr=err.(ks), p0=[1e8, 1e4], redraw=false, unc=true)
    println(popt)

    popt2, ci2 = bootstrap(func2, Tlit, klit, p0=[1.6e11, 8.4e4], redraw=false, unc=true)
    println(popt2)

    # add fitparameters to plot
    plt.text(0.03, 0.875, L"f(T) = A\exp\left( -E_a/RT \right)", fontsize=14, transform=ax.transAxes)

    plt.text(0.03, 0.725, L"A = 3.4(3) \times 10^{8}\, \mathrm{s}^{-1}", fontsize=14, transform=ax.transAxes)
    plt.text(0.03, 0.625, L"E_a = 6.362(12) \times 10^{4}\, \mathrm{J/mol}", fontsize=14, transform=ax.transAxes)

    rect = mpl.patches.FancyBboxPatch((0.04, 0.6), 0.37, 0.2, lw=1.5, ec="C0", fc="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
    ax.add_patch(rect)

    plt.text(0.03, 0.475, L"A = 2(8) \times 10^{11}\, \mathrm{s}^{-1}", fontsize=14, transform=ax.transAxes)
    plt.text(0.03, 0.375, L"E_a = 8.4(1.3) \times 10^{4}\, \mathrm{J/mol}", fontsize=14, transform=ax.transAxes)

    rect = mpl.patches.FancyBboxPatch((0.04, 0.35), 0.37, 0.2, lw=1.5, ec="C1", fc="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
    ax.add_patch(rect)

    xlim, ylim = plt.xlim(), plt.ylim()

    # dummy plots for legend
    plt.fill_between(0, [0, 0], color="C0", label=L"\mathrm{Data}", ec="k")
    plt.fill_between(0, [0, 0 ], color="C1", ec="k", label=L"\mathrm{Literature}")

    ax.plot(ci2.x, func2(ci2.x, nom.(popt)), c="C0", lw=2)
    ax.plot(ci2.x, func2(ci2.x, nom.(popt2)), c="C1", lw=2)

    plt.xlabel(L"T\ (K)")
    plt.ylabel(L"k\ (\mathrm{s}^{-1})")

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.legend(loc="center right")
    plt.tight_layout()
    # plt.savefig(joinpath(@__DIR__, "bilder/plot2.pdf"), bbox_inches="tight")
    plt.show()
end
