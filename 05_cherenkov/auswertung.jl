using CSV, Roots, LaTeXStrings, PythonPlot, StatsBase, PythonCall, Unitful, DataFrames, LsqFit, Measurements
using PhysicalConstants.CODATA2018: e

include(string(@__DIR__, "/../Source.jl"))
# include(string(@__DIR__, "/../SourceStatistics.jl"))

@py import matplotlib as mpl
@py import matplotlib.pyplot as plt
@py import scipy.signal as ss
@py import numpy as np

mpl.use("pgf")
mpl.use("TkAgg")

function nom(x)
    return Measurements.value(ustrip.(x))
end

function err(x)
    return Measurements.uncertainty(ustrip.(x))
end

# estimate muon flux through cylinder with radius r and height h
r = 15 * u"cm" 
h = 25 * u"cm"
A = π * r^2

# calculate solid angle of cylinder
Ω = 2π * (1 - sqrt(1 - (r/h)^2)) |> u"sr"

dN = 2e3 *u"1/(s*m^2*sr)"
N = dN * A * Ω |> u"s^-1"

# fit gauss to peaks
gaussian(x, p) = @. p[1] * exp(-((x - p[2]) / p[3])^2)
gauss_int(p) = @. p[1] * p[3] * sqrt(pi) 

int_to_time(x) = @. x * 100 * u"μs"

begin
myflag = 0
integrals = []
for j in 1:9
    # load data
    begin
    data = CSV.read(string(@__DIR__, "/data/tek000$j.csv"), DataFrame, header=21)
    # convert time to μs
    data.TIME *= 1e6
    end

    # calculate find_peaks
    maxima, height = ss.find_peaks(data.CH1, height=0.001, distance=100)
    # convert to julia array
    maxima = pyconvert(Array, maxima) .+ 1

    # remove infs from maxima by filtering out dat.CH1[maxima]
    mask = isfinite.(data.CH1[maxima])  
    maxima = maxima[mask]
    # filter out peaks too close to start
    maxima = maxima[maxima .> 10]

    # integrate peaks
    for (i, peak) in enumerate(maxima)
        # find where falls blow 0.002
        threshold = 0.002
        zero_crossing1 =  maxima[i] - findfirst(data.CH1[maxima[i]:-1:1] .< threshold)

        # fit gaussian to peak
        popt = curve_fit(gaussian, data.TIME[zero_crossing1:peak], data.CH1[zero_crossing1:peak], [0.1, data.TIME[peak], 1e-3]).param

        # plot data for specific peak
        if (myflag < 10)
            myflag += 1
            if myflag == 3
                popt = curve_fit(gaussian, data.TIME[zero_crossing1:peak+10], data.CH1[zero_crossing1:peak+10], [0.1, data.TIME[peak], 1e-3]).param

                ax = subplots(figsize=(4, 3))[1]
                plotx = LinRange(data.TIME[peak+50], data.TIME[peak-50], 1000) .- data.TIME[peak]
                scatter(data.TIME[peak-50:peak+50] .- data.TIME[peak], data.CH1[peak-50:peak+50], label=L"\mathrm{data}", c="gray", s=10)
                scatter(data.TIME[zero_crossing1:peak] .- data.TIME[peak], data.CH1[zero_crossing1:peak], c="crimson", s=10)
                plot(plotx, gaussian(plotx, popt - [0, data.TIME[peak], 0]), label=L"\mathrm{fit}", c="C1")

                ylabel(L"\mathrm{Voltage}\ (\mathrm{V})")
                legend()
                tight_layout()
                # savefig(string(@__DIR__, "/bilder/peak_fit.pdf"), bbox_inches="tight")
                plt.show()
            end
        end
        
        # check if fit actually fits data by checking peak height and fit height at peak
        if (abs(data.CH1[peak] - maximum(gaussian(data.TIME, popt))) < 0.1)
            push!(integrals, gauss_int(popt))
        end
    end
end
end
# filter out too large values
integrals2 = Vector{Float64}(integrals[integrals .< .5e-3])

# int_to_time(data.TIME[2] - data.TIME[1]) |> u"ns"

# bin data
begin
bins = 200
bin_width = 1e-3 / bins
# use python to calculate histogram
binned_data, x = np.histogram(integrals2, bins=bins)
binned_data = pyconvert(Array, binned_data)
binned_data = measurement.(binned_data, sqrt.(binned_data))
x = pyconvert(Array, x)
newx = (x[1:end-1] + x[2:end]) / 2
end
print(bin_width)

# fit functions
begin
start1, stop1 = 1, 12
start2, stop2 = 50, 123

# fit gaussian to data
fit1 = curve_fit(gaussian, newx[start1:stop1], nom.(binned_data[start1:stop1]), [500., 1e-6, 1e-5])
popt1 = measurement.(fit1.param, stderror(fit1))

fit2 = curve_fit(gaussian, newx[start2:stop2], nom.(binned_data[start2:stop2]), [50., 2e-4, 1e-4])
popt2 = measurement.(fit2.param, stderror(fit2))
end

# compute difference
HWHM1 = sqrt(2 * log(2)) * popt1[3]
HWHM2 = sqrt(2 * log(2)) * popt2[3]

mu1 = measurement(nom(popt1[2]), nom(HWHM1))
mu2 = measurement(nom(popt2[2]), nom(HWHM2))

systematic_error = 3e-5
diff = mu2 - mu1 
# diff = mu2 - mu1 + systematic_error

# calculate amplification
R = 50 * u"Ω"
A = diff*u"V*μs"/(R*e) |> NoUnits

# calculate anode dark current
I_dark = mu1 * u"V*μs" / R |> u"nA*µs"

plotx = LinRange(0, 5e-4, 1000)

begin
    start1, stop1 = 10, 15
    start2, stop2 = 50, 123
    
    # fit gaussian to data
    fit1 = curve_fit(gaussian, newx[start1:stop1], nom.(binned_data[start1:stop1]), [500., 1e-6, 1e-5])
    popt1 = measurement.(fit1.param, stderror(fit1))
    
    fit2 = curve_fit(gaussian, newx[start2:stop2], nom.(binned_data[start2:stop2]), [50., 2e-4, 1e-4])
    popt2 = measurement.(fit2.param, stderror(fit2))
end

# plot histogram
begin
ax = subplots(3, 1, figsize=(10, 9), sharex=true)[1]

ax[0].hist(integrals2, bins=250, color="C0", edgecolor="black", alpha=0.7, histtype="stepfilled", label="\$\\mathrm{binned\\ data}\$")

for i in 1:2
ax[i].errorbar(newx, nom.(binned_data), err.(binned_data), color="gray", fmt=".", ecolor="k", label=L"\mathrm{data}")
ax[i].errorbar(newx[start1:stop1], nom.(binned_data[start1:stop1]), err.(binned_data[start1:stop1]), color="crimson", fmt=".", ecolor="k")
ax[i].errorbar(newx[start2:stop2], nom.(binned_data[start2:stop2]), err.(binned_data[start2:stop2]), color="deepskyblue", fmt=".", ecolor="k")

ax[i].plot(plotx[1:stop1*15], gaussian(plotx[1:stop1*15], nom.(popt1)), color="C1", zorder=5)
ax[i].plot(plotx, gaussian(plotx, nom.(popt2)), color="C0", zorder=5)

# create dummy plot for legend
ax[i].plot([0], [0], color="gray", label=L"\mathrm{Fit}")
# ax[i].plot(plotx, gaussian(plotx, [3, 3.2e-4, 1e-5]), color="C3", label="Fit", zorder=5)
# ax[i].plot(plotx, gaussian(plotx, nom.(popt1)) .+ gaussian(plotx, nom.(popt2)) .+ gaussian(plotx, [3, 3.2e-4, 1e-5]), color="C2", label="Fit")
end

# add fit parameters to ax[1]
ax[1].text(0.05, 0.75, L"A = 4.0(3) \times 10^2", transform=ax[1].transAxes)
ax[1].text(0.05, 0.65, L"\mu = 3.4(5) \times 10^{-6}\ \mathrm{V\mu s}", transform=ax[1].transAxes)
ax[1].text(0.05, 0.55, L"\sigma = 3.2(7) \times 10^{-6}\ \sqrt{\mathrm{V\mu s}}", transform=ax[1].transAxes)

rect = mpl.patches.FancyBboxPatch((0.05, 0.55), 0.27, 0.3, linewidth=1.5, edgecolor="C1", facecolor="none", transform=ax[1].transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax[1].add_patch(rect)

ax[1].text(0.65, 0.4, L"A = 28.9(7)", transform=ax[1].transAxes)
ax[1].text(0.65, 0.3, L"\mu = 19.1(3) \times 10^{-5}\ \mathrm{V\mu s}", transform=ax[1].transAxes)
ax[1].text(0.65, 0.2, L"\sigma = 9.2(4) \times 10^{-5}\ \sqrt{\mathrm{V\mu s}}", transform=ax[1].transAxes)

rect = mpl.patches.FancyBboxPatch((0.65, 0.18), 0.2, 0.3, linewidth=1.5, edgecolor="C0", facecolor="none", transform=ax[1].transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax[1].add_patch(rect)

# add fit function as text
ax[1].text(0.45, 0.65, L"f(x) = A \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)", transform=ax[1].transAxes, fontsize=16)

ax[0].legend()
ax[1].legend()
ax[2].legend()

ax[0].set_ylabel(L"\mathrm{Counts}")
ax[1].set_ylabel(L"\mathrm{Counts}")
ax[2].set_ylabel(L"\mathrm{Counts}")
ax[2].set_yscale("log")

ax[2].set_ylim(1, 1e3)

# set xaxis ticks to scientific
ax[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))

xlim(-1e-5, .5e-3)
xlabel(L"\mathrm{Charge}\ (\mathrm{V}\mathrm{s})")
tight_layout()
# savefig(string(@__DIR__, "/bilder/histogram.pdf"), bbox_inches="tight")
plt.show()
end


