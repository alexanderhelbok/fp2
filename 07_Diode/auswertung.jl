using CSV, Roots, LaTeXStrings, PythonCall
using PhysicalConstants.CODATA2018: c_0

include(string(@__DIR__, "/../Source.jl"))
# include(string(@__DIR__, "/../SourceStatistics.jl"))
@py import matplotlib as mpl
@py import scipy.signal as ss

mpl.use("pgf")
mpl.use("TkAgg")

c = ["C0", "C1", "C4", "C3"]

begin
ax = plt.subplots(figsize=(7, 4))[1]
peaks1, peaks2 = [], []
tFSR, µ = [], 0
for i in 1:4

data1 = CSV.read(joinpath(@__DIR__, "data/TR0$(2*i).CSV"), DataFrame, header=["t", "CH1"], skipto=2)
data2 = CSV.read(joinpath(@__DIR__, "data/TR0$(2*i-1).CSV"), DataFrame, header=["t", "CH2"], skipto=2)
# merge data
data = innerjoin(data1, data2, on=:t)

data = data[argmin(data.CH2) + 170:argmax(data.CH2) + 200, :]
data.t .-= data.t[1]
data.t .*= 1e3

if i == 1
    µ = mean(data.CH1[1:200])
end

gauss(x, p) = @. p[1] * exp( - ( x - p[2] )^2 / ( 2 * p[3]^2 ) ) + µ

maxima, height = ss.find_peaks(data.CH1, height=0.5, distance=100)
# convert to julia array
maxima = pyconvert(Array, maxima) .+ 1

start, stop = maxima[1] - 75, maxima[1] + 50
start2, stop2 = maxima[2] - 75, maxima[2] + 50

popt, ci = bootstrap(gauss, data.t[start:stop], data.CH1[start:stop], p0=[1., data.t[maxima[1]], 0.2, 0.], redraw=false, unc=true)
popt2, ci2 = bootstrap(gauss, data.t[start2:stop2], data.CH1[start2:stop2], p0=[1., data.t[maxima[2]], 0.2, 0.], redraw=false, unc=true)

FWHM = 2 * sqrt(2 * log(2)) * popt[3]
FWHM2 = 2 * sqrt(2 * log(2)) * popt2[3]

push!(tFSR, measurement(nom(popt2[2]), nom(FWHM2)) * u"ms" - measurement(nom(popt[2]), nom(FWHM)) * u"ms") 

push!(peaks1, measurement(nom(popt[2]), nom(FWHM)) * u"ms")
push!(peaks2, measurement(nom(popt2[2]), nom(FWHM2)) * u"ms")

ax.scatter(data.t, data.CH1, s=5, c="gray")
# ax.scatter(data.t[start:stop], data.CH1[start:stop], s=5, c="crimson")
# ax.scatter(data.t[start2:stop2], data.CH1[start2:stop2], s=5, c="crimson")

x = range(data.t[1], stop=data.t[end], length=1000)
ax.plot(x, gauss(x, nom.(popt)), c=c[i], lw=2)
ax.plot(x, gauss(x, nom.(popt2)), c=c[i], lw=2)

end

# plot arrows
ax.arrow(0.4, 0.93, -0.275, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
ax.arrow(0.4, 0.93,  0.275, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
ax.text(0.7, 0.93, L"\overline{t_{\mathrm{FSR}}} = 5.4(2)\, \mathrm{ms}", fontsize=14, transform=ax.transAxes, verticalalignment="center")

ax.arrow(0.16, 0.85, -0.035, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
ax.arrow(0.16, 0.85,  0.035, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
ax.text(0.22, 0.85, L"\overline{\Delta t} = 0.5(2)\, \mathrm{ms}", fontsize=14, transform=ax.transAxes, verticalalignment="center")

ylim = ax.get_ylim()

# create dummy plots for legend
ax.plot(0, 0, c="gray", label=L"\mathrm{Fit}")
ax.scatter(0, 0, c="gray", label=L"\mathrm{Data}")

ax.set_xlim(0, 9.75)  
ax.set_ylim(ylim[0], ylim[1] + 0.2)

ax.set_ylabel(L"U\ (\mathrm{V})")
ax.set_xlabel(L"t\ (\mathrm{ms})")

ax.legend()

plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/peaks.pdf"), bbox_inches="tight")
plt.show()
end

FSR = mean(tFSR) 

Δt = mean([diff(sort(peaks1)), diff(sort(peaks2))])
νFSR = FSR/mean(Δt)*mean(diff(Δν)) |> u"GHz"

nom(νFSR)
err(νFSR)




dMax = measurement.([1.5, 1.5, 2, 2, 1, 1.55, 1, 2, 2, 1, 1.5, 1.5, 1.5, 2, 2], 0.1)
dAlex = measurement.([1.75, 1.75, 2, 1.5, 2, 1.75, 2, 1.5, 2, 2, 1, 2, 1.75, 2, 2], 0.1)

dMean = mean([dMax, dAlex]) .* u"mm"

# transform dMean as cumsum centered around the central vector entry
dMean = cumsum(dMean)
dMean .-= dMean[8]

D = (210 ± 2) * u"cm"

Δβ = @. atand(dMean / D)
d = 1/1200 * u"mm"
α = 45 ± 2
β = 8 ± 2
β2 = 65 ± 2
λ0 = d/2 * (sind(α) + sind(β2)) |> u"nm"
λ1 = @. d/2 * (sind(α) + sind(β2 + Δβ))
Δν = @. c_0* (1/λ1 - 1/λ0) |> u"GHz"

nom.(Δν) 

mean(diff(Δν)) |> u"GHz"

Δν[end] - Δν[1] |> u"GHz"

α = 25 ± 3

2*d*sind(α) |> u"nm"

nom(2*d*sind(α) |> u"nm")