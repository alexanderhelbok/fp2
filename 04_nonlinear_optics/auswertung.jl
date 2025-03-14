using CSV, Roots, LaTeXStrings, PythonPlot
using PhysicalConstants.CODATA2018: e, k_B

include(string(@__DIR__, "/../Source.jl"))
# include(string(@__DIR__, "/../SourceStatistics.jl"))
@py import matplotlib as mpl

mpl.use("pgf")
mpl.use("TkAgg")

# load visibility data
begin
data = CSV.read(string(@__DIR__, "/data/data1.csv"), DataFrame, header=["Angle", "P1", "P2", "P3"], skipto=2)
# create new column for the mean of the three measurements
data[!, "P"] = measurement.(mean.(eachrow(data[:, 2:4])), std.(eachrow(data[:, 2:4])))
# create relaitve angle to the polarizer by subtracting the angle where P is maximal
data.Angle .-= data.Angle[18]
# shift y up by min value
# data.P .-= minimum(data.P)
x = data.Angle[data.Angle .> 0]
y = data.P[data.Angle .> 0]
# convert angle to power using cos^2 law
data.x = @. cos(2*(data.Angle-0)*pi/180)^2
x = @. cos(2*(x-0)*pi/180)^2
end

model(x, p) = @. p[1]*(x - p[2])^2 + p[3]
parabola(x, p) = @. p[1]*x^2 + p[2]

# fit to data
popt, ci = bootstrap(parabola, nom.(x), nom.(y), yerr=err.(y), p0=[14., -3.], redraw=false, unc=true)

# plot data
begin
ax = subplots(figsize=(7, 4.5))[1]
# myerrorbar(data.x, data.P, fmt=".k", capsize=3, label="Data")
myerrorbar(x, y, fmt=".k", capsize=3, label="Data")
xlim = ax.get_xlim()
plot(ci.x, parabola(ci.x, nom.(popt)), label="Fit")

# add fit parameters to plot
text(0.08, 0.6, L"$f(x) = a \cdot x^2 + c$", transform=ax.transAxes, fontsize=14)
text(0.08, 0.5, L"$a = 15.7(2)\ \mathrm{mW}$", transform=ax.transAxes, fontsize=14)
text(0.08, 0.4, L"$c = -3.23(7)\ \mathrm{mW}$", transform=ax.transAxes, fontsize=14)

rect = mpl.patches.FancyBboxPatch((0.07, 0.4), 0.27, 0.25, linewidth=1.5, edgecolor="C0", facecolor="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax.add_patch(rect)

ax.set_xlabel(L"P\ (\mathrm{a.u.})")
ax.set_ylabel(L"I_{2\omega}\ (\mathrm{mW})")
ax.set_xlim(xlim)
ax.legend()
plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/plot1.pdf"), bbox_inches="tight")
plt.show()
end

# load data
begin
data = CSV.read(string(@__DIR__, "/data/data2.csv"), DataFrame, header=["Angle", "P1", "P2", "P3"], skipto=2)
# create new column for the mean of the three measurements
data[!, "P"] = measurement.(mean.(eachrow(data[:, 2:4])), std.(eachrow(data[:, 2:4])))
end

# convert micrometer screwticks to angles, 6 degrees = 1.9  micrometer ticks
data.Angle = data.Angle .* 6/1.9

# fit sinc function to data
sincfit(x, p) = @. p[1] * sinc(p[2] * x + p[3]) + p[4]

popt, ci = bootstrap(sincfit, nom.(data.Angle), nom.(data.P), yerr=err.(data.P), p0=[20., 0.02, 0., 0.], redraw=false, unc=true)

# plot data
begin
ax = subplots(figsize=(7, 4.5))[1]
myerrorbar(data.Angle, data.P, fmt=".k", label="Data", capsize=3)
xlim = ax.get_xlim()
plot(ci.x, sincfit(ci.x, nom.(popt)), label="Fit")

# add fit parameters to plot
text(0.08, 0.5, L"$f(x) = a \cdot \mathrm{sinc}(b \cdot x + c) + d$", transform=ax.transAxes, fontsize=14)
text(0.08, 0.4, L"$a = 16.02(12)\ \mathrm{mW}$", transform=ax.transAxes, fontsize=14)
text(0.08, 0.3, L"$b = 0.0270(4)$", transform=ax.transAxes, fontsize=14)
text(0.08, 0.2, L"$c = -0.112(15)$", transform=ax.transAxes, fontsize=14)
text(0.08, 0.1, L"$d = 2.44(13)\ \mathrm{mW}$", transform=ax.transAxes, fontsize=14)

rect = mpl.patches.FancyBboxPatch((0.07, 0.1), 0.4, 0.45, linewidth=1.5, edgecolor="C0", facecolor="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax.add_patch(rect)

ax.set_xlabel(L"\theta\ (^\circ)")
ax.set_ylabel(L"I_{2\omega}\ (\mathrm{mW})")
ax.set_xlim(xlim)
ax.legend()
plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/plot2.pdf"), bbox_inches="tight")
plt.show()
end


# load data
begin
data = CSV.read(string(@__DIR__, "/data/data3.csv"), DataFrame, header=["Angle", "P", "Perr", "V1", "V2", "V3"], skipto=2)
# create new column for the mean of the three measurements
data[!, "V"] = measurement.(mean.(eachrow(data[:, 4:6])), std.(eachrow(data[:, 4:6])))
data[!, "P"] = measurement.(data.P, data.Perr/1000)
data.P = data.P/maximum(data.P)
end

popt, ci = bootstrap(parabola, nom.(data.P./maximum(data.P)), nom.(data.V), yerr=err.(data.V), xerr=err.(data.P./maximum(data.P)),  p0=[1., 0.], redraw=false, unc=true)
nom(popt[2]), err(popt[2])
# plot data
begin
ax = subplots(figsize=(7, 4.5))[1]
myerrorbar(data.P, data.V, fmt=".k", capsize=3, label=L"\mathrm{Data}")
xlim, ylim = ax.get_xlim(), ax.get_ylim()

plot(ci.x, parabola(ci.x, nom.(popt)), label=L"\mathrm{Fit}")

# write fit parameters in plot
text(0.08, 0.6, L"$f(x) = a \cdot x^2 + c$", transform=ax.transAxes, fontsize=14)
text(0.08, 0.5, L"$a = 20.7(3)\ \mathrm{mW}$", transform=ax.transAxes, fontsize=14)
text(0.08, 0.4, L"$c = -0.74(5)\ \mathrm{mW}$", transform=ax.transAxes, fontsize=14)

rect = mpl.patches.FancyBboxPatch((0.07, 0.4), 0.27, 0.25, linewidth=1.5, edgecolor="C0", facecolor="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax.add_patch(rect)

ax.set_xlabel(L"I_{\mathrm{Laser}}\ (\mathrm{a.u.})")
ax.set_ylabel(L"I_{2\omega}\ (\mathrm{mW})")
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.legend()
plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/plot3.pdf"), bbox_inches="tight")
plt.show()
end

Plaser = 11.46 ± 0.002
Pfilter = [0.101±0.0011, 0.089±0.002, 0.096±0.009]
Pblue = mean([18.3, 17.7, 17.7]) ± std([18.3, 17.7, 17.7])
Pred = mean([55.4, 55.3, 55.4]) ± std([55.4, 55.3, 55.4])

# calculate filter efficiency
η = Pfilter ./ Plaser
η = η[1] * η[2] * η[3] * 100
Plaser / η /1000
η2 = Pblue / (Plaser / η + Pblue) * 100
