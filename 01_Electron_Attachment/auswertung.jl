using CSV, Roots, LaTeXStrings, PythonPlot
using PhysicalConstants.CODATA2018: e, k_B

include(string(@__DIR__, "/../Source.jl"))
# include(string(@__DIR__, "/../SourceStatistics.jl"))
@py import matplotlib as mpl

mpl.use("pgf")
mpl.use("TkAgg")

# load data
df = CSV.read(joinpath(@__DIR__, "data/group3_Cl-_I01F_35.txt"), DataFrame, skipto=8, header=["x","y1","y2","ymean"])

df.ymean .*= 1e-3

# define gaussian
gaussian(x, p) = @. p[1] * exp( - 4*log(2)*( x - p[2] )^2 / ( p[3]^2 ) )

start1 = 1
stop1 = 80
start2 = 80
stop2 = 220
# fit to data
popt1, ci1 = bootstrap(gaussian, df.x[start1:stop1], df.ymean[start1:stop1], p0=[14., 2., 1.], redraw=false, unc=true)
popt2, ci2 = bootstrap(gaussian, df.x[start2:stop2], df.ymean[start2:stop2], p0=[1., 3., .5], redraw=false, unc=true)

# plot data
begin
ax = subplots(figsize=(7, 4.5))[1]
scatter(df.x[start1:stop1], df.ymean[start1:stop1], color="k", label="data")
plot(ci1.x, gaussian(ci1.x, nom.(popt1)), label="fit", color="C0")
scatter(df.x[start2:stop2], df.ymean[start2:stop2], color="black")
plot(df.x[start2:stop2], gaussian(df.x[start2:stop2], nom.(popt2)), label="fit", color="C1")

text(0.3, 0.88, L"$f(x) = A \cdot \exp\left(- 4\log(2)\frac{(x-x_0)^2}{\mathrm{FWHM}^2}\right)$", transform=ax.transAxes, fontsize=14)
text(0.34, 0.68, L"$x_0 = 2.1373(13)\ \mathrm{eV}$", transform=ax.transAxes, fontsize=14)
text(0.34, 0.58, L"$\mathrm{FWHM} = 0.147(3)\ \mathrm{eV}$", transform=ax.transAxes, fontsize=14)

text(0.34, 0.38, L"$x_0 = 2.992(4)\ \mathrm{eV}$", transform=ax.transAxes, fontsize=14)
text(0.34, 0.28, L"$\mathrm{FWHM} = 0.713(10)\ \mathrm{eV}$", transform=ax.transAxes, fontsize=14)

rect1 = mpl.patches.FancyBboxPatch((0.345, 0.56), 0.3, 0.16, linewidth=1.5, edgecolor="C0", facecolor="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax.add_patch(rect1)

rect2 = mpl.patches.FancyBboxPatch((0.345, 0.26), 0.325, 0.16, linewidth=1.5, edgecolor="C1", facecolor="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax.add_patch(rect2)

xlabel(L"E_{\mathrm{kin}}\ (\mathrm{eV})")
ylabel(L"\mathrm{counts}\ (\mathrm{kcps})")
legend()
tight_layout()
# savefig(string(@__DIR__, "/bilder/plot1.pdf"), bbox_inches="tight")
plt.show()
end

# print fit parameters
println("ΔE_1 =  $(popt1[2]) eV, FWHM = $(abs(popt1[3])) eV")
println("ΔE_2 =  $(popt2[2]) eV, FWHM = $(abs(popt2[3])) eV")
E1 = popt1[2]
E2 = popt2[2]
# print calibrated energy of second peak
println("E_1 = $(E2 - E1) eV")

# load SF6- data
df = CSV.read(joinpath(@__DIR__, "data/group3_Sf6-_I025_146p6.txt"), DataFrame, skipto=8, header=["x","y1","y2","ymean"])

# convert ymean to kcps
df.ymean .*= 1e-3

# fit to data
popt, ci = bootstrap(gaussian, df.x, df.ymean, p0=[40., 2., .1], redraw=false, unc=true)

begin
ax = subplots(figsize=(7, 3))[1]
ax.scatter(df.x, df.ymean, label="data", color="black")
xlims, ylims = xlim(), ylim()
ax.plot(ci.x, gaussian(ci.x, nom.(popt)), label="fit")

text(0.5, 0.8, L"$f(x) = A \cdot \exp\left(- 4\log(2)\frac{(x-x_0)^2}{\mathrm{FWHM}^2}\right)$", transform=ax.transAxes, fontsize=14)
text(0.54, 0.5, L"$x_0 = 2.1369(5)\ \mathrm{eV}$", transform=ax.transAxes, fontsize=14)
text(0.54, 0.3, L"$\mathrm{FWHM} = 0.1360(11)\ \mathrm{eV}$", transform=ax.transAxes, fontsize=14)

rect1 = mpl.patches.FancyBboxPatch((0.545, 0.28), 0.33, 0.3, linewidth=1.5, edgecolor="C0", facecolor="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax.add_patch(rect1)

# disable ticks on top
ax.tick_params(axis="x", top=false)

xlabel(L"E_{\mathrm{kin}}\ (\mathrm{eV})")
ylabel(L"\mathrm{counts}\ (\mathrm{kcps})")
legend()
tight_layout()
xlim(xlims)
# savefig(string(@__DIR__, "/bilder/plot2.pdf"), bbox_inches="tight")
plt.show()
end

E_SF6 = nom(popt[2]) ± err(popt[2])

println("ΔE =  $(popt[2]) eV, FWHM = $(abs(popt[3])) eV")
println("E_SF6 - E_1 = $(E_SF6 - E1) eV")

# load SF5- data
df = CSV.read(joinpath(@__DIR__, "data/group3_SF5-_I037_127p7.txt"), DataFrame, skipto=8, header=["x","y1","y2","y3","y4","y5","ymean"])

# convert ymean to kcps
df.ymean .*= 1e-3

# fit to data
popt, ci = bootstrap(gaussian, df.x, df.ymean, p0=[.6, 2.5, .6], redraw=false, unc=true)

begin
ax = subplots(figsize=(7, 3.5))[1]
ax.scatter(df.x, df.ymean, label="data", color="black")

xlims, ylims = xlim(), ylim()
ax.plot(ci.x, gaussian(ci.x, nom.(popt)), label="fit")

text(0.5, 0.8, L"$f(x) = A \cdot \exp\left(- 4\log(2)\frac{(x-x_0)^2}{\mathrm{FWHM}^2}\right)$", transform=ax.transAxes, fontsize=14)
text(0.6, 0.6, L"$x_0 = 2.462(7)\ \mathrm{eV}$", transform=ax.transAxes, fontsize=14)
text(0.6, 0.45, L"$\mathrm{FWHM} = 0.617(15)\ \mathrm{eV}$", transform=ax.transAxes, fontsize=14)

rect1 = mpl.patches.FancyBboxPatch((0.595, 0.38), 0.33, 0.3, linewidth=1.5, edgecolor="C0", facecolor="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax.add_patch(rect1)

xlabel(L"E_{\mathrm{kin}}\ (\mathrm{eV})")
ylabel(L"\mathrm{counts}\ (\mathrm{kcps})")
xlim(xlims)
legend()
tight_layout()
# savefig(string(@__DIR__, "/bilder/plot3.pdf"), bbox_inches="tight")
plt.show()
end

# print fit parameters
println("ΔE =  $(popt[2]) eV, FWHM = $(abs(popt[3])) eV")
E_SF5 = popt[2]
println("E_SF5 - E_1 = $(E_SF5 - E1) eV")

