using CSV, Roots, LaTeXStrings, PythonPlot
using PhysicalConstants.CODATA2018: e, k_B

include(string(@__DIR__, "/../Source.jl"))
# include(string(@__DIR__, "/../SourceStatistics.jl"))
@py import matplotlib as mpl

mpl.use("pgf")
mpl.use("TkAgg")

# load visibility data
begin
data = CSV.read(string(@__DIR__, "/data/visibility.txt"), DataFrame)
# add poisson error to data
for column in names(data)
    if column != "YAngle"
        data[!, column] = data[!, column] .± sqrt.(data[!, column])
    end
end
end

# define sine^2 function
model(x, p) = @. (p[1]*(x-p[2])^2 + p[3]) * (sin(p[4] * x + p[5])^2 + p[6])
parabola(x, p) = @. p[1]*(x-p[2])^2 + p[3]
sine(x, p) = @. p[4]*(sin(p[1] * x + p[2])^2 + p[3])
# fit to data


# define fourier series to fit to data

popt, ci = bootstrap(poly, nom.(data.YAngle), nom.(data[!, "Si_2"]), p0=[1.42e5, 7e3, 0.05, 9e3, 0.025, 2e3, 0.01, 2e3, 0.04], redraw=false, unc=true)
round.(nom.(popt),digits=7)
begin
    ax = subplots(figsize=(7, 4.5))[1]
    for i in 2:2:8
        ax.errorbar(nom.(data.YAngle), nom.(data[!, "Si_$i"]), yerr=err.(data[!, "Si_$i"]), fmt=".", capsize=2, )
        # plot fit
        ax.plot(ci.x, poly(ci.x, nom.(popt)), color="C$i")
        ax.plot(ci.x, poly(ci.x, nom.(popt) .+ [0, 3000, 0, 0, 0, 0, 0, 0, 0]), color="C1")
        ax.plot(ci.x, poly(ci.x, nom.(popt) .+ [0, 2500, 0, 3000, 0, -1000, 0, 0, 0]), color="C0")
    end
    xlabel("Polarizer B angle (°)")
    ylabel(L"$\mathrm{Coincidences}\ (cps)$")
    legend(title="Polarizer A")
    ylim(120000, 160000)
    tight_layout()
    # savefig(string(@__DIR__, "/bilder/plot1.pdf"), bbox_inches="tight")
    plt.show()
end

legendlabels = [L"$0 ^\circ$", L"$45 ^\circ$", L"$90 ^\circ$", L"$135 ^\circ$"]
markers = [".", "D"]
ms = [10, 5]

Vs, Vs2 = [], []
# plot coincidences
begin
    ax = subplots(2, 1, figsize=(10, 7.5), sharex=true)[1]
    for i in 1:8
        ax[0].errorbar(nom.(data.YAngle), nom.(data[!, "Si_$i"]), yerr=err.(data[!, "Si_$i"]), fmt=markers[(i+1)%2+1], capsize=2, c="C$(floor(Int, (i-1)/2))", ms=ms[(i+1)%2+1])
    end
    # create dummy plot for legend
    # create gray Diamond and Circle
    ax[0].errorbar(0, 0, yerr=1, fmt="D", c="gray", ms=5, label=L"$\mathrm{Polarizer\ A}$")
    ax[0].errorbar(0, 0, yerr=1, fmt=".", c="gray", ms=10, label=L"$\mathrm{Polarizer\ B}$")
    # create colored squares for angles
    ax[0].fill_between([0], [0], color="C0", label=L"$0 ^\circ$")
    ax[0].fill_between([0], [0], color="C1", label=L"$45 ^\circ$")
    ax[0].fill_between([0], [0], color="C2", label=L"$90 ^\circ$")
    ax[0].fill_between([0], [0], color="C3", label=L"$135 ^\circ$")

    ax[0].set_ylim(105e3, 16e4)
    ax[0].set_ylabel(L"$\mathrm{Total\ Counts}\ (\mathrm{per\ 5s})$")
    ax[0].legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.3))
    # set yticks to scitentific notation
    ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    for i in 1:4
        popt, ci = bootstrap(model, nom.(data.YAngle), nom.(data[!, "Co_$i"]), yerr=err.(data[!, "Co_$i"]), p0=[0.1, 150, 5000, 0.02, -1.5, 1.], redraw=false, unc=true)
        popt2, ci2 = bootstrap(sine, nom.(data.YAngle), nom.(data[!, "Co_$i"]), p0=[0.02, -1.5, 0.01, 5e3], redraw=false, unc=true)
        # calculate visibility    
        Sinemax, Sinemin = maximum(model(ci.x, popt)./parabola(ci.x, popt)), minimum(model(ci.x,popt)./parabola(ci.x,popt))
        Sine2max, Sine2min = maximum(sine(ci.x, popt2)), minimum(sine(ci.x,popt2))
        V = (Sinemax - Sinemin) / (Sinemax + Sinemin)
        V2 = (Sine2max - Sine2min) / (Sine2max + Sine2min)
        Vs = push!(Vs, V)
        Vs2 = push!(Vs2, V2)
        println("Visibility: ", V)
        println("Visibility2: ", V2)
        ax[1].errorbar(nom.(data.YAngle), nom.(data[!, "Co_$i"]), yerr=err.(data[!, "Co_$i"]), label=legendlabels[i], fmt=".", capsize=2, ms=15, mfc="C$(i-1)", mec="k", ecolor="k")
        # plot fit
        ax[1].plot(ci.x, model(ci.x, nom.(popt)), color="C$(i-1)", zorder=5, lw=2)
        ax[1].plot(ci.x, sine(ci.x, nom.(popt2)), color="C$(i-1)", zorder=5, lw=2)
    end
    ax[1].set_ylim(0, 6700)
    xlabel(L"$\mathrm{Polarizer\ B\ angle\ (^\circ)}$")
    ylabel(L"$\mathrm{Coincidences}\ (\mathrm{per\ 5s})$")
    legend(title=L"$\mathrm{Polarizer\ A}$", ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    xlim(0, 360)
    tight_layout()
    # savefig(string(@__DIR__, "/bilder/plot1.pdf"), bbox_inches="tight")
    plt.show()
end

V_HV = mean([Vs[1], Vs[3]])
V_DA = mean([Vs[2], Vs[4]])

V_HV2 = mean([Vs2[1], Vs2[3]])
V_DA2 = mean([Vs2[2], Vs2[4]])

# calculate psi+abundance using V = 1 - 2F
F = (1-V_DA)/2
# calculate psi- abundance


# calculate bell parameter
# create filename matrix
filemat = [["90;337,5", "90;247,5", "180;337,5", "180;247,5"],
            ["90;292,5", "90;202,5", "180;292,5", "180;202,5"],
            ["135;337,5", "135;247,5", "225;337,5", "225;247,5"],
            ["135;292,5", "135;202,5", "225;292,5", "225;202,5"]]
# create matrix to store Measurement values
mat = Matrix{Measurement{Float64}}(undef, 4, 4)


# calculate bell parameter
E(vec) = (vec[1] - vec[2] - vec[3] + vec[4])/sum(vec)

function loadmat(file)
    df = CSV.read(string(@__DIR__, "/data/$file.txt"), DataFrame, skipto=6, delim='\t', header=["a", "b", "c", "d"])
    counts = df[!, "d"] .± sqrt.(df[!, "d"])
    return (sum(counts) / length(counts))
end

for i in 1:4
    mat[i, :] = loadmat.(filemat[i])
end

mat

S = abs(E(mat[1, :]) - E(mat[2, :])) + abs(E(mat[3, :]) + E(mat[4, :]))

# calculate visibility from bell parameter
Smax = 2 * sqrt(2)
# fraction of psi- 
F = S / Smax
1-F
# visibility from F
V = 2 * F - 1
