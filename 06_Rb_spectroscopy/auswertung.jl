using CSV, Roots, LaTeXStrings, PythonCall, PythonPlot
using PhysicalConstants.CODATA2018: c_0

include(string(@__DIR__, "/../Source.jl"))
# include(string(@__DIR__, "/../SourceStatistics.jl"))
@py import matplotlib as mpl

mpl.use("pgf")
mpl.use("TkAgg")


L = 10 * u"cm"

begin
    df = CSV.read(joinpath(@__DIR__, "data2/TEK00000.CSV"), DataFrame, header=["t", "CH1"], skipto=17)
    df.CH1 /= maximum(df.CH1)  
    df = df[400:end-100, :]
    
    # scatter(df.t, df.CH1)
    # plt.show()

    FSR = c_0/(2*L) / u"GHz"|> NoUnits
    
    # find 2 maxima in the data
    maxima, height = ss.find_peaks(df.CH1, height=0.02, distance=100)
    # convert to julia array
    maxima = pyconvert(Array, maxima) .+ 1
    
    dt = df.t[maxima[end]] - df.t[maxima[1]]
    
    t_to_f(t) = @. FSR / dt * (t - t[1])
    df.f = t_to_f(df.t)
    
    didx = maxima[end] - maxima[1]
    
    idx_to_f(idx) = @. FSR / didx * (idx - 1)
    f_to_idx(f) = @. didx / FSR * f + 1 |> round |> Int
    
    plot(df.f, df.CH1)
    scatter(df.f[maxima], df.CH1[maxima])
end
plt.show()

lorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4]
fivelorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4] / ( 1 + ( ( x - p[5] )/ p[6] )^2 ) + p[7] / ( 1 + ( ( x - p[8] )/ p[9] )^2 ) + p[10] / ( 1 + ( ( x - p[11] )/ p[12] )^2 ) + p[13] / ( 1 + ( ( x - p[14] )/ p[15] )^2 ) + p[16]



idx_to_f(maxima[1]), idx_to_f(maxima[2]), idx_to_f(maxima[3]), idx_to_f(maxima[4]), idx_to_f(maxima[5])

begin
    ax = plt.subplots(figsize=(7, 3.5))[1]
    # mid
    popt, ci = bootstrap(fivelorentzian, df.f, df.CH1, p0=[1., idx_to_f(maxima[1]), 0.003, .1, idx_to_f(maxima[2]), 0.003, .3, idx_to_f(maxima[3]), 0.003, .1, idx_to_f(maxima[4]), 0.003, 1., idx_to_f(maxima[5]), 0.003, 0.01], redraw=false, unc=true)

    println(popt)

    scatter(df.f, df.CH1, s=10, label=L"\mathrm{data}")

    xlims, ylims = xlim(), ylim() 
    x = range(ci.x[1], ci.x[end], length=10000)
    plot(x, fivelorentzian(x, nom.(popt)), label=L"\mathrm{multilorentzian\ fit}", color="C1", zorder=5)

    arrow(0.55, 0.5,  0.25, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    arrow(0.55, 0.5, -0.35, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    ax.text(0.45, 0.53, L"\nu_{\mathrm{FSR}} \stackrel{!}{=} 1.5\ \mathrm{GHz}", fontsize=14, transform=ax.transAxes)

    # arrow(0.4, 0.2,  0.07, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # arrow(0.4, 0.2, -0.09, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # ax.text(0.33, 0.23, L"0.380\ \mathrm{GHz}", fontsize=14, transform=ax.transAxes)

    # arrow(0.65, 0.2,  0.03, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # arrow(0.65, 0.2, -0.15, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # ax.text(0.52, 0.23, L"0.375\ \mathrm{GHz}", fontsize=14, transform=ax.transAxes)

    # arrow(0.2, 0.2,  0.07, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # arrow(0.2, 0.2, -0.09, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # ax.text(0.13, 0.23, L"0.373\ \mathrm{GHz}", fontsize=14, transform=ax.transAxes)

    # arrow(0.83, 0.2,  0.03, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # arrow(0.83, 0.2, -0.14, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # ax.text(0.72, 0.23, L"0.70\ \mathrm{GHz}", fontsize=14, transform=ax.transAxes)

    xlabel(L"\Delta\nu\ (\mathrm{GHz})")
    ylabel(L"V\ \mathrm{(arb.u.)}")

    xlim(xlims)
    ylim(ylims)

    legend()
    tight_layout()
    # savefig(string(@__DIR__, "/bilder/multilorentzian.pdf"), bbox_inches="tight")
    plt.show()
end

println("Δν1 = $(popt[5] - popt[2]), Δν2 = $(popt[8] - popt[5]) Δν3 = $(popt[11] - popt[8]), Δν4 = $(popt[14] - popt[11])")

# load data
begin
    data1 = CSV.read(string(@__DIR__, "/data2/TEK00003.CSV"), DataFrame, skipto=17, header=["t", "CH1", "CH2"])
    data2 = CSV.read(string(@__DIR__, "/data2/TEK00004.CSV"), DataFrame, skipto=17, header=["t", "CH1", "CH2"])
end

data

newy = data1.CH2 .- data2.CH2

# find peaks
maxima, height = ss.find_peaks(newy, height=0.01, distance=50)
maxima = pyconvert(Array, maxima) .+ 1
ymaxima = data2.t[maxima]

sixlorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4] / ( 1 + ( ( x - p[5] )/ p[6] )^2 ) + p[7] / ( 1 + ( ( x - p[8] )/ p[9] )^2 ) + p[10] / ( 1 + ( ( x - p[11] )/ p[12] )^2 ) + p[13] / ( 1 + ( ( x - p[14] )/ p[15] )^2 ) + p[16] / ( 1 + ( ( x - p[17] )/ p[18] )^2 ) - 0.03

popt, ci = bootstrap(fivelorentzian, data1.t[175:950], newy[175:950], p0=[0.02, ymaxima[1], 0.0001, 0.01, ymaxima[2], 0.0001, 0.02, ymaxima[3], 0.0001, 0.04, ymaxima[4], 0.0001, 0.01, 0.00065, 0.0002, 0.], unc=true, redraw=false)
popt, ci = bootstrap(sixlorentzian, data1.t[175:950], newy[175:950], p0=[0.005, -0.0003, 0.0002, 0.02, ymaxima[1], 0.0003, 0.01, ymaxima[2], 0.0003, 0.02, ymaxima[3], 0.0003, 0.04, ymaxima[4], 0.0003, 0.01, 0.00065, 0.0002, 0.], unc=true, redraw=false)


newt_to_f(t) = @. nom.(FSR / dt * t /2 * u"GHz" / u"MHz" |> NoUnits)
newf_to_t(f) = @. @. 2* dt / FSR * f |> NoUnits

newt_to_f(-0.0005)
newf_to_t(-.100)

begin
ax = plt.subplots(2, 1, figsize=(7, 6), sharex=true)[1]
ax[0].scatter(data1.t, data1.CH2, s=5, label=L"\mathrm{saturated\ spectrum}")
ax[0].scatter(data2.t, data2.CH2, s=5, label=L"\mathrm{linear\ spectrum}")

ax[0].set_ylabel(L"V\ \mathrm{(arb.u.)}")
ax[0].legend()

ax[1].scatter(data1.t, newy, s=5, label=L"\mathrm{data}")

ax[1].plot(data1.t, sixlorentzian(data1.t, nom.(popt)), color="C1", label=L"\mathrm{fit}")

ax[1].text(0.075, 0.225, L"\nu_1", fontsize=14, transform=ax[1].transAxes)
ax[1].text(0.16, 0.55, L"\frac{\nu_1 + \nu_2}{2}", fontsize=14, transform=ax[1].transAxes)
ax[1].text(0.325, 0.425, L"\nu_2", fontsize=14, transform=ax[1].transAxes)
ax[1].text(0.375, 0.625, L"\frac{\nu_1 + \nu_3}{2}", fontsize=14, transform=ax[1].transAxes)
ax[1].text(0.5, 0.86, L"\frac{\nu_2 + \nu_3}{2}", fontsize=14, transform=ax[1].transAxes)
ax[1].text(0.75, 0.25, L"\nu_3", fontsize=14, transform=ax[1].transAxes)

# transform x-axis to frequency by setting xticks
ax[1].set_xticks(newf_to_t.([-0.1, -0.05, 0, .05, .1, .15, .2]), [L"0", L"50", L"100", L"150", L"200", L"250", L"300"])

ax[1].set_xlim(newf_to_t(-0.1), 0.001)
ax[1].set_ylim(-0.01, 0.055)

ax[1].set_xlabel(L"\Delta\nu\ (\mathrm{MHz})")
ax[1].set_ylabel(L"V\ \mathrm{(arb.u.)}")

ax[1].legend()
tight_layout()
savefig(string(@__DIR__, "/bilder/sixlorentzian.pdf"), bbox_inches="tight")
plt.show()
end
# end

begin
peak3 = measurement(newt_to_f(popt[5]), newt_to_f(popt[6]))*2
peak1 = measurement(newt_to_f(popt[2]), newt_to_f(popt[3]))*2
peak2 = measurement(newt_to_f(popt[8]), newt_to_f(popt[9]))*2
peak4 = measurement(newt_to_f(popt[11]), newt_to_f(popt[12]))*2
peak5 = measurement(newt_to_f(popt[14]), newt_to_f(popt[15]))*2
peak6 = measurement(newt_to_f(popt[17]), newt_to_f(popt[18]))*2

ν1 = peak1
ν2 = peak3
ν3 = peak6
ω1 = peak2
ω2 = peak4
ω3 = peak5
end

# δ1 = 72.21 
δ1 = 156.94
δ2 = 266.65

# check relations
2*ν3 - 2*ω3
δ2

# check relations
2*ν2 - 2*ω1
δ1

2*ν3 - 2*ω2
δ2 + δ1

C1, C2, D1, D2 = 4.5, -1.5, 0.403, -1.23

Δν1 = 194 * u"MHz"
Δν2 = (2*ν3 - 2*ω3) * u"MHz"

A = -(2(Δν2 - Δν1 )D1 + 2*D2*Δν1) / (C2*D1 - C1*D2)
B = -((Δν2-Δν1)C1 + C2*Δν1)/(C1*D2 - C2*D1)

nom(A)
