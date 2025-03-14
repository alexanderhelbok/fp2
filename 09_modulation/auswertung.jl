using CSV, Roots, LaTeXStrings, PythonCall, Bessels
using PhysicalConstants.CODATA2018: c_0

include(string(@__DIR__, "/../Source.jl"))
# include(string(@__DIR__, "/../SourceStatistics.jl"))
@py import matplotlib as mpl
@py import scipy.signal as ss

mpl.use("pgf")
mpl.use("TkAgg")

sinefit(x, p) = @. p[1] * sin(2*pi*p[2] * x + p[3])
fitarr = [2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7]

begin
A, f, phi = [[], []], [[], []], [[], []]
for (j, i) in enumerate(1:24)
if i < 10
    i = "0$i"
else
    i = "$i"
end
data = CSV.read(joinpath(@__DIR__, "data/A6/A6_CH$i.CSV"), DataFrame, header=["t", "CH"], skipto=2)
offset = mean(data.CH)
data.CH .-= offset

# fit sinefit
popt, ci = bootstrap(sinefit, data.t[1:end], data.CH[1:end], p0=[.2, fitarr[ceil(Int, j/2)], 1.], redraw=false, unc=true)
# println(popt[2])

if j in [18, 19, 20, 21]
plot(data.t, data.CH)
end
# plot(data.t, sinefit(data.t, nom.(popt)))
# # scatter(data.t[maxima], data.CH[maxima], c="C1")
# if j%2 == 0
#     plt.show()
# end
# plt.show() 

push!(A[j%2 + 1], popt[1])
push!(f[j%2 + 1], popt[2])
push!(phi[j%2 + 1], popt[3])
end
# plt.show()

A = abs.(A[1] ./ A[1][1])
phi = phi[2] - phi[1]
f = (f[1] + f[2])/2

f .*= 1e-3
phi2 = -(phi .+ 2*pi).%(2 * pi)
phi2[end] -= pi
phi2[end-2] -= pi
end

# fit bandpass filter transfer function to data
# Φ(x, p) = @. pi - 2*atan(x/p[1])
Φ(x, p) = @. - 4*(atan(1/p[2]*(2*x/p[1] + sqrt(1 + 4*(x/p[1])^2))) + atan(1/p[2]*(2*x/p[1] - sqrt(1 + 4*(x/p[1])^2))))
popt2, ci2 = bootstrap(Φ, nom.(f[1:end-2]), nom.(phi2[1:end-2]), xerr=err.(f[1:end-2]), yerr=err.(phi2[1:end-2]), p0=[1e3, 1.], unc=true, redraw=false)

H(x, p) = @. 1 / (1 + (x / p[1])^2)^(p[2]/2)
popt, ci = bootstrap(H, nom.(f), nom.(A), xerr=err.(f), yerr=err.(A), p0=[1e3, 1.], unc=true, redraw=false)
A = 20*log10.(A)
popty = 20*log10.(H(ci.x, nom.(popt)))

# calculate chisq of fit
chisq(nom.(A), H(nom.(f), nom.(popt)), sigma=err.(A))

begin
ax = plt.subplots(2, 1, sharex=true, figsize=(7, 5))[1]
ax[0].errorbar(nom.(f), [nom(p) for p in A], [err(p) for p in A], fmt=".k", capsize=3, label=L"\mathrm{Data}")
ax[1].errorbar(nom.(f), [nom(p) for p in phi2], [err(p) for p in phi2], fmt=".k", capsize=3, label=L"\mathrm{Data}")

ax[0].plot(ci.x, popty, c="C1", label=L"\mathrm{Fit}")
ax[1].plot(ci.x, Φ(ci.x, nom.(popt2)), c="C1", label=L"\mathrm{Fit}")

# add red box around excluded datapoints
rect = mpl.patches.FancyBboxPatch((0.875, 0.03), 0.075, 0.335, lw=2, ls="--", ec="crimson", fc="none", transform=ax[1].transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax[1].add_patch(rect)
# add legendentry for excluded datapoints by creating dummyplot
ax[1].plot(0, 0, c="crimson", label=L"\mathrm{Excluded\ Data}", lw=2, ls="--")

# add fitfunction to top plot
ax[0].text(0.5, 0.6, L"H(f) = \frac{1}{\left( 1 + \left(\frac{f}{f_c}\right)^2 \right)^{n/2}}", transform=ax[0].transAxes, fontsize=12)
ax[0].text(0.5, 0.35, L"f_c = 5.82(6)\ \mathrm{MHz}", transform=ax[0].transAxes, fontsize=12)
ax[0].text(0.5, 0.2, L"n = 4.29(7)", transform=ax[0].transAxes, fontsize=12)

rect = mpl.patches.FancyBboxPatch((0.5, 0.18), 0.225, 0.5, lw=1.5, ec="C1", fc="white", transform=ax[0].transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax[0].add_patch(rect)

# add fitfunction to top plot
ax[1].text(0.45, 0.35, L"f_c = 1.9(2)\ \mathrm{MHz}", transform=ax[1].transAxes, fontsize=12)
ax[1].text(0.45, 0.2, L"\alpha = 0.4(1.6)", transform=ax[1].transAxes, fontsize=12)

rect = mpl.patches.FancyBboxPatch((0.45, 0.18), 0.19, 0.25, lw=1.5, ec="C1", fc="white", transform=ax[1].transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
ax[1].add_patch(rect)

ax[0].set_ylabel(L"\mathrm{Gain}\ (\mathrm{dBm})")
ax[1].set_ylabel(L"\phi")

ax[1].set_yticks([0, -pi/2, -pi, -3pi/2], [L"0", L"\frac{-\pi}{2}", L"-\pi", L"\frac{-3\pi}{2}"])
ax[1].set_ylim(-5, 0.5)

plt.xscale("log")
plt.grid(true)
plt.xlabel(L"f\ (\mathrm{kHz})")

for axis in ax
    axis.grid(true, which="minor", linestyle="--")
    axis.grid(true, which="major", lw=1)
    axis.set_axisbelow(true)
end

ax[0].legend()
ax[1].legend()

plt.tight_layout()
plt.savefig(string(@__DIR__, "/bilder/bode.pdf"), bbox_inches="tight")
plt.show()
end
popt
# =================== Ex 1 ========================
begin
ax = plt.subplots(1, 2, figsize=(8.5, 2.5))[1]
data = CSV.read(joinpath(@__DIR__, "data/A1/A1_CH02.CSV"), DataFrame, header=["t", "CH"], skipto=2)
data = data[825:2000, :]
data.CH .-= minimum(data.CH)
data.CH = 2* data.CH ./ (maximum(data.CH)) .- 1
data.t .-= data.t[1]

maxima, height = ss.find_peaks(data.CH, height=0.01, distance=20)
height = pyconvert(Array, height["peak_heights"])
maxima = pyconvert(Array, maxima) .+ 1


envelope = ((maximum(height) - minimum(height)) * cos.(2*pi*50e3*data.t .+ pi) .+ (maximum(height) + minimum(height))) /2

ax[0].plot(data.t, data.CH)
ax[0].plot(data.t, envelope, ls="--", lw=3, c="gray")
ax[0].plot(data.t, -envelope, ls="--", lw=3, c="gray")

ax[0].set_xlabel(L"t\ (\mathrm{s})")
ax[0].set_ylabel(L"A\ \mathrm{(arb.U.)}")

ax[0].set_xlim(0, 5e-5)

data = CSV.read(joinpath(@__DIR__, "data/A1/A1_FFT02.CSV"), DataFrame, header=["t", "CH"], skipto=2)
data = data[530e3 .< data.t .< 720e3, :]
data.t ./= 1e3

ax[1].plot(data.t, data.CH)
ax[1].set_xlabel(L"f\ (\mathrm{kHz})")
ax[1].set_ylabel(L"A\ (\mathrm{dBm})")

ax[1].set_xticks([575, 625, 675])
ax[1].set_xlim(530, 720)
ax[1].set_yticks([-125, -100, -75, -50, -25])

plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/AM.pdf"), bbox_inches="tight")
plt.show()
end




threelorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4] / ( 1 + ( ( x - p[5] )/ p[6] )^2 ) + p[7] / ( 1 + ( ( x - p[8] )/ p[9] )^2 ) + p[10]

begin
leglabels = [L"\mu = 10\%", L"\mu = 50\%", L"\mu = 100\%"]
c = ["C0", "C1", "C4"]
ax = plt.subplots(figsize=(7, 3.5))[1]
for i in 1:3
data = CSV.read(joinpath(@__DIR__, "data/A1/A1_FFT0$i.CSV"), DataFrame, header=["t", "CH"], skipto=2)
data = data[5e5 .< data.t .< 7.5e5, :]
data.t ./= 1e3

# get three maxima of data
maxima, height = ss.find_peaks(data.CH, height=-60, distance=25)
height = pyconvert(Array, height["peak_heights"])
maxima = pyconvert(Array, maxima) .+ 1
    
# fit lornezian to data
popt, ci = bootstrap(threelorentzian, data.t, data.CH, p0=[height[1], data.t[maxima[1]], 1., height[2], data.t[maxima[2]], 1., height[3], data.t[maxima[3]], 1., mean(data.CH)], unc=true, redraw=false)

newx = LinRange(data.t[1], data.t[end], floor(Int, 1e5))

scatter(data.t, data.CH, c=c[i], s=10)
xlim = ax.get_xlim()

# create dummy fill_between for legend
ax.fill_between(0, [0, 0], color=c[i], label=leglabels[i], ec="black")

plot(newx, threelorentzian(newx, nom.(popt)), c=c[i])
ax.set_xlim(xlim)
end

# create dummyplot for legend
scatter(0, 0, c="gray", label=L"\mathrm{Data}")
plot(0, 0, c="gray", label=L"\mathrm{Fit}")

ax.set_xlim(500, 750)
ax.set_ylim(-110, -15)
ax.set_xticks([525, 575, 625, 675, 725])
plt.legend(loc="upper left", borderaxespad=0.5)

plt.xlabel(L"f\ (\mathrm{kHz})")
plt.ylabel(L"A\ (\mathrm{dBm})")

plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/AM_fft.pdf"), bbox_inches="tight")
plt.show()
end

# =================== Ex 2 =======================
begin
    ax = plt.subplots(1, 2, figsize=(8.5, 2.5))[1]
    data = CSV.read(joinpath(@__DIR__, "data/pater/A7_CH12.CSV"), DataFrame, header=["t", "CH"], skipto=2)
    data = data[1:15000, :]
    # normalize data
    data.CH .-= minimum(data.CH)
    data.CH = 2* data.CH ./ (maximum(data.CH)) .- 1
    data.t .-= data.t[1]
    
    # find minima of CH
    minima, height = ss.find_peaks(-data.CH, height=0.02, distance=500)
    minima = pyconvert(Array, minima) .+ 1

    # put 8th minima in center of data.t
    # data.t .-= data.t[minima[8]]    

    A = maximum(data.CH)

    f = 625e3

    carrier_y = @. A*cos(2*pi*data.t*f)

    # plot
    ax[0].plot(data.t, data.CH)
    ax[0].plot(data.t, carrier_y, ls="--", lw=3, c="gray")

    ax[0].set_xlabel(L"t\ (\mathrm{s})")
    ax[0].set_ylabel(L"A\ \mathrm{(arb.U.)}")

    ax[0].set_xlim(0, 1.5e-5)

    data = CSV.read(joinpath(@__DIR__, "data/alldata/A3_FF6.CSV"), DataFrame, header=["t", "CH"], skipto=2)
    data = data[125e3 .< data.t .< 1125e3, :]
    data.t ./= 1e3

    ax[1].plot(data.t, data.CH)
    ax[1].set_xlabel(L"f\ (\mathrm{kHz})")
    ax[1].set_ylabel(L"A\ (\mathrm{dBm})")

    ax[1].set_xlim(125, 1125)
    ax[1].set_yticks([-125, -100, -75, -50, -25])

    plt.tight_layout()
    # plt.savefig(string(@__DIR__, "/bilder/FM.pdf"), bbox_inches="tight")
    plt.show()
end

# =================== Ex 3 ========================

begin
peak_loc, peak_height = [], []
for i in 1:13
tempdf = CSV.read(joinpath(@__DIR__, "data/alldata/A3_FF$i.CSV"), DataFrame, header=["t", "CH"], skipto=2)

data = tempdf[10:sum(tempdf.t .< 2e6), :]

µ, σ = mean(tempdf.CH), std(tempdf.CH)

lorentz(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + µ

maxima, height = ss.find_peaks(data.CH, height=µ + 5*σ, distance=10)
# convert to julia array
maxima = pyconvert(Array, maxima) .+ 1
    
ci, start, stop = 0, 0, 0
temp1, temp2 = [], []
for peak in maxima
step = 5
start, stop = peak - step, peak + step
popt, ci = bootstrap(lorentz, data.t[start:stop], data.CH[start:stop], p0 = [70., data.t[peak], 1e4], unc=true, redraw=false)

push!(temp1, measurement(nom(popt[2]), nom(popt[3])))
push!(temp2, popt[1])
end
# combine opposind peak_heights due to symmetry

temparr = Vector{Measurement}(undef, floor(Int, length(temp2)/2)+1)

temparr[1] = temp2[floor(Int, length(temp2)/2 + 1)]

for i in 1:floor(Int, length(temp2)/2)
    temparr[end-i+1] = (temp2[i] + temp2[end-i+1])/2
end
# convert dbm to linear scale
temparr = @. 10^(temparr/10)
# normalize peak_height
temparr ./= sum(temparr)

push!(peak_loc, temp1)
push!(peak_height, temparr)
end
end
peak_loc
peak_height[1]

modulation_index = [25, 35, 250, 50, 75, 100, 5, 10, 15, 150, 200, 300, 60] ./ 50
theory = [[besselj(i-1, modulation_index[j]) for i in 1:length(peak_height[j])] for j in 1:length(peak_height)]


# plot first sideband amplitude against modulation index
begin
    ax = plt.subplots(4, 1, figsize=(10, 7), constrained_layout=true)[1]
    
    tempdf = CSV.read(joinpath(@__DIR__, "data/alldata/A3_FF2.CSV"), DataFrame, header=["t", "CH"], skipto=2)
        
    data = tempdf[10:sum(tempdf.t .< 2e6), :]
    data.t ./= 1e3
        
    µ, σ = mean(tempdf.CH), std(tempdf.CH)
        
    lorentz(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + µ
        
    maxima, height = ss.find_peaks(data.CH, height=µ + 5*σ, distance=10)
    # convert to julia array
    maxima = pyconvert(Array, maxima) .+ 1
            
    ci, start, stop = 0, 0, 0
    temp1, temp2 = [], []
    legendflag = false
    
    for peak in maxima
    step = 5
    start, stop = peak - step, peak + step
    popt, ci = bootstrap(lorentz, data.t[start:stop], data.CH[start:stop], p0 = [70., data.t[peak], 1e4], unc=true, redraw=false)
    
    if legendflag == false
        ax[0].plot(ci.x, lorentz(ci.x, nom.(popt)), c="C0", label=L"\mathrm{Lorentz fits}")
        legendflag = true
    else
        ax[0].plot(ci.x, lorentz(ci.x, nom.(popt)), c="C0")
    end
    
    end
    ax[0].scatter(data.t, data.CH, c="C1", s=15, label=L"\mathrm{data}")
    # scatter(data.t[start:stop], data.CH[start:stop], c="C0", s=15)
    ax[0].set_xlim(325, 925)
    
    # ax[0].legend()
    
    ax[0].set_xlabel(L"f\ (\mathrm{kHz})")
    ax[0].set_ylabel(L"A\ \mathrm{(dBm})")

xs = LinRange(0, 6.25, 100)

sidebands = [0, 2, 7]

for (i, band) in enumerate(sidebands)
    y = [besselj(band, x) for x in xs]
    ax[i].plot(xs, y, c=c[i], label=latexstring("J_$band(x)"))
    ax[i].set_ylabel(latexstring("J_$(band)(x)"))
end
for (j, index) in enumerate(peak_height)
    count = 1
    for (i, peak) in enumerate(index)
        if i - 1 == sidebands[1]
            if modulation_index[j] < 3
                ax[count].errorbar(modulation_index[j], nom.(peak), err.(peak), fmt=".", capsize=3, mfc=c[count], mec="k", ms=11, ecolor="k")
            end
        count += 1
        elseif i - 1 == sidebands[2]
            if modulation_index[j] < 6
                ax[count].errorbar(modulation_index[j], nom.(peak), err.(peak), fmt=".", capsize=3, mfc=c[count], mec="k", ms=11, ecolor="k")
            end
        count += 1  
        elseif i == sidebands[3]
            # if modulation_index[j] < 6
                ax[count].errorbar(modulation_index[j], nom.(peak), err.(peak), fmt=".", capsize=3, mfc=c[count], mec="k", ms=11, ecolor="k")
            # end
        end
    end
end

ax[0].set_yticks([-125, -100, -75, -50, -25])

# turn off xticks for ax1 and ax2
ax[1].set_xticklabels([])
ax[2].set_xticklabels([])
# set xlim
for i in 1:3
    ax[i].set_xlim(0, 6.25)
end

plt.xlabel(L"\mathrm{Modulation\ Index}\ \mu")

# plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/FM_FFT.pdf"), bbox_inches="tight")
plt.show()
end



# =================== Ex 4 ========================
labels = [L"f_m = 1\ \mathrm{kHz}", L"f_m = 100\ \mathrm{Hz}"]
begin
    ax = plt.subplots(figsize=(7, 3))[1]
for i in 1:2
data = CSV.read(joinpath(@__DIR__, "data/alldata/A4_FF0$(i+1).CSV"), DataFrame, header=["t", "CH"], skipto=2)
data.t ./= 1e3
    
plot(data.t, data.CH, c="C$(i-1)", label=labels[i])
end

plt.xlabel(L"f\ (\mathrm{kHz})")
plt.ylabel(L"A\ (\mathrm{dBm})")

plt.xlim(625-7.5, 625+7.5)
plt.ylim(-125, -10)
plt.legend(borderaxespad=0.5)
plt.tight_layout()
plt.savefig(string(@__DIR__, "/bilder/AM_fft2.pdf"), bbox_inches="tight")
plt.show()
end



begin
ax = plt.subplots(2, 1, figsize=(7, 4.5), sharex=true)[1]
starts, stops = [], []
for i in 1:2
data = CSV.read(joinpath(@__DIR__, "data/alldata/A7_CH0$i.CSV"), DataFrame, header=["t", "CH"], skipto=2)
data.t .-= data.t[1]
δt = data.t[2] - data.t[1]
# data.CH .-= minimum(data.CH)

start1, stop1 = 1, 5000
µ1, σ1 = mean(data.CH[start1:stop1]), std(data.CH[start1:stop1])
start2, stop2 = 9000, 16000
µ2, σ2 = mean(data.CH[start2:stop2]), std(data.CH[start2:stop2])

sigstart = findfirst(data.CH .< µ1 - 4*σ1)
sigstop = findfirst(data.CH .< µ2 + 4*σ2)

push!(starts, measurement(data.t[sigstart], δt))
push!(stops, measurement(data.t[sigstop], δt))

Δt = stops[i] - starts[i]
println(1/(2*Δt))


ax[i-1].scatter(data.t, data.CH, s=3, label=L"\mathrm{Data}", c="C0")

ax[i-1].scatter(data.t[sigstart], data.CH[sigstart], c="C1", s=15)
ax[i-1].scatter(data.t[sigstop], data.CH[sigstop], c="C1", s=15)

ax[i-1].axvline(data.t[sigstart], color="C1", ls="--")
ax[i-1].axvline(data.t[sigstop], color="C1", ls="--")

ax[i-1].hlines(µ1, data.t[start1], data.t[sigstart], color="crimson", ls="--", label=L"\mathrm{mean}")
ax[i-1].hlines(µ2, data.t[sigstop], data.t[stop2], color="crimson", ls="--")

# fill between sigma band
ax[i-1].fill_between(data.t[start1:sigstart], µ1 - 4*σ1, µ1 + 4*σ1, color="crimson", alpha=0.3, label=L"4\sigma-\mathrm{Band}")
ax[i-1].fill_between(data.t[sigstop:stop2], µ2 - 4*σ2, µ2 + 4*σ2, color="crimson", alpha=0.3)

ax[i-1].set_xlim(0.5e-5, 1e-5)
ax[i-1].legend()
end
ax[0].text(0.275, 0.5, L"\Delta t = 4.1(14) \times 10^{-8}\ \mathrm{s}", transform=ax[0].transAxes, fontsize=12, c="C1")
ax[1].text(0.375, 0.5, L"\Delta t = 2.870(14) \times 10^{-7}\ \mathrm{s}", transform=ax[1].transAxes, fontsize=12, c="C1")

ax[0].set_ylabel(L"\mathrm{input\ signal\ (V)}")
ax[1].set_ylabel(L"\mathrm{output\ signal\ (V)}")
plt.xlabel(L"t\ (\mathrm{s})")

plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/delay.pdf"), bbox_inches="tight")
plt.show()
end

starts[2] - starts[1]
stops[2] - stops[1]

# calucluate phase shift from time delay
f = 1/(2*(stops[2] - starts[2]))
δt = starts[2] - starts[1]
δφ = 2*f*δt