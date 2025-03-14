using FourierAnalysis, Plots

# Set sampling rate (sr) and FFT window length (wl):
sr, wl = 128, 128

# Generate a sinusoidal wave at 10Hz with peak amplitude 0.5 and add some white noise:
v = sinusoidal(0.5, 10, sr, wl*16) + randn(wl*16)

# Get the power spectrum with a rectangular tapering window:
S = spectra(v, sr, wl; tapering=rectangular)

# Plot the power spectrum:
plot(S; fmax=24)

# The same syntax applies in the case of multivariate data (e.g., 4 time-series):
V = randn(wl*16, 4)
S = spectra(V, sr, wl; tapering=hamming)
plot(S)

# Get the analytic amplitude in the time-Frequency domain:
A = TFamplitude(v, sr, wl; fmax=24)

# plot the analytic amplitude:
heatmap(A.y)
