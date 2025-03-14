using CSV, Roots, LaTeXStrings, PythonCall
using SavitzkyGolay

include(string(@__DIR__, "/../Source.jl"))
# include(string(@__DIR__, "/../SourceStatistics.jl"))
@py import matplotlib as mpl
@py import scipy.signal as ss

mpl.use("pgf")
mpl.use("TkAgg")


for i in 1:4
    # open file and replace "," with "."
    open(joinpath(@__DIR__, "Scherung_03/Aramid_0$i.csv"), "r") do file
        data = read(file, String)
        data = replace(data, "," => ".")
        data = replace(data, ";" => ",")
    end

    # load data
    begin
        data = CSV.read(joinpath(@__DIR__, "Scherung_03/Aramid_0$i.csv"), DataFrame, skipto=4, header=["t", "x", "F", "def", ""])
        # dtop last column
        data = data[:, 1:end-1]
        # only keep every 300th data point
        data = data[1:300:end, :]
    end

    # write data to csv
    CSV.write(joinpath(@__DIR__, "reduced/Aramid_0$i.csv"), data)
end
# smooth data using SavitzkyGolay
y = savitzky_golay(data.F, 11, 2)

# plot data
begin
    ax = plt.subplots(figsize=(7, 4))[1]
    ax.scatter(data.x, data.F, s=10)
    # ax.plot(data.x, y.y, c="C1", lw=2)
    ax.set_xlabel(L"t\ (\mathrm{s})")
    ax.set_ylabel(L"x\ (\mathrm{mm})")
    plt.tight_layout()
    # plt.savefig(joinpath(@__DIR__, "bilder/plot0.pdf"), bbox_inches="tight")
    plt.show()
end