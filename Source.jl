using DataFrames, Random, LsqFit, Statistics, Measurements, PythonCall, Distributions, Unitful
import PythonPlot.plot, PythonPlot.scatter
import Measurements.Measurement
# pygui(true)

include("SourceStatistics.jl")

# @py import pip  
# @py pip.main(["install", "mpl_axes_aligner"])
# @py pip.main(["install", "scipy"])
# @py pip.main(["install", "matplotlib"])
# @py pip.main(["install", "uncertainties"])

@py import scipy.signal as ss
@py import scipy.optimize as sciop
@py import matplotlib.pyplot as plt
@py import matplotlib as mpl
@py import mpl_axes_aligner as aligner
plt.style.use("Source.mplstyle")

plt.rc("text", usetex=true)  # enable use of LaTeX in matplotlib
plt.rc("font", family="sans-serif", serif="Times New Roman", size=14)  # font settings
plt.rc("text.latex", preamble="\\usepackage{mtpro2} \\usepackage{siunitx} \\usepackage{amsmath}")
plt.rcParams["pgf.texsystem"] = "xelatex"

# missing type combination is x = Vector{Any}, y = Vector{<:Number}
function plot(xdata::Union{LinRange{<:Number}, Vector{<:Number}, Vector{<:Measurement}, Vector{<:Quantity}}, ydata::Union{Vector{<:Measurement}, Vector{<:Quantity}}; kwargs...)
    plot(nom.(xdata), nom.(ydata); kwargs...)
end

# missing type combination is x = Vector{Any}, y = Vector{<:Number}
function scatter(xdata::Union{Vector{<:Number}, Vector{<:Measurement}, Vector{<:Quantity}}, ydata::Union{Vector{<:Measurement}, Vector{<:Quantity}}; kwargs...)
    scatter(nom.(xdata), nom.(ydata); kwargs...)
end

function myerrorbar(xdata, ydata; kwargs...) 
    errorbar(nom.(xdata), nom.(ydata), xerr=err.(xdata), yerr=err.(ydata); kwargs...)
end

function compile(file::String)
    @eval using Glob
    
    function rdir(dir::AbstractString, pat::AbstractString)
        pat = Glob.FilenameMatch(pat)
        result = String[]
        for (root, dirs, files) in walkdir(dir)
            append!(result, filter!(f -> occursin(pat, f), joinpath.(root, files)))
        end
        return result
    end

    fulldir = rdir(pwd(), "*$file.pgf")
    if length(fulldir) == 0
        throw(ArgumentError("No file found"))
    elseif length(fulldir) > 1
        throw(ArgumentError("Multiple files found"))
    else 
        fulldir = fulldir[1]
    end

    dir = splitdir(fulldir)[1]
    name = splitdir(fulldir)[end]

    # compile pgf to pdf
    run(`xelatex -output-directory=$dir "\newcommand{\file}{$name}\input{./imgen}"`)

    # Cleanup
    rm("$dir/imgen.aux")
    rm("$dir/imgen.log")
    mv("$dir/imgen.pdf", "$dir/$file.pdf", force=true)
end

# kwargs = (xminorticksvisible = true,
#     yminorticksvisible = true,
#     spinewidth = 2,
#     xminorticks = IntervalsBetween(4),
#     yminorticks = IntervalsBetween(4),
#     xtickwidth = 2,
#     ytickwidth = 2,
#     xticksize = -14,
#     yticksize = -14,
#     xminorticksize = -7,
#     yminorticksize = -7,
#     xticksmirrored = true,
#     yticksmirrored = true,
#     xgridvisible = false,
#     ygridvisible = false,
#     xgridwidth = 2,
#     ygridwidth = 2,
#     xticklabelsize = 20,
#     yticklabelsize = 20,
#     xticklabelfont = "Times New Roman",
#     yticklabelfont = "Times New Roman",
#     xlabelfont = "Times New Roman",
#     xlabelsize = 24,
#     ylabelfont = "Times New Roman",
#     ylabelsize = 24,
#     xlabelpadding = 10,
#     ylabelpadding = 10)


function chisq(obs, exp; sigma=nothing, dof=nothing, pcount=nothing)
    """
    Calculate chi-squared value for a given set of observed and expected values.
    Parameters:
    obs: observed values
    exp: expected values
    sigma: optional error on observed values (default nothing)
    dof: degrees of freedom (default 0)
    pcount: number of parameters (default 0)
    Returns:
    chi-squared value
    """
    if dof === nothing && pcount === nothing
        dof = 1
    elseif dof === nothing
        dof = length(obs) - pcount
    end

    if sigma === nothing
        return sum((obs - exp).^2)/dof
    else
        # replace 0 values in sigma with 1
        sigma[sigma .== 0] .= 1
        return sum(((obs - exp)./sigma).^2)/dof
    end
end


function de(fit, xdata, ydata, bounds; mut=0.8, crossp=0.7, popsize=20, its=1000, fobj=chisq, seed=nothing, sigma=nothing)
    """
    Differential evolution algorithm to fit a function fobj(xdata, p...) with length(p) parameters.
    Parameters:
    fit: LsqFit object
    xdata: x data to be fitted
    ydata: y data to be fitted
    bounds: bounds for each parameter
    mut: mutation factor (default 0.8)
    crossp: crossover probability (default 0.7)
    popsize: population size (default 20)
    its: number of iterations (default 1000)
    fobj: objective function (default chisq)
    seed: random seed (default None)
    sigma: optional error on observed values (default None)
    Returns:
    optimal parameters, 1 sigma error of parameters
    """
    if seed !== nothing
        Random.seed!(seed)
    end

    dimensions = length(bounds)
    # create population with random parameters (between 0 and 1)
    pop = rand(popsize, dimensions)
    # scale parameters to the given bounds
    bounds = permutedims(hcat(bounds...))
    min_b, max_b = bounds[:, 1], bounds[:, 2]
    diff = diff = abs.(max_b - min_b)
    pop_denorm = min_b' .+ pop .* diff'


    fitness = [fobj(ydata, fit(xdata, p), sigma=sigma) for p in eachrow(pop_denorm)]
    best_idx = argmin(fitness) 
    best = pop_denorm[best_idx, :]

    for i in 1:its
        for j in 1:popsize
            idxs = filter(x -> x != j, 1:dimensions)
            mu = @view pop_denorm[idxs[sample(1:dimensions-1, 3, replace=false)], :]
            mutant = mu[1, :] .+ mut .* (mu[2, :] .- mu[3, :])
            cross_points = rand(dimensions) .< crossp
            
            trial = [if cross_points[k] 
                        mutant[k] 
                    else 
                        pop_denorm[j, k] 
                    end for k in 1:dimensions]

            trial = clamp.(trial, bounds[:, 1], bounds[:, 2])
            
            f = fobj(ydata, fit(xdata, trial), sigma=sigma)
            if f < fitness[j]
                fitness[j] = f
                pop[j, :] = trial
                if f < fitness[best_idx]
                    best_idx = j
                    best = trial
                end
            end
        end
        yield(best, fitness[best_idx])
    end
end

function bootstrap(fobj, xdata, ydata; xerr=zeros(length(xdata)), yerr=zeros(length(ydata)), 
    p=0.95, its=1000, samples=nothing, p0=nothing, smoothing=false, unc=false, xlim=0.1, xlimconst=false, redraw=false)
    """
    Bootstrap fit including confidence bands to a function fobj(xdata, p...) with length(p) parameters.
    Parameters:
    fobj: function to be fitted of the form fobj(xdata, p...)
    xdata: x data to be fitted
    ydata: y data to be fitted
    xerr: optional x error (default zeros(length(xdata)))
    yerr: optional y error (default zeros(length(ydata)))
    p: confidence interval (default 0.95)
    its: number of iterations (default 1000)
    p0: initial parameters; if none given, using 1 as starting value (default nothing)
    smoothing: whether to smooth the confidence band using Savitzky-Golay (default false)
    Returns:
    optimal parameters, 1 sigma error of parameters, x and y values for confidence band
    """
    if length(xdata) != length(ydata)
        throw(ArgumentError("x and y must be of the same size"))
    end
    if p < 0
        throw(ArgumentError("p must be positive"))
    elseif p > 1
        warn("p > 1, assuming p is a percentage")
        p = p / 100
    end
    if p0 === nothing
        throw(ArgumentError("p0 must be given (either as a vector or as a number)"))
    elseif typeof(p0) == Int
        p0 = ones(p0)
    end
    if samples === nothing
        samples = length(xdata)
    elseif typeof(samples) <: Number
        samples = round(Int, samples)
    end

    # if no errors are given and no redraw is requested, only one iteration is necessary
    if xerr == zeros(length(xdata)) && yerr == zeros(length(ydata)) && redraw == false
        its = 1
    end

    # initialize array for parameters and interpolated values for each iteration
    arr2 = Matrix{Float64}(undef, 1000, its)
    var = zeros(length(p0))
    sum = zeros(length(p0))
    sigma = 0
    # initialize DataFrame for confidence band
    if typeof(xlim) == Float64
        if 0 <= xlim < 2 && xlimconst == false
            span = maximum(xdata) - minimum(xdata)
            ci = DataFrame(x=range(minimum(xdata) - span*xlim, maximum(xdata) + span*xlim, length=1000), c0=Vector(undef, 1000), c1=Vector(undef, 1000), mean=Vector(undef, 1000))
        elseif xlimconst == true
            ci = DataFrame(x=range(minimum(xdata) - xlim, maximum(xdata) + xlim, length=1000), c0=Vector(undef, 1000), c1=Vector(undef, 1000), mean=Vector(undef, 1000))
        end
    else
        println(xlim, typeof(xlim))
        ci = DataFrame(x=range(xlim[1], xlim[2], length=1000), c0=Vector(undef, 1000), c1=Vector(undef, 1000), mean=Vector(undef, 1000))
    end
    for i in 1:its
        if redraw 
            ind = rand(1:length(xdata), samples)
        else
            ind = [1:length(xdata);]
        end
        newx, newy = rand.(Normal.(xdata[ind], xerr[ind])), rand.(Normal.(ydata[ind], yerr[ind]))
        fit = curve_fit(fobj, newx, newy, p0)
        popt = fit.param
        try
            sigma = stderror(fit)
        catch
            sigma = 0
        end
        arr2[:, i] = fobj(ci.x, popt)

        sum += popt
        var += popt.^2
        # scatter!(newx, newy, alpha=0.1, c=:gray, label=nothing)
    end

    pmean = sum/its

    # calculate the error of the parameters, if no bootsrapping is done, the error is the fit error
    if xerr == zeros(length(xdata)) && yerr == zeros(length(ydata)) && redraw == false
        perr = sigma
    else
        perr = sqrt.(abs.(its/(its - 1)*(var/its - pmean.^2)))
    end

    ci.c0 = quantile.(eachrow(arr2), 0.5 * (1 - p))
    ci.c1 = quantile.(eachrow(arr2), 1 - 0.5 * (1 - p))
    ci.mean = mean(arr2, dims=2)[:, 1]

    popt = curve_fit(fobj, ci.x, ci.mean, pmean).param
    if smoothing
        # smooth the confidence band
        ci.c0 = ss.savgol_filter(ci.c0, 75, 1)
        ci.c1 = ss.savgol_filter(ci.c1, 75, 1)
    end
    if unc
        popt = measurement.(popt, perr)
        return popt, ci
    else
        return popt, perr, ci
    end
end


function Base.show(io::IO, m::Measurement)
    # set default format for the uncertainty (when printing)
    str = errprint(m)

    # Print the value followed by the uncertainty in bracket notation
    print(io, str)
end

function errprint(number::Union{Measurement, Quantity}, sigfigs::Int=0)
    @py import uncertainties
    # check wether the number is a measurement or a quantity
    u = ""
    if typeof(number) <: Quantity
        u *= " $(unit(number))"
        number = ustrip(number)
    end

    if sigfigs < 0
        error("sigfigs must be a positive integer")
    end


    val, unc = Measurements.value(number), Measurements.uncertainty(number)
    pynum = uncertainties.ufloat(val, unc)

    if sigfigs == 0
        pyunc = pyconvert(String, @py "%.2e" % unc)

        if parse(Float64, pyunc[1:4]) < 1.95
            # show 2 significant digits if leading digit is 1
            str = pyconvert(String, @py "{:.2uS}".format(pynum))
        else
            str = pyconvert(String, @py "{:.1uS}".format(pynum))
        end
    else
        formatter = "{:.$(sigfigs)uS}"
        str = pyconvert(String, @py formatter.format(pynum))
    end
    return str * u
end

# function newleg(elems::Function, labels, x, y; fig = figure[1, 1], xgap = 0.05, xmarkergap = 0.04, ygap = 0.05, labelsize = 22)
# 	# create a temporary axis (necessary for nonlinear axes) to get the relative projection
# 	tempax = Axis(fig, backgroundcolor = :transparent)
# 	hidedecorations!(tempax)
# 	hidespines!(tempax)
# 	relative_projection = Makie.camrelative(tempax.scene)
# 	# draw legend markers
# 	elems(relative_projection, x, y, ygap, xmarkergap)
# 	# draw legend labels
# 	for (i, label) in enumerate(labels)
# 		text!(relative_projection, 
# 			"$label", 
# 			position=Point2f(x + xgap, y - (i-1)*ygap), 
# 			align = (:left, 0.4),
# 			fontsize = labelsize,
# 			font = "Times New Roman")
# 	end
# end

# function mylegend(figure, elems, labels, x, y; fig = figure[1, 1], rgs...)
# 	tempax = Axis(fig)
# 	hidedecorations!(tempax)
# 	hidespines!(tempax)
# 	leg_projetion = campixel(tempax.scene)
# 	@lift translate!(leg_projetion, Vec2f($(figure.scene.camera.resolution)[1]*x, $(figure.scene.camera.resolution)[2]*y))
# 	Legend(leg_projetion, elems, labels; rgs...)
# end

# legargs = (labelfont = "Times New Roman", 
#     labelsize = 20, 
#     margin = ones(4).*18,
#     patchlabelgap = 10,
#     backgroundcolor = :transparent)