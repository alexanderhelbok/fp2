using Unitful, Measurements
using Statistics


function nom(x)
    return Measurements.value(ustrip.(x))
end

function err(x)
    return Measurements.uncertainty(ustrip.(x))
end

error_weight(x) = 1/(err(x))^2

function weighted_mean(x)
    weights = error_weight.(x)
    nominal = sum(nom.(x).*weights) / sum(weights)
    error = 1/√(sum(weights))
    return measurement(nominal, error)
end

normal_mean(x) = measurement(mean(x), std(x))

