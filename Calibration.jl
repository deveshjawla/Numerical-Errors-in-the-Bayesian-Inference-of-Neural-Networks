# For the binning we assume here that the range of values is 0.0-1.0, that each bin closes right, that each bin would have a different size

function conf_bin_indices(n, conf, test, predictions)
    bins = Dict{Int,Vector}()
    mean_conf = Dict{Int,Float32}()
    bin_acc = Dict{Int,Float32}()
    calibration_gaps = Dict{Int,Float32}()
    for i in 1:n
        lower = (i - 1) / n
        upper = i / n
        # println(lower, upper)
        bin = findall(x -> x > lower && x <= upper, conf)
        bins[i] = bin
        if length(predictions[bin]) > 1
            mean_conf_ = mean(conf[bin])
            mean_acc_ = count(==(1), test[bin]) / length(test[bin])
        else
            mean_conf_ = NaN
            mean_acc_ = NaN
        end
        println(length(predictions[bin]), ' ', mean_acc_)
        mean_conf[i] = mean_conf_
        bin_acc[i] = mean_acc_
        calibration_gaps[i] = abs(mean_acc_ - mean_conf_)
    end
    return bins, mean_conf, bin_acc, calibration_gaps
end

#input is the number of bins, confidence scores of the predictions, true labels
function conf_bin_indices(n, conf, test)
    bins = Dict{Int,Vector}()
    mean_conf = Dict{Int,Float32}()
    bin_acc = Dict{Int,Float32}()
    calibration_gaps = Dict{Int,Float32}()
    for i in 1:n
        lower = (i - 1) / n
        upper = i / n
        # println(lower, upper)
        bin = findall(x -> x > lower && x <= upper, conf)
        bins[i] = bin
        if lastindex(test[bin]) > 1
            mean_conf_ = mean(conf[bin])
            mean_acc_ = count(==(1), test[bin]) / length(test[bin])
        else
            mean_conf_ = NaN
            mean_acc_ = NaN
        end
        println(lastindex(test[bin]), ' ', mean_acc_)
        mean_conf[i] = mean_conf_
        bin_acc[i] = mean_acc_
        calibration_gaps[i] = abs(mean_acc_ - mean_conf_)
    end
    return bins, mean_conf, bin_acc, calibration_gaps
end


using Distributions
using Optim

function ece_mce(bins, calibration_gaps, total_samples)
    n_bins = length(bins)
    ece_ = []
    for i in 1:n_bins
        append!(ece_, length(bins[i]) * calibration_gaps[i])
    end
    ece = sum(filter(!isnan, ece_)) / total_samples
    mce = maximum(filter(!isnan, collect(values(calibration_gaps))))
    return ece, mce
end

# Logistic function for a scalar input:
function platt(conf::Float64)
    1.0 / (1.0 + exp(-conf))
end

function platt(conf)
    1.0 ./ (1.0 .+ exp.(-conf))
end

# pred_conf and labels are on the dataset which we use for calibration
function _loss(a, b, pred_conf, labels)
    return -sum(labels .* log.(platt.(pred_conf .* a .+ b)) + (1.0 .- labels) .* log.(1.0 .- platt.(pred_conf .* a .+ b)))
end