using Random

function data_balancing(data_xy; balancing::String)
    normal_data = data_xy[data_xy[:, end].==0.0, :]
    anomaly = data_xy[data_xy[:, end].==1.0, :]
    size_anomaly = size(anomaly)[1]
    size_normal = size(normal_data)[1]
    multiplier = div(size_normal, size_anomaly)
    leftover = mod(size_normal, size_anomaly)
    if balancing == "undersampling"
        data_xy = vcat(normal_data[1:size(anomaly)[1], :], anomaly)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "generative"
        new_anomaly = vcat(repeat(anomaly, outer=multiplier - 1), anomaly[1:leftover, :], anomaly)
        data_x = select(new_anomaly, Not([:target]))
        data_y = select(new_anomaly, [:target])
        new_anomaly = mapcols(x -> x + x * rand(collect(-0.05:0.01:0.05)), data_x)
        new_anomaly = hcat(data_x, data_y)
        data_xy = vcat(normal_data, new_anomaly)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "none"
        nothing
    end
    # data_x = Matrix(data_xy)[:, 1:end-1]
    # data_y = data_xy.target
    return data_xy
end

# A handy helper function to normalize our dataset.
    function standardize(x, mean_, std_)
        return (x .- mean_) ./ (std_ .+ 0.000001)
    end

    # A handy helper function to normalize our dataset.
    function scaling(x, max_, min_)
        return (x .- min_) ./ (max_ - min_)
    end