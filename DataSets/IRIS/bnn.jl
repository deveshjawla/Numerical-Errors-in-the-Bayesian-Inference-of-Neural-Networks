using Distributed
using Turing
# Add four processes to use for sampling.
addprocs(5; exeflags=`--project`)

@everywhere begin
    PATH = @__DIR__
    cd(PATH)

    include("../../BNNUtils.jl")
    include("../../Calibration.jl")
    include("../../DataUtils.jl")

    ###
    ### Data
    ###
    using DataFrames
    using CSV

    iris = CSV.read("Iris_cleaned.csv", DataFrame, header=1)

    # Function to split samples.
    function split_data(df; at=0.70)
        r = size(df, 1)
        index = Int(round(r * at))
        train = df[1:index, :]
        test = df[(index+1):end, :]
        return train, test
    end

    target = "Species"
    using Random
    iris = iris[shuffle(axes(iris, 1)), :]
    train, test = split_data(iris, at=0.8)



    # A handy helper function to normalize our dataset.
    function standardize(x, mean_, std_)
        return (x .- mean_) ./ (std_ .+ 0.000001)
    end

    # A handy helper function to normalize our dataset.
    function scaling(x, max_, min_)
        return (x .- min_) ./ (max_ - min_)
    end

    train_x = Matrix(train[:, 1:4])
    # train_max = maximum(train_x, dims=1)
    # train_mini = minimum(train_x, dims=1)
    # train_x = scaling(train_x, train_max, train_mini)
    train_mean = mean(train_x, dims=1)
    train_std = std(train_x, dims=1)
    train_x = standardize(train_x, train_mean, train_std)
    train_y = train[:, end]

    test_x = Matrix(test[:, 1:4])
    # test_x = scaling(test_x, train_max, train_mini)
    test_x = standardize(test_x, train_mean, train_std)
    test_y = test[:, end]

    train_x = Array(train_x')
    test_x = Array(test_x')

    name = "Initializing_10101010_3_with_0.2"
    mkpath("./experiments/$(name)")


    ###
    ### Dense Network specifications
    ###

    input_size = size(train_x)[1]
    l1, l2, l3, l4, l5 = 10,10,10,10,3
    nl1 = input_size * l1 + l1
    nl2 = l1 * l2 + l2
    nl3 = l2 * l3 + l3
    nl4 = l3 * l4 + l4
    ol5 = l4 * l5 + l5

    total_num_params = nl1 + nl2 + nl3 + nl4 + ol5

    using Flux


    function feedforward(θ::AbstractVector)
        W0 = reshape(θ[1:40], 10, 4)
        b0 = θ[41:50]
        W1 = reshape(θ[51:150], 10, 10)
        b1 = θ[151:160]
        W2 = reshape(θ[161:260], 10, 10)
        b2 = θ[261:270]
        W3 = reshape(θ[271:370], 10, 10)
        b3 = θ[371:380]
        W4 = reshape(θ[381:410], 3, 10)
        b4 = θ[411:413]
        model = Chain(
            Dense(W0, b0, relu),
            Dense(W1, b1, relu),
            Dense(W2, b2, relu),
            Dense(W3, b3, relu),
            Dense(W4, b4),
            softmax
        )
        return model
    end

    # function feedforward(θ::AbstractVector)
    #     W0 = reshape(θ[1:20], 5, 4)
    #     b0 = reshape(θ[21:25], 5)
    #     W1 = reshape(θ[26:40], 3, 5)
    #     b1 = reshape(θ[41:43], 3)
    #     model = Chain(
    #         Dense(W0, b0, relu),
    #         Dense(W1, b1, relu),
    #         softmax
    #     )
    #     return model
    # end


    ###
    ### Bayesian Network specifications
    ###
    using Turing

    # setprogress!(false)
    # using Zygote
    # Turing.setadbackend(:zygote)
    using ReverseDiff
    Turing.setadbackend(:reversediff)

    sigma = 0.2

    #Here we define the layer by layer initialisation
    # sigma = vcat(sqrt(2 / (input_size + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), sqrt(2 / (l4 + l5)) * ones(ol5))
end

# Define a model on all processes.
@everywhere @model bayesnn(x, y) = begin
    θ ~ MvNormal(zeros(total_num_params), ones(total_num_params) .* sigma)
    nn = feedforward(θ)

    ŷ = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(ŷ[:, i])
    end
end

@everywhere model = bayesnn(train_x, train_y)
nsteps = 1000
chain_timed = @timed sample(model, NUTS(), MCMCDistributed(), nsteps, 5)
chain = chain_timed.value
elapsed = chain_timed.time
θ = MCMCChains.group(chain, :θ).value
using DelimitedFiles
writedlm("./experiments/$(name)/elapsed.txt", elapsed)
# using EvalMetrics
function accuracy(true_labels, predictions)
    accuracy = mean(true_labels .== predictions)
end
for i in 1:5
    params_set = collect.(eachrow(θ[:, :, i]))
    param_matrix = mapreduce(permutedims, vcat, params_set)

    independent_param_matrix = Array{Float64}(undef, Int(nsteps / 10), total_num_params)
    for i in 1:lastindex(param_matrix, 1)
        if i % 10 == 0
            independent_param_matrix[Int((i) / 10), :] = param_matrix[i, :]
        end
    end
    writedlm("./experiments/$(name)/param_matrix_$(i).csv", independent_param_matrix, ',')

    predictions, pred_probabilities = predicitons_analyzer_multiclass(test_x, test_y, independent_param_matrix)

    writedlm("./experiments/$(name)/predictions_$(i).csv", hcat(predictions, pred_probabilities), ',')

    acc = accuracy(test_y, predictions)
    ch = chain[:, :, i]
    summaries, quantiles = describe(ch)
    large_rhat = count(>=(1.01), sort(summaries[:, :rhat]))
    small_rhat = count(<=(0.99), sort(summaries[:, :rhat]))
    avg_acceptance_rate = mean(ch[:acceptance_rate])
    total_numerical_error = sum(ch[:numerical_error])
    avg_ess = mean(summaries[:, :ess])
    # println(describe(summaries[:, :mean]))
    writedlm("./experiments/$(name)/statistics_chain_$(i).csv", [["oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess", "Accuracy"] [large_rhat + small_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, acc]], ',')


end

# sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
# _, i = findmax(chain[:lp])
# i = i.I[1]
# elapsed = chain_timed.time
# θ = MCMCChains.group(chain, :θ).value
# θ[i, :]

PATH = @__DIR__
cd(PATH)
using DelimitedFiles
begin
    name = "Initializing_10101010_3_with_0.2"
	data = Array{Any}(undef, 5, 5)
    for i = 1:5
        m = readdlm("./experiments/$(name)/statistics_chain_$(i).csv", ',')
        data[:, i] = m[:, 2]
    end
    d = mean(data, dims=2)
    writedlm("./experiments/$(name)/statistics_chain_mean.txt", d)
end