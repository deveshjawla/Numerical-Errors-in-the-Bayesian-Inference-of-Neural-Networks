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

    train_xy = CSV.read("train.csv", DataFrame, header=1)
    shap_importances = CSV.read("./shap_importances.csv", DataFrame, header=1)
    train_xy = select(train_xy, vcat(shap_importances.feature_name[1:6], "stroke"))
    balanced_data = data_balancing(train_xy, balancing="undersampling")

    train_x = Matrix(train_xy[:, 1:end-1])
    # train_max = maximum(train_x, dims=1)
    # train_mini = minimum(train_x, dims=1)
    # train_x = scaling(train_x, train_max, train_mini)
    train_mean = mean(train_x, dims=1)
    train_std = std(train_x, dims=1)
    train_x = standardize(train_x, train_mean, train_std)
    train_y = train_xy[:, end]

    test_xy = CSV.read("./test.csv", DataFrame, header=1)
    test_xy = select(test_xy, vcat(shap_importances.feature_name[1:6], "stroke"))
    test_xy = data_balancing(test_xy, balancing="none")
    test_x = Matrix(test_xy[:, 1:end-1])
    # test_x = scaling(test_x, train_max, train_mini)
    test_x = standardize(test_x, train_mean, train_std)
    test_y = test_xy[:, end]
	n_test = lastindex(test_y)

    train_x = Array(train_x')
    test_x = Array(test_x')
    train_y[train_y.==0] .= 2
    test_y[test_y.==0] .= 2
    name = "Initializing_5_5_2_with_3.0"
    mkpath("./experiments/$(name)")

    ###
    ### Dense Network specifications
    ###

    input_size = size(train_x)[1]
    l1, l2, l3, l4, l5 = 5, 5, 2, 0, 0
    nl1 = input_size * l1 + l1
    nl2 = l1 * l2 + l2
    nl3 = l2 * l3 + l3
    nl4 = l3 * l4 + l4
    ol5 = l4 * l5 + l5

    total_num_params = nl1 + nl2 + nl3 + nl4 + ol5

    using Flux


    # function feedforward(θ::AbstractVector)
    #     W0 = reshape(θ[1:60], 10, 6)
    #     b0 = θ[61:70]
    #     W1 = reshape(θ[71:170], 10, 10)
    #     b1 = θ[171:180]
    #     W2 = reshape(θ[181:280], 10, 10)
    #     b2 = θ[281:290]
    #     W3 = reshape(θ[291:390], 10, 10)
    #     b3 = θ[391:400]
    #     W4 = reshape(θ[401:420], 2, 10)
    #     b4 = θ[421:422]
    #     model = Chain(
    #         Dense(W0, b0, relu),
    #         Dense(W1, b1, relu),
    #         Dense(W2, b2, relu),
    #         Dense(W3, b3, relu),
    #         Dense(W4, b4),
    #         softmax
    #     )
    #     return model
    # end

    function feedforward(θ::AbstractVector)
        W0 = reshape(θ[1:30], 5, 6)
        b0 = reshape(θ[31:35], 5)
        W1 = reshape(θ[36:60], 5, 5)
        b1 = reshape(θ[61:65], 5)
        W2 = reshape(θ[66:75], 2, 5)
        b2 = reshape(θ[76:77], 2)
        model = Chain(
            Dense(W0, b0, relu),
            Dense(W1, b1, relu),
            Dense(W2, b2),
            softmax
        )
        return model
    end


    ###
    ### Bayesian Network specifications
    ###
    using Turing

    # setprogress!(false)
    # using Zygote
    # Turing.setadbackend(:zygote)
    using ReverseDiff
    Turing.setadbackend(:reversediff)

    sigma = 3.0

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
# Learning with the train set only
chain_timed = @timed sample(model, NUTS(), MCMCDistributed(), nsteps, 5)
chain = chain_timed.value
elapsed = chain_timed.time
θ = MCMCChains.group(chain, :θ).value
using DelimitedFiles
writedlm("./experiments/$(name)/elapsed.txt", elapsed)

using EvalMetrics
set_encoding(OneTwo())
using Plots

#before Calibration
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

    ŷ_test, pŷ_test = predicitons_analyzer_multiclass(test_x, test_y, independent_param_matrix)

    # println(countmap(ŷ_test))

    writedlm("./experiments/$(name)/ŷ_test_$(i).csv", hcat(ŷ_test, pŷ_test), ',')

    f1 = f1_score(test_y, ŷ_test)
    # gr()
    # prplot(test_y, pŷ_test)
    # no_skill(x) = count(==(1), test_y) / length(test_y)
    # plot!(no_skill, 0, 1, label="No Skill Classifier")
    # savefig("./experiments/$(name)/PRCurve_$(i).png")
    mcc = matthews_correlation_coefficient(test_y, ŷ_test)
    acc = accuracy(test_y, ŷ_test)
    fpr = false_positive_rate(test_y, ŷ_test)
    # fnr = fnr(test_y, ŷ_test)
    # tpr = tpr(test_y, ŷ_test)
    # tnr = tnr(test_y, ŷ_test)
    prec = precision(test_y, ŷ_test)
    recall = true_positive_rate(test_y, ŷ_test)
    # prauc = au_prcurve(test_y, pŷ_test)


    ch = chain[:, :, i]
    summaries, quantiles = describe(ch)
    large_rhat = count(>=(1.01), sort(summaries[:, :rhat]))
    small_rhat = count(<=(0.99), sort(summaries[:, :rhat]))
    avg_acceptance_rate = mean(ch[:acceptance_rate])
    total_numerical_error = sum(ch[:numerical_error])
    avg_ess = mean(summaries[:, :ess])
    # println(describe(summaries[:, :mean]))
    # writedlm("./experiments/$(name)/statistics_chain_$(i).csv", [["oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess", "Accuracy", "f1", "fpr", "precision", "PRAUC"] [large_rhat + small_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, acc, f1, fpr, prec, prauc]], ',')
    writedlm("./experiments/$(name)/statistics_chain_$(i).csv", [["oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess", "Accuracy", "MCC", "f1", "fpr", "precision", "recall"] [large_rhat + small_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, acc, mcc, f1, fpr, prec, recall]], ',')
end

# sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
# _, i = findmax(chain[:lp])
# i = i.I[1]
# elapsed = chain_timed.time
# θ = MCMCChains.group(chain, :θ).value
# θ[i, :]
using DelimitedFiles
begin
    name = "Initializing_5_5_2_with_3.0"
    data = Array{Any}(undef, 10, 5)
    for i = 1:5
        m = readdlm("./experiments/$(name)/statistics_chain_$(i).csv", ',')
        data[:, i] = m[:, 2]
    end
    d = mean(data, dims=2)
    writedlm("./experiments/$(name)/statistics_chain_mean.txt", d)
end
