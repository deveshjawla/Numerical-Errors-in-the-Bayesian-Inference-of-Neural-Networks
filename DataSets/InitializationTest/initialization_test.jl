PATH = @__DIR__
cd(PATH)

using Random
coeffs = rand(MersenneTwister(0), 2)
X = rand(MersenneTwister(2), 2, 100)
Y_logits = [(sum(coeffs .* i) - 6) for i in eachcol(X)]
Y = [(i > 0 ? 1 : 0) for i in Y_logits]

name = "initialisation_1.0"

mkpath("./experiments/$(name)")

###
### Dense Network specifications
###

input_size = size(X)[1]
l1, l2, l3, l4, l5 = 100, 100, 100, 100, 1
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
ol5 = l4 * l5 + l5

total_num_params = nl1 + nl2 + nl3 + nl4 + ol5

using Flux

function feedforward(θ::AbstractVector)
    W0 = reshape(θ[1:200], 100, 2)
    b0 = θ[201:300]
    W1 = reshape(θ[301:10300], 100, 100)
    b1 = θ[10301:10400]
    W2 = reshape(θ[10401:20400], 100, 100)
    b2 = θ[20401:20500]
    W3 = reshape(θ[20501:30500], 100, 100)
    b3 = θ[30501:30600]
    W4 = reshape(θ[30601:30700], 1, 100)
    b4 = θ[30701:30701]
    model = Chain(
        Dense(W0, b0, relu),
        Dense(W1, b1, relu),
        Dense(W2, b2, relu),
        Dense(W3, b3, relu),
        Dense(W4, b4, σ)
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

sigma = 1.0
#Here we define the layer by layer initialisation
# sigma = vcat(sqrt(2 / (input_size + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), 4 * sqrt(2 / (l4 + l5)) * ones(ol5))

using DelimitedFiles
@model bayesnn(x, y) = begin
    θ ~ MvNormal(zeros(total_num_params), ones(total_num_params) .* sigma)
    nn = feedforward(θ)
    ŷ = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Bernoulli(ŷ[i])
    end
end

# if isfile("./experiments/$(name)/W0.csv") == false
#     weights = Flux.params(nn)
#     W0 = weights[1]
#     b0 = weights[2]
#     W1 = weights[3]
#     b1 = weights[4]
#     W2 = weights[5]
#     b2 = weights[6]
#     W3 = weights[7]
#     b3 = weights[8]
#     W4 = weights[9]
#     b4 = weights[10]
#     activations_ = Flux.activations(nn, x)
#     a1 = activations_[1]
#     a2 = activations_[2]
#     a3 = activations_[3]
#     a4 = activations_[4]
#     a5 = activations_[5]
#     writedlm("./experiments/$(name)/W0.csv", W0, ',')
#     writedlm("./experiments/$(name)/b0.csv", b0, ',')
#     writedlm("./experiments/$(name)/W1.csv", W1, ',')
#     writedlm("./experiments/$(name)/b1.csv", b1, ',')
#     writedlm("./experiments/$(name)/W2.csv", W2, ',')
#     writedlm("./experiments/$(name)/b2.csv", b2, ',')
#     writedlm("./experiments/$(name)/W3.csv", W3, ',')
#     writedlm("./experiments/$(name)/b3.csv", b3, ',')
#     writedlm("./experiments/$(name)/W4.csv", W4, ',')
#     writedlm("./experiments/$(name)/b4.csv", b4, ',')
#     writedlm("./experiments/$(name)/a1.csv", a1, ',')
#     writedlm("./experiments/$(name)/a2.csv", a2, ',')
#     writedlm("./experiments/$(name)/a3.csv", a3, ',')
#     writedlm("./experiments/$(name)/a4.csv", a4, ',')
#     writedlm("./experiments/$(name)/a5.csv", a5, ',')
# end


###
### Inference
###

# ScikitLearn.CrossValidation.StratifiedKFold([ones(10)...,zeros(5)...], n_folds=5)
# for i in 1:10
initial_dist = MvNormal(zeros(total_num_params), ones(total_num_params) .* sigma)
init_params = rand(initial_dist)
chain_timed = @timed Turing.sample(bayesnn(X, Y), NUTS(10, 0.65), 100, verbose = true)
chain = chain_timed.value
elapsed = chain_timed.time
writedlm("./experiments/$(name)/elapsed.txt", elapsed)
θ = MCMCChains.group(chain, :θ).value
params_after_100_steps = collect.(eachrow(θ[:, :, 1]))[100]

summaries, quantiles = describe(chain)

writedlm("./experiments/$(name)/init_params.csv", init_params, ',')
writedlm("./experiments/$(name)/params_after_100_steps.csv", params_after_100_steps, ',')

function break_params(θ::AbstractVector)
    W0 = θ[1:200]
    b0 = θ[201:300]
    W1 = θ[301:10300]
    b1 = θ[10301:10400]
    W2 = θ[10401:20400]
    b2 = θ[20401:20500]
    W3 = θ[20501:30500]
    b3 = θ[30501:30600]
    W4 = θ[30601:30700]
    b4 = θ[30701:30701]
    return W0, W1, W2, W3, W4
end

W0, W1, W2, W3, W4 = break_params(init_params)
using Distributions

W0_dist = fit(Normal, W0)
W1_dist = fit(Normal, W1)
W2_dist = fit(Normal, W2)
W3_dist = fit(Normal, W3)

using Plots

plot(x -> pdf(W0_dist, x), xlims=(-1.0, 1.0), label="Layer 1, μ=$(round(W0_dist.μ, digits =4)), σ=$(round(W0_dist.σ, digits = 4))", xlabel="Weights", size=(800, 600))
plot!(x -> pdf(W1_dist, x), xlims=(-1.0, 1.0), label="Layer 2, μ=$(round(W1_dist.μ, digits =4)), σ=$(round(W1_dist.σ, digits = 4))")
plot!(x -> pdf(W2_dist, x), xlims=(-1.0, 1.0), label="Layer 3, μ=$(round(W2_dist.μ, digits =4)), σ=$(round(W2_dist.σ, digits = 4))")
plot!(x -> pdf(W3_dist, x), xlims=(-1.0, 1.0), label="Layer 4, μ=$(round(W3_dist.μ, digits =4)), σ=$(round(W3_dist.σ, digits = 4))")
savefig("./experiments/$(name)/Layers.png")

model = feedforward(init_params)
activations = Flux.activations(model, X[:, 1])

histogram(activations[1])
histogram(activations[2])
histogram(activations[3])
histogram(activations[4])
activations_dist_1 = fit(Normal, activations[1])
activations_dist_2 = fit(Normal, activations[2])
activations_dist_3 = fit(Normal, activations[3])
activations_dist_4 = fit(Normal, activations[4])

plot(x -> pdf(activations_dist_1, x), xlims=(-1.0, 1.0), label="Layer 1, μ=$(round(activations_dist_1.μ, digits =4)), σ=$(round(activations_dist_1.σ, digits = 4))", xlabel="Activations", size=(800, 600))
plot!(x -> pdf(activations_dist_2, x), xlims=(-1.0, 1.0), label="Layer 2, μ=$(round(activations_dist_2.μ, digits =4)), σ=$(round(activations_dist_2.σ, digits = 4))")
plot!(x -> pdf(activations_dist_3, x), xlims=(-1.0, 1.0), label="Layer 3, μ=$(round(activations_dist_3.μ, digits =4)), σ=$(round(activations_dist_3.σ, digits = 4))")
plot!(x -> pdf(activations_dist_4, x), xlims=(-1.0, 1.0), label="Layer 4, μ=$(round(activations_dist_4.μ, digits =4)), σ=$(round(activations_dist_4.σ, digits = 4))")
savefig("./experiments/$(name)/Activations.png")

W0, W1, W2, W3, W4 = break_params(params_after_100_steps)
using Distributions

W0_dist = fit(Normal, W0)
W1_dist = fit(Normal, W1)
W2_dist = fit(Normal, W2)
W3_dist = fit(Normal, W3)

using Plots

plot(x -> pdf(W0_dist, x), xlims=(-1.0, 1.0), label="Layer 1 μ=$(round(W0_dist.μ, digits =4)), σ=$(round(W0_dist.σ, digits = 4))", xlabel="Weights", size=(800, 600))
plot!(x -> pdf(W1_dist, x), xlims=(-1.0, 1.0), label="Layer 2, μ=$(round(W1_dist.μ, digits =4)), σ=$(round(W1_dist.σ, digits = 4))")
plot!(x -> pdf(W2_dist, x), xlims=(-1.0, 1.0), label="Layer 3, μ=$(round(W2_dist.μ, digits =4)), σ=$(round(W2_dist.σ, digits = 4))")
plot!(x -> pdf(W3_dist, x), xlims=(-1.0, 1.0), label="Layer 4, μ=$(round(W3_dist.μ, digits =4)), σ=$(round(W3_dist.σ, digits = 4))")
savefig("./experiments/$(name)/Layers_100.png")

model = feedforward(params_after_100_steps)
activations = Flux.activations(model, X[:, 1])

activations_dist_1 = fit(Normal, activations[1])
activations_dist_2 = fit(Normal, activations[2])
activations_dist_3 = fit(Normal, activations[3])
activations_dist_4 = fit(Normal, activations[4])


plot(x -> pdf(activations_dist_1, x), xlims=(-1.0, 1.0), label="Layer 1, μ=$(round(activations_dist_1.μ, digits =4)), σ=$(round(activations_dist_1.σ, digits = 4))", xlabel="Activations", size=(800, 600))
plot!(x -> pdf(activations_dist_2, x), xlims=(-1.0, 1.0), label="Layer 2, μ=$(round(activations_dist_2.μ, digits =4)), σ=$(round(activations_dist_2.σ, digits = 4))")
plot!(x -> pdf(activations_dist_3, x), xlims=(-1.0, 1.0), label="Layer 3, μ=$(round(activations_dist_3.μ, digits =4)), σ=$(round(activations_dist_3.σ, digits = 4))")
plot!(x -> pdf(activations_dist_4, x), xlims=(-1.0, 1.0), label="Layer 4, μ=$(round(activations_dist_4.μ, digits =4)), σ=$(round(activations_dist_4.σ, digits = 4))")
savefig("./experiments/$(name)/Activations_100.png")
# # A helper to create NN from weights `theta` and run it through data `x`
# nn_forward(x, θ) = reconstruct(θ)(x)

# # Return the average predicted value across
# # multiple weights.
# function nn_predict(x, theta, start, step, stop)
#     return mean([nn_forward(x, theta[i, :])[1] for i in start:step:stop])
# end;


# param_matrix = mapreduce(permutedims, vcat, params_set)
