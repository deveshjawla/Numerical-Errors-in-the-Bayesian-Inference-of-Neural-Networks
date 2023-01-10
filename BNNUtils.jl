"""
Returns means, stds, classifications, majority_voting, majority_conf

where

means, stds are the average logits and std for all the chains

classifications are obtained using threshold

majority_voting, majority_conf are averaged classifications of each chain, if the majority i.e more than 0.5 chains voted in favor then we have a vote in favor of 1. the conf here means how many chains on average voted in favor of 1. majority vote is just rounding of majority conf.
"""
function predicitons_analyzer(test_xs, test_ys, params_set, threshold)::Tuple{Array{Float32},Array{Float32},Array{Int},Array{Int},Array{Float32}}
    means = []
    stds = []
    classifications = []
    majority_voting = []
    majority_conf = []
    for (test_x, test_y) in zip(eachrow(test_xs), test_ys)
        predictions = []
        for theta in params_set
            model = feedforward(theta)
            # make probabilistic by inserting bernoulli distributions, we can make each prediction as probabilistic and then average out the predicitons to give us the final predictions_mean and std
            ŷ = model(collect(test_x))
            append!(predictions, ŷ)

        end
        individual_classifications = map(x -> ifelse(x > threshold, 1, 0), predictions)
        majority_vote = ifelse(mean(individual_classifications) > 0.5, 1, 0)
        majority_conf_ = mean(individual_classifications)
        ensemble_pred_prob = mean(predictions) #average logit
        std_pred_prob = std(predictions)
        ensemble_class = ensemble_pred_prob > threshold ? 1 : 0
        append!(means, ensemble_pred_prob)
        append!(stds, std_pred_prob)
        append!(classifications, ensemble_class)
        append!(majority_voting, majority_vote)
        append!(majority_conf, majority_conf_)
    end

    # for each samples mean, std in zip(means, stds)
    # plot(histogram, mean, std)
    # savefig(./plots of each sample)
    # end
    return means, stds, classifications, majority_voting, majority_conf
end

using StatsBase

"""
Returns a tuple of {Prediction, Prediction probability}

Uses a simple argmax and percentage of samples in the ensemble respectively
"""
function predicitons_analyzer_multiclass(test_xs, test_ys, params_set)::Tuple{Array{Int},Array{Float32}}
    majority_voting = []
    majority_conf = []
    for test_x in eachcol(test_xs)
        predictions = []
        for theta = 1:lastindex(params_set, 1)
            model = feedforward(params_set[theta, :])
            # make probabilistic by inserting bernoulli distributions, we can make each prediction as probabilistic and then average out the predicitons to give us the final predictions_mean and std
            ŷ = model(test_x)
			# println(ŷ)
            append!(predictions, argmax(ŷ)) #the argmax outputs the index of the maximum element in the array
        end
        count_map = countmap(predictions)
		# println(count_map)
        uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
        index_max = argmax(nUniques)
        prediction = uniques[index_max]
        pred_probability = maximum(nUniques) / sum(nUniques)
		# println(prediction, "\t", pred_probability)
        append!(majority_voting, prediction)
        append!(majority_conf, pred_probability)
    end
    return majority_voting, majority_conf
end