function print_exmn(x)
    x=Float32.(x)
    min_x = minimum(x)
    max_x = maximum(x)
    mean_x = mean(x)
    println("   min: $min_x, max: $max_x, mean: $mean_x")
end

function print_grad_rssm(grad)
    println("RSSM gradients -----------")
    keys(grad) |> println
    # Core components -
    begin
        println("\nCore components -")
        keys(grad.core) |> println
        # layer deter
        println("grad.core.layer_deter")
        keys(grad.core.layer_deter) |> println
        println("grad.core.layer_deter.layer_1")
        keys(grad.core.layer_deter.layer_1) |> println
        print_exmn(grad.core.layer_deter.layer_1.weight)
        print_exmn(grad.core.layer_deter.layer_1.bias)
        println("grad.core.layer_deter.layer_2")
        keys(grad.core.layer_deter.layer_2) |> println
        print_exmn(grad.core.layer_deter.layer_2.scale)
        # layer stoch
        println("grad.core.layer_stoch")
        keys(grad.core.layer_stoch) |> println
        println("grad.core.layer_stoch.layer_1")
        keys(grad.core.layer_stoch.layer_1) |> println
        print_exmn(grad.core.layer_stoch.layer_1.weight)
        print_exmn(grad.core.layer_stoch.layer_1.bias)
        println("grad.core.layer_stoch.layer_2")
        keys(grad.core.layer_stoch.layer_2) |> println
        print_exmn(grad.core.layer_stoch.layer_2.scale)
        # layer action
        println("grad.core.layer_action")
        keys(grad.core.layer_action) |> println
        println("grad.core.layer_action.layer_1")
        keys(grad.core.layer_action.layer_1) |> println
        print_exmn(grad.core.layer_action.layer_1.weight)
        print_exmn(grad.core.layer_action.layer_1.bias)
        println("grad.core.layer_action.layer_2")
        keys(grad.core.layer_action.layer_2) |> println
        print_exmn(grad.core.layer_action.layer_2.scale)
        # layers gru
        println("grad.core.gru_layers")
        keys(grad.core.gru_layers) |> println
        println("grad.core.gru_layers.layer_1")
        keys(grad.core.gru_layers.layer_1) |> println
        println("grad.core.gru_layers.layer_1.layer_1")
        keys(grad.core.gru_layers.layer_1.layer_1) |> println
        print_exmn(grad.core.gru_layers.layer_1.layer_1.weight)
        print_exmn(grad.core.gru_layers.layer_1.layer_1.bias)
        println("grad.core.gru_layers.layer_1.layer_2")
        keys(grad.core.gru_layers.layer_1.layer_2) |> println
        print_exmn(grad.core.gru_layers.layer_1.layer_2.scale)
        println("grad.core.gru_layers.layer_2")
        keys(grad.core.gru_layers.layer_2) |> println
        print_exmn(grad.core.gru_layers.layer_2.weight)
        print_exmn(grad.core.gru_layers.layer_2.bias)
    end
    
    # Observation components -
    begin
        println("\nObservation components -")
        keys(grad.observation) |> println
        
        println("grad.observation.posterior_layers")
        keys(grad.observation.posterior_layers) |> println
        println("grad.observation.posterior_layers.layer_1")
        keys(grad.observation.posterior_layers.layer_1)
        println("grad.observation.posterior_layers.layer_1.layer_1")
        keys(grad.observation.posterior_layers.layer_1.layer_1)
        print_exmn(grad.observation.posterior_layers.layer_1.layer_1.weight)
        print_exmn(grad.observation.posterior_layers.layer_1.layer_1.bias)
        println("grad.observation.posterior_layers.layer_1.layer_2")
        keys(grad.observation.posterior_layers.layer_1.layer_2)
        print_exmn(grad.observation.posterior_layers.layer_1.layer_2.scale)
        
        println("grad.observation.logit_posterior")
        keys(grad.observation.logit_posterior) |> println
        println("grad.observation.logit_posterior.layer_1")
        keys(grad.observation.logit_posterior.layer_1)
        println("grad.observation.logit_posterior.layer_1.weight")
        print_exmn(grad.observation.logit_posterior.layer_1.weight)
        println("grad.observation.logit_posterior.layer_1.bias")
        print_exmn(grad.observation.logit_posterior.layer_1.bias)
        println("grad.observation.logit_posterior.layer_2")
        println(grad.observation.logit_posterior.layer_2)
    end
    
    # imagination
    begin
        println("\nimagination")
        keys(grad.imagination) |> println
        println("grad.imagination.prior_layers")
        keys(grad.imagination.prior_layers) |> println
        println("grad.imagination.prior_layers.layer_1")
        keys(grad.imagination.prior_layers.layer_1)
        println("grad.imagination.prior_layers.layer_1.layer_1")
        keys(grad.imagination.prior_layers.layer_1.layer_1)
        print_exmn(grad.imagination.prior_layers.layer_1.layer_1.weight)
        print_exmn(grad.imagination.prior_layers.layer_1.layer_1.bias)
        println("grad.imagination.prior_layers.layer_1.layer_2")
        keys(grad.imagination.prior_layers.layer_1.layer_2)
        print_exmn(grad.imagination.prior_layers.layer_1.layer_2.scale)
    
    
        println("grad.imagination.logit_prior")
        keys(grad.imagination.logit_prior) |> println
        println("grad.imagination.logit_prior.layer_1")
        keys(grad.imagination.logit_prior.layer_1)
        println("grad.imagination.logit_prior.layer_1.weight")
        print_exmn(grad.imagination.logit_prior.layer_1.weight)
        println("grad.imagination.logit_prior.layer_1.bias")
        print_exmn(grad.imagination.logit_prior.layer_1.bias)
        println("grad.imagination.logit_prior.layer_2")
        println(grad.imagination.logit_prior.layer_2)
    end
end




# Loos at Gradients ----------------------------
begin
    # Encoder gradients -----------
    begin
        println(keys(grad[1].encoder))
        println(keys(grad[1].encoder.convolve))
        println(keys(grad[1].encoder.convolve))
        all(grad[1].encoder.convolve.layer_1.weight .≈ 0) |> println
        all(grad[1].encoder.convolve.layer_1.bias .≈ 0) |> println
        grad[1].encoder.convolve.layer_2 |> println
        all(grad[1].encoder.convolve.layer_3.scale .≈ 0) |> println
        all(grad[1].encoder.convolve.layer_4.weight .≈ 0) |> println
        all(grad[1].encoder.convolve.layer_4.bias .≈ 0) |> println
        grad[1].encoder.convolve.layer_5 |> println
        all(grad[1].encoder.convolve.layer_6.scale .≈ 0) |> println
        
        all(grad[1].encoder.convolve.layer_7.weight .≈ 0) |> println
        all(grad[1].encoder.convolve.layer_7.bias .≈ 0) |> println
        grad[1].encoder.convolve.layer_8 |> println
        all(grad[1].encoder.convolve.layer_9.scale .≈ 0) |> println
        
        all(grad[1].encoder.convolve.layer_10.weight .≈ 0) |> println
        all(grad[1].encoder.convolve.layer_10.bias .≈ 0) |> println
        grad[1].encoder.convolve.layer_11 |> println
    end
    
    # RSSM gradients -----------
    print_grad_rssm(grad[1].rssm)
    
    # Decoder gradients -----------
    begin
        println("Decoder gradients")
        keys(grad[1].decoder) |> println
    
        println("spatialize_deter")
        keys(grad[1].decoder.spatialize_deter) |> println
        println("spatialize_deter.layer_1")
        keys(grad[1].decoder.spatialize_deter.layer_1) |> println
        print_exmn(grad[1].decoder.spatialize_deter.layer_1.weight)
        print_exmn(grad[1].decoder.spatialize_deter.layer_1.bias)
        println("spatialize_deter.layer_2")
        println(grad[1].decoder.spatialize_deter.layer_2)
    
        println("spatialize_stoch")
        keys(grad[1].decoder.spatialize_stoch) |> println
        println("spatialize_stoch.layer_1")
        keys(grad[1].decoder.spatialize_stoch.layer_1) |> println
        print_exmn(grad[1].decoder.spatialize_stoch.layer_1.weight)
        println("spatialize_stoch.layer_2")
        keys(grad[1].decoder.spatialize_stoch.layer_2) |> println
        print_exmn(grad[1].decoder.spatialize_stoch.layer_2.scale)
        println("spatialize_stoch.layer_3")
        keys(grad[1].decoder.spatialize_stoch.layer_3) |> println
        println("spatialize_stoch.layer_3.weight")
        print_exmn(grad[1].decoder.spatialize_stoch.layer_3.weight)
        println("spatialize_stoch.layer_3.bias")
        print_exmn(grad[1].decoder.spatialize_stoch.layer_3.bias)
        println("spatialize_stoch.layer_4")
        println(grad[1].decoder.spatialize_stoch.layer_4)
    
        println("deconvolve")
        keys(grad[1].decoder.deconvolve) |> println
        println("deconvolve.layer_1")
        keys(grad[1].decoder.deconvolve.layer_1) |> println
        print_exmn(grad[1].decoder.deconvolve.layer_1.scale)
        println("deconvolve.layer_2")
        grad[1].decoder.deconvolve.layer_2 |> println
        println("deconvolve.layer_3")
        keys(grad[1].decoder.deconvolve.layer_3) |> println
        print_exmn(grad[1].decoder.deconvolve.layer_3.weight)
        print_exmn(grad[1].decoder.deconvolve.layer_3.bias)
        println("deconvolve.layer_4")
        keys(grad[1].decoder.deconvolve.layer_4) |> println
        print_exmn(grad[1].decoder.deconvolve.layer_4.scale)
        println("deconvolve.layer_5")
        grad[1].decoder.deconvolve.layer_5 |> println
        println("deconvolve.layer_6")
        keys(grad[1].decoder.deconvolve.layer_6) |> println
        print_exmn(grad[1].decoder.deconvolve.layer_6.weight)
        println("deconvolve.layer_6.bias")
        print_exmn(grad[1].decoder.deconvolve.layer_6.bias)
        println("deconvolve.layer_7")
        keys(grad[1].decoder.deconvolve.layer_7) |> println
        print_exmn(grad[1].decoder.deconvolve.layer_7.scale)
        println("deconvolve.layer_8")
        grad[1].decoder.deconvolve.layer_8 |> println
        println("deconvolve.layer_9")
        keys(grad[1].decoder.deconvolve.layer_9) |> println
        println("deconvolve.layer_9.weight")
        print_exmn(grad[1].decoder.deconvolve.layer_9.weight)
        println("deconvolve.layer_9.bias")
        print_exmn(grad[1].decoder.deconvolve.layer_9.bias)
        println("deconvolve.layer_10")
        keys(grad[1].decoder.deconvolve.layer_10) |> println
        println("deconvolve.layer_10.scale")
        print_exmn(grad[1].decoder.deconvolve.layer_10.scale)
        println("deconvolve.layer_11")
        grad[1].decoder.deconvolve.layer_11 |> println
        println("deconvolve.layer_12")
        keys(grad[1].decoder.deconvolve.layer_12) |> println
        println("deconvolve.layer_12.weight")
        print_exmn(grad[1].decoder.deconvolve.layer_12.weight)
        println("deconvolve.layer_12.bias")
        print_exmn(grad[1].decoder.deconvolve.layer_12.bias)    
    end
end


