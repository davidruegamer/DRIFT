neat_model <- new_model_class(
  classname = "neat_model",
  loss_fn_unnorm = function(y_true, y_pred) { 
    -1 * tfd_log_prob(tfd_normal(loc = 0, scale = 1), y_pred)
  },
  train_step = function(data) {
    
    # Compute gradients
    trainable_vars <- self$trainable_variables
    
    # Exact LL part
    with(tf$GradientTape(persistent = TRUE) %as% tape, {
      
      c(x, y) %<-% data
      
      # Create tensor that you will watch
      x = lapply(x, function(xx) tf$convert_to_tensor(xx, dtype = tf$float32))
      # y = x[[2]]

      # print(x)
      # print(y)
      
      # Watch x and y
      tape$watch(x)
      # tape$watch(y)
      
      # Feed forward
      h = self(x, training=TRUE)
      # print(h)

      # Gradient and the corresponding loss function
      h_prime = tape$gradient(h, x[[2]])
      # print(h_prime)
      loss_value = self$loss_fn_unnorm(x[[2]], h)
      # print(loss_value)
      logLik = tf$reduce_sum(tf$subtract(
        loss_value, 
        tf$math$log(tf$clip_by_value(h_prime, 1e-8, Inf))))
      # print(logLik)
      gradients = tape$gradient(logLik, trainable_vars)
      # print(str(gradients))
      
    })
    
    # Update weights
    self$optimizer$apply_gradients(zip_lists(gradients, trainable_vars))
    
    # Return a named list mapping metric names to current value
    return(list(#unnormalized=tf$reduce_sum(loss_value),
                #normconstant=tf$reduce_mean(h_prime),
                logLik=logLik
                )
    )
  },
  test_step = function(data) {

    with(tf$GradientTape(persistent = TRUE) %as% tape, {

      c(x, y) %<-% data

      # Create tensor that you will watch
      x = lapply(x, function(xx) tf$convert_to_tensor(xx, dtype = tf$float32))
      # y = x[[2]]

      # print(x)
      # print(y)

      # Watch x and y
      tape$watch(x)
      # tape$watch(y)

      # Feed forward
      h = self(x, training=FALSE)

      # Gradient and the corresponding loss function
      h_prime = tape$gradient(h, x[[2]])

      loss_value = self$loss_fn_unnorm(x[[2]], h)
      logLik = tf$reduce_sum(tf$subtract(
        loss_value, 
        tf$math$log(tf$clip_by_value(h_prime, 1e-8, Inf))))

    })

    return(list(#unnormalized=tf$reduce_mean(loss_value),
                #normconstant=tf$reduce_mean(h_prime),
                logLik=logLik
                )
    )

  }
)


