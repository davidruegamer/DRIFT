library(keras)
library(tensorflow)

neat_model <- new_model_class(
  classname = "neat_model",
  train_step = function(data) {
    
    # Compute gradients
    trainable_vars <- self$trainable_variables
    # Exact LL part
    with(tf$GradientTape(persistent = TRUE) %as% tape, {
      
      c(x, y) %<-% data
      
      # Create tensor that you will watch
      x = lapply(x, function(xx) tf$convert_to_tensor(xx, dtype = tf$float32))
      y = x[[2]]

      # print(x)
      # print(y)
      
      # Watch x and y
      tape$watch(x)
      tape$watch(y)
      
      # Feed forward
      h = self(x, training=TRUE)
      
      # Gradient and the corresponding loss function
      h_prime = tape$gradient(h, y)
      loss_value = self$compiled_loss(y, h)
      gradients = tape$gradient(tf$subtract(loss_value, h_prime), trainable_vars)
      
    })
    
    # Update weights
    self$optimizer$apply_gradients(zip_lists(gradients, trainable_vars))
    
    # Return a named list mapping metric names to current value
    return(list(loss=loss_value))
  },
  test_step = function(data) {
    
    h = self(data[[1]], training=FALSE)
    loss_value = self$compiled_loss(data[[1]][[2]], h)
    return(list(loss=loss_value))
    
  }
)


