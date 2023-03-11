source("neat_model_class.R")
library(tfprobability)

######################## Networks #############################
### Generic function to built an MLP
mlp_with_default_layer <- function(size, default_layer)
{
  
  function(input){
    
    x <- input %>% default_layer(units = size[1])
    
    for(i in 2:length(size))
      x <- x %>% default_layer(units = size[i])
    
    return(x)
    
  }
  
}

### ReLU network
relu_network <- function(size) mlp_with_default_layer(size, 
                         default_layer = function(...) 
                           layer_dense(..., activation = "relu")
                         )

### Special layer for monotonocity
layer_nonneg_tanh <- function(...) layer_dense(..., activation = "tanh", 
                                               kernel_constraint = 
                                                 keras$constraints$non_neg(),
                                               kernel_initializer = 
                                                 keras$initializers$random_uniform(minval = 0, 
                                                                                   maxval = 1))

### Monotonic NN
nonneg_tanh_network <- function(size) mlp_with_default_layer(
  size, 
  default_layer = layer_nonneg_tanh
)

### Tensor-product network
tensorproduct_network <- function(inpY, inpX)
  deepregression::tf_row_tensor(inpY, inpX) %>% 
  layer_dense(units = 1,
              kernel_constraint = 
                keras$constraints$non_neg(),
              kernel_initializer = 
                keras$initializers$random_uniform(minval = 0, 
                                                  maxval = 1),
              use_bias = FALSE)

### Monotonic NN with interactions
interconnected_network <- function(inpY, inpX, 
                                   network_default = 
                                     nonneg_tanh_network(c(50, 50, 10)),
                                   top_layer = layer_nonneg_tanh(units = 1L))
{
  
  layer_concatenate(list(inpX, inpY)) %>% 
    network_default() %>% 
    top_layer %>% 
    layer_activation(activation = "softplus")
  
}

### Layer for inverse sigma
layer_inverse_exp <- function(object, units, ...)
{
  
  function(object) tf$multiply(layer_dense(object, units, ...), -0.5)
  
}

### Multiplication NN
locscale_network <- function(inpY, inpX,
                             mu_top_layer = layer_dense(units = 1L),
                             sd_top_layer = layer_inverse_exp(units = 1L),
                             top_layer = layer_nonneg_tanh(units = 1L))
{
  
  loc <- inpX %>% mu_top_layer()
  scale_inv <- inpX %>% sd_top_layer()
  outpY <- inpY %>% 
    top_layer %>% 
    layer_activation(activation = "softplus")
  
  tf$subtract(tf$multiply(scale_inv, outpY), loc)
  
}

neat <- function(
    dim_features,
    net_y_size_trunk = nonneg_tanh_network(c(50, 50, 10)),
    net_x_arch_trunk = relu_network(c(100,100)),
    type = c("tp", "ls", "inter"),
    base_distribution = tfd_normal(loc = 0, scale = 1),
    optimizer = optimizer_adam()
)
{
  
  type <- match.arg(type)
  
  # inputs
  inpX <- layer_input(dim_features)
  inpY <- layer_input(1L)
  
  # (intermediate) outputs
  outpX <- inpX %>% net_x_arch_trunk()
  
  # outputs
  outp <- switch (type,
    tp = tensorproduct_network(net_y_size_trunk(inpY), outpX),
    ls = locscale_network(net_y_size_trunk(inpY), outpX),
    inter = interconnected_network(inpY, outpX)
  )
  
  mod <- neat_model(list(inpX, inpY), outp)
  
  bd <- base_distribution
  
  mod %>% compile(
    loss = function(y_true, y_pred) -1 * tfd_log_prob(bd, y_pred),
    optimizer = optimizer
  )
  
  return(mod)
  
}