######################## Networks #############################
### Generic function to built an MLP
#' 
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

### Feature-specific network for NAMs
feature_specific_network <- function(size = c(64,64,32),
                                     default_layer = function(...)
                                       layer_dense(..., activation = "relu")
                                     )
{
  
  function(x){
    tf$concat(
      lapply(tf$split(x, num_or_size_splits = x$shape[[2]], axis=1L),
             function(xx) xx %>% mlp_with_default_layer(
               size = size,
               default_layer = default_layer)()
      ), axis=1L
      )
  }
  
}

### Semi-structured NAMs
semi_structured_nams <- function(size_nam = c(64,64,32),
                                 size_deep = c(100,100,10),
                                 default_layer_nam = function(...)
                                   layer_dense(..., activation = "relu"),
                                 default_layer_deep = function(...)
                                   layer_dense(..., activation = "relu")
)
{
  
  function(x){
    tf$concat(
      c(
        lapply(tf$split(x, num_or_size_splits = x$shape[[2]], axis=1L),
               function(xx) xx %>% mlp_with_default_layer(
                 size = size_nam,
                 default_layer = default_layer_nam)()
        ),
        list(
          x %>% mlp_with_default_layer(
            size = size_deep,
            default_layer = default_layer_deep
          )()
        )
      ), axis=1L
    )
  }
  
}

### Special layer for monotonocity
layer_nonneg_tanh <- function(...) layer_dense(..., activation = "tanh", 
                                               kernel_constraint = 
                                                 keras$constraints$non_neg(),
                                               kernel_initializer = 
                                                 keras$initializers$random_uniform(minval = 0, 
                                                                                   maxval = 1))

layer_nonneg_lin <- function(...) layer_dense(..., activation = "linear", 
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
  deeptrafo:::layer_mono_multi(units = 1, 
                              dim_bsp = inpY$shape[[2]]*inpX$shape[[2]],
                              trafo = deeptrafo:::mono_trafo_multi, 
                              kernel_regularizer = NULL, 
                              trainable = TRUE)()

### Monotonic NN with interactions
interconnected_network <- function(inpY, inpX, 
                                   network_default = 
                                     nonneg_tanh_network(c(50, 50, 10)),
                                   top_layer = layer_nonneg_lin(units = 1L))
{
  
  layer_concatenate(list(inpX, inpY)) %>% 
    network_default() %>% 
    top_layer
  
}

### Layer for inverse sigma
layer_inverse_exp <- function(object, units, ...)
{
  
  function(object) tf$math$exp(tf$multiply(layer_dense(object, units, ...), -0.5))
  
}

### Multiplication NN
locscale_network <- function(inpY, inpX,
                             mu_top_layer = layer_dense(units = 1L),
                             sd_top_layer = layer_inverse_exp(units = 1L),
                             top_layer = layer_nonneg_lin(units = 1L))
{
  
  loc <- inpX %>% mu_top_layer()
  scale_inv <- inpX %>% sd_top_layer()
  outpY <- inpY %>% 
    top_layer 
  
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
  
  mod <- neat_model(list(inpX, inpY), outp, 
                    base_distribution = base_distribution)
  
  mod %>% compile(
    loss = function(y_true, y_pred) -1 * tfd_log_prob(bd, y_pred),
    optimizer = optimizer
  )
  
  return(mod)
  
}

sneat <- function(
    dim_features,
    ...
    )
{
  
  neat(
    dim_features = dim_features,
    net_x_arch_trunk = feature_specific_network(c(64,64,32)),
    ...
  )
  
}

sesneat <- function(
    dim_features,
    ...
)
{
  
  neat(
    dim_features = dim_features,
    net_x_arch_trunk = semi_structured_nams(),
    ...
  )
  
}