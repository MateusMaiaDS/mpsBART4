# Getting the BART wrapped function
#' @export
mpsbart <- function(x_train,
                  y,
                  x_test,
                  n_tree = 2,
                  n_mcmc = 2000,
                  n_burn = 500,
                  alpha = 0.95,
                  beta = 2,
                  dif_order = 2,
                  # df_splines = 10,
                  nIknots = 1,
                  df = 3,
                  sigquant = 0.9,
                  kappa = 2,
                  scale_bool = TRUE,
                  # Hyperparam for tau_b and tau_b_0
                  nu = 2,
                  delta = 1,
                  a_delta = 0.0001,
                  d_delta = 0.0001,
                  df_tau_b = 3,
                  prob_tau_b = 0.9,
                  intercept_model = TRUE,
                  stump = FALSE,
                  a # "a" are the constant for the constant values to penalise the number of trees
                  ) {

     # Verifying if x_train and x_test are matrices
     if(!is.data.frame(x_train) || !is.data.frame(x_test)){
          stop("Insert valid data.frame for both data and xnew.")
     }

     # Creating the col
     original_p <- 1:ncol(x_train)-1


     # Getting the valid
     dummy_x <- caret::dummyVars(~.,data = x_train)

     col_names <- attr(dummy_x$terms,"term.labels")
     dummy_x_train_m <- predict(object = dummy_x,newdata = x_train)
     dummy_x_test_m <- predict(object = dummy_x,newdata = x_test)

     recode_names <- recode_vars(x_train = x_train,dummy_obj = dummy_x)
     n_levels <- c(table(recode_names)-1) # It also counts the zero

     x_train_scale <- dummy_x_train_m
     x_test_scale <- dummy_x_test_m

     # Scaling x
     x_min <- apply(as.matrix(x_train_scale),2,min)
     x_max <- apply(as.matrix(x_train_scale),2,max)

     # Storing the original
     x_train_original <- x_train
     x_test_original <- x_test


     # Normalising all the columns
     for(i in 1:ncol(x_train)){
             x_train_scale[,i] <- normalize_covariates_bart(y = x_train_scale[,i],a = x_min[i], b = x_max[i])
             x_test_scale[,i] <- normalize_covariates_bart(y = x_test_scale[,i],a = x_min[i], b = x_max[i])
     }


     # Scaling the y
     min_y <- min(y)
     max_y <- max(y)

     # Getting the min and max for each column
     min_x <- apply(x_train_scale,2,min)
     max_x <- apply(x_train_scale, 2, max)

     # Getting the internal knots
     # knots <- apply(x_train_scale,
     #                2,
     #                function(x){quantile(x,seq(0,1,length.out = nIknots+2))[-c(1,nIknots+2)]})

     knots <- apply(x_train_scale,
                    2,
                    function(x){seq(min(x),max(x),length.out = nIknots)})


     # Creating a array of basis functions (only for continuous variables)
     continuous_vars <- col_names[!(col_names %in% dummy_x$facVars)]

     B_train_arr <- array(data = NA,
                          dim = c(nrow(x_train_scale),
                                  nrow(knots)+3, # +3 here because is a natural spline
                                  ncol(x_train_scale[,continuous_vars, drop = FALSE])))

     B_test_arr <- array(data = NA,
                          dim = c(nrow(x_test_scale),
                                  nrow(knots)+3,  # +3 here because is a natural spline
                                  ncol(x_test_scale[,continuous_vars, drop = FALSE])))

     # Setting new parameters for the spline
     ndx <- nIknots
     dx <- (max_x-min_x)/ndx

     # New_knots
     new_knots <- matrix()
     new_knots <- mapply(min_x,max_x, FUN = function(MIN,MAX){seq(MIN-3*dx,MAX+3*dx, by = dx)})

     # Creating the natural B-spline for each predictor
     for(i in 1:length(continuous_vars)){
             # B_train_obj <- splines::bs(x = x_train_scale[,continuous_vars[i], drop = FALSE],knots = knots[,continuous_vars[i]],
             #                            intercept = FALSE,
             #                            Boundary.knots = c((min_x[i]),max_x[i]))
             B_train_obj <- splines::spline.des(x = x_train_scale[,continuous_vars[i], drop = FALSE],
                                                knots = new_knots[,continuous_vars[i]],
                                                ord = 4,
                                                derivs = 0*x_train_scale[,continuous_vars[i], drop = FALSE])$design

             B_train_arr[,,i] <- as.matrix(B_train_obj)
             # B_test_arr[,,i] <- as.matrix(predict(B_train_obj,newx = x_test_scale[,continuous_vars[i], drop = FALSE]))
             B_test_arr[,,i] <- splines::spline.des(x = x_test_scale[,continuous_vars[i], drop = FALSE],
                                                    knots = new_knots[,continuous_vars[i]],
                                                    ord = 4,
                                                    derivs = 0*x_test_scale[,continuous_vars[i], drop = FALSE])$design

     }

     # plot(x_train_scale, B_train_obj[,1], ylim = range(B_train_obj), type = 'n',
     #      ylab = "B")
     # for(j in 1:ncol(B_train_obj)) {
     #         points(x_train_scale, B_train_obj[,j], col = j)
     # }

     # R-th difference order matrix
     if(dif_order!=0){
        D <- D_gen(p = ncol(B_train_arr[,,1]),n_dif = dif_order)
     } else {
        D <- diag(nrow = ncol(B_train_arr[,,1]))
     }

     # Scaling "y"
     if(scale_bool){
        y_scale <- normalize_bart(y = y,a = min_y,b = max_y)
        tau_b_0 <- (4*n_tree*(kappa^2))
        tau_b <- tau_mu <- (4*n_tree*(kappa^2))
        # tau_b <- tau_mu <- 0.1

     } else {
        y_scale <- y
        tau_b_0 <- tau_b <- tau_mu <- (4*n_tree*(kappa^2))/((max_y-min_y)^2)
     }


     # Getting the naive sigma value
     nsigma <- naive_sigma(x = x_train_scale,y = y_scale)

     # Calculating tau hyperparam
     a_tau <- df/2

     # Calculating lambda
     qchi <- stats::qchisq(p = 1-sigquant,df = df,lower.tail = 1,ncp = 0)
     lambda <- (nsigma*nsigma*qchi)/df
     d_tau <- (lambda*df)/2


     # Call the bart function
     tau_init <- nsigma^(-2)
     mu_init <- mean(y_scale)
     # tau_init <- 0.01

     # Creating the vector that stores all trees
     all_tree_post <- vector("list",length = round(n_mcmc-n_burn))


     # Another way to define the prior for \tau_b
     a_tau_b <- 0.5*df_tau_b

     # Getting the \tau_b
     # naive_tau_b <- optim(par = rep(1, 2), fn = nll, dat=y_scale,
     #                      x = x_train_scale,
     #                      B = B_train_arr,
     #                      P = crossprod(D),n_tree = n_tree,
     #                      tau_b_0_ = tau_b_0,
     #                      method = "L-BFGS-B",
     #                      hessian = TRUE,
     #                      lower = rep(0.0001, 2))$par[2]
     #
     # d_tau_b <- optim(par = 0.001,d_tau_b_rate,method = "L-BFGS-B",
     #                  lower = 0.001,df_tau_b = df_tau_b,
     #                  prob_tau_b = prob_tau_b,
     #                  naive_tau_b = naive_tau_b)$par

     # Getting the number of basis
     d_pred <- dim(B_train_arr)[3]

     a_tau_b <- d_tau_b <- 0.01

     # Generating the BART obj
     bart_obj <- sbart(x_train_scale,
          y_scale,
          x_test_scale,
          B_train = B_train_arr,
          B_test = B_test_arr,
          D = D,
          n_tree,
          n_mcmc,
          n_burn,
          tau_init,
          mu_init,
          tau_mu,
          tau_b,
          tau_b_0, # Same initial value as tau_b
          alpha,
          beta,
          a_tau,d_tau,
          nu,delta,
          a_delta,d_delta, # Hypeparameters from delta
          a_tau_b,d_tau_b,
          original_p, # Getting the p available variables
          n_levels, # Getting the sample levels
          intercept_model = intercept_model,
          stump,
          a)


     if(scale_bool){
             # Tidying up the posterior elements
             y_train_post <- unnormalize_bart(z = bart_obj[[1]],a = min_y,b = max_y)
             y_test_post <- unnormalize_bart(z = bart_obj[[2]],a = min_y,b = max_y)
             for(i in 1:round(n_mcmc-n_burn)){
                     all_tree_post[[i]] <-  unnormalize_bart(z = bart_obj[[4]][,,i],a = min_y,b = max_y)
             }
             tau_post <- bart_obj[[3]]/((max_y-min_y)^2)
             tau_b_post <-  bart_obj[[5]]/((max_y-min_y)^2)
             tau_b_post_intercept <-  bart_obj[[6]]/((max_y-min_y)^2)

     } else {
             y_train_post <- bart_obj[[1]]
             y_test_post <- bart_obj[[2]]
             tau_post <- bart_obj[[3]]
             for(i in 1:round(n_mcmc-n_burn)){
                     all_tree_post[[i]] <-  bart_obj[[4]][,,i]
             }
             tau_b_post <-  bart_obj[[5]]
             tau_b_post_intercept <-  bart_obj[[6]]

     }

     # Return the list with all objects and parameters
     return(list(y_hat = y_train_post,
                 y_hat_test = y_test_post,
                 tau_post = tau_post,
                 tau_b_post = tau_b_post,
                 tau_b_post_intercept = tau_b_post_intercept,
                 all_tree_post = all_tree_post,
                 prior = list(n_tree = n_tree,
                              alpha = alpha,
                              beta = beta,
                              tau_mu = tau_mu,
                              a_tau = a_tau,
                              d_tau = d_tau),
                 mcmc = list(n_mcmc = n_mcmc,
                             n_burn = n_burn),
                 data = list(x_train = x_train,
                             y = y,
                             x_test = x_test,
                             grow_accept = bart_obj[[7]])))
}


#
