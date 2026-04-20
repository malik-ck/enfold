
<!-- README.md is generated from README.Rmd. Please edit that file -->

# enfold

<!-- badges: start -->

<!-- badges: end -->

`enfold` is a model ensembling toolbox that makes nested
cross-validation convenient and accessible. Applicable to all supervised
learning problems, but designed with nuisance model estimation for
causal inference in mind. Supports many types of advanced learner
objects found in modern machine learning ecosystems, such as pipelines
and hyperparameter optimizers. There are three distinctive features of
`enfold`:

- List learners, which train one learner and output a list of valid
  predictions (e.g., fitting an entire elastic net regularization path)
- Predictions from cross-validated ensembles as first-class feature
- Very simple learner creation via built-in factory makers

The brief example below is followed by a list learner illustration.

## Installation

You can install the development version of `enfold` from
[GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("malik-ck/enfold")
```

## Example

We use the built-in `enfold_demo` data set to predict HbA1c. We use a
simple mean learner, a random forest, and a generalized linear model
(GLM). For illustration, we create a GLM template in the code block
below, though `enfold` of course ships one as `lrn_glmnet()`. We do so
by using `make_learner_factory()`, which needs two arguments: `fit` and
`preds`. The former needs to be a function with arguments `x` and `y`.
The latter needs to be a function with arguments `object` and `data`.
Additional arguments we want our learner factory to have (e.g.,
`family`) are passed as named arguments to `make_learner_factory()`.

``` r
set.seed(7441)
library(enfold)
data(enfold_demo, package = "enfold")

predictors <- enfold_demo[, setdiff(names(enfold_demo), "hba1c")]
outcome <- enfold_demo$hba1c

# Create a GLM template
# 'family' is passed without value, which makes it have no default in the resulting factory.
my_glm_maker <- make_learner_factory(
  fit = function(x, y) glm(y ~ ., data = data.frame(y = y, x), family = family),
  preds = function(object, data) {
    predict(object, newdata = data, type = "response")
  },
  family
)

# Now instantiate a learner
# The first argument is 'name', added by the learner factory maker, and required
my_glm <- my_glm_maker(name = "My GLM", family = gaussian())

# The resulting template can also be used for other GLM families. 
# Say we wanted a logistic regression instead:
# my_logistic <- my_glm_maker(name = "My Logistic", family = binomial())

# Other learners to add: mean learner, random forest
mean_learner <- lrn_mean("Mean")
ranger_learner <- lrn_ranger("Random Forest", num.trees = 300)
```

We now build superlearner ensembles using the three learners. We build
the ensemble using three-fold cross-validation, and run additionally an
outer loop of cross-validation to get pure out-of-fold predictions for
the superlearner.

``` r
superlearner_ensembles <- initialize_enfold(predictors, outcome) |>
  add_learners(my_glm, mean_learner, ranger_learner) |>
  add_metalearners(mtl_superlearner("SL")) |>
  add_cv_folds(inner_cv = 3L, outer_cv = 3L) |>
  fit()
#> Warning: package 'future' was built under R version 4.5.3
```

We have a number of functions available to evaluate fit ensembles.
Below, we use `risk()` to assess mean loss for our superlearner as well
as individual learners via `risk_learners()`. Passing `type = "cv"`
calculates risk exclusively on out-of-fold data using the fold
specification added by `add_cv_folds`.

``` r
risk(superlearner_ensembles, type = "cv", loss_fun = loss_gaussian())
#>        SL 
#> 0.1850677

risk_learners(
  superlearner_ensembles,
  type = "cv",
  loss_fun = loss_gaussian()
)
#>        My GLM          Mean Random Forest 
#>     0.2095792     0.4487277     0.1858747
```

Check the package vignettes for advanced functionality, such as
pipelines and grids.

## List learner illustration

To show list learner functionality, we use `enfold` below to fit one
elastic net learner without a metalearner. When calling `risk_learners`,
we can see that `enfold` treats the elastic net as one learner per
lambda.

``` r
# Slight quirk: Need to pre-create lambda sequence
make_lambdas <- make_lambda_sequence(
  x = predictors,
  y = outcome,
  nlambda = 10L
)

# Now fit...
elastic_net_illustration <- initialize_enfold(x = predictors, y = outcome) |>
  add_learners(lrn_glmnet(
    "Elnet",
    family = "gaussian",
    lambda = make_lambdas
  )) |>
  add_cv_folds(inner_cv = NA, outer_cv = 5L) |>
  fit()

# ...and estimate risk across the lambda sequence
risk_learners(elastic_net_illustration, type = "cv", loss_fun = loss_gaussian())
#>    Elnet/0.357660725441877    Elnet/0.128536600209929 
#>                  0.4471337                  0.2817041 
#>   Elnet/0.0461936590133436   Elnet/0.0166011402943286 
#>                  0.2188772                  0.2101855 
#>  Elnet/0.00596614048244957  Elnet/0.00214411971860052 
#>                  0.2090618                  0.2089009 
#> Elnet/0.000770556674153949 Elnet/0.000276923710431031 
#>                  0.2088797                  0.2088768 
#> Elnet/9.95212214378516e-05 Elnet/3.57660725441877e-05 
#>                  0.2088765                  0.2088762
```
