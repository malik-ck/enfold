## tests/test_complex_tasks.R
##
## Integration tests covering more complex task configurations:
##   - Multi-stage pipelines (screener → basis expander → learner)
##   - Branching pipelines (one screener feeding multiple learners)
##   - Grids nested inside pipelines
##   - Multiple metalearners
##   - Incremental fitting (adding learners / metalearners after first fit)
##
## Source after: devtools::load_all()

stopifnot(requireNamespace("future", quietly = TRUE))
future::plan("sequential")

## ---- Shared data: more covariates, larger n --------------------------------
set.seed(7)
n <- 200
x <- data.frame(
  a = rnorm(n),
  b = rnorm(n),
  c = rnorm(n),
  d = rnorm(n),
  e = rnorm(n),
  f = rnorm(n)
)
y <- 3 * x$a - 2 * x$c + 0.5 * x$e + rnorm(n, sd = 0.8)

## ============================================================================
## 1.  3-stage pipeline: screener → basis expander → GLM
## ============================================================================

test_that("3-stage pipeline (screener → splines → GLM) fits and predicts", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  bex <- bex_splines("Splines", type = "bs", degree = 3, max_knots = 5)
  glm_lr <- lrn_glm("GLM", family = gaussian())
  pipe3 <- make_pipeline(scr, bex, glm_lr)

  task <- initialize_enfold(x, y) |>
    add_learners(pipe3) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
})

test_that("3-stage pipeline path names contain all three node names", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  bex <- bex_splines("Splines", type = "bs", degree = 3, max_knots = 5)
  glm_lr <- lrn_glm("GLM", family = gaussian())
  pipe3 <- make_pipeline(scr, bex, glm_lr)

  folds <- new_fold_list(list(new_fold(validation_set = 101:200, n = n)))
  res <- cv_fit(pipe3, folds, x, y)

  expect_true(is.list(res))
  expect_true(all(grepl("Scr", names(res))))
  expect_true(all(grepl("Splines", names(res))))
  expect_true(all(grepl("GLM", names(res))))
})


## ============================================================================
## 2.  Branching pipeline: one screener feeding two different learners
## ============================================================================

test_that("branching pipeline (screener → [mean, GLM]) produces two cv_fit outputs", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  mean_l <- lrn_mean("Mean")
  glm <- lrn_glm("GLM", family = gaussian())
  pipe_b <- make_pipeline(scr, list(mean_l, glm))

  folds <- new_fold_list(list(new_fold(validation_set = 101:200, n = n)))
  res <- cv_fit(pipe_b, folds, x, y)

  expect_length(res, 2L)
  expect_true(any(grepl("Mean", names(res))))
  expect_true(any(grepl("GLM", names(res))))
  expect_true(all(grepl("^Scr/", names(res))))
})

test_that("branching pipeline fits and predicts inside a full task", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  mean_l <- lrn_mean("Mean")
  glm <- lrn_glm("GLM", family = gaussian())
  pipe_b <- make_pipeline(scr, list(mean_l, glm))

  task <- initialize_enfold(x, y) |>
    add_learners(pipe_b) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
})


## ============================================================================
## 3.  Grid wrapping a pipeline (directory-based targeting)
## ============================================================================

test_that("grid wrapping pipeline produces paths prefixed with screener name", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  rf_lrn <- lrn_ranger("RF")
  pl <- make_pipeline(scr, rf_lrn)
  params <- specify_hyperparameters(num.trees = c(50L, 100L))
  g <- grd_random(
    "RF",
    pl,
    params,
    directory = "RF",
    n_candidates = 2L,
    seed = 1L
  )

  folds <- new_fold_list(list(new_fold(validation_set = 101:200, n = n)))
  res <- cv_fit(g, folds, x, y)

  expect_true(is.list(res))
  expect_true(length(res) >= 1L)
  expect_true(all(grepl("^Scr/RF/", names(res))))
})

test_that("grid wrapping pipeline fits end-to-end inside a full task", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  rf_lrn <- lrn_ranger("RF")
  pl <- make_pipeline(scr, rf_lrn)
  params <- specify_hyperparameters(num.trees = c(50L, 100L))
  g <- grd_random(
    "RF",
    pl,
    params,
    directory = "RF",
    n_candidates = 2L,
    seed = 1L
  )

  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), g) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
})


## ============================================================================
## 4.  Multiple metalearners: selector and superlearner together
## ============================================================================

test_that("multiple metalearners can be fitted and predict returns a named list", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel"), mtl_superlearner("SL")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  preds <- predict(fitted, type = "ensemble", metalearner_name = c("Sel", "SL"))
  expect_true(is.list(preds))
  expect_setequal(names(preds), c("Sel", "SL"))
  expect_length(preds$Sel, n)
  expect_length(preds$SL, n)
})

test_that("multiple metalearners with outer CV each produce cv predictions", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel"), mtl_superlearner("SL")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  preds <- predict(fitted, type = "cv", metalearner_name = c("Sel", "SL"))
  expect_true(is.list(preds))
  expect_length(preds$Sel, n)
  expect_length(preds$SL, n)
  # Both should carry an indices attribute from cv predictions
  expect_false(is.null(attr(preds$Sel, "indices")))
  expect_false(is.null(attr(preds$SL, "indices")))
})


## ============================================================================
## 5.  Incremental fitting — add a learner, triggering full re-fit
## ============================================================================

test_that("adding a learner after fit triggers a full re-fit", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean")) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted1 <- fit(task)
  expect_length(fitted1$fit_objects[[1L]], 1L)

  fitted2 <- fitted1 |>
    add_learners(lrn_glm("GLM", family = gaussian())) |>
    fit()

  expect_s3_class(fitted2, "enfold_task_fitted")
  expect_length(fitted2$fit_objects[[1L]], 2L)
  expect_true("GLM" %in% fitted2$fitted_learner_names)
})

test_that("adding a metalearner after fit does not alter fit_objects", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted1 <- fit(task)

  fit_objects_before <- fitted1$fit_objects

  fitted2 <- fitted1 |>
    add_metalearners(mtl_superlearner("SL")) |>
    fit()

  # Learner fit_objects must not change when only metalearners are added
  expect_identical(fitted2$fit_objects, fit_objects_before)
  expect_true("SL" %in% fitted2$fitted_metalearner_names)
  expect_length(fitted2$ensembles[[1L]], 2L)
})


## ============================================================================
## 6.  Screener → interaction basis expander → GLM pipeline
## ============================================================================

test_that("screener → interaction expander → GLM pipeline fits and predicts", {
  scr <- scr_correlation("Scr", cutoff = 0.05, min_vars = 3L)
  bex <- bex_interactions("Ints", depth = 2L)
  glm <- lrn_glm("GLM", family = gaussian())
  pipe <- make_pipeline(scr, bex, glm)

  task <- initialize_enfold(x, y) |>
    add_learners(pipe) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
})

## ============================================================================
## 7. enfold_list at the end of a pipeline
## ============================================================================

test_that("enfold_list works at the end of pipelines", {
  test_fit <- initialize_enfold(x, y) %>%
    add_learners(
      make_pipeline(
        bex_splines("SPL", max_knots = 10),
        bex_interactions("Ints"),
        lrn_glmnet("GLMN", family = gaussian(), lambda = make_lambda_sequence(x, y, nlambda = 20L))
      )
    ) %>%
    add_metalearners(mtl_superlearner("SL")) %>%
    add_cv_folds(10, NA) %>%
    fit()

  expect_s3_class(test_fit, "enfold_task_fitted")
  preds <- predict(test_fit, type = "ensemble")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
})


## ============================================================================
## 8. enfold_list screener in a non-terminal pipeline stage
## ============================================================================

test_that("enfold_list screener expands into sub-paths at non-terminal stage", {
  # Screener that returns two filtered datasets: top-2 and top-4 predictors
  # by absolute correlation with y. Using fixed counts avoids empty-column
  # failures in small folds.
  list_scr <- make_learner_factory(
    fit = function(x, y) {
      cors <- abs(cor(as.matrix(x), y))
      ord <- order(cors, decreasing = TRUE)
      list(
        top2 = ord[seq_len(min(2L, length(ord)))],
        top4 = ord[seq_len(min(4L, length(ord)))]
      )
    },
    preds = function(object, data) {
      list(
        top2 = data[, object$top2, drop = FALSE],
        top4 = data[, object$top4, drop = FALSE]
      )
    },
    expect_list = TRUE
  )(name = "ListScr")

  glm <- lrn_glm("GLM", family = gaussian())
  pipe <- make_pipeline(list_scr, glm)

  # cv_fit should produce exactly two paths, one per screener output
  folds <- new_fold_list(list(new_fold(validation_set = 101:200, n = n)))
  res <- cv_fit(pipe, folds, x, y)
  expect_length(res, 2L)
  expect_true(all(grepl("^ListScr/", names(res))))
  expect_true(any(grepl("top2", names(res))))
  expect_true(any(grepl("top4", names(res))))

  # Full end-to-end: both sub-paths fed into the metalearner
  task <- initialize_enfold(x, y) |>
    add_learners(pipe) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
})


## ============================================================================
## 9. Risk-set expansion and collapsing pipeline
## ============================================================================

test_that("risk-set expand/collapse wrapper works with CV", {
  set.seed(123)
  n <- 100
  x <- data.frame(
    age = runif(n, 20, 70),
    sex = factor(sample(c("m", "f"), n, TRUE)),
    time = sample(1:5, n, TRUE)
  )
  y <- rbinom(n, 1, 0.25)

  # Plain learners wrapped in risk-set
  risk_set_glm <- wrp_risk_set(
    "RS GLM",
    lrn_glm("GLM", family = binomial()),
    time = "time",
    collapse_back = TRUE
  )
  risk_set_glmnet <- wrp_risk_set(
    "RS GLMNET",
    lrn_glmnet("GLMNET", binomial(), lambda = make_lambda_sequence(x[, c("age", "sex")], y, nlambda = 20L)),
    time = "time",
    collapse_back = TRUE
  )

  # Risk-set learner inside a pipeline, and a pipeline inside a risk-set learner
  pipe_risk_set <- make_pipeline(
    scr_lasso("LSS", lambda = 1e-5),
    risk_set_glmnet
  )
  risk_set_pipe <- wrp_risk_set(
    "RS Pipe",
    make_pipeline(
      scr_lasso("LSS2", lambda = 1e-5),
      lrn_glm("GLM2", family = binomial())
    ),
    time = "time",
    collapse_back = TRUE
  )

  task <- initialize_enfold(x, y) |>
    add_learners(risk_set_glm, risk_set_glmnet, pipe_risk_set, risk_set_pipe) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)

  fitted <- fit(task)
  preds <- predict(fitted, type = "ensemble")

  expect_length(preds, n)
  expect_true(is.numeric(preds))
  expect_false(any(is.na(preds)))

  # And check none failed
  expect_true(all(
    vapply(
      fitted$ensembles,
      function(en) is.null(attr(en, "failed_learners")),
      logical(1)
    )
  ))
})

## ============================================================================
## 10.  Stratification wrapper
## ============================================================================

test_that("stratify wrapper fits and predicts for plain learners", {
  set.seed(321)
  n <- 100
  x <- data.frame(
    a = rnorm(n),
    b = rnorm(n),
    sex = factor(sample(c("m", "f"), n, TRUE))
  )
  y <- rbinom(n, 1, 0.3)

  strat_glm <- wrp_stratify(
    "Strat GLM",
    lrn_glm("GLM", family = binomial()),
    strata = "sex"
  )

  task <- initialize_enfold(x, y) |>
    add_learners(strat_glm) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)

  fitted <- fit(task)
  preds <- predict(fitted, type = "ensemble")

  expect_s3_class(fitted, "enfold_task_fitted")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
  expect_false(any(is.na(preds)))
})

test_that("stratify wrapper works with pipelines", {
  set.seed(322)
  n <- 100
  x <- data.frame(
    a = rnorm(n),
    b = rnorm(n),
    sex = factor(sample(c("m", "f"), n, TRUE))
  )
  y <- rbinom(n, 1, 0.3)

  strat_pipe <- wrp_stratify(
    "Strat Pipe",
    make_pipeline(
      scr_lasso("LSS", lambda = 1e-9),
      lrn_glm("GLM", family = binomial())
    ),
    strata = "sex"
  )

  task <- initialize_enfold(x, y) |>
    add_learners(strat_pipe) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)

  fitted <- fit(task)
  preds <- predict(fitted, type = "ensemble")

  expect_s3_class(fitted, "enfold_task_fitted")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
  expect_false(any(is.na(preds)))
})

test_that("stratify wrapper composes with risk-set wrapper", {
  set.seed(323)
  n <- 100
  x <- data.frame(
    age = runif(n, 20, 70),
    sex = factor(sample(c("m", "f"), n, TRUE)),
    time = sample(1:5, n, TRUE)
  )
  y <- rbinom(n, 1, 0.3)

  strat_then_risk <- wrp_risk_set(
    "RS Strat",
    wrp_stratify(
      "Strat GLM",
      lrn_glm("GLM", family = binomial()),
      strata = "sex"
    ),
    time = "time",
    collapse_back = TRUE
  )

  task <- initialize_enfold(x, y) |>
    add_learners(strat_then_risk) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)

  fitted <- fit(task)
  preds <- predict(fitted, type = "ensemble")

  expect_s3_class(fitted, "enfold_task_fitted")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
  expect_false(any(is.na(preds)))
})


message("\n\u2713  All complex task tests passed.\n")
