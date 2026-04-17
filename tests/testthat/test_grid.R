## tests/test_grid.R
## Verification tests for make_score, make_grid_factory, and grd_* constructors.
## Source after: devtools::load_all()

future::plan("sequential")

set.seed(1)
n <- 100
x <- data.frame(a = rnorm(n), b = rnorm(n))
y <- x$a + rnorm(n, sd = 0.5)

folds <- new_fold_list(list(new_fold(validation_set = 51:100, n = n)))

## ============================================================================
## 1.  make_range
## ============================================================================

test_that("make_range returns an enfold_range", {
  r <- make_range(0, 1)
  expect_s3_class(r, "enfold_range")
  expect_equal(r$min, 0)
  expect_equal(r$max, 1)
})

test_that("make_range rejects min >= max", {
  expect_error(make_range(5, 1), "strictly less")
  expect_error(make_range(1, 1), "strictly less")
})

test_that("make_range rejects non-scalar inputs", {
  expect_error(make_range(c(0, 1), 2), "single numeric")
})

## ============================================================================
## 2.  make_score
## ============================================================================

test_that("make_score returns an enfold_score with correct fields", {
  s <- make_score(loss_gaussian())
  expect_s3_class(s, "enfold_score")
  expect_false(s$higher_is_better)
  expect_null(s$metalearner)
  expect_s3_class(s$loss_function, "mtl_loss")
})

test_that("make_score stores higher_is_better = TRUE", {
  s <- make_score(loss_gaussian(), higher_is_better = TRUE)
  expect_true(s$higher_is_better)
})

test_that("make_score stores a metalearner", {
  s <- make_score(loss_gaussian(), metalearner = mtl_selector("sel"))
  expect_false(is.null(s$metalearner))
  expect_equal(s$metalearner$name, "sel")
})

test_that("make_score rejects a raw function as loss_function", {
  expect_error(
    make_score(function(y, yh) (y - yh)^2),
    "mtl_loss"
  )
})

test_that("make_score rejects a non-logical higher_is_better", {
  expect_error(make_score(loss_gaussian(), higher_is_better = "yes"), "TRUE or FALSE")
  expect_error(make_score(loss_gaussian(), higher_is_better = NA), "TRUE or FALSE")
})

test_that("make_score rejects an invalid metalearner class", {
  expect_error(
    make_score(loss_gaussian(), metalearner = function(x, y) x),
    "enfold_learner"
  )
})

## ============================================================================
## 3.  make_grid (internal constructor)
## ============================================================================

dummy_search <- function(
  hyperparams,
  name_prefix,
  learner_object,
  directory,
  x,
  y,
  folds
) {
  list()
}

test_that("make_grid returns an enfold_grid with correct fields", {
  params <- specify_hyperparameters(num.trees = c(100L, 200L), mtry = c(1L, 2L))
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = dummy_search)
  expect_s3_class(g, "enfold_grid")
  expect_equal(g$name, "RF")
  expect_equal(names(g$hyperparams), c("num.trees", "mtry"))
  expect_true(is.function(g$search_engine))
})

test_that("make_grid rejects non-enfold_hyperparameters input", {
  expect_error(
    make_grid(
      "RF",
      lrn_ranger("RF"),
      list(num.trees = c(100L, 200L)),
      search_engine = dummy_search
    ),
    "enfold_hyperparameters"
  )
})

test_that("make_grid accepts enfold_range hyperparams", {
  params <- specify_hyperparameters(
    num.trees = c(100L, 200L),
    min.node.size = make_range(1, 20)
  )
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = dummy_search)
  expect_s3_class(g$hyperparams$min.node.size, "enfold_range")
})

## ============================================================================
## 4.  fit.enfold_grid always errors
## ============================================================================

test_that("fit.enfold_grid stops with informative message", {
  params <- specify_hyperparameters(num.trees = c(100L, 200L))
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = dummy_search)
  expect_error(fit(g, x, y), "add_learners")
})

## ============================================================================
## 5.  make_grid_factory — validation
## ============================================================================

test_that("make_grid_factory rejects search with extra formals", {
  expect_error(
    make_grid_factory(
      search = function(
        hyperparams,
        name_prefix,
        learner_object,
        directory,
        x,
        y,
        folds,
        extra
      ) {
        NULL
      }
    ),
    "hyperparams, name_prefix, learner_object, directory, x, y, folds"
  )
})

test_that("make_grid_factory rejects reserved config param names", {
  valid_search <- function(
    hyperparams,
    name_prefix,
    learner_object,
    directory,
    x,
    y,
    folds
  ) {
    list()
  }
  expect_error(
    make_grid_factory(
      search = valid_search,
      name_prefix = "x"
    ),
    "name_prefix"
  )
  expect_error(
    make_grid_factory(
      search = valid_search,
      search = 1
    ),
    "search"
  )
})

test_that("make_grid_factory returns a constructor that produces enfold_grid", {
  grd_exhaustive <- make_grid_factory(
    search = function(
      hyperparams,
      name_prefix,
      learner_object,
      directory,
      x,
      y,
      folds
    ) {
      combo_df <- draw(hyperparams, n = NULL)
      combo_list <- lapply(seq_len(nrow(combo_df)), function(i) {
        as.list(combo_df[i, , drop = FALSE])
      })
      results <- list()
      for (combo in combo_list) {
        nm <- make_combo_name(name_prefix, combo)
        modified <- tryCatch(
          change_arguments(learner_object, directory, combo, name = nm),
          error = function(e) NULL
        )
        if (is.null(modified)) next
        contrib <- tryCatch(cv_fit(modified, folds, x, y), error = function(e) NULL)
        if (is.null(contrib) || !is.null(attr(contrib, "failed_learner"))) next
        results <- c(results, list(list(name = nm, combo = combo, contrib = contrib)))
      }
      results
    }
  )
  params <- specify_hyperparameters(num.trees = c(100L, 200L))
  g <- grd_exhaustive("RF", lrn_ranger("RF"), params)
  expect_s3_class(g, "enfold_grid")
  expect_equal(g$name, "RF")
  expect_true(is.function(g$search_engine))
})

## ============================================================================
## 6.  grd_random — grid construction and search behaviour
## ============================================================================

test_that("grd_random returns an enfold_grid with correct fields", {
  params <- specify_hyperparameters(num.trees = c(100L, 200L), mtry = c(1L, 2L))
  g <- grd_random("RF", lrn_ranger("RF"), params, n_candidates = 2L)
  expect_s3_class(g, "enfold_grid")
  expect_equal(g$name, "RF")
  expect_true(is.function(g$search_engine))
})

test_that("grd_random returns all combinations when n_candidates is NULL", {
  params <- specify_hyperparameters(alpha = c(0.1, 0.5), depth = c(1L, 2L))
  g <- grd_random("M", lrn_mean("M"), params)
  res <- cv_fit(g, folds, x, y)
  expect_length(res, 4L) # 2 x 2
})

test_that("grd_random limits to n_candidates", {
  params <- specify_hyperparameters(
    alpha = c(0.1, 0.5, 1.0),
    depth = c(1L, 2L, 3L)
  )
  g <- grd_random("M", lrn_mean("M"), params, n_candidates = 3L)
  res <- cv_fit(g, folds, x, y)
  expect_length(res, 3L)
})

test_that("grd_random samples from enfold_range values within bounds", {
  params <- specify_hyperparameters(
    num.trees = c(100L),
    min.node.size = make_range(1, 20)
  )
  g <- grd_random("RF", lrn_ranger("RF"), params, n_candidates = 5L, seed = 42L)
  res <- cv_fit(g, folds, x, y)
  expect_length(res, 5L)
  vals <- vapply(res, function(p) attr(p, "combo")$min.node.size, numeric(1L))
  expect_true(all(vals >= 1 & vals <= 20))
})

test_that("grd_random respects forbid constraints", {
  params <- specify_hyperparameters(
    alpha = c(0.1, 0.5, 1.0),
    depth = c(3L, 5L)
  ) |>
    forbid(alpha > 0.4 && depth == 3L)
  g <- grd_random("M", lrn_mean("M"), params)
  res <- cv_fit(g, folds, x, y)
  # Forbidden: alpha=0.5+depth=3 and alpha=1+depth=3 → 4 valid combos from 6
  expect_length(res, 4L)
})

## ============================================================================
## 7.  cv_fit.enfold_grid — standalone
## ============================================================================

test_that("cv_fit.enfold_grid returns a named list with indices attr", {
  params <- specify_hyperparameters(num.trees = c(100L, 200L))
  g <- grd_random("RF", lrn_ranger("RF"), params, n_candidates = 2L, seed = 1L)
  res <- cv_fit(g, folds, x, y)
  expect_true(is.list(res))
  expect_true(length(res) > 0L)
  expect_true(all(grepl("^RF/", names(res))))
  expect_false(is.null(attr(res[[1L]], "indices")))
})

test_that("cv_fit.enfold_grid name format is prefix/param=val,...", {
  params <- specify_hyperparameters(num.trees = c(100L))
  g <- grd_random("RF", lrn_ranger("RF"), params)
  res <- cv_fit(g, folds, x, y)
  expect_true(any(grepl("num.trees=100", names(res))))
})

## ============================================================================
## 8.  cv_fit.enfold_grid inside fit() / build_ensembles
## ============================================================================

test_that("enfold_grid works inside fit() with inner CV", {
  params <- specify_hyperparameters(num.trees = c(100L, 200L))
  g <- grd_random("RF", lrn_ranger("RF"), params, n_candidates = 2L, seed = 1L)
  glmn <- lrn_glmnet("GLMNET", gaussian(), alpha = 1, lambda = make_lambda_sequence(x, y, nlambda = 20L))
  task <- initialize_enfold(x, y) |>
    add_learners(g, lrn_mean("Mean"), glmn) |>
    add_metalearners(mtl_selector("Sel"), mtl_superlearner("SL")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)
  expect_s3_class(fitted, "enfold_task_fitted")
})

## ============================================================================
## 9.  enfold_grid wrapping a pipeline (directory-based targeting)
## ============================================================================

test_that("cv_fit.enfold_grid with pipeline+directory creates branched paths", {
  scr <- scr_correlation("Screen", cutoff = 0.05)
  rf_lrn <- lrn_ranger("RF")
  pl <- make_pipeline(scr, rf_lrn)
  params <- specify_hyperparameters(num.trees = c(100L, 200L))
  g <- grd_random("RF", pl, params, directory = "RF", seed = 1L)
  fold2 <- new_fold_list(list(new_fold(validation_set = 51:100, n = n)))
  res <- cv_fit(g, fold2, x, y)

  expect_true(is.list(res))
  expect_true(length(res) >= 1L)
  expect_true(all(grepl("^Screen/RF/", names(res))))
})

test_that("cv_fit.enfold_pipeline still works for grid-free pipelines", {
  scr <- scr_correlation("Screen", cutoff = 0.05)
  glm <- lrn_glm("GLM", family = gaussian())
  pl <- make_pipeline(scr, glm)
  fold2 <- new_fold_list(list(new_fold(validation_set = 51:100, n = n)))
  res <- cv_fit(pl, fold2, x, y)
  expect_true(is.list(res))
  expect_equal(names(res), "Screen/GLM")
})

## ============================================================================
## 10.  grd_bayes — standalone cv_fit.enfold_grid
## ============================================================================

test_that("grd_bayes evaluates n_init + n_iter candidates and returns results", {
  skip_if_not_installed("rBayesianOptimization")

  params <- specify_hyperparameters(
    num.trees = make_range(50, 200),
    mtry = make_range(1, 2)
  )
  g <- grd_bayes(
    "RF_bayes",
    lrn_ranger("RF_bayes"),
    params,
    n_init = 2L,
    n_iter = 3L,
    score = make_score(loss_gaussian()),
    seed = 1L
  )

  res <- cv_fit(g, folds, x, y)

  expect_true(is.list(res))
  expect_true(length(res) > 0L)
  # n_init + n_iter = 5 candidates total (minus any failed)
  expect_lte(length(res), 5L)
  expect_true(all(grepl("^RF_bayes/", names(res))))
  expect_false(is.null(attr(res[[1L]], "indices")))
})

test_that("grd_bayes rejects non-range hyperparameters", {
  skip_if_not_installed("rBayesianOptimization")

  params <- specify_hyperparameters(
    num.trees = c(100L, 200L), # discrete — not allowed
    mtry = make_range(1, 2)
  )
  g <- grd_bayes(
    "RF_bayes",
    lrn_ranger("RF_bayes"),
    params,
    n_init = 2L,
    n_iter = 2L,
    score = make_score(loss_gaussian())
  )

  expect_error(cv_fit(g, folds, x, y), "enfold_range")
})

test_that("grd_bayes surviving entries carry combo attribute", {
  skip_if_not_installed("rBayesianOptimization")

  params <- specify_hyperparameters(num.trees = make_range(50, 200))
  g <- grd_bayes(
    "RF_bayes",
    lrn_ranger("RF_bayes"),
    params,
    n_init = 2L,
    n_iter = 2L,
    score = make_score(loss_gaussian()),
    seed = 42L
  )
  res <- cv_fit(g, folds, x, y)

  combos <- lapply(res, function(p) attr(p, "combo"))
  expect_true(all(vapply(combos, is.list, logical(1L))))
  expect_true(all(vapply(
    combos,
    function(co) "num.trees" %in% names(co),
    logical(1L)
  )))
})

test_that("grd_bayes errors with informative message when score is a raw mtl_loss", {
  skip_if_not_installed("rBayesianOptimization")

  params <- specify_hyperparameters(num.trees = make_range(50, 200))
  g <- grd_bayes(
    "RF_bayes",
    lrn_ranger("RF_bayes"),
    params,
    n_init = 2L,
    n_iter = 2L,
    score = loss_gaussian()
  )
  expect_error(cv_fit(g, folds, x, y), "make_score")
})

## ============================================================================
## 11.  grd_bayes — grid wrapping a pipeline
## ============================================================================

test_that("grd_bayes works with pipeline+directory", {
  skip_if_not_installed("rBayesianOptimization")

  fold_pl <- new_fold_list(list(new_fold(validation_set = 51:100, n = n)))

  params <- specify_hyperparameters(
    num.trees = make_range(50, 200),
    mtry = make_range(1, 2)
  )
  scr <- scr_correlation("Screen", cutoff = 0.05)
  rf_lrn <- lrn_ranger("RF_bayes")
  pl <- make_pipeline(scr, rf_lrn)
  g <- grd_bayes(
    "RF_bayes",
    pl,
    params,
    n_init = 2L,
    n_iter = 3L,
    score = make_score(loss_gaussian()),
    seed = 1L,
    directory = "RF_bayes"
  )

  res <- cv_fit(g, fold_pl, x, y)

  expect_true(is.list(res))
  expect_true(length(res) > 0L)
  expect_true(all(grepl("^Screen/RF_bayes/", names(res))))
})

## ============================================================================
## 12.  grd_early_stop
## ============================================================================

test_that("grd_early_stop stops early when xgboost search stops improving", {
  skip_if_not_installed("xgboost")

  set.seed(1)
  n2 <- 300
  p <- 8
  x2 <- as.data.frame(matrix(rnorm(n2 * p), n2, p))
  y2 <- 1.5 * x2[[1]] - 0.8 * x2[[2]] + rnorm(n2, sd = 0.3)

  folds2 <- new_fold_list(
    list(new_fold(validation_set = 201:300, n = n2))
  )

  params <- specify_hyperparameters(
    nrounds = c(3L, 5L, 8L, 12L),
    max_depth = c(1L, 2L),
    eta = c(0.05, 0.1),
    min_child_weight = make_range(1, 20)
  )

  g <- grd_early_stop(
    "XGB_early",
    lrn_xgboost("XGB_early"),
    params,
    seed = 1L,
    max_candidates = 100L,
    n_early_stop = 3L,
    tol = 0.05
  )

  res <- cv_fit(g, folds2, x2, y2)

  expect_true(length(res) < 100L)
  expect_true(length(res) >= 1L)
  expect_true(all(grepl("^XGB_early/", names(res))))
})

test_that("grd_early_stop accepts an explicit make_score object", {
  params <- specify_hyperparameters(num.trees = c(100L, 200L, 300L))
  g <- grd_early_stop(
    "RF",
    lrn_ranger("RF"),
    params,
    score = make_score(loss_gaussian()),
    n_early_stop = 2L
  )
  res <- cv_fit(g, folds, x, y)
  expect_true(is.list(res))
  expect_true(length(res) >= 1L)
})

test_that("grd_early_stop errors with informative message when score is a raw mtl_loss", {
  params <- specify_hyperparameters(num.trees = c(100L, 200L))
  g <- grd_early_stop(
    "RF",
    lrn_ranger("RF"),
    params,
    score = loss_gaussian(),
    n_early_stop = 2L
  )
  expect_error(cv_fit(g, folds, x, y), "make_score")
})

test_that("grd_early_stop with multi-output pipeline errors without metalearner", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  glm <- lrn_glm("GLM", family = gaussian())
  rf  <- lrn_ranger("RF")
  pl  <- make_pipeline(scr, list(glm, rf))
  params <- specify_hyperparameters(num.trees = c(50L, 100L))

  # Without metalearner: evaluate_score should error on multi-output contrib
  g_no_mtl <- grd_early_stop(
    "RF", pl, params,
    directory = "RF",
    score = make_score(loss_gaussian()),
    n_early_stop = 5L
  )
  expect_error(cv_fit(g_no_mtl, folds, x, y), "metalearner")
})

test_that("grd_early_stop with multi-output pipeline succeeds with metalearner", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  glm <- lrn_glm("GLM", family = gaussian())
  rf  <- lrn_ranger("RF")
  pl  <- make_pipeline(scr, list(glm, rf))
  params <- specify_hyperparameters(num.trees = c(50L, 100L))

  g_with_mtl <- grd_early_stop(
    "RF", pl, params,
    directory = "RF",
    score = make_score(loss_gaussian(), metalearner = mtl_selector("sel")),
    n_early_stop = 5L
  )
  res <- cv_fit(g_with_mtl, folds, x, y)
  expect_true(length(res) >= 1L)
})


message("\n\u2713  All grid tests passed.\n")
