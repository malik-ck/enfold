## tests/test_grid.R
## Verification tests for make_score, make_grid_factory, and grd_* constructors.
## Source after: devtools::load_all()

future::plan("sequential")

set.seed(1)
n <- 100
x <- data.frame(a = rnorm(n), b = rnorm(n), d = rnorm(n), e = rnorm(n))
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
## 1b.  make_discrete
## ============================================================================

test_that("make_discrete returns enfold_discrete with scalar candidates", {
  d <- make_discrete(0, 0.5, 1)
  expect_s3_class(d, "enfold_discrete")
  expect_equal(length(d$values), 3L)
  expect_equal(d$values[[1L]], 0)
  expect_true(d$is_separate)
})

test_that("make_discrete with is_separate = TRUE expands vectors", {
  d <- make_discrete(c(1L, 2L))
  expect_equal(length(d$values), 2L)
})

test_that("make_discrete with is_separate = FALSE keeps vectors whole", {
  d <- make_discrete(c(1L, 2L), 3L, is_separate = FALSE)
  expect_equal(length(d$values), 2L)
  expect_equal(d$values[[1L]], c(1L, 2L))
})

test_that("make_discrete errors with no arguments", {
  expect_error(make_discrete(), "at least one")
})

test_that("print.enfold_discrete does not error", {
  d <- make_discrete(1, 2, 3)
  expect_output(print(d), "enfold_discrete")
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
  expect_error(
    make_score(loss_gaussian(), higher_is_better = "yes"),
    "TRUE or FALSE"
  )
  expect_error(
    make_score(loss_gaussian(), higher_is_better = NA),
    "TRUE or FALSE"
  )
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

dummy_search <- function(search_space, learner, x, y, folds) {
  list()
}

test_that("make_grid returns an enfold_grid with correct fields", {
  params <- specify_hyperparameters(
    num.trees = make_discrete(100L, 200L),
    mtry = make_discrete(1L, 2L)
  )
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = dummy_search)
  expect_s3_class(g, "enfold_grid")
  expect_equal(g$name, "RF")
  expect_equal(names(g$hyperparams), c("num.trees", "mtry"))
  expect_true(is.function(g$search_engine))
})

test_that("make_grid allows search_engine = NULL (bare grid)", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = NULL)
  expect_s3_class(g, "enfold_grid")
  expect_null(g$search_engine)
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
    num.trees = make_discrete(100L, 200L),
    min.node.size = make_range(1, 20)
  )
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = dummy_search)
  expect_s3_class(g$hyperparams$min.node.size, "enfold_range")
})

## ============================================================================
## 4.  fit.enfold_grid — bare vs engine
## ============================================================================

test_that("fit.enfold_grid with engine errors with informative message", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = dummy_search)
  expect_error(fit(g, x, y), "add_learners")
})

test_that("fit.enfold_grid bare grid fits exhaustively and returns enfold_grid_fitted", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = NULL)
  fitted_g <- fit(g, x, y)
  expect_s3_class(fitted_g, "enfold_grid_fitted")
  expect_equal(length(fitted_g$models), 2L)
  expect_true(all(grepl("^RF/", names(fitted_g$models))))
})

test_that("predict.enfold_grid_fitted returns named list of predictions", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L))
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = NULL)
  fitted_g <- fit(g, x, y)
  preds <- predict(fitted_g, newdata = x)
  expect_true(is.list(preds))
  expect_equal(length(preds), 1L)
  expect_equal(length(preds[[1L]]), nrow(x))
})

test_that("fit.enfold_grid bare grid errors for continuous ranges", {
  params <- specify_hyperparameters(num.trees = make_range(100, 500))
  g <- make_grid("RF", lrn_ranger("RF"), params, search_engine = NULL)
  expect_error(fit(g, x, y), "ranges without a search engine")
})

## ============================================================================
## 5.  lrn_*(name, parameters = ...) — bare grid via constructor
## ============================================================================

test_that("lrn_ranger with parameters returns enfold_grid (bare)", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- lrn_ranger("RF", parameters = params)
  expect_s3_class(g, "enfold_grid")
  expect_null(g$search_engine)
  expect_equal(g$name, "RF")
})

test_that("lrn_ranger bare grid fit/predict round-trip", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- lrn_ranger("RF", parameters = params)
  fitted_g <- fit(g, x, y)
  preds <- predict(fitted_g, newdata = x)
  expect_equal(length(preds), 2L)
})

test_that("lrn_glm with parameters returns bare enfold_grid", {
  params <- specify_hyperparameters(
    family = make_discrete(gaussian(), binomial(), is_separate = FALSE)
  )
  g <- lrn_glm("GLM", family = gaussian(), parameters = params)
  expect_s3_class(g, "enfold_grid")
})

test_that("parameters with unknown names errors at constructor time", {
  params <- specify_hyperparameters(not_a_param = make_discrete(1, 2))
  expect_error(
    lrn_ranger("RF", parameters = params),
    "not recognised"
  )
})

## ============================================================================
## 5b.  grd_* accepts bare enfold_grid as first arg
## ============================================================================

test_that("grd_random accepts bare enfold_grid as first arg", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L, 300L))
  bare_g <- lrn_ranger("RF", parameters = params)
  g <- grd_random(bare_g, n_candidates = 2L)
  expect_s3_class(g, "enfold_grid")
  expect_true(is.function(g$search_engine))
  expect_equal(g$name, "RF")
})

test_that("grd_random rejects enfold_grid that already has a search engine", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  bare_g <- lrn_ranger("RF", parameters = params)
  g_with_engine <- grd_random(bare_g, n_candidates = 2L)
  expect_error(grd_random(g_with_engine), "already has one")
})

## ============================================================================
## 5c.  make_pipeline with embedded bare grid node
## ============================================================================

test_that("make_pipeline accepts bare enfold_grid node", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- lrn_ranger("RF", parameters = params)
  scr <- scr_correlation("Screen", cutoff = 0.05)
  pl <- make_pipeline(scr, g)
  expect_s3_class(pl, "enfold_pipeline")
})

test_that("make_pipeline rejects enfold_grid with search engine", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- grd_random(lrn_ranger("RF", parameters = params), n_candidates = 2L)
  scr <- scr_correlation("Screen", cutoff = 0.05)
  expect_error(make_pipeline(scr, g), "search engine")
})

test_that("fit.enfold_pipeline expands bare grid node exhaustively", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- lrn_ranger("RF", parameters = params)
  scr <- scr_correlation("Screen", cutoff = 0.05)
  pl <- make_pipeline(scr, g)
  fitted_pl <- fit(pl, x, y)
  preds <- predict(fitted_pl, newdata = x)
  expect_equal(length(preds), 2L)
  expect_true(all(grepl("^Screen/RF/", names(preds))))
})

test_that("grd_random accepts pipeline with single embedded bare grid", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- lrn_ranger("RF", parameters = params)
  scr <- scr_correlation("Screen", cutoff = 0.05)
  pl <- make_pipeline(scr, g)
  grid <- grd_random(pl, n_candidates = 2L)
  expect_s3_class(grid, "enfold_grid")
  expect_equal(grid$name, "RF")
})

test_that("grd_random accepts pipeline with multiple embedded bare grids (multi-node)", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g1  <- lrn_ranger("RF1", parameters = params)
  g2  <- lrn_ranger("RF2", parameters = params)
  scr <- scr_correlation("Screen", cutoff = 0.05)
  pl  <- make_pipeline(scr, list(g1, g2))
  grid <- grd_random(pl, n_candidates = 3L, seed = 1L)
  expect_s3_class(grid, "enfold_grid")
  expect_equal(grid$name, "RF1+RF2")

  res <- cv_fit(grid, folds, x, y)
  expect_true(is.list(res))
  expect_true(length(res) >= 1L)
})

## ============================================================================
## 6.  make_grid_factory — validation
## ============================================================================

test_that("make_grid_factory rejects search with extra formals", {
  expect_error(
    make_grid_factory(
      search = function(search_space, learner, x, y, folds, extra) NULL
    ),
    "search_space, learner, x, y, folds"
  )
})

test_that("make_grid_factory rejects reserved config param names", {
  valid_search <- function(search_space, learner, x, y, folds) list()
  expect_error(
    make_grid_factory(search = valid_search, grid = "x"),
    "grid"
  )
})

test_that("make_grid_factory returns a constructor that produces enfold_grid", {
  grd_exhaustive <- make_grid_factory(
    search = function(search_space, learner, x, y, folds) {
      combo_df <- draw(search_space, n = NULL)
      results <- list()
      for (i in seq_len(nrow(combo_df))) {
        combo <- combo_row(combo_df, i)
        modified <- tryCatch(apply_combo(learner, combo), error = function(e) {
          NULL
        })
        if (is.null(modified)) {
          next
        }
        contrib <- tryCatch(cv_fit(modified, folds, x, y), error = function(e) {
          NULL
        })
        if (is.null(contrib) || !is.null(attr(contrib, "failed_learner"))) {
          next
        }
        results <- c(results, list(list(combo = combo, contrib = contrib)))
      }
      results
    }
  )
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- grd_exhaustive(lrn_ranger("RF", parameters = params))
  expect_s3_class(g, "enfold_grid")
  expect_equal(g$name, "RF")
  expect_true(is.function(g$search_engine))
})

## ============================================================================
## 7.  grd_random — grid construction and search behaviour
## ============================================================================

test_that("grd_random returns an enfold_grid with correct fields", {
  params <- specify_hyperparameters(
    num.trees = make_discrete(100L, 200L),
    mtry = make_discrete(1L, 2L)
  )
  g <- grd_random(lrn_ranger("RF", parameters = params), n_candidates = 2L)
  expect_s3_class(g, "enfold_grid")
  expect_equal(g$name, "RF")
  expect_true(is.function(g$search_engine))
})

test_that("grd_random returns all combinations when n_candidates is NULL", {
  params <- specify_hyperparameters(
    num.trees = make_discrete(100L, 200L),
    mtry = make_discrete(1L, 2L)
  )
  g <- grd_random(lrn_ranger("RF", parameters = params))
  res <- cv_fit(g, folds, x, y)
  expect_length(res, 4L) # 2 x 2
})

test_that("grd_random limits to n_candidates", {
  params <- specify_hyperparameters(
    num.trees = make_discrete(100L, 200L, 300L),
    mtry = make_discrete(1L, 2L, 3L)
  )
  g <- grd_random(lrn_ranger("RF", parameters = params), n_candidates = 3L)
  res <- cv_fit(g, folds, x, y)
  expect_length(res, 3L)
})

test_that("grd_random samples from enfold_range values within bounds", {
  params <- specify_hyperparameters(
    num.trees = make_discrete(100L),
    min.node.size = make_range(1, 20)
  )
  g <- grd_random(
    lrn_ranger("RF", parameters = params),
    n_candidates = 5L,
    seed = 42L
  )
  res <- cv_fit(g, folds, x, y)
  expect_length(res, 5L)
  vals <- vapply(res, function(p) attr(p, "combo")$min.node.size, numeric(1L))
  expect_true(all(vals >= 1 & vals <= 20))
})

test_that("grd_random respects forbid constraints", {
  params <- specify_hyperparameters(
    num.trees = make_discrete(100L, 200L, 300L),
    min.node.size = make_discrete(3L, 5L)
  ) |>
    forbid(num.trees > 150 && min.node.size == 3L)
  g <- grd_random(lrn_ranger("RF", parameters = params))
  res <- cv_fit(g, folds, x, y)
  # Forbidden: 200+3 and 300+3 → 4 valid combos from 6
  expect_length(res, 4L)
})

## ============================================================================
## 8.  cv_fit.enfold_grid — standalone
## ============================================================================

test_that("cv_fit.enfold_grid returns a named list with indices attr", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- grd_random(
    lrn_ranger("RF", parameters = params),
    n_candidates = 2L,
    seed = 1L
  )
  res <- cv_fit(g, folds, x, y)
  expect_true(is.list(res))
  expect_true(length(res) > 0L)
  expect_true(all(grepl("^RF/", names(res))))
  expect_false(is.null(attr(res[[1L]], "indices")))
})

test_that("cv_fit.enfold_grid name format is prefix/param=val,...", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L))
  g <- grd_random(lrn_ranger("RF", parameters = params))
  res <- cv_fit(g, folds, x, y)
  expect_true(any(grepl("num.trees=100", names(res))))
})

test_that("cv_fit.enfold_grid bare grid works without search engine", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- lrn_ranger("RF", parameters = params)
  res <- cv_fit(g, folds, x, y)
  expect_true(is.list(res))
  expect_length(res, 2L)
  expect_true(all(grepl("^RF/", names(res))))
})

## ============================================================================
## 9.  cv_fit.enfold_grid inside fit() / build_ensembles
## ============================================================================

test_that("enfold_grid works inside fit() with inner CV", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- grd_random(
    lrn_ranger("RF", parameters = params),
    n_candidates = 2L,
    seed = 1L
  )
  glmn <- lrn_glmnet(
    "GLMNET",
    gaussian(),
    alpha = 1,
    lambda = make_lambda_sequence(x, y, nlambda = 20L)
  )
  task <- initialize_enfold(x, y) |>
    add_learners(g, lrn_mean("Mean"), glmn) |>
    add_metalearners(mtl_selector("Sel"), mtl_superlearner("SL")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)
  expect_s3_class(fitted, "enfold_task_fitted")
})

## ============================================================================
## 10.  enfold_grid wrapping a pipeline (directory-based targeting)
## ============================================================================

test_that("cv_fit.enfold_grid with pipeline containing embedded bare grid", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  scr <- scr_correlation("Screen", cutoff = 0.05)
  rf_bare <- lrn_ranger("RF", parameters = params)
  pl <- make_pipeline(scr, rf_bare)
  g <- grd_random(pl, seed = 1L)
  res <- cv_fit(g, folds, x, y)

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
## 11.  grd_bayes — standalone cv_fit.enfold_grid
## ============================================================================

test_that("grd_bayes evaluates n_init + n_iter candidates and returns results", {
  skip_if_not_installed("rBayesianOptimization")

  params <- specify_hyperparameters(
    num.trees = make_range(50, 200),
    mtry = make_range(1, 2)
  )
  g <- grd_bayes(
    lrn_ranger("RF_bayes", parameters = params),
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
    num.trees = make_discrete(100L, 200L), # discrete — not allowed
    mtry = make_range(1, 2)
  )
  g <- grd_bayes(
    lrn_ranger("RF_bayes", parameters = params),
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
    lrn_ranger("RF_bayes", parameters = params),
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
    lrn_ranger("RF_bayes", parameters = params),
    n_init = 2L,
    n_iter = 2L,
    score = loss_gaussian()
  )
  expect_error(cv_fit(g, folds, x, y), "make_score")
})

## ============================================================================
## 12.  grd_bayes — grid wrapping a pipeline
## ============================================================================

test_that("grd_bayes works with pipeline containing embedded bare grid", {
  skip_if_not_installed("rBayesianOptimization")

  params <- specify_hyperparameters(
    num.trees = make_range(50, 200),
    mtry = make_range(1, 2)
  )
  scr <- scr_correlation("Screen", cutoff = 0.05)
  rf_lrn <- lrn_ranger("RF_bayes", parameters = params)
  pl <- make_pipeline(scr, rf_lrn)
  g <- grd_bayes(
    pl,
    n_init = 2L,
    n_iter = 3L,
    score = make_score(loss_gaussian()),
    seed = 1L
  )

  res <- cv_fit(g, folds, x, y)

  expect_true(is.list(res))
  expect_true(length(res) > 0L)
  expect_true(all(grepl("^Screen/RF_bayes/", names(res))))
})

## ============================================================================
## 13.  grd_early_stop
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
    nrounds = make_discrete(3L, 5L, 8L, 12L),
    max_depth = make_discrete(1L, 2L),
    eta = make_discrete(0.05, 0.1),
    min_child_weight = make_range(1, 20)
  )

  g <- grd_early_stop(
    lrn_xgboost("XGB_early", parameters = params),
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
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L, 300L))
  g <- grd_early_stop(
    lrn_ranger("RF", parameters = params),
    score = make_score(loss_gaussian()),
    n_early_stop = 2L
  )
  res <- cv_fit(g, folds, x, y)
  expect_true(is.list(res))
  expect_true(length(res) >= 1L)
})

test_that("grd_early_stop errors with informative message when score is a raw mtl_loss", {
  params <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  g <- grd_early_stop(
    lrn_ranger("RF", parameters = params),
    score = loss_gaussian(),
    n_early_stop = 2L
  )
  expect_error(cv_fit(g, folds, x, y), "make_score")
})

test_that("grd_early_stop with multi-output pipeline errors without metalearner", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  glm <- lrn_glm("GLM", family = gaussian())
  params <- specify_hyperparameters(num.trees = make_discrete(50L, 100L))
  rf_bare <- lrn_ranger("RF", parameters = params)
  pl <- make_pipeline(scr, list(glm, rf_bare))

  g_no_mtl <- grd_early_stop(
    pl,
    score = make_score(loss_gaussian()),
    n_early_stop = 5L
  )
  expect_error(cv_fit(g_no_mtl, folds, x, y), "metalearner")
})

test_that("grd_early_stop with multi-output pipeline succeeds with metalearner", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  glm <- lrn_glm("GLM", family = gaussian())
  params <- specify_hyperparameters(num.trees = make_discrete(50L, 100L))
  rf_bare <- lrn_ranger("RF", parameters = params)
  pl <- make_pipeline(scr, list(glm, rf_bare))

  g_with_mtl <- grd_early_stop(
    pl,
    score = make_score(loss_gaussian(), metalearner = mtl_selector("sel")),
    n_early_stop = 5L
  )
  res <- cv_fit(g_with_mtl, folds, x, y)
  expect_true(length(res) >= 1L)
})

test_that("grd_early_stop searches over multiple HPs inside a pipeline", {
  scr <- scr_correlation("Scr", cutoff = 0.05)
  params <- specify_hyperparameters(
    num.trees = make_discrete(50L, 100L, 200L),
    mtry = make_discrete(1L, 2L)
  )
  rf_bare <- lrn_ranger("RF", parameters = params)
  pl <- make_pipeline(scr, rf_bare)

  g <- grd_early_stop(
    pl,
    score = make_score(loss_gaussian()),
    n_early_stop = 3L,
    seed = 1L
  )
  res <- cv_fit(g, folds, x, y)

  expect_true(is.list(res))
  expect_true(length(res) >= 1L)
  expect_true(all(grepl("^Scr/RF/", names(res))))
  combos <- lapply(res, function(p) attr(p, "combo"))
  expect_true(all(vapply(
    combos,
    function(co) {
      "num.trees" %in% names(co) && "mtry" %in% names(co)
    },
    logical(1L)
  )))
})


## ============================================================================
## 14.  Multi-parameterized-node pipeline
## ============================================================================

test_that("grd_random with two sequential parameterized pipeline nodes works", {
  scr_params <- specify_hyperparameters(cutoff = make_discrete(0.05, 0.2))
  scr_bare   <- scr_correlation("Scr", parameters = scr_params)
  rf_params  <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  rf_bare    <- lrn_ranger("RF", parameters = rf_params)
  pl         <- make_pipeline(scr_bare, rf_bare)

  grid <- grd_random(pl, n_candidates = 4L, seed = 1L)
  expect_s3_class(grid, "enfold_grid")
  expect_equal(grid$name, "Scr+RF")

  res <- cv_fit(grid, folds, x, y)
  expect_true(is.list(res))
  expect_true(length(res) >= 1L)

  combos <- lapply(res, function(p) attr(p, "combo"))
  expect_true(all(vapply(combos, function(co) {
    any(grepl("Scr/cutoff",   names(co))) &&
      any(grepl("RF/num.trees", names(co)))
  }, logical(1L))))
})

test_that("multi-node pipeline grid works inside full enfold task", {
  scr_params <- specify_hyperparameters(cutoff = make_discrete(0.05, 0.2))
  scr_bare   <- scr_correlation("Scr", parameters = scr_params)
  rf_params  <- specify_hyperparameters(num.trees = make_discrete(100L, 200L))
  rf_bare    <- lrn_ranger("RF", parameters = rf_params)
  pl         <- make_pipeline(scr_bare, rf_bare)
  g          <- grd_random(pl, n_candidates = 3L, seed = 1L)

  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), g) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
})

message("\n\u2713  All grid tests passed.\n")
