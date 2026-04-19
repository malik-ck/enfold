## tests/test_graceful_failure.R
##
## Tests documenting how enfold handles learner failures at different levels:
##   - cv_fit level (inner-CV fold loop): graceful per-path failure
##   - build_ensembles level: failed learners excluded from preds_list
##   - fit_ensemble level (final fit): see note in Section 4
##
## KEY FINDINGS (from reading R/enfold_4_train.R + R/cv_contributing.R):
##
##   Plain learner failures:
##     build_ensembles correctly excludes the learner from the metalearner's
##     preds_list, but learners_for_fit is not filtered for non-grid learners.
##     fit_ensemble therefore still tries to fit the failed learner on the full
##     data. End-to-end graceful failure for plain learners is NOT complete.
##
##   Pipeline failures:
##     cv_fit.enfold_pipeline excludes only the failing path, not the whole
##     pipeline. If all paths fail, the pipeline itself gets a failed_learner
##     attribute. fit.enfold_pipeline (used in fit_ensemble) re-throws errors
##     rather than isolating per-path failures, so the same gap applies.
##
##   Grid failures:
##     Failing candidates are excluded both in cv_fit.enfold_grid and via the
##     resolved_learner mechanism in fit.enfold_task. End-to-end graceful
##     failure for grids IS complete.
##
## Source after: devtools::load_all()

stopifnot(requireNamespace("future", quietly = TRUE))
future::plan("sequential")

## ── Shared data ────────────────────────────────────────────────────────────
set.seed(42)
n <- 80
x <- data.frame(a = rnorm(n), b = rnorm(n), c = rnorm(n))
y <- x$a - 0.5 * x$b + rnorm(n, sd = 0.5)

## ── Test-only learner factories ────────────────────────────────────────────

# Always fails on fit — used to simulate a totally broken learner.
lrn_bomb <- make_learner_factory(
  fit = function(x, y) stop("intentional fit failure"),
  preds = function(object, data) rep(0, nrow(data))
)

# Fails when should_fail = TRUE; otherwise returns the mean of y as predictions.
# Used to test grids where some candidates fail and some succeed.
lrn_maybe_fail <- make_learner_factory(
  fit = function(x, y) {
    if (should_fail) {
      stop("intentional conditional failure")
    }
    list(mu = mean(y))
  },
  preds = function(object, data) rep(object$mu, nrow(data)),
  should_fail = FALSE
)

## ── Shared fold list for cv_fit unit tests ─────────────────────────────────
folds <- create_cv_folds(nrow(x), 3, NA)$build_sets[[1L]]


## ============================================================================
## 1.  cv_fit.default — plain learner failure
## ============================================================================

test_that("cv_fit on a bomb learner returns failed_learner attribute", {
  bomb <- lrn_bomb("Bomb")
  result <- expect_warning(
    cv_fit(bomb, folds, x, y),
    "Bomb"
  )
  expect_length(result, 0L)
  expect_equal(attr(result, "failed_learner"), "Bomb")
})

test_that("cv_fit on a working learner returns named prediction vector", {
  good <- lrn_mean("Mean")
  result <- cv_fit(good, folds, x, y)
  expect_named(result, "Mean")
  # Predictions are combined across folds — length equals n
  expect_length(result[["Mean"]], n)
  expect_false(is.null(attr(result[["Mean"]], "indices")))
})


## ============================================================================
## 2.  cv_fit.enfold_pipeline — per-path failure
## ============================================================================

test_that("cv_fit on a pipeline excludes only the failing branch", {
  # Two terminal paths: Screen/Mean (good) and Screen/Bomb (bad)
  scr <- scr_correlation("Screen", cutoff = 0.0)
  good_lrn <- lrn_mean("Mean")
  bomb <- lrn_bomb("Bomb")
  pl <- make_pipeline(scr, list(good_lrn, bomb))

  result <- expect_warning(
    cv_fit(pl, folds, x, y),
    "Bomb"
  )

  # Good path survives, bad path is gone
  expect_true("Screen/Mean" %in% names(result))
  expect_false("Screen/Bomb" %in% names(result))
  expect_null(attr(result, "failed_learner"))

  # Surviving path covers all n observations
  expect_length(result[["Screen/Mean"]], n)
})

test_that("cv_fit on a pipeline returns failed_learner when all paths fail", {
  scr <- scr_correlation("Screen", cutoff = 0.0)
  bomb <- lrn_bomb("Bomb")
  pl <- make_pipeline(scr, bomb)

  result <- expect_warning(
    cv_fit(pl, folds, x, y),
    regexp = "Bomb|Screen"
  )

  expect_length(result, 0L)
  expect_false(is.null(attr(result, "failed_learner")))
})

test_that("cv_fit pipeline: stage-1 failure excludes the entire pipeline", {
  # A bomb as the first stage should fail all downstream paths
  bomb_scr <- lrn_bomb("BombScr")
  good_lrn <- lrn_mean("Mean")
  pl <- make_pipeline(bomb_scr, good_lrn)

  result <- expect_warning(
    cv_fit(pl, folds, x, y),
    regexp = "BombScr|stage 1"
  )

  expect_length(result, 0L)
  expect_false(is.null(attr(result, "failed_learner")))
})


## ============================================================================
## 3.  cv_fit.enfold_grid — per-candidate failure
## ============================================================================

test_that("cv_fit on a grid excludes only failing candidates", {
  params <- specify_hyperparameters(should_fail = make_discrete(FALSE, TRUE))
  grid <- grd_random(lrn_maybe_fail("MF", parameters = params))

  result <- expect_warning(
    cv_fit(grid, folds, x, y),
    "MF"
  )

  # One combo survives (should_fail = FALSE), one is dropped
  expect_length(result, 1L)
  expect_match(names(result)[[1L]], "should_fail=FALSE")
  expect_null(attr(result, "failed_learner"))
  expect_length(result[[1L]], n)
})

test_that("cv_fit on a grid returns failed_learner when all candidates fail", {
  params <- specify_hyperparameters(should_fail = make_discrete(TRUE))
  grid <- grd_random(lrn_maybe_fail("MF", parameters = params))

  result <- expect_warning(
    cv_fit(grid, folds, x, y),
    "MF"
  )

  expect_length(result, 0L)
  expect_equal(attr(result, "failed_learner"), "MF")
})


## ============================================================================
## 4.  build_ensembles — failed_learners attribute
## ============================================================================

## NOTE: build_ensembles excludes failing learners from preds_list (correct),
## but fit.enfold_task does NOT remove non-grid failed learners from
## learners_for_fit. fit_ensemble would therefore still try to fit them on the
## full data. End-to-end handling is only complete for grids (see Section 5).

test_that("build_ensembles stores failed learner names in attribute", {
  cv <- create_cv_folds(n, inner_cv = 3L, outer_cv = NA)
  good <- lrn_mean("Mean")
  bomb <- lrn_bomb("Bomb")
  mtl <- mtl_selector("Sel")

  ensembles <- expect_warning(
    build_ensembles(
      cv = cv,
      learners = list(good, bomb),
      metalearners = list(mtl),
      x = x,
      y = y
    ),
    "Bomb"
  )

  # One build fold; check its failed_learners attribute
  expect_false(is.null(attr(ensembles[[1L]], "failed_learners")))
  expect_true("Bomb" %in% attr(ensembles[[1L]], "failed_learners"))
})

test_that("build_ensembles errors when all learners fail", {
  cv <- create_cv_folds(n, inner_cv = 3L, outer_cv = NA)
  bomb <- lrn_bomb("Bomb")
  mtl <- mtl_selector("Sel")

  expect_error(
    suppressWarnings(build_ensembles(
      cv = cv,
      learners = list(bomb),
      metalearners = list(mtl),
      x = x,
      y = y
    )),
    "All learners failed"
  )
})


## ============================================================================
## 5.  End-to-end fit() with grid failures (graceful — grid resolution works)
## ============================================================================

test_that("fit() with a grid where one candidate fails: surviving candidate used", {
  params <- specify_hyperparameters(should_fail = make_discrete(FALSE, TRUE))
  grid <- grd_random(lrn_maybe_fail("MF", parameters = params))

  task <- initialize_enfold(x, y) |>
    add_learners(grid) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)

  fitted <- expect_warning(fit(task), "MF")
  expect_s3_class(fitted, "enfold_task_fitted")

  # The ensemble should produce predictions from the surviving combo
  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
})

test_that("fit() with a grid where all candidates fail: grid excluded, error if no other learner", {
  params <- specify_hyperparameters(should_fail = make_discrete(TRUE))
  grid <- grd_random(lrn_maybe_fail("MF", parameters = params))

  task <- initialize_enfold(x, y) |>
    add_learners(grid) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)

  # All candidates fail → grid excluded → no learners left → error
  expect_error(
    suppressWarnings(fit(task)),
    regexp = "All learners failed|no candidates survived"
  )
})

test_that("fit() with grid where all fail but another good learner exists: ensemble uses survivor", {
  params <- specify_hyperparameters(should_fail = make_discrete(TRUE))
  grid <- grd_random(lrn_maybe_fail("MF", parameters = params))
  good <- lrn_mean("Mean")

  task <- initialize_enfold(x, y) |>
    add_learners(good, grid) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)

  fitted <- expect_warning(fit(task), "MF")
  expect_s3_class(fitted, "enfold_task_fitted")

  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
})


message("\n\u2713  All graceful-failure tests passed.\n")
