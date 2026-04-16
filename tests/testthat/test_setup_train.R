## tests/test_setup_train.R
##
## Tests for the enfold_task workflow:
##   initialize_enfold() → add_learners() → add_metalearners() → add_cv_folds() → fit()
## Source after: devtools::load_all()

stopifnot(requireNamespace("future", quietly = TRUE))
future::plan("sequential")

## ---- Shared data -----------------------------------------------------------
set.seed(42)
n <- 120
x <- data.frame(
  a = rnorm(n),
  b = rnorm(n),
  c = rnorm(n),
  d = rnorm(n)
)
y <- 2 * x$a - x$b + rnorm(n, sd = 0.5)

## ---- Learners --------------------------------------------------------------
my_mean <- lrn_mean("Mean")
my_glm <- lrn_glm("GLM", family = gaussian())

my_screen <- scr_correlation("Screen", cutoff = 0.1)
my_pipeline <- make_pipeline(my_screen, list(my_mean, my_glm))


## ============================================================================
## 1.  initialize_enfold() — structure checks
## ============================================================================

test_that("initialize_enfold returns an enfold_task", {
  task <- initialize_enfold(x, y)
  expect_s3_class(task, "enfold_task")
  expect_identical(task$x_env$x, x)
  expect_identical(task$y_env$y, y)
  expect_length(task$learners, 0L)
  expect_length(task$metalearners, 0L)
})

test_that("initialize_enfold stores x in a locked environment", {
  task <- initialize_enfold(x, y)
  expect_true(environmentIsLocked(task$x_env))
})

test_that("initialize_enfold rejects non-matrix/data-frame x", {
  expect_error(initialize_enfold(as.list(x), y), "data frame or matrix")
})

test_that("initialize_enfold rejects mismatched nrow(x) and length(y)", {
  expect_error(initialize_enfold(x, y[-1L]), "same number")
})


## ============================================================================
## 2.  add_learners() / add_metalearners()
## ============================================================================

test_that("add_learners appends learners", {
  task <- initialize_enfold(x, y) |> add_learners(my_mean, my_glm)
  expect_length(task$learners, 2L)
})

test_that("add_learners rejects duplicate names", {
  dup <- lrn_mean("GLM") # same name as my_glm
  expect_error(
    initialize_enfold(x, y) |> add_learners(my_glm, dup),
    "unique"
  )
})

test_that("add_learners rejects already-fitted learners", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted_task <- fit(task)
  # The fitted learner objects inside the task should not be re-addable
  fitted_lrn <- fitted_task$fit_objects[[1L]][[1L]]
  expect_error(add_learners(fitted_task, fitted_lrn), "fitted")
})

test_that("add_metalearners appends metalearners", {
  task <- initialize_enfold(x, y) |> add_metalearners(mtl_selector("Sel"))
  expect_length(task$metalearners, 1L)
})


## ============================================================================
## 3.  add_cv_folds()
## ============================================================================

test_that("add_cv_folds populates task$cv with both fold sets", {
  task <- initialize_enfold(x, y) |>
    add_cv_folds(inner_cv = 5L, outer_cv = 3L)
  expect_s3_class(task$cv, "enfold_cv")
  expect_false(is.null(task$cv$build_sets))
  expect_false(is.null(task$cv$performance_sets))
  expect_length(task$cv$performance_sets, 3L)
  expect_length(task$cv$build_sets[[1L]], 5L)
})

test_that("add_cv_folds with inner-only CV leaves performance_sets NULL", {
  task <- initialize_enfold(x, y) |>
    add_cv_folds(inner_cv = 5L, outer_cv = NA)
  expect_false(is.null(task$cv$build_sets))
  expect_null(task$cv$performance_sets)
})

test_that("add_cv_folds accepts a pre-built enfold_cv object", {
  cv <- create_cv_folds(n, inner_cv = 3L, outer_cv = NA)
  task <- initialize_enfold(x, y) |> add_cv_folds(cv = cv)
  expect_s3_class(task$cv, "enfold_cv")
})


## ============================================================================
## 4.  fit() — inner CV only (no outer folds)
## ============================================================================

test_that("fit with inner-only CV returns enfold_task_fitted", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Selector")) |>
    add_cv_folds(inner_cv = 5L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  expect_s3_class(fitted, "enfold_task")
  expect_false(fitted$is_cv_ensemble)
  expect_length(fitted$fit_objects, 1L) # one synthetic full-data fold
  expect_length(fitted$fit_objects[[1L]], 2L) # two learners
})

test_that("fit always retains x and y in locked environments", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Selector")) |>
    add_cv_folds(inner_cv = 5L, outer_cv = NA)
  fitted <- fit(task)

  expect_identical(fitted$x_env$x, x)
  expect_identical(fitted$y_env$y, y)
})

test_that("fit records fitted_learner_names and fitted_metalearner_names", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_setequal(fitted$fitted_learner_names, c("Mean", "GLM"))
  expect_setequal(fitted$fitted_metalearner_names, "Sel")
})


## ============================================================================
## 5.  fit() — inner + outer CV
## ============================================================================

test_that("fit with inner + outer CV sets is_cv_ensemble = TRUE", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Selector")) |>
    add_cv_folds(inner_cv = 5L, outer_cv = 3L)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  expect_true(fitted$is_cv_ensemble)
  expect_length(fitted$fit_objects, 3L) # three outer folds
  expect_length(fitted$ensembles, 3L)
})


## ============================================================================
## 6.  fit() — pipeline learner
## ============================================================================

test_that("fit works when a pipeline learner is included", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_pipeline) |>
    add_metalearners(mtl_selector("Selector")) |>
    add_cv_folds(inner_cv = 5L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  classes_fold1 <- lapply(fitted$fit_objects[[1L]], class)
  has_pipeline <- any(vapply(
    classes_fold1,
    function(cls) {
      any(grepl("pipeline", cls, ignore.case = TRUE))
    },
    logical(1L)
  ))
  expect_true(has_pipeline)
})


## ============================================================================
## 7.  fit() — input validation errors
## ============================================================================

test_that("fit stops when no CV folds have been added", {
  task <- initialize_enfold(x, y) |> add_learners(my_mean)
  expect_error(fit(task), "add_cv_folds")
})

test_that("fit stops when no learners have been added", {
  task <- initialize_enfold(x, y) |>
    add_cv_folds(inner_cv = 5L, outer_cv = NA)
  expect_error(fit(task), "learners")
})

test_that("fit stops when outer CV is requested without a metalearner", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_glm) |>
    add_cv_folds(inner_cv = 5L, outer_cv = 3L)
  expect_error(fit(task), "metalearner")
})

## ============================================================================
## 8.  predict() — ensemble type
## ============================================================================

test_that("predict(type='ensemble') returns a numeric vector of length n", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Selector")) |>
    add_cv_folds(inner_cv = 5L, outer_cv = NA)
  fitted <- fit(task)

  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
})

test_that("predict errors when type is not provided", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)
  expect_error(predict(fitted), "type")
})


## ============================================================================
## 9.  predict() — cv type (requires outer folds)
## ============================================================================

test_that("predict(type='cv') returns a vector of length n with an indices attr", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Selector")) |>
    add_cv_folds(inner_cv = 5L, outer_cv = 3L)
  fitted <- fit(task)
  cv_preds <- predict(fitted, type = "cv")

  expect_length(cv_preds, n)
  expect_true(is.numeric(cv_preds))
  expect_false(is.null(attr(cv_preds, "indices")))
})

test_that("predict(type='cv') errors without outer CV", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Selector")) |>
    add_cv_folds(inner_cv = 5L, outer_cv = NA)
  fitted <- fit(task)
  expect_error(predict(fitted, type = "cv"), "outer CV")
})


## ============================================================================
## 10.  predict() — ensemble type with pipeline learner
## ============================================================================

test_that("predict(type='ensemble') works when a pipeline learner is included", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_pipeline) |>
    add_metalearners(mtl_selector("Selector")) |>
    add_cv_folds(inner_cv = 5L, outer_cv = NA)
  fitted <- fit(task)

  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
  expect_true(is.numeric(preds))
})


## ============================================================================
## 11.  Incremental fitting — add metalearner, refit
## ============================================================================

test_that("adding a metalearner and refitting appends it to ensembles", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 5L, outer_cv = NA)
  fitted <- fit(task)

  n_mtl_before <- length(fitted$ensembles[[1L]])

  fitted2 <- fitted |>
    add_metalearners(mtl_superlearner("SL")) |>
    fit()

  expect_s3_class(fitted2, "enfold_task_fitted")
  expect_gt(length(fitted2$ensembles[[1L]]), n_mtl_before)
  expect_true("SL" %in% fitted2$fitted_metalearner_names)
})

test_that("fit returns unchanged object and messages when nothing new was added", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_message(fit(fitted), "No new")
})

test_that("adding a learner and refitting triggers full re-fit", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  fitted2 <- fitted |> add_learners(my_glm) |> fit()

  expect_s3_class(fitted2, "enfold_task_fitted")
  expect_true("GLM" %in% fitted2$fitted_learner_names)
  # Two learners now in final fit objects
  expect_length(fitted2$fit_objects[[1L]], 2L)
})


message("\n\u2713  All setup/train tests passed.\n")
