## tests/testthat/test_cols_subset.R
##
## Tests for the `cols` argument in initialize_enfold(), which restricts
## predictors used during training and prediction.

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

my_mean <- lrn_mean("Mean")
my_glm  <- lrn_glm("GLM", family = gaussian())

## ============================================================================
## 1.  initialize_enfold — cols validation
## ============================================================================

test_that("initialize_enfold accepts integer cols", {
  # Integer cols are normalised to character names for matrix/data.frame
  task <- initialize_enfold(x, y, cols = c(1L, 3L))
  expect_identical(task$cols, c("a", "c"))
})

test_that("initialize_enfold accepts character cols", {
  task <- initialize_enfold(x, y, cols = c("a", "d"))
  expect_identical(task$cols, c("a", "d"))
})

test_that("initialize_enfold rejects empty cols", {
  expect_error(initialize_enfold(x, y, cols = character(0L)), "empty")
})

test_that("initialize_enfold rejects out-of-range integer cols", {
  expect_error(initialize_enfold(x, y, cols = c(1L, 99L)), "between")
})

test_that("initialize_enfold rejects invalid character cols", {
  expect_error(initialize_enfold(x, y, cols = c("a", "z")), "not found")
})

test_that("initialize_enfold stores NULL cols by default", {
  task <- initialize_enfold(x, y)
  expect_null(task$cols)
})

## ============================================================================
## 2.  subset_x with cols
## ============================================================================

test_that("subset_x respects integer cols for data frames", {
  sub <- subset_x(x, 1:10, cols = c(1L, 3L))
  expect_equal(ncol(sub), 2L)
  expect_equal(nrow(sub), 10L)
  expect_identical(names(sub), c("a", "c"))
})

test_that("subset_x respects character cols for data frames", {
  sub <- subset_x(x, 1:10, cols = c("b", "d"))
  expect_equal(ncol(sub), 2L)
  expect_identical(names(sub), c("b", "d"))
})

test_that("subset_x with NULL cols returns all columns", {
  sub <- subset_x(x, 1:10, cols = NULL)
  expect_equal(ncol(sub), 4L)
})

test_that("subset_x with cols on matrix input works", {
  xm <- as.matrix(x)
  sub <- subset_x(xm, 1:10, cols = c(2L, 4L))
  expect_equal(ncol(sub), 2L)
  expect_equal(nrow(sub), 10L)
})

## ============================================================================
## 3.  fit + predict with character cols
## ============================================================================

test_that("fit and predict work with character cols", {
  task <- initialize_enfold(x, y, cols = c("a", "c")) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  expect_identical(fitted$cols, c("a", "c"))

  # Ensemble prediction on training data
  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)

  # newdata must match the subset width
  newdata_ok <- x[, c("a", "c"), drop = FALSE]
  preds_new <- predict(fitted, newdata = newdata_ok, type = "ensemble")
  expect_length(preds_new, nrow(newdata_ok))
})

## ============================================================================
## 4.  fit + predict with integer cols
## ============================================================================

test_that("fit and predict work with integer cols", {
  task <- initialize_enfold(x, y, cols = 1:2) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  expect_s3_class(fitted, "enfold_task_fitted")
  preds <- predict(fitted, type = "ensemble")
  expect_length(preds, n)
})

## ============================================================================
## 5.  CV predict with cols
## ============================================================================

test_that("predict(type='cv') works with cols", {
  task <- initialize_enfold(x, y, cols = c("a", "b")) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  cv_preds <- predict(fitted, type = "cv")
  expect_length(cv_preds, n)
  expect_true(is.numeric(cv_preds))
})

## ============================================================================
## 6.  risk() with cols
## ============================================================================

test_that("risk() works with cols", {
  task <- initialize_enfold(x, y, cols = c("a", "c")) |>
    add_learners(my_mean, my_glm) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  r <- risk(fitted, loss_fun = loss_gaussian(), type = "ensemble")
  expect_true(is.numeric(r))
  expect_length(r, 1L)
})

## ============================================================================
## 7.  Predictors line in print.enfold_task_fitted
## ============================================================================

test_that("fitted task with cols prints Predictors line", {
  task <- initialize_enfold(x, y, cols = c("a", "b")) |>
    add_learners(my_mean) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  output <- capture.output(print(fitted))
  expect_true(any(grepl("Predictors", output)))
})

test_that("fitted task without cols does not print Predictors line", {
  task <- initialize_enfold(x, y) |>
    add_learners(my_mean) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  output <- capture.output(print(fitted))
  expect_false(any(grepl("Predictors", output)))
})

## ============================================================================
## 8.  newdata dimension mismatch with cols
## ============================================================================

test_that("predict(type='cv') errors on wrong ncol when cols is set", {
  task <- initialize_enfold(x, y, cols = c("a", "b")) |>
    add_learners(my_mean) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  # newdata with wrong number of columns
  expect_error(
    predict(fitted, newdata = x, type = "cv"),
    "ncol"
  )
})

message("\n\u2713  All cols subset tests passed.\n")