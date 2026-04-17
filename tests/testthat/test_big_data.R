## tests/test_big_data.R
##
## Tests for big data backends: subset_x dispatch, enfold_arrow_file,
## in-memory Arrow Table, file-path Arrow, and FBM backends.
##
## Source after: devtools::load_all()

set.seed(42)
future::plan("sequential")

n <- 30L
df <- data.frame(a = rnorm(n), b = rnorm(n), c = rnorm(n))
mat <- as.matrix(df)
y   <- rnorm(n)
idx <- c(1L, 5L, 10L, 15L, 20L)


# ── subset_x — default (matrix / data frame) ─────────────────────────────────

test_that("subset_x.default works on a matrix", {
  result <- subset_x(mat, idx)
  expect_true(is.matrix(result))
  expect_equal(nrow(result), length(idx))
  expect_equal(result, mat[idx, , drop = FALSE])
})

test_that("subset_x.default works on a data frame", {
  result <- subset_x(df, idx)
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), length(idx))
  expect_equal(result, df[idx, , drop = FALSE])
})


# ── enfold_arrow_file — unit tests ───────────────────────────────────────────

test_that("enfold_arrow_file: new_arrow_file validates path and caches nrow", {
  skip_if_not_installed("arrow")
  tmp <- tempfile(fileext = ".feather")
  on.exit(unlink(tmp), add = TRUE)
  arrow::write_feather(df, tmp)

  af <- new_arrow_file(tmp)
  expect_s3_class(af, "enfold_arrow_file")
  expect_equal(af$nrow, n)
  expect_equal(nrow(af), n)
})

test_that("enfold_arrow_file: new_arrow_file errors on missing file", {
  skip_if_not_installed("arrow")
  expect_error(new_arrow_file("/no/such/file.feather"), regexp = "")
})

test_that("enfold_arrow_file: subset_x returns a data frame with correct rows", {
  skip_if_not_installed("arrow")
  tmp <- tempfile(fileext = ".feather")
  on.exit(unlink(tmp), add = TRUE)
  arrow::write_feather(df, tmp)

  af <- new_arrow_file(tmp)
  result <- subset_x(af, idx)
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), length(idx))
  expect_equal(result, data.frame(df[idx,], row.names = NULL), ignore_attr = TRUE)
})

test_that("enfold_arrow_file: print method works", {
  skip_if_not_installed("arrow")
  tmp <- tempfile(fileext = ".feather")
  on.exit(unlink(tmp), add = TRUE)
  arrow::write_feather(df, tmp)
  af <- new_arrow_file(tmp)
  expect_output(print(af), "enfold_arrow_file")
})


# ── In-memory Arrow Table ─────────────────────────────────────────────────────

test_that("subset_x.ArrowTabular returns a data frame with correct rows", {
  skip_if_not_installed("arrow")
  tbl <- arrow::as_arrow_table(df)
  result <- subset_x(tbl, idx)
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), length(idx))
  expect_equal(result, data.frame(df[idx, ], row.names = NULL), ignore_attr = TRUE)
})

test_that("initialize_enfold accepts an in-memory Arrow Table", {
  skip_if_not_installed("arrow")
  tbl <- arrow::as_arrow_table(df)
  task <- initialize_enfold(tbl, y)
  expect_s3_class(task, "enfold_task")
  expect_true("arrow" %in% task$future_pkgs)
})


# ── File-path Arrow (enfold_arrow_file via initialize_enfold) ─────────────────

test_that("initialize_enfold accepts a Feather file path", {
  skip_if_not_installed("arrow")
  tmp <- tempfile(fileext = ".feather")
  on.exit(unlink(tmp), add = TRUE)
  arrow::write_feather(df, tmp)

  task <- initialize_enfold(tmp, y)
  expect_s3_class(task, "enfold_task")
  expect_s3_class(task$x_env$x, "enfold_arrow_file")
  expect_true("arrow" %in% task$future_pkgs)
})

test_that("full workflow with Feather file path completes without error", {
  skip_if_not_installed("arrow")
  tmp <- tempfile(fileext = ".feather")
  on.exit(unlink(tmp), add = TRUE)
  arrow::write_feather(df, tmp)

  task <- initialize_enfold(tmp, y) |>
    add_learners(lrn_mean("mean"), lrn_glm("GLM", "auto")) |>
    add_metalearners(mtl_selector("SL")) |>
    add_cv_folds(inner_cv = 10L, outer_cv = 10L) |>
    fit()
  expect_s3_class(task, "enfold_task_fitted")
})


# ── FBM ──────────────────────────────────────────────────────────────────────

test_that("subset_x.FBM returns a matrix with correct rows", {
  skip_if_not_installed("bigstatsr")
  fbm <- bigstatsr::as_FBM(mat, type = "double")
  result <- subset_x(fbm, idx)
  expect_true(is.matrix(result))
  expect_equal(nrow(result), length(idx))
  expect_equal(result, unname(mat[idx, , drop = FALSE]))
})

test_that("initialize_enfold accepts an FBM", {
  skip_if_not_installed("bigstatsr")
  fbm <- bigstatsr::as_FBM(mat)
  task <- initialize_enfold(fbm, y)
  expect_s3_class(task, "enfold_task")
  expect_true("bigstatsr" %in% task$future_pkgs)
})

test_that("full workflow with FBM completes without error", {
  skip_if_not_installed("bigstatsr")
  fbm <- bigstatsr::as_FBM(mat)
  task <- initialize_enfold(fbm, y) |>
    add_learners(lrn_mean("mean"), lrn_glm("GLM", "auto")) |>
    add_metalearners(mtl_selector("SL")) |>
    add_cv_folds(inner_cv = 10L, outer_cv = 10L) |>
    fit()
  expect_s3_class(task, "enfold_task_fitted")
})


# ── Error for unsupported type ────────────────────────────────────────────────

test_that("initialize_enfold rejects unsupported x types", {
  expect_error(initialize_enfold(list(1, 2, 3), y), regexp = "must be")
})
