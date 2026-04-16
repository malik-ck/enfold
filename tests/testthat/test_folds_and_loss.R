## tests/test_folds_and_loss.R
##
## Tests for fold primitives (new_fold, new_fold_list, new_cv, exclude,
## training_set, validation_set, create_cv_folds) and loss evaluation.
##
## Source after: devtools::load_all()

future::plan("sequential")

set.seed(99)
n <- 80
x <- data.frame(a = rnorm(n), b = rnorm(n))
y <- x$a + rnorm(n, sd = 0.3)


## ============================================================================
## 1.  new_fold — complementary storage
## ============================================================================

test_that("new_fold with n and no training_set is complementary", {
  fold <- new_fold(validation_set = 1:20, n = 80L)
  expect_s3_class(fold, "enfold_fold")
  expect_true(fold$complementary)
  expect_equal(training_set(fold), 21:80)
  expect_equal(validation_set(fold), 1:20)
})

test_that("new_fold with explicit training_set that is complementary collapses to complementary", {
  fold <- new_fold(validation_set = 1:20, training_set = 21:80, n = 80L)
  expect_true(fold$complementary)
  expect_null(fold$training_set)
})

test_that("new_fold with explicit non-complementary training_set stores it", {
  fold <- new_fold(validation_set = 1:20, training_set = 41:80)
  expect_false(fold$complementary)
  expect_equal(training_set(fold), 41:80)
  expect_equal(validation_set(fold), 1:20)
})

test_that("new_fold errors if neither training_set nor n is provided", {
  expect_error(new_fold(validation_set = 1:20), "training_set.*n")
})


## ============================================================================
## 2.  exclude()
## ============================================================================

test_that("exclude returns a new fold, does not mutate in place", {
  fold <- new_fold(validation_set = 1:20, n = 80L)
  fold2 <- exclude(fold, 5:10)
  # Original unchanged
  expect_equal(validation_set(fold), 1:20)
  # New fold excludes the indices
  expect_false(any(5:10 %in% validation_set(fold2)))
  expect_false(any(5:10 %in% training_set(fold2)))
})

test_that("exclude.enfold_fold_list propagates to all folds", {
  fl <- new_fold_list(list(
    new_fold(validation_set = 1:20, n = 80L),
    new_fold(validation_set = 21:40, n = 80L)
  ))
  fl2 <- exclude(fl, 1:5)
  expect_false(any(1:5 %in% validation_set(fl2[[1L]])))
  # Fold 2 is not in 1:5's validation set, but they are removed from training
  expect_false(any(1:5 %in% training_set(fl2[[2L]])))
})


## ============================================================================
## 3.  new_fold_list
## ============================================================================

test_that("new_fold_list rejects non-fold elements", {
  expect_error(new_fold_list(list(1, 2)), "enfold_fold")
})

test_that("new_fold_list length matches input", {
  folds <- lapply(1:5, function(i) {
    new_fold(validation_set = (i * 10 + 1):(i * 10 + 10), n = 80L)
  })
  fl <- new_fold_list(folds)
  expect_length(fl, 5L)
  expect_s3_class(fl, "enfold_fold_list")
})


## ============================================================================
## 4.  new_cv / create_cv_folds
## ============================================================================

test_that("new_cv requires at least one non-NULL argument", {
  expect_error(new_cv(), "non-NULL")
})

test_that("create_cv_folds with both inner and outer CV yields correct structure", {
  cv <- create_cv_folds(n = 80L, inner_cv = 3L, outer_cv = 2L)
  expect_s3_class(cv, "enfold_cv")
  expect_length(cv$performance_sets, 2L)
  expect_length(cv$build_sets, 2L) # one inner fold set per outer fold
  expect_length(cv$build_sets[[1L]], 3L) # 3 inner folds
})

test_that("create_cv_folds inner-only leaves performance_sets NULL", {
  cv <- create_cv_folds(n = 80L, inner_cv = 5L, outer_cv = NA)
  expect_null(cv$performance_sets)
  expect_false(is.null(cv$build_sets))
})

test_that("create_cv_folds outer-only leaves build_sets NULL", {
  cv <- create_cv_folds(n = 80L, inner_cv = NA, outer_cv = 4L)
  expect_null(cv$build_sets)
  expect_false(is.null(cv$performance_sets))
})

test_that("create_cv_folds inner fold indices are subsets of outer training set", {
  cv <- create_cv_folds(n = 80L, inner_cv = 3L, outer_cv = 2L)
  outer_tr <- training_set(cv$performance_sets[[1L]])
  inner_all <- unlist(lapply(cv$build_sets[[1L]], function(f) {
    c(training_set(f), validation_set(f))
  }))
  expect_true(all(inner_all %in% outer_tr))
})

test_that("create_cv_folds rejects both inner_cv and outer_cv as NA", {
  expect_error(create_cv_folds(n = 80L, inner_cv = NA, outer_cv = NA), "non-NA")
})

test_that("exclude.enfold_cv propagates to both sets", {
  cv <- create_cv_folds(n = 80L, inner_cv = 3L, outer_cv = 2L)
  cv2 <- exclude(cv, 1L)
  # index 1 should not appear in any fold's validation or training
  perf_val <- unlist(lapply(cv2$performance_sets, validation_set))
  expect_false(1L %in% perf_val)
})


## ============================================================================
## 5.  make_learner_factory basics
## ============================================================================

test_that("make_learner_factory returns a constructor function", {
  my_factory <- make_learner_factory(
    fit = function(x, y) mean(y),
    preds = function(object, data) rep(object, nrow(data))
  )
  expect_true(is.function(my_factory))
})

test_that("constructor from factory returns an enfold_learner", {
  my_factory <- make_learner_factory(
    fit = function(x, y) mean(y),
    preds = function(object, data) rep(object, nrow(data))
  )
  lrn <- my_factory(name = "MeanLearner")
  expect_s3_class(lrn, "enfold_learner")
  expect_equal(lrn$name, "MeanLearner")
})

test_that("fit then predict on a simple learner works end-to-end", {
  my_factory <- make_learner_factory(
    fit = function(x, y) mean(y),
    preds = function(object, data) rep(object, nrow(data))
  )
  lrn <- my_factory(name = "MeanLearner")
  fitted <- fit(lrn, x, y)
  expect_s3_class(fitted, "enfold_learner_fitted")
  preds <- predict(fitted, newdata = x)
  expect_length(preds, n)
  expect_true(all(preds == mean(y)))
})

test_that("make_learner_factory hyperparameter is trapped in closure", {
  my_factory <- make_learner_factory(
    fit = function(x, y) list(mu = mean(y) + shift),
    preds = function(object, data) rep(object$mu, nrow(data)),
    shift = 0
  )
  lrn_shifted <- my_factory(name = "ShiftedMean", shift = 10)
  fitted <- fit(lrn_shifted, x, y)
  preds <- predict(fitted, newdata = x)
  expect_true(all(abs(preds - (mean(y) + 10)) < 1e-10))
})

test_that("make_learner_factory rejects fit/preds args with extra formals", {
  expect_error(
    make_learner_factory(
      fit = function(x, y, extra) mean(y),
      preds = function(object, data) rep(object, nrow(data))
    ),
    "only have"
  )
})

test_that("get_params retrieves trapped hyperparameters", {
  my_factory <- make_learner_factory(
    fit = function(x, y) mean(y),
    preds = function(object, data) rep(object, nrow(data)),
    alpha = 0.5
  )
  lrn <- my_factory(name = "L", alpha = 0.9)
  params <- get_params(lrn)
  expect_equal(params$alpha, 0.9)
})


## ============================================================================
## 6.  risk.enfold_task_fitted
## ============================================================================

test_that("risk() errors if loss_fun is not provided", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean")) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)
  expect_error(risk(fitted, type = "ensemble"), "loss_fun")
})

test_that("risk() errors if loss_fun is not an mtl_loss", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean")) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)
  expect_error(
    risk(
      fitted,
      loss_fun = function(y, y_hat) (y - y_hat)^2,
      type = "ensemble"
    ),
    "mtl_loss"
  )
})

test_that("risk() returns named numeric vector for each metalearner", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel"), mtl_superlearner("SL")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  risks <- risk(fitted, loss_fun = loss_gaussian(), type = "ensemble")
  expect_true(is.numeric(risks))
  expect_setequal(names(risks), c("Sel", "SL"))
  expect_true(all(risks >= 0))
})

test_that("risk() with type='cv' requires outer CV", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean")) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)
  expect_error(
    risk(fitted, loss_fun = loss_gaussian(), type = "cv"),
    "outer CV"
  )
})

test_that("risk() with type='cv' returns named numeric vector", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  risks <- risk(fitted, loss_fun = loss_gaussian(), type = "cv")
  expect_true(is.numeric(risks))
  expect_true("Sel" %in% names(risks))
  expect_true(all(risks >= 0))
})

test_that("risk(type='cv') equals fold-size-weighted mean of fold_risk()", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  r <- risk(fitted, loss_fun = loss_gaussian(), type = "cv")
  fr <- fold_risk(fitted, loss_fun = loss_gaussian())
  fold_sizes <- vapply(
    fitted$cv$performance_sets,
    function(f) length(validation_set(f)),
    integer(1L)
  )
  weighted <- apply(fr, 2L, function(col) weighted.mean(col, fold_sizes))
  expect_equal(r, weighted, tolerance = 1e-12)
})


## ============================================================================
## 7.  fold_risk, cv_loss, predict_learners, risk_learners
## ============================================================================

test_that("fold_risk() returns a folds x metalearners numeric matrix", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel"), mtl_superlearner("SL")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 4L)
  fitted <- fit(task)

  fr <- fold_risk(fitted, loss_fun = loss_gaussian())
  expect_true(is.matrix(fr))
  expect_equal(nrow(fr), 4L)
  expect_setequal(colnames(fr), c("Sel", "SL"))
  expect_true(all(fr >= 0))
  expect_equal(rownames(fr), paste0("fold_", 1:4))
})

test_that("fold_risk() requires outer CV", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean")) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)
  expect_error(fold_risk(fitted, loss_fun = loss_gaussian()), "outer CV")
})

test_that("fold_risk() respects metalearner_name subsetting", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel"), mtl_superlearner("SL")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  fr <- fold_risk(fitted, loss_fun = loss_gaussian(), metalearner_name = "Sel")
  expect_equal(ncol(fr), 1L)
  expect_equal(colnames(fr), "Sel")
})

test_that("cv_loss() returns a data frame with .index and metalearner columns", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  df <- cv_loss(fitted, loss_fun = loss_gaussian())
  expect_true(is.data.frame(df))
  expect_equal(nrow(df), n)
  expect_true(".index" %in% names(df))
  expect_true("Sel" %in% names(df))
  expect_true(all(df$Sel >= 0))
})

test_that("cv_loss() requires outer CV", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean")) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)
  expect_error(cv_loss(fitted, loss_fun = loss_gaussian()), "outer CV")
})

test_that("predict_learners(type='ensemble') returns named list of learner predictions", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)

  pl <- predict_learners(fitted, type = "ensemble")
  expect_true(is.list(pl))
  expect_setequal(names(pl), c("Mean", "GLM"))
  expect_length(pl$Mean, n)
  expect_length(pl$GLM, n)
})

test_that("predict_learners(type='cv') returns named list with indices attribute", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  pl <- predict_learners(fitted, type = "cv")
  expect_true(is.list(pl))
  expect_setequal(names(pl), c("Mean", "GLM"))
  expect_length(pl$Mean, n)
  expect_false(is.null(attr(pl$Mean, "indices")))
})

test_that("predict_learners(type='cv') requires outer CV", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean")) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = NA)
  fitted <- fit(task)
  expect_error(predict_learners(fitted, type = "cv"), "outer CV")
})

test_that("predict_learners(type='ensemble') requires fold_id for cv_ensemble objects", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean")) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)
  expect_error(predict_learners(fitted, type = "ensemble"), "fold_id")
})

test_that("risk_learners() returns named numeric vector of mean losses", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean"), lrn_glm("GLM", family = gaussian())) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  rl <- risk_learners(fitted, loss_fun = loss_gaussian(), type = "cv")
  expect_true(is.numeric(rl))
  expect_setequal(names(rl), c("Mean", "GLM"))
  expect_true(all(rl >= 0))
})

test_that("risk_learners() is consistent with manual mean of cv_loss per learner pred", {
  task <- initialize_enfold(x, y) |>
    add_learners(lrn_mean("Mean")) |>
    add_metalearners(mtl_selector("Sel")) |>
    add_cv_folds(inner_cv = 3L, outer_cv = 3L)
  fitted <- fit(task)

  rl <- risk_learners(fitted, loss_fun = loss_gaussian(), type = "cv")
  pl <- predict_learners(fitted, type = "cv")
  idx <- attr(pl$Mean, "indices")
  manual <- mean(loss_gaussian()$loss_fun(y[idx], pl$Mean))
  expect_equal(rl[["Mean"]], manual, tolerance = 1e-12)
})


## ============================================================================
## 8.  print methods smoke tests
## ============================================================================

test_that("print.enfold_learner does not emit message('NULL') to stderr", {
  lrn <- lrn_mean("Mean")
  # Should print to stdout and return invisibly — NOT produce a warning or message
  expect_no_warning(print(lrn))
  expect_no_message(print(lrn))
})

test_that("print.enfold_fold does not error", {
  fold <- new_fold(validation_set = 1:20, n = 80L)
  expect_no_error(print(fold))
})

test_that("print.enfold_cv does not error", {
  cv <- create_cv_folds(n = 80L, inner_cv = 3L, outer_cv = 2L)
  expect_no_error(print(cv))
})
