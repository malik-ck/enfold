set.seed(1)

test_that("specify_hyperparameters creates object with sample functions", {
  hp <- specify_hyperparameters(
    alpha = c(0.1, 1),
    lambda = make_range(1e-3, 1e-1)
  )
  expect_s3_class(hp, "enfold_hyperparameters")
  expect_equal(names(hp), c("alpha", "lambda"))
  expect_true(is.list(attr(hp, "sample_funs")))
  expect_true(is.function(attr(hp, "sample_funs")$alpha))
  expect_true(is.function(attr(hp, "sample_funs")$lambda))
})

test_that("log_transform replaces range sampling with log-uniform sampling", {
  hp <- specify_hyperparameters(
    alpha = c(0.1, 1),
    lambda = make_range(1e-3, 1e-1)
  )
  hp <- log_transform(hp, lambda)

  draws <- attr(hp, "sample_funs")$lambda(1000)
  expect_true(all(draws >= 1e-3))
  expect_true(all(draws <= 1e-1))
  expect_true(abs(median(log(draws)) - mean(log(c(1e-3, 1e-1)))) < 0.2)
})

test_that("forbid excludes invalid discrete combinations and draw returns valid rows", {
  hp <- specify_hyperparameters(
    alpha = seq(0, 1, by = 0.02),
    penalty = c("l1", "l2")
  )
  hp <- forbid(hp, alpha < 0.5 && penalty == "l2")

  draws <- draw(hp, 20)
  expect_equal(nrow(draws), 20)
  expect_false(any(draws$alpha < 0.5 & draws$penalty == "l2"))
  expect_true(all(draws$alpha %in% seq(0, 1, by = 0.02)))
  expect_true(all(draws$penalty %in% c("l1", "l2")))
})

test_that("draw exhausts discrete grid when n is NULL", {
  hp <- specify_hyperparameters(alpha = c(0.1, 1), penalty = c("l1", "l2"))
  hp <- forbid(hp, alpha == 0.1 && penalty == "l2")

  combos <- draw(hp, n = NULL)
  expect_equal(nrow(combos), 3)
  expect_setequal(combos$alpha, c(0.1, 1))
  expect_false(any(combos$alpha == 0.1 & combos$penalty == "l2"))
})

test_that("draw with log-transformed range returns values within original bounds", {
  hp <- specify_hyperparameters(
    alpha = c(0.1, 1),
    lambda = make_range(1e-3, 1e-1)
  )
  hp <- log_transform(hp, lambda)

  draws <- draw(hp, 50)
  expect_equal(nrow(draws), 50)
  expect_true(all(draws$alpha %in% c(0.1, 1)))
  expect_true(all(draws$lambda >= 1e-3))
  expect_true(all(draws$lambda <= 1e-1))
})

message("\n\u2713  All hyperparameter tests passed.\n")
