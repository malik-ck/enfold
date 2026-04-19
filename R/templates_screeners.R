#' Correlation screener
#'
#' Retains only those predictor columns whose (absolute) Pearson correlation
#' with \code{y} exceeds \code{cutoff}. Designed for use as the first stage
#' of a \code{\link{make_pipeline}}.
#'
#' @details
#' All \code{scr_*} templates return an \code{enfold_learner} whose
#' \code{predict()} method returns a subsetted or transformed data matrix.
#' They are designed as the first stage of a \code{\link{make_pipeline}},
#' feeding reduced features into downstream learners. Example:
#'
#' \preformatted{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' pl <- make_pipeline(
#'   scr_correlation("scr", cutoff = 0.2),
#'   list(lrn_glm("glm", family = gaussian()), lrn_mean("mean"))
#' )
#' task <- initialize_enfold(x, y) |>
#'   add_learners(pl) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = NA) |>
#'   fit()
#' }
#'
#' @param name Character scalar. Name of the screener.
#' @param cutoff Numeric scalar. Minimum correlation required to retain a
#'   variable. Default \code{0.1}.
#' @param abs Logical. If \code{TRUE} (default), the absolute value of the
#'   correlation is used.
#' @param min_vars Positive integer. Minimum number of variables to retain
#'   even if fewer than \code{min_vars} pass the cutoff. Default \code{2}.
#' @param parameters An \code{enfold_hyperparameters} object from
#'   \code{\link{specify_hyperparameters}()}, or \code{NULL} (default). When
#'   provided, returns a bare \code{enfold_grid} ready to wrap with a
#'   \code{grd_*()} constructor.
#' @return An \code{enfold_learner} whose \code{predict()} method returns a
#'   column-subset of the input data.
#' @seealso \code{\link{scr_lasso}}, \code{\link{scr_ranger}},
#'   \code{\link{make_pipeline}}
#' @export
scr_correlation <- function(name, cutoff = 0.1, abs = TRUE, min_vars = 2L, parameters = NULL) {
  if (min_vars < 1L || !is.numeric(min_vars) || length(min_vars) != 1L) {
    stop("'min_vars' must be a single positive integer.", call. = FALSE)
  }
  min_vars <- as.integer(min_vars)

  make_learner_factory(
    fit = function(x, y) {
      if (is.data.frame(x)) {
        var_classes <- vapply(x, class, character(1L))
        if (!all(var_classes %in% c("numeric", "integer"))) {
          stop(
            "Correlation screener not sensible for non-numeric variables; please convert factors to dummies or use a different screener.",
            call. = FALSE
          )
        }
      }

      get_cors <- apply(x, 2, function(clmn) stats::cor(clmn, y))
      if (abs) {
        get_cors <- abs(get_cors)
      }

      if (sum(get_cors > cutoff, na.rm = TRUE) < min_vars) {
        warning(
          "scr_correlation: fewer than 'min_vars' variables meet the cutoff; retaining top 'min_vars' variables.",
          call. = FALSE
        )
        return(order(get_cors, decreasing = TRUE)[seq_len(min_vars)])
      }

      return(which(get_cors > cutoff))
    },
    preds = function(object, data) {
      data[, object, drop = FALSE]
    },
    cutoff = 0.1,
    abs = TRUE,
    min_vars = 2L
  )(name = name, cutoff = cutoff, abs = abs, min_vars = min_vars, parameters = parameters)
}

#' LASSO (or elastic net) variable screener
#'
#' Fits a LASSO (or elastic net) on the training data and retains only those
#' variables with non-zero coefficients.  Variable selection is controlled by
#' either a fixed regularisation strength or a target number of variables.
#'
#' @param name Character scalar. Name of the screener.
#' @param alpha Numeric in \eqn{[0, 1]}. Elastic net mixing parameter passed to
#'   \code{glmnet}: \code{1} (default) is the LASSO, \code{0} is ridge.
#' @param lambda Numeric scalar or \code{NULL}. When provided, glmnet is called
#'   with this single \eqn{\lambda} value and the variables with non-zero
#'   coefficients are retained.  Exactly one of \code{lambda} or \code{n_vars}
#'   must be supplied.
#' @param n_vars Positive integer or \code{NULL}. When provided, the full
#'   regularisation path is fitted and the \eqn{\lambda} whose number of
#'   non-zero coefficients is closest to \code{n_vars} is used.  Exactly one of
#'   \code{lambda} or \code{n_vars} must be supplied.
#' @param parameters An \code{enfold_hyperparameters} object from
#'   \code{\link{specify_hyperparameters}()}, or \code{NULL} (default). When
#'   provided, returns a bare \code{enfold_grid} ready to wrap with a
#'   \code{grd_*()} constructor.
#' @return An \code{enfold_learner} whose \code{predict()} method returns a
#'   numeric matrix containing only the selected columns.
#' @seealso \code{\link{scr_correlation}}, \code{\link{scr_ranger}},
#'   \code{\link{make_pipeline}}
#' @examples
#' \dontrun{
#' scr <- scr_lasso("lasso_scr", n_vars = 5L)
#' fitted <- fit(scr, x = mtcars[, -1], y = mtcars$mpg)
#' ncol(predict(fitted, newdata = mtcars[, -1]))  # 5 columns
#' }
#' @export
scr_lasso <- function(name, alpha = 1, lambda = NULL, n_vars = NULL, parameters = NULL) {
  .msg_pkg("glmnet")

  if (is.null(lambda) && is.null(n_vars)) {
    stop(
      "Provide exactly one of 'lambda' or 'n_vars' to scr_lasso.",
      call. = FALSE
    )
  }
  if (!is.null(lambda) && !is.null(n_vars)) {
    stop(
      "Provide exactly one of 'lambda' or 'n_vars' to scr_lasso, not both.",
      call. = FALSE
    )
  }
  if (!is.null(n_vars)) {
    integer_checker(n_vars, "n_vars")
    n_vars <- as.integer(n_vars)
  }

  make_learner_factory(
    fit = function(x, y) {
      .check_pkg("glmnet")
      if (!is.null(dim(y))) {
        y <- y[, 1L]
      }
      if (is.data.frame(x)) {
        instr_list <- create_instruction_list(x)
        x_mat <- make_safe_matrix(x, instr_list)
      } else {
        instr_list <- "skip"
        x_mat <- as.matrix(x)
      }
      nms <- colnames(x_mat)
      if (is.null(nms)) {
        nms <- paste0("V", seq_len(ncol(x_mat)))
      }
      colnames(x_mat) <- nms
      col_means <- colMeans(x_mat)
      col_sds <- apply(x_mat, 2, sd)
      col_sds[col_sds == 0] <- 1
      x_sc <- sweep(sweep(x_mat, 2, col_means, "-"), 2, col_sds, "/")
      if (!is.null(lambda)) {
        fit_obj <- glmnet::glmnet(x_sc, y, alpha = alpha, lambda = lambda)
        coef_vec <- as.vector(stats::coef(fit_obj))[-1L]
      } else {
        fit_obj <- glmnet::glmnet(x_sc, y, alpha = alpha)
        n_nonzero <- colSums(fit_obj$beta != 0)
        idx <- which.min(abs(n_nonzero - n_vars))
        coef_mat <- as.matrix(fit_obj$beta[, idx, drop = FALSE])
        coef_vec <- coef_mat[, 1L, drop = TRUE]
      }
      selected <- which(coef_vec != 0)
      if (length(selected) == 0L) {
        warning(
          "scr_lasso: no variables selected; retaining all.",
          call. = FALSE
        )
        selected <- seq_len(ncol(x_mat))
      }
      list(selected_vars = nms[selected], instructions = instr_list)
    },
    preds = function(object, data) {
      .check_pkg("glmnet")
      x_mat <- if (!identical(object$instructions, "skip")) {
        make_safe_matrix(data, object$instructions)
      } else {
        x <- as.matrix(data)
        if (is.null(colnames(x))) {
          colnames(x) <- paste0("V", seq_len(ncol(x)))
        }
        x
      }
      x_mat[, object$selected_vars, drop = FALSE]
    },
    alpha = 1,
    lambda = NULL,
    n_vars = NULL
  )(name = name, alpha = alpha, lambda = lambda, n_vars = n_vars, parameters = parameters)
}


#' Random forest variable screener
#'
#' Fits a \pkg{ranger} random forest on the training data and retains the
#' \code{n_vars} variables with the highest variable importance.
#'
#' @param name Character scalar. Name of the screener.
#' @param n_vars Positive integer. Number of variables to retain.
#' @param importance Character scalar passed to \code{ranger::ranger}:
#'   \code{"impurity"} (default, Gini/variance impurity) or
#'   \code{"permutation"} (out-of-bag permutation importance, slower but more
#'   reliable for correlated predictors).
#' @param parameters An \code{enfold_hyperparameters} object from
#'   \code{\link{specify_hyperparameters}()}, or \code{NULL} (default). When
#'   provided, returns a bare \code{enfold_grid} ready to wrap with a
#'   \code{grd_*()} constructor.
#' @return An \code{enfold_learner} whose \code{predict()} method returns a
#'   numeric matrix containing only the selected columns.
#' @seealso \code{\link{scr_correlation}}, \code{\link{scr_lasso}},
#'   \code{\link{make_pipeline}}
#' @examples
#' \dontrun{
#' scr <- scr_ranger("rf_scr", n_vars = 4L)
#' fitted <- fit(scr, x = mtcars[, -1], y = mtcars$mpg)
#' ncol(predict(fitted, newdata = mtcars[, -1]))  # 4 columns
#' }
#' @export
scr_ranger <- function(name, n_vars, importance = "impurity", parameters = NULL) {
  .msg_pkg("ranger")

  if (missing(n_vars)) {
    stop("'n_vars' is required for scr_ranger.", call. = FALSE)
  }
  integer_checker(n_vars, "n_vars")
  n_vars <- as.integer(n_vars)

  make_learner_factory(
    fit = function(x, y) {
      .check_pkg("ranger")
      if (!is.null(dim(y))) {
        y <- y[, 1L]
      }
      if (is.data.frame(x)) {
        instr_list <- create_instruction_list(x)
        x_mat <- make_safe_matrix(x, instr_list)
      } else {
        instr_list <- "skip"
        x_mat <- as.matrix(x)
      }
      if (is.null(colnames(x_mat))) {
        colnames(x_mat) <- paste0("V", seq_len(ncol(x_mat)))
      }
      fit_obj <- ranger::ranger(
        x = data.frame(x_mat),
        y = y,
        num.trees = 500,
        importance = importance
      )
      imp <- fit_obj$variable.importance
      top_n <- min(n_vars, length(imp))
      selected_vars <- names(sort(imp, decreasing = TRUE))[seq_len(top_n)]
      list(selected_vars = selected_vars, instructions = instr_list)
    },
    preds = function(object, data) {
      .check_pkg("ranger")
      x_mat <- if (!identical(object$instructions, "skip")) {
        make_safe_matrix(data, object$instructions)
      } else {
        x <- as.matrix(data)
        if (is.null(colnames(x))) {
          colnames(x) <- paste0("V", seq_len(ncol(x)))
        }
        x
      }
      x_mat[, object$selected_vars, drop = FALSE]
    },
    n_vars,
    importance = "impurity"
  )(name = name, n_vars = n_vars, importance = importance, parameters = parameters)
}

#' Truncated SVD basis-expansion screener
#'
#' Projects the data onto its leading right singular vectors.  If the
#' \pkg{RSpectra} package is installed, a fast truncated SVD is used; otherwise
#' the computation falls back to base \code{svd()}.  Because the projection is
#' computed from the training data alone, this is a valid unsupervised
#' pre-processing step inside cross-validation folds.
#'
#' @param name Character scalar. Name of the screener.
#' @param n_components Positive integer. Number of singular vectors (components)
#'   to retain.
#' @param center Logical. If \code{TRUE} (default), columns are mean-centred
#'   before the decomposition.
#' @param scale Logical. If \code{FALSE} (default), no column scaling is
#'   applied.  Set to \code{TRUE} to divide each column by its standard
#'   deviation, which is advisable when predictors are on very different scales.
#' @param parameters An \code{enfold_hyperparameters} object from
#'   \code{\link{specify_hyperparameters}()}, or \code{NULL} (default). When
#'   provided, returns a bare \code{enfold_grid} ready to wrap with a
#'   \code{grd_*()} constructor.
#' @return An \code{enfold_learner} whose \code{predict()} method returns an
#'   \eqn{n \times k} numeric matrix of projected scores.
#' @seealso \code{\link{scr_correlation}}, \code{\link{bex_splines}},
#'   \code{\link{make_pipeline}}
#' @examples
#' scr <- scr_svd("svd_scr", n_components = 3L)
#' fitted <- fit(scr, x = as.matrix(mtcars[, -1]), y = mtcars$mpg)
#' dim(predict(fitted, newdata = as.matrix(mtcars[, -1])))  # n x 3
#' @export
scr_svd <- function(name, n_components, center = TRUE, scale = FALSE, parameters = NULL) {
  if (!requireNamespace("RSpectra", quietly = TRUE)) {
    message(
      "Package 'RSpectra' not found; scr_svd will use base R's svd(), which may be slow for large datasets.\nInstall 'RSpectra' for faster truncated SVD.",
      call. = FALSE
    )
  }

  if (missing(n_components)) {
    stop("'n_components' is required for scr_svd.", call. = FALSE)
  }
  integer_checker(n_components, "n_components")
  n_components <- as.integer(n_components)

  make_learner_factory(
    fit = function(x, y = NULL) {
      if (is.data.frame(x)) {
        instr_list <- create_instruction_list(x)
        x_mat <- make_safe_matrix(x, instr_list)
      } else {
        instr_list <- "skip"
        x_mat <- as.matrix(x)
      }

      center_vec <- if (isTRUE(center)) colMeans(x_mat) else rep(0, ncol(x_mat))
      scale_vec <- if (isTRUE(scale)) {
        sds <- apply(x_mat, 2, sd)
        sds[sds == 0] <- 1
        sds
      } else {
        rep(1, ncol(x_mat))
      }

      x_sc <- sweep(sweep(x_mat, 2, center_vec, "-"), 2, scale_vec, "/")

      k <- min(n_components, nrow(x_sc) - 1L, ncol(x_sc))

      V <- if (requireNamespace("RSpectra", quietly = TRUE)) {
        RSpectra::svds(x_sc, k = k)$v
      } else {
        svd(x_sc, nu = 0, nv = k)$v
      }

      list(
        V = V,
        center_vec = center_vec,
        scale_vec = scale_vec,
        instructions = instr_list
      )
    },
    preds = function(object, data) {
      x_mat <- if (!identical(object$instructions, "skip")) {
        make_safe_matrix(data, object$instructions)
      } else {
        as.matrix(data)
      }
      x_sc <- sweep(
        sweep(x_mat, 2, object$center_vec, "-"),
        2,
        object$scale_vec,
        "/"
      )
      x_sc %*% object$V
    },
    n_components,
    center = TRUE,
    scale = FALSE
  )(name = name, n_components = n_components, center = center, scale = scale, parameters = parameters)
}
