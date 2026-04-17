#' Empirical mean learner
#'
#' A benchmark learner that ignores predictors and always predicts the
#' training-set mean of \code{y}. Useful as a lower bound for comparison.
#'
#' @details
#' All \code{lrn_*} templates return an \code{enfold_learner} that can be
#' passed to \code{\link{add_learners}}. General workflow (works with any
#' \code{lrn_*} constructor):
#'
#' \preformatted{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(
#'     lrn_mean("mean"),
#'     lrn_glm("glm", family = gaussian())
#'   ) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#' predict(task, type = "cv")
#' }
#'
#' @param name Character. Name for this learner instance.
#' @return An \code{enfold_learner} object.
#' @export
lrn_mean <- make_learner_factory(
  fit = function(x, y) mean(y),
  preds = function(object, data) rep(object, nrow(data))
)

#' Generalized linear model learner
#'
#' Fits a GLM via \code{stats::glm}. Handles data frames with factor and
#' character columns automatically.
#'
#' @param name Character. Name for this learner instance.
#' @param family A family object (e.g. \code{gaussian()}, \code{binomial()},
#'   \code{poisson()}) or the string \code{"auto"} to let the learner guess
#'   from \code{y}.
#' @param offset Character or \code{NULL}. Name of a column in \code{x} to
#'   use as an offset.
#' @return An \code{enfold_learner} object.
#' @seealso \code{\link{lrn_gam}}, \code{\link{lrn_glmnet}},
#'   \code{\link{make_learner_factory}}
#' @examples
#' lrn <- lrn_glm("glm_gauss", family = gaussian())
#' fitted <- fit(lrn, x = mtcars[, -1], y = mtcars$mpg)
#' head(predict(fitted, newdata = mtcars[, -1]))
#' @export
lrn_glm <- function(name, family, offset = NULL) {
  if (missing(family)) {
    stop(
      "Argument 'family' is missing. You need to specify a family object ",
      "(e.g., gaussian()), or use family = 'auto' to let the learner guess.",
      call. = FALSE
    )
  }

  return(
    make_learner_factory(
      fit = function(x, y) {
        if (identical(family, "auto")) {
          family <- family_guesser(y)
        }

        if (is.data.frame(x)) {
          instr_list <- create_instruction_list(x)
          x <- make_safe_matrix(x, instr_list)
        } else {
          instr_list <- "skip"
          x <- as.matrix(x)
          if (is.null(colnames(x))) colnames(x) <- paste0("X", seq_len(ncol(x)))
        }

        fd <- data.frame(y = y, x)

        # Formula logic
        if (!is.null(offset)) {
          # Ensure offset is handled as a character string for the formula
          ofs_char <- paste0("offset(", offset, ")")
          offset_col <- which(colnames(fd) == offset)
          # Remove y and the offset column from predictors
          predictors <- colnames(fd)[-c(1, offset_col)]
        } else {
          ofs_char <- character(0)
          predictors <- colnames(fd)[-1]
        }

        get_frm <- stats::as.formula(
          paste0("y ~ ", paste(c(ofs_char, predictors), collapse = " + "))
        )

        get_model <- stats::glm(get_frm, family = family, data = fd)

        return(list(
          model = get_model,
          instructions = instr_list,
          col_names = colnames(x)
        ))
      },

      preds = function(object, data) {
        if (!identical(object[["instructions"]], "skip")) {
          data <- make_safe_matrix(data, object[["instructions"]])
        } else {
          data <- as.matrix(data)
          colnames(data) <- object[["col_names"]]
        }
        data <- data.frame(data)
        return(as.vector(stats::predict(
          object[["model"]],
          newdata = data,
          type = "response"
        )))
      },
      family,
      offset = NULL
    )(name = name, family = family, offset = offset)
  )
}


#' Generalized additive model learner
#'
#' Fits a GAM via \code{mgcv::gam} (or the faster \code{mgcv::bam} when
#' \code{fast = TRUE}). Numeric columns with more than two unique values are
#' smoothed automatically; binary and categorical columns enter as linear
#' terms. Requires the \pkg{mgcv} package.
#'
#' @param name Character. Name for this learner instance.
#' @param family A family object (e.g. \code{gaussian()}) or \code{"auto"}.
#' @param offset Character or \code{NULL}. Column name in \code{x} to use as
#'   an offset.
#' @param k Integer. Maximum basis dimension for smooth terms. Defaults to
#'   \code{10}, capped at the number of unique values when fewer are present.
#' @param frm Formula or \code{NULL}. If supplied, used as-is instead of the
#'   automatically constructed formula.
#' @param smoother Character. Basis type passed to \code{mgcv::s()}.
#'   Default \code{"tp"} (thin plate regression splines).
#' @param fast Logical. Use \code{mgcv::bam} with \code{discrete = TRUE} and
#'   \code{method = "fREML"} for faster fitting on large data sets.
#' @param method Character or \code{NULL}. Smoothing parameter estimation
#'   method. Ignored when \code{fast = TRUE} (which always uses
#'   \code{"fREML"}).
#' @return An \code{enfold_learner} object.
#' @seealso \code{\link{lrn_glm}}, \code{\link{lrn_mboost}}
#' @examples
#' \dontrun{
#' lrn <- lrn_gam("gam", family = gaussian())
#' fitted <- fit(lrn, x = mtcars[, -1], y = mtcars$mpg)
#' head(predict(fitted, newdata = mtcars[, -1]))
#' }
#' @export
lrn_gam <- function(
  name,
  family,
  offset = NULL,
  k = 10,
  frm = NULL,
  smoother = "tp",
  fast = TRUE,
  method = NULL
) {
  .msg_pkg("mgcv")

  if (missing(family)) {
    stop(
      "Argument 'family' is missing. You need to specify a family object ",
      "(e.g., gaussian()), or use family = 'auto' to let the learner guess.",
      call. = FALSE
    )
  }

  if (fast && !is.null(method)) {
    stop(
      "Cannot specify 'method' when 'fast = TRUE', since method will be set to 'fREML'.",
      call. = FALSE
    )
  }

  make_learner_factory(
    fit = function(x, y) {
      .check_pkg("mgcv")
      if (identical(family, "auto")) {
        family <- family_guesser(y)
      }
      # Make a safe matrix if necessary and no formula provided
      if (is.data.frame(x) & is.null(frm)) {
        instr_list <- create_instruction_list(x)
        x <- make_safe_matrix(x, instr_list)
      } else {
        instr_list <- "skip"
        x <- as.matrix(x)
        if (is.null(colnames(x))) colnames(x) <- paste0("X", seq_len(ncol(x)))
      }
      # Need a data frame for GAMs
      fd <- data.frame(y = y, x)
      # Now need to construct formula
      # If provided, just use provided formula
      if (!is.null(frm)) {
        use_frm <- frm
      } else {
        # Get numeric columns with sufficient length
        filter_numeric <- which(unlist(lapply(fd[, -1], function(x) {
          is.numeric(x) & length(unique(x)) > 2
        })))
        # For those of sufficient length, get the number of unique values (capped at k)
        unique_vals <- unlist(lapply(
          fd[, (filter_numeric + 1), drop = FALSE],
          function(x) ifelse(length(unique(x)) <= k, length(unique(x)), k)
        ))
        # Now can construct formula
        if (!is.null(offset)) {
          offset_part <- paste("offset(", offset, ")", sep = "")
        } else {
          offset_part <- character(0)
        }
        offset_col <- which(colnames(fd) == offset)
        numeric_part_vars <- colnames(fd)[(filter_numeric + 1)]
        indicator_part_vars <- colnames(fd[-c(1, filter_numeric + 1)])
        # If there is an offset, remove it from numeric and indicator parts
        if (!is.null(offset)) {
          if (any(numeric_part_vars == offset)) {
            unique_vals <- unique_vals[-which(numeric_part_vars == offset)]
            numeric_part_vars <- numeric_part_vars[
              -which(numeric_part_vars == offset)
            ]
          }
          if (any(indicator_part_vars == offset)) {
            indicator_part_vars <- indicator_part_vars[
              -which(indicator_part_vars == offset)
            ]
          }
        }
        numeric_part <- paste0(
          "s(",
          numeric_part_vars,
          ", k = ",
          unique_vals,
          ", bs = \"",
          smoother,
          "\")"
        )
        indicator_part <- paste0(
          indicator_part_vars
        )
        # Combine all and remove last two of string (since that is an overhang +)
        use_frm <- paste(
          "y ~ ",
          paste0(
            c(offset_part, numeric_part, indicator_part),
            collapse = " + "
          ),
          collapse = ""
        )
        use_frm <- stats::formula(use_frm)
      }
      # Fit here
      if (fast) {
        model_obj <- mgcv::bam(
          use_frm,
          nthreads = 1,
          family = family,
          data = fd,
          discrete = TRUE,
          method = "fREML"
        )
      } else {
        method <- ifelse(is.null(method), "REML", method)
        model_obj <- mgcv::gam(
          use_frm,
          family = family,
          data = fd,
          method = method
        )
      }
      # Can now fit!
      return(list(
        model = model_obj,
        instructions = instr_list,
        col_names = colnames(x)
      ))
    },
    preds = function(object, data) {
      .check_pkg("mgcv")
      # Apply instruction list if necessary
      if (!identical(object[["instructions"]], "skip")) {
        data <- make_safe_matrix(data, object[["instructions"]])
      } else {
        data <- as.matrix(data)
        colnames(data) <- object[["col_names"]]
      }
      # Then need to coerce to data frame either way
      data <- data.frame(data)
      # Then output
      return(as.vector(stats::predict(
        object[["model"]],
        newdata = data,
        type = "response"
      )))
    },
    family,
    offset = NULL,
    k = 10,
    frm = NULL,
    smoother = "tp",
    fast = TRUE,
    method = NULL
  )(
    name = name,
    family = family,
    offset = offset,
    k = k,
    frm = frm,
    smoother = smoother
  )
}


#' Gradient boosting learner (mboost)
#'
#' Fits a component-wise gradient boosting model via \code{mboost::mboost}.
#' Numeric variables are fitted with P-spline base learners (\code{bbs});
#' binary and categorical variables with linear base learners (\code{bols}).
#' Requires the \pkg{mboost} package.
#'
#' @param name Character. Name for this learner instance.
#' @param family An \pkg{mboost} family object (e.g.
#'   \code{mboost::Gaussian()}, \code{mboost::Binomial(type = "glm")}).
#'   Unlike other \code{lrn_*} learners, \code{"auto"} is not supported
#'   because \pkg{mboost} uses its own family system.
#' @param offset Character or \code{NULL}. Column name in \code{x} to use
#'   as an offset.
#' @param mstop Integer. Number of boosting iterations. Default \code{100}.
#' @param nu Numeric. Shrinkage (learning rate). Default \code{0.1}.
#' @param frm Formula or \code{NULL}. If supplied, used as-is.
#' @param max_df Integer. Maximum effective degrees of freedom for numeric
#'   smooth base learners. Default \code{5}.
#' @param knots Integer. Number of B-spline knots for \code{bbs} base
#'   learners. Default \code{20}.
#' @param df_factor Numeric in \eqn{(0, 1]}. Scales the degrees of freedom
#'   relative to the number of unique values. Default \code{0.99}.
#' @return An \code{enfold_learner} object.
#' @seealso \code{\link{lrn_gam}}, \code{\link{lrn_glm}}
#' @examples
#' \dontrun{
#' lrn <- lrn_mboost("boost", family = mboost::Gaussian(), mstop = 200)
#' fitted <- fit(lrn, x = mtcars[, -1], y = mtcars$mpg)
#' head(predict(fitted, newdata = mtcars[, -1]))
#' }
#' @export
lrn_mboost <- function(
  name,
  family,
  offset = NULL,
  mstop = 100,
  nu = 0.1,
  frm = NULL,
  max_df = 5,
  knots = 20,
  df_factor = 0.99
) {
  .msg_pkg("mboost")

  if (missing(family)) {
    stop(
      "Argument 'family' is missing. You need to specify a family object. There is no 'auto' for mboost, which has custom families.",
      call. = FALSE
    )
  }

  make_learner_factory(
    fit = function(x, y) {
      .check_pkg("mboost")
      # Create safe matrix if necessary
      if (is.data.frame(x) & is.null(frm)) {
        instr_list <- create_instruction_list(x)
        x <- make_safe_matrix(x, instr_list)
      } else {
        instr_list <- "skip"
        x <- as.matrix(x)
        if (is.null(colnames(x))) colnames(x) <- paste0("X", seq_len(ncol(x)))
      }
      # Check y and family
      if (!is.factor(y) && identical(family@name, mboost::Binomial()@name)) {
        stop(
          "For binary regression, mboost expects 'Binomial(type = 'glm')' for family. With 'Binomial()' it defaults to adaboost and fails on numeric y."
        )
      }
      # Need a data frame for Mboost
      fd <- data.frame(y = y, intr = 1, x)
      # Now need to construct formula
      # If provided, just use provided formula
      if (!is.null(frm)) {
        use_frm <- frm
      } else {
        # Get numeric columns with sufficient length
        filter_numeric <- which(unlist(lapply(
          fd[, -1, drop = FALSE],
          function(x) is.numeric(x) & length(unique(x)) > 2
        )))
        # Also get if too few unique values
        filter_tiny <- which(unlist(lapply(fd[, -1, drop = FALSE], function(x) {
          length(unique(x)) == 1
        })))
        # For those of sufficient length, get the number of unique values (capped at k)
        get_n_knot <- unlist(lapply(
          fd[, (filter_numeric + 1), drop = FALSE],
          function(x) {
            ifelse(length(unique(x)) <= knots + 2, length(unique(x)), knots + 2)
          }
        ))
        get_df <- unlist(lapply(
          fd[, (filter_numeric + 1), drop = FALSE],
          function(x) {
            ifelse(length(unique(x)) <= max_df, length(unique(x)) - 1, max_df)
          }
        ))
        # Get variable names and remove offset if it exists
        numeric_part_vars <- colnames(fd)[(filter_numeric + 1)]
        indicator_part_vars <- colnames(fd)[-c(1, filter_tiny + 1)]
        if (!is.null(offset)) {
          offset_part <- paste("offset(", offset, ")", sep = "")
        } else {
          offset_part <- character(0)
        }
        offset_col <- which(colnames(fd) == offset)
        if (!is.null(offset)) {
          if (any(numeric_part_vars == offset)) {
            get_n_knot <- get_n_knot[-which(numeric_part_vars == offset)]
            get_df <- get_df[-which(numeric_part_vars == offset)]
            numeric_part_vars <- numeric_part_vars[
              -which(numeric_part_vars == offset)
            ]
          }
          if (any(indicator_part_vars == offset)) {
            indicator_part_vars <- indicator_part_vars[
              -which(indicator_part_vars == offset)
            ]
          }
        }
        # Now can construct formula
        numeric_part <- paste0(
          "bbs(",
          numeric_part_vars,
          ", df = ",
          get_df * df_factor,
          ", center = TRUE, knots = ",
          get_n_knot,
          ")"
        )
        indicator_part <- paste0(
          "bols(",
          indicator_part_vars,
          ", intercept = FALSE, df = ",
          df_factor,
          ")"
        )
        # Combine all and remove last two of string (since that is an overhang +)
        use_frm <- stats::formula(paste(
          "y ~ ",
          paste0(c(numeric_part, indicator_part), collapse = " + "),
          collapse = ""
        ))
      }
      # Can now fit!
      return(list(
        model = mboost::mboost(
          use_frm,
          data = fd,
          family = family,
          control = mboost::boost_control(mstop = mstop, nu = nu)
        ),
        instructions = instr_list,
        col_names = colnames(x)
      ))
    },
    preds = function(object, data) {
      .check_pkg("mboost")
      # Apply instruction list if necessary
      if (!identical(object[["instructions"]], "skip")) {
        data <- make_safe_matrix(data, object[["instructions"]])
      } else {
        data <- as.matrix(data)
        colnames(data) <- object[["col_names"]]
      }
      # Then need to coerce to data frame either way
      data <- data.frame(data)
      # Then output
      return(as.vector(stats::predict(
        object[["model"]],
        newdata = data,
        type = "response"
      )))
    },
    family,
    offset = NULL,
    mstop = 100,
    nu = 0.1,
    frm = NULL,
    max_df = 5,
    knots = 20,
    df_factor = 0.99
  )(
    name = name,
    family = family,
    offset = offset,
    mstop = mstop,
    nu = nu,
    frm = frm,
    max_df = max_df,
    knots = knots,
    df_factor = df_factor
  )
}


#' Highly adaptive LASSO (HAL) learner
#'
#' Fits a Highly Adaptive Lasso via \code{hal9001::fit_hal}. Predictors are
#' internally normalized and can optionally be expanded by a user-supplied
#' formula. Requires the \pkg{hal9001} package.
#'
#' @param name Character. Name for this learner instance.
#' @param family A family object or \code{"auto"}.
#' @param offset Character or \code{NULL}. Column name in \code{x} to use
#'   as an offset.
#' @param frm Formula or \code{NULL}. Passed to \code{model.matrix()} for
#'   custom basis expansion before fitting.
#' @param max_degree Integer. Maximum interaction degree for the HAL basis.
#'   Default \code{2}.
#' @param smoothness_orders Integer. Smoothness order for the HAL basis.
#'   Default \code{1}.
#' @param num_knots Integer. Number of knots per basis function. Default
#'   \code{50}.
#' @return An \code{enfold_list} object. Fitting it
#' creates 100 learners, one for each value of \eqn{\lambda} in the regularization path.
#' @seealso \code{\link{lrn_glmnet}}, \code{\link{lrn_glm}}
#' @examples
#' \dontrun{
#' lrn <- lrn_hal("hal", family = gaussian(), max_degree = 1)
#' fitted <- fit(lrn, x = as.matrix(mtcars[, -1]), y = mtcars$mpg)
#' head(predict(fitted, newdata = as.matrix(mtcars[, -1])))
#' }
#' @export
lrn_hal <- function(
  name,
  family,
  offset = NULL,
  frm = NULL,
  max_degree = 2,
  smoothness_orders = 1,
  num_knots = 50
) {
  .msg_pkg("hal9001")

  if (missing(family)) {
    stop(
      "Argument 'family' is missing. You need to specify a family object ",
      "(e.g., gaussian()), or use family = 'auto' to let the learner guess.",
      call. = FALSE
    )
  }

  # Some checks ensuring there are integers where necessary
  integer_checker(max_degree, "the interaction degree.")
  integer_checker(smoothness_orders, "the smoothness.")

  make_learner_factory(
    fit = function(x, y) {
      .check_pkg("hal9001")
      if (identical(family, "auto")) {
        family <- family_guesser(y)
      }
      # If there is an offset, retrieve as variable and remove from data
      if (!is.null(offset)) {
        get_offset <- x[, offset]
        if (!is.numeric(get_offset)) {
          stop(
            "Called from lrn_hal: please ensure that the offset you provide is numeric."
          )
        }
        x <- x[, -which(colnames(x) == offset)]
      } else {
        get_offset <- NULL
      }
      # Make a safe matrix if necessary and no formula provided
      if (is.data.frame(x) & is.null(frm)) {
        instr_list <- create_instruction_list(x)
        x <- make_safe_matrix(x, instr_list)
      } else if (is.data.frame(x) & !is.null(frm)) {
        # If frm provided, use in model.matrix call
        instr_list <- list(coerce_df = FALSE, frm = frm)
        x <- stats::model.matrix(frm, data = x)[, -1, drop = FALSE]
      } else if (is.matrix(x) & is.null(frm)) {
        # Simplest case: take x as is
        instr_list <- "skip"
      } else if (is.matrix(x) & !is.null(frm)) {
        # If formula provided for matrix, coerce to df and warn
        instr_list <- list(coerce_df = TRUE, frm = frm)
        x <- stats::model.matrix(frm, data = data.frame(x))[, -1, drop = FALSE]
        warning(
          "Formula provided for matrix in hal.\nCoerced to data frame for call to model.matrix(); can be (silently) very unsafe!"
        )
      } else {
        stop("Unexpected data input in hal! Should not happen.")
      }
      # Normalize...
      flex_list <- list()
      get_sds <- apply(x, 2, sd)
      flex_list[["rescale"]] <- list(
        means = apply(x, 2, mean),
        sds = ifelse(get_sds == 0, 1, get_sds)
      )
      x <- sweep(x, 2, flex_list[["rescale"]][["means"]], "-")
      x <- sweep(x, 2, flex_list[["rescale"]][["sds"]], "/")

      # Can now fit!
      old_controls <- glmnet::glmnet.control()
      glmnet::glmnet.control(fdev = 0)
      get_model <- hal9001::fit_hal(
        x,
        y,
        family = family,
        offset = get_offset,
        max_degree = max_degree,
        smoothness_orders = smoothness_orders,
        num_knots = num_knots,
        fit_control = list(cv_select = FALSE)
      )

      do.call(glmnet::glmnet.control, old_controls)

      return(list(
        model = get_model,
        instructions = instr_list,
        flexibility = flex_list
      ))
    },
    preds = function(object, data) {
      .check_pkg("hal9001")
      # If there is an offset, retrieve as variable and remove from data
      if (!is.null(offset)) {
        get_offset <- data[, offset]
        if (!is.numeric(get_offset)) {
          stop(
            "Called from lrn_hal: please ensure that the offset you provide is numeric."
          )
        }
        data <- data[, -which(colnames(data) == offset)]
      } else {
        get_offset <- NULL
      }
      # Apply instruction list if necessary
      if (!(identical(object[["instructions"]], "skip"))) {
        # Only need logic if data not already right
        # If matrix, need to coerce to df again for model matrix call
        if (is.matrix(data)) {
          data <- stats::model.matrix(
            object[["instructions"]][["frm"]],
            data = data.frame(data)
          )[, -1, drop = FALSE]
        } else if (!is.null(object[["instructions"]][["frm"]])) {
          data <- stats::model.matrix(
            object[["instructions"]][["frm"]],
            data = data
          )[,
            -1,
            drop = FALSE
          ]
        } else {
          # Should be the last option here: df without formula
          data <- make_safe_matrix(data, object[["instructions"]])
        }
      }
      # Now rescale
      get_means <- object[["flexibility"]][["rescale"]][["means"]]
      get_sds <- object[["flexibility"]][["rescale"]][["sds"]]
      data <- sweep(data, 2, get_means, "-")
      data <- sweep(data, 2, get_sds, "/")

      # Can now output

      preds_mat <- matrix(
        stats::predict(
          object[["model"]],
          new_data = data,
          type = "response",
          newoffset = get_offset
        ),
        ncol = 100
      )

      preds_list <- asplit(preds_mat, 2)
      names(preds_list) <- as.character(seq_len(ncol(preds_mat)))
      return(preds_list)
    },
    family,
    offset = NULL,
    frm = NULL,
    max_degree = 2,
    smoothness_orders = 1,
    num_knots = 50
  )(
    name = name,
    family = family,
    offset = offset,
    frm = frm,
    max_degree = max_degree,
    smoothness_orders = smoothness_orders,
    num_knots = num_knots
  )
}


#' Elastic net / LASSO regularization path learner
#'
#' Fits an entire regularization path via \code{glmnet::glmnet} and returns
#' predictions for each \eqn{\lambda} value as a named list. Because it
#' returns multiple outputs it is an \code{enfold_list} learner — each
#' \eqn{\lambda} column is treated as an independent learner by the
#' metalearner. Predictors are internally centered and scaled. Requires the
#' \pkg{glmnet} package.
#'
#' @param name Character. Name for this learner instance.
#' @param family A family object or \code{"auto"}.
#' @param offset Character or \code{NULL}. Column name in \code{x} to use
#'   as an offset.
#' @param frm Formula or \code{NULL}. Passed to \code{model.matrix()} for
#'   custom basis expansion.
#' @param alpha Numeric in \eqn{[0, 1]}. Elastic net mixing parameter.
#'   \code{1} (default) = LASSO, \code{0} = ridge.
#' @param lambda Numeric vector of \eqn{\lambda} values for the regularization
#'   path, or an \code{enfold_lambda_request} produced by
#'   \code{\link{make_lambda_sequence}}. Using \code{make_lambda_sequence}
#'   ensures the same lambda grid is used across all cross-validation folds.
#' @return An \code{enfold_list} learner (a subclass of
#'   \code{enfold_learner}) whose \code{predict()} method returns a named
#'   list with one entry per \eqn{\lambda}.
#' @seealso \code{\link{lrn_bigGlm}}, \code{\link{lrn_hal}},
#'   \code{\link{make_lambda_sequence}}
#' @examples
#' \dontrun{
#' x <- as.matrix(mtcars[, -1]); y <- mtcars$mpg
#' lrn <- lrn_glmnet("en", family = gaussian(), alpha = 0.5,
#'   lambda = make_lambda_sequence(x, y, nlambda = 50L))
#' fitted <- fit(lrn, x = x, y = y)
#' preds <- predict(fitted, newdata = x)
#' length(preds)  # 50 entries, one per lambda
#' }
#' @export
lrn_glmnet <- function(
  name,
  family,
  offset = NULL,
  frm = NULL,
  alpha = 1,
  lambda
) {
  .msg_pkg("glmnet")

  if (missing(family)) {
    stop(
      "Argument 'family' is missing. You need to specify a family object ",
      "(e.g., gaussian()), or use family = 'auto' to let the learner guess.",
      call. = FALSE
    )
  }

  if (inherits(lambda, "enfold_lambda_request")) {
    lambda <- .resolve_lambda_request(lambda, family = family, alpha = alpha, frm = frm, offset = offset)
  }

  make_learner_factory(
    fit = function(x, y) {
      .check_pkg("glmnet")
      if (identical(family, "auto")) {
        family <- family_guesser(y)
      }
      if (!is.null(offset)) {
        get_offset <- x[, offset]
        if (!is.numeric(get_offset)) {
          stop(
            "Called from lrn_glmnet: please ensure that the offset you provide is numeric."
          )
        }
        x <- x[, -which(colnames(x) == offset)]
      } else {
        get_offset <- NULL
      }
      if (is.data.frame(x) & is.null(frm)) {
        instr_list <- create_instruction_list(x)
        x <- make_safe_matrix(x, instr_list)
      } else if (is.data.frame(x) & !is.null(frm)) {
        instr_list <- list(coerce_df = FALSE, frm = frm)
        x <- stats::model.matrix(frm, data = x)[, -1, drop = FALSE]
      } else if (is.matrix(x) & is.null(frm)) {
        instr_list <- "skip"
      } else if (is.matrix(x) & !is.null(frm)) {
        instr_list <- list(coerce_df = TRUE, frm = frm)
        x <- stats::model.matrix(frm, data = data.frame(x))[, -1, drop = FALSE]
        warning(
          "Formula provided for matrix in glmnet.\nCoerced to data frame for call to model.matrix(); can be (silently) very unsafe!"
        )
      } else {
        stop("Unexpected data input in glmnet! Should not happen.")
      }
      get_sds <- apply(x, 2, sd)
      rescale_list <- list(
        means = apply(x, 2, mean),
        sds = ifelse(get_sds == 0, 1, get_sds)
      )
      x <- sweep(x, 2, rescale_list[["means"]], "-")
      x <- sweep(x, 2, rescale_list[["sds"]], "/")
      old_controls <- glmnet::glmnet.control()
      glmnet::glmnet.control(fdev = 0)
      get_model <- glmnet::glmnet(
        x,
        y,
        family = family,
        alpha = alpha,
        offset = get_offset,
        lambda = lambda,
        standardize = FALSE
      )
      do.call(glmnet::glmnet.control, old_controls)

      # Strip call; expensive to store
      get_model[["call"]] <- NULL
      return(list(
        model = get_model,
        instructions = instr_list,
        rescale = rescale_list
      ))
    },
    preds = function(object, data) {
      .check_pkg("glmnet")
      if (!is.null(offset)) {
        get_offset <- data[, offset]
        if (!is.numeric(get_offset)) {
          stop(
            "Called from lrn_glmnet: please ensure that the offset you provide is numeric."
          )
        }
        data <- data[, -which(colnames(data) == offset)]
      } else {
        get_offset <- NULL
      }
      if (!(identical(object[["instructions"]], "skip"))) {
        if (is.matrix(data)) {
          data <- stats::model.matrix(
            object[["instructions"]][["frm"]],
            data = data.frame(data)
          )[, -1, drop = FALSE]
        } else if (!is.null(object[["instructions"]][["frm"]])) {
          data <- stats::model.matrix(
            object[["instructions"]][["frm"]],
            data = data
          )[, -1, drop = FALSE]
        } else {
          data <- make_safe_matrix(data, object[["instructions"]])
        }
      }
      data <- sweep(data, 2, object[["rescale"]][["means"]], "-")
      data <- sweep(data, 2, object[["rescale"]][["sds"]], "/")

      pred_mat <- stats::predict(
        object[["model"]],
        newx = data,
        newoffset = get_offset,
        type = "link"
      )
      preds_list <- asplit(pred_mat, 2)
      names(preds_list) <- as.character(object[["model"]]$lambda)
      return(preds_list)
    },
    family,
    offset = NULL,
    frm = NULL,
    alpha = 1,
    lambda,
    expect_list = TRUE
  )(
    name = name,
    family = family,
    offset = offset,
    frm = frm,
    alpha = alpha,
    lambda = lambda
  )
}


#' Multivariate adaptive regression splines (MARS) learner
#'
#' Fits a MARS model via \code{earth::earth}. Supports all GLM families.
#' Requires the \pkg{earth} package.
#'
#' @param name Character. Name for this learner instance.
#' @param family A family object, \code{"auto"}, or \code{NULL} for a plain
#'   (non-GLM) MARS model.
#' @param offset Character or \code{NULL}. Column name in \code{x} to use
#'   as an offset.
#' @param degree Integer. Maximum interaction degree. \code{1} = additive
#'   (no interactions). Default \code{2}.
#' @param penalty Numeric. GCV penalty per knot. Default \code{3}.
#' @param nk Integer. Maximum number of knots before pruning. Default
#'   \code{100}.
#' @param thresh Numeric. Forward pass stopping threshold. Default \code{0.01}.
#' @param pmethod Character. Pruning method passed to \code{earth}. Default
#'   \code{"backward"}.
#' @param nfold Integer. Number of cross-validation folds internal to
#'   \code{earth} (\code{0} disables internal CV). Default \code{0}.
#' @return An \code{enfold_learner} object.
#' @seealso \code{\link{lrn_gam}}, \code{\link{lrn_glm}}
#' @examples
#' \dontrun{
#' lrn <- lrn_earth("mars", family = gaussian(), degree = 2)
#' fitted <- fit(lrn, x = mtcars[, -1], y = mtcars$mpg)
#' head(predict(fitted, newdata = mtcars[, -1]))
#' }
#' @export
lrn_earth <- function(
  name,
  family,
  offset = NULL,
  degree = 2,
  penalty = 3,
  nk = 100,
  thresh = 0.01,
  pmethod = "backward",
  nfold = 0
) {
  .msg_pkg("earth")

  if (missing(family)) {
    stop(
      "Argument 'family' is missing. You need to specify a family object ",
      "(e.g., gaussian()), or use family = 'auto' to let the learner guess. For 'earth', family can be NULL for a normal model.",
      call. = FALSE
    )
  }

  # Some checks ensuring there are integers where necessary
  integer_checker(degree, "the interaction degree (1 = additive model).")
  integer_checker(penalty, "the GCV penalty.")
  integer_checker(nk, "the number of knots.")
  integer_checker(
    nfold,
    "the number of cv folds (0 = no cv).",
    require_positive = FALSE
  )

  make_learner_factory(
    fit = function(x, y) {
      .check_pkg("earth")
      if (identical(family, "auto")) {
        family <- family_guesser(y)
      }
      # Very similar to how I did it for GLMs, actually!
      if (is.data.frame(x)) {
        instr_list <- create_instruction_list(x)
        x <- make_safe_matrix(x, instr_list)
      } else {
        instr_list <- "skip"
        x <- as.matrix(x)
        if (is.null(colnames(x))) colnames(x) <- paste0("X", seq_len(ncol(x)))
      }
      fd <- data.frame(y = y, x)
      # Formula, maybe with offset
      if (!is.null(offset)) {
        ofs_char <- paste("offset(", offset, ")", sep = "")
      } else {
        ofs_char <- character(0)
      }
      offset_col <- which(colnames(fd) == offset)
      get_frm <- stats::formula(paste0(
        "y ~ ",
        paste(c(ofs_char, colnames(fd)[-c(1, offset_col)]), collapse = " + ")
      ))
      # Can now fit!
      # Some logic to ensure that family is only handed in if necessary
      if (!is.null(family)) {
        get_fm <- list(family = family)
      } else {
        get_fm <- NULL
      }
      get_model <- earth::earth(
        formula = get_frm,
        data = fd,
        pmethod = pmethod,
        degree = degree,
        nfold = nfold,
        thresh = thresh,
        penalty = penalty,
        nk = nk,
        glm = get_fm
      )
      return(list(
        model = get_model,
        instructions = instr_list,
        col_names = colnames(x)
      ))
    },
    preds = function(object, data) {
      .check_pkg("earth")
      # Apply instruction list if necessary
      if (!identical(object[["instructions"]], "skip")) {
        data <- make_safe_matrix(data, object[["instructions"]])
      } else {
        data <- as.matrix(data)
        colnames(data) <- object[["col_names"]]
      }
      # Then need to coerce to data frame either way
      data <- data.frame(data)
      # Then output
      return(as.vector(stats::predict(
        object[["model"]],
        newdata = data,
        type = "response"
      )))
    },
    family,
    offset = NULL,
    degree = 2,
    penalty = 3,
    nk = 100,
    thresh = 0.01,
    pmethod = "backward",
    nfold = 0
  )(
    name = name,
    family = family,
    offset = offset,
    degree = degree,
    penalty = penalty,
    nk = nk,
    thresh = thresh,
    pmethod = pmethod,
    nfold = nfold
  )
}


#' Big GLM learner (glmnet back-end)
#'
#' Fits a single-penalty GLM using \code{glmnet::bigGlm}, which is more
#' numerically stable than \code{stats::glm} for high-dimensional or
#' ill-conditioned designs. Predictors are internally standardized. Requires
#' the \pkg{glmnet} package.
#'
#' @param name Character. Name for this learner instance.
#' @param family A family object or \code{"auto"}.
#' @param offset Character or \code{NULL}. Column name in \code{x} to use
#'   as an offset.
#' @param frm Formula or \code{NULL}. Passed to \code{model.matrix()} for
#'   custom basis expansion.
#' @return An \code{enfold_learner} object.
#' @seealso \code{\link{lrn_glm}}, \code{\link{lrn_glmnet}}
#' @examples
#' \dontrun{
#' lrn <- lrn_bigGlm("big_glm", family = gaussian())
#' fitted <- fit(lrn, x = as.matrix(mtcars[, -1]), y = mtcars$mpg)
#' head(predict(fitted, newdata = as.matrix(mtcars[, -1])))
#' }
#' @export
lrn_bigGlm <- function(name, family, offset = NULL, frm = NULL) {
  .msg_pkg("glmnet")

  if (missing(family)) {
    stop(
      "Argument 'family' is missing. You need to specify a family object ",
      "(e.g., gaussian()), or use family = 'auto' to let the learner guess.",
      call. = FALSE
    )
  }

  make_learner_factory(
    fit = function(x, y) {
      .check_pkg("glmnet")
      if (identical(family, "auto")) {
        family <- family_guesser(y)
      }
      # If there is an offset, retrieve as variable and remove from data
      if (!is.null(offset)) {
        get_offset <- x[, offset]
        if (!is.numeric(get_offset)) {
          stop(
            "Called from lrn_bigGlm: please ensure that the offset you provide is numeric."
          )
        }
        x <- x[, -which(colnames(x) == offset)]
      } else {
        get_offset <- NULL
      }
      # Make a safe matrix if necessary and no formula provided
      if (is.data.frame(x) & is.null(frm)) {
        instr_list <- create_instruction_list(x)
        x <- make_safe_matrix(x, instr_list)
      } else if (is.data.frame(x) & !is.null(frm)) {
        # If frm provided, use in model.matrix call
        instr_list <- list(coerce_df = FALSE, frm = frm)
        x <- stats::model.matrix(frm, data = x)[, -1, drop = FALSE]
      } else if (is.matrix(x) & is.null(frm)) {
        # Simplest case: take x as is
        instr_list <- "skip"
      } else if (is.matrix(x) & !is.null(frm)) {
        # If formula provided for matrix, coerce to df and warn
        instr_list <- list(coerce_df = TRUE, frm = frm)
        x <- stats::model.matrix(frm, data = data.frame(x))[, -1, drop = FALSE]
        warning(
          "Formula provided for matrix in bigGlm.\nCoerced to data frame for call to model.matrix(); can be (silently) very unsafe!"
        )
      } else {
        stop("Unexpected data input in bigGlm! Should not happen.")
      }
      flex_list <- list()
      # Normalize; for constant columns set sd to 1
      get_sds <- apply(x, 2, sd)
      flex_list[["rescale"]] <- list(
        means = apply(x, 2, mean),
        sds = ifelse(get_sds == 0, 1, get_sds)
      )
      x <- sweep(x, 2, flex_list[["rescale"]][["means"]], "-")
      x <- sweep(x, 2, flex_list[["rescale"]][["sds"]], "/")
      # Can now fit!
      get_model <- glmnet::bigGlm(
        x,
        y,
        family = family,
        offset = get_offset,
        standardize = FALSE
      )
      return(list(
        model = get_model,
        instructions = instr_list,
        flexibility = flex_list
      ))
    },
    preds = function(object, data) {
      .check_pkg("glmnet")
      # If there is an offset, retrieve as variable and remove from data
      if (!is.null(offset)) {
        get_offset <- data[, offset]
        if (!is.numeric(get_offset)) {
          stop(
            "Called from lrn_bigGlm: please ensure that the offset you provide is numeric."
          )
        }
        data <- data[, -which(colnames(data) == offset)]
      } else {
        get_offset <- NULL
      }
      # Apply instruction list if necessary
      if (!(identical(object[["instructions"]], "skip"))) {
        # Only need logic if data not already right
        # If matrix, need to coerce to df again for model matrix call
        if (is.matrix(data)) {
          data <- stats::model.matrix(
            object[["instructions"]][["frm"]],
            data = data.frame(data)
          )[, -1, drop = FALSE]
        } else if (!is.null(object[["instructions"]][["frm"]])) {
          data <- stats::model.matrix(
            object[["instructions"]][["frm"]],
            data = data
          )[,
            -1,
            drop = FALSE
          ]
        } else {
          # Should be the last option here: df without formula
          data <- make_safe_matrix(data, object[["instructions"]])
        }
      }
      # Now need to match scaling
      get_means <- object[["flexibility"]][["rescale"]][["means"]]
      get_sds <- object[["flexibility"]][["rescale"]][["sds"]]
      data <- sweep(data, 2, get_means, "-")
      data <- sweep(data, 2, get_sds, "/")
      return(as.vector(stats::predict(
        object[["model"]],
        newx = data,
        type = "response",
        newoffset = get_offset
      )))
    },
    family,
    offset = NULL,
    frm = NULL
  )(name = name, family = family, offset = offset, frm = frm)
}


#' Random forest learner (ranger back-end)
#'
#' Fits a random forest via \code{ranger::ranger}. Requires the \pkg{ranger}
#' package.
#'
#' @param name Character. Name for this learner instance.
#' @param select_vars Character vector or \code{NULL}. If provided, only these
#'   columns of \code{x} are used.
#' @param num.trees Integer. Number of trees. Default \code{500}.
#' @param mtry Integer or \code{NULL}. Number of variables to try at each
#'   split; \code{NULL} uses the ranger default.
#' @param probability Logical. If \code{TRUE}, fits a probability forest
#'   (suitable for binary outcomes). Default \code{FALSE}.
#' @param min.node.size Integer or \code{NULL}. Minimum node size.
#' @param min.bucket Integer or \code{NULL}. Minimum bucket size (terminal
#'   node observations).
#' @param max.depth Integer or \code{NULL}. Maximum tree depth.
#' @param splitrule Character or \code{NULL}. Splitting rule passed to
#'   \code{ranger}.
#' @return An \code{enfold_learner} object.
#' @seealso \code{\link{lrn_gam}}, \code{\link{grd_random}}
#' @examples
#' \dontrun{
#' lrn <- lrn_ranger("rf", num.trees = 200L)
#' fitted <- fit(lrn, x = mtcars[, -1], y = mtcars$mpg)
#' head(predict(fitted, newdata = mtcars[, -1]))
#' }
#' @export
lrn_ranger <- function(
  name,
  select_vars = NULL,
  num.trees = 500,
  mtry = NULL,
  probability = FALSE,
  min.node.size = NULL,
  min.bucket = NULL,
  max.depth = NULL,
  splitrule = NULL
) {
  return(
    make_learner_factory(
      fit = function(x, y) {
        # First, if select_vars is provided, subset x
        if (!is.null(select_vars)) {
          x <- x[, select_vars]
        }

        # Make a safe matrix if necessary
        if (is.data.frame(x)) {
          instr_list <- create_instruction_list(x)
          x <- make_safe_matrix(x, instr_list)
        } else {
          instr_list <- "skip"
        }

        # Need a data frame for ranger
        x_df <- data.frame(x)

        # Can now fit!
        return(list(
          model = ranger::ranger(
            x = x_df,
            y = y,
            num.trees = num.trees,
            mtry = mtry,
            probability = probability,
            min.node.size = min.node.size,
            min.bucket = min.bucket,
            max.depth = max.depth,
            splitrule = splitrule
          ),
          instructions = instr_list
        ))
      },
      preds = function(object, data) {
        # Apply instruction list if necessary
        if (!identical(object[["instructions"]], "skip")) {
          data <- make_safe_matrix(data, object[["instructions"]])
        }

        # Then need to coerce to data frame either way
        data <- data.frame(data)

        # Then output
        return(as.vector(
          stats::predict(
            object[["model"]],
            data = data,
            type = "response"
          )$predictions
        ))
      },
      select_vars = NULL,
      num.trees = 500,
      mtry = NULL,
      probability = FALSE,
      min.node.size = NULL,
      min.bucket = NULL,
      max.depth = NULL,
      splitrule = NULL
    )(
      name = name,
      select_vars = select_vars,
      num.trees = num.trees,
      mtry = mtry,
      probability = probability,
      min.node.size = min.node.size,
      min.bucket = min.bucket,
      max.depth = max.depth,
      splitrule = splitrule
    )
  )
}


#' XGBoost learner with optional snapshotting
#'
#' Trains a gradient boosted tree model via \code{xgboost::xgb.train} and
#' returns one prediction column per snapshot. Snapshots are taken at evenly-spaced boosting rounds
#' (the final snapshot is always at \code{nrounds}), so a downstream
#' metalearner can weight the rounds optimally.
#'
#' @param name Character scalar. Name of the learner.
#' @param select_vars Column indices or names to subset \code{x} before
#'   fitting, or \code{NULL} (default) to use all columns.
#' @param nrounds Positive integer. Total number of boosting rounds.
#' @param n_snapshots Positive integer. Number of evenly-spaced rounds at
#' which a prediction column is generated.  Default is \code{NULL}, which means no snapshotting.
#' n_snapshots == 1 is treated the same as no snapshotting.
#' @param max_depth Positive integer. Maximum tree depth.
#' @param eta Numeric. Learning rate (step-size shrinkage).
#' @param objective Character. XGBoost objective string, e.g.
#'   \code{"reg:squarederror"} (default), \code{"binary:logistic"}.
#' @param nthread Positive integer. Number of parallel threads.
#' @param verbose Integer. Verbosity level passed to \code{xgb.train}:
#'   \code{0} = silent (default).
#' @param lambda Numeric. L2 regularisation weight on leaf weights.
#' @param alpha Numeric. L1 regularisation weight on leaf weights. Default \code{0}.
#' @param min_child_weight Numeric. Minimum sum of instance weight in a leaf.
#' @return An \code{enfold_list} whose \code{predict()} method returns a named
#'   list with one numeric vector per snapshot round.
#' @seealso \code{\link{lrn_ranger}}, \code{\link{lrn_glmnet}}
#' @examples
#' \dontrun{
#' lrn <- lrn_xgboost("xgb", nrounds = 100L, n_snapshots = 5L)
#' fitted <- fit(lrn, x = as.matrix(mtcars[, -1]), y = mtcars$mpg)
#' preds <- predict(fitted, newdata = as.matrix(mtcars[, -1]))
#' length(preds)  # 5 snapshots
#' }
#' @export
lrn_xgboost <- function(
  name,
  select_vars = NULL,
  nrounds = 200L,
  n_snapshots = NULL,
  max_depth = 3L,
  eta = 0.1,
  objective = "reg:squarederror",
  nthread = 1L,
  verbose = 0L,
  lambda = 2,
  alpha = 0,
  min_child_weight = 5
) {
  integer_checker(nrounds, "nrounds")
  integer_checker(n_snapshots, "n_snapshots")

  is_enfold_list <- if (!is.null(n_snapshots)) TRUE else FALSE

  # expect_list is TRUE only if n_snapshots > 1
  if (!is.null(n_snapshots)) {
    if (n_snapshots == 1) n_snapshots <- NULL
  }

  make_learner_factory(
    fit = function(x, y) {
      if (!is.null(dim(y))) {
        y <- y[, 1L]
      }
      if (!is.null(select_vars)) {
        x <- x[, select_vars, drop = FALSE]
      }

      if (is.data.frame(x)) {
        instr_list <- create_instruction_list(x)
        x <- make_safe_matrix(x, instr_list)
      } else {
        instr_list <- "skip"
        x <- as.matrix(x)
      }

      dtrain <- xgboost::xgb.DMatrix(data = x, label = y)

      model <- xgboost::xgb.train(
        params = list(
          max_depth = max_depth,
          eta = eta,
          objective = objective,
          nthread = nthread,
          lambda = lambda,
          alpha = alpha,
          min_child_weight = min_child_weight
        ),
        data = dtrain,
        nrounds = nrounds,
        verbose = verbose
      )

      # If no snapshots just output
      if (is.null(n_snapshots)) {
        return(list(model = model, instructions = instr_list))
      }

      # Otherwise evenly-spaced snapshot rounds; duplicates removed (can arise when n_snapshots > nrounds)
      snap_rounds <- unique(pmax(
        1L,
        as.integer(round(
          seq_len(n_snapshots) * nrounds / n_snapshots
        ))
      ))

      list(model = model, snap_rounds = snap_rounds, instructions = instr_list)
    },
    preds = function(object, data) {
      # No snapshot case
      if (is.null(object[["snap_rounds"]])) {
        if (!is.null(select_vars)) {
          data <- data[, select_vars, drop = FALSE]
        }

        if (!identical(object$instructions, "skip")) {
          data <- make_safe_matrix(data, object$instructions)
        } else {
          data <- as.matrix(data)
        }

        dtest <- xgboost::xgb.DMatrix(data = data)

        return(as.vector(stats::predict(object$model, newdata = dtest)))
      }

      # Else this branch is taken
      if (!is.null(select_vars)) {
        data <- data[, select_vars, drop = FALSE]
      }

      if (!identical(object$instructions, "skip")) {
        data <- make_safe_matrix(data, object$instructions)
      } else {
        data <- as.matrix(data)
      }

      dtest <- xgboost::xgb.DMatrix(data = data)

      preds_list <- lapply(object$snap_rounds, function(sr) {
        as.vector(stats::predict(
          object$model,
          newdata = dtest,
          iterationrange = c(1L, as.integer(sr))
        ))
      })
      names(preds_list) <- as.character(object$snap_rounds)
      preds_list
    },
    select_vars = NULL,
    nrounds = 200L,
    n_snapshots = 5L,
    max_depth = 3L,
    eta = 0.1,
    objective = "reg:squarederror",
    nthread = 1L,
    verbose = 0L,
    alpha = 0,
    lambda = 2,
    min_child_weight = 5,
    expect_list = is_enfold_list
  )(
    name = name,
    select_vars = select_vars,
    nrounds = nrounds,
    n_snapshots = n_snapshots,
    max_depth = max_depth,
    eta = eta,
    objective = objective,
    nthread = nthread,
    verbose = verbose,
    lambda = lambda,
    alpha = alpha,
    min_child_weight = min_child_weight
  )
}


### HELPERS
# Family guesser so that family can be auto
family_guesser <- function(y) {
  unique_vals <- unique(y)

  # Binary (Strictly 0 and 1)
  if (length(unique_vals) == 2 && all(unique_vals %in% c(0, 1))) {
    return(stats::binomial())
  }

  # Proportions (between 0 and 1, but not strictly binary)
  if (all(y >= 0 & y <= 1)) {
    return(stats::quasibinomial())
  }

  # 3. Counts (non-negative Integers)
  if (all(y >= 0) && all(y == floor(y))) {
    return(stats::poisson())
  }

  # Anything else
  return(stats::gaussian())
}
