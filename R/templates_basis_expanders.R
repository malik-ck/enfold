#' B-spline or truncated power spline basis expander
#'
#' Expands every column of the predictor matrix into a spline basis, then
#' concatenates the resulting columns.  The knot sequences chosen on the
#' training data are stored and replayed at prediction time, so the basis is
#' fully consistent across folds.
#'
#' @param name Character scalar. Name of the expander.
#' @param type One of \code{"bs"} (B-splines via \code{splines::bs}, the
#'   default) or \code{"tp"} (truncated power splines, which combine well with ridge penalties).
#' @param degree Non-negative integer. Polynomial degree of the spline. Default \code{3} gives cubic splines.
#' @param max_knots Positive integer. Maximum number of interior knots per
#'   column. For columns with fewer than \code{max_knots + 2} unique values
#'   the actual number of knots is capped automatically so the basis remains
#'   identifiable.
#' @return An \code{enfold_learner} whose \code{predict()} method returns a
#'   numeric matrix of spline basis columns, one block per input column.
#' @details
#' All \code{bex_*} templates return an \code{enfold_learner} whose
#' \code{predict()} method returns a transformed feature matrix. They are
#' designed as non-terminal stages in a \code{\link{make_pipeline}}, feeding
#' expanded features into downstream learners. Example:
#'
#' \preformatted{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' pl <- make_pipeline(
#'   bex_splines("splines", degree = 3, max_knots = 5),
#'   lrn_glm("glm", family = gaussian())
#' )
#' task <- initialize_enfold(x, y) |>
#'   add_learners(pl) |>
#'   add_cv_folds(inner_cv = NA, outer_cv = 3L) |>
#'   fit()
#' }
#'
#' @seealso \code{\link{bex_interactions}}, \code{\link{bex_formula}},
#'   \code{\link{make_pipeline}}
#' @examples
#' bex <- bex_splines("splines", degree = 3, max_knots = 5)
#' fitted <- fit(bex, x = as.matrix(mtcars[, -1]), y = mtcars$mpg)
#' dim(predict(fitted, newdata = as.matrix(mtcars[, -1])))
#' @export
bex_splines <- function(
  name,
  type = c("bs", "tp"),
  degree = 3,
  max_knots = 10
) {
  type <- match.arg(type)
  integer_checker(degree, "degree", require_positive = FALSE)
  if (degree < 0L) {
    stop("'degree' must be non-negative.", call. = FALSE)
  }
  integer_checker(max_knots, "max_knots")

  make_learner_factory(
    fit = function(x, y) {
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
      col_nms <- colnames(x_mat)
      p <- ncol(x_mat)

      col_info <- lapply(seq_len(p), function(j) {
        col <- x_mat[, j]
        n_unique <- length(unique(col))
        actual_knots <- max(0L, min(as.integer(max_knots), n_unique - 2L))

        if (type == "bs") {
          df_val <- max(1L, actual_knots + as.integer(degree))
          basis_obj <- splines::bs(
            col,
            degree = degree,
            df = df_val,
            intercept = FALSE
          )
          list(type = "bs", basis_obj = basis_obj, n_cols = ncol(basis_obj))
        } else {
          # Compute knot sequence outside tps to avoid integer_checker blocking 0 knots
          knot_seq <- if (actual_knots > 0L) {
            unname(stats::quantile(
              col,
              seq(0, 1, length.out = actual_knots + 2L)
            )[-c(1L, actual_knots + 2L)])
          } else {
            numeric(0)
          }
          basis_mat <- tps(
            col,
            num_knots = NULL,
            knot_seq = knot_seq,
            degree = degree,
            intercept = FALSE
          )
          list(
            type = "tp",
            knot_seq = knot_seq,
            degree = degree,
            n_cols = ncol(basis_mat)
          )
        }
      })

      out_col_nms <- unlist(lapply(seq_len(p), function(j) {
        paste0(col_nms[j], "_s", seq_len(col_info[[j]]$n_cols))
      }))

      list(
        col_info = col_info,
        out_col_nms = out_col_nms,
        instr_list = instr_list
      )
    },
    preds = function(object, data) {
      if (!identical(object$instr_list, "skip")) {
        data <- make_safe_matrix(data, object$instr_list)
      } else {
        data <- as.matrix(data)
      }

      basis_list <- lapply(seq_along(object$col_info), function(j) {
        info <- object$col_info[[j]]
        col <- data[, j]
        if (info$type == "bs") {
          stats::predict(info$basis_obj, newx = col)
        } else {
          tps(
            col,
            num_knots = NULL,
            knot_seq = info$knot_seq,
            degree = info$degree,
            intercept = FALSE
          )
        }
      })

      out <- do.call(cbind, basis_list)
      colnames(out) <- object$out_col_nms
      out
    },
    type = "bs",
    degree = 3,
    max_knots = 10
  )(name = name, type = type, degree = degree, max_knots = max_knots)
}


#' Interaction basis expander
#'
#' Expands the predictor matrix to include all pairwise (or higher-order) interaction terms up to the specified depth.
#' The column set produced on the training data is recorded and enforced at prediction time.
#'
#' @param name Character scalar. Name of the expander.
#' @param depth Positive integer. Maximum interaction order.  \code{2}
#'   (default) adds all pairwise products; \code{3} additionally adds all
#'   triple products, and so on.  For most SuperLearner use cases \code{2}
#'   is plenty.
#' @return An \code{enfold_learner} whose \code{predict()} method returns a
#'   numeric matrix containing main effects and all interaction columns.
#' @seealso \code{\link{bex_splines}}, \code{\link{bex_formula}},
#'   \code{\link{make_pipeline}}
#' @examples
#' bex <- bex_interactions("inter2", depth = 2L)
#' fitted <- fit(bex, x = as.matrix(mtcars[, -1]), y = mtcars$mpg)
#' dim(predict(fitted, newdata = as.matrix(mtcars[, -1])))
#' @export
bex_interactions <- function(name, depth = 2L) {
  integer_checker(depth, "depth")

  make_learner_factory(
    fit = function(x, y = NULL) {
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

      frm <- stats::as.formula(paste0("~ .^", depth, " - 1"))
      mm <- stats::model.matrix(frm, data = data.frame(x_mat))

      list(col_nms = colnames(mm), instr_list = instr_list)
    },
    preds = function(object, data) {
      if (!identical(object$instr_list, "skip")) {
        data <- make_safe_matrix(data, object$instr_list)
      } else {
        data <- as.matrix(data)
        if (is.null(colnames(data))) {
          colnames(data) <- paste0("V", seq_len(ncol(data)))
        }
      }

      frm <- stats::as.formula(paste0("~ .^", depth, " - 1"))
      mm <- stats::model.matrix(frm, data = data.frame(data))
      mm[, object$col_nms, drop = FALSE]
    },
    depth = 2L
  )(name = name, depth = depth)
}


#' Random forest tree-prediction basis expander
#'
#' Trains a \pkg{ranger} random forest and expands the predictor matrix by
#' replacing it with the individual tree predictions: each tree contributes one
#' column. A subsequent learner can then weight the trees
#' optimally. This is sometimes called \emph{random forest stacking} or
#' \emph{forest embedding}.
#'
#' @param name Character scalar. Name of the expander.
#' @param num.trees Positive integer. Number of trees. Default \code{100}.
#' @param mtry Positive integer or \code{NULL}. Number of variables to
#'   consider at each split; \code{NULL} lets \pkg{ranger} use its default
#'   (floor of sqrt of number of variables for classification, one third for
#'   regression).
#' @param min.node.size Positive integer or \code{NULL}. Minimum node size;
#'   \code{NULL} uses the ranger default.
#' @return An \code{enfold_learner} whose \code{predict()} method returns an
#'   \eqn{n \times \text{num.trees}} numeric matrix of individual tree
#'   predictions.
#' @seealso \code{\link{bex_splines}}, \code{\link{make_pipeline}}
#' @examples
#' \dontrun{
#' bex <- bex_ranger("rf_emb", num.trees = 50L)
#' fitted <- fit(bex, x = mtcars[, -1], y = mtcars$mpg)
#' dim(predict(fitted, newdata = mtcars[, -1]))  # n x 50
#' }
#' @export
bex_ranger <- function(
  name,
  num.trees = 100L,
  mtry = NULL,
  min.node.size = NULL
) {
  .msg_pkg("ranger")

  integer_checker(num.trees, "num.trees")

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
      model <- ranger::ranger(
        x = data.frame(x_mat),
        y = y,
        num.trees = num.trees,
        mtry = mtry,
        min.node.size = min.node.size
      )
      list(model = model, instr_list = instr_list)
    },
    preds = function(object, data) {
      .check_pkg("ranger")
      if (!identical(object$instr_list, "skip")) {
        data <- make_safe_matrix(data, object$instr_list)
      } else {
        data <- as.matrix(data)
        if (is.null(colnames(data))) {
          colnames(data) <- paste0("V", seq_len(ncol(data)))
        }
      }
      preds_mat <- stats::predict(
        object$model,
        data = data.frame(data),
        predict.all = TRUE
      )$predictions
      colnames(preds_mat) <- paste0("tree_", seq_len(ncol(preds_mat)))
      preds_mat
    },
    num.trees = 100L,
    mtry = NULL,
    min.node.size = NULL
  )(
    name = name,
    num.trees = num.trees,
    mtry = mtry,
    min.node.size = min.node.size
  )
}


#' Formula-based basis expander
#'
#' Uses \code{model.matrix()} to expand the predictor matrix according to a user-specified formula
#'
#' @param name Character scalar. Name of the expander.
#' @param formula A formula object specifying the desired basis expansion.
#' The formula should use \code{.} to represent the predictor matrix and should not include a response variable.
#' For example, \code{~ . + I(.^2) - 1} would add squared terms for each predictor without an intercept, while \code{~ . + age^2 + age*sex} would add squared and interaction terms for an \code{age} variable and a \code{sex} variable.
#' @return An \code{enfold_learner} whose \code{predict()} method returns the
#'   numeric model matrix produced by applying \code{formula} to the new data.
#' @details
#' Include \code{-1} in the formula to suppress the intercept column, which
#' is often needed for downstream learners that add their own intercept.
#' @seealso \code{\link{bex_splines}}, \code{\link{bex_interactions}},
#'   \code{\link{make_pipeline}}
#' @examples
#' bex <- bex_formula("poly2", formula = ~ . + I(hp^2) + I(wt^2) - 1)
#' fitted <- fit(bex, x = mtcars[, -1], y = mtcars$mpg)
#' dim(predict(fitted, newdata = mtcars[, -1]))
#' @export
bex_formula <- function(name, formula) {
  if (!inherits(formula, "formula")) {
    stop("'formula' must be a formula object.", call. = FALSE)
  }

  make_learner_factory(
    fit = function(x, y) {
      # Make x to safe matrix if it is a data frame, otherwise just ensure it's a matrix and has column names
      if (is.data.frame(x)) {
        instr_list <- create_instruction_list(x)
        x_mat <- make_safe_matrix(x, instr_list)
      } else {
        instr_list <- "skip"
        x_mat <- as.matrix(x)
      }

      # If no column names, add them
      if (is.null(colnames(x_mat))) {
        colnames(x_mat) <- paste0("V", seq_len(ncol(x_mat)))
      }

      data_for_mm <- data.frame(x_mat)
      mm <- stats::model.matrix(formula, data = data_for_mm)

      list(col_nms = colnames(mm), instr_list = instr_list)
    },
    preds = function(object, data) {
      if (!identical(object$instr_list, "skip")) {
        data <- make_safe_matrix(data, object$instr_list)
      } else {
        data <- as.matrix(data)
        if (is.null(colnames(data))) {
          colnames(data) <- paste0("V", seq_len(ncol(data)))
        }
      }

      mm <- stats::model.matrix(formula, data = data.frame(data))
      mm[, object$col_nms, drop = FALSE]
    },
    formula = formula
  )(name = name, formula = formula)
}
