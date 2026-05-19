#' Initialize an enfold task
#'
#' Creates the base \code{enfold_task} object that holds predictor and outcome
#' data. Learners, metalearners, and CV folds are added afterwards with
#' \code{\link{add_learners}}, \code{\link{add_metalearners}}, and
#' \code{\link{add_cv_folds}}, respectively. The task is then fitted via
#' \code{\link{fit.enfold_task}}.
#'
#' @param x A data frame, matrix, \code{arrow::Table}, \code{FBM} (from
#'   \pkg{bigstatsr}), or a length-1 character path to a Feather / Arrow IPC
#'   file. Feather paths are opened as memory-mapped tables so that only the
#'   requested rows are read into memory inside each fold loop.
#' @param y A vector or matrix (or an object inheriting from either) of outcome
#'   values. Must have the same number of rows (or elements) as \code{x}.
#' @param cols Optional column specification to restrict predictors.  Can be a
#'   character vector of column names or an integer vector of column positions.
#'   When supplied, only these columns are used for fitting, prediction, and
#'   risk evaluation.  The default (\code{NULL}) uses all columns.
#' @return An object of class \code{enfold_task}.
#' @seealso \code{\link{add_learners}}, \code{\link{add_cv_folds}},
#'   \code{\link{fit.enfold_task}}
#' @export
#' @examples
#' x <- mtcars[, -1]
#' y <- mtcars$mpg
#' task <- initialize_enfold(x, y)
#' task
#'
initialize_enfold <- function(x, y, cols = NULL) {
  # Convert file path to an enfold_arrow_file reference (validates + caches nrow)
  if (is.character(x) && length(x) == 1L) {
    x <- new_arrow_file(x)
  }

  # Validate x type
  is_big_x <- inherits(x, c("ArrowTabular", "enfold_arrow_file", "FBM"))

  if (!is.matrix(x) && !is.data.frame(x) && !is_big_x) {
    stop(
      "`x` must be a data frame, matrix, Arrow Table, Arrow IPC file path, or FBM.",
      call. = FALSE
    )
  }

  if (is.matrix(x) && !is.numeric(x)) {
    stop("If `x` is a matrix, it must be numeric.", call. = FALSE)
  }

  if (!is.matrix(y) && !is.data.frame(y) && !is.vector(y)) {
    stop("`y` must be a vector, data frame, or matrix.")
  }

  y_len <- if (is.vector(y)) length(y) else nrow(y)

  if (nrow(x) != y_len) {
    stop("`x` and `y` must have the same number of observations.")
  }

  # Add data in proper slots
  x_env <- new.env(parent = emptyenv())
  x_env$x <- x
  lockEnvironment(x_env, bindings = TRUE)

  y_env <- new.env(parent = emptyenv())
  y_env$y <- y
  lockEnvironment(y_env, bindings = TRUE)

  # Validate and normalise cols
  if (!is.null(cols)) {
    if (length(cols) == 0L) {
      stop("`cols` must not be empty. Use NULL to select all columns.", call. = FALSE)
    }
    x_ncol <- ncol(x)
    if (is.character(cols)) {
      # Check names exist for matrix / data.frame (Arrow backends handled later)
      if (is.matrix(x) || is.data.frame(x)) {
        x_nms <- colnames(x)
        if (is.null(x_nms)) {
          stop("`x` has no column names; use integer `cols` instead.", call. = FALSE)
        }
        bad <- setdiff(cols, x_nms)
        if (length(bad) > 0L) {
          stop(
            "Column(s) not found in `x`: ",
            paste(sprintf("'%s'", bad), collapse = ", "),
            call. = FALSE
          )
        }
      }
    } else if (is.numeric(cols)) {
      cols <- as.integer(cols)
      if (any(cols < 1L | cols > x_ncol)) {
        stop(
          "Column indices in `cols` must be between 1 and ncol(x) = ",
          x_ncol, ".",
          call. = FALSE
        )
      }
      # Normalise integer indices to character names for matrix / data.frame
      if (is.matrix(x) || is.data.frame(x)) {
        x_nms <- colnames(x)
        if (!is.null(x_nms)) cols <- x_nms[cols]
      }
    } else {
      stop("`cols` must be a character or integer vector, or NULL.", call. = FALSE)
    }
  }

  # Get a starting list
  structure(
    list(
      x_env = x_env,
      y_env = y_env,
      cols = cols,
      # Also the ones initialized as NULL
      learners = NULL,
      metalearners = NULL,
      future_pkgs = detect_x_pkgs(x),
      cv = NULL,
      is_cv_ensemble = NULL,
      fit_objects = NULL,
      ensembles = NULL
    ),
    class = "enfold_task"
  )
}

# Add print method

#' @export
print.enfold_task <- function(x, ...) {

  cat("Enfold Task\n\n")
  cat("Data:\n")
  cat(sprintf("  Observations : %d\n", nrow(x$x_env$x)))
  if (!is.null(x$cols)) {
    cat(sprintf("  Predictors   : %d (of %d)\n", length(x$cols), ncol(x$x_env$x)))
  } else {
    cat(sprintf("  Predictors   : %d\n", ncol(x$x_env$x)))
  }
  cat("\n")
  cv_word <- if (is.null(x$cv)) {
    cat("CV specified   : No\n")
  } else {
    cat("CV specified   : Yes\n")
  }

  learner_word <- if (is.null(x$learners)) {
    cat("Learners       : None\n")
  } else {
    cat(sprintf("  Learners       : %d\n", length(x$learners)))
  }

  metalarner_word <- if (is.null(x$metalearners)) {
    cat("Metalearners   : None\n")
  } else {
    cat(sprintf("  Metalearners   : %d\n", length(x$metalearners)))
  }
  cat("\n")
  cat("Not yet fitted.")

  invisible(x)
}
