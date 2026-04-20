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
#' @return An object of class \code{enfold_task}.
#' @seealso \code{\link{add_learners}}, \code{\link{add_cv_folds}},
#'   \code{\link{fit.enfold_task}}
#' @export
#' @examples
#' x <- mtcars[, -1]
#' y <- mtcars$mpg
#' task <- initialize_enfold(x, y)
#' task
initialize_enfold <- function(x, y) {
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

  # Get a starting list
  structure(
    list(
      x_env = x_env,
      y_env = y_env,
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
  cat(sprintf("  Predictors   : %d\n", ncol(x$x_env$x)))
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
