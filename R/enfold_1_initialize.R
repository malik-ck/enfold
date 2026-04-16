#' Initialize an enfold task
#'
#' Creates the base \code{enfold_task} object that holds predictor and outcome
#' data. Learners, metalearners, and CV folds are added afterwards with
#' \code{\link{add_learners}}, \code{\link{add_metalearners}}, and
#' \code{\link{add_cv_folds}}, respectively. The task is then fitted via
#' \code{\link{fit.enfold_task}}.
#'
#' @param x A data frame or matrix (or an object inheriting from either) of
#'   predictor variables.
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
  # Validate inputs
  if (!is.matrix(x) && !is.data.frame(x)) {
    stop("`x` must be a data frame or matrix.")
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
      future_pkgs = NULL,
      cv = NULL,
      is_cv_ensemble = NULL,
      fit_objects = NULL,
      ensembles = NULL
    ),
    class = "enfold_task"
  )
}
