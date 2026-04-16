#' Create a custom loss object for metalearning
#'
#' Bundles a loss function and an optional gradient into an \code{mtl_loss}
#' object that can be passed to \code{\link{risk.enfold_task_fitted}} and to
#' metalearners that require a differentiable loss (e.g.
#' \code{\link{mtl_superlearner}}).
#'
#' @param loss A \code{function(y, y_hat)} returning a numeric vector of
#'   per-observation losses.
#' @param gradient An optional \code{function(y, y_hat)} returning a numeric
#'   vector of first derivatives with respect to \code{y_hat}. Required by
#'   \code{\link{mtl_superlearner}} and some other metalearners.
#' @param ... Additional named elements stored on the returned list (e.g.
#'   \code{family = "gaussian"} for documentation purposes).
#' @return An object of class \code{mtl_loss}.
#' @seealso \code{\link{loss_gaussian}}, \code{\link{loss_logistic}},
#'   \code{\link{loss_poisson}}, \code{\link{loss_gamma}}
#' @examples
#' # Mean absolute error (no gradient required for mtl_selector)
#' mae_loss <- loss_custom(
#'   loss = function(y, y_hat) abs(y - y_hat)
#' )
#'
#' # Huber loss with gradient
#' delta <- 1
#' huber_loss <- loss_custom(
#'   loss = function(y, y_hat) {
#'     r <- y - y_hat
#'     ifelse(abs(r) <= delta, 0.5 * r^2, delta * (abs(r) - 0.5 * delta))
#'   },
#'   gradient = function(y, y_hat) {
#'     r <- y - y_hat
#'     ifelse(abs(r) <= delta, -r, -delta * sign(r))
#'   }
#' )
#' @export
loss_custom <- function(loss, gradient = NULL, ...) {
  stopifnot(is.function(loss))
  out <- list(
    loss_fun = loss,
    grad_fun = gradient
  )
  class(out) <- "mtl_loss"
  out
}

#' Compute element-wise loss
#'
#' Applies the loss function stored in an \code{mtl_loss} object to a pair of
#' observed and predicted values.
#'
#' @param obj An \code{mtl_loss} object.
#' @param y Observed outcomes (vector or matrix).
#' @param y_hat Predicted values matching the shape of \code{y}.
#' @param ... Ignored.
#' @return A numeric vector of per-observation losses.
#' @examples
#' compute_loss(loss_gaussian(), y = 1:5, y_hat = c(1.1, 1.9, 3.2, 3.8, 5.1))
#' @export
compute_loss <- function(obj, y, y_hat, ...) UseMethod("compute_loss")

#' @export
compute_loss.mtl_loss <- function(obj, y, y_hat, ...) {
  obj$loss_fun(y, y_hat)
}

# A little printer because why not
#' @export
print.mtl_loss <- function(x, ...) {
  cat("<mtl_loss>\n")
  invisible(x)
}

#' Pre-built loss functions
#'
#' Convenience constructors that return ready-to-use \code{mtl_loss} objects
#' for the most common outcome types.
#'
#' \describe{
#'   \item{\code{loss_gaussian()}}{Squared error: \eqn{(y - \hat{y})^2}.
#'     Suitable for continuous outcomes.}
#'   \item{\code{loss_logistic()}}{Binary cross-entropy:
#'     \eqn{-[y \log \hat{y} + (1-y) \log (1-\hat{y})]}.
#'     Suitable for binary outcomes where \eqn{\hat{y} \in (0,1)}.}
#'   \item{\code{loss_poisson()}}{Poisson deviance contribution:
#'     \eqn{\hat{y} - y \log \hat{y}}.
#'     Suitable for count outcomes where \eqn{\hat{y} > 0}.}
#'   \item{\code{loss_gamma()}}{Gamma deviance contribution:
#'     \eqn{y / \hat{y} + \log \hat{y}}.
#'     Suitable for strictly positive skewed outcomes.}
#' }
#'
#' @return An \code{mtl_loss} object with \code{loss_fun} and \code{grad_fun}
#'   populated.
#' @seealso \code{\link{loss_custom}} for user-defined losses.
#' @examples
#' loss_gaussian()
#' loss_logistic()
#' loss_poisson()
#' loss_gamma()
#'
#' # Use in loss evaluation
#' compute_loss(loss_gaussian(), y = 1:5, y_hat = c(1.1, 1.9, 3.2, 3.8, 5.1))
#' @name loss_builtins
NULL

#' @rdname loss_builtins
#' @export
loss_gaussian <- function() {
  loss_custom(
    loss = function(y, y_hat) (y - y_hat)^2,
    gradient = function(y, y_hat) -2 * (y - y_hat),
    family = "gaussian"
  )
}

#' @rdname loss_builtins
#' @export
loss_logistic <- function() {
  loss_custom(
    loss = function(y, y_hat) {
      # Add small epsilon to prevent log(0)
      eps <- 1e-15
      y_hat <- pmax(pmin(y_hat, 1 - eps), eps)
      -(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    },
    gradient = function(y, y_hat) {
      eps <- 1e-15
      y_hat <- pmax(pmin(y_hat, 1 - eps), eps)
      -(y / y_hat - (1 - y) / (1 - y_hat))
    },
    family = "binomial"
  )
}

#' @rdname loss_builtins
#' @export
loss_poisson <- function() {
  loss_custom(
    loss = function(y, y_hat) {
      y_hat <- pmax(y_hat, 1e-15)
      y_hat - y * log(y_hat)
    },
    gradient = function(y, y_hat) {
      y_hat <- pmax(y_hat, 1e-15)
      1 - y / y_hat
    },
    family = "poisson"
  )
}

#' @rdname loss_builtins
#' @export
loss_gamma <- function() {
  loss_custom(
    loss = function(y, y_hat) {
      y_hat <- pmax(y_hat, 1e-15)
      y / y_hat + log(y_hat)
    },
    gradient = function(y, y_hat) {
      y_hat <- pmax(y_hat, 1e-15)
      -y / y_hat^2 + 1 / y_hat
    },
    family = "gamma"
  )
}
