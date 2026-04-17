#' Build a consistent lambda sequence for lrn_glmnet
#'
#' Returns an \code{enfold_lambda_request} sentinel that is passed to
#' \code{\link{lrn_glmnet}} as its \code{lambda} argument.  The actual
#' sequence is computed inside \code{lrn_glmnet} once \code{family},
#' \code{alpha}, \code{frm}, and \code{offset} are in scope — so those
#' arguments never need to be repeated here.
#'
#' @param x Data to compute the sequence from. Accepts a plain matrix or data
#'   frame, an \code{enfold_task} or \code{enfold_task_fitted} (in which case
#'   \code{y} is extracted automatically and the \code{y} argument is ignored),
#'   an \code{FBM} (bigstatsr), an in-memory Arrow Table, or an
#'   \code{enfold_arrow_file} reference.
#' @param y Numeric response vector. Required when \code{x} is not a task
#'   object; ignored otherwise.
#' @param nlambda Integer. Number of lambda values in the sequence. Default
#'   \code{100L}.
#' @return An S3 object of class \code{"enfold_lambda_request"}. Passing it as
#'   \code{lambda} to \code{lrn_glmnet} triggers resolution using the
#'   learner's own \code{family}, \code{alpha}, \code{frm}, and
#'   \code{offset}.
#' @seealso \code{\link{lrn_glmnet}}
#' @examples
#' \dontrun{
#' x <- as.matrix(mtcars[, -1]); y <- mtcars$mpg
#' lrn <- lrn_glmnet("en", family = gaussian(),
#'   lambda = make_lambda_sequence(x, y, nlambda = 50L))
#' }
#' @export
make_lambda_sequence <- function(x, y = NULL, nlambda = 100L) {
  structure(
    list(x = x, y = y, nlambda = nlambda),
    class = "enfold_lambda_request"
  )
}

# ── Internal resolver called by lrn_glmnet ────────────────────────────────────

.resolve_lambda_request <- function(req, family, alpha, frm, offset) {
  x <- req$x
  y <- req$y
  nlambda <- req$nlambda

  if (inherits(x, c("enfold_task", "enfold_task_fitted"))) {
    y <- x$y_env$y
    x <- x$x_env$x
  } else {
    if (is.null(y)) {
      stop("y is required when x is not an enfold_task", call. = FALSE)
    }
  }

  if (!is.null(dim(y))) {
    y <- y[, 1L]
  }

  # Materialise big-data backends
  if (inherits(x, "FBM")) {
    x <- x[seq_len(nrow(x)), , drop = FALSE]
  } else if (inherits(x, "enfold_arrow_file")) {
    x <- as.data.frame(arrow::read_feather(x$path))
  } else if (inherits(x, "ArrowTabular")) {
    x <- as.data.frame(x)
  }

  # Drop offset column
  if (!is.null(offset)) {
    x <- x[, -which(colnames(x) == offset), drop = FALSE]
  }

  # Preprocess to numeric matrix (mirrors lrn_glmnet fit logic)
  if (!is.null(frm)) {
    if (is.data.frame(x)) {
      x <- stats::model.matrix(frm, data = x)[, -1L, drop = FALSE]
    } else {
      x <- stats::model.matrix(frm, data = data.frame(x))[, -1L, drop = FALSE]
    }
  } else if (is.data.frame(x)) {
    x <- make_safe_matrix(x, create_instruction_list(x))
  } else {
    x <- as.matrix(x)
  }

  # Center and scale
  col_means <- apply(x, 2L, mean)
  col_sds <- apply(x, 2L, sd)
  col_sds <- ifelse(col_sds == 0, 1, col_sds)
  x <- sweep(x, 2L, col_means, "-")
  x <- sweep(x, 2L, col_sds, "/")

  family <- .resolve_family_for_lambda(family, y)

  lambda_max <- .lambda_max_glmnet(x, y, family, alpha)
  lambda_min_ratio <- if (nrow(x) >= ncol(x)) 1e-4 else 1e-2
  exp(seq(
    log(lambda_max),
    log(lambda_max * lambda_min_ratio),
    length.out = nlambda
  ))
}

.resolve_family_for_lambda <- function(family, y) {
  if (is.list(family) && is.function(family$linkfun)) {
    return(family)
  }
  if (identical(family, "auto")) {
    return(family_guesser(y))
  }
  if (!is.character(family)) {
    stop(
      "'family' must be a family object, 'auto', or a character string",
      call. = FALSE
    )
  }
  switch(
    family,
    gaussian = stats::gaussian(),
    binomial = stats::binomial(),
    poisson = stats::poisson(),
    Gamma = stats::Gamma(),
    inverse.gaussian = stats::inverse.gaussian(),
    quasi = stats::quasi(),
    quasibinomial = stats::quasibinomial(),
    quasipoisson = stats::quasipoisson(),
    stop(
      "Only families that have equivalents in the stats-package are allowed in lrn_glmnet",
      call. = FALSE
    )
  )
}

.lambda_max_glmnet <- function(x, y, family, alpha) {
  nobs <- nrow(x)
  weights <- rep(1 / nobs, nobs)
  mu <- rep(weighted.mean(y, weights), nobs)
  eta <- family$linkfun(mu)
  v <- family$variance(mu)
  m.e <- family$mu.eta(eta)
  r <- y - mu
  rv <- r / v * m.e * weights
  g <- abs(drop(t(rv) %*% x))
  max(g) / max(alpha, 1e-3)
}
