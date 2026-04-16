#' Discrete model selector metalearner
#'
#' Selects the single best base learner from the ensemble based on minimum
#' mean loss on the inner cross-validation predictions. At prediction time,
#' only that learner's predictions are returned.
#'
#' @details
#' All \code{mtl_*} templates return an \code{enfold_learner} that can be
#' passed to \code{\link{add_metalearners}}. General workflow (works with any
#' \code{mtl_*} constructor):
#'
#' \preformatted{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(
#'     mtl_selector("selector"),
#'     mtl_superlearner("superlearner")
#'   ) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#' risk(task, loss_fun = loss_gaussian(), type = "cv")
#' }
#'
#' @param name Character. Name of the metalearner.
#' @param loss_fun An \code{mtl_loss} object (default \code{loss_gaussian()}).
#'   Loss is computed per learner over the full inner-fold predictions.
#' @return An \code{enfold_learner} object.
#' @seealso \code{\link{mtl_superlearner}}, \code{\link{loss_gaussian}}
#' @export
mtl_selector <- function(name, loss_fun = loss_gaussian()) {
  make_learner_factory(
    fit = function(x, y) {
      all_losses <- vapply(
        x,
        function(preds) {
          mean(loss_fun$loss_fun(y = y, y_hat = preds))
        },
        numeric(1L)
      )

      weights <- rep(0, length(all_losses))
      weights[which.min(all_losses)] <- 1
      names(weights) <- names(x)
      return(weights)
    },
    preds = function(object, data) {
      selected_name <- names(which(object == 1))
      to_return <- data[[selected_name]]
      attr(to_return, "selected_learner") <- selected_name
      to_return
    },
    loss_fun = loss_gaussian()
  )(name = name, loss_fun = loss_fun)
}


#' SuperLearner metalearner (Frank-Wolfe optimization)
#'
#' Learns a convex combination of base learner predictions that minimizes the
#' given loss function using the Frank-Wolfe algorithm. The resulting weights
#' are non-negative and sum to one.
#'
#' @param name Character. Name of the metalearner.
#' @param loss_fun An \code{mtl_loss} object with a gradient function
#'   (\code{grad_fun}) (default \code{loss_gaussian()}). The gradient is
#'   required for the Frank-Wolfe update step.
#' @param max_iter Integer. Maximum number of Frank-Wolfe iterations.
#'   Default \code{1000}.
#' @param tol Numeric. Convergence tolerance (Frank-Wolfe gap). Default
#'   \code{1e-7}.
#' @return An \code{enfold_learner} object.
#' @seealso \code{\link{mtl_selector}}, \code{\link{loss_gaussian}}
#' @examples
#' \dontrun{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(mtl_superlearner("sl")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = NA) |>
#'   fit()
#' }
#' @export
mtl_superlearner <- function(
  name,
  loss_fun = loss_gaussian(),
  max_iter = 1000,
  tol = 1e-7
) {
  make_learner_factory(
    fit = function(x, y) {
      col_names <- names(x)
      n_learners <- length(x)

      initial_losses <- vapply(
        x,
        function(preds) {
          mean(loss_fun$loss_fun(y, preds))
        },
        numeric(1L)
      )

      best_idx <- which.min(initial_losses)
      w <- rep(0, n_learners)
      w[best_idx] <- 1
      y_hat <- x[[best_idx]]

      for (t in seq_len(max_iter)) {
        res_grad <- loss_fun$grad_fun(y = y, y_hat = y_hat)

        all_grads <- vapply(
          x,
          function(preds) mean(res_grad * preds),
          numeric(1L)
        )
        idx_min <- which.min(all_grads)

        fw_gap <- sum(w * all_grads) - all_grads[idx_min]
        if (fw_gap < tol) {
          break
        }

        d_y <- x[[idx_min]] - y_hat
        opt_res <- stats::optimize(
          f = function(g) mean(loss_fun$loss_fun(y, y_hat + g * d_y)),
          interval = c(0, 1)
        )
        gamma <- opt_res$minimum

        w <- (1 - gamma) * w
        w[idx_min] <- w[idx_min] + gamma
        y_hat <- y_hat + gamma * d_y
      }

      names(w) <- col_names
      w
    },
    preds = function(object, data) {
      out <- 0
      for (i in seq_along(data)) {
        out <- out + object[[i]] * data[[i]]
      }
      out
    },
    loss_fun = loss_gaussian(),
    max_iter = 1000,
    tol = 1e-7
  )(name = name, loss_fun = loss_fun, max_iter = max_iter, tol = tol)
}


mtl_nnls <- function(name, normalize = FALSE) {
  make_learner_factory(
    fit = function(x, y) {
      if (is.matrix(y)) {
        if (ncol(y) != 1L) {
          stop("`y` must be a vector or a one-column matrix")
        }
        y <- y[, 1L]
      } else if (!is.null(dim(y)) || !is.atomic(y)) {
        stop("`y` must be a vector or a one-column matrix")
      }

      preds_mat <- do.call("cbind", x)
      get_coefs <- nnls::nnls(preds_mat, y)

      coef_vec <- get_coefs$x
      names(coef_vec) <- colnames(preds_mat)

      if (normalize == TRUE) {
        coef_vec <- coef_vec / sum(coef_vec)
      }

      return(coef_vec)
    },
    preds = function(object, data) {
      preds_mat <- do.call("cbind", data)
      return(as.vector(preds_mat %*% object))
    },
    normalize = FALSE
  )(name = name, normalize = normalize)
}


# Template to define individual learners as metalearners
# Not exported; just used internally
mtl_learner <- function(name, index_pos) {
  make_learner_factory(
    fit = function(x, y) return(index_pos),
    preds = function(object, data) return(as.vector(data[[object]])),
    index_pos
  )(name = name, index_pos = index_pos)
}
