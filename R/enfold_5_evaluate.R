# ══════════════════════════════════════════════════════════════════════════════
# predict.enfold_task_fitted
# ══════════════════════════════════════════════════════════════════════════════

#' Predict from a fitted enfold task
#'
#' Returns predictions from one or more metalearners stored in a fitted
#' \code{enfold_task}.
#'
#' @param object An \code{enfold_task_fitted} object returned by
#'   \code{\link{fit.enfold_task}}.
#' @param newdata Optional predictor data (matrix or data frame). When
#'   \code{NULL}, predictions are made on the stored training data.
#' @param metalearner_name Character vector of metalearner names to predict
#'   from. \code{NULL} (default) uses all metalearners.
#' @param ensemble_fold_id Integer. For \code{type = "ensemble"} with outer CV,
#'   selects which outer fold's ensemble to apply to \code{newdata}.
#' @param type Character. Either \code{"cv"} for cross-validated
#'   (out-of-fold) predictions on the training data, or \code{"ensemble"} for
#'   predictions from the ensemble fitted on a fold's training set.
#'   \code{"cv"} requires outer CV (\code{outer_cv} non-\code{NULL} when
#'   calling \code{\link{add_cv_folds}}).
#' @param ... Ignored.
#' @return When a single metalearner is selected, returns its predictions
#'   directly. Otherwise returns a named list with one entry per metalearner.
#'   CV predictions carry \code{attr(result, "indices")} with original row
#'   indices (sorted).
#' @seealso \code{\link{risk.enfold_task_fitted}}, \code{\link{fold_risk}},
#'   \code{\link{predict_learners}}
#' @examples
#' \dontrun{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#'
#' # Cross-validated predictions
#' cv_preds <- predict(task, type = "cv")
#'
#' # Ensemble predictions on new data
#' ens_preds <- predict(task, newdata = x, type = "ensemble",
#'                      ensemble_fold_id = 1L)
#' }
#' @export
predict.enfold_task_fitted <- function(
  object,
  newdata = NULL,
  metalearner_name = NULL,
  ensemble_fold_id = NULL,
  type = NULL,
  ...
) {
  # ── Validate type ──────────────────────────────────────────────────────────
  if (is.null(type)) {
    stop(
      "Please provide `type`: 'cv' for cross-validated predictions or ",
      "'ensemble' for ensemble predictions.",
      call. = FALSE
    )
  }
  type <- match.arg(type, c("cv", "ensemble"))

  # ── Resolve metalearner names ──────────────────────────────────────────────
  use_names <- resolve_metalearner_names(object, metalearner_name)

  # ══════════════════════════════════════════════════════════════════════════
  # type = "cv"
  # ══════════════════════════════════════════════════════════════════════════
  if (type == "cv") {
    if (!object$is_cv_ensemble) {
      stop(
        "Cross-validated predictions require outer CV. ",
        "Refit with a non-NULL `outer_cv`.",
        call. = FALSE
      )
    }

    x <- object$x_env$x
    if (!is.null(newdata)) {
      if (!is.null(dim(newdata)) && !is.null(dim(x))) {
        if (ncol(newdata) != ncol(x)) {
          stop(
            "ncol(newdata) must match ncol of the training data.",
            call. = FALSE
          )
        }
      }
      x <- newdata
    }

    perf_folds <- object$cv$performance_sets
    n_folds <- length(perf_folds)

    result <- lapply(use_names, function(nm) {
      chunks <- lapply(seq_len(n_folds), function(i) {
        val <- validation_set(perf_folds[[i]])
        preds_list <- make_preds_list(
          object$fit_objects[[i]],
          x[val, , drop = FALSE]
        )
        preds <- apply_metalearner(object$ensembles[[i]], nm, preds_list)
        list(idx = val, preds = preds)
      })

      all_idx <- unlist(lapply(chunks, `[[`, "idx"))
      combined <- combine_preds(lapply(chunks, `[[`, "preds"))
      sort_ord <- order(all_idx)

      sorted <- if (is.data.frame(combined) || is.matrix(combined)) {
        combined[sort_ord, , drop = FALSE]
      } else {
        combined[sort_ord]
      }

      structure(sorted, indices = all_idx[sort_ord])
    })
    names(result) <- use_names

    if (length(result) == 1L) {
      return(result[[1L]])
    }
    return(result)
  }

  # ══════════════════════════════════════════════════════════════════════════
  # type = "ensemble"
  # ══════════════════════════════════════════════════════════════════════════
  if (!is.null(ensemble_fold_id) && !object$is_cv_ensemble) {
    warning("`ensemble_fold_id` is ignored for non-cross-validated objects.")
  }

  if (object$is_cv_ensemble && is.null(ensemble_fold_id)) {
    stop(
      "For a cross-validated object, provide `ensemble_fold_id` to select ",
      "which fold's ensemble to use.",
      call. = FALSE
    )
  }

  fold_id <- if (object$is_cv_ensemble) ensemble_fold_id else 1L
  if (is.null(newdata)) {
    newdata <- object$x_env$x
  }

  preds_list <- make_preds_list(object$fit_objects[[fold_id]], newdata)

  result <- lapply(use_names, function(nm) {
    apply_metalearner(object$ensembles[[fold_id]], nm, preds_list)
  })
  names(result) <- use_names

  if (length(result) == 1L) {
    return(result[[1L]])
  }
  result
}


# ══════════════════════════════════════════════════════════════════════════════
# risk
# ══════════════════════════════════════════════════════════════════════════════

#' Compute empirical risk for a fitted enfold task
#'
#' Generic that dispatches to \code{\link{risk.enfold_task_fitted}}.
#'
#' @param object An object of class \code{enfold_task_fitted}.
#' @param ... Further arguments passed to the method.
#' @return A named numeric vector of mean losses, one entry per metalearner.
#' @seealso \code{\link{risk.enfold_task_fitted}}, \code{\link{fold_risk}},
#'   \code{\link{loss_gaussian}}, \code{\link{loss_custom}}
#' @export
risk <- function(object, ...) UseMethod("risk")

#' Compute empirical risk for a fitted enfold task
#'
#' Evaluates mean loss (empirical risk) for each metalearner on either
#' cross-validated predictions (\code{type = "cv"}) or ensemble predictions
#' (\code{type = "ensemble"}). This function has the same signature as
#' \code{\link{predict.enfold_task_fitted}}, with the addition of
#' \code{loss_fun}.
#'
#' @param object An \code{enfold_task_fitted} object.
#' @param loss_fun An \code{mtl_loss} object created by
#'   \code{\link{loss_gaussian}}, \code{\link{loss_logistic}},
#'   \code{\link{loss_poisson}}, \code{\link{loss_gamma}}, or
#'   \code{\link{loss_custom}}.
#' @param newdata Optional predictor data. Defaults to the stored training
#'   data.
#' @param metalearner_name Character vector of metalearner names to evaluate.
#'   \code{NULL} (default) uses all metalearners.
#' @param ensemble_fold_id Integer. For \code{type = "ensemble"} with outer
#'   CV, selects which fold's ensemble to evaluate.
#' @param type Character. One of \code{"cv"} or \code{"ensemble"}.
#' @param ... Ignored.
#' @return A named numeric vector of mean losses, one per metalearner.
#' @seealso \code{\link{fold_risk}}, \code{\link{loss.enfold_task_fitted}},
#'   \code{\link{loss_gaussian}}, \code{\link{loss_custom}},
#'   \code{\link{predict.enfold_task_fitted}}
#' @examples
#' \dontrun{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#'
#' risk(task, loss_fun = loss_gaussian(), type = "cv")
#' }
#' @export
risk.enfold_task_fitted <- function(
  object,
  loss_fun,
  newdata = NULL,
  metalearner_name = NULL,
  ensemble_fold_id = NULL,
  type = NULL,
  ...
) {
  loss_fun <- validate_loss_fun(loss_fun)
  y <- object$y_env$y
  use_names <- resolve_metalearner_names(object, metalearner_name)

  get_preds <- suppressWarnings(stats::predict(
    object = object,
    newdata = newdata,
    metalearner_name = use_names,
    ensemble_fold_id = ensemble_fold_id,
    type = type
  ))

  # predict() simplifies to bare object when only one name; normalise to list
  if (!is.list(get_preds) || !is.null(attr(get_preds, "indices"))) {
    get_preds <- stats::setNames(list(get_preds), use_names[[1L]])
  }

  vapply(
    names(get_preds),
    function(nm) {
      preds <- get_preds[[nm]]
      idx <- attr(preds, "indices")
      y_use <- if (!is.null(idx)) subset_y(y, idx) else y
      mean(loss_fun$loss_fun(y_use, preds))
    },
    numeric(1L)
  )
}


# ══════════════════════════════════════════════════════════════════════════════
# fold_risk
# ══════════════════════════════════════════════════════════════════════════════

#' Per-fold empirical risk
#'
#' Returns a matrix of mean losses broken down by outer fold and metalearner.
#' Where \code{\link{risk.enfold_task_fitted}} returns a single mean per
#' metalearner, \code{fold_risk} preserves the fold-level detail so that you
#' can assess how stable each metalearner's performance is and whether one
#' metalearner consistently outperforms another across folds.
#'
#' @param object An \code{enfold_task_fitted} object with outer CV (fitted
#'   with a non-\code{NULL} \code{outer_cv}).
#' @param loss_fun An \code{mtl_loss} object created by
#'   \code{\link{loss_gaussian}}, \code{\link{loss_logistic}},
#'   \code{\link{loss_poisson}}, \code{\link{loss_gamma}}, or
#'   \code{\link{loss_custom}}.
#' @param metalearner_name Character vector of metalearner names to evaluate.
#'   \code{NULL} (default) uses all metalearners.
#' @param ... Ignored.
#' @return A numeric matrix with rows corresponding to outer folds
#'   (\code{"fold_1"}, \code{"fold_2"}, ...) and columns corresponding to
#'   metalearners. Each cell is the mean loss on that fold's validation set.
#' @seealso \code{\link{risk.enfold_task_fitted}}, \code{\link{loss.enfold_task_fitted}},
#' @examples
#' \dontrun{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#'
#' fr <- fold_risk(task, loss_fun = loss_gaussian())
#' # fr is a 3 x 1 matrix; colMeans(fr) equals risk(task, ..., type = "cv")
#' }
#' @export
fold_risk <- function(object, loss_fun, metalearner_name = NULL, ...) {
  if (!inherits(object, "enfold_task_fitted")) {
    stop("`object` must be an enfold_task_fitted.", call. = FALSE)
  }
  if (!object$is_cv_ensemble) {
    stop(
      "fold_risk() requires outer CV. Refit with a non-NULL `outer_cv`.",
      call. = FALSE
    )
  }
  loss_fun <- validate_loss_fun(loss_fun)
  use_names <- resolve_metalearner_names(object, metalearner_name)

  x <- object$x_env$x
  y <- object$y_env$y
  perf_folds <- object$cv$performance_sets
  n_folds <- length(perf_folds)

  out <- matrix(
    NA_real_,
    nrow = n_folds,
    ncol = length(use_names),
    dimnames = list(
      paste0("fold_", seq_len(n_folds)),
      use_names
    )
  )

  for (i in seq_len(n_folds)) {
    val <- validation_set(perf_folds[[i]])
    preds_lst <- make_preds_list(
      object$fit_objects[[i]],
      x[val, , drop = FALSE]
    )
    y_fold <- subset_y(y, val)
    for (nm in use_names) {
      preds <- apply_metalearner(object$ensembles[[i]], nm, preds_lst)
      out[i, nm] <- mean(loss_fun$loss_fun(y_fold, preds))
    }
  }

  out
}


# ══════════════════════════════════════════════════════════════════════════════
# loss
# ══════════════════════════════════════════════════════════════════════════════

#' Compute per-observation loss for a fitted enfold task
#'
#' Generic that dispatches to \code{\link{loss.enfold_task_fitted}}.
#'
#' @param object An object of class \code{enfold_task_fitted}.
#' @param ... Further arguments passed to the method.
#' @return A \code{data.frame} with one row per observation and one column per
#'   metalearner, plus a leading \code{.index} column.
#' @seealso \code{\link{loss.enfold_task_fitted}}, \code{\link{risk}},
#'   \code{\link{fold_risk}}, \code{\link{loss_gaussian}}, \code{\link{loss_custom}}
#' @export
loss <- function(object, ...) UseMethod("loss")

#' Compute per-observation loss for a fitted enfold task
#'
#' Returns a data frame of per-observation losses for each metalearner.
#' Unlike \code{\link{risk}}, which returns a single mean per metalearner,
#' this function preserves observation-level detail so that you can plot loss
#' distributions, inspect high-error observations, or compute weighted risks.
#' This function has the same signature as \code{\link{risk.enfold_task_fitted}}.
#'
#' @param object An \code{enfold_task_fitted} object.
#' @param loss_fun An \code{mtl_loss} object created by
#'   \code{\link{loss_gaussian}}, \code{\link{loss_logistic}},
#'   \code{\link{loss_poisson}}, \code{\link{loss_gamma}}, or
#'   \code{\link{loss_custom}}.
#' @param newdata Optional predictor data. Defaults to the stored training
#'   data.
#' @param metalearner_name Character vector of metalearner names to evaluate.
#'   \code{NULL} (default) uses all metalearners.
#' @param ensemble_fold_id Integer. For \code{type = "ensemble"} with outer
#'   CV, selects which fold's ensemble to evaluate.
#' @param type Character. One of \code{"cv"} or \code{"ensemble"}.
#'   \code{"cv"} requires outer CV (\code{outer_cv} non-\code{NULL}).
#' @param ... Ignored.
#' @return A \code{data.frame} with one row per observation and one column per
#'   metalearner, plus a leading \code{.index} column giving the original row
#'   position in the training data.
#' @seealso \code{\link{risk.enfold_task_fitted}}, \code{\link{fold_risk}},
#'   \code{\link{loss_learners}}, \code{\link{loss_gaussian}},
#'   \code{\link{loss_custom}}
#' @examples
#' \dontrun{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#'
#' df <- loss(task, loss_fun = loss_gaussian(), type = "cv")
#' # df has columns: .index, selector
#' }
#' @export
loss.enfold_task_fitted <- function(
  object,
  loss_fun,
  newdata = NULL,
  metalearner_name = NULL,
  ensemble_fold_id = NULL,
  type = NULL,
  ...
) {
  if (is.null(type)) {
    stop(
      "Please provide `type`: 'cv' for cross-validated per-observation losses or ",
      "'ensemble' for ensemble per-observation losses.",
      call. = FALSE
    )
  }
  type <- match.arg(type, c("cv", "ensemble"))
  loss_fun <- validate_loss_fun(loss_fun)
  use_names <- resolve_metalearner_names(object, metalearner_name)
  y <- object$y_env$y

  preds_all <- suppressWarnings(stats::predict(
    object = object,
    newdata = newdata,
    metalearner_name = use_names,
    ensemble_fold_id = ensemble_fold_id,
    type = type
  ))

  # Normalise to named list
  if (!is.list(preds_all) || !is.null(attr(preds_all, "indices"))) {
    preds_all <- stats::setNames(list(preds_all), use_names[[1L]])
  }

  first_preds <- preds_all[[1L]]
  idx_vec <- attr(first_preds, "indices")

  out <- data.frame(
    .index = if (!is.null(idx_vec)) idx_vec else seq_len(NROW(first_preds)),
    stringsAsFactors = FALSE
  )

  for (nm in use_names) {
    preds <- preds_all[[nm]]
    idx <- attr(preds, "indices")
    y_use <- if (!is.null(idx)) subset_y(y, idx) else y
    out[[nm]] <- loss_fun$loss_fun(y_use, preds)
  }

  out
}


# ══════════════════════════════════════════════════════════════════════════════
# predict_learners
# ══════════════════════════════════════════════════════════════════════════════

#' Predict from individual base learners
#'
#' Returns predictions from the fitted base learners (not metalearners) stored
#' in a fitted \code{enfold_task}. Pipeline paths, grid survivors, and list
#' learner entries are each returned as their own named entry, exactly as they
#' are presented to metalearners during training.
#'
#' @param object An \code{enfold_task_fitted} object.
#' @param newdata Optional predictor data. For \code{type = "ensemble"},
#'   defaults to the stored training data. Ignored for \code{type = "cv"}.
#' @param type Character. Either \code{"cv"} for cross-validated (out-of-fold)
#'   learner predictions on the training data, or \code{"ensemble"} for
#'   predictions from the ensemble trained on a fold's training set.
#' @param fold_id Integer. For \code{type = "ensemble"}, selects which outer
#'   fold's fitted learners to use. Defaults to \code{1L}.
#' @param ... Ignored.
#' @return A named list of learner predictions (one entry per
#'   learner/path/grid entry). For \code{type = "cv"}, each entry carries
#'   \code{attr(result, "indices")} with original row indices.
#' @seealso \code{\link{risk_learners}}, \code{\link{predict.enfold_task_fitted}}
#' @examples
#' \dontrun{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#'
#' predict_learners(task, type = "cv")
#' predict_learners(task, type = "ensemble", fold_id = 1L, newdata = x)
#' }
#' @export
predict_learners <- function(
  object,
  newdata = NULL,
  type = c("cv", "ensemble"),
  fold_id = NULL,
  ...
) {
  if (!inherits(object, "enfold_task_fitted")) {
    stop("`object` must be an enfold_task_fitted.", call. = FALSE)
  }
  type <- match.arg(type)

  if (type == "ensemble") {
    if (!is.null(fold_id) && !object$is_cv_ensemble) {
      warning("`fold_id` is ignored for non-cross-validated objects.")
    }
    if (object$is_cv_ensemble && is.null(fold_id)) {
      stop(
        "For a cross-validated object, provide `fold_id` to select ",
        "which fold's fitted learners to use.",
        call. = FALSE
      )
    }
    fold_id <- if (object$is_cv_ensemble) fold_id else 1L
    if (is.null(newdata)) {
      newdata <- object$x_env$x
    }
    return(make_preds_list(object$fit_objects[[fold_id]], newdata))
  }

  # type == "cv"
  if (!object$is_cv_ensemble) {
    stop(
      "Cross-validated learner predictions require outer CV. ",
      "Refit with a non-NULL `outer_cv`.",
      call. = FALSE
    )
  }

  cv_learner_preds(object)
}


# ══════════════════════════════════════════════════════════════════════════════
# risk_learners
# ══════════════════════════════════════════════════════════════════════════════

#' Compute empirical risk for individual base learners
#'
#' Returns mean loss for each fitted base learner (not metalearner). Wraps
#' \code{\link{predict_learners}} and applies the provided loss function.
#'
#' @param object An \code{enfold_task_fitted} object.
#' @param loss_fun An \code{mtl_loss} object created by
#'   \code{\link{loss_gaussian}}, \code{\link{loss_logistic}},
#'   \code{\link{loss_poisson}}, \code{\link{loss_gamma}}, or
#'   \code{\link{loss_custom}}.
#' @param type Character. Either \code{"cv"} or \code{"ensemble"}.
#' @param fold_id Integer. For \code{type = "ensemble"}, selects which outer
#'   fold's fitted learners to use. Defaults to \code{1L}.
#' @param ... Ignored.
#' @return A named numeric vector of mean losses, one per
#'   learner/path/grid entry.
#' @seealso \code{\link{predict_learners}}, \code{\link{risk.enfold_task_fitted}}
#' @examples
#' \dontrun{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#'
#' risk_learners(task, loss_fun = loss_gaussian(), type = "cv")
#' }
#' @export
risk_learners <- function(
  object,
  loss_fun,
  type = c("cv", "ensemble"),
  fold_id = NULL,
  ...
) {
  if (!inherits(object, "enfold_task_fitted")) {
    stop("`object` must be an enfold_task_fitted.", call. = FALSE)
  }
  type <- match.arg(type)
  loss_fun <- validate_loss_fun(loss_fun)
  y <- object$y_env$y

  preds <- predict_learners(
    object,
    newdata = NULL,
    type = type,
    fold_id = fold_id
  )

  vapply(
    names(preds),
    function(nm) {
      p <- preds[[nm]]
      idx <- attr(p, "indices")
      y_use <- if (!is.null(idx)) subset_y(y, idx) else y
      mean(loss_fun$loss_fun(y_use, p))
    },
    numeric(1L)
  )
}


# ══════════════════════════════════════════════════════════════════════════════
# loss_learners
# ══════════════════════════════════════════════════════════════════════════════

#' Compute per-observation loss for individual base learners
#'
#' Returns a data frame of per-observation losses for each fitted base learner
#' (not metalearner). Where \code{\link{risk_learners}} returns a single mean
#' per learner, this function preserves observation-level detail. Wraps
#' \code{\link{predict_learners}} and applies the provided loss function.
#'
#' @param object An \code{enfold_task_fitted} object.
#' @param loss_fun An \code{mtl_loss} object created by
#'   \code{\link{loss_gaussian}}, \code{\link{loss_logistic}},
#'   \code{\link{loss_poisson}}, \code{\link{loss_gamma}}, or
#'   \code{\link{loss_custom}}.
#' @param type Character. Either \code{"cv"} or \code{"ensemble"}.
#' @param fold_id Integer. For \code{type = "ensemble"}, selects which outer
#'   fold's fitted learners to use. Defaults to \code{1L}.
#' @param ... Ignored.
#' @return A \code{data.frame} with one row per observation and one column per
#'   learner/path/grid entry, plus a leading \code{.index} column giving the
#'   original row position in the training data.
#' @seealso \code{\link{risk_learners}}, \code{\link{loss.enfold_task_fitted}},
#'   \code{\link{predict_learners}}
#' @examples
#' \dontrun{
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#'
#' loss_learners(task, loss_fun = loss_gaussian(), type = "cv")
#' }
#' @export
loss_learners <- function(
  object,
  loss_fun,
  type = c("cv", "ensemble"),
  fold_id = NULL,
  ...
) {
  if (!inherits(object, "enfold_task_fitted")) {
    stop("`object` must be an enfold_task_fitted.", call. = FALSE)
  }
  type <- match.arg(type)
  loss_fun <- validate_loss_fun(loss_fun)
  y <- object$y_env$y

  preds <- predict_learners(
    object,
    newdata = NULL,
    type = type,
    fold_id = fold_id
  )

  first <- preds[[1L]]
  idx_vec <- attr(first, "indices")

  out <- data.frame(
    .index = if (!is.null(idx_vec)) idx_vec else seq_len(NROW(first)),
    stringsAsFactors = FALSE
  )

  for (nm in names(preds)) {
    p <- preds[[nm]]
    idx <- attr(p, "indices")
    y_use <- if (!is.null(idx)) subset_y(y, idx) else y
    out[[nm]] <- loss_fun$loss_fun(y_use, p)
  }

  out
}


# ══════════════════════════════════════════════════════════════════════════════
# print.enfold_task_fitted
# ══════════════════════════════════════════════════════════════════════════════

#' @export
print.enfold_task_fitted <- function(x, ...) {
  cv_word <- if (x$is_cv_ensemble) {
    sprintf(
      "Cross-validated across %d outer fold(s).",
      length(x$cv$performance_sets)
    )
  } else {
    "Not cross-validated."
  }

  n_build <- if (!is.null(x$cv$build_sets)) {
    length(x$cv$build_sets[[1L]])
  } else {
    0L
  }

  cat(
    "\u2500\u2500 enfold_task_fitted ",
    paste(rep("\u2500", 30), collapse = ""),
    "\n",
    sep = ""
  )
  cat(sprintf("  Learners     : %d\n", length(x$learners)))
  cat(sprintf("  Metalearners : %d\n", length(x$metalearners)))
  cat(sprintf("  Inner folds  : %d\n", n_build))
  cat(sprintf("  %s\n", cv_word))
  cat(paste(rep("\u2500", 50), collapse = ""), "\n")
  invisible(x)
}


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

# Apply one named metalearner from a fold's ensemble list to a preds_list
apply_metalearner <- function(ensembles_fold, mtl_name, preds_list) {
  mtl_names <- get_learner_names(ensembles_fold)
  mtl_pos <- which(mtl_names == mtl_name)
  if (length(mtl_pos) == 0L) {
    stop(sprintf(
      "Metalearner '%s' not found in this fold's ensembles.\nAvailable: %s",
      mtl_name,
      paste(mtl_names, collapse = ", ")
    ))
  }
  stats::predict(ensembles_fold[[mtl_pos]], preds_list)
}

# Resolve which metalearner names to use given user input and object state
resolve_metalearner_names <- function(object, metalearner_name) {
  all_names <- get_learner_names(object$metalearners)

  if (!is.null(metalearner_name)) {
    bad <- setdiff(metalearner_name, all_names)
    if (length(bad)) {
      stop(sprintf(
        "Metalearner name(s) not found: %s\nAvailable: %s",
        paste(bad, collapse = ", "),
        paste(all_names, collapse = ", ")
      ))
    }
    return(metalearner_name)
  }

  all_names
}

# Build a named preds_list from a single fold's list of fitted learners
make_preds_list <- function(fit_objects_fold, newdata) {
  preds_list <- list()
  for (j in seq_along(fit_objects_fold)) {
    fitted_lrn <- fit_objects_fold[[j]]
    raw <- tryCatch(
      stats::predict(fitted_lrn, newdata = newdata),
      error = function(e) {
        warning(sprintf(
          "Learner '%s' failed during prediction: %s",
          get_lrn_display_name(fitted_lrn),
          conditionMessage(e)
        ))
        NULL
      }
    )
    if (is.null(raw)) {
      next
    }

    # Splice pipeline/grid outputs as independent entries
    if (
      is.list(raw) &&
        !is.data.frame(raw) &&
        inherits(fitted_lrn, c("enfold_pipeline_fitted", "enfold_grid_fitted"))
    ) {
      for (nm in names(raw)) {
        if (inherits(raw[[nm]], "enfold_pipeline_path_error")) {
          warning(sprintf(
            "Pipeline path '%s' errored during prediction and is skipped.",
            nm
          ))
          next
        }
        preds_list[[nm]] <- raw[[nm]]
      }
    } else if (
      is.list(raw) &&
        !is.data.frame(raw) &&
        inherits(fitted_lrn, "enfold_list_fitted")
    ) {
      prefix <- paste0(fitted_lrn$name, "/")
      for (nm in names(raw)) {
        preds_list[[paste0(prefix, nm)]] <- raw[[nm]]
      }
    } else {
      preds_list[[get_lrn_display_name(fitted_lrn)]] <- raw
    }
  }
  if (length(preds_list) == 0L) {
    stop("All learners failed during prediction.")
  }
  preds_list
}

# Compute out-of-fold learner predictions over outer performance folds
cv_learner_preds <- function(object) {
  x <- object$x_env$x
  perf_folds <- object$cv$performance_sets
  n_folds <- length(perf_folds)

  # Collect per-fold chunks: list indexed by learner name, each element a list
  # of (idx, preds) chunks across folds
  chunks_by_learner <- list()

  for (i in seq_len(n_folds)) {
    val <- validation_set(perf_folds[[i]])
    chunk <- make_preds_list(object$fit_objects[[i]], x[val, , drop = FALSE])
    for (nm in names(chunk)) {
      if (is.null(chunks_by_learner[[nm]])) {
        chunks_by_learner[[nm]] <- list()
      }
      chunks_by_learner[[nm]] <- c(
        chunks_by_learner[[nm]],
        list(list(idx = val, preds = chunk[[nm]]))
      )
    }
  }

  # Combine chunks for each learner, sort by index
  lapply(chunks_by_learner, function(fold_chunks) {
    all_idx <- unlist(lapply(fold_chunks, `[[`, "idx"))
    combined <- combine_preds(lapply(fold_chunks, `[[`, "preds"))
    sort_ord <- order(all_idx)

    sorted <- if (is.data.frame(combined) || is.matrix(combined)) {
      combined[sort_ord, , drop = FALSE]
    } else {
      combined[sort_ord]
    }

    structure(sorted, indices = all_idx[sort_ord])
  })
}

# Validate that loss_fun is a proper mtl_loss object
validate_loss_fun <- function(loss_fun) {
  if (missing(loss_fun) || is.null(loss_fun)) {
    stop(
      "`loss_fun` must be provided (e.g. `loss_gaussian()`).\n",
      "You can also specify a custom loss via `loss_custom()`. ",
      "Example: loss_custom(function(y, y_hat) abs(y - y_hat)).",
      call. = FALSE
    )
  }
  if (!inherits(loss_fun, "mtl_loss")) {
    stop(
      "`loss_fun` must be of class 'mtl_loss'.\n",
      "You can specify a custom loss via loss_custom(). ",
      "Example: loss_custom(function(y, y_hat) abs(y - y_hat)).",
      call. = FALSE
    )
  }
  loss_fun
}
