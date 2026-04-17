# cv_fit generic and methods — internal only
# These are the workhorses used by build_ensembles and the grid search engines.
#' @keywords internal
cv_fit <- function(learner, folds, x, y, ...) UseMethod("cv_fit")

# Default: one fit per fold, combined across folds, one output name.
#' @exportS3Method enfold::cv_fit
cv_fit.default <- function(learner, folds, x, y, ...) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.")
  }
  lrn_nm <- get_lrn_display_name(learner)
  all_idx <- unlist(lapply(folds, validation_set))
  chunks <- fit_predict_folds(learner, folds, x, y, lrn_nm)
  if (any(vapply(chunks, `[[`, logical(1L), "failed"))) {
    failed <- which(vapply(chunks, `[[`, logical(1L), "failed"))
    warning(sprintf(
      "Learner '%s' failed on fold(s) %s; excluded.",
      lrn_nm,
      paste(failed, collapse = ", ")
    ))
    return(structure(list(), failed_learner = lrn_nm))
  }
  combined <- combine_preds(lapply(chunks, `[[`, "preds"))
  attr(combined, "indices") <- all_idx
  stats::setNames(list(combined), lrn_nm)
}

# Pipeline: fit the full pipeline per fold via fit_predict_folds(), then split
# the named-list predictions by path name and combine across folds.
#' @exportS3Method enfold::cv_fit
cv_fit.enfold_pipeline <- function(learner, folds, x, y, ...) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.")
  }
  lrn_nm <- paste(active_path_names(learner), collapse = "|")
  all_idx <- unlist(lapply(folds, validation_set))

  chunks <- fit_predict_folds(learner, folds, x, y, lrn_nm)

  if (any(vapply(chunks, `[[`, logical(1L), "failed"))) {
    failed_folds <- which(vapply(chunks, `[[`, logical(1L), "failed"))
    warning(sprintf(
      "Pipeline '%s' failed on fold(s) %s; excluded.",
      lrn_nm,
      paste(failed_folds, collapse = ", ")
    ))
    return(structure(list(), failed_learner = lrn_nm))
  }

  # Only keep paths that survived in every fold
  path_nms <- Reduce(intersect, lapply(chunks, function(ch) names(ch$preds)))
  if (length(path_nms) == 0L) {
    warning(sprintf("All paths in pipeline '%s' failed; excluded.", lrn_nm))
    return(structure(list(), failed_learner = lrn_nm))
  }

  out <- lapply(path_nms, function(nm) {
    p <- combine_preds(lapply(chunks, function(chunk) chunk$preds[[nm]]))
    attr(p, "indices") <- all_idx
    p
  })
  stats::setNames(out, path_nms)
}

# ── cv_fit.enfold_grid ─────────────────────────────────────────────────────
#' @exportS3Method enfold::cv_fit
cv_fit.enfold_grid <- function(learner, folds, x, y, ...) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.", call. = FALSE)
  }

  results <- learner$search_engine(
    learner$hyperparams,
    learner$name,
    learner$learner_object,
    learner$directory,
    x, y, folds
  )

  if (length(results) == 0L) {
    return(structure(list(), failed_learner = learner$name))
  }

  # Flatten all contribution paths across all winning combos
  out <- list()
  for (r in results) {
    for (nm in names(r$contrib)) {
      p <- r$contrib[[nm]]
      attr(p, "combo") <- r$combo
      out[[nm]] <- p
    }
  }

  attr(out, "resolved_learner") <- build_resolved_learner(learner, results)
  out
}

# enfold_list: single learner whose predict() returns a named list.
# Each list entry is treated as an independent learner output.
#' @exportS3Method enfold::cv_fit
cv_fit.enfold_list <- function(learner, folds, x, y, ...) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.")
  }
  lrn_nm <- get_lrn_display_name(learner)
  all_idx <- unlist(lapply(folds, validation_set))
  chunks <- fit_predict_folds(learner, folds, x, y, lrn_nm)
  if (any(vapply(chunks, `[[`, logical(1L), "failed"))) {
    return(structure(list(), failed_learner = lrn_nm))
  }

  entry_nms <- names(chunks[[1L]]$preds)
  if (
    any(vapply(
      chunks,
      function(fc) !identical(names(fc$preds), entry_nms),
      logical(1L)
    ))
  ) {
    stop(
      "Inconsistent list output names across folds for enfold_list learner '",
      lrn_nm,
      "'."
    )
  }

  out <- lapply(entry_nms, function(nm) {
    p <- combine_preds(lapply(chunks, function(fc) fc$preds[[nm]]))
    attr(p, "indices") <- all_idx
    p
  })
  stats::setNames(out, paste0(lrn_nm, "/", entry_nms))
}

# Fit a learner on each fold and collect validation-set predictions.
# Returns a list with one element per fold; each element has fields
# $idx (validation indices), $preds (predictions or NULL), $failed (logical).
fit_predict_folds <- function(learner, folds, x, y, lrn_nm) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.")
  }
  lapply(seq_along(folds), function(k) {
    fold <- folds[[k]]
    tr <- training_set(fold)
    val <- validation_set(fold)
    x_tr <- subset_x(x, tr)
    y_tr <- subset_y(y, tr)
    x_val <- subset_x(x, val)
    tryCatch(
      {
        f <- fit(learner, x_tr, y_tr)
        out <- stats::predict(f, newdata = x_val)
        list(idx = val, preds = out, failed = FALSE)
      },
      error = function(e) {
        warning(sprintf(
          "Learner '%s' failed on fold %d: %s",
          lrn_nm,
          k,
          conditionMessage(e)
        ))
        list(idx = val, preds = NULL, failed = TRUE)
      }
    )
  })
}
