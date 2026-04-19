# cv_fit generic and methods — internal only
# These are the workhorses used by build_ensembles and the grid search engines.
#' @keywords internal
cv_fit <- function(learner, folds, x, y, future_pkgs = character(0L), ...) UseMethod("cv_fit")

# Default: one fit per fold, combined across folds, one output name.
#' @exportS3Method enfold::cv_fit
cv_fit.default <- function(learner, folds, x, y, future_pkgs = character(0L), ...) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.")
  }
  lrn_nm <- get_lrn_display_name(learner)
  all_idx <- unlist(lapply(folds, validation_set))
  chunks <- fit_predict_folds(learner, folds, x, y, lrn_nm, future_pkgs)
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
cv_fit.enfold_pipeline <- function(learner, folds, x, y, future_pkgs = character(0L), ...) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.")
  }
  lrn_nm <- paste(active_path_names(learner), collapse = "|")
  all_idx <- unlist(lapply(folds, validation_set))

  chunks <- fit_predict_folds(learner, folds, x, y, lrn_nm, future_pkgs)

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
cv_fit.enfold_grid <- function(learner, folds, x, y, future_pkgs = character(0L), ...) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.", call. = FALSE)
  }

  if (is.null(learner$search_engine)) {
    # Bare grid: exhaustive search over all discrete combinations
    all_discrete <- all(vapply(
      learner$hyperparams,
      function(p) !inherits(p, "enfold_range"),
      logical(1L)
    ))
    if (!all_discrete) {
      stop(
        "A bare enfold_grid (no search engine) cannot have continuous enfold_range ",
        "hyperparameters. Wrap with a grd_* constructor or a custom strategy to use a search engine.",
        call. = FALSE
      )
    }
    combo_df <- draw(learner$hyperparams, n = NULL)
    results <- lapply(seq_len(nrow(combo_df)), function(i) {
      combo <- combo_row(combo_df, i)
      nm <- make_combo_name(learner$name, combo)
      mod <- tryCatch(apply_combo(learner, combo), error = function(e) NULL)
      if (is.null(mod)) {
        return(NULL)
      }
      contrib <- tryCatch(cv_fit(mod, folds, x, y, future_pkgs = future_pkgs), error = function(e) NULL)
      if (is.null(contrib) || !is.null(attr(contrib, "failed_learner"))) {
        return(NULL)
      }
      list(name = nm, combo = combo, contrib = contrib)
    })
    results <- Filter(Negate(is.null), results)
  } else {
    is_pipe <- inherits(learner$learner_object, "enfold_pipeline")
    ss <- if (is_pipe) {
      search_space(learner$learner_object)
    } else {
      search_space(learner)
    }
    results <- learner$search_engine(ss, learner, x, y, folds)
  }

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

  # For multi-node pipeline grids with multiple winners, the resolved learner will be
  # an enfold_list (make_modified_list_pipeline). make_preds_list prefixes enfold_list
  # output names with the grid name; align the CV names now so they match at predict time.
  is_pipeline   <- inherits(learner$learner_object, "enfold_pipeline")
  is_multi_node <- is_pipeline && any(grepl("/", names(learner$hyperparams), fixed = TRUE))
  if (is_multi_node && length(results) > 1L) {
    pfx    <- paste0(learner$name, "/")
    names(out) <- paste0(pfx, names(out))
  }

  attr(out, "resolved_learner") <- build_resolved_learner(learner, results)
  out
}

# enfold_list: single learner whose predict() returns a named list.
# Each list entry is treated as an independent learner output.
#' @exportS3Method enfold::cv_fit
cv_fit.enfold_list <- function(learner, folds, x, y, future_pkgs = character(0L), ...) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.")
  }
  lrn_nm <- get_lrn_display_name(learner)
  all_idx <- unlist(lapply(folds, validation_set))
  chunks <- fit_predict_folds(learner, folds, x, y, lrn_nm, future_pkgs)
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
fit_predict_folds <- function(learner, folds, x, y, lrn_nm, future_pkgs = character(0L)) {
  if (!inherits(folds, "enfold_fold_list")) {
    stop("`folds` must be an `enfold_fold_list`.")
  }
  future.apply::future_lapply(
    seq_along(folds),
    function(k) {
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
    },
    future.globals = list(
      folds    = folds,
      learner  = learner,
      x        = x,
      y        = y,
      lrn_nm   = lrn_nm,
      training_set               = training_set,
      `training_set.enfold_fold` = training_set.enfold_fold,
      validation_set               = validation_set,
      `validation_set.enfold_fold` = validation_set.enfold_fold,
      subset_x = subset_x,
      subset_y = subset_y
    ),
    future.packages = c(future_pkgs, "enfold"),
    future.seed = TRUE
  )
}
