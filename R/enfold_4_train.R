# ══════════════════════════════════════════════════════════════════════════════
# fit.enfold_task — main training entry point
# ══════════════════════════════════════════════════════════════════════════════

#' Fit a cross-validated SuperLearner
#'
#' Trains all learners and metalearners stored on an \code{enfold_task}. If
#' called again on an already-fitted task (class \code{enfold_task_fitted}),
#' only newly added learners or metalearners are fitted:
#' \itemize{
#'   \item New learners detected → full re-fit (build ensembles and final
#'         learner fit from scratch).
#'   \item Only new metalearners detected → re-run inner-CV build step with
#'         the new metalearners only, then append to the existing ensembles.
#'   \item Nothing new → message and return unchanged.
#' }
#'
#' @param object An object of class \code{enfold_task}, constructed via
#'   \code{initialize_enfold()} with learners, optional metalearners, and CV
#'   folds added.
#' @param add_future_pkgs A character vector of additional packages to load in
#'   each future worker. Useful when using parallel backends with the
#'   \code{future.apply} package. \code{enfold} tries to find these automatically,
#'  but you can use this argument to specify any packages that were missed.
#' @param ... Ignored.
#'
#' @return The same \code{enfold_task} with class updated to
#'   \code{c("enfold_task_fitted", "enfold_task")} and the following fields
#'   populated: \code{fit_objects}, \code{ensembles}, \code{is_cv_ensemble},
#'   \code{fitted_learner_names}, \code{fitted_metalearner_names}.
#' @seealso \code{\link{initialize_enfold}}, \code{\link{add_learners}},
#'   \code{\link{add_cv_folds}}, \code{\link{predict.enfold_task_fitted}},
#'   \code{\link{risk.enfold_task_fitted}}
#' @examples
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()), lrn_mean("mean")) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = 3L) |>
#'   fit()
#' task
#' @export
fit.enfold_task <- function(object, add_future_pkgs = NULL, ...) {
  # ── Typechecks ──────────────────────────────────────────────────────────────
  if (!inherits(object, "enfold_task")) {
    stop("`object` must be an enfold_task.", call. = FALSE)
  }

  if (is.null(object$cv)) {
    stop("No CV folds found. Call add_cv_folds() first.", call. = FALSE)
  }

  if (length(object$learners) == 0L) {
    stop("No learners found. Call add_learners() first.", call. = FALSE)
  }

  cv <- object$cv
  x <- object$x_env$x
  y <- object$y_env$y

  object$future_pkgs <- unique(c(object$future_pkgs, add_future_pkgs))

  # Metalearners require inner CV; warn and drop if missing
  if (length(object$metalearners) > 0L && is.null(cv$build_sets)) {
    warning(
      "Metalearners were supplied without inner cross-validation (build_sets). ",
      "They are ignored.",
      call. = FALSE
    )
    object$metalearners <- list()
  }

  # Cross-validated ensembles require at least one metalearner
  if (length(object$metalearners) == 0L && !is.null(cv$performance_sets)) {
    stop(
      "For cross-validated ensembles (outer_cv non-NULL), ",
      "please provide at least one metalearner via add_metalearners().",
      call. = FALSE
    )
  }

  # Grid learners require inner CV
  has_grids <- any(vapply(
    object$learners,
    function(lrn) inherits(lrn, "enfold_grid"),
    logical(1L)
  ))
  if (has_grids && is.null(cv$build_sets)) {
    stop(
      "inner_cv must be non-NULL when any learner is an `enfold_grid`.",
      call. = FALSE
    )
  }

  # ── Incremental fitting ────────────────────────────────────────────────────
  already_fitted <- inherits(object, "enfold_task_fitted")

  if (already_fitted) {
    current_lrn_names <- get_learner_names(object$learners)
    current_mtl_names <- if (length(object$metalearners) > 0L) {
      get_learner_names(object$metalearners)
    } else {
      character(0L)
    }

    new_lrn_names <- setdiff(current_lrn_names, object$fitted_learner_names)
    new_mtl_names <- setdiff(current_mtl_names, object$fitted_metalearner_names)

    if (length(new_lrn_names) == 0L && length(new_mtl_names) == 0L) {
      message("No new learners or metalearners detected; nothing to fit.")
      return(object)
    }

    if (length(new_lrn_names) > 0L) {
      # New learners added: must re-fit everything — fall through to full fit
      already_fitted <- FALSE
    } else {
      # Only new metalearners: rebuild inner-CV predictions and fit new metalearners
      new_mtl_idx <- get_learner_names(object$metalearners) %in% new_mtl_names
      new_metalearners <- object$metalearners[new_mtl_idx]

      new_ensembles <- build_ensembles(
        cv = cv,
        learners = object$learners,
        metalearners = new_metalearners,
        x = x,
        y = y,
        future_pkgs = object$future_pkgs
      )

      # Append new fitted metalearners; preserve existing ensemble attributes
      for (i in seq_along(object$ensembles)) {
        old_attrs <- attributes(object$ensembles[[i]])
        object$ensembles[[i]] <- c(object$ensembles[[i]], new_ensembles[[i]])
        attributes(object$ensembles[[i]]) <- old_attrs
      }

      object$fitted_metalearner_names <- current_mtl_names
      return(object)
    }
  }

  # ── Full fit ─────────────────────────────────────────────────────────────────

  # Build ensembles via inner CV (if build_sets present)
  get_all_ensembles <- if (!is.null(cv$build_sets)) {
    build_ensembles(
      cv = cv,
      learners = object$learners,
      metalearners = object$metalearners,
      x = x,
      y = y,
      future_pkgs = object$future_pkgs
    )
  } else {
    NULL
  }

  # Resolve grid learners (and grid-containing pipelines) using the resolved_learners
  # collected by build_ensembles. Position-indexed to avoid name ambiguity.
  learners_for_fit <- if (!has_grids) {
    object$learners
  } else {
    all_resolved <- vector("list", length(object$learners))
    for (ens in get_all_ensembles) {
      rl_list <- attr(ens, "resolved_learners")
      for (i in seq_along(rl_list)) {
        if (!is.null(rl_list[[i]]) && is.null(all_resolved[[i]])) {
          all_resolved[[i]] <- rl_list[[i]]
        }
      }
    }

    Filter(
      Negate(is.null),
      lapply(seq_along(object$learners), function(i) {
        lrn <- object$learners[[i]]
        rl <- all_resolved[[i]]
        if (!is.null(rl)) {
          return(rl)
        }
        if (inherits(lrn, "enfold_grid")) {
          warning(
            sprintf(
              "Learner '%s': no candidates survived build_ensembles; excluded from final fit.",
              get_lrn_display_name(lrn)
            ),
            call. = FALSE
          )
          return(NULL)
        }
        lrn
      })
    )
  }

  # Performance (outer) folds for the final learner fit
  ensemble_cv <- !is.null(cv$performance_sets)
  perf_folds <- if (ensemble_cv) {
    cv$performance_sets
  } else {
    new_fold_list(list(new_fold(
      validation_set = seq_len(nrow(x)),
      training_set = seq_len(nrow(x)),
      n = nrow(x)
    )))
  }

  get_all_learners <- fit_ensemble(
    x = x,
    y = y,
    perf_folds = perf_folds,
    learners = learners_for_fit,
    future_pkgs = object$future_pkgs
  )

  # Store results on the task
  object$learners <- learners_for_fit
  object$fit_objects <- get_all_learners
  object$ensembles <- get_all_ensembles
  object$is_cv_ensemble <- ensemble_cv

  object$fitted_learner_names <- get_learner_names(object$learners)
  object$fitted_metalearner_names <- if (length(object$metalearners) > 0L) {
    get_learner_names(object$metalearners)
  } else {
    character(0L)
  }

  class(object) <- c("enfold_task_fitted", "enfold_task")
  object
}


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

#' Fit all learners on each performance fold's training set
#'
#' By the time this is called from \code{fit.enfold_task}, any
#' \code{enfold_grid} objects have already been resolved to concrete
#' \code{enfold_learner} or \code{enfold_list} objects.
#'
#' @param x Full predictor matrix or data frame.
#' @param y Full outcome vector or matrix.
#' @param perf_folds A \code{enfold_fold_list} of performance (outer) folds.
#' @param learners A list of \code{enfold_learner} objects (no grids).
#' @param future_pkgs Character vector of packages for parallel workers.
#' @return A list of length \code{length(perf_folds)}, each element a list of
#'   fitted learner objects.
fit_ensemble <- function(x, y, perf_folds, learners, future_pkgs) {
  future.apply::future_lapply(
    perf_folds,
    function(fold) {
      tr <- training_set(fold)
      lapply(learners, function(lrn) {
        fit(lrn, x[tr, , drop = FALSE], subset_y(y, tr))
      })
    },
    future.packages = future_pkgs,
    future.globals = list(
      x = x,
      y = y,
      learners = learners,
      subset_y = subset_y,
      training_set = training_set,
      `training_set.enfold_fold` = training_set.enfold_fold
    ),
    future.seed = TRUE
  )
}


#' Fit ensembles across inner folds without retaining fitted learner objects
#'
#' Iterates learners in the outer loop and folds in the inner loop, enabling
#' grid-based early stopping. For each learner, predictions are accumulated
#' across all folds before moving to the next learner. List outputs from
#' \code{enfold_pipeline} and \code{enfold_grid} objects are spliced into
#' \code{preds_list} as independent entries; list outputs from other learner
#' types are stored as-is.
#'
#' @param cv A \code{enfold_cv} with \code{build_sets} populated.
#' @param learners A list of \code{enfold_learner} or \code{enfold_pipeline} objects.
#' @param metalearners A list of \code{enfold_learner} objects.
#' @param x Full predictor matrix or data frame.
#' @param y Full outcome vector or matrix.
#' @param future_pkgs Character vector of packages for parallel workers.
#' @return A list of length \code{length(cv$build_sets)}, each element a
#'   list of fitted metalearner objects with an \code{indices} attribute.
#' @keywords internal
#' @export
build_ensembles <- function(
  cv,
  learners,
  metalearners,
  x,
  y,
  future_pkgs = character(0L)
) {
  if (!inherits(cv, "enfold_cv")) {
    stop("`cv` must be a enfold_cv object.")
  }
  if (is.null(cv$build_sets)) {
    stop(
      "No build folds found. Please add them before calling build_ensembles()."
    )
  }

  future.apply::future_lapply(
    cv$build_sets,
    function(inner_folds) {
      all_idx <- unlist(lapply(inner_folds, validation_set))

      preds_list <- list()
      failed_learners <- character(0L)
      resolved_learners <- vector("list", length(learners))

      for (i in seq_along(learners)) {
        lrn <- learners[[i]]
        contrib <- cv_fit(lrn, inner_folds, x, y)
        failed <- attr(contrib, "failed_learner")
        if (!is.null(failed)) {
          failed_learners <- c(failed_learners, failed)
          next
        }
        for (nm in names(contrib)) {
          preds_list[[nm]] <- contrib[[nm]]
        }
        rl <- attr(contrib, "resolved_learner")
        if (!is.null(rl)) resolved_learners[[i]] <- rl
      }

      if (length(preds_list) == 0L) {
        stop("All learners failed on all folds. Cannot build ensemble.")
      }

      y_sub <- structure(subset_y(y, all_idx), indices = all_idx)

      fitted_metalearners <- lapply(metalearners, function(mtl) {
        fit(mtl, x = preds_list, y = y_sub)
      })

      rm(preds_list, y_sub)

      structure(
        fitted_metalearners,
        indices = all_idx,
        failed_learners = if (length(failed_learners) > 0L) {
          failed_learners
        } else {
          NULL
        },
        resolved_learners = resolved_learners
      )
    },
    future.packages = c(future_pkgs, "enfold"),
    future.globals = list(
      x = x,
      y = y,
      learners = learners,
      metalearners = metalearners,
      combine_preds = combine_preds,
      subset_y = subset_y,
      get_lrn_display_name = get_lrn_display_name,
      fit_predict_folds = fit_predict_folds,
      training_set = training_set,
      `training_set.enfold_fold` = training_set.enfold_fold,
      validation_set = validation_set,
      `validation_set.enfold_fold` = validation_set.enfold_fold
    ),
    future.seed = TRUE
  )
}
