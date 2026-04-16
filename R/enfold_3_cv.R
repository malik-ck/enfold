# ══════════════════════════════════════════════════════════════════════════════
# CV fold adder to tasks
# ══════════════════════════════════════════════════════════════════════════════

#' Add cross-validation folds to an \code{enfold_task}
#'
#' Attaches a cross-validation structure to an \code{enfold_task}. Can be
#' called with explicit fold counts, custom fold functions, or a pre-built
#' \code{enfold_cv} object.
#'
#' @param task An object of class \code{enfold_task}.
#' @param inner_cv Integer \eqn{\geq 2}, a function \code{function(n, ...)}
#'   returning fold index sets, or \code{NA}/\code{NULL} for no inner CV.
#'   Inner folds are used to build ensemble metalearners.
#' @param outer_cv Integer \eqn{\geq 2}, a function, or \code{NA}/\code{NULL}
#'   for no outer CV. Outer folds cross-validate the fitted ensemble.
#' @param cv An optional pre-built \code{enfold_cv} object (e.g. from
#'   \code{\link{create_cv_folds}}). When supplied, \code{inner_cv} and
#'   \code{outer_cv} are ignored.
#' @param ... Additional named arguments forwarded to custom fold functions
#'   that declare them (e.g. \code{cluster}, \code{strata}).
#' @return The updated \code{enfold_task} with the \code{cv} slot populated.
#' @seealso \code{\link{create_cv_folds}}, \code{\link{fit.enfold_task}}
#' @examples
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' # 3-fold CV to evaluate learner performance (no ensemble metalearners)
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian())) |>
#'   add_cv_folds(inner_cv = NA, outer_cv = 3L)
#'
#' # 5-fold inner CV to build an ensemble; no outer CV
#' task2 <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian())) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = NA)
#' @export
add_cv_folds <- function(task, inner_cv = 5L, outer_cv = 5L, cv = NULL, ...) {
  UseMethod("add_cv_folds")
}

#' @export
add_cv_folds.enfold_task <- function(
  task,
  inner_cv = 5L,
  outer_cv = 5L,
  cv = NULL,
  ...
) {
  if (!inherits(task, "enfold_task")) {
    stop("`task` must be an enfold_task object.", call. = FALSE)
  }

  if (!is.null(task$cv)) {
    stop("CV folds already exist on this task.", call. = FALSE)
  }

  if (!is.null(cv)) {
    if (!inherits(cv, "enfold_cv")) {
      stop("`cv` must be an enfold_cv object.", call. = FALSE)
    }
    task$cv <- cv
  } else {
    n <- nrow(task$x_env$x)
    task$cv <- create_cv_folds(
      n = n,
      inner_cv = inner_cv,
      outer_cv = outer_cv,
      ...
    )
  }

  # Can see now whether it was a CV ensemble
  task$is_cv_ensemble <- !is.null(task$cv$build_sets)

  task
}


make_vfold_fun <- function(v) {
  force(v)
  function(n, ...) {
    origami::make_folds(n = n, V = v, fold_fun = origami::folds_vfold)
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# enfold_fold — single fold
# ══════════════════════════════════════════════════════════════════════════════

# new_fold: internal constructor for a single cross-validation fold.
# Use create_cv_folds() or add_cv_folds() to create folds in normal usage.
new_fold <- function(
  validation_set,
  n = NULL,
  training_set = NULL,
  excluded = NULL
) {
  if (is.null(training_set) && is.null(n)) {
    stop("Either `training_set` or `n` must be provided.")
  }

  validation_set <- as.integer(validation_set)
  excluded <- if (!is.null(excluded)) as.integer(excluded) else NULL

  # Determine storage type
  if (!is.null(training_set)) {
    training_set <- as.integer(training_set)
    # Promote to complementary if the sets actually are complementary
    if (!is.null(n)) {
      full <- seq_len(as.integer(n))
      if (
        length(intersect(training_set, validation_set)) == 0L &&
          identical(sort(c(training_set, validation_set)), full)
      ) {
        training_set <- NULL # redundant, derive on access
      }
    }
  }

  structure(
    list(
      validation_set = validation_set,
      training_set = training_set,
      n = if (!is.null(n)) as.integer(n) else NULL,
      excluded = excluded,
      complementary = is.null(training_set)
    ),
    class = "enfold_fold"
  )
}

# ── Accessors ──────────────────────────────────────────────────────────────

#' Extract training or validation indices from a fold
#'
#' \code{training_set()} returns the integer indices of observations used for
#' training; \code{validation_set()} returns those used for evaluation.
#' Excluded indices (set via \code{\link{exclude}}) are always removed from
#' both sets.
#'
#' @param fold An \code{enfold_fold} object.
#' @param ... Ignored.
#' @return An integer vector of row indices.
#' @examples
#' cv <- create_cv_folds(n = 100, inner_cv = NA, outer_cv = 5L)
#' fold <- cv$performance_sets[[1]]
#' training_set(fold)
#' @export
training_set <- function(fold, ...) UseMethod("training_set")

#' @export
training_set.enfold_fold <- function(fold, ...) {
  idx <- if (fold$complementary) {
    setdiff(seq_len(fold$n), fold$validation_set)
  } else {
    fold$training_set
  }
  if (!is.null(fold$excluded)) {
    idx <- setdiff(idx, fold$excluded)
  }
  idx
}

#' @rdname training_set
#' @examples
#' cv <- create_cv_folds(n = 100, inner_cv = NA, outer_cv = 5L)
#' fold <- cv$performance_sets[[1]]
#' validation_set(fold)
#' @export
validation_set <- function(fold, ...) UseMethod("validation_set")

#' @export
validation_set.enfold_fold <- function(fold, ...) {
  idx <- fold$validation_set
  if (!is.null(fold$excluded)) {
    idx <- setdiff(idx, fold$excluded)
  }
  idx
}

# ── exclude() generic ──────────────────────────────────────────────────────

#' Exclude indices from a fold or fold list
#'
#' Returns a new object with the specified indices removed from both training
#' and validation sets on access. The original object is not modified.
#'
#' @param x An \code{enfold_fold}, \code{enfold_fold_list}, or
#'   \code{enfold_cv} object.
#' @param indices Integer vector of row indices to exclude.
#' @param ... Ignored.
#' @return An object of the same class as \code{x} with the excluded indices
#'   registered. Calling \code{\link{training_set}} or
#'   \code{\link{validation_set}} on the result will omit these rows.
#' @details
#' Useful in survival analysis settings where some observations are censored
#' before the first event and should be excluded from the validation set.
#' If applied to an object that already has excluded indices, the previous
#' exclusions are replaced.
#' @examples
#' cv <- create_cv_folds(n = 100, inner_cv = NA, outer_cv = 5L)
#' fold <- cv$performance_sets[[1]]
#' fold_excl <- exclude(fold, indices = c(5L, 10L, 15L))
#' validation_set(fold_excl)  # excluded indices are absent
#' @export
exclude <- function(x, indices, ...) UseMethod("exclude")

#' @export
exclude.enfold_fold <- function(x, indices, ...) {
  x$excluded <- as.integer(unique(indices))
  x
}

# ── print ──────────────────────────────────────────────────────────────────

#' @export
print.enfold_fold <- function(x, ...) {
  tr <- training_set(x)
  val <- validation_set(x)
  cat(sprintf(
    "enfold_fold | %s | train: %d | val: %d%s\n",
    if (x$complementary) "complementary" else "explicit",
    length(tr),
    length(val),
    if (!is.null(x$excluded) && length(x$excluded) > 0L) {
      sprintf(" | %d excluded", length(x$excluded))
    } else {
      ""
    }
  ))
  invisible(x)
}


# ══════════════════════════════════════════════════════════════════════════════
# enfold_fold_list — ordered collection of folds
# ══════════════════════════════════════════════════════════════════════════════

# new_fold_list: internal constructor wrapping a list of enfold_fold objects.
# Used by create_cv_folds() and internally by build_ensembles().
new_fold_list <- function(folds) {
  if (
    !is.list(folds) || !all(vapply(folds, inherits, logical(1L), "enfold_fold"))
  ) {
    stop("All elements must be `enfold_fold` objects.")
  }
  structure(folds, class = c("enfold_fold_list", "list"))
}

#' @export
exclude.enfold_fold_list <- function(x, indices, ...) {
  indices <- as.integer(indices)
  new_fold_list(lapply(x, exclude, indices = indices))
}

#' @export
print.enfold_fold_list <- function(x, ...) {
  n_excl <- sum(vapply(
    x,
    function(f) {
      !is.null(f$excluded) && length(f$excluded) > 0L
    },
    logical(1L)
  ))
  cat(sprintf(
    "enfold_fold_list | %d fold(s)%s\n",
    length(x),
    if (n_excl > 0L) sprintf(" | %d fold(s) with exclusions", n_excl) else ""
  ))
  invisible(x)
}

# ══════════════════════════════════════════════════════════════════════════════
# enfold_cv — full nested CV structure
# ══════════════════════════════════════════════════════════════════════════════

# new_cv: internal constructor for an enfold_cv object.
# Use create_cv_folds() or add_cv_folds() to create CV structures in normal usage.
new_cv <- function(performance_sets = NULL, build_sets = NULL) {
  if (is.null(performance_sets) && is.null(build_sets)) {
    stop("At least one of `performance_sets` or `build_sets` must be non-NULL.")
  }
  if (
    !is.null(performance_sets) &&
      !inherits(performance_sets, "enfold_fold_list")
  ) {
    stop("`performance_sets` must be a enfold_fold_list or NULL.")
  }
  if (!is.null(build_sets)) {
    if (
      !is.list(build_sets) ||
        !all(vapply(build_sets, inherits, logical(1L), "enfold_fold_list"))
    ) {
      stop("`build_sets` must be a list of enfold_fold_list objects or NULL.")
    }
  }
  structure(
    list(performance_sets = performance_sets, build_sets = build_sets),
    class = "enfold_cv"
  )
}

#' @export
exclude.enfold_cv <- function(x, indices, ...) {
  new_cv(
    performance_sets = if (!is.null(x$performance_sets)) {
      exclude(x$performance_sets, indices)
    } else {
      NULL
    },
    build_sets = if (!is.null(x$build_sets)) {
      lapply(x$build_sets, exclude, indices = indices)
    } else {
      NULL
    }
  )
}

#' @export
print.enfold_cv <- function(x, ...) {
  cat("enfold_cv\n")
  if (!is.null(x$performance_sets)) {
    cat(sprintf("  Performance folds : %d\n", length(x$performance_sets)))
  } else {
    cat("  Performance folds : none\n")
  }
  if (!is.null(x$build_sets)) {
    cat(sprintf(
      "  Build fold sets   : %d outer x %d inner\n",
      length(x$build_sets),
      length(x$build_sets[[1L]])
    ))
  } else {
    cat("  Build fold sets   : none\n")
  }
  invisible(x)
}


# ══════════════════════════════════════════════════════════════════════════════
# create_cv_folds — internal constructor
# ══════════════════════════════════════════════════════════════════════════════

#' Create cross-validation folds
#'
#' Convenience function that creates an \code{enfold_cv} object from simple
#' integer fold counts or custom fold functions. This is typically the easiest
#' way to create reusable fold specifications before calling
#' \code{\link{add_cv_folds}}.
#'
#' @param n Total number of observations.
#' @param inner_cv Integer \eqn{\geq 2}, a function \code{function(n, ...)}
#'   returning a list of fold index sets (as produced by \code{origami}), or
#'   \code{NA}/\code{NULL} for no inner folds. Inner folds are used to build
#'   the ensemble metalearner.
#' @param outer_cv Integer \eqn{\geq 2}, a function, or \code{NA}/\code{NULL}
#'   for no outer folds. Outer folds are used to cross-validate the fitted
#'   ensemble.
#' @param ... Additional named arguments forwarded to custom fold functions
#'   that declare them (e.g. \code{cluster}, \code{strata}).
#' @return An \code{enfold_cv} object.
#' @seealso \code{\link{add_cv_folds}}
#' @examples
#' # 5-fold outer CV only (no ensemble)
#' cv1 <- create_cv_folds(n = 200, inner_cv = NA, outer_cv = 5L)
#' cv1
#'
#' # 5-fold inner CV to build ensemble; no outer CV
#' cv2 <- create_cv_folds(n = 200, inner_cv = 5L, outer_cv = NA)
#' cv2
#'
#' # Nested: 3 outer x 5 inner
#' cv3 <- create_cv_folds(n = 200, inner_cv = 5L, outer_cv = 3L)
#' cv3
#' @export
create_cv_folds <- function(n, inner_cv = NA, outer_cv = NA, ...) {
  # ── Resolve cv_instructions to inner_cv / outer_cv ────────────────────────
  resolve_one <- function(x, arg_name) {
    if (is.function(x)) {
      return(x)
    }
    if (is.null(x) || (length(x) == 1L && is.na(x))) {
      return(NULL)
    }
    if (is.numeric(x) && length(x) == 1L && x >= 2L) {
      return(make_vfold_fun(as.integer(x)))
    }
    if (is.list(x)) {
      return(x)
    }
    stop(sprintf(
      "`%s`: must be NA/NULL (no CV), a positive integer >= 2, a function of (n, ...), or a pre-specified list of fold index sets.",
      arg_name
    ))
  }

  inner_cv <- resolve_one(inner_cv, "inner_cv")
  outer_cv <- resolve_one(outer_cv, "outer_cv")

  if (is.null(inner_cv) && is.null(outer_cv)) {
    stop("At least one of `inner_cv` or `outer_cv` must be non-NA/NULL.")
  }

  dots <- list(...)

  call_cv_fun <- function(fun, n) {
    declared <- names(formals(fun))
    if ("..." %in% declared) {
      do.call(fun, c(list(n = n), dots))
    } else {
      do.call(fun, c(list(n = n), dots[intersect(names(dots), declared)]))
    }
  }

  wrap_folds <- function(raw, n) {
    new_fold_list(lapply(raw, function(f) {
      new_fold(
        validation_set = f$validation_set,
        training_set = f$training_set,
        n = n
      )
    }))
  }

  outer_raw <- if (is.null(outer_cv)) {
    list(list(training_set = seq_len(n), validation_set = seq_len(n)))
  } else if (is.function(outer_cv)) {
    call_cv_fun(outer_cv, n)
  } else {
    outer_cv
  }

  outer_fold_list <- wrap_folds(outer_raw, n)

  inner_fold_lists <- if (!is.null(inner_cv)) {
    lapply(outer_fold_list, function(outer_fold) {
      tr <- training_set(outer_fold)
      n_tr <- length(tr)
      raw <- if (is.function(inner_cv)) {
        call_cv_fun(inner_cv, n_tr)
      } else {
        inner_cv
      }
      new_fold_list(lapply(raw, function(f) {
        new_fold(
          validation_set = tr[f$validation_set],
          training_set = tr[f$training_set],
          n = n
        )
      }))
    })
  } else {
    NULL
  }

  new_cv(
    performance_sets = if (!is.null(outer_cv)) outer_fold_list else NULL,
    build_sets = inner_fold_lists
  )
}

# ══════════════════════════════════════════════════════════════════════════════
# A few templates for CV fold creation
# ══════════════════════════════════════════════════════════════════════════════
