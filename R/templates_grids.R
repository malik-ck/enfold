# ── make_score ────────────────────────────────────────────────────────────────

#' Bundle a loss function and optional metalearner into a grid scorer
#'
#' Creates an \code{enfold_score} object that tells \code{\link{grd_early_stop}}
#' and \code{\link{grd_bayes}} how to reduce a grid candidate's out-of-fold
#' predictions to a single scalar. When the candidate produces only one output
#' stream (e.g. a plain \code{enfold_learner}), \code{metalearner} is not
#' needed. When the candidate produces multiple streams (e.g. a pipeline with
#' branching paths or an \code{enfold_list}), \code{metalearner} is required;
#' no default is applied — a missing metalearner causes an informative error at
#' evaluation time so the omission is explicit.
#'
#' @param loss_function An \code{mtl_loss} object, e.g. \code{loss_gaussian()}.
#' @param metalearner An \code{enfold_learner} used to combine multiple output
#'   streams into one prediction before scoring. \code{NULL} (default) is fine
#'   for single-output candidates; multi-output candidates will error if
#'   \code{metalearner} is \code{NULL}.
#' @param higher_is_better Logical. \code{FALSE} (default) means lower scores
#'   are better (minimising a loss). \code{TRUE} means higher scores are better
#'   (e.g. a log-likelihood or accuracy).
#' @return An object of class \code{enfold_score}.
#' @seealso \code{\link{grd_early_stop}}, \code{\link{grd_bayes}},
#'   \code{\link{loss_gaussian}}
#' @examples
#' # Single-output learner — no metalearner needed
#' make_score(loss_gaussian())
#'
#' # Multi-output pipeline — provide a metalearner to combine paths
#' make_score(loss_gaussian(), metalearner = mtl_selector("sel"))
#'
#' # Higher-is-better score (e.g. negative loss or accuracy)
#' make_score(loss_gaussian(), higher_is_better = TRUE)
#' @export
make_score <- function(loss_function, metalearner = NULL, higher_is_better = FALSE) {
  if (!inherits(loss_function, "mtl_loss")) {
    stop(
      "`loss_function` must be an `mtl_loss` object (e.g. `loss_gaussian()`).",
      call. = FALSE
    )
  }
  if (
    !is.null(metalearner) &&
      !inherits(metalearner, c("enfold_learner", "enfold_pipeline", "enfold_list"))
  ) {
    stop(
      "`metalearner` must be an `enfold_learner`, `enfold_pipeline`, or `enfold_list` (or NULL).",
      call. = FALSE
    )
  }
  if (
    !is.logical(higher_is_better) ||
      length(higher_is_better) != 1L ||
      is.na(higher_is_better)
  ) {
    stop("`higher_is_better` must be TRUE or FALSE.", call. = FALSE)
  }
  structure(
    list(
      loss_function    = loss_function,
      metalearner      = metalearner,
      higher_is_better = higher_is_better
    ),
    class = "enfold_score"
  )
}

#' @export
print.enfold_score <- function(x, ...) {
  cat(sprintf(
    "enfold_score | %s | metalearner: %s\n",
    if (x$higher_is_better) "higher is better" else "lower is better",
    if (is.null(x$metalearner)) "none" else x$metalearner$name
  ))
  invisible(x)
}


# ── grd_random ────────────────────────────────────────────────────────────────

#' Random hyperparameter search grid constructor
#'
#' A \code{\link{make_grid_factory}}-produced constructor that builds an
#' \code{enfold_grid} using random hyperparameter sampling via
#' \code{\link{draw}}.
#'
#' @param name_prefix Character. Prefix for generated learner names
#'   (e.g. \code{"rf"} yields names like \code{"rf/num.trees=500,mtry=3"}).
#' @param learner_object An \code{enfold_learner}, \code{enfold_pipeline}, or
#'   \code{enfold_list} instance to search over.
#' @param parameters An \code{enfold_hyperparameters} object from
#'   \code{\link{specify_hyperparameters}}.
#' @param n_candidates Integer or \code{NULL}. Number of random draws.
#'   When \code{NULL} and all parameters are discrete, all valid
#'   combinations are evaluated.
#' @param seed Integer or \code{NULL}. Passed to \code{set.seed()} before
#'   drawing. \code{NULL} leaves the RNG state untouched.
#' @param directory \code{NULL} for a plain learner, or a character vector
#'   naming the target node inside a pipeline.
#' @return An \code{enfold_grid} object.
#' @seealso \code{\link{make_grid_factory}}, \code{\link{grd_early_stop}},
#'   \code{\link{grd_bayes}}
#' @examples
#' \dontrun{
#' params <- specify_hyperparameters(
#'   num.trees = c(100L, 500L),
#'   mtry      = c(2L, 4L)
#' )
#' grid <- grd_random("rf", lrn_ranger("rf"), params, n_candidates = 5L, seed = 42L)
#'
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(grid) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = NA) |>
#'   fit()
#' }
#' @export
grd_random <- make_grid_factory(
  search = function(hyperparams, name_prefix, learner_object, directory, x, y, folds) {
    if (!is.null(n_candidates)) {
      if (
        !is.numeric(n_candidates) ||
          length(n_candidates) != 1L ||
          n_candidates < 1L
      ) {
        stop("`n_candidates` must be a positive integer or NULL.", call. = FALSE)
      }
      n_candidates <- as.integer(n_candidates)
    }

    if (!is.null(seed)) set.seed(seed)

    all_discrete <- all(vapply(
      hyperparams,
      function(x) !inherits(x, "enfold_range"),
      logical(1L)
    ))

    if (!all_discrete && is.null(n_candidates)) {
      stop(
        "At least one hyperparameter is continuous and `n_candidates` is NULL. ",
        "Please specify `n_candidates`.",
        call. = FALSE
      )
    }

    combo_df <- draw(hyperparams, n = n_candidates)
    combo_list <- lapply(
      seq_len(nrow(combo_df)),
      function(i) as.list(combo_df[i, , drop = FALSE])
    )

    results <- list()
    for (combo in combo_list) {
      nm <- make_combo_name(name_prefix, combo)
      modified <- tryCatch(
        change_arguments(learner_object, directory, combo, name = nm),
        error = function(e) {
          warning(sprintf(
            "Grid '%s': change_arguments failed for '%s': %s",
            name_prefix, nm, conditionMessage(e)
          ))
          NULL
        }
      )
      if (is.null(modified)) next
      contrib <- tryCatch(
        cv_fit(modified, folds, x, y),
        error = function(e) {
          warning(sprintf(
            "Grid '%s': cv_fit failed for '%s': %s",
            name_prefix, nm, conditionMessage(e)
          ))
          NULL
        }
      )
      if (is.null(contrib) || !is.null(attr(contrib, "failed_learner"))) next
      results <- c(results, list(list(name = nm, combo = combo, contrib = contrib)))
    }
    results
  },
  n_candidates = NULL,
  seed = NULL
)


# ── grd_early_stop ────────────────────────────────────────────────────────────

#' Early-stopping hyperparameter search grid constructor
#'
#' A \code{\link{make_grid_factory}}-produced constructor that builds an
#' \code{enfold_grid} that evaluates candidates sequentially and stops
#' when no meaningful improvement has been observed for
#' \code{n_early_stop} consecutive candidates.
#'
#' @param name_prefix Character. Prefix for generated learner names.
#' @param learner_object An \code{enfold_learner}, \code{enfold_pipeline}, or
#'   \code{enfold_list} instance to search over.
#' @param parameters An \code{enfold_hyperparameters} object.
#' @param seed Integer or \code{NULL}. Passed to \code{set.seed()} before
#'   sampling candidates.
#' @param max_candidates Integer or \code{NULL}. Maximum number of candidates
#'   to evaluate. Required when any parameter is a continuous
#'   \code{enfold_range}.
#' @param n_early_stop Integer. Consecutive evaluations without relative
#'   improvement of at least \code{tol} before stopping. Default \code{10L}.
#' @param tol Numeric. Relative improvement threshold. Default \code{0.01}.
#' @param score An \code{enfold_score} object from \code{\link{make_score}}
#'   specifying the loss function, an optional metalearner for multi-output
#'   candidates, and the optimisation direction. Default
#'   \code{make_score(loss_gaussian())}.
#' @param directory \code{NULL} for a plain learner, or a character vector
#'   naming the target node inside a pipeline.
#' @return An \code{enfold_grid} object.
#' @seealso \code{\link{make_grid_factory}}, \code{\link{make_score}},
#'   \code{\link{grd_random}}, \code{\link{grd_bayes}}
#' @examples
#' \dontrun{
#' params <- specify_hyperparameters(num.trees = c(100L, 200L, 500L))
#' grid   <- grd_early_stop("rf", lrn_ranger("rf"), params, n_early_stop = 2L)
#'
#' # Pipeline with branching paths: supply a metalearner to combine them
#' pipe   <- make_pipeline(some_screener, list(lrn_glm("glm", gaussian()), lrn_mean("mean")))
#' params <- specify_hyperparameters(threshold = c(5L, 10L, 20L))
#' grid   <- grd_early_stop(
#'   "pipe_grid", pipe, params,
#'   score = make_score(loss_gaussian(), metalearner = mtl_selector("sel"))
#' )
#' }
#' @export
grd_early_stop <- make_grid_factory(
  search = function(hyperparams, name_prefix, learner_object, directory, x, y, folds) {
    if (inherits(score, "mtl_loss")) {
      stop(
        "`score` must be an `enfold_score` object, not an `mtl_loss`. ",
        "Wrap your loss function: `make_score(your_loss_fun)`.",
        call. = FALSE
      )
    }
    if (!inherits(score, "enfold_score")) {
      stop(
        "`score` must be an `enfold_score` object from `make_score()`.",
        call. = FALSE
      )
    }

    if (!is.null(max_candidates)) {
      max_candidates <- integer_checker(max_candidates, "max_candidates", return = TRUE)
    }
    n_early_stop <- integer_checker(n_early_stop, "n_early_stop", return = TRUE)
    if (!is.numeric(tol) || length(tol) != 1L || tol < 0) {
      stop("`tol` must be a single non-negative number.", call. = FALSE)
    }

    if (!is.null(seed)) set.seed(seed)

    all_discrete <- all(vapply(
      hyperparams,
      function(x) !inherits(x, "enfold_range"),
      logical(1L)
    ))

    if (!all_discrete && is.null(max_candidates)) {
      stop(
        "At least one hyperparameter is continuous and `max_candidates` is NULL. ",
        "Please specify `max_candidates`.",
        call. = FALSE
      )
    }

    combo_df <- draw(hyperparams, n = max_candidates)
    combo_list <- lapply(
      seq_len(nrow(combo_df)),
      function(i) as.list(combo_df[i, , drop = FALSE])
    )

    best_score <- if (score$higher_is_better) -Inf else Inf
    no_improve <- 0L
    results <- list()

    for (combo in combo_list) {
      nm <- make_combo_name(name_prefix, combo)
      modified <- tryCatch(
        change_arguments(learner_object, directory, combo, name = nm),
        error = function(e) {
          warning(sprintf(
            "Grid '%s': change_arguments failed for '%s': %s",
            name_prefix, nm, conditionMessage(e)
          ))
          NULL
        }
      )
      if (is.null(modified)) next
      contrib <- tryCatch(
        cv_fit(modified, folds, x, y),
        error = function(e) {
          warning(sprintf(
            "Grid '%s': cv_fit failed for '%s': %s",
            name_prefix, nm, conditionMessage(e)
          ))
          NULL
        }
      )
      if (is.null(contrib) || !is.null(attr(contrib, "failed_learner"))) next

      scalar <- evaluate_score(score, contrib, y)

      if (!is.numeric(scalar) || length(scalar) != 1L || is.na(scalar)) {
        stop(
          "grd_early_stop: `score` produced a non-scalar result. ",
          "Check that `score` is compatible with the learner output.",
          call. = FALSE
        )
      }

      improved <- if (is.infinite(best_score)) {
        TRUE
      } else if (score$higher_is_better) {
        (scalar - best_score) / (abs(best_score) + 1e-15) >= tol
      } else {
        (best_score - scalar) / (abs(best_score) + 1e-15) >= tol
      }

      if (improved) {
        best_score <- scalar
        no_improve <- 0L
      } else {
        no_improve <- no_improve + 1L
      }

      results <- c(results, list(list(name = nm, combo = combo, contrib = contrib)))

      if (no_improve >= n_early_stop) break
    }

    results
  },
  seed = NULL,
  max_candidates = NULL,
  n_early_stop = 10L,
  tol = 0.01,
  score = make_score(loss_gaussian(), metalearner = mtl_selector("sel"))
)


# ── grd_bayes ─────────────────────────────────────────────────────────────────

#' Bayesian hyperparameter search grid constructor
#'
#' A \code{\link{make_grid_factory}}-produced constructor that builds an
#' \code{enfold_grid} using Gaussian-process Bayesian optimisation to
#' select candidate combinations.
#'
#' @param name_prefix Character. Prefix for generated learner names.
#' @param learner_object An \code{enfold_learner}, \code{enfold_pipeline}, or
#'   \code{enfold_list} instance to search over.
#' @param parameters An \code{enfold_hyperparameters} object. All parameters
#'   must be continuous \code{enfold_range} objects.
#' @param n_init Integer. Number of random initial evaluations before
#'   Bayesian updating begins. Default \code{5L}.
#' @param n_iter Integer. Number of Bayesian optimisation iterations after
#'   the random phase. Default \code{10L}.
#' @param score An \code{enfold_score} object from \code{\link{make_score}}
#'   specifying the loss function, an optional metalearner for multi-output
#'   candidates, and the optimisation direction. Required; no default is
#'   provided.
#' @param seed Integer or \code{NULL}. Passed to \code{set.seed()} before
#'   the optimisation run.
#' @param directory \code{NULL} for a plain learner, or a character vector
#'   naming the target node inside a pipeline.
#' @return An \code{enfold_grid} object.
#' @details
#' Requires the \pkg{rBayesianOptimization} package. Discrete parameter
#' vectors are not supported; define all ranges as \code{enfold_range}
#' objects via \code{\link{make_range}}.
#' @seealso \code{\link{make_grid_factory}}, \code{\link{make_score}},
#'   \code{\link{grd_random}}, \code{\link{grd_early_stop}}
#' @examples
#' \dontrun{
#' params <- specify_hyperparameters(lambda = make_range(1e-4, 10))
#' grid   <- grd_bayes(
#'   "en", lrn_glmnet("en", gaussian()), params,
#'   score = make_score(loss_gaussian())
#' )
#'
#' # Multi-output pipeline: supply a metalearner
#' grid <- grd_bayes(
#'   "pipe", my_pipeline, params,
#'   score = make_score(loss_gaussian(), metalearner = mtl_selector("sel"))
#' )
#' }
#' @export
grd_bayes <- make_grid_factory(
  search = function(hyperparams, name_prefix, learner_object, directory, x, y, folds) {
    if (inherits(score, "mtl_loss")) {
      stop(
        "`score` must be an `enfold_score` object, not an `mtl_loss`. ",
        "Wrap your loss function: `make_score(your_loss_fun)`.",
        call. = FALSE
      )
    }
    if (!inherits(score, "enfold_score")) {
      stop(
        "`score` must be an `enfold_score` object from `make_score()`.",
        call. = FALSE
      )
    }

    not_range <- !vapply(hyperparams, inherits, logical(1L), "enfold_range")
    if (any(not_range)) {
      stop(
        sprintf(
          paste0(
            "grd_bayes() requires all parameters to be continuous enfold_range objects. ",
            "Non-range parameters: %s."
          ),
          paste(names(hyperparams)[not_range], collapse = ", ")
        ),
        call. = FALSE
      )
    }

    if (!requireNamespace("rBayesianOptimization", quietly = TRUE)) {
      stop(
        "Package 'rBayesianOptimization' is required for grd_bayes().",
        call. = FALSE
      )
    }

    if (!is.null(seed)) set.seed(seed)

    bounds <- lapply(hyperparams, function(p) c(p$min, p$max))

    bayes_fn <- function(...) {
      combo <- list(...)
      nm <- make_combo_name(name_prefix, combo)
      modified <- tryCatch(
        change_arguments(learner_object, directory, combo, name = nm),
        error = function(e) NULL
      )
      if (is.null(modified)) return(list(Score = -Inf))
      contrib <- tryCatch(
        cv_fit(modified, folds, x, y),
        error = function(e) NULL
      )
      if (is.null(contrib) || !is.null(attr(contrib, "failed_learner"))) {
        return(list(Score = -Inf))
      }
      scalar <- evaluate_score(score, contrib, y)
      list(Score = if (score$higher_is_better) scalar else -scalar)
    }

    opt <- rBayesianOptimization::BayesianOptimization(
      FUN = bayes_fn,
      bounds = bounds,
      init_points = as.integer(n_init),
      n_iter = as.integer(n_iter),
      verbose = FALSE
    )

    best_combo <- as.list(opt$Best_Par)
    best_nm    <- make_combo_name(name_prefix, best_combo)
    best_mod   <- change_arguments(learner_object, directory, best_combo, name = best_nm)
    best_contrib <- cv_fit(best_mod, folds, x, y)

    list(list(name = best_nm, combo = best_combo, contrib = best_contrib))
  },
  n_init = 5L,
  n_iter = 10L,
  score,
  seed = NULL
)


# ── Internal helpers ───────────────────────────────────────────────────────────

# Apply an enfold_score to a named list of out-of-fold predictions (contrib)
# and return a scalar mean loss. For single-output contrib, scores directly.
# For multi-output contrib, requires score$metalearner — errors if NULL.
evaluate_score <- function(score, contrib, y) {
  if (length(contrib) == 1L) {
    p   <- contrib[[1L]]
    idx <- attr(p, "indices")
    return(mean(score$loss_function$loss_fun(subset_y(y, idx), p)))
  }
  if (is.null(score$metalearner)) {
    stop(
      "The grid candidate produced ", length(contrib), " output stream(s) but ",
      "`score$metalearner` is NULL. ",
      "Provide a metalearner in make_score() to combine multi-output candidates.",
      call. = FALSE
    )
  }
  idx       <- attr(contrib[[1L]], "indices")
  y_sub     <- structure(subset_y(y, idx), indices = idx)
  fitted_mtl <- fit(score$metalearner, x = contrib, y = y_sub)
  preds     <- stats::predict(fitted_mtl, contrib)
  mean(score$loss_function$loss_fun(subset_y(y, idx), preds))
}
