# ── grd_random ────────────────────────────────────────────────────────────

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


# ── grd_early_stop ────────────────────────────────────────────────────────

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
#' @param loss_fun An \code{mtl_loss} object (e.g. \code{loss_gaussian()})
#'   used to compute candidate loss from out-of-fold predictions.
#' @param directory \code{NULL} for a plain learner, or a character vector
#'   naming the target node inside a pipeline.
#' @return An \code{enfold_grid} object.
#' @seealso \code{\link{make_grid_factory}}, \code{\link{grd_random}},
#'   \code{\link{grd_bayes}}
#' @examples
#' \dontrun{
#' params <- specify_hyperparameters(num.trees = c(100L, 200L, 500L))
#' grid   <- grd_early_stop("rf", lrn_ranger("rf"), params, n_early_stop = 2L)
#' }
#' @export
grd_early_stop <- make_grid_factory(
  search = function(hyperparams, name_prefix, learner_object, directory, x, y, folds) {
    if (!is.null(max_candidates)) {
      max_candidates <- integer_checker(max_candidates, "max_candidates", return = TRUE)
    }
    n_early_stop <- integer_checker(n_early_stop, "n_early_stop", return = TRUE)
    if (!is.numeric(tol) || length(tol) != 1L || tol < 0) {
      stop("`tol` must be a single non-negative number.", call. = FALSE)
    }
    if (!inherits(loss_fun, "mtl_loss")) {
      stop("`loss_fun` must be a loss object created by loss_*().", call. = FALSE)
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

    best_loss <- Inf
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

      p <- extract_best_preds_for_loss(contrib, y, loss_fun)
      idx <- attr(p, "indices")
      loss <- mean(compute_loss(loss_fun, subset_y(y, idx), p))

      if (!is.numeric(loss) || length(loss) != 1L || is.na(loss)) {
        stop(
          "grd_early_stop requires a scalar numeric loss from `loss_fun`. ",
          "Check that `loss_fun` is compatible with the learner output.",
          call. = FALSE
        )
      }

      improved <- if (is.infinite(best_loss)) {
        TRUE
      } else {
        best_loss > 0 && (best_loss - loss) / best_loss >= tol
      }

      if (improved) {
        best_loss <- loss
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
  loss_fun = loss_gaussian()
)


# ── grd_bayes ─────────────────────────────────────────────────────────────

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
#' @param score_fn A \code{function(result, y) -> numeric(1)} that extracts a
#'   scalar score (higher is better) from a candidate result. \code{result}
#'   carries fields \code{$name} (learner name string), \code{$combo} (named
#'   list of hyperparameter values), and \code{$contrib} (named list of
#'   out-of-fold predictions, one entry per terminal output, each with
#'   \code{attr(., "indices")} carrying the corresponding row indices).
#'   \code{y} is the full outcome vector or matrix passed to the grid search.
#' @param seed Integer or \code{NULL}. Passed to \code{set.seed()} before
#'   the optimisation run.
#' @param directory \code{NULL} for a plain learner, or a character vector
#'   naming the target node inside a pipeline.
#' @return An \code{enfold_grid} object.
#' @details
#' Requires the \pkg{rBayesianOptimization} package. Discrete parameter
#' vectors are not supported; define all ranges as \code{enfold_range}
#' objects via \code{\link{make_range}}.
#' @seealso \code{\link{make_grid_factory}}, \code{\link{grd_random}},
#'   \code{\link{grd_early_stop}}
#' @examples
#' \dontrun{
#' params <- specify_hyperparameters(lambda = make_range(1e-4, 10))
#' grid   <- grd_bayes(
#'   "en", lrn_glmnet("en", gaussian()), params,
#'   n_init   = 3L,
#'   n_iter   = 7L,
#'   score_fn = function(result, y) {
#'     p   <- result$contrib[[1L]]
#'     idx <- attr(p, "indices")
#'     -mean((p - y[idx])^2)
#'   }
#' )
#' }
#' @export
grd_bayes <- make_grid_factory(
  search = function(hyperparams, name_prefix, learner_object, directory, x, y, folds) {
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

    results_env <- new.env(parent = emptyenv())
    results_env$collected <- list()

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
      result <- list(name = nm, combo = combo, contrib = contrib)
      score <- score_fn(result, y)
      results_env$collected <- c(results_env$collected, list(result))
      list(Score = score)
    }

    rBayesianOptimization::BayesianOptimization(
      FUN = bayes_fn,
      bounds = bounds,
      init_points = as.integer(n_init),
      n_iter = as.integer(n_iter),
      verbose = FALSE
    )

    results_env$collected
  },
  n_init = 5L,
  n_iter = 10L,
  score_fn,
  seed = NULL
)


# ── Internal helpers ───────────────────────────────────────────────────────

# For early-stopping search engines that need a scalar loss: pick the single
# prediction stream most representative of the candidate.
# Single-output contrib -> contrib[[1]].
# Multi-output -> the stream with the lowest mean loss under loss_fun.
extract_best_preds_for_loss <- function(contrib, y, loss_fun) {
  if (length(contrib) == 1L) return(contrib[[1L]])
  losses <- vapply(contrib, function(p) {
    idx <- attr(p, "indices")
    mean(compute_loss(loss_fun, subset_y(y, idx), p))
  }, numeric(1L))
  contrib[[which.min(losses)]]
}
