# ── change_arguments ──────────────────────────────────────────────────────

#' Modify hyperparameters of a learner or a node inside a pipeline
#'
#' Returns a modified copy of \code{object} with hyperparameters updated
#' according to \code{combo}. Does not mutate the original.
#'
#' For a plain \code{enfold_learner}, \code{directory} must be \code{NULL}.
#' For an \code{enfold_pipeline}, \code{directory} is a character vector
#' giving the name of the target node (and optionally the names of nested
#' nodes for multi-level pipelines, e.g. \code{c("outer", "inner")}).
#'
#' @param object An \code{enfold_learner} or \code{enfold_pipeline}.
#' @param directory \code{NULL} for a plain learner; a non-empty character
#'   vector navigating to the target node for pipelines.
#' @param combo A named list of hyperparameter values to apply. Merged into
#'   the existing parameters via \code{modifyList}.
#' @param name Optional new name for the returned object.
#' @return A modified copy of \code{object}.
#' @export
change_arguments <- function(object, directory, combo, name = NULL) {
  UseMethod("change_arguments")
}

#' @export
change_arguments.enfold_learner <- function(
  object,
  directory,
  combo,
  name = NULL
) {
  if (!is.null(directory)) {
    stop("'directory' must be NULL for a plain enfold_learner.", call. = FALSE)
  }
  new_p <- utils::modifyList(get_params(object), combo)
  orig_fit <- get_original_fit(object)
  orig_preds <- get_original_preds(object)
  old_env <- environment(object$fit)

  # New closure env with updated params; preserve parent chain
  new_env <- list2env(new_p, parent = parent.env(old_env))
  # Re-seat user functions in new env (environment<- uses copy-on-modify semantics)
  environment(orig_fit) <- new_env
  environment(orig_preds) <- new_env
  new_env$fit <- orig_fit
  new_env$preds <- orig_preds

  # Recreate thin wrappers (mirrors make_learner_factory pattern)
  new_wrapped_fit <- function(x, y) fit(x = x, y = y)
  environment(new_wrapped_fit) <- new_env
  new_wrapped_preds <- function(object, data) {
    preds(object = object, data = data)
  }
  environment(new_wrapped_preds) <- new_env

  new_obj <- object
  new_obj$fit <- new_wrapped_fit
  new_obj$preds <- new_wrapped_preds
  if (!is.null(name)) {
    new_obj$name <- name
  }
  new_obj
}

#' @export
change_arguments.enfold_pipeline <- function(
  object,
  directory,
  combo,
  name = NULL
) {
  if (is.null(directory) || length(directory) == 0L) {
    stop(
      "'directory' must be a non-empty character vector for enfold_pipeline.",
      call. = FALSE
    )
  }
  target <- directory[[1L]]
  sub_dir <- if (length(directory) > 1L) directory[-1L] else NULL

  found <- FALSE
  new_stages <- vector("list", length(object$stages))
  for (s in seq_along(object$stages)) {
    new_stages[[s]] <- vector("list", length(object$stages[[s]]))
    for (j in seq_along(object$stages[[s]])) {
      node <- object$stages[[s]][[j]]
      if (!is_passthrough(node) && node$name == target) {
        found <- TRUE
        new_stages[[s]][[j]] <- if (!is.null(sub_dir)) {
          change_arguments(node, sub_dir, combo, name = NULL)
        } else {
          # Rename node with combo so pipeline paths are distinguishable per combo
          change_arguments(
            node,
            NULL,
            combo,
            name = make_combo_name(node$name, combo)
          )
        }
      } else {
        new_stages[[s]][[j]] <- node
      }
    }
  }

  if (!found) {
    stop(
      sprintf("Node '%s' not found in any pipeline stage.", target),
      call. = FALSE
    )
  }

  new_paths <- enumerate_paths(new_stages)
  new_nms <- vapply(
    new_paths,
    function(p) path_name(new_stages, p),
    character(1L)
  )
  new_paths_df <- build_paths_df(new_stages, new_paths)
  structure(
    list(
      stages = new_stages,
      paths = new_paths,
      path_names = new_nms,
      paths_df = new_paths_df
    ),
    class = "enfold_pipeline"
  )
}


# ── make_grid_factory ─────────────────────────────────────────────────────

#' Create a grid search factory
#'
#' Returns a constructor function for \code{enfold_grid} objects. The
#' constructor captures \code{search} in a minimal closure together with
#' whatever configuration parameters you declare via \code{...}, keeping
#' the closure environment small.
#'
#' This is the grid-search analogue of \code{\link{make_learner_factory}}.
#' The built-in constructors \code{\link{grd_random}},
#' \code{\link{grd_early_stop}}, and \code{\link{grd_bayes}} are produced
#' by this factory.
#'
#' @param search A function with arguments
#'   \code{(hyperparams, name_prefix, learner_object, directory, x, y, folds)}
#'   that implements the search strategy. \code{hyperparams} is an
#'   \code{enfold_hyperparameters} object; the remaining arguments give the search
#'   function direct access to the grid's learner object, pipeline directory,
#'   the full data, and the cross-validation folds. The body of \code{search}
#'   may freely reference any configuration parameter declared in \code{...}.
#'   It must return a list of result objects, each with fields \code{$name}
#'   (combo name string), \code{$combo} (named list of hyperparameter values),
#'   and \code{$contrib} (named list of out-of-fold predictions for that
#'   candidate, one entry per terminal output, each with
#'   \code{attr(., "indices")} carrying the corresponding row indices).
#' @param ... Configuration parameters. Bare names (e.g. \code{seed})
#'   become required arguments of the returned constructor; named values
#'   (e.g. \code{seed = NULL}) become optional arguments with defaults.
#'   Names \code{name_prefix}, \code{learner_object}, \code{parameters},
#'   \code{directory}, and \code{search} are reserved and will cause an error.
#' @return A constructor
#'   \code{function(name_prefix, learner_object, parameters, ..., directory = NULL)}
#'   that builds an \code{enfold_grid} with the baked-in search engine.
#' @details
#' The closure environment of the resulting search engine contains only the
#' configuration parameter values and the \code{search} function, nothing
#' else.
#' @seealso \code{\link{make_learner_factory}}, \code{\link{grd_random}},
#'   \code{\link{grd_early_stop}}, \code{\link{grd_bayes}}
#' @examples
#' # A custom exhaustive search (evaluates every discrete combination).
#' # Inside the search function, enfold's internal cross-validation helpers
#' # are accessible automatically because the function runs in the package
#' # namespace.
#' grd_exhaustive <- make_grid_factory(
#'   search = function(hyperparams, name_prefix, learner_object, directory, x, y, folds) {
#'
#'     combo_df <- draw(hyperparams, n = NULL)
#'     results  <- list()
#'
#'     for (i in seq_len(nrow(combo_df))) {
#'       combo <- as.list(combo_df[i, , drop = FALSE])
#'       nm <- paste0(name_prefix, "/", paste(names(combo), combo, sep = "=", collapse = ","))
#'       modified <- change_arguments(learner_object, directory, combo, name = nm)
#'       contrib <- tryCatch(cv_fit(modified, folds, x, y), error = function(e) NULL)
#'       if (is.null(contrib) || !is.null(attr(contrib, "failed_learner"))) next
#'       results <- c(results, list(list(name = nm, combo = combo, contrib = contrib)))
#'     }
#'     results
#'   }
#' )
#' \dontrun{
#' params <- specify_hyperparameters(alpha = c(0, 0.5, 1))
#' grid   <- grd_exhaustive("en", lrn_glmnet("en", gaussian()), params)
#' }
#' @export
make_grid_factory <- function(search, ...) {
  factory_ns <- parent.env(environment())
  raw_dots <- substitute(list(...))[-1]

  # Validate search function signature
  required_search_args <- c(
    "hyperparams",
    "name_prefix",
    "learner_object",
    "directory",
    "x",
    "y",
    "folds"
  )
  extra_search_args <- setdiff(names(formals(search)), required_search_args)
  if (length(extra_search_args) > 0) {
    stop(
      "search must only have (hyperparams, name_prefix, learner_object, directory, x, y, folds) ",
      "as arguments. Every configuration parameter you pass into make_grid_factory() is ",
      "available inside search() via the closure, not as function arguments.",
      call. = FALSE
    )
  }

  # Check for reserved names in config params
  bad_names <- c(
    "name_prefix",
    "learner_object",
    "parameters",
    "search",
    "directory"
  )
  clashing <- intersect(names(raw_dots), bad_names)
  if (length(clashing) > 0) {
    stop(
      "It is not allowed to pass arguments to 'make_grid_factory' ",
      "named 'name_prefix', 'learner_object', 'parameters', 'directory', or 'search'. Clashing: ",
      paste(clashing, collapse = ", "),
      call. = FALSE
    )
  }

  # Build constructor formals: fixed grid args first, then config params
  constr_args <- alist(
    name_prefix = ,
    learner_object = ,
    parameters = ,
    directory = NULL
  )

  for (i in seq_along(raw_dots)) {
    arg_name <- names(raw_dots)[[i]]
    if (is.null(arg_name) || arg_name == "") {
      actual_name <- as.character(raw_dots[[i]])
      new_arg <- alist(x = )
      names(new_arg) <- actual_name
      constr_args <- c(constr_args, new_arg)
    } else {
      constr_args[arg_name] <- list(raw_dots[[i]])
    }
  }

  constructor <- function() {}
  config_param_names <- setdiff(
    names(constr_args),
    c("name_prefix", "learner_object", "parameters", "directory")
  )
  formals(constructor) <- constr_args

  body(constructor) <- bquote({
    p <- mget(.(config_param_names), envir = environment())

    # Minimal environment: only config params and search function
    closure_env <- list2env(p, parent = .(factory_ns))
    environment(search) <- closure_env
    closure_env$search <- search

    wrapped_search <- function(
      hyperparams,
      name_prefix,
      learner_object,
      directory,
      x,
      y,
      folds
    ) {
      search(hyperparams, name_prefix, learner_object, directory, x, y, folds)
    }
    environment(wrapped_search) <- closure_env

    make_grid(
      name_prefix,
      learner_object,
      parameters,
      search_engine = wrapped_search,
      directory = directory
    )
  })

  constructor
}


# ── make_grid ──────────────────────────────────────────────────────────────
# Internal constructor used by make_grid_factory. Not exported.
# Prefer the grd_* constructors (grd_random, grd_early_stop, grd_bayes) or
# make_grid_factory() for custom grid searches.

make_grid <- function(
  name_prefix,
  learner_object,
  parameters,
  search_engine,
  directory = NULL
) {
  if (
    !is.character(name_prefix) ||
      length(name_prefix) != 1L ||
      nchar(name_prefix) == 0L
  ) {
    stop("`name_prefix` must be a non-empty character string.", call. = FALSE)
  }
  if (
    !inherits(
      learner_object,
      c("enfold_learner", "enfold_pipeline", "enfold_list")
    )
  ) {
    stop(
      "`learner_object` must be an enfold_learner, enfold_pipeline, or enfold_list.",
      call. = FALSE
    )
  }
  if (!inherits(parameters, "enfold_hyperparameters")) {
    stop(
      "`parameters` must be an `enfold_hyperparameters` object from `specify_hyperparameters()`.",
      call. = FALSE
    )
  }
  if (!is.function(search_engine)) {
    stop(
      "`search_engine` must be a function.",
      call. = FALSE
    )
  }
  if (
    !is.null(directory) && (!is.character(directory) || length(directory) == 0L)
  ) {
    stop(
      "`directory` must be NULL or a non-empty character vector.",
      call. = FALSE
    )
  }

  structure(
    list(
      name = name_prefix,
      learner_object = learner_object,
      hyperparams = parameters,
      search_engine = search_engine,
      directory = directory
    ),
    class = "enfold_grid"
  )
}


# ── fit.enfold_grid ────────────────────────────────────────────────────────

#' @export
fit.enfold_grid <- function(object, x, y, ...) {
  stop(
    "Cannot call fit() on an enfold_grid directly. ",
    "Pass it to add_learners() and let fit.enfold_task() with inner_cv non-NA handle it.",
    call. = FALSE
  )
}

# ── internal helpers ──────────────────────────────────────────────────────

# Build the resolved learner from grid + winning results.
# Single winner → the modified object directly.
# Multiple pipeline winners → expand the target stage with all winner nodes.
# Multiple plain-learner winners → bundle into enfold_list.
build_resolved_learner <- function(grid, results) {
  is_pipeline <- inherits(grid$learner_object, "enfold_pipeline")

  if (length(results) == 1L) {
    change_arguments(
      grid$learner_object,
      grid$directory,
      results[[1L]]$combo,
      name = results[[1L]]$name
    )
  } else if (is_pipeline && !is.null(grid$directory)) {
    target_node <- find_stage_node(grid$learner_object, grid$directory[[1L]])
    winner_nodes <- lapply(results, function(r) {
      change_arguments(
        target_node,
        NULL,
        r$combo,
        name = make_combo_name(target_node$name, r$combo)
      )
    })
    expand_pipeline_at_node(
      grid$learner_object,
      grid$directory[[1L]],
      winner_nodes
    )
  } else {
    modified_lrns <- lapply(results, function(r) {
      change_arguments(
        grid$learner_object,
        grid$directory,
        r$combo,
        name = r$name
      )
    })
    make_modified_list_learner(grid$name, modified_lrns)
  }
}

# Extract the first node with the given name from a pipeline's stages.
find_stage_node <- function(pipeline, node_name) {
  for (stage in pipeline$stages) {
    for (node in stage) {
      if (!is_passthrough(node) && node$name == node_name) return(node)
    }
  }
  stop(sprintf("Node '%s' not found in pipeline.", node_name), call. = FALSE)
}

# Replace the single node named `target` in a pipeline with `replacement_nodes`
# (a list). All other stage entries and all other stages are preserved.
expand_pipeline_at_node <- function(pipeline, target, replacement_nodes) {
  new_stages <- lapply(pipeline$stages, function(stage) {
    target_idx <- which(vapply(
      stage,
      function(n) !is_passthrough(n) && n$name == target,
      logical(1L)
    ))
    if (length(target_idx) == 0L) {
      return(stage)
    }
    c(stage[setdiff(seq_along(stage), target_idx)], replacement_nodes)
  })

  new_paths <- enumerate_paths(new_stages)
  new_nms <- vapply(
    new_paths,
    function(p) path_name(new_stages, p),
    character(1L)
  )
  new_paths_df <- build_paths_df(new_stages, new_paths)
  structure(
    list(
      stages = new_stages,
      paths = new_paths,
      path_names = new_nms,
      paths_df = new_paths_df
    ),
    class = "enfold_pipeline"
  )
}

# Bundle already-modified learner instances into an enfold_list.
# Replaces the old make_grid_list_learner (which reconstructed from constructor+combos).
make_modified_list_learner <- function(name_prefix, modified_learners) {
  lrn_names <- vapply(modified_learners, `[[`, character(1L), "name")
  short_nms <- sub(paste0("^", name_prefix, "/"), "", lrn_names)

  fit_fn <- function(x, y) {
    models <- lapply(modified_learners, function(lrn) fit(lrn, x, y))
    stats::setNames(models, short_nms)
  }
  preds_fn <- function(object, data) {
    stats::setNames(
      lapply(names(object), function(nm) {
        stats::predict(object[[nm]], newdata = data)
      }),
      names(object)
    )
  }
  structure(
    list(name = name_prefix, fit = fit_fn, preds = preds_fn),
    class = c("enfold_list", "enfold_learner")
  )
}


# ── print methods ──────────────────────────────────────────────────────────

#' @export
print.enfold_range <- function(x, ...) {
  cat(sprintf("enfold_range [%g, %g]\n", x$min, x$max))
  invisible(x)
}

#' @export
print.enfold_grid <- function(x, ...) {
  cat(sprintf("enfold_grid | prefix: '%s'\n", x$name))
  cat("  Hyperparameters:\n")
  for (nm in names(x$hyperparams)) {
    val <- x$hyperparams[[nm]]
    if (inherits(val, "enfold_range")) {
      cat(sprintf("    %s: enfold_range [%g, %g]\n", nm, val$min, val$max))
    } else {
      cat(sprintf("    %s: %s\n", nm, paste(val, collapse = ", ")))
    }
  }
  invisible(x)
}


# Build a canonical learner name from a grid prefix and a param combo.
# Format: "prefix/param1=val1,param2=val2"
make_combo_name <- function(prefix, params) {
  parts <- vapply(
    names(params),
    function(nm) {
      val <- params[[nm]]
      sprintf("%s=%s", nm, paste(val, collapse = "+"))
    },
    character(1L)
  )
  paste0(prefix, "/", paste(parts, collapse = ","))
}
