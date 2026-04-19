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
#' ## Calling the produced constructor
#'
#' The produced constructor accepts two usage forms:
#'
#' \describe{
#'   \item{Bare-grid form}{
#'     \code{grd_*(bare_grid, ...)} where \code{bare_grid} is an
#'     \code{enfold_grid} without a search engine, created by passing
#'     \code{parameters =} to a learner constructor. The learner, its
#'     hyperparameter spec, and the internal \code{directory} are extracted
#'     from the bare grid automatically.
#'   }
#'   \item{Pipeline form}{
#'     \code{grd_*(pipeline, ...)} where \code{pipeline} is an
#'     \code{enfold_pipeline} that contains exactly one embedded bare grid node.
#'     All needed information is harvested automatically.
#'   }
#' }
#'
#' @param search A function with arguments \code{(search_space, learner, x, y, folds)}
#'   that implements the search strategy. Call \code{apply_combo(learner, combo)} to
#'   instantiate concrete learners from sampled parameter combinations.
#' @param ... Configuration parameters for the search engine.
#' @return A constructor that builds an \code{enfold_grid} with the baked-in
#'   search engine.
#' @seealso \code{\link{make_learner_factory}}, \code{\link{grd_random}},
#'   \code{\link{grd_early_stop}}, \code{\link{grd_bayes}}
#' @export
make_grid_factory <- function(search, ...) {
  factory_ns <- parent.env(environment())
  raw_dots <- substitute(list(...))[-1]

  # Validate search function signature
  required_search_args <- c("search_space", "learner", "x", "y", "folds")
  extra_search_args <- setdiff(names(formals(search)), required_search_args)
  if (length(extra_search_args) > 0) {
    stop(
      "search must only have (search_space, learner, x, y, folds) as arguments. ",
      "Every configuration parameter passed to make_grid_factory() ",
      "is available via the closure.",
      call. = FALSE
    )
  }

  # Check for reserved names in config params
  bad_names <- c("grid", "search")
  clashing <- intersect(names(raw_dots), bad_names)
  if (length(clashing) > 0) {
    stop(
      "It is not allowed to pass arguments to 'make_grid_factory' ",
      "named 'grid' or 'search'. Clashing: ",
      paste(clashing, collapse = ", "),
      call. = FALSE
    )
  }

  constr_args <- alist(grid = )

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
  config_param_names <- setdiff(names(constr_args), "grid")
  formals(constructor) <- constr_args

  body(constructor) <- bquote({
    if (inherits(grid, "enfold_grid")) {
      if (!is.null(grid$search_engine)) {
        stop(
          "Cannot add a search engine to an enfold_grid that already has one.",
          call. = FALSE
        )
      }
      parameters <- grid$hyperparams
      learner_object <- grid$learner_object
      grid_name <- grid$name
    } else if (inherits(grid, "enfold_pipeline")) {
      extracted <- extract_bare_grid_from_pipeline(grid)
      grid_name <- extracted$name
      parameters <- extracted$parameters
      learner_object <- grid # original pipeline stored directly
    } else {
      stop(
        "The first argument to a grd_*() constructor must be a bare enfold_grid ",
        "(created via lrn_*(name, parameters = ...)) or an enfold_pipeline ",
        "containing at least one embedded bare grid node.",
        call. = FALSE
      )
    }

    p <- mget(.(config_param_names), envir = environment())

    closure_env <- list2env(p, parent = .(factory_ns))
    search_copy <- search
    environment(search_copy) <- closure_env
    closure_env$search <- search_copy

    wrapped_search <- function(search_space, learner, x, y, folds) {
      search(search_space, learner, x, y, folds)
    }
    environment(wrapped_search) <- closure_env

    make_grid(
      grid_name,
      learner_object,
      parameters,
      search_engine = wrapped_search
    )
  })

  constructor
}


# ── make_grid ──────────────────────────────────────────────────────────────
# Internal constructor. Prefer grd_* or make_bare_grid.

make_grid <- function(
  name_prefix,
  learner_object,
  parameters,
  search_engine = NULL
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
  if (!is.null(search_engine) && !is.function(search_engine)) {
    stop("`search_engine` must be a function or NULL.", call. = FALSE)
  }

  structure(
    list(
      name = name_prefix,
      learner_object = learner_object,
      hyperparams = parameters,
      search_engine = search_engine
    ),
    class = "enfold_grid"
  )
}

# Internal: create an enfold_grid without a search engine (bare grid).
# Created when a learner constructor is called with parameters = specify_hyperparameters(...).
make_bare_grid <- function(name, learner_object, parameters) {
  make_grid(name, learner_object, parameters, search_engine = NULL)
}


# ── fit.enfold_grid ────────────────────────────────────────────────────────

#' @export
fit.enfold_grid <- function(object, x, y, ...) {
  # Check that passed parameters are all discrete (no ranges) since we have no search engine to resolve them
  if (
    any(
      unlist(
        lapply(
          object$hyperparams,
          function(p) {
            inherits(p, "enfold_range") ||
              (inherits(p, "enfold_mixture") && !p$discrete_mixture)
          }
        )
      )
    )
  ) {
    stop(
      "Cannot fit an enfold_grid with hyperparameter ranges without a search engine.\n",
      "Use a grd_* constructor or a custom search engine.",
      call. = FALSE
    )
  }

  if (!is.null(object$search_engine)) {
    stop(
      "fit() is disabled for grids with a search engine. ",
      "Pass it to add_learners() and let fit.enfold_task() handle it.",
      call. = FALSE
    )
  }
  # draw() with n = NULL errors if any param is enfold_range, as intended
  combo_df <- draw(object$hyperparams, n = NULL)
  n_combos <- nrow(combo_df)
  if (n_combos == 0L) {
    stop("No valid hyperparameter combinations to fit.", call. = FALSE)
  }

  fitted_models <- vector("list", n_combos)
  nms <- character(n_combos)
  for (i in seq_len(n_combos)) {
    combo <- combo_row(combo_df, i)
    lrn <- apply_combo(object, combo)
    nms[[i]] <- lrn$name
    fitted_models[[i]] <- fit(lrn, x, y)
  }
  names(fitted_models) <- nms
  structure(
    list(name = object$name, models = fitted_models),
    class = "enfold_grid_fitted"
  )
}

#' @export
predict.enfold_grid_fitted <- function(object, newdata, ...) {
  lapply(object$models, function(m) stats::predict(m, newdata = newdata))
}

#' @export
print.enfold_grid_fitted <- function(x, ...) {
  cat(sprintf(
    "enfold_grid_fitted | '%s' | %d model(s)\n",
    x$name,
    length(x$models)
  ))
  for (nm in names(x$models)) {
    cat(sprintf("  %s\n", nm))
  }
  invisible(x)
}


# ── internal helpers ──────────────────────────────────────────────────────

# Build the resolved learner from grid + winning results.
# Single winner (any)          → the modified object directly.
# Multiple winners, plain      → bundle into enfold_list.
# Multiple winners, single-node pipeline → expand the target stage with all winner nodes.
# Multiple winners, multi-node pipeline  → bundle complete winner pipelines as enfold_list.
build_resolved_learner <- function(grid, results) {
  is_pipeline <- inherits(grid$learner_object, "enfold_pipeline")
  is_multi_node <- is_pipeline &&
    any(grepl("/", names(grid$hyperparams), fixed = TRUE))

  if (length(results) == 1L) {
    apply_combo(grid, results[[1L]]$combo)
  } else if (is_multi_node) {
    pipelines <- lapply(results, function(r) apply_combo(grid, r$combo))
    combo_names <- vapply(
      results,
      function(r) make_combo_name(grid$name, r$combo),
      character(1L)
    )
    make_modified_list_pipeline(grid$name, pipelines, combo_names)
  } else if (is_pipeline) {
    bare_node <- find_bare_grid_node(grid$learner_object, grid$name)
    winner_nodes <- lapply(results, function(r) apply_combo(bare_node, r$combo))
    expand_pipeline_at_node(grid$learner_object, grid$name, winner_nodes)
  } else {
    modified <- lapply(results, function(r) apply_combo(grid, r$combo))
    make_modified_list_learner(grid$name, modified)
  }
}

# Find the first bare enfold_grid node (no search engine) with a given name.
find_bare_grid_node <- function(pipeline, node_name) {
  for (stage in pipeline$stages) {
    for (node in stage) {
      if (
        inherits(node, "enfold_grid") &&
          is.null(node$search_engine) &&
          node$name == node_name
      ) {
        return(node)
      }
    }
  }
  stop(
    sprintf("Bare grid node '%s' not found in pipeline.", node_name),
    call. = FALSE
  )
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
make_modified_list_learner <- function(name_prefix, modified_learners) {
  lrn_names <- vapply(modified_learners, `[[`, character(1L), "name")
  short_nms <- sub(paste0("^", name_prefix, "/"), "", lrn_names)

  fit_fn <- function(x, y) {
    models <- lapply(modified_learners, function(lrn) fit(lrn, x, y))
    stats::setNames(models, short_nms)
  }
  preds_fn <- function(object, data) {
    out <- list()
    for (nm in names(object)) {
      inner_preds <- stats::predict(object[[nm]], newdata = data)
      if (is.list(inner_preds) && !is.data.frame(inner_preds)) {
        for (inner_nm in names(inner_preds)) {
          out[[paste0(nm, "/", inner_nm)]] <- inner_preds[[inner_nm]]
        }
      } else {
        out[[nm]] <- inner_preds
      }
    }
    out
  }
  structure(
    list(name = name_prefix, fit = fit_fn, preds = preds_fn),
    class = c("enfold_list", "enfold_learner")
  )
}

# Bundle multiple concrete pipelines (from a multi-node grid search) into an
# enfold_list. fit_fn trains each independently; preds_fn flattens all path
# predictions into one named list matching the flat CV output from the search.
make_modified_list_pipeline <- function(name_prefix, pipelines, combo_names) {
  fit_fn <- function(x, y) {
    models <- lapply(pipelines, function(pl) fit(pl, x, y))
    stats::setNames(models, combo_names)
  }
  preds_fn <- function(object, data) {
    out <- list()
    for (nm in names(object)) {
      path_preds <- stats::predict(object[[nm]], newdata = data)
      for (pnm in names(path_preds)) {
        out[[pnm]] <- path_preds[[pnm]]
      }
    }
    out
  }
  structure(
    list(name = name_prefix, fit = fit_fn, preds = preds_fn),
    class = c("enfold_list", "enfold_learner")
  )
}

# Scan a pipeline for bare enfold_grid nodes (no search engine).
# Returns name and hyperparams for the constructor; the original pipeline is
# used as learner_object so apply_combo can find and replace the node(s) at fit time.
# Single node: returns name = node$name, parameters = node$hyperparams (unchanged).
# Multiple nodes: returns name = paste(node_names, "+"), parameters = combined
#   enfold_hyperparameters with "node/param" style keys.
extract_bare_grid_from_pipeline <- function(pipeline) {
  grid_locations <- list()
  for (s in seq_along(pipeline$stages)) {
    for (j in seq_along(pipeline$stages[[s]])) {
      node <- pipeline$stages[[s]][[j]]
      if (inherits(node, "enfold_grid") && is.null(node$search_engine)) {
        grid_locations <- c(
          grid_locations,
          list(list(s = s, j = j, node = node))
        )
      }
    }
  }

  if (length(grid_locations) == 0L) {
    stop(
      "Pipeline contains no embedded hyperparameter specification (no bare enfold_grid node). ",
      "Use specify_hyperparameters() in a learner constructor before adding it as a pipeline stage.",
      call. = FALSE
    )
  }

  if (length(grid_locations) == 1L) {
    gn <- grid_locations[[1L]]$node
    return(list(name = gn$name, parameters = gn$hyperparams))
  }

  all_params <- list()
  for (loc in grid_locations) {
    node <- loc$node
    for (nm in names(node$hyperparams)) {
      all_params[[paste0(node$name, "/", nm)]] <- node$hyperparams[[nm]]
    }
  }
  node_names <- vapply(grid_locations, function(l) l$node$name, character(1L))
  list(
    name = paste(node_names, collapse = "+"),
    parameters = structure(all_params, class = "enfold_hyperparameters")
  )
}


# ── print methods ──────────────────────────────────────────────────────────

#' @export
print.enfold_grid <- function(x, ...) {
  engine_str <- if (is.null(x$search_engine)) "none (bare)" else "present"
  cat(sprintf(
    "enfold_grid | prefix: '%s' | search engine: %s\n",
    x$name,
    engine_str
  ))
  is_multi_node <- any(grepl("/", names(x$hyperparams), fixed = TRUE))
  if (is_multi_node) {
    node_names <- unique(sub("/.*", "", names(x$hyperparams)))
    for (node_nm in node_names) {
      cat(sprintf("  Node '%s':\n", node_nm))
      node_keys <- grep(
        paste0("^", node_nm, "/"),
        names(x$hyperparams),
        value = TRUE
      )
      for (nm in node_keys) {
        param_nm <- sub(paste0("^", node_nm, "/"), "", nm)
        val <- x$hyperparams[[nm]]
        cat(sprintf("    %s: %s\n", param_nm, format_hyperparam(val)))
      }
    }
  } else {
    cat("  Hyperparameters:\n")
    for (nm in names(x$hyperparams)) {
      val <- x$hyperparams[[nm]]
      cat(sprintf("    %s: %s\n", nm, format_hyperparam(val)))
    }
  }
  invisible(x)
}

format_hyperparam <- function(val) {
  if (inherits(val, "enfold_range")) {
    sprintf("enfold_range [%g, %g]", val$min, val$max)
  } else if (inherits(val, "enfold_discrete")) {
    n <- length(val$values)
    preview <- utils::head(val$values, 3L)
    pstr <- paste(vapply(preview, deparse, character(1L)), collapse = ", ")
    if (n > 3L) {
      pstr <- paste0(pstr, ", ...")
    }
    sprintf("make_discrete(%s)", pstr)
  } else {
    paste(val, collapse = ", ")
  }
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
