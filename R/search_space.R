# ── search_space ──────────────────────────────────────────────────────────

#' Extract the hyperparameter search space from a grid or pipeline
#'
#' Returns the search space as a structured \code{enfold_search_space} object.
#' Each entry is named by the learner or pipeline node it belongs to and holds
#' the corresponding \code{enfold_hyperparameters} specification.
#'
#' Calling \code{search_space()} on a plain \code{enfold_learner} is an error
#' because a plain learner carries no hyperparameter specification. Use
#' \code{lrn_*(name, parameters = specify_hyperparameters(...))} to attach a
#' specification; that call already returns a bare \code{enfold_grid}.
#'
#' Calling \code{search_space()} on a pipeline that contains no embedded bare
#' grid nodes is also an error.
#'
#' @param x An \code{enfold_grid} (bare, i.e. no search engine) or an
#'   \code{enfold_pipeline} that contains at least one bare grid node.
#' @param ... Ignored.
#' @return An object of class \code{enfold_search_space}: a named list where
#'   each entry is an \code{enfold_hyperparameters} object keyed by the node
#'   name. Use \code{\link{flatten}} to convert this into a single flat
#'   parameter list suitable for Bayesian or other black-box optimizers.
#' @seealso \code{\link{flatten}}, \code{\link{draw}}, \code{\link{apply_combo}}
#' @examples
#' \dontrun{
#' params <- specify_hyperparameters(num.trees = make_discrete(50L, 100L, 200L))
#' g      <- lrn_ranger("rf", parameters = params)  # bare enfold_grid
#' search_space(g)
#'
#' # Pipeline with two parameterised nodes
#' p1 <- specify_hyperparameters(cutoff = make_range(0.01, 0.5))
#' p2 <- specify_hyperparameters(num.trees = make_discrete(50L, 100L))
#' pl <- make_pipeline(
#'   scr_correlation("scr", parameters = p1),
#'   lrn_ranger("rf", parameters = p2)
#' )
#' search_space(pl)
#' }
#' @export
search_space <- function(x, ...) {
  UseMethod("search_space")
}

#' @export
search_space.enfold_learner <- function(x, ...) {
  stop(
    "search_space() requires an enfold_grid or a parameterised pipeline, not ",
    "a plain enfold_learner. Attach a specification via ",
    "lrn_*(name, parameters = specify_hyperparameters(...)).",
    call. = FALSE
  )
}

#' @export
search_space.enfold_grid <- function(x, ...) {
  if (inherits(x$learner_object, "enfold_pipeline"))
    return(search_space(x$learner_object))
  structure(
    stats::setNames(list(x$hyperparams), x$name),
    class = "enfold_search_space"
  )
}

#' @export
search_space.enfold_pipeline <- function(x, ...) {
  found <- list()
  for (stage in x$stages) {
    for (node in stage) {
      if (inherits(node, "enfold_grid") && is.null(node$search_engine)) {
        found[[node$name]] <- node$hyperparams
      }
    }
  }
  if (length(found) == 0L) {
    stop(
      "search_space() found no parameterised nodes in this pipeline. ",
      "Embed a bare grid by passing parameters = specify_hyperparameters(...) ",
      "to a learner constructor before adding it as a pipeline stage.",
      call. = FALSE
    )
  }
  structure(found, class = "enfold_search_space")
}


# ── flatten ───────────────────────────────────────────────────────────────

#' Flatten a structured search space into a single parameter listsetnames
#'
#' Combines all per-node \code{enfold_hyperparameters} specs from an
#' \code{enfold_search_space} into a single named list, keying each parameter
#' as \code{"node/param"}. The result is suitable for passing to Bayesian or
#' other black-box optimizers that expect a flat list of parameter bounds.
#'
#' @param x An \code{enfold_search_space} returned by \code{\link{search_space}}.
#' @param ... Ignored.
#' @return An object of class \code{enfold_search_space_flat}: a named list of
#'   \code{enfold_range} and \code{enfold_discrete} objects with
#'   \code{"node/param"} keys. The original per-node structure is available via
#'   \code{attr(result, "nodes")}. Pass the flat object to
#'   \code{\link{apply_combo}} alongside the original grid or pipeline; it will
#'   automatically be unflattened.
#' @seealso \code{\link{search_space}}, \code{\link{apply_combo}}
#' @examples
#' \dontrun{
#' params <- specify_hyperparameters(
#'   alpha  = make_discrete(0, 0.5, 1),
#'   lambda = make_range(1e-4, 1)
#' )
#' g  <- lrn_glmnet("enet", parameters = params)
#' ss <- search_space(g)
#' flatten(ss)
#' }
#' @export
flatten <- function(x, ...) {
  UseMethod("flatten")
}

#' @export
flatten.enfold_search_space <- function(x, ...) {
  node_names <- names(x)
  result    <- list()
  param_map <- list()
  target_of <- character(0L)

  for (nm in node_names) {
    hp <- x[[nm]]
    for (param_nm in names(hp)) {
      key              <- paste0(nm, "/", param_nm)
      result[[key]]    <- hp[[param_nm]]
      param_map[[key]] <- list(node = nm, param = param_nm)
      target_of[[key]] <- nm
    }
  }

  structure(
    result,
    class     = "enfold_search_space_flat",
    nodes     = node_names,
    param_map = param_map,
    target_of = target_of
  )
}


# ── unflatten ─────────────────────────────────────────────────────────────

#' Unflatten a flat hyperparameter combo back to structured form
#'
#' Converts a flat \code{"node/param"}-keyed list (as produced by an optimizer
#' working with the output of \code{\link{flatten}}) back to the structured
#' \code{list(node = list(param = val))} form that \code{\link{apply_combo}}
#' expects.
#'
#' @param flat_combo A named list with \code{"node/param"} keys.
#' @param search_space An \code{enfold_search_space} giving the node structure.
#' @param ... Ignored.
#' @return A named list: \code{list(node_name = list(param = val), ...)}.
#' @seealso \code{\link{flatten}}, \code{\link{apply_combo}}
#' @export
unflatten <- function(flat_combo, search_space, ...) {
  UseMethod("unflatten")
}

#' @rdname unflatten
#' @export
unflatten.default <- function(flat_combo, search_space, ...) {
  node_names <- names(search_space)
  result <- vector("list", length(node_names))
  names(result) <- node_names
  for (nm in node_names) result[[nm]] <- list()

  for (key in names(flat_combo)) {
    slash_pos <- regexpr("/", key, fixed = TRUE)
    if (slash_pos < 1L) {
      stop(
        sprintf(
          "Flat combo key '%s' has no '/' separator. Expected 'node/param' format.",
          key
        ),
        call. = FALSE
      )
    }
    node_nm  <- substr(key, 1L, slash_pos - 1L)
    param_nm <- substr(key, slash_pos + 1L, nchar(key))
    if (!node_nm %in% node_names) {
      stop(
        sprintf(
          "Flat combo key '%s' refers to node '%s' which is not in the search space.",
          key, node_nm
        ),
        call. = FALSE
      )
    }
    result[[node_nm]][[param_nm]] <- flat_combo[[key]]
  }
  result
}


# ── normalize_combo ───────────────────────────────────────────────────────
# Internal: accept named list or 1-row data frame, always return named list.
normalize_combo <- function(combo) {
  if (is.data.frame(combo)) combo_row(combo, 1L) else combo
}


# ── apply_combo ───────────────────────────────────────────────────────────

#' Instantiate a concrete learner or pipeline from a hyperparameter combo
#'
#' Applies a named list of hyperparameter values to an \code{enfold_grid} or
#' \code{enfold_pipeline} and returns the resulting concrete learner or
#' pipeline with the parameters pinned.
#'
#' \describe{
#'   \item{Grid}{
#'     \code{combo} is a flat named list: one entry per hyperparameter
#'     (e.g. \code{list(alpha = 0.5, lambda = 0.01)}).
#'   }
#'   \item{Pipeline — structured}{
#'     \code{combo} is a named list of named lists: each top-level key is a
#'     node name and each value is the flat combo for that node
#'     (e.g. \code{list(rf = list(num.trees = 100L))}).
#'   }
#'   \item{Pipeline — flat (from \code{flatten})}{
#'     \code{combo} may also be a flat \code{"node/param"}-keyed list as
#'     produced by an optimizer working with the output of
#'     \code{\link{flatten}}. It is automatically unflattened before
#'     application.
#'   }
#' }
#'
#' @param x An \code{enfold_grid} or \code{enfold_pipeline}.
#' @param combo A named list of hyperparameter values. See Details.
#' @param ... Ignored.
#' @return For a grid: an \code{enfold_learner} or \code{enfold_pipeline}
#'   with parameters pinned to the values in \code{combo}. For a pipeline: a
#'   concrete \code{enfold_pipeline} with all targeted bare grid nodes
#'   replaced by instantiated learners.
#' @seealso \code{\link{search_space}}, \code{\link{flatten}}, \code{\link{draw}}
#' @examples
#' \dontrun{
#' params <- specify_hyperparameters(num.trees = make_discrete(50L, 100L, 200L))
#' g <- lrn_ranger("rf", parameters = params)
#' lrn <- apply_combo(g, list(num.trees = 100L))
#' }
#' @export
apply_combo <- function(x, combo, ...) {
  UseMethod("apply_combo")
}

#' @export
apply_combo.enfold_grid <- function(x, combo, ...) {
  combo <- normalize_combo(combo)
  if (inherits(x$learner_object, "enfold_pipeline")) {
    if (any(grepl("/", names(combo), fixed = TRUE))) {
      # Multi-node: "node/param" keys — pass flat combo directly to pipeline dispatch
      apply_combo(x$learner_object, combo)
    } else {
      # Single-node: wrap as {node_name: {param: val}}
      apply_combo(x$learner_object, stats::setNames(list(combo), x$name))
    }
  } else {
    nm <- make_combo_name(x$name, combo)
    pin_learner_params(x$learner_object, combo, name = nm)
  }
}

# Rebuild a plain enfold_learner's closure env with new hyperparameter values.
pin_learner_params <- function(learner, combo, name = NULL) {
  new_p <- utils::modifyList(get_params(learner), combo)
  orig_fit <- get_original_fit(learner)
  orig_preds <- get_original_preds(learner)
  old_env <- environment(learner$fit)

  new_env <- list2env(new_p, parent = parent.env(old_env))
  environment(orig_fit) <- new_env
  environment(orig_preds) <- new_env
  new_env$fit <- orig_fit
  new_env$preds <- orig_preds

  new_wrapped_fit <- function(x, y) fit(x = x, y = y)
  environment(new_wrapped_fit) <- new_env
  new_wrapped_preds <- function(object, data) preds(object = object, data = data)
  environment(new_wrapped_preds) <- new_env

  new_obj <- learner
  new_obj$fit <- new_wrapped_fit
  new_obj$preds <- new_wrapped_preds
  if (!is.null(name)) new_obj$name <- name
  new_obj
}

#' @export
apply_combo.enfold_pipeline <- function(x, combo, ...) {
  combo <- normalize_combo(combo)

  # If flat "node/param" keys are detected, unflatten first
  if (any(grepl("/", names(combo), fixed = TRUE))) {
    ss    <- search_space(x)
    combo <- unflatten(combo, ss)
  }

  # combo is now structured: list(node_name = list(param = val, ...), ...)
  new_stages <- x$stages
  for (node_name in names(combo)) {
    node_combo <- combo[[node_name]]
    found <- FALSE
    for (s in seq_along(new_stages)) {
      for (j in seq_along(new_stages[[s]])) {
        node <- new_stages[[s]][[j]]
        if (
          inherits(node, "enfold_grid") &&
            is.null(node$search_engine) &&
            node$name == node_name
        ) {
          new_stages[[s]][[j]] <- apply_combo(node, node_combo)
          found <- TRUE
          break
        }
      }
      if (found) break
    }
    if (!found) {
      stop(
        sprintf("No bare grid node named '%s' found in the pipeline.", node_name),
        call. = FALSE
      )
    }
  }

  new_paths    <- enumerate_paths(new_stages)
  new_nms      <- vapply(new_paths, function(p) path_name(new_stages, p), character(1L))
  new_paths_df <- build_paths_df(new_stages, new_paths)
  structure(
    list(stages = new_stages, paths = new_paths, path_names = new_nms, paths_df = new_paths_df),
    class = "enfold_pipeline"
  )
}

# ── draw extensions ───────────────────────────────────────────────────────

#' @rdname draw
#' @export
draw.enfold_search_space <- function(obj, n = 1L) {
  if (length(obj) == 1L) {
    # Single-node: same data frame as draw.enfold_hyperparameters
    draw(obj[[1L]], n = n)
  } else {
    if (is.null(n)) {
      # Multi-node n = NULL: Cartesian product of each node's full combo set
      per_node <- lapply(names(obj), function(nm) {
        df        <- draw(obj[[nm]], n = NULL)
        names(df) <- paste0(nm, "/", names(df))
        df
      })
      cross_search_space_dfs(per_node)
    } else {
      # Multi-node n given: draw n rows per node independently, cbind
      per_node <- lapply(names(obj), function(nm) {
        df        <- draw(obj[[nm]], n = n)
        names(df) <- paste0(nm, "/", names(df))
        df
      })
      do.call(cbind, per_node)
    }
  }
}


# Internal: Cartesian product of a list of data frames (for n = NULL multi-node draw).
cross_search_space_dfs <- function(dfs) {
  if (length(dfs) == 1L) return(dfs[[1L]])
  sizes    <- vapply(dfs, nrow, integer(1L))
  idx_grid <- do.call(expand.grid, lapply(sizes, seq_len))
  result_cols <- list()
  for (d_i in seq_along(dfs)) {
    df          <- dfs[[d_i]]
    row_indices <- idx_grid[[d_i]]
    for (col_nm in names(df)) {
      col_vals            <- df[[col_nm]]
      result_cols[[col_nm]] <- if (inherits(col_vals, "AsIs")) {
        I(col_vals[row_indices])
      } else {
        col_vals[row_indices]
      }
    }
  }
  result <- as.data.frame(result_cols, stringsAsFactors = FALSE)
  rownames(result) <- NULL
  result
}


# ── print methods ─────────────────────────────────────────────────────────

#' @export
print.enfold_search_space <- function(x, ...) {
  cat(sprintf("enfold_search_space | %d node(s)\n", length(x)))
  for (nm in names(x)) {
    cat(sprintf("  Node '%s':\n", nm))
    hp <- x[[nm]]
    for (param_nm in names(hp)) {
      val <- hp[[param_nm]]
      if (inherits(val, "enfold_range")) {
        cat(sprintf("    %s: range [%g, %g]\n", param_nm, val$min, val$max))
      } else if (inherits(val, "enfold_discrete")) {
        n_vals <- length(val$values)
        preview <- utils::head(val$values, 3L)
        pstr <- paste(vapply(preview, deparse, character(1L)), collapse = ", ")
        if (n_vals > 3L) {
          pstr <- paste0(pstr, ", ...")
        }
        cat(sprintf("    %s: discrete(%s)\n", param_nm, pstr))
      } else {
        cat(sprintf("    %s: %s\n", param_nm, paste(val, collapse = ", ")))
      }
    }
  }
  invisible(x)
}

#' @export
print.enfold_search_space_flat <- function(x, ...) {
  cat(sprintf(
    "enfold_search_space_flat | %d parameter(s) across node(s): %s\n",
    length(x),
    paste(attr(x, "nodes"), collapse = ", ")
  ))
  for (key in names(x)) {
    val <- x[[key]]
    if (inherits(val, "enfold_range")) {
      cat(sprintf("  %s: range [%g, %g]\n", key, val$min, val$max))
    } else if (inherits(val, "enfold_discrete")) {
      n_vals <- length(val$values)
      preview <- utils::head(val$values, 3L)
      pstr <- paste(vapply(preview, deparse, character(1L)), collapse = ", ")
      if (n_vals > 3L) {
        pstr <- paste0(pstr, ", ...")
      }
      cat(sprintf("  %s: discrete(%s)\n", key, pstr))
    } else {
      cat(sprintf("  %s: %s\n", key, paste(val, collapse = ", ")))
    }
  }
  invisible(x)
}
