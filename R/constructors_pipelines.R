# ── Internal DAG helpers ───────────────────────────────────────────────────

# Each pipeline stage is a list of enfold_learner objects or enfold_passthrough
# sentinels. The pipeline is stored as an ordered list of stages, where each
# stage contains one or more nodes. Every node in stage i+1 is fitted on
# the output of every node in stage i, creating independent branches.
# A passthrough sentinel passes its input unchanged to the next stage.
#
# A pipeline with stages list(A, list(B, NA), D) produces branches:
#   A -> B -> D  (named "A/B/D")
#   A -> . -> D  (named "A/./D")

# Enumerate all root-to-leaf paths through the stage list.
# Returns a list of integer vectors, each giving one path as stage indices
# (always one per stage since branching multiplies learners within stages,
# not across them — each path picks exactly one learner per stage).
enumerate_paths <- function(stages) {
  stage_sizes <- vapply(stages, length, integer(1L))
  n_paths <- prod(stage_sizes)

  paths <- vector("list", n_paths)
  for (p in seq_len(n_paths)) {
    idx <- integer(length(stages))
    rem <- p - 1L
    for (s in seq_along(stages)) {
      idx[[s]] <- (rem %% stage_sizes[[s]]) + 1L
      rem <- rem %/% stage_sizes[[s]]
    }
    paths[[p]] <- idx
  }
  paths
}

# Build the name for a path by concatenating node names with "/".
# Passthrough nodes contribute "." to the path name.
path_name <- function(stages, path_idx) {
  paste(
    vapply(
      seq_along(stages),
      function(s) {
        node <- stages[[s]][[path_idx[[s]]]]
        if (is_passthrough(node)) "." else node$name
      },
      character(1L)
    ),
    collapse = "/"
  )
}

# Build the paths data frame: one row per path, one column per stage.
# Cell values are learner names (character) or NA for passthrough nodes.
# Integer rownames allow the user to subset rows and recover original indices.
build_paths_df <- function(stages, paths) {
  n_stages <- length(stages)
  cols <- lapply(seq_len(n_stages), function(s) {
    vapply(
      paths,
      function(p) {
        node <- stages[[s]][[p[[s]]]]
        if (is_passthrough(node)) NA_character_ else node$name
      },
      character(1L)
    )
  })
  df <- as.data.frame(cols, stringsAsFactors = FALSE)
  names(df) <- paste0("stage_", seq_len(n_stages))
  df
}


# ── make_pipeline ──────────────────────────────────────────────────────────

#' Make a pipeline of learners for use in ensembles
#'
#' Constructs a DAG-based pipeline where each stage feeds its predictions as
#' \code{x} to the next stage. If a stage contains multiple learners (or an
#' \code{NA} passthrough), the pipeline branches independently for each node,
#' multiplying terminal outputs. Pipelines can be used anywhere an
#' \code{enfold_learner} is accepted.
#'
#' @param ... Learners (\code{enfold_learner}), \code{NA} (passthrough),
#'   or lists of these, in stage order. The first argument must be a single
#'   \code{enfold_learner}. Each subsequent
#'   argument receives the predictions of all nodes in the previous stage as
#'   its \code{x}. \code{NA} at a non-first stage creates a passthrough branch
#'   that forwards its input unchanged.
#' @details
#' Paths are stored in \code{$paths_df}: a data frame with one row per path
#' and one column per stage. Cells are learner names or \code{NA} for
#' passthrough nodes. Rows can be removed before fitting to disable specific
#' paths:
#' \preformatted{pl$paths_df <- pl$paths_df[pl$paths_df$stage_2 != "glm", ]}
#' Alternatively, use \code{\link{remove_paths}}.
#'
#' Path names concatenate node names with \code{"/"}; passthrough segments
#' appear as \code{"."} (e.g. \code{"screener/./rf"}).
#'
#' Learners designed for pipeline use should return objects that can serve as
#' \code{x} for the next stage. Type compatibility between adjacent nodes is
#' the user's responsibility.
#' @return An object of class \code{enfold_pipeline}.
#' @seealso \code{\link{remove_paths}}, \code{\link{grd_random}}
#' @examples
#' \dontrun{
#' # Screen then fit a GLM and random forest in parallel
#' pl <- make_pipeline(
#'   scr_correlation("scr", cutoff = 0.1),
#'   list(lrn_glm("glm", family = gaussian()),
#'        lrn_ranger("rf"))
#' )
#' pl
#'
#' # Use in a task (produces two predictions: "scr/glm" and "scr/rf")
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(pl) |>
#'   add_metalearners(mtl_selector("selector")) |>
#'   add_cv_folds(inner_cv = 5L, outer_cv = NA) |>
#'   fit()
#' }
#' @export
make_pipeline <- function(...) {
  raw <- list(...)

  if (length(raw) < 2L) {
    stop(
      "`make_pipeline` requires at least two arguments (a source learner and at least one successor)."
    )
  }

  if (is.list(raw[[1L]]) && !inherits(raw[[1L]], c("enfold_learner", "enfold_grid"))) {
    stop(
      "The first argument to `make_pipeline` must be a single node, not a list of nodes.",
      call. = FALSE
    )
  }

  # Resolve each argument into a flat list of nodes (enfold_learner, enfold_grid, or enfold_passthrough)
  stages <- lapply(seq_along(raw), function(i) {
    x <- raw[[i]]

    if (inherits(x, "enfold_learner")) {
      list(x)
    } else if (inherits(x, "enfold_grid")) {
      if (!is.null(x$search_engine)) {
        stop(
          sprintf(
            "Argument %d to `make_pipeline` is an enfold_grid with a search engine. ",
            i
          ),
          "Only bare grids (created via lrn_*(name, parameters = ...)) are allowed in pipelines. ",
          "Apply grd_*() to the full pipeline instead.",
          call. = FALSE
        )
      }
      list(x)
    } else if (length(x) == 1L && is.na(x)) {
      if (i == 1L) {
        stop("The first argument to `make_pipeline` cannot be NA.")
      }
      list(make_passthrough())
    } else if (is.list(x)) {
      lapply(seq_along(x), function(k) {
        item <- x[[k]]
        if (length(item) == 1L && is.na(item)) {
          if (i == 1L) {
            stop("The first argument to `make_pipeline` cannot contain NA.")
          }
          make_passthrough()
        } else if (inherits(item, "enfold_grid")) {
          if (!is.null(item$search_engine)) {
            stop(
              sprintf(
                "Argument %d to `make_pipeline`, element %d is an enfold_grid with a search engine. ",
                i, k
              ),
              "Only bare grids (created via lrn_*(name, parameters = ...)) are allowed in pipelines.",
              call. = FALSE
            )
          }
          item
        } else if (inherits(item, "enfold_learner")) {
          item
        } else {
          stop(sprintf(
            "Argument %d to `make_pipeline`, element %d: must be an enfold_learner, enfold_grid, or NA.",
            i,
            k
          ))
        }
      })
    } else {
      stop(sprintf(
        "Argument %d to `make_pipeline` must be an enfold_learner, enfold_grid, NA, or a list of these.",
        i
      ))
    }
  })

  paths <- enumerate_paths(stages)
  path_nms <- vapply(paths, function(p) path_name(stages, p), character(1L))

  if (length(unique(path_nms)) != length(path_nms)) {
    stop(
      "Some pipeline paths have identical names. Rename learners to disambiguate."
    )
  }

  paths_df <- build_paths_df(stages, paths)

  structure(
    list(
      stages = stages,
      paths = paths,
      path_names = path_nms,
      paths_df = paths_df
    ),
    class = "enfold_pipeline"
  )
}


# ── fit.enfold_pipeline ────────────────────────────────────────────────────────

#' @export
fit.enfold_pipeline <- function(object, x, y, ...) {
  stages <- object$stages
  n_stages <- length(stages)

  # Expand any bare enfold_grid nodes into their concrete learners.
  # Multiple bare grids across stages are expanded independently (Cartesian product).
  has_bare_grid <- any(vapply(stages, function(stage) {
    any(vapply(stage, function(node) inherits(node, "enfold_grid"), logical(1L)))
  }, logical(1L)))

  if (has_bare_grid) {
    stages <- lapply(stages, function(stage) {
      expanded <- list()
      for (node in stage) {
        if (inherits(node, "enfold_grid")) {
          combo_df <- draw(node$hyperparams, n = NULL)
          new_nodes <- lapply(seq_len(nrow(combo_df)), function(i) {
            apply_combo(node, combo_row(combo_df, i))
          })
          expanded <- c(expanded, new_nodes)
        } else {
          expanded <- c(expanded, list(node))
        }
      }
      expanded
    })
    new_paths <- enumerate_paths(stages)
    active_names <- vapply(new_paths, function(p) path_name(stages, p), character(1L))
  } else {
    active_names <- active_path_names(object)
  }

  path_states <- list(list(
    name_parts = character(0L),
    x = x,
    y = y,
    fitted = list()
  ))

  for (s in seq_len(n_stages)) {
    new_states <- list()
    for (j in seq_along(stages[[s]])) {
      node <- stages[[s]][[j]]
      for (state in path_states) {
        if (is_passthrough(node)) {
          new_states <- c(
            new_states,
            list(list(
              name_parts = c(state$name_parts, "."),
              x = state$x,
              y = state$y,
              fitted = c(state$fitted, list(node))
            ))
          )
        } else if (inherits(node, "enfold_list") && s < n_stages) {
          # Non-terminal enfold_list: expand into one sub-path per list entry.
          f <- tryCatch(
            fit(node, state$x, state$y),
            error = function(e) {
              warning(sprintf(
                "Pipeline stage %d, '%s': fit failed; path excluded.\n  %s",
                s, node$name, conditionMessage(e)
              ))
              NULL
            }
          )
          if (is.null(f)) next
          raw <- tryCatch(
            stats::predict(f, newdata = state$x),
            error = function(e) {
              warning(sprintf(
                "Pipeline stage %d, '%s': predict failed; path excluded.\n  %s",
                s, node$name, conditionMessage(e)
              ))
              NULL
            }
          )
          if (is.null(raw)) next
          for (nm in names(raw)) {
            xy <- extract_xy(raw[[nm]], state$y)
            new_states <- c(
              new_states,
              list(list(
                name_parts = c(state$name_parts, node$name, nm),
                x = xy$x,
                y = xy$y,
                fitted = c(
                  state$fitted,
                  list(
                    structure(
                      list(fitted_list = f, entry_nm = nm),
                      class = "enfold_list_entry_fitted"
                    )
                  )
                )
              ))
            )
          }
        } else {
          f <- tryCatch(
            fit(node, state$x, state$y),
            error = function(e) {
              warning(sprintf(
                "Pipeline stage %d, '%s': fit failed; path excluded.\n  %s",
                s, node$name, conditionMessage(e)
              ))
              NULL
            }
          )
          if (is.null(f)) next
          if (s < n_stages) {
            raw <- tryCatch(
              stats::predict(f, newdata = state$x),
              error = function(e) {
                warning(sprintf(
                  "Pipeline stage %d, '%s': predict failed; path excluded.\n  %s",
                  s, node$name, conditionMessage(e)
                ))
                NULL
              }
            )
            if (is.null(raw)) next
            xy <- extract_xy(raw, state$y)
            new_states <- c(
              new_states,
              list(list(
                name_parts = c(state$name_parts, node$name),
                x = xy$x,
                y = xy$y,
                fitted = c(state$fitted, list(f))
              ))
            )
          } else {
            new_states <- c(
              new_states,
              list(list(
                name_parts = c(state$name_parts, node$name),
                x = state$x,
                y = state$y,
                fitted = c(state$fitted, list(f))
              ))
            )
          }
        }
      }
    }
    path_states <- new_states
  }

  path_states <- Filter(
    function(s) {
      canonical_pipeline_path_name(s$name_parts, stages) %in% active_names
    },
    path_states
  )

  structure(
    list(
      stages = stages,
      path_names = vapply(
        path_states,
        function(s) paste(s$name_parts, collapse = "/"),
        character(1L)
      ),
      fitted_by_path = lapply(path_states, `[[`, "fitted")
    ),
    class = c(
      "enfold_pipeline_fitted",
      "enfold_learner_fitted",
      "enfold_pipeline"
    )
  )
}


# ── predict.enfold_pipeline_fitted ────────────────────────────────────────────

#' @export
predict.enfold_pipeline_fitted <- function(object, newdata, ...) {
  if (missing(newdata) || is.null(newdata)) {
    stop("`newdata` is required for `predict.enfold_pipeline_fitted`.")
  }
  results <- list()
  for (p in seq_along(object$fitted_by_path)) {
    path_nm <- object$path_names[[p]]
    path_out <- tryCatch({
      current_x <- newdata
      for (node in object$fitted_by_path[[p]]) {
        if (!is_passthrough(node)) {
          raw <- stats::predict(node, newdata = current_x)
          current_x <- if (inherits(node, "enfold_list_fitted")) {
            raw
          } else if (is.list(raw) && !is.null(raw$x)) {
            raw$x
          } else {
            raw
          }
        }
      }
      current_x
    }, error = function(e) {
      warning(sprintf(
        "Pipeline predict, path '%s': failed; excluded.\n  %s",
        path_nm, conditionMessage(e)
      ))
      NULL
    })
    if (is.null(path_out)) next
    last_node <- utils::tail(object$fitted_by_path[[p]], 1L)[[1L]]
    if (inherits(last_node, "enfold_list_fitted")) {
      for (nm in names(path_out)) {
        results[[paste0(path_nm, "/", nm)]] <- path_out[[nm]]
      }
    } else {
      results[[path_nm]] <- path_out
    }
  }
  results
}


# ── print methods ──────────────────────────────────────────────────────────

#' @export
print.enfold_pipeline <- function(x, ...) {
  active_idx <- as.integer(rownames(x$paths_df))
  active_names <- x$path_names[active_idx]
  n_total <- length(x$paths)
  n_active <- length(active_idx)

  if (n_active == n_total) {
    cat(sprintf(
      "enfold_pipeline | %d stage(s) | %d path(s)\n",
      length(x$stages),
      n_active
    ))
  } else {
    cat(sprintf(
      "enfold_pipeline | %d stage(s) | %d/%d active path(s)\n",
      length(x$stages),
      n_active,
      n_total
    ))
  }

  for (s in seq_along(x$stages)) {
    node_names <- vapply(
      x$stages[[s]],
      function(n) {
        if (is_passthrough(n)) "(passthrough)" else n$name
      },
      character(1L)
    )
    cat(sprintf("  Stage %d: %s\n", s, paste(node_names, collapse = ", ")))
  }

  if (n_active < n_total) {
    cat(sprintf(
      "Note: %d paths are currently inactive (removed by call to `remove_paths` or subsetting `paths_df`).\n",
      n_total - n_active
    ))
  }

  cat("Active paths:\n")
  for (nm in active_names) {
    cat(sprintf("  %s\n", nm))
  }
  invisible(x)
}

#' @export
print.enfold_pipeline_fitted <- function(x, ...) {
  cat(sprintf(
    "enfold_pipeline_fitted | %d stage(s) | %d terminal path(s)\n",
    length(x$stages),
    length(x$path_names)
  ))
  for (s in seq_along(x$stages)) {
    node_names <- vapply(
      x$stages[[s]],
      function(n) {
        if (is_passthrough(n)) "(passthrough)" else n$name
      },
      character(1L)
    )
    cat(sprintf("  Stage %d: %s\n", s, paste(node_names, collapse = ", ")))
  }
  cat("Paths:\n")
  for (nm in x$path_names) {
    cat(sprintf("  %s\n", nm))
  }
  invisible(x)
}

# ── remove_paths ───────────────────────────────────────────────────────-------

#' Remove paths from a pipeline
#'
#' Deactivates one or more paths in an \code{enfold_pipeline} before fitting.
#' Path names use \code{"/"} as a separator and \code{"."} for passthrough
#' segments (e.g. \code{"screener/./rf"}).
#'
#' @param object An unfitted \code{enfold_pipeline} object.
#' @param ... One or more path name strings to remove. Available names can be
#'   inspected with \code{print(object)} or via
#'   \code{rownames(object$paths_df)}.
#' @return The updated \code{enfold_pipeline} with the specified paths
#'   deactivated. The pipeline's full \code{paths} and \code{path_names} slots
#'   are unchanged; only the active set tracked by \code{$paths_df} is
#'   modified.
#' @seealso \code{\link{make_pipeline}}
#' @examples
#' \dontrun{
#' pl <- make_pipeline(
#'   scr_correlation("scr", cutoff = 0.1),
#'   list(lrn_glm("glm", family = gaussian()), lrn_ranger("rf"))
#' )
#' # Remove the GLM branch
#' pl_no_glm <- remove_paths(pl, "scr/glm")
#' pl_no_glm
#' }
#' @export
remove_paths <- function(object, ...) {
  if (!inherits(object, "enfold_pipeline")) {
    stop("`object` must be an `enfold_pipeline`.")
  }

  if (inherits(object, "enfold_pipeline_fitted")) {
    stop(
      "`object` must be an unfitted `enfold_pipeline`. Remove paths before fitting."
    )
  }

  path_nms <- active_path_names(object)
  to_remove <- c(...)

  if (length(to_remove) == 0L) {
    warning("No paths specified for removal. Returning original pipeline.")
    return(object)
  }

  if (!all(to_remove %in% path_nms)) {
    stop("Some specified paths not found among active paths.")
  }

  remove_inds <- which(path_nms %in% to_remove)

  if (length(remove_inds) == length(path_nms)) {
    stop("Cannot remove all paths from the pipeline.")
  }

  # Only subset paths_df — paths and path_names stay as the full canonical lists.
  # active_path_names() reads rownames(paths_df) to determine what is active.
  new_paths_df <- object$paths_df[-remove_inds, , drop = FALSE]

  object$paths_df <- new_paths_df
  object
}


# ── Internal helpers ───────────────────────────────────────────────────────

# Thin wrapper stored in fitted_by_path when a non-terminal enfold_list is
# expanded into sub-paths. At predict time it calls the underlying
# enfold_list_fitted and selects the one entry this sub-path owns.
#' @export
predict.enfold_list_entry_fitted <- function(object, newdata, ...) {
  stats::predict(object$fitted_list, newdata = newdata)[[object$entry_nm]]
}

# Normalise a stage predict() output into list(x, y).
# If the stage returns list(x, y), use it; otherwise wrap plain x.
extract_xy <- function(raw, current_y) {
  if (is.list(raw) && !is.null(raw$x)) {
    list(x = raw$x, y = if (!is.null(raw$y)) raw$y else current_y)
  } else {
    list(x = raw, y = current_y)
  }
}

# Return the active path names for a pipeline (respecting paths_df subsetting).
active_path_names <- function(pipeline) {
  idx <- as.integer(rownames(pipeline$paths_df))
  pipeline$path_names[idx]
}

# Create a passthrough sentinel node. A passthrough forwards its input
# unchanged to the next stage — no fit or predict is performed.
make_passthrough <- function() {
  structure(list(name = NA_character_), class = "enfold_passthrough")
}

is_passthrough <- function(x) inherits(x, "enfold_passthrough")

# Create nice path names
canonical_pipeline_path_name <- function(name_parts, stages) {
  n_stages <- length(stages)
  parts <- character(n_stages)
  idx <- 1L

  for (s in seq_len(n_stages)) {
    parts[s] <- name_parts[[idx]]
    node_name <- parts[s]

    node <- NULL
    for (candidate in stages[[s]]) {
      if (is_passthrough(candidate) && node_name == ".") {
        node <- candidate
        break
      }
      if (!is_passthrough(candidate) && candidate$name == node_name) {
        node <- candidate
        break
      }
    }

    if (inherits(node, "enfold_list") && s < n_stages) {
      idx <- idx + 2L
    } else {
      idx <- idx + 1L
    }
  }

  paste(parts, collapse = "/")
}
