#' Specify hyperparameters for a grid search
#'
#' Creates an \code{enfold_hyperparameters} object that describes the search
#' space for grid constructors such as \code{\link{grd_random}}. Each named
#' argument defines one hyperparameter as either a discrete vector of candidate
#' values or a continuous range created by \code{\link{make_range}}.
#'
#' @param ... Named hyperparameter specifications. Accepted types:
#'   \itemize{
#'     \item Numeric or character vectors — discrete candidates.
#'     \item An \code{enfold_range} object from \code{\link{make_range}} —
#'       continuous uniform range, sampled at run time.
#'   }
#' @return An object of class \code{enfold_hyperparameters}.
#' @details
#' Pipe the result through \code{\link{forbid}} to exclude invalid parameter
#' combinations, and through \code{\link{log_transform}} to sample a
#' continuous parameter on the log scale.
#' @seealso \code{\link{make_range}}, \code{\link{forbid}},
#'   \code{\link{log_transform}}, \code{\link{draw}}, \code{\link{grd_random}}
#' @examples
#' # Discrete grid
#' specify_hyperparameters(
#'   alpha  = c(0, 0.5, 1),
#'   lambda = c(0.01, 0.1, 1)
#' )
#'
#' # Mixed: discrete + continuous range
#' specify_hyperparameters(
#'   alpha  = c(0, 0.5, 1),
#'   lambda = make_range(1e-4, 10)
#' )
#' @export
specify_hyperparameters <- function(...) {
  hyperparams <- rlang::list2(...)
  if (length(hyperparams) == 0L) {
    stop(
      "Provide at least one hyperparameter specification via `...`.",
      call. = FALSE
    )
  }
  if (is.null(names(hyperparams)) || any(names(hyperparams) == "")) {
    stop(
      "All hyperparameter specifications in `...` must be named.",
      call. = FALSE
    )
  }
  if (
    any(
      !sapply(hyperparams, function(x) {
        is.vector(x) || inherits(x, "enfold_range")
      })
    )
  ) {
    stop(
      "Each hyperparameter specification must be a vector (numeric or character) or an enfold_range object.",
      call. = FALSE
    )
  }

  attr(hyperparams, "forbidden_logic") <- list()

  sample_funs <- lapply(hyperparams, function(x) {
    if (inherits(x, "enfold_range")) {
      function(n) runif(n, x$min, x$max)
    } else {
      function(n) sample(x, n, replace = TRUE)
    }
  })

  attr(hyperparams, "sample_funs") <- sample_funs

  structure(hyperparams, class = "enfold_hyperparameters")
}

#' Forbid specific hyperparameter combinations
#'
#' Marks one or more parameter combinations as inadmissible so that
#' \code{\link{draw}} and grid search engines never evaluate them.
#'
#' @param obj An object of class \code{enfold_hyperparameters}.
#' @param ... Unquoted logical expressions referencing hyperparameter names
#'   (e.g. \code{alpha == 0 && lambda < 0.01}). Each expression should
#'   evaluate to a single \code{TRUE}/\code{FALSE} for a given combination.
#' @return An updated \code{enfold_hyperparameters} object.
#' @seealso \code{\link{specify_hyperparameters}}, \code{\link{draw}}
#' @examples
#' params <- specify_hyperparameters(
#'   alpha  = c(0, 0.5, 1),
#'   lambda = c(0.01, 0.1, 1)
#' ) |>
#'   forbid(alpha == 0 && lambda == 0.01)
#'
#' draw(params)  # 0,0.01 combination absent
#' @export
forbid <- function(obj, ...) {4
  UseMethod("forbid")
}

#' @rdname forbid
#' @export
forbid.enfold_hyperparameters <- function(obj, ...) {
  constraints <- rlang::enquos(...)

  for (constr in constraints) {
    used_vars <- all.vars(rlang::quo_get_expr(constr))
    missing_vars <- setdiff(used_vars, names(obj))
    if (length(missing_vars) > 0) {
      stop(
        paste0(
          "The following hyperparameters in `forbid()` are not defined: ",
          paste(missing_vars, collapse = ", ")
        ),
        call. = FALSE
      )
    }
  }

  current_constraints <- attr(obj, "forbidden_logic") %||% list()
  attr(obj, "forbidden_logic") <- c(current_constraints, constraints)

  obj
}

#' Log-scale sampling for continuous hyperparameters
#'
#' Causes the specified continuous (\code{enfold_range}) hyperparameters to be
#' sampled on the log scale, so that \code{draw()} produces values
#' \eqn{\exp(U[\log(\text{min}), \log(\text{max})])} rather than
#' \eqn{U[\text{min}, \text{max}]}.
#'
#' @param obj An object of class \code{enfold_hyperparameters}.
#' @param ... Unquoted names of continuous hyperparameters to log-transform.
#'   Each referenced parameter must be an \code{enfold_range} with
#'   \eqn{\text{min} > 0}.
#' @return An updated \code{enfold_hyperparameters} object.
#' @seealso \code{\link{specify_hyperparameters}}, \code{\link{make_range}}
#' @examples
#' params <- specify_hyperparameters(lambda = make_range(1e-4, 10)) |>
#'   log_transform(lambda)
#'
#' # Draws are now log-uniformly distributed
#' draw(params, n = 5L)
#' @export
log_transform <- function(obj, ...) {
  UseMethod("log_transform")
}

#' @rdname log_transform
#' @export
log_transform.enfold_hyperparameters <- function(obj, ...) {
  vars_to_log <- rlang::enquos(...)

  for (var in vars_to_log) {
    var_name <- rlang::as_label(var)
    if (!var_name %in% names(obj)) {
      stop(
        paste0(
          "The following hyperparameter in `log_transform()` is not defined: ",
          var_name
        ),
        call. = FALSE
      )
    }
  }

  sample_funs <- attr(obj, "sample_funs")

  for (var in vars_to_log) {
    var_name <- rlang::as_label(var)
    param_val <- obj[[var_name]]
    if (!inherits(param_val, "enfold_range")) {
      stop(
        "log_transform() only supports continuous ranges from make_range().",
        call. = FALSE
      )
    }
    if (param_val$min <= 0) {
      stop(
        "Cannot log-transform range values that are not strictly positive. You can instead define a very small number like 1e-10.",
        call. = FALSE
      )
    }

    sample_funs[[var_name]] <- local({
      lower <- log(param_val$min)
      upper <- log(param_val$max)
      function(n) exp(runif(n, lower, upper))
    })
  }

  attr(obj, "sample_funs") <- sample_funs
  obj
}

#' Draw hyperparameter combinations
#'
#' Samples \code{n} valid hyperparameter combinations from an
#' \code{enfold_hyperparameters} specification, respecting \code{\link{forbid}}
#' constraints and \code{\link{log_transform}} sampling instructions.
#'
#' @param obj An object of class \code{enfold_hyperparameters}.
#' @param n Integer. Number of combinations to draw. When \code{NULL} and all
#'   parameters are discrete, every valid combination is returned.
#' @return A data frame with one row per combination and one column per
#'   hyperparameter.
#' @details
#' When all parameters are discrete, sampling is performed without
#' replacement. For specifications containing at least one continuous
#' \code{enfold_range}, \code{n} must be a positive integer.
#' @seealso \code{\link{specify_hyperparameters}}, \code{\link{forbid}}
#' @examples
#' params <- specify_hyperparameters(
#'   alpha  = c(0, 0.5, 1),
#'   lambda = c(0.01, 0.1, 1)
#' )
#' draw(params)         # all 9 combinations
#' draw(params, n = 3L) # 3 random combinations
#' @export
draw <- function(obj, n = 1L) {
  UseMethod("draw")
}

#' @rdname draw
#' @export
draw.enfold_hyperparameters <- function(obj, n = 1L) {
  constraints <- attr(obj, "forbidden_logic")
  sample_funs <- attr(obj, "sample_funs")
  param_names <- names(obj)

  all_discrete <- all(vapply(
    obj,
    function(x) !inherits(x, "enfold_range"),
    logical(1L)
  ))

  if (all_discrete) {
    combos <- expand_grid_params(obj)
    valid <- Filter(
      function(combo) !is_forbidden_combo(combo, constraints),
      combos
    )

    if (length(valid) == 0L) {
      return(data.frame(matrix(
        ncol = length(obj),
        nrow = 0,
        dimnames = list(NULL, param_names)
      )))
    }

    if (is.null(n) || n >= length(valid)) {
      out <- do.call(
        rbind,
        lapply(valid, function(combo) {
          as.data.frame(combo, stringsAsFactors = FALSE)
        })
      )
      rownames(out) <- NULL
      return(out)
    }

    if (!is.numeric(n) || length(n) != 1L || n < 1L) {
      stop("`n` must be a single positive integer or NULL.", call. = FALSE)
    }
    n <- as.integer(n)

    chosen <- sample.int(length(valid), n)
    out <- do.call(
      rbind,
      lapply(valid[chosen], function(combo) {
        as.data.frame(combo, stringsAsFactors = FALSE)
      })
    )
    rownames(out) <- NULL
    return(out)
  }

  if (is.null(n) && !all_discrete) {
    stop(
      "n must be provided for hyperparameters containing continuous ranges.",
      call. = FALSE
    )
  }

  if (!is.numeric(n) || length(n) != 1L || n < 1L) {
    stop("`n` must be a single positive integer or NULL.", call. = FALSE)
  }
  n <- as.integer(n)

  valid_rows <- vector("list", 0)
  attempts <- 0L
  max_attempts <- 100L

  while (length(valid_rows) < n && attempts < max_attempts) {
    batch_size <- max((n - length(valid_rows)) * 4L, 50L)
    draws <- lapply(sample_funs, function(fn) fn(batch_size))
    draws_df <- as.data.frame(draws, stringsAsFactors = FALSE)

    passed <- !vapply(
      seq_len(nrow(draws_df)),
      function(i) {
        combo <- as.list(draws_df[i, , drop = FALSE])
        is_forbidden_combo(combo, constraints)
      },
      logical(1L)
    )

    if (any(passed)) {
      valid_draws <- draws_df[passed, , drop = FALSE]
      valid_rows <- c(
        valid_rows,
        split(valid_draws, seq_len(nrow(valid_draws)))
      )
    }
    attempts <- attempts + 1L
  }

  if (length(valid_rows) < n) {
    stop(
      "Unable to draw the requested number of valid hyperparameter combinations after repeated sampling.",
      call. = FALSE
    )
  }

  out <- do.call(
    rbind,
    lapply(valid_rows[seq_len(n)], function(combo) {
      as.data.frame(combo, stringsAsFactors = FALSE)
    })
  )
  rownames(out) <- NULL
  out
}

# Helpers

is_forbidden_combo <- function(combo, constraints) {
  if (length(constraints) == 0L) {
    return(FALSE)
  }
  any(vapply(
    constraints,
    function(constr) {
      rlang::eval_tidy(constr, data = combo)
    },
    logical(1L)
  ))
}

#' Create a continuous hyperparameter range
#'
#' Defines a closed interval \eqn{[\text{min}, \text{max}]} from which
#' \code{\link{draw}} and search engines sample uniformly (or log-uniformly if
#' \code{\link{log_transform}} is applied).
#'
#' @param min Numeric scalar. Lower bound (inclusive).
#' @param max Numeric scalar. Upper bound (inclusive). Must be strictly greater
#'   than \code{min}.
#' @return An object of class \code{enfold_range}.
#' @seealso \code{\link{specify_hyperparameters}}, \code{\link{log_transform}}
#' @examples
#' make_range(0.001, 10)
#'
#' # Use inside specify_hyperparameters
#' specify_hyperparameters(lambda = make_range(1e-4, 1))
#' @export
make_range <- function(min, max) {
  if (
    !is.numeric(min) ||
      length(min) != 1L ||
      !is.numeric(max) ||
      length(max) != 1L
  ) {
    stop("`min` and `max` must each be a single numeric value.", call. = FALSE)
  }
  if (min >= max) {
    stop("`min` must be strictly less than `max`.", call. = FALSE)
  }
  structure(list(min = min, max = max), class = "enfold_range")
}

# Enumerate all combinations of discrete hyperparameters.
# Returns a list of named lists, one per combination.
expand_grid_params <- function(hyperparams) {
  nms <- names(hyperparams)
  levels <- lapply(hyperparams, function(v) {
    if (inherits(v, "enfold_range")) {
      stop(
        "expand_grid_params() cannot enumerate continuous enfold_range values.",
        call. = FALSE
      )
    }
    v
  })
  grid_df <- do.call(expand.grid, c(levels, list(stringsAsFactors = FALSE)))
  lapply(seq_len(nrow(grid_df)), function(i) {
    row <- as.list(grid_df[i, , drop = FALSE])
    names(row) <- nms
    row
  })
}

# Print method for enfold_hyperparameters
#' @export
print.enfold_hyperparameters <- function(x, ...) {
  cat("enfold_hyperparameters:\n")
  for (nm in names(x)) {
    val <- x[[nm]]
    if (inherits(val, "enfold_range")) {
      cat(sprintf("  %s: enfold_range [%g, %g]\n", nm, val$min, val$max))
    } else {
      cat(sprintf("  %s: %s\n", nm, paste(val, collapse = ", ")))
    }
  }
  invisible(x)
}
