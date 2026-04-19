#' Create a discrete set of hyperparameter candidates
#'
#' Defines a discrete set of values that \code{\link{draw}} and
#' \code{\link{specify_hyperparameters}} can sample from. Use this instead of
#' passing raw vectors to \code{specify_hyperparameters}.
#'
#' @param ... Candidate values. How these are interpreted depends on
#'   \code{is_separate}.
#' @param is_separate Logical. When \code{TRUE} (default), each argument is
#'   expanded element-wise: \code{make_discrete(c(1, 2), 3)} yields three
#'   candidates \code{1}, \code{2}, and \code{3}. When \code{FALSE}, each
#'   argument is kept as-is as one possible value:
#'   \code{make_discrete(c(1, 2), 3, is_separate = FALSE)} yields two
#'   candidates — the vector \code{c(1, 2)} and the scalar \code{3}.
#' @return An object of class \code{enfold_discrete}.
#' @seealso \code{\link{specify_hyperparameters}}, \code{\link{make_range}},
#'   \code{\link{draw}}
#' @examples
#' make_discrete(0, 0.5, 1)
#' make_discrete(c(0, 0.5, 1))            # same: is_separate = TRUE expands
#' make_discrete(1:5, is_separate = FALSE) # one candidate: the vector 1:5
#' @export
make_discrete <- function(..., is_separate = TRUE) {
  args <- list(...)
  if (length(args) == 0L) {
    stop("Provide at least one candidate value.", call. = FALSE)
  }
  if (is_separate) {
    values <- unlist(args, recursive = FALSE)
  } else {
    values <- args
  }

  sample_fun <- function(n) {
    values[sample.int(length(values), n, replace = TRUE)]
  }

  structure(
    list(values = values, is_separate = is_separate, sample_fun = sample_fun),
    class = "enfold_discrete"
  )
}

#' @export
print.enfold_discrete <- function(x, ...) {
  n <- length(x$values)
  cat(sprintf(
    "enfold_discrete | %d candidate(s) | is_separate = %s\n",
    n,
    x$is_separate
  ))
  preview <- utils::head(x$values, 5L)
  for (v in preview) {
    cat(" ", deparse(v), "\n")
  }
  if (n > 5L) {
    cat(sprintf("  ... (%d more)\n", n - 5L))
  }
  invisible(x)
}


#' Make a Mixture of Hyperparameter Distributions
#'
#' Takes an arbitrary amount of sample spaces made via
#' \code{\link{make_discrete}} and \code{\link{make_range}} and combines them
#' into a single mixture distrubution. The \code{weights} argument
#' controls the relative sampling frequency of each component distribution.
#'
#' @param ... Distributions created by \code{\link{make_discrete}} or \code{\link{make_range}}.
#' @param weights Numeric vector of non-negative weights corresponding to each distribution in \code{...}.
#' The length of \code{weights} must match the number of distributions provided.
#' If weights do not sum to 1, they will be normalized internally.
#' @return An object of class \code{enfold_mixture}.
#' @seealso \code{\link{specify_hyperparameters}}, \code{\link{make_range}}, \code{\link{make_discrete}},
#'   \code{\link{draw}}
#' @examples
#' # Draw a 0 half the time, and a number between 0 and 10 half the time
#' mixture_dist <- mixture(
#'   make_discrete(0),
#'   make_range(0, 10),
#'   weights = c(0.5, 0.5)
#' )
#' @export
mixture <- function(..., weights) {
  dists <- rlang::list2(...)
  if (length(dists) < 2L) {
    stop("Please provide at least two distributions.", call. = FALSE)
  }
  if (
    !all(sapply(dists, function(x) {
      inherits(x, c("enfold_discrete", "enfold_range"))
    }))
  ) {
    stop(
      "All components must be created by make_discrete() or make_range().",
      call. = FALSE
    )
  }
  if (length(weights) != length(dists)) {
    stop(
      "Length of weights must match the number of distributions.",
      call. = FALSE
    )
  }
  if (any(weights < 0)) {
    stop("Weights must be non-negative.", call. = FALSE)
  }
  if (sum(weights) == 0) {
    stop("At least one weight must be positive.", call. = FALSE)
  }

  # Check if mixture is fully discrete
  all_discrete <- all(vapply(
    dists,
    function(x) inherits(x, "enfold_discrete"),
    logical(1L)
  ))

  # Normalize weights to sum to 1
  weights <- weights / sum(weights)

  sample_fun <- function(n) {
    comps <- sample.int(length(dists), n, replace = TRUE, prob = weights)
    sapply(comps, function(i) dists[[i]]$sample_fun(1L))
  }

  structure(
    list(
      distributions = dists,
      weights = weights,
      sample_fun = sample_fun,
      discrete_mixture = all_discrete
    ),
    class = "enfold_mixture"
  )
}

#' @export
print.enfold_mixture <- function(x, ...) {
  cat("enfold_mixture:\n")
  for (i in seq_along(x$distributions)) {
    dist <- x$distributions[[i]]
    weight <- x$weights[i]
    cat(sprintf("  Component %d (weight = %.2f):\n", i, weight))

    dist_lines <- capture.output(print(dist, ...))
    cat(paste0("    ", dist_lines, collapse = "\n"), "\n", sep = "")
  }
  invisible(x)
}


#' Specify hyperparameters for a grid search
#'
#' Creates an \code{enfold_hyperparameters} object that describes the search
#' space for grid constructors such as \code{\link{grd_random}}. Each named
#' argument defines one hyperparameter and must be either a
#' \code{\link{make_discrete}} or a \code{\link{make_range}} object.
#'
#' @param ... Named hyperparameter specifications. Accepted types:
#'   \itemize{
#'     \item An \code{enfold_discrete} object from \code{\link{make_discrete}} —
#'       discrete candidates, sampled at run time.
#'     \item An \code{enfold_range} object from \code{\link{make_range}} —
#'       continuous range, sampled at run time. Pass
#'       \code{sample_space = sample_log_uniform()} to \code{make_range} for
#'       log-scale sampling.
#'     \item An \code{enfold_mixture} object from \code{\link{mixture}} —
#'       mixture of discrete and/or continuous distributions.
#'   }
#' @return An object of class \code{enfold_hyperparameters}.
#' @details
#' Pipe the result through \code{\link{forbid}} to exclude invalid parameter
#' combinations. To sample on the log scale, pass
#' \code{sample_space = sample_log_uniform()} to \code{\link{make_range}}.
#' @seealso \code{\link{make_discrete}}, \code{\link{make_range}},
#'   \code{\link{mixture}}, \code{\link{sample_uniform}},
#'   \code{\link{sample_log_uniform}}, \code{\link{forbid}},
#'   \code{\link{draw}}, \code{\link{grd_random}}
#' @examples
#' specify_hyperparameters(
#'   alpha  = make_discrete(0, 0.5, 1),
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
  bad <- !sapply(hyperparams, function(x) {
    inherits(x, "enfold_discrete") ||
      inherits(x, "enfold_range") ||
      inherits(x, "enfold_mixture")
  })
  if (any(bad)) {
    stop(
      "Each hyperparameter must be make_discrete(), make_range(), or mixture(). Bad args: ",
      paste(names(hyperparams)[bad], collapse = ", "),
      call. = FALSE
    )
  }

  # Weird check: ensure no name passed has forward slash, which would be... never, hopefully?
  if (any(grepl("/", names(hyperparams)))) {
    stop(
      "Hyperparameter names cannot contain forward slashes; they are a reserved symbol.",
      call. = FALSE
    )
  }

  attr(hyperparams, "forbidden_logic") <- list()

  sample_funs <- lapply(hyperparams, function(x) {
    if (
      inherits(x, "enfold_range") ||
        inherits(x, "enfold_discrete") ||
        inherits(x, "enfold_mixture")
    ) {
      x$sample_fun
    } else {
      stop("Invalid hyperparameter specification.", call. = FALSE)
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
#'   alpha  = make_discrete(0, 0.5, 1),
#'   lambda = make_discrete(0.01, 0.1, 1)
#' ) |>
#'   forbid(alpha == 0 && lambda == 0.01)
#'
#' draw(params)  # 0,0.01 combination absent
#' @export
forbid <- function(obj, ...) {
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

#' Draw hyperparameter combinations
#'
#' Samples \code{n} valid hyperparameter combinations from an
#' \code{enfold_hyperparameters} specification, respecting \code{\link{forbid}}
#' constraints.
#'
#' @param obj An object of class \code{enfold_hyperparameters}.
#' @param n Integer. Number of combinations to draw. When \code{NULL} and all
#'   parameters are discrete, every valid combination is returned.
#' @return A data frame with one row per combination and one column per
#'   hyperparameter. Columns are regular atomic vectors when candidates are
#'   scalar; list-columns (via \code{I()}) when candidates are vector-valued
#'   (\code{make_discrete(..., is_separate = FALSE)}).
#' @details
#' When all parameters are discrete, sampling is performed without replacement.
#' For specifications containing at least one continuous \code{enfold_range},
#' \code{n} must be a positive integer.
#' @seealso \code{\link{specify_hyperparameters}}, \code{\link{forbid}}
#' @examples
#' params <- specify_hyperparameters(
#'   alpha  = make_discrete(0, 0.5, 1),
#'   lambda = make_discrete(0.01, 0.1, 1)
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
    function(x) {
      inherits(x, "enfold_discrete") ||
        (inherits(x, "enfold_mixture") && x$discrete_mixture)
    },
    logical(1L)
  ))

  if (all_discrete) {
    combos <- expand_grid_params(obj)
    valid <- Filter(
      function(combo) !is_forbidden_combo(combo, constraints),
      combos
    )

    if (length(valid) == 0L) {
      return(combos_to_df(list(), param_names))
    }

    if (is.null(n) || n >= length(valid)) {
      return(combos_to_df(valid, param_names))
    }

    if (!is.numeric(n) || length(n) != 1L || n < 1L) {
      stop("`n` must be a single positive integer or NULL.", call. = FALSE)
    }
    n <- as.integer(n)

    chosen <- valid[sample.int(length(valid), n)]
    return(combos_to_df(chosen, param_names))
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

  valid_combos <- list()
  attempts <- 0L
  max_attempts <- 100L

  while (length(valid_combos) < n && attempts < max_attempts) {
    batch_size <- max((n - length(valid_combos)) * 4L, 50L)

    drawn <- lapply(sample_funs, function(fn) fn(batch_size))

    batch <- lapply(seq_len(batch_size), function(i) {
      lapply(
        stats::setNames(param_names, param_names),
        function(nm) {
          d <- drawn[[nm]]
          if (is.list(d)) d[[i]] else d[[i]]
        }
      )
    })

    passed <- Filter(
      function(combo) !is_forbidden_combo(combo, constraints),
      batch
    )
    valid_combos <- c(valid_combos, passed)
    attempts <- attempts + 1L
  }

  if (length(valid_combos) < n) {
    stop(
      "Unable to draw the requested number of valid hyperparameter combinations after repeated sampling.",
      call. = FALSE
    )
  }

  combos_to_df(valid_combos[seq_len(n)], param_names)
}


#' Create a continuous hyperparameter range
#'
#' Defines a closed interval \eqn{[\text{min}, \text{max}]} from which
#' \code{\link{draw}} and search engines sample. Pass
#' \code{sample_space = \link{sample_log_uniform}()} for log-scale sampling.
#'
#' @param min Numeric scalar. Lower bound (inclusive).
#' @param max Numeric scalar. Upper bound (inclusive). Must be strictly greater
#'   than \code{min}.
#' @param sample_space A zero-argument factory that returns a sampling function
#'   with a single argument \code{n}. Use \code{\link{sample_uniform}} (default)
#'   for uniform sampling or \code{\link{sample_log_uniform}} for log-scale
#'   sampling.
#' @return An object of class \code{enfold_range}.
#' @seealso \code{\link{specify_hyperparameters}}, \code{\link{sample_uniform}},
#'   \code{\link{sample_log_uniform}}
#' @examples
#' make_range(0.001, 10)
#' make_range(1e-4, 1, sample_space = sample_log_uniform())
#'
#' # Use inside specify_hyperparameters
#' specify_hyperparameters(lambda = make_range(1e-4, 1))
#' @export
make_range <- function(min, max, sample_space = sample_uniform()) {
  if (
    !is.numeric(min) ||
      length(min) != 1L ||
      !is.numeric(max) ||
      length(max) != 1L
  ) {
    stop("`min` and `max` must each be a single numeric value.", call. = FALSE)
  }

  if (!identical(names(formals(sample_space)), "n")) {
    stop(
      "`sample_space` must be a function with a single argument named `n`.",
      call. = FALSE
    )
  }

  if (min >= max) {
    stop("`min` must be strictly less than `max`.", call. = FALSE)
  }

  sample_fun <- sample_space
  env <- new.env(parent = environment(sample_fun))
  env$lower <- min
  env$upper <- max
  environment(sample_fun) <- env

  # Validate single draw
  test_draw <- try(sample_fun(1L), silent = TRUE)
  if (
    inherits(test_draw, "try-error") ||
      is.nan(test_draw) ||
      is.infinite(test_draw)
  ) {
    stop(
      "The provided `sample_space` function produced an invalid value when called with n = 1.\n",
      "A common issue is using a range with lower bound 0 for log-uniform sampling.",
      call. = FALSE
    )
  }

  structure(
    list(
      min = min,
      max = max,
      sample_fun = sample_fun
    ),
    class = "enfold_range"
  )
}


# ── Print methods ──────────────────────────────────────────────────────────────

#' @export
print.enfold_range <- function(x, ...) {
  cat(sprintf("enfold_range [%g, %g]\n", x$min, x$max))
  invisible(x)
}

#' @export
print.enfold_hyperparameters <- function(x, ...) {
  cat("enfold_hyperparameters:\n")
  for (nm in names(x)) {
    val <- x[[nm]]
    if (inherits(val, "enfold_range")) {
      cat(sprintf("  %s: enfold_range [%g, %g]\n", nm, val$min, val$max))
    } else if (inherits(val, "enfold_discrete")) {
      n <- length(val$values)
      preview <- utils::head(val$values, 3L)
      pstr <- paste(vapply(preview, deparse, character(1L)), collapse = ", ")
      if (n > 3L) {
        pstr <- paste0(pstr, ", ...")
      }
      cat(sprintf("  %s: make_discrete(%s)\n", nm, pstr))
    } else {
      cat(sprintf("  %s: %s\n", nm, paste(val, collapse = ", ")))
    }
  }
  invisible(x)
}


# ── Helpers ────────────────────────────────────────────────────────────────────

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

# Enumerate all combinations of discrete hyperparameters.
# Returns a list of named lists, one per combination.
# Handles both scalar and vector-valued candidates (enfold_discrete).
expand_grid_params <- function(hyperparams) {
  nms <- names(hyperparams)
  candidates <- lapply(hyperparams, function(v) {
    if (inherits(v, "enfold_range")) {
      stop(
        "expand_grid_params() cannot enumerate continuous enfold_range values.",
        call. = FALSE
      )
    }
    if (inherits(v, "enfold_discrete")) {
      if (is.list(v$values)) v$values else as.list(v$values)
    } else {
      as.list(v)
    }
  })

  sizes <- vapply(candidates, length, integer(1L))
  n_combos <- prod(sizes)

  result <- vector("list", n_combos)
  for (p in seq_len(n_combos)) {
    combo <- vector("list", length(nms))
    names(combo) <- nms
    rem <- p - 1L
    for (i in seq_along(candidates)) {
      idx <- (rem %% sizes[[i]]) + 1L
      combo[[i]] <- candidates[[i]][[idx]]
      rem <- rem %/% sizes[[i]]
    }
    result[[p]] <- combo
  }
  result
}

# Convert a list of named lists (one per combo) to a data frame.
# Uses list-columns (I()) for vector-valued candidates; plain columns otherwise.
combos_to_df <- function(combos, param_names) {
  if (length(combos) == 0L) {
    return(data.frame(
      matrix(
        ncol = length(param_names),
        nrow = 0L,
        dimnames = list(NULL, param_names)
      )
    ))
  }
  cols <- lapply(param_names, function(nm) {
    vals <- lapply(combos, function(combo) combo[[nm]])
    if (
      all(vapply(
        vals,
        function(v) is.atomic(v) && length(v) == 1L,
        logical(1L)
      ))
    ) {
      unlist(vals)
    } else {
      I(vals)
    }
  })
  names(cols) <- param_names
  df <- do.call(data.frame, c(cols, list(stringsAsFactors = FALSE)))
  rownames(df) <- NULL
  df
}

# Extract row i from a draw() data frame as a proper named list.
# Works for both regular atomic columns and list-columns (from vector-valued candidates).
combo_row <- function(df, i) {
  lapply(
    stats::setNames(names(df), names(df)),
    function(nm) df[[nm]][[i]]
  )
}

# Some helpers to create search spaces

#' Uniform Sampler for Continuous Hyperparameters
#'
#' Provides a sampling function that generates uniform random values.
#' Is the default sampling method for continuous hyperparameters defined by \code{make_range}.
#'
#' @returns A function of n that generates n random values uniformly distributed between the specified min and max of the range.
#' @export
sample_uniform <- function() {
  sample_fun <- function(n) runif(n, min = lower, max = upper)
}


#' Log-Uniform Sampler for Continuous Hyperparameters
#'
#' Provides a sampling function that generates log-uniform random values.
#'
#' @returns A function of n that generates n random values log-uniformly distributed between the specified min and max of the range.
#' @export
sample_log_uniform <- function() {
  sample_fun <- function(n) exp(runif(n, min = log(lower), max = log(upper)))
}
