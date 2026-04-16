#' Print the date of the last git commit
#'
#' A development helper that prints the date of the most recent commit.
#' Requires git to be available on the system PATH.
#'
#' @return Called for its side effect (message). Returns \code{invisible(NULL)}.
#' @keywords internal
#' @export
last_update <- function() {
  message(
    "Last push: ",
    system(
      "git log -1 --format=%cd",
      intern = TRUE
    )
  )
}


# Write a transform-type function that applies to variable names
quick_transform <- function(data, name, replacement) {
  data[, name] <- replacement
  return(data)
}

# Column applier
apply_cols <- function(X, FUN, ...) {
  if (is.data.frame(X)) {
    lapply(X, FUN, ...)
  } else if (is.matrix(X)) {
    apply(X, 2, FUN, ...)
  } else {
    stop("Input must be a data frame or matrix")
  }
}

# Instruction list creator for the use with GLMs and the likes that handles factors well
create_instruction_list <- function(data) {
  # First get column indices
  potential_vars <- seq_len(ncol(data))

  # Start by identifying columns with zero variance
  identify_constants <- which(unlist(apply_cols(data, function(x) {
    length(unique(x)) <= 1
  })))

  # Now iterate over all others and, if not numeric, explicitly enumerate their levels
  instruction_list <- vector("list", length(potential_vars))

  for (i in potential_vars) {
    # Ensure that the current class is valid
    if (
      !inherits(
        data[, i],
        c(
          "numeric",
          "logical",
          "POSIXct",
          "POSIXt",
          "difftime",
          "Date",
          "integer",
          "character",
          "factor",
          "ordered"
        )
      )
    ) {
      get_class <- class(data[, i])

      stop(
        paste0(
          "Invalid column class detected: ",
          paste(get_class, collapse = ", "),
          ". Please ensure each predictor inherits one of the following classes:\nnumeric, logical, POSIXct, POSIXt, difftime, Date, integer, character, factor, ordered."
        )
      )
    }

    # First, set instructions to NULL if constant (i.e., not part of design matrix)
    if (i %in% identify_constants) {
      instruction_list[[i]] <- "ignore"
    } else if (inherits(data[, i], "numeric")) {
      # If numeric, just return the vector
      instruction_list[[i]] <- "identity"
    } else if (
      inherits(
        data[, i],
        c("logical", "POSIXct", "POSIXt", "difftime", "Date", "integer")
      )
    ) {
      # If one of these, coerce to numeric and return
      instruction_list[[i]] <- "coerce_numeric"
    } else if (inherits(data[, i], c("character", "factor", "ordered"))) {
      # Most complex handling here: Enumerate all levels in data and dummy code, taking as reference the class with most 1's
      get_lvls <- unique(data[, i])

      # Get the largest class...
      get_class_sizes <- rep(NA, length(get_lvls))
      get_current_predictor <- data[, i]

      get_class_sizes <- tabulate(match(get_current_predictor, get_lvls))

      # Remove the largest level from dummy variable creation to make it the reference, also coerce to character for it to lose unnecessary attributes
      get_reference <- get_lvls[which.max(get_class_sizes)]
      to_indicate <- as.character(setdiff(get_lvls, get_reference))

      # Put that character vector into the instruction list
      instruction_list[[i]] <- list(search_lvls = to_indicate)
    }
  }

  names(instruction_list) <- colnames(data)

  return(instruction_list)
}

# Safe matrix creator given x and an instruction list
make_safe_matrix <- function(data, instr_list) {
  # First, determine how many columns we will need
  track_col_n <- 0

  for (i in seq_along(instr_list)) {
    if (instr_list[[i]] == "ignore") {
      track_col_n <- track_col_n # Do not add anything here
    } else if (!is.list(instr_list[[i]])) {
      track_col_n <- track_col_n + 1 # Single column added whenever not a list
    } else {
      track_col_n <- track_col_n + length(instr_list[[i]][[1]]) # Add number of categories if list
    }
  }

  # Create empty matrix
  built_mat <- matrix(ncol = track_col_n, nrow = nrow(data))

  # Now build matrix, track column names and filled columns along the way
  current_col <- 1
  assigned_names <- rep(NA, track_col_n)

  for (i in seq_along(instr_list)) {
    if (instr_list[[i]] == "ignore") {
      next
    } else if (instr_list[[i]] == "identity") {
      built_mat[, current_col] <- data[, i]
      assigned_names[[current_col]] <- colnames(data)[[i]]
      current_col <- current_col + 1
    } else if (instr_list[[i]] == "coerce_numeric") {
      built_mat[, current_col] <- as.numeric(data[, i])
      assigned_names[[current_col]] <- colnames(data)[[i]]
      current_col <- current_col + 1
    } else if (is.list(instr_list[[i]])) {
      add_levels <- instr_list[[i]][[1]]

      current_var <- data[, i]

      for (k in seq_along(add_levels)) {
        current_dummy <- ifelse(current_var == add_levels[[k]], 1, 0)
        built_mat[, current_col] <- current_dummy
        assigned_names[[current_col]] <- paste0(
          colnames(data)[[i]],
          add_levels[[k]]
        )
        current_col <- current_col + 1
      }
    } else {
      stop("Whoops, the instruction list contained an unexpected value!")
    }
  }

  # Now have matrix filled, can assign names and return it
  colnames(built_mat) <- assigned_names

  return(built_mat)
}

# Truncated power basis splines, useful in combination with ridge penalties.
tps <- function(
  x,
  num_knots = 20,
  knot_seq = NULL,
  degree = 3,
  intercept = FALSE
) {
  integer_checker(degree, "the spline degree.", require_positive = FALSE)
  integer_checker(num_knots, "the number of knots.")
  if (degree < 0) {
    stop("Degree needs to be non-negative.")
  }

  # Some typechecks
  if (!is.null(num_knots) & !is.null(knot_seq)) {
    warning(
      "Both num_knots and knot_seq are provided. knot_seq used for construction."
    )
    num_knots <- length(knot_seq)
  }

  if (is.null(num_knots) & is.null(knot_seq)) {
    warning(
      "Both num_knots and knot_seq are NULL. Creating basis with knots at all unique values."
    )
    knot_seq <- unique(sort(x))[-c(1, length(unique(x)))]
    num_knots <- length(knot_seq)
  }

  # Keep the order of x for re-ordering later
  order_x <- order(x)
  sorted_x <- sort(x)

  # Get knot values if not provided (at quantiles defined via num_knots)
  if (is.null(knot_seq)) {
    knot_seq <- stats::quantile(
      sorted_x,
      seq(0, 1, length.out = num_knots + 2)
    )[-c(1, num_knots + 2)]
  } else {
    num_knots <- length(knot_seq)
  }

  # Build the degree-th polynomial, which is the start of the design matrix
  start_mat <- matrix(ncol = intercept + degree, nrow = length(x))

  if (intercept == TRUE) {
    start_mat[, 1] <- 1
  }
  for (i in 1:degree) {
    start_mat[, i + intercept] <- sorted_x^i
  }

  # Project a sorted x into a matrix with all columns needed
  # Also adjust following columns by subtracting knot values
  # Cube at the end

  spline_mat <- `colnames<-`(
    cbind(
      start_mat,
      pmax(
        matrix(
          rep(sorted_x, times = num_knots),
          ncol = num_knots
        ) -
          matrix(
            rep(knot_seq, each = length(x)),
            ncol = num_knots
          ),
        0
      )^degree
    )[order(order_x), ],
    NULL
  )

  # Now just set some attributes
  attr(spline_mat, "knots") <- knot_seq
  attr(spline_mat, "range") <- c(sorted_x[[1]], sorted_x[[length(sorted_x)]])
  attr(spline_mat, "has_intercept") <- intercept
  attr(spline_mat, "degree") <- degree

  # Set new class
  # Might be helpful for the future, if I want to reconstruct bases via generics
  class(spline_mat) <- c("TruncatedPowerSpline", class(spline_mat))

  return(spline_mat)
}

# To get packages a function depends on
get_funs <- function(x) {
  if (is.function(x)) {
    x <- body(x)
  }
  if (is.name(x)) {
    return(NULL)
  }
  if (is.call(x)) {
    # The first element of a call is the function name
    # We want that, plus anything found inside the arguments
    f <- if (is.name(x[[1]])) as.character(x[[1]]) else NULL
    return(unique(c(f, unlist(lapply(x, get_funs)))))
  }
  if (is.recursive(x)) {
    return(unique(unlist(lapply(x, get_funs))))
  }
  return(NULL)
}

# Type-agnostic y subsetter
subset_y <- function(y, idx) {
  if (is.null(dim(y))) y[idx] else y[idx, , drop = FALSE]
}

# To extract learner names
get_learner_names <- function(learners) {
  unlist(lapply(learners, function(x) {
    if (inherits(x, "enfold_pipeline")) {
      x$path_names
    } else {
      x$name
    }
  }))
}

get_lrn_display_name <- function(lrn) {
  if (inherits(lrn, "enfold_pipeline")) {
    paste(lrn$path_names, collapse = "|")
  } else {
    lrn$name
  }
}

# Function to check whether something is an integer, with some tolerance just in case
# Can alternatively return the rounded value to make sure
integer_checker <- function(
  x,
  object_name = NULL,
  require_positive = TRUE,
  return = FALSE
) {
  if (!is.null(x)) {
    if ((abs(round(x) - x)) > 1e-16 || !is.numeric(x)) {
      if (is.null(object_name)) {
        stop(
          "Non-whole number or integer detected where one was expected. Please check inputs."
        )
      } else {
        stop(paste(
          "Please specify NULL, an integer or a whole number for",
          object_name,
          collapse = " "
        ))
      }
    }

    # Just do this separately
    if (isTRUE(require_positive) && x < 1) {
      if (is.null(object_name)) {
        stop(
          "Non-positive integer detected where one was expected. Please check inputs."
        )
      } else {
        stop(paste(
          "Please specify NULL, a positive integer or a whole number for",
          object_name,
          collapse = " "
        ))
      }
    }
  }

  if (return == FALSE) {
    return(NULL)
  } else {
    return(round(x))
  }
}

# Type-agnostic prediction combiner
combine_preds <- function(chunks) {
  first <- chunks[[1L]]
  if (is.data.frame(first) || is.matrix(first)) {
    do.call(rbind, chunks)
  } else {
    do.call(c, chunks)
  }
}

# Check package exists
.check_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      paste0(
        "Package '",
        pkg,
        "' is required for this learner. Please install it."
      ),
      call. = FALSE
    )
  }
}

.msg_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste0(
      "Package '",
      pkg,
      "' not installed. Will be needed at fit time."
    ))
  }
}

# Making risk sets
#' Helper to expand data into person-time risk sets
#'
#' @param data A matrix or data frame with one row per subject.
#' @param time_col Character. Name of the column containing the event/exit time.
#' @param time_start_col Character or \code{NULL}. Name of the column containing
#'   the entry time. If \code{NULL}, entry is assumed to be 0 for all subjects.
#' @param time_grid Numeric vector or \code{NULL}. The grid of time points at
#'   which risk sets are evaluated. If \code{NULL}, all unique event times in
#'   the training data are used.
#' @return A list with elements \code{expanded} (expanded data frame),
#'   \code{orig_row} (integer vector mapping each row back to the original
#'   subject), and \code{time_grid} (the time grid used).
#' @keywords internal
expand_to_risk_sets <- function(
  data,
  time_col,
  time_start_col = NULL,
  time_grid = NULL
) {
  n <- nrow(data)
  is_df <- is.data.frame(data)

  # Extract time boundaries safely (works for matrix and data.frame)
  t_stop <- if (time_col %in% colnames(data)) {
    data[, time_col]
  } else {
    rep(max(time_grid, na.rm = TRUE), n)
  }
  t_start <- if (
    !is.null(time_start_col) && time_start_col %in% colnames(data)
  ) {
    data[, time_start_col]
  } else {
    rep(0, n)
  }

  # If training (no grid provided), define the grid as all unique event times
  if (is.null(time_grid)) {
    time_grid <- sort(unique(t_stop[t_stop > 0]))
  }

  # Expand rows based on the valid time window for each subject
  expanded_list <- lapply(seq_len(n), function(i) {
    # Find grid points the subject was actively at risk for
    valid_times <- time_grid[time_grid > t_start[i] & time_grid <= t_stop[i]]

    # Fallback to single row if no grid points match
    if (length(valid_times) == 0) {
      valid_times <- t_stop[i]
    }

    # Replicate the row
    reps <- length(valid_times)
    row_reps <- if (is_df) {
      data[rep(i, reps), , drop = FALSE]
    } else {
      data[rep(i, reps), , drop = FALSE]
    }

    # Overwrite the time column to the evaluated grid point
    row_reps[, time_col] <- valid_times

    # Track metadata as a separate list (to avoid polluting a purely numeric matrix)
    list(
      data = row_reps,
      ids = rep(i, reps),
      eval_times = valid_times
    )
  })

  # Bind back together
  expanded_data <- do.call(rbind, lapply(expanded_list, `[[`, "data"))
  original_ids <- unlist(lapply(expanded_list, `[[`, "ids"))
  eval_times <- unlist(lapply(expanded_list, `[[`, "eval_times"))

  list(
    data = expanded_data,
    time_grid = time_grid,
    original_ids = original_ids,
    eval_times = eval_times
  )
}
