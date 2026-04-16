#' Add learners to an \code{enfold_task}
#'
#' Appends one or more learner objects to the learner slot of an
#' \code{enfold_task}. Learner names must be unique across all learners and
#' metalearners on the task.
#'
#' @param obj An object of class \code{enfold_task}.
#' @param ... One or more learner objects of class \code{enfold_learner},
#'   \code{enfold_pipeline}, \code{enfold_list}, or \code{enfold_grid}.
#' @return The updated \code{enfold_task} with the new learners appended.
#' @seealso \code{\link{add_metalearners}}, \code{\link{initialize_enfold}}
#' @examples
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian()),
#'                lrn_mean("mean"))
#' @export
add_learners <- function(obj, ...) {
  UseMethod("add_learners")
}

#' Add metalearners to an \code{enfold_task}
#'
#' Appends one or more metalearner objects to the metalearner slot of an
#' \code{enfold_task}. Metalearners combine the out-of-fold predictions
#' produced by the base learners. They require inner cross-validation folds
#' (i.e. \code{inner_cv} must be non-\code{NA} when calling
#' \code{\link{add_cv_folds}}).
#'
#' @param obj An object of class \code{enfold_task}.
#' @param ... One or more metalearner objects of class \code{enfold_learner}.
#'   Typically created via \code{mtl_*} constructors.
#' @return The updated \code{enfold_task} with the new metalearners appended.
#' @seealso \code{\link{add_learners}}, \code{\link{initialize_enfold}}
#' @examples
#' x <- mtcars[, -1]; y <- mtcars$mpg
#' task <- initialize_enfold(x, y) |>
#'   add_learners(lrn_glm("glm", family = gaussian())) |>
#'   add_metalearners(mtl_selector("selector"))
#' @export
add_metalearners <- function(obj, ...) {
  UseMethod("add_metalearners")
}

#' @rdname add_learners
#' @export
add_learners.enfold_task <- function(obj, ...) {
  new_learners <- list(...)
  validate_new_learners(new_learners)
  obj$learners <- c(obj$learners, new_learners)

  # Now check name uniqueness across learners and metalearners
  check_unique_names(obj$learners, obj$metalearners)

  # Now append dependencies
  new_pkgs <- get_lrn_packages(new_learners)
  obj$future_pkgs <- unique(c(obj$future_pkgs, new_pkgs))

  obj
}

#' @rdname add_metalearners
#' @export
add_metalearners.enfold_task <- function(obj, ...) {
  new_learners <- list(...)
  validate_new_learners(new_learners)
  obj$metalearners <- c(obj$metalearners, new_learners)

  # Now check name uniqueness across learners and metalearners
  check_unique_names(obj$learners, obj$metalearners)

  # Check that metalearners are never grids and never contain grids
  ### Still needs implementation

  # Now append dependencies
  new_pkgs <- get_lrn_packages(new_learners)
  obj$future_pkgs <- unique(c(obj$future_pkgs, new_pkgs))

  obj
}

# Some validators
validate_new_learners <- function(new_learners) {
  if (
    !all(vapply(
      new_learners,
      inherits,
      logical(1L),
      c("enfold_learner", "enfold_pipeline", "enfold_list", "enfold_grid")
    ))
  ) {
    stop(
      "All learners must be of class 'enfold_learner', 'enfold_pipeline', 'enfold_list', or 'enfold_grid'.",
      call. = FALSE
    )
  }

  if (
    any(vapply(
      new_learners,
      inherits,
      logical(1L),
      c("enfold_learner_fitted", "enfold_pipeline_fitted", "enfold_list_fitted")
    ))
  ) {
    stop("None of the learners to add can be fitted.", call. = FALSE)
  }

  invisible(NULL)
}

# Check that names across learners and metalearners are unique
check_unique_names <- function(learners, metalearners) {
  names_lrn <- lapply(learners, function(x) {
    if (inherits(x, "enfold_pipeline")) {
      enum_names <- x$path_names
    } else {
      enum_names <- x$name
    }
    return(enum_names)
  })

  names_mtl <- lapply(metalearners, function(x) {
    if (inherits(x, "enfold_pipeline")) {
      enum_names <- x$path_names
    } else {
      enum_names <- x$name
    }
    return(enum_names)
  })

  enum_all_names <- c(unlist(names_lrn), unlist(names_mtl))

  if (length(enum_all_names) != length(unique(enum_all_names))) {
    stop(
      "Please ensure that learner names are unique.\nNames can also not be shared between learners and metalearners."
    )
  }
}


# Helper: Get learner packages
get_lrn_packages <- function(learners) {
  # Small internal helper
  extract_pkgs <- function(fun_expr) {
    funs <- get_funs(fun_expr)
    unique(unlist(lapply(funs, function(fun_name) {
      loc <- utils::getAnywhere(fun_name)$where
      if (length(loc) == 0L) {
        return(NULL)
      }
      pkg_loc <- grep("^(package:|namespace:)", loc, value = TRUE)
      if (!length(pkg_loc)) {
        return(NULL)
      }
      sub("^(package:|namespace:)", "", pkg_loc)
    })))
  }

  pkgs <- unique(unlist(lapply(learners, function(x) {
    unique(c(extract_pkgs(x$fit), extract_pkgs(x$preds)))
  })))

  pkgs <- setdiff(pkgs, "base")
  unique(c(pkgs, "enfold"))
}
