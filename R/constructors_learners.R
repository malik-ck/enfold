#==============================================================================
# For learners
#==============================================================================

# fit() and predict() generics for individual learners

#' Fit a learner or task
#'
#' Generic function for fitting learners, pipelines, lists, and complete
#' \code{enfold_task} objects. Dispatches to the appropriate method based on
#' the class of \code{object}.
#'
#' @param object An \code{enfold_learner}, \code{enfold_pipeline},
#'   \code{enfold_list}, or \code{enfold_task} object.
#' @param ... Additional arguments passed to the specific method. For
#'   \code{enfold_learner} and related classes, \code{x} (predictors) and
#'   \code{y} (outcome) are required positional arguments. For
#'   \code{enfold_task}, see \code{\link{fit.enfold_task}}.
#' @return A fitted version of the input object. The class gains a
#'   \code{_fitted} suffix (e.g. \code{enfold_learner_fitted}).
#' @seealso \code{\link{fit.enfold_task}}
#' @export
fit <- function(object, ...) {
  UseMethod("fit")
}

#' @export
fit.enfold_learner <- function(object, x, y, ...) {
  copy_structure <- object
  copy_structure$model <- object$fit(x = x, y = y)
  class(copy_structure) <- c("enfold_learner_fitted", "enfold_learner")
  copy_structure
}

#' @export
fit.enfold_list <- function(object, x, y, ...) {
  copy_structure <- object
  copy_structure$model <- object$fit(x = x, y = y)
  class(copy_structure) <- c(
    "enfold_list_fitted",
    "enfold_list",
    "enfold_learner"
  )
  copy_structure
}

#' @export
predict.enfold_learner <- function(object, newdata, ...) {
  stop("Please train this learner before using it for predictions.")
}

#' @export
predict.enfold_learner_fitted <- function(object, newdata, ...) {
  object$preds(object$model, newdata)
}

#' @export
predict.enfold_list_fitted <- function(object, newdata, ...) {
  get_preds <- object$preds(object$model, newdata)

  if (!is.list(get_preds)) {
    stop(
      "While trying to predict with learner '",
      object$name,
      "':\n",
      "Expected predictions to be a list, but got an object of class '",
      class(get_preds)[1L],
      "'.",
      call. = FALSE
    )
  }
  if (is.null(names(get_preds)) || any(names(get_preds) == "")) {
    stop(
      "While trying to predict with learner '",
      object$name,
      "':\n",
      "The predictions list must have names for each entry, but some are missing.",
      call. = FALSE
    )
  }
  if (any(duplicated(names(get_preds)))) {
    stop(
      "While trying to predict with learner '",
      object$name,
      "':\n",
      "The predictions list must have unique names for each entry, but duplicates were found.",
      call. = FALSE
    )
  }

  return(get_preds)
}

# Also printing!
#' @export
print.enfold_learner <- function(x, ...) {
  cat(
    "Learner object with name ",
    x$name,
    ".\nNot yet fitted.\n",
    sep = ""
  )
  invisible(x)
}

#' @export
print.enfold_learner_fitted <- function(x, ...) {
  cat(
    "Learner object with name ",
    x$name,
    ".\nHas been fitted.\n",
    sep = ""
  )
  invisible(x)
}

#' @export
print.enfold_list <- function(x, ...) {
  cat(
    "List learner object with name ",
    x$name,
    ".\nNot yet fitted.\n",
    sep = ""
  )
  invisible(x)
}

#' @export
print.enfold_list_fitted <- function(x, ...) {
  cat(
    "List learner object with name ",
    x$name,
    ".\nHas been fitted.\n",
    sep = ""
  )
  invisible(x)
}

#' Create a learner factory
#'
#' Returns a constructor function for \code{enfold_learner} objects. The
#' constructor captures \code{fit} and \code{preds} in a minimal closure
#' together with whatever hyperparameters you declare via \code{...}, keeping
#' the closure environment small.
#'
#' @param fit A function with arguments \code{(x, y)} that trains a model and
#'   returns any object that \code{preds} can consume.
#' @param preds A function with arguments \code{(object, data)} where
#'   \code{object} is the return value of \code{fit} and \code{data} is the
#'   new predictor matrix or data frame. Returns raw predictions — no type
#'   constraints.
#' @param ... Hyperparameters. Bare names (e.g. \code{alpha}) become required
#'   arguments of the constructor; named values (e.g. \code{alpha = 1}) become
#'   optional arguments with defaults. Names \code{name}, \code{fit}, and
#'   \code{preds} are reserved and will cause an error.
#' @param expect_list Logical. If \code{TRUE} the returned object gets class
#'   \code{enfold_list} and \code{predict()} enforces that the output is a
#'   fully-named list with no duplicate names.
#' @return A constructor \code{function(name, ...)} that builds an
#'   \code{enfold_learner} (or \code{enfold_list} when \code{expect_list =
#'   TRUE}).
#' @details
#' The closure environment of the returned learner contains only the
#' hyperparameter values, the \code{fit} function, and the \code{preds}
#' function — nothing else. Use \code{\link{get_params}} to inspect
#' hyperparameter values and \code{\link{inspect}} to recover the original
#' function bodies.
#'
#' @examples
#' # Simple mean-prediction factory
#' lrn_mean_factory <- make_learner_factory(
#'   fit   = function(x, y) mean(y),
#'   preds = function(object, data) rep(object, nrow(data))
#' )
#' lrn <- lrn_mean_factory(name = "mean")
#' fitted_lrn <- fit(lrn, x = mtcars[, -1], y = mtcars$mpg)
#' head(predict(fitted_lrn, newdata = mtcars[, -1]))
#'
#' # Factory with a required hyperparameter
#' lrn_poly_factory <- make_learner_factory(
#'   fit = function(x, y) {
#'     lm(y ~ poly(x[, 1], degree = degree))
#'   },
#'   preds = function(object, data) {
#'     predict(object, newdata = data.frame(x = data[, 1]))
#'   },
#'   degree  # required — no default
#' )
#' @export
make_learner_factory <- function(fit, preds, ..., expect_list = FALSE) {
  factory_ns <- parent.env(environment())
  raw_dots <- substitute(list(...))[-1]

  # Check whether any names are bad
  bad_names <- c("name", "fit", "preds")
  clashing <- intersect(names(raw_dots), bad_names)
  if (length(clashing) > 0) {
    stop(
      "It is not allowed to pass arguments to 'make_learner_factory' ",
      "named 'name', 'fit', or 'preds'. Clashing: ",
      paste(clashing, collapse = ", ")
    )
  }

  # Check that fit and preds only have as arguments the two arguments they need
  extra_fit_args <- setdiff(names(formals(fit)), c("x", "y"))
  extra_preds_args <- setdiff(names(formals(preds)), c("object", "data"))
  if (length(c(extra_fit_args, extra_preds_args)) > 0) {
    stop(
      "fit and preds must only have (x, y) and (object, data) as arguments, ",
      "respectively.\nEvery argument you pass into make_learner_factory() ",
      "is available inside fit() and predict() regardless.",
      call. = FALSE
    )
  }

  # Build formals as a plain list using alist() for missing-value sentinels
  constr_args <- alist(name = ) # name is always required

  for (i in seq_along(raw_dots)) {
    arg_name <- names(raw_dots)[[i]]
    if (is.null(arg_name) || arg_name == "") {
      actual_name <- as.character(raw_dots[[i]])
      new_arg <- alist(x = )
      names(new_arg) <- actual_name
      constr_args <- c(constr_args, new_arg)
    } else {
      constr_args[arg_name] <- list(raw_dots[[i]]) # single bracket + list()
    }
  }

  # Always inject parameters = NULL as the last named arg (before expect_list)
  constr_args["parameters"] <- list(NULL)

  constructor <- function() {}
  hyperparam_names <- setdiff(names(constr_args), c("name", "parameters"))
  formals(constructor) <- constr_args

  body(constructor) <- bquote({
    p <- mget(.(hyperparam_names), envir = environment())

    # Minimal environment: only fit, preds, p, nothing else
    closure_env <- list2env(p, parent = .(factory_ns))
    environment(fit) <- closure_env
    environment(preds) <- closure_env
    closure_env$fit <- fit
    closure_env$preds <- preds

    wrapped_fit <- function(x, y) fit(x = x, y = y)
    environment(wrapped_fit) <- closure_env

    wrapped_preds <- function(object, data) preds(object = object, data = data)
    environment(wrapped_preds) <- closure_env

    class_assign <- if (expect_list) {
      c("enfold_list", "enfold_learner")
    } else {
      "enfold_learner"
    }

    learner_obj <- structure(
      list(name = name, fit = wrapped_fit, preds = wrapped_preds),
      class = class_assign
    )

    # If parameters provided, validate names and return a bare enfold_grid
    if (!is.null(parameters)) {
      bad_params <- setdiff(names(parameters), .(hyperparam_names))
      if (length(bad_params) > 0L) {
        stop(
          "parameters contains names not recognised by this constructor: ",
          paste(bad_params, collapse = ", "),
          call. = FALSE
        )
      }
      return(make_bare_grid(name, learner_obj, parameters))
    }

    learner_obj
  })

  constructor
}

#' Retrieve hyperparameters from a learner
#'
#' Extracts the named hyperparameter list captured in an
#' \code{enfold_learner}'s closure environment.
#'
#' @param learner An \code{enfold_learner} or \code{enfold_learner_fitted}
#'   object.
#' @param ... Ignored.
#' @return A named list of hyperparameter values.
#' @seealso \code{\link{inspect}} for a more verbose inspection including
#'   the original \code{fit} and \code{preds} function bodies.
#' @examples
#' lrn <- lrn_glm("my_glm", family = gaussian())
#' get_params(lrn)
#' @export
get_params <- function(learner, ...) {
  UseMethod("get_params")
}

#' @export
get_params.enfold_learner <- function(learner, ...) {
  env <- environment(learner$fit)
  reserved <- c("fit", "preds")
  keys <- setdiff(ls(env), reserved)
  mget(keys, envir = env)
}

#' @export
get_params.enfold_learner_fitted <- function(learner, ...) {
  env <- environment(learner$fit)
  reserved <- c("fit", "preds")
  keys <- setdiff(ls(env), reserved)
  mget(keys, envir = env)
}

get_original_fit <- function(learner) {
  environment(learner$fit)$fit
}

get_original_preds <- function(learner) {
  environment(learner$fit)$preds
}

#' Inspect an enfold learner
#'
#' Prints a human-readable summary of an \code{enfold_learner}, including its
#' name, hyperparameters, and original \code{fit}/\code{preds} function bodies.
#'
#' @param learner An \code{enfold_learner} object.
#' @param ... Ignored.
#' @return Invisibly returns a named list with elements \code{fit},
#'   \code{preds}, and \code{params}.
#' @examples
#' lrn <- lrn_glm("my_glm", family = gaussian())
#' inspect(lrn)
#' @export
inspect <- function(learner, ...) {
  UseMethod("inspect")
}

#' @rdname inspect
#' @export
inspect.enfold_learner <- function(learner, ...) {
  cat("Learner Name: ", learner$name, "\n")
  cat("============================\n\n")

  # 1. Show Hyperparameters (using our environment trick)
  params <- get_params(learner)
  cat("--- Additional arguments ---\n")
  if (length(params) > 0) {
    print(params)
  } else {
    cat("None\n")
  }

  # 2. Extract the original functions
  # Note: I updated the indices to match the 'snapped' constructor body
  orig_fit <- get_original_fit(learner)
  orig_preds <- get_original_preds(learner)

  cat("\n--- Original Fit Logic ---\n")
  print(orig_fit)

  cat("\n--- Original Predict Logic ---\n")
  print(orig_preds)

  # Return them in a named list so the user can "intercept" them
  invisible(list(
    fit = orig_fit,
    preds = orig_preds,
    params = params
  ))
}
