#' Risk set learner wrapper
#'
#' Wraps a base learner so that it operates on person-time risk sets rather
#' than the original row-per-person data. During fitting, each row is expanded
#' to one row per event time the subject was at risk for; during prediction,
#' the per-time-point probabilities are optionally collapsed back to one
#' event probability per subject via the complement of the survival product.
#'
#' @param name Character. Name for the wrapped learner.
#' @param learner An \code{enfold_learner}, \code{enfold_list},
#'   \code{enfold_pipeline}, or \code{enfold_grid}. Should return predicted
#'   probabilities (values in \eqn{[0, 1]}) and accept binary outcomes.
#' @param time Character. Column name in \code{x} that stores the
#'   (stop-)time for each subject.
#' @param time_start Character or \code{NULL}. Column name for the start
#'   time. When \code{NULL}, a start time of 0 is assumed for all subjects.
#' @param collapse_back Logical. If \code{TRUE} (default), per-time-point
#'   predictions are collapsed back to one probability per subject using
#'   \eqn{1 - \prod_t (1 - p_t)}. If \code{FALSE}, predictions are returned
#'   at the expanded (person-time) level.
#' @param expect_list Ignored (determined automatically from \code{learner}
#'   class).
#' @return An \code{enfold_learner} (or \code{enfold_grid} if \code{learner}
#'   is a grid) with class \code{enfold_risk_set} prepended.
#' @seealso \code{\link{wrp_stratify}}, \code{\link{make_pipeline}}
#' @export
wrp_risk_set <- function(
  name,
  learner,
  time,
  time_start = NULL,
  collapse_back = TRUE,
  expect_list = NULL
) {
  if (inherits(learner, "enfold_risk_set")) {
    stop(
      "`wrp_risk_set()` has already been applied to this learner.",
      call. = FALSE
    )
  }

  if (inherits(learner, "enfold_grid")) {
    return(wrp_risk_set_grid(
      name = name,
      grid = learner,
      time = time,
      time_start = time_start,
      collapse_back = collapse_back
    ))
  }

  if (
    !inherits(learner, c("enfold_learner", "enfold_list", "enfold_pipeline"))
  ) {
    stop(
      "Learner must be an enfold_learner, enfold_list, enfold_pipeline, or enfold_grid."
    )
  }

  wrap_risk_set_learner(
    name = name,
    learner = learner,
    time = time,
    time_start = time_start,
    collapse_back = collapse_back
  )
}

wrap_risk_set_learner <- function(
  name,
  learner,
  time,
  time_start,
  collapse_back
) {
  do_expect_list <- inherits(learner, c("enfold_list", "enfold_pipeline"))

  wrapped <- make_learner_factory(
    fit = function(x, y) {
      expansion <- expand_to_risk_sets(
        data = x,
        time_col = time,
        time_start_col = time_start,
        time_grid = NULL
      )

      raw_expanded_y <- y[expansion$original_ids]
      is_last_row <- !duplicated(expansion$original_ids, fromLast = TRUE)
      expanded_y <- ifelse(is_last_row, raw_expanded_y, 0)

      fitted_base <- enfold::fit(learner, x = expansion$data, y = expanded_y)

      list(
        fitted_base = fitted_base,
        time_grid = expansion$time_grid,
        time_col = time,
        time_start_col = time_start,
        collapse_back = collapse_back
      )
    },

    preds = function(object, data) {
      expansion <- expand_to_risk_sets(
        data = data,
        time_col = object$time_col,
        time_start_col = object$time_start_col,
        time_grid = object$time_grid
      )

      raw_preds <- stats::predict(object$fitted_base, newdata = expansion$data)

      if (!object$collapse_back) {
        attr(raw_preds, "original_ids") <- expansion$original_ids
        attr(raw_preds, "eval_times") <- expansion$eval_times
        return(raw_preds)
      }

      if (
        !inherits(
          object$fitted_base,
          c("enfold_list", "enfold_grid", "enfold_pipeline")
        )
      ) {
        collapsed <- as.numeric(
          tapply(
            as.numeric(raw_preds),
            INDEX = expansion$original_ids,
            FUN = function(x) 1 - prod(1 - x)
          )
        )
      } else {
        collapsed <- lapply(
          raw_preds,
          function(x) {
            as.numeric(tapply(
              as.numeric(x),
              INDEX = expansion$original_ids,
              FUN = function(x) 1 - prod(1 - x)
            ))
          }
        )
        names(collapsed) <- names(raw_preds)
      }

      collapsed
    },
    learner = learner,
    time = time,
    time_start = time_start,
    collapse_back = collapse_back,
    expect_list = do_expect_list
  )(
    name = name,
    learner = learner,
    time = time,
    time_start = time_start,
    collapse_back = collapse_back
  )

  class(wrapped) <- c("enfold_risk_set", class(wrapped))
  wrapped
}

wrp_risk_set_grid <- function(name, grid, time, time_start, collapse_back) {
  stop(
    "Passing an `enfold_grid` to `wrp_risk_set()` is not yet supported. ",
    "Apply `wrp_risk_set()` to a plain learner and pass the result to `make_grid()` instead.",
    call. = FALSE
  )
}


#' Stratified learner wrapper
#'
#' Trains one separate instance of a base learner per stratum of the data
#' (defined by one or more columns). At prediction time, each row is routed
#' to its corresponding fitted sub-model.
#'
#' @param name Character. Name for the wrapped learner.
#' @param learner An \code{enfold_learner}, \code{enfold_list},
#'   \code{enfold_pipeline}, or \code{enfold_grid} to stratify.
#' @param strata Character vector of column names in \code{x} used to form
#'   strata keys. Multiple columns are combined (paste-separated).
#' @param fail_on_new Logical. If \code{TRUE} (default), predicting on a
#'   stratum not seen during training raises an error. If \code{FALSE}, the
#'   unseen stratum silently receives \code{NA} predictions.
#' @return An \code{enfold_learner} (or \code{enfold_grid} when
#'   \code{learner} is a grid) with class \code{enfold_stratify} prepended.
#' @seealso \code{\link{wrp_risk_set}}, \code{\link{make_pipeline}}
#' @examples
#' \dontrun{
#' base_lrn <- lrn_glm("glm", family = gaussian())
#' strat_lrn <- wrp_stratify("strat_glm", base_lrn, strata = "cyl")
#'
#' fitted <- fit(strat_lrn, x = mtcars[, -1], y = mtcars$mpg)
#' head(predict(fitted, newdata = mtcars[, -1]))
#' }
#' @export
wrp_stratify <- function(
  name,
  learner,
  strata,
  fail_on_new = TRUE
) {
  if (inherits(learner, "enfold_stratify")) {
    stop(
      "`wrp_stratify()` has already been applied to this learner.",
      call. = FALSE
    )
  }

  if (!is.character(strata) || length(strata) < 1L) {
    stop("`strata` must be a character vector of column names.", call. = FALSE)
  }

  if (inherits(learner, "enfold_grid")) {
    return(wrp_stratify_grid(
      name = name,
      grid = learner,
      strata = strata,
      fail_on_new = fail_on_new,
      expect_list = TRUE
    ))
  }

  if (
    !inherits(learner, c("enfold_learner", "enfold_list", "enfold_pipeline"))
  ) {
    stop(
      "Learner must be an enfold_learner, enfold_list, enfold_pipeline, or enfold_grid."
    )
  }

  wrap_stratify_learner(
    name = name,
    learner = learner,
    strata = strata,
    fail_on_new = fail_on_new,
    expect_list = inherits(learner, c("enfold_list", "enfold_pipeline"))
  )
}

wrap_stratify_learner <- function(
  name,
  learner,
  strata,
  fail_on_new,
  expect_list
) {
  do_expect_list <- inherits(
    learner,
    c("enfold_list", "enfold_pipeline", "enfold_grid")
  )

  wrapped <- make_learner_factory(
    fit = function(x, y) {
      strata_keys <- make_strata_keys(extract_strata_data(x, strata))
      strata_levels <- unique(strata_keys)

      models <- stats::setNames(
        lapply(strata_levels, function(level) {
          idx <- which(strata_keys == level)
          enfold::fit(learner, x = x[idx, , drop = FALSE], y = subset_y(y, idx))
        }),
        strata_levels
      )

      list(
        models = models,
        strata = strata,
        strata_levels = strata_levels,
        fail_on_new = fail_on_new
      )
    },

    preds = function(object, data) {
      strata_keys <- make_strata_keys(extract_strata_data(data, object$strata))
      unknown <- setdiff(unique(strata_keys), object$strata_levels)

      if (length(unknown) > 0L && object$fail_on_new) {
        stop(
          "Prediction data contains unseen strata: ",
          paste0("'", unknown, "'", collapse = ", "),
          call. = FALSE
        )
      }

      result <- NULL
      for (level in unique(strata_keys)) {
        rows <- which(strata_keys == level)
        fitted_model <- object$models[[level]]
        if (is.null(fitted_model)) {
          stop(
            "No fitted model found for strata level '",
            level,
            "'.",
            call. = FALSE
          )
        }

        preds_sub <- stats::predict(
          fitted_model,
          newdata = data[rows, , drop = FALSE]
        )
        if (is.null(result)) {
          result <- allocate_predictions_like(preds_sub, nrow(data))
        }

        result <- assign_predictions(result, rows, preds_sub)
      }

      result
    },
    learner = learner,
    strata = strata,
    fail_on_new = fail_on_new,
    expect_list = do_expect_list
  )(
    name = name,
    learner = learner,
    strata = strata,
    fail_on_new = fail_on_new
  )

  class(wrapped) <- c("enfold_stratify", class(wrapped))
  wrapped
}

wrp_stratify_grid <- function(name, grid, strata, fail_on_new, expect_list) {
  stop(
    "Passing an `enfold_grid` to `wrp_stratify()` is not yet supported. ",
    "Apply `wrp_stratify()` to a plain learner and pass the result to `make_grid()` instead.",
    call. = FALSE
  )
}

extract_strata_data <- function(data, strata) {
  if (is.data.frame(data)) {
    data[strata]
  } else {
    as.data.frame(data, stringsAsFactors = FALSE)[strata]
  }
}

make_strata_keys <- function(strata_df) {
  if (ncol(strata_df) == 1L) {
    as.character(strata_df[[1L]])
  } else {
    do.call(paste, c(strata_df, sep = "\r"))
  }
}

allocate_predictions_like <- function(preds, n) {
  if (is.list(preds)) {
    stats::setNames(
      lapply(preds, allocate_single_prediction, n = n),
      names(preds)
    )
  } else {
    allocate_single_prediction(preds, n = n)
  }
}

allocate_single_prediction <- function(pred, n) {
  if (is.data.frame(pred)) {
    if (n == 0L) {
      return(pred[0L, , drop = FALSE])
    }
    out <- pred[rep(1L, n), , drop = FALSE]
    out[,] <- NA
    out
  } else if (is.matrix(pred)) {
    if (n == 0L) {
      return(pred[0L, , drop = FALSE])
    }
    out <- pred[rep(1L, n), , drop = FALSE]
    out[,] <- NA
    class(out) <- class(pred)
    out
  } else if (is.list(pred)) {
    vector("list", n)
  } else {
    vector(typeof(pred), n)
  }
}

assign_predictions <- function(result, rows, preds_sub) {
  if (!is.list(preds_sub)) {
    result[rows] <- preds_sub
    return(result)
  }

  for (nm in names(preds_sub)) {
    result[[nm]][rows] <- preds_sub[[nm]]
  }
  result
}
