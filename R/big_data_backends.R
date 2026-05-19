# ══════════════════════════════════════════════════════════════════════════════
# Big data backends — Arrow IPC (Feather) and FBM support
# ══════════════════════════════════════════════════════════════════════════════

# ── enfold_arrow_file ─────────────────────────────────────────────────────────

# Internal constructor for a file-backed Arrow IPC / Feather reference.
# Stores only the normalised path and cached row count — tiny object that is
# safe to serialise to multisession future workers. Workers re-open the
# memory-mapped file on demand inside subset_x().
new_arrow_file <- function(path) {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop(
      "Package 'arrow' is required to use a file path for `x`. ",
      "Install it with: install.packages('arrow')",
      call. = FALSE
    )
  }
  path <- normalizePath(path, mustWork = TRUE)
  tbl <- arrow::read_feather(path, as_data_frame = FALSE)
  n  <- tbl$num_rows
  nc <- tbl$num_columns
  rm(tbl)
  structure(list(path = path, nrow = n, ncol = nc), class = "enfold_arrow_file")
}

# dim() is an implicit S3 generic in R, so this method is dispatched by nrow()
# and ncol() (both call dim(x)[1L] and dim(x)[2L] respectively).
#' @export
dim.enfold_arrow_file <- function(x) c(x$nrow, x$ncol)

#' @export
print.enfold_arrow_file <- function(x, ...) {
  cat(sprintf("enfold_arrow_file | %d rows | %s\n", x$nrow, x$path))
  invisible(x)
}

# ── subset_x ──────────────────────────────────────────────────────────────────

# Type-aware row (and optionally column) subsetter for x. Returns a plain R
# type (matrix or data frame) ready to pass to learners. Materialisation always
# happens inside a fold loop, never before it, so large backends are never fully
# loaded by the calling process.
#
# @param x    Predictor object (matrix, data.frame, FBM, Arrow, or
#             enfold_arrow_file).
# @param idx  Integer vector of row indices to keep.
# @param cols Column specification to restrict to: NULL (all columns),
#             character vector of column names, or integer vector of column
#             positions.  NULL is the default and preserves full-width behaviour.
subset_x <- function(x, idx, cols = NULL) {
  # Resolve column positions for backends that require integer indices (FBM).
  cols_idx <- if (!is.null(cols) && inherits(x, "FBM")) {
    if (is.character(cols)) {
      match(cols, colnames(x))
    } else {
      as.integer(cols)
    }
  }

  if (inherits(x, "FBM")) {
    # bigstatsr: [ reads only the requested rows from the backing file → matrix.
    if (is.null(cols)) {
      x[idx, , drop = FALSE]
    } else {
      x[idx, cols_idx, drop = FALSE]
    }
  } else if (inherits(x, "enfold_arrow_file")) {
    # File-backed Feather/IPC: re-open as memory-mapped table, subset, materialise.
    tbl <- arrow::read_feather(x$path, as_data_frame = FALSE)
    if (is.null(cols)) {
      as.data.frame(tbl[idx, ])
    } else {
      as.data.frame(tbl[idx, cols])
    }
  } else if (inherits(x, "ArrowTabular")) {
    # In-memory Arrow Table or RecordBatch → data frame.
    if (is.null(cols)) {
      as.data.frame(x[idx, ])
    } else {
      as.data.frame(x[idx, cols])
    }
  } else {
    # Matrix or data frame: standard 2D subsetting.
    if (is.null(cols)) {
      x[idx, , drop = FALSE]
    } else {
      x[idx, cols, drop = FALSE]
    }
  }
}

# ── detect_x_pkgs ─────────────────────────────────────────────────────────────

# Returns the package names that future workers need to load to handle x.
detect_x_pkgs <- function(x) {
  pkgs <- character(0L)
  if (inherits(x, c("ArrowTabular", "enfold_arrow_file"))) pkgs <- c(pkgs, "arrow")
  if (inherits(x, "FBM"))                                  pkgs <- c(pkgs, "bigstatsr")
  pkgs
}
