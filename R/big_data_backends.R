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

# Type-aware row subsetter for x. Returns a plain R type (matrix or data frame)
# ready to pass to learners. Materialisation always happens inside a fold loop,
# never before it, so large backends are never fully loaded by the calling process.
subset_x <- function(x, idx) {
  if (inherits(x, "FBM")) {
    # bigstatsr: [ reads only the requested rows from the backing file → matrix.
    x[idx, , drop = FALSE]
  } else if (inherits(x, "enfold_arrow_file")) {
    # File-backed Feather/IPC: re-open as memory-mapped table, subset, materialise.
    tbl <- arrow::read_feather(x$path, as_data_frame = FALSE)
    as.data.frame(tbl[idx, ])
  } else if (inherits(x, "ArrowTabular")) {
    # In-memory Arrow Table or RecordBatch → data frame.
    as.data.frame(x[idx, ])
  } else {
    # Matrix or data frame: standard 2D subsetting.
    x[idx, , drop = FALSE]
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
