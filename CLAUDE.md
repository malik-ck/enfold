# CLAUDE.md — enfold development guide

## Status
This package is mid-refactor. Assume nothing is final without reading the code.
- Several exported functions may still return NULL or have incomplete implementations
- Test coverage is growing but not exhaustive

## For AIs
Read every file in `R/` before making changes. Suggested reading order:
1. `enfold_1_initialize.R` → `enfold_2_add_learners.R` → `enfold_3_cv.R` → `enfold_4_train.R` → `enfold_5_evaluate.R`
2. `constructors_learners.R`, `constructors_grids.R`, `constructors_pipelines.R`, `constructors_hyperparameters.R`
3. `cv_contributing.R`
4. `templates_*.R`, `define_losses.R`, `helpers.R`

Place user-facing functions at the top of files, helpers below them, helpers-of-helpers further down. Do not define functions inside functions unless they are very brief anonymous functions.

## Core design principles
- **S3 throughout.** No R6, no reference classes.
- **Type-agnostic.** Do not add `is.numeric`, `is.vector`, or similar checks on learner/metalearner inputs or outputs unless a specific function genuinely requires a type.
- **Closures over global state.** Learner factories use `bquote` and `list2env` to create minimal closure environments. Never close over large data objects inadvertently.

## Class overview
No class inherits from another except where methods are genuinely shared. `enfold_pipeline` and `enfold_grid` do NOT inherit `enfold_learner` — they are duck-typed to respond to `fit()` and `predict()` generics.

| Class | Created by | Notes |
|---|---|---|
| `enfold_task` / `enfold_task_fitted` | `initialize_enfold(x, y)` / `fit()` | Main workflow object; data stored in locked `x_env`/`y_env` |
| `enfold_cv` | `create_cv_folds()` or `add_cv_folds()` | `performance_sets` = outer folds; `build_sets` = inner folds (one per outer fold) |
| `enfold_fold` / `enfold_fold_list` | `new_fold()` / `new_fold_list()` | Internal fold primitives |
| `enfold_learner` / `enfold_learner_fitted` | `make_learner_factory()(...)` / `fit()` | Closure-based; hyperparams in env |
| `enfold_pipeline` / `enfold_pipeline_fitted` | `make_pipeline(...)` / `fit()` | DAG of stages; branching via multiple nodes per stage |
| `enfold_grid` | `make_grid_factory()(...)` | Search engine baked into closure |
| `enfold_list` | `make_learner_factory(..., expect_list=TRUE)(...)` | Thin wrapper indicating predict returns a named list |
| `enfold_hyperparameters` | `specify_hyperparameters(...)` | Discrete values or `make_range()`; supports `forbid()`; log-scale via `sample_log_uniform()` |

## Workflow
```
initialize_enfold(x, y)
  |> add_learners(lrn_1, pipeline_1, grid_1, ...)
  |> add_metalearners(mtl_1, mtl_2, ...)
  |> add_cv_folds(inner_cv = 5, outer_cv = 10)
  |> fit()        # returns enfold_task_fitted
  |> predict(type = "cv")       # out-of-fold preds with indices attr
  |> predict(type = "ensemble") # metalearner preds
  |> risk()       # mean loss per metalearner
```

## Key contracts

### enfold_fold
- Always stores `validation_set` as integer indices into the full data
- Stores `training_set` only if non-complementary; otherwise derives it as `setdiff(seq_len(n), validation_set)` via `training_set()` accessor
- `exclude(fold, indices)` returns a NEW fold — never mutates in place
- Downstream code ALWAYS uses `training_set(fold)` and `validation_set(fold)` accessors

### enfold_learner
- Constructors for `enfold_learner` objects created via `make_learner_factory(fit, preds, ...)`
- Calling constructor then creates a learner instance
- `fit(learner, x, y)` returns `enfold_learner_fitted`
- `predict(fitted, newdata)` returns raw output — no type constraints
- Closure environment contains ONLY `fit`, `preds`, and declared hyperparameters
- `get_params(learner)`, `inspect(learner)` are the inspection interface

### enfold_pipeline
- `fit.enfold_pipeline(object, x, y)` traverses DAG per path, returns `enfold_pipeline_fitted`
- `predict.enfold_pipeline_fitted(object, newdata)` replays DAG, returns named list (one entry per terminal path)
- Path names: node names concatenated with "/"; passthrough nodes shown as "."
- Per-path errors are isolated — a failing path does not fail the whole pipeline

### cv_fit / fit_predict_folds (cv_contributing.R)
- `cv_fit()` is the inner-CV workhorse: dispatches on learner class, returns named prediction list
- A learner that fails on ANY fold is excluded entirely — partial predictions never reach metalearners
- Failed learner names stored as `attr(result, "failed_learners")`
- Grids return a `resolved_learner` attribute indicating the winning configuration
- List-valued outputs are spliced for `enfold_pipeline`, `enfold_grid`, and `enfold_list` classes
- `combine_preds(chunks)`: `rbind` for matrix/df, `c` for vectors

### predict.enfold_task_fitted
- `type = "cv"`: out-of-fold predictions via outer folds; result carries `attr(result, "indices")`
- `type = "ensemble"`: predictions from fitted metalearners
- `make_preds_list()` splices pipeline/grid/list outputs automatically

### Incremental fitting
- `fitted_learner_names` / `fitted_metalearner_names` track what was fit
- Adding only metalearners → fast metadata-only refit (skips learner training)
- Adding learners → full refit

## Parallelism
- `future.apply::future_lapply` used in `fit_ensemble()` and inner `build_ensembles()`
- Always pass `future.globals` explicitly — do not rely on automatic detection
- `future.seed = TRUE` always
- Users set `future::plan()` before calling `fit()`; recommend `future::multicore` on Linux/macOS

## Memory discipline
- `rm()` intermediate objects (fitted learners, fold subsets) immediately after use inside loops
- Never store full data in fold objects — only indices
- `subset_y(y, idx)` handles both vector and matrix y: `if (is.null(dim(y))) y[idx] else y[idx, , drop = FALSE]`
- `combine_y(chunks)`: `do.call(c, chunks)` for vectors; `do.call(rbind, chunks)` for matrices

## Testing
- Framework: `testthat`; all test files start with `set.seed()` and `future::plan("sequential")`
- Test structure: class checks first (`expect_s3_class`), then slot contents, then behavior
- Use custom factory-built test learners (e.g. a "bomb" learner that always errors) for graceful-failure tests
- Test levels: unit for fold primitives and hyperparameters; integration for the full workflow; complex scenarios for pipelines + grids combined
- Known gaps: pipeline error isolation in `fit.enfold_pipeline`; graceful failure consistency in `fit_ensemble` for plain learners

## R-specific practices
- Generics: define with `UseMethod()`; register every method in NAMESPACE via `devtools::document()`
- Isolated closure envs: `new.env(parent = emptyenv())`; use `lockEnvironment()` to signal immutability
- Quoting: `alist()` for formal lists with missing defaults; `bquote()` for splicing values into expressions
- Validated string args: `match.arg()` at the top of the function
- R is copy-on-modify — never assume in-place mutation; always return modified objects
- Pre-allocate lists in loops: `vector("list", n)` rather than growing with `[[]]`

## What NOT to do
- Do not add type checks on learner prediction outputs
- Do not use `<<-`; use return values or explicit `assign(nm, val, envir = e)`
- Do not call `as.vector()` on data frame columns; use `df[[col]]`
- Do not use `pairlist()` with `quote(expr=)` for function formals; use `alist()`
- Do not use `body()<-` for anything other than the learner factory pattern already established
- Do not inherit classes speculatively — only add a class if at least one method dispatches on it differently
