#' @importFrom stats runif sd
#' @importFrom rlang %||%
NULL

#' Simulated Cardiovascular Disease Risk Score Data
#'
#' Simulated medical records data with a continuous cardiovascular disease risk
#' score and a number of predictors.
#'
#' @format ## `enfold_demo`
#' A data frame with 1,274 rows and 9 columns:
#' \describe{
#'   \item{age}{Numeric. Age of the individual.}
#'   \item{sex}{Male or female.}
#'   \item{smoking}{Binary. Whether an individual is currently a smoker.}
#'   \item{bmi}{Numeric. BMI of the individual}
#'   \item{physical_activity}{Numeric integer. Physical activity score.}
#'   \item{systolic_bp}{Numeric. Systolic blood pressure.}
#'   \item{ldl_cholesterol}{Numeric. LDL cholesterol.}
#'   \item{coronary_heart_disease}{Binary. Whether an individual has coronary heart disease.}
#'   \item{hba1c}{Numeric. Glycosylated hemoglobin.}
#' }
#' @source Data simulation.
"enfold_demo"
