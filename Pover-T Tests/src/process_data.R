PreProcess <- function(raw.data, country.name){
  #' @title 
  #' Preprocess raw input data to model
  #'
  #' @description 
  #' Performs several preprocessing steps on raw input data to model:
  #' 1. Removes zero-variances columns
  #' 2. Converts binary response to factor data type (necessary for xgboost)
  #' 3. Impact encode categorical variables (xgboost requires all numeric data)
  #' 
  #' @param raw.data 
  #' list of list of data.table: list of data.tables for a single country
  #'   Top list level names: "hh" and "ind" (household and individual datasets);
  #'   bottom list level names: "train" and "test"
  #' @param country.name 
  #' character: country name associated with data/model
  #'
  #' @return
  #' list of list of data.table: same structure as raw.data, after preprocessing
  
  
  log_info(glue("Preprocessing raw data for Country {country.name}..."))
  start.time <- proc.time()
  
  # Country-level data
  preproc.data <- copy(raw.data)
  
  # Note order of IDs, test rows must be in same order at the end for prediction
  order.hh.IDs <- data.table(id = preproc.data$hh$test$id)
  order.ind.IDs <- data.table(id = preproc.data$ind$test$id)
  
  # Remove zero-variance columns
  lapply(preproc.data, RemoveZeroVarCols)
  
  # Remove columns with large proportion missing values
  # Verified through EDA that relatively few columns are dropped this way
  lapply(preproc.data, RemoveHighMissingDataCols, cutoff = 0.8)
  
  # Convert binary response column to factors
  lapply(preproc.data,
         function(x){
           x$train[, poor := ifelse(poor == TRUE, "Poor", "Not_Poor")]
           x$train[, poor := relevel(as.factor(poor), "Poor")]
         }
  )

  # Reorder the test data rows (necessary for submission script)
  preproc.data$hh$test <- preproc.data$hh$test[order.hh.IDs, on = "id"]

  # Re-encode categorical data as numeric using impact encoding
  preproc.data <- lapply(preproc.data, 
                         ImpactEncode, 
                         response = "poor",
                         pos.class = "Poor"
                         )

  # Reorder the columns (necessary for XGBoost)
  lapply(preproc.data, AlignTrainAndTestColumns)
  
  duration <- round((proc.time() - start.time)[['elapsed']], 2)
  log_info(glue("Raw data successfully preprocessed in {duration} seconds."))
  
  return(preproc.data)
}  

PostProcess <- function(feature.data, country.name) {
  #' @title 
  #' Postprocess raw input data to model
  #'
  #' @description 
  #' Postprocessing of input data to model (after feature engineering).
  #' Removes columns that are redundant or inappropriate for modeling
  #'  (e.g., identifier columns)
  #' 
  #' @param feature.data 
  #' list of data.table: list of data.tables for a single country.
  #'  List should contain two named elements, "train" and "test.
  #' @param country.name 
  #' character: country name associated with data/model
  #'
  #' @return
  #' list of data.table: same structure as feature.data, after postprocessing
  
  log_info(glue("Postprocessing raw data for Country {country.name}..."))
  start.time <- proc.time()
  
  # Country-level data
  model.data <- copy(feature.data)
  
  DropColumnsRegex <- function(df) {
    drop.cols <- colnames(df)[grep(".iid.|.hh_size", colnames(df))]
    df[, (drop.cols) := NULL]
  }
  
  model.data <- lapply(feature.data, DropColumnsRegex)
  
  duration <- round((proc.time() - start.time)[['elapsed']], 2)
  log_info(glue("Raw data successfully postprocessed in {duration} seconds."))
  
  return(model.data)
}
