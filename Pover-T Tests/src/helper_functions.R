# General utility functions

ImportPackages <- function(...) {
 
  #' @title
  #' Import necessary packages
  #'
  #' @description
  #' Function for importing necessary packages.
  #' Installs any missing necessary packages as well.
  #'
  #' @param
  #' ...: Comma separated list of packages to import.
  #'
  #' @return
  #' N/A
  #'
  #' @examples
  #' ImportPackages('data.table', 'caret', 'ggplot2')
 
  libs.to.import <- unlist(list(...))
  lib.present <- unlist(lapply(libs.to.import, require, character.only = TRUE))
  missing.libs <- libs.to.import[lib.present == FALSE]
 
  if(length(missing.libs) > 0){
    install.packages(missing.libs)
    lapply(missing.libs, require, character.only = TRUE)
  }
}

# Pre- and post-processing helper functions

RemoveZeroVarCols <- function(data) {
  # Helper functions
  #' @title
  #' Remove zero-variance columns
  #'
  #' @description
  #' Removes columns with no variance from input data to models.
  #' Uses only training data to determine which columns are considered
  #'     "zero-variance" to avoid data leakage, then removes said columns
  #'     from both training and test data.
  #'
  #' @param data
  #' list of data.table: two data.tables with list names "train" and "test" to
  #'     remove zero-variance columns from
  #'
  #' @return
  #' data.table: data with zero variance columns removed
  #'
  #' @examples
  #' input.data = list(train = data.table(...), test= data.table(...))
  #' RemoveZeroVarCols(input.data)
 
  unique.vals <- lapply(data$train, function(y) length(unique(y)))
  no.var.cols <- names(unlist(which(!(unique.vals > 1))))
 
  if(length(no.var.cols) > 0){
    data$train[, (no.var.cols) := NULL]
    data$test[, (no.var.cols) := NULL]
  }
}

RemoveHighMissingDataCols <- function(data, cutoff) {
  # Helper functions
  #' @title
  #' Remove columns with high proportion of missing data
  #'
  #' @description
  #' Remove columns with high proportion of missing data from data.
  #' "High proportion of missing data" is assumed to be a certain percentage
  #'   of missing values in a column or greater; the cutoff is provided
  #'   by the user as an argument.
  #' Uses only training data to determine which columns are considered
  #'   "zero-variance" to avoid data leakage, then removes said columns
  #'   from both training and test data.
  #'
  #' @param data
  #' list of data.table: two data.tables with list names "train" and "test" to
  #'   remove high missing data columns from
  #' @param cutoff  
  #' numeric: value between 0 and 1 indicating the threshold for 
  #'   "high percentage" of missing values
  #'
  #' @return
  #' data.table: data with high missing data columns removed
  #'
  #' @examples
  #' input.data = list(train = data.table(...), test= data.table(...))
  #' RemoveHighMissingDataCols(input.data)
 
  missing.data.cols <- 
    names(which(colSums(is.na(data$train)) / dim(data$train)[1] > cutoff))
  
  if(length(missing.data.cols) > 0){
    data$train[, (missing.data.cols) := NULL]
    data$test[, (missing.data.cols) := NULL]
  }
}

ImpactEncode <- function(data, 
                         response, 
                         type = 'binary_classification',
                         pos.class = NULL, 
                         exclude.cols = NULL){
  #' @title
  #' Impact encoder for categorical data
  #'
  #' @description 
  #' Converts categorical variables to numeric through impact encoding.
  #' For regression, replaces category level with mean of all response/target
  #'   values for that level.
  #' For binary classification, replaces category level with proportion
  #'   of all responses/targets values with "positive" label.
  #' 
  #' @param data 
  #' list of data.table: two data.tables with list names "train" and "test" to 
  #'   remove zero-variance columns from
  #' @param response 
  #' character: name of response column in dataset
  #' @param type:
  #' character: type of supervised learning problem;
  #'   valid values are "binary_classification" and "regression"
  #' @param pos.class:
  #' character: if binary classification problem, label of positive class.
  #' @param exclude.cols 
  #' list of character: categorical columns to exclude from impact encoding
  #'
  #' @return
  #' list of data.table: original list of data.tables after preprocessing
  
  # Extract categorical columns, excluding response and those passed in params
  catCols <- colnames(data$train)[vapply(data$train, 
                                         function (x) {!(is.numeric(x))}, 
                                         logical(1))]
  catCols <- catCols[!(catCols %in% c(response, exclude.cols))]
  
  # Impact encode categorical columns
  for (catCol in catCols){
    if (type == 'binary_classification'){ 
      temp <- data$train[, .(temp = sum(get(response) == pos.class)/.N), 
                         by = eval(catCol)]
    } else if (type == 'regression'){
      temp <- data$train[, .(temp = sum(get(response))/.N), 
                         by = eval(catCol)]
    } else {
      stop("Invalid model type.")
    }
    
    data <- lapply(data,
                   function(x){
                     x <- temp[x, on = eval(catCol)]
                     x[, eval(catCol) := NULL]
                     setnames(x, "temp", eval(catCol))
                     }
                   )
  }
  
  return(data)
}

AlignTrainAndTestColumns <- function(x){
  #' @title 
  #' Align train and test dataset columns
  #' 
  #' @description 
  #' Puts columns in the same order across train and test datasets (in-place).
  #' Any columns not present in both datasets (e.g., response) will be placed
  #'   after all overlapping columns.
  #' Necessary for XGBoost model training to work as intended.
  #'
  #' @param x 
  #' list of data.table: list containing train and test data; list names should 
  #'   be "train" and "test" 
  #'   
  #' @return
  #' N/A
  
  common.cols <- intersect(colnames(x$train), colnames(x$test))
  train.only.cols <- setdiff(colnames(x$train), colnames(x$test))
  test.only.cols <- setdiff(colnames(x$test), colnames(x$train))
  
  setcolorder(x$train, c(sort(common.cols), sort(train.only.cols)))
  setcolorder(x$test, c(sort(common.cols), sort(test.only.cols)))
}

# Feature engineering helper functions

AggInfToNA <- function(agg, x){
  #' @title
  #' Aggregate data and return NAs
  #' 
  #' @description 
  #' Performs common aggregations (e.g., median, min, max),
  #'   but returns NA instead of Inf or -Inf if all data is missing.
  #'
  #' @param agg 
  #' function: an aggregation (e.g., mean, median, etc.) with na.rm arg
  #' @param x 
  #' vector: column of data.table/dataframe to apply aggregation to
  #'   Vector type must make sense for the aggregation provided.
  #'
  #' @return
  #' vector: result of applying aggregation to supplied vector/column
  #'
  #' @examples
  #' AggInfToNA(mean, df$col1)
  #' AggInfToNA(max, df$col1)
  
  agg.result <- 
    ifelse(all(is.na(x)), 
           as.numeric(NA), # type inconsistency error without numeric coercion
           agg(x, na.rm=TRUE))
  return(agg.result)
}

GenerateNewAggregate <- function(agg, data, group.by){
  #' @title
  #' Generate new aggregated feature
  #' 
  #' @description 
  #' Generates a new feature by aggregating individual-level data up to
  #'   household-level
  #'
  #' @param agg 
  #' string: name of an R function that aggregates numeric data 
  #'   (e.g., mean, median, etc.) with na.rm arg
  #' @param data
  #' data.table: data.table containing individual-level data. 
  #' @param group.by
  #' character: column name specifying level of aggregation
  #'
  #' @return
  #' data.table: data.table containing newly calculated aggregate features.
  #'
  #' @examples
  #' GenerateNewAggregate("mean", df)
  #' GenerateNewAggregate("max", df)
  
  new.feature.data <- data[, 
                           lapply(.SD, 
                                  function(x) {
                                    AggInfToNA(match.fun(agg), x)}), 
                           by = group.by]
  
  # If specified aggregation produces NA, replace with corresponding aggregate 
  # computed from entire dataset (across all households)
  new.feature.data <- 
    new.feature.data[, lapply(.SD, function(x) {
      nafill(x, 
             type='const', 
             do.call(agg, c(x, list(na.rm = TRUE))))})]
  
  # Add three-character suffix to column name to indicate performed aggregation
  colnames(new.feature.data) <- paste0(colnames(new.feature.data), 
                                       glue(".{substring(agg, 1, 3)}"))
  
  return(new.feature.data)
}

# Modeling helper functions

BinaryClassSummary <- function(data, lev = NULL, model = NULL) {
  #' @title 
  #' Custom binary classification summary
  #' 
  #' @description 
  #' Custom set of binary classification performance evaluation metrics to pass 
  #'   to caret while training models.
  #' Refer to caret documentation for argument details:
  #' https://topepo.github.io/caret/model-training-and-tuning.html#metrics
  #' 
  #' @param data 
  #' See caret documentation linked above.
  #' @param lev 
  #' See caret documentation linked above.
  #' @param model 
  #' See caret documentation linked above.
  #'
  #' @return
  #' See caret documentation linked above.
  
  brier.val <- DescTools::BrierScore(ifelse(data$obs == "Poor", 1, 0), 
                                     data$Poor)
  
  logloss.val <- MLmetrics::LogLoss(data$Poor,
                                    ifelse(data$obs == "Poor", 1, 0))
  
  auc.val <- MLmetrics::AUC(data$Poor,
                            ifelse(data$obs == "Poor", 1, 0))
  
  sens.val <- MLmetrics::Sensitivity(data$obs,
                                     data$pred,
                                     positive = lev[[1]])
  
  spec.val <- MLmetrics::Specificity(data$obs,
                                     data$pred,
                                     positive = lev[[1]])
  
  c(LogLoss = round(logloss.val, 4),
    BrierScore = round(brier.val, 4),
    AUC = round(auc.val, 4),
    Sensitivity = round(sens.val, 4),
    Specificity = round(spec.val, 4)
  )
}

# Visualization helper functions

PlotVarImps <- function(model, country.name, top.n = 1, scale = TRUE){
  #' @title
  #' Plot variable importance
  #' 
  #' @description 
  #' Plots variable importance generated by caret-trained model (assuming model
  #'   has some built-in variable importance metrics implemented, as is the
  #'   case with XGBoost)
  #'
  #' @param model 
  #' train: trained model returned from caret train() function
  #' @param country.name 
  #' character: country name associated with data/model
  #' @param top.n 
  #' integer: how many of the most important variables to display
  #' @param scale 
  #' logical: whether or not to scale variable importance values to 0-100 scale
  #'
  #' @return
  #' data.table: data.table containing top.n variable importance values
  
  if (scale == TRUE){
    var.imps <- varImp(model)
  } else {
    var.imps <- varImp(model, scale = FALSE)
  }
  
  var.imps <- data.table(variable = rownames(var.imps$importance),
                         importance = var.imps$importance$Overall)
  
  var.imps <- var.imps[order(-importance)]
  var.imps[, variable := factor(variable, 
                                levels = var.imps$variable[
                                  order(var.imps$importance)])]
  
  cat("Variable importance in tabular form:\n\n")
  print(var.imps)
  cat("\n")
  
  cat("Plotting variable importances...\n\n")
  title1 <- "Variable Importances \n Model: "
  title2 <- glue("Top {top.n} Variable Importance for Country {country.name}
                             Model: {model$modelInfo$label}")
 
  variable.imp.plot <-
    ggplot(data = var.imps[1:top.n], aes(x = variable, y = importance)) +
    geom_bar(stat = 'identity', fill = "blue") +
    coord_flip() +
    labs(title = ifelse(top.n == nrow(var.imps),
                        paste0(title1, model$modelInfo$label),
                        title2)) +
    theme(plot.title = element_text(hjust = 0.5, size = 14))
 
  print(variable.imp.plot)
 
  # Plot ALE and PD graphs of most important variable
  PlotEffects(model, toString(var.imps$variable[1]), "hh.poor")
 
  return(var.imps)
}

ModelSummary <- function(model, country.name){
  #' @title
  #' Summary of model results
  #'
  #' @description
  #' Summarizes model results by printing following to console:
  #' 1. Best evaluation metric value
  #' 2. Confusion matrix and associated metrics (specificity, sensitivity, etc.)
  #' 3. Distribution of holdout CV predictions vs. actual (i.e., calibration)
  #' 4. Variable importance in tabular form
  #'
  #' @param model
  #' train: trained model returned from caret train() function
  #' @param country.name
  #' character: country name associated with data/model
  #'
  #' @return
  #' list: list containing calibration results (output of caret's
  #'   calibration() function)
 
  cat("#############")
  cat(glue("       MODEL SUMMARY FOR COUNTRY {country.name}     "))
  cat("#############\n\n")
 
  cat(paste0("Best ", model$metric, " across tuning grid: "))
  cat(min(model$results[[eval(model$metric)]]))
  cat("\n\n")
 
  if (model$modelType == "Classification"){
    print(confusionMatrix(data = model$pred$obs,
                          reference = model$pred$pred))
    cat("\n")
   
  }
 
  cat("Distribution of holdout CV predictions vs. observed (absolute):\n")
  print(
    table(cut(model$pred$Poor,
              breaks = seq(0, 1, 0.05),
              include.lowest = TRUE),
          model$pred$obs)
  )
  cat("\n")
 
  cat("Distribution of holdout CV predictions vs. observed (relative):\n")
  print(prop.table(
    table(cut(model$pred$Poor,
              breaks = seq(0, 1, 0.05),
              include.lowest = TRUE),
          model$pred$obs),
    margin = 1)
  )
  cat("\n")
 
  calibration.results <- calibration(pred$obs ~ pred$Poor,
                                     data = model,
                                     cuts = 10)
 
  calib.plot <- ggplot(data = calibration.results$data, aes(x = midpoint,
                                                            y = Percent)) +
    geom_line() +
    geom_point() +
    xlim(0, 100) +
    ylim(0, 100) +
    geom_abline(slope = 1, intercept = 0, colour='forestgreen') +
    ylab("Observed Event Rate") +
    xlab("Prediction (Probability Bin Midpoint)") +
    labs(title = "Calibration Plot for XGBoost") +
    theme(plot.title = element_text(hjust = 0.5, size = 14))
 
  print(calib.plot)
 
  return(calibration.results)
 
}

PlotEffects <- function(model, variable, response){
  #' @title
  #' Plot estimated variable effects
  #'
  #' @description
  #' Plots partial dependence (PD) and accumulated local effects (ALE) plots
  #'   for a single variable in a trained model.
  #'
  #' @param model
  #' train: trained model returned from caret train() function
  #' @param variable
  #' character: name of variable to plot estimated effects for
  #' @param response
  #' character: name of response for trained model
  #'
  #' @return
  #' N/A
 
  yhat <- function(X.model, newdata) as.numeric(predict(X.model, newdata))
 
  model.data <- data.table(model$trainingData)[,-c(".outcome")]
 
  ALEEffects <- ALEPlot(model.data, 
                        model, 
                        pred.fun = yhat,
                        J = grep(variable, colnames(model.data)))
 
  PDEffects <- PDPlot(model.data, 
                      model, 
                      pred.fun = yhat,
                      J = grep(variable, colnames(model.data)),
                      K = 40)
 
  ALEPlotData <- data.table(Predictor = ALEEffects$x.values,
                            ALE = ALEEffects$f.values)
  PDPlotData <- data.table(Predictor = PDEffects$x.values,
                           PD = PDEffects$f.values)
 
  ALE.Plot <- ggplot(data = ALEPlotData,
                     aes(x = Predictor, y = ALE)) +
    geom_line() +
    xlab(variable) +
    labs(title = paste0("ALE of ", variable, " on Response (", response, ")")) +
    theme(plot.title = element_text(hjust = 0.5))
 
  PD.Plot <- ggplot(data = PDPlotData,
                    aes(x = Predictor, y = PD)) +
    geom_line() +
    xlab(variable) +
    labs(title = paste0("PD of ", variable, " on Response (", response, ")")) +
    theme(plot.title = element_text(hjust = 0.5))
 
  grid.arrange(ALE.Plot, PD.Plot, nrow = 1)
}
