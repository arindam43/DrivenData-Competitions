GeneratePredictionFile <- function(model, data, predictions.path){
  #' @title 
  #' Generate predictions and save to file
  #' 
  #' @description 
  #' Generate predictions using best model trained on training data and save 
  #'   to .csv file locally for upload to leaderboard
  #'
  #' @param model 
  #' train: trained model returned from caret train() function
  #' @param data 
  #' list of data.table: list containing train and test data; list names should 
  #'   be "train" and "test" 
  #' @param predictions.path 
  #' character: directory in which to save predictions
  #' 
  #' @return
  #' data.table: data.table containing predictions on entire test set
  #'   (across all three countries)
  
  log_info("Generating predictions on test data and saving to local file...")
  start.time <- proc.time()
  
  predictions.list <- mapply(MakePredictions, 
                            model, 
                            data, 
                            toupper(names(model)), 
                            SIMPLIFY = FALSE)
   
  prediction.submission <- rbindlist(predictions.list)
  
  current.time <- format(Sys.time(), format = "%Y-%m-%d %H.%M.%S")
  fwrite(prediction.submission, 
         glue("{predictions.path}/predictions {current.time}.csv"))
  
  duration <- round(proc.time() - start.time, 2)
  log_info("Predictions generated in {duration[['elapsed']]} seconds.")
  
  return(prediction.submission)
}

MakePredictions <- function(train, data, country.name){
  #' @title 
  #' Predict outcomes for test data
  #'
  #' @description 
  #' Make predictions for test data (one country) using best trained model
  #' 
  #' @param train 
  #' train: trained model returned from caret train() function
  #' @param data 
  #' list of data.table: list containing train and test data; list names should 
  #'   be "train" and "test" 
  #' @param country.name 
  #' character: country name associated with data/model
  #'
  #' @return
  #' data.table: data.table containing predictions on test set
  
  output <- copy(data$test)
  
  output[, poor := predict(train, output, type = "prob")$Poor]
  output[, country := country.name]
  output <- output[,.(id, country, poor)]
  
  output
}
