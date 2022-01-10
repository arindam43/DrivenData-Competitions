# Primary functions
VisualizeInsights <- function(xgb.model, country.name){
  
  #' @title 
  #' Visualize performance and insights
  #' 
  #' @description 
  #' Visualizes results of model training.
  #' Visualizations include model performance and estimated effects of 
  #'   variables on predictions (PD and ALE plots).
  #'
  #' @param xgb.model 
  #' train: trained model returned from caret train() function
  #' @param country.name 
  #' character: country name associated with data/model
  #'
  #' @return
  #' list: first element is data.table containing variable importance values,
  #'   while second element is list containing calibration results
  #'   (output of caret's calibration() function)
  
  log_info("Visualizing model performance and insights...")
  start.time <- proc.time()
  
  calibration.results <- ModelSummary(xgb.model, country.name)
  var.imps <- PlotVarImps(xgb.model, country.name, top.n = 20) 
  
  duration <- round(proc.time() - start.time, 2)
  log_info("Visualization complete in {duration[['elapsed']]} seconds.")
  
  return(list(var.imps = var.imps,
              calibration.results = calibration.results)
         )
  }
  