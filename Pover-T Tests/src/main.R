# Main function for data processing, model building, and predictions
main <- function(){
  root.dir <- getwd()
  predictions.path <- paste0(root.dir, "/predictions")
  setwd(paste0(getwd(), "/src"))
  
  source("helper_functions.R")
  source("read_input_data.R")
  source("process_data.R")
  source("engineer_features.R")
  source("build_models.R")
  source("visualize_insights.R")
  source("make_predictions.R")
  
  ImportPackages("data.table", "caret", "glue", "logger", "ggplot2", "ALEPlot", 
                 "gridExtra", "xgboost", "MLmetrics", "DescTools", "docstring")
  
  input.data <- ReadInputData(glue("{root.dir}/data"))
  
  preproc.data <- mapply(PreProcess,
                         input.data,
                         toupper(names(input.data)),
                         SIMPLIFY = FALSE)
  
  feature.data <- mapply(EngineerFeatures,
                         preproc.data,
                         toupper(names(preproc.data)),
                         SIMPLIFY = FALSE)
  
  model.data <- mapply(PostProcess,
                       feature.data, 
                       toupper(names(feature.data)),
                       SIMPLIFY = FALSE)
  
  models <- mapply(BuildModels,
                   model.data,
                   toupper(names(model.data)),
                   SIMPLIFY = FALSE)
  
  insights = mapply(VisualizeInsights,
                    models,
                    toupper(names(models)),
                    SIMPLIFY = FALSE)
  
  predictions = GeneratePredictionFile(models,
                                       model.data,
                                       predictions.path)
  
  return(list(input.data = input.data,
              preproc.data = preproc.data,
              feature.data = feature.data,
              model.data = model.data,
              models = models,
              insights = insights,
              predictions = predictions))
}

results <- main()