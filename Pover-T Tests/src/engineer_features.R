EngineerFeatures <- function(data, country.name){
  #' @title 
  #' Engineer new features for modeling
  #' 
  #' @description 
  #' Create new features from those provided in raw/preprocessed data.
  #' All features are created prior to model training, including aggregates 
  #'   across the full training dataset. Strictly speaking, this leads to data 
  #'   leakage; the proper approach would be to calculate aggregated features 
  #'   for each new training dataset generated by each fold of CV. However, the
  #'   proper implementation would lead to a dramatic increase in runtime 
  #'   (i.e., ten function calls per 10-fold CV run instead of one), and the 
  #'   leaderboard results generally tracked closely enough with local estimates 
  #'   of generalization error that this seemed the more practical approach.
  #'
  #' @param data 
  #' list of list of data.table: list of data.tables for a single country
  #'   Top list level names: "hh" and "ind" (household and individual datasets);
  #'   bottom list level names: "train" and "test"
  #' @param country.name 
  #' character: country name associated with data/model
  #'
  #' @return
  #' list of data.table: list of data.tables for a single country.
  #'   List should contain two named elements, "train" and "test".
  #'   All data returned is at a household-level; new features derived from
  #'     individual-level data have been aggregated up to the household level.
  
  log_info(glue("Engineering new features for Country {country.name}..."))
  start.time <- proc.time()
  
  feature.data <- copy(data)
  
  # Record original order of records in test data
  order.IDs <- data.table(id = feature.data$hh$test$id)
  
  # Estimate household size using # of responses in individual data set
  feature.data$ind <- lapply(feature.data$ind, 
                             function(x) {
                               x[, hh_size := .N, by = "id"]
                             }
  )
  
  # Join estimated household sizes to household-level data
  ind.temp <- unique(feature.data$ind$train[,.(hh_size, id)])
  feature.data$hh$train <- feature.data$hh$train[ind.temp, on = "id"]
  
  ind.temp <- unique(feature.data$ind$test[,.(hh_size, id)])
  feature.data$hh$test <- feature.data$hh$test[ind.temp, on = "id"]
  
  AggregateIndividualData <- function(hh.data, ind.data){
    #' @title 
    #' Aggregate individual data up to household level
    #' 
    #' @description 
    #' Create new household-level features by aggregating individual-level data.
    #' Performs common aggregations (median, min, max) across all valid columns.
    #'
    #' @param hh.data 
    #' List of data.table: list of data.tables containing household data;
    #'   list names should be "train" and "test".
    #' @param ind.data 
    #' list of data.table: list containing train and test data; list names 
    #'   should be "train" and "test"
    #'
    #' @return
    #' list of data.table: list of data.tables containing features derived
    #'   from both household and individual data (all at a household level).
    #'   List should contain two named elements, "train" and "test".
    
    hh <- copy(hh.data)
    ind <- copy(ind.data)
      
    # Prepend dataset identifiers to column names
    colnames(hh) <- paste0("hh.",colnames(hh))
    colnames(ind) <- paste0("ind.",colnames(ind))
    
    # Aggregate individual data to household level using common aggregations
    agg.list <- c("median", "max", "min")
    new.features <- lapply(agg.list, 
                           GenerateNewAggregate, 
                           data = ind, 
                           group.by = "ind.id")
    names(new.features) <- agg.list
    
    new.features$prop.na <- ind[, 
                                lapply(.SD, function(x) sum(is.na(x))/.N),
                                by = "ind.id"]
    colnames(new.features$prop.na) <- 
      paste0(colnames(new.features$prop.na), ".na")
    
    # Join aggregated individual data to household-level data
    combined <- 
      new.features$prop.na[
        new.features$min[
          new.features$max[
            new.features$median[
              hh, 
              on = c("ind.id.med" = "hh.id")],
            on = c("ind.id.max" = "ind.id.med")],
          on = c("ind.id.min" = "ind.id.max")],
        on = c("ind.id.na" = "ind.id.min")]
    
    setnames(combined, "ind.id.na", "id")
    setnames(combined, "hh.hh_size", "hh_size")
    
    return(combined)
    }

  # Convert all integer columns in individual data to double before creating
  #   new individual-derived features (necessary for median calculation)
  double.cols <- colnames(feature.data$ind$train)
  double.cols <- double.cols[!(colnames(feature.data$ind$train) == "poor")]
  
  feature.data$ind$train[, (double.cols) := lapply(.SD, as.double), 
                         .SDcols = double.cols]
  feature.data$ind$test[, (double.cols) := lapply(.SD, as.double), 
                        .SDcols = double.cols]
  
  # Join newly created features to original preprocessed dataset
  feature.data <- list(train = AggregateIndividualData(feature.data$hh$train, 
                                       feature.data$ind$train[,-c("poor")]),
                      test = AggregateIndividualData(feature.data$hh$test, 
                                       feature.data$ind$test))
  
  # Match order of records in test data to original order for submission script
  feature.data$test <- feature.data$test[order.IDs, on = "id"]
  
  duration <- round((proc.time() - start.time)[['elapsed']], 2)
  log_info(glue("New features successfully engineered in {duration} seconds."))
  
  return(feature.data)
}
