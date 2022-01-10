ReadInputData <- function(path) {
  #' @title 
  #' Read input data
  #'
  #' @description 
  #' Reads relevant input data files for analysis into a list of data.tables.
  #' 
  #' The following .csv files are assumed to be present in the directory 
  #' specified by the 'path' argument:
  #' 
  #' country_a_hh_test
  #' country_a_hh_train
  #' country_a_ind_test
  #' country_a_ind_train
  #' 
  #' The equivalent files for countries B and C should be present as well
  #'   (12 input data files in total).
  #' Note that these file names are different than those of the raw source files 
  #'   downloaded from DrivenData. 
  #' The file names must be converted to the expected ones manually; 
  #'   the correspondence between original file names and those expected by
  #'   this script should be self-explanatory.
  #' 
  #' @param 
  #' path: directory containing raw data files
  #'
  #' @return
  #' List of list of lists (all named) containing input data in data.tables.
  #' Top level of list names corresponds to country ("a", "b", or "c");
  #'   middle level corresponds to individual/household ("ind", "hh"),
  #'   bottom level corresponds to train/test ("train", "test").
  #'
  #' @examples
  #' x = ReadInputData("C:/poverttests/data/")
  #' 
  #' x$a$ind$train returns a data.table containing the individual-level
  #'  training survey data responses for country a,
  #'  x$b$hh$test returns a data.table containing the household-level
  #'  test survey data responses for country b, and so on.
  
  log_info("Reading in raw data...")
  start.time <- proc.time()
  
  countries <- c('a', 'b', 'c')
  
  GetCountryData <- function(country.name) {
    ind.train.path <- glue("{path}/country_{country.name}_ind_train.csv")
    ind.test.path <- glue("{path}/country_{country.name}_ind_test.csv")
    hh.train.path <- glue("{path}/country_{country.name}_hh_train.csv")
    hh.test.path <- glue("{path}/country_{country.name}_hh_test.csv")
    
    country.data <- 
      list(ind = list(train = fread(ind.train.path, stringsAsFactors = TRUE),
                      test = fread(ind.test.path, stringsAsFactors = TRUE)),
           hh = list(train = fread(hh.train.path, stringsAsFactors = TRUE),
                     test = fread(hh.test.path, stringsAsFactors = TRUE))
      )
    
    return(country.data)
  }
  
  input.data <- lapply(countries, GetCountryData)
  names(input.data) <- countries
  
  duration <- round((proc.time() - start.time)[['elapsed']], 2)
  log_info(glue("Raw data successfully imported in {duration} seconds."))
  
  return(input.data)
}
