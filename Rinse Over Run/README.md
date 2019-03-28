# Rinse Over Run  
All of the scripts I developed for DrivenData's Rinse Over Run competition can be found here. A detailed guide for reproducing my results is provided below. All directions provided are for Windows 10.  
  
After completing the steps below, simply run main.py to generate the test set predictions. They will be saved to the 'predictions' directory as 'Test Predictions YYYY-MM-DD HH.MM.SS.csv'.  
  
## Detailed Installation Guide  
  
This detailed guide assumes that you are using Windows 10, Anaconda, and PyCharm.  
  
1. Create a new environment in anaconda prompt:  
    *conda create -n rinseoverrundemo python=3.6.6*  
2. Activate the newly created environment:  
    *conda activate rinseoverrundemo*  
3. Install the required packages listed in the **Install Requirements** section above using *conda install*.  
4. Clone the LightGBM git repo (can be done with git bash) into the directory that contains the environment (i.e. C:/.../rinseoverrundemo):  
    *git clone --recursive https://github.com/Microsoft/LightGBM.git*  
5. Make the appropriate changes to the source code outlined in the **LightGBM Installation Instructions** section below.  
6. Run the following two commands **in anaconda prompt with the rinseoverrundemo environment active**:  
    *cd LightGBM/python-package*  
    *python setup.py install*  
   This will ensure that the modified version of LightGBM is installed to the correct environment, 'rinseoverrundemo'!  
7. After all packages have been installed, create a new project in PyCharm and set up the directory structure according to the **Directory and File Structure** section below.  
8. In PyCharm, go to File -> Settings -> Project: XYZ -> Project Interpreter and set the path to the correct version of python. When you are finished, the Project Interpreter dropdown menu should be populated with something that looks like this: 'Python 3.6 C:/.../rinseoverrundemo\python.exe'. The list of packages displayed should also correspond to those installed in the 'rinseoverrundemo' environment created through conda.  
9. Run main.py to generate the test set predictions.  
  
## Install Requirements  
Below are the basic requirements; a full list of packages, including dependencies, can be found in requirements.txt.  
  
| Package  | Version |  
| --- | --- |  
| python | 3.6.6 |  
| pandas | 0.23.4 |  
| numpy | 1.15.4 |  
| matplotlib | 3.0.2 |  
| shap | 0.27.0 |  
| lightgbm** | 2.2.3 |  
  
** Must be installed from source using git clone, since changes are made to the source code before installation. Cannot be installed using 'conda install' or 'pip install'!  
  
## LightGBM Installation Instructions  
1. Make sure you have CMake installed, version 3.8 or higher.  
2. Clone the LightGBM git repo using this command: *git clone --recursive https://github.com/Microsoft/LightGBM.git*  
3. Make the following changes to the source code (line numbers are as of 3-28-2017, version 2.2.3):  
    - /src/objective/regression_objective.hpp: change line 640 from  
        *label_weight_[i] = 1.0f / std::max(1.0f, std::fabs(label_[i]));*  
        to  
        *label_weight_[i] = 1.0f / std::max(290000.0f, std::fabs(label_[i]));*  
        
      In the same vein, change line 645 from  
        *label_weight_[i] = 1.0f / std::max(1.0f, std::fabs(label_[i])) * weights_[i];*  
        to  
        *label_weight_[i] = 1.0f / std::max(290000.0f, std::fabs(label_[i])) * weights_[i];*  
    - /src/metric/regression_metric.hpp: change line 243 from  
        *return std::fabs((label - score)) / std::max(1.0f, std::fabs(label));*  
        to  
        *return std::fabs((label - score)) / std::max(290000.0f, std::fabs(label));*  
4. Install the package with the modified source code using the following commands:  
    *cd LightGBM/python-package*  
    *python setup.py install*  
  
## Directory and File Structure  
  
Directories are indicated in bold.  
  
**Rinse-Over-Run**  
  * **data** 
    * train_values.pkl  
    * test_values.pkl  
    * recipe_metadata.csv  
    * train_labels.csv  
  * **predictions**  
    * *initially empty*  
  * **src**  
    * build_models.py  
    * engineer_features.py  
    * ingest_data.py  
    * main.py  
    * make_predictions.py  
    * visualize_insights.py  
  
Additional notes:  
  
- All of the .py scripts assume that os.getcwd() will return the top level folder indicated in the structure above (i.e. C:/..../Rinse-Over-Run). Please add os.chdir() statements to the beginning of each .py script to change the working directory if this is not the case.  
- The raw train_values.csv and test_values.csv files should have the following changes made to them before running main.py:  
        1. Remove all rows with phase = "final_rinse". This step should only affect train_values.csv, as test_values.csv should not have any final rinse data in it. This is easily accomplished by reading in the data using *pd.read_csv()* and subsetting the dataframe using pandas.  
        2. Change the file types from .csv to .pkl. I did this by taking the subsetted dataframes from step 1 and writing them back out to my local directory using *pd.to_pickle()*. This was primarily done for speed when reading the files in to Python, as they are quite large.  
