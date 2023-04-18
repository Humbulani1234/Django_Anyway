***DESIGN PATTERN AND PROGRAMMING PARADIGM***

    1. Design Pattern - the Python file have a corresponding Jupyter Notebook file, so an interactive jupyter
                        notebook code layout has been adopted
    
    2. Software Paradigm - a Functional paradigm has been adopted
    
    3. Work in progress - the code has been developed merely to show case coding abilities, nor is it designed
                          according to software beased parctices.

YOU MUST BE REMOVED!!!!!
***PROJECT DESCRIPTION AND ITS MOTIVATION***

    1. EAD Definition:
    
          1.1 Outstanding loan balance at the time of default
  
    1. We present a model building framework, functions (for ease of unit testing) and algorithms in python for
       various problems for the development of credit risk models - Exposure at Default (EAD)
    
    3. Mathematical object of interest: The (p+1)-dimensional joint Conditional Probability distribution where the
       vector of independent variables has p-dimension and dependent variable has 1-dimension.

    4. We proceed by modelling the mean of the distribution under the assumption that the observed responses
       are draw from a particular statistical model - in this case Least Squares
      
***Resources***
 
    1. Resources - https://doi.org/10.1177%2F1536867X1301300407, Little Test of Missing Completely at Random
    by Cheng Li: only up to section 2.2 of the paper has been applied.
      
    2. Resources - http://www.ce.memphis.edu/7012/L17_CategoricalVariableAssociation.pdf, Point Biserial Test
       by University of Memphis

    3.  Resources - Developing Credit Risk Models by Dr. Iain Brown

    4.  Resources - Generalized Linear Models with Examples in R by Peter K Dunn and Gordon K Smith

    5.  Resources - https://www.statlect.com/, Lectures by Marc Taboga  
   
***DATA COLLECTION AND DATA DEFINITIONS***
  
    1. Data definitions and collection methods are detailed in Iain Brown book referenced above.
    
***Libraries and Settings***

    1. Libraries
    2. Settings
  
***DATA DOWNLOAD AND CLEANING***
 
     1. Data Download
     2. Data Cleaning
 
***ANALYSIS OF MISSING VALUES AND IMPUTATION***
 
      1. Missing values count per column
      2. Visualization of missing values patterns

 **Missing values analysis:**

      1. Little Test hypothesis - MCAR test (missing completely at random)
      2. MCAR vs (MNAR, MAR) - Adhoc tests
      3. Imputation via python Simple imputer and KNN

            1.1 Little's Test for MCAR -- Hypothesis testing

                  1.1.1 Resources: Little's Test of Missing Completely at Random by Cheng Li, Northwestern University,
                        Evanston, IL

                  1.1.2 Algorithm presented below

                              1.1.2.1 Inputs
                              1.1.2.2 The Test

             2.1. MCAR adhoc tests vs MNAR, MAR

                   2.1.1 Plots
                   2.1.2 Tests

             3.1 Simple Imputation -- through Python API's

                   3.1.1 KNN Imputation

**concatenate** the imputed dataframes(categorical/float) into one **total dataframe** for further analysis

***EXPLORATORY DATA ANALYSIS***

      1. Investigating Distribution of LGD

      2. Hypothesis Tests and Visual Plots:

          2.1  Pearson correlation - numeric variables
          2.2. Chi Square test - categorical variables
          2.3. Point Bisserial test - categorical and numeric variables
          2.4. Correlation and variance inflation factor (VIF)
  
***Point Biserial Test for Binary vs Numerical***

      1. Plot
      2. The Test

***Categorical vs Categorical Chi-square test and plots***

      1. Plot
      2. The Test

***Pearson correlation test for Numerical variables***

      1. Plot
      2. The Test

***Multicollinearity investigation***

      1. VIF Test
  
***DATA CLUSTERING AND DIMENSION REDUCTION***

      1. K_Prototype Clustering
      2. K_Prototype Plots

***TRAIN AND TESTING SAMPLES***
    
      1. One Hot Encoding - Statistical methods and Machine learning
      2. Train and Testing sample split

          2.1 Sample partitioning into train and testing sets

              2.1.1 Defining Independent and Dependent variables - Statistics
              2.1.2 Sample imbalance investigation
              2.1.3 Training and Testing samples

***EAD MODELING:***

      1. Least Squares Regression
      
            1.1. Linear Least Sqaures Regression via using Python built-in API
            1.2 Model Fit Assessment
            
***MODEL ASSESSMENT***

      1. Diagonostics Tests

            1.1 Hypothesis Tests and Visual Plots:

                    1.1.1 Quantile Continuos Residuals
                    1.2.2 Breush Pagan Test - Heteroskedasticity of Variance
                    1.3.3 Normal Residuals Test
                    1.4.4 Durbin Watson Test - Test for Errors Serial Correlation
                    1.5.5 Leverage Studentized Quantile Residuals
                    1.6.6 Partial Residuals Plots
                    1.7.7 Cooks Distance Quantile Residuals

***DIAGNOSTICS REMEDIES***

        1. Conduct Hypothesis tests to drop insignificant variables:

              1.1 Wald test
              1.2. Score test
              1.3. Likelihood ratio test

        2. Conduct Chi square test for model fit using Deviance residuals

        3. Investigate Diagnostics checks and respond accordingly per each test by adding/dropping required varaibles

***FINAL STATISTICAL MODEL - after diagnostics remedies***

        1. Refit the Least Square after incoporating above changes

***MODEL DEPLOYMENT***
  
        1. Save with pickle

        2. Options:

                2.1. Cloud deployment
                2.2. Build a GUI for the model
