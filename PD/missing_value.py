import pandas as pd
import missingno as msn
import matplotlib.pyplot as plt
import matplotlib

import ED
#from ED import df_loan_categorical, df_loan_float

def Missing_values_analysis(dataframe_1, dataframe_2):
    
    missing_count_column_cat = dataframe_1.isnull().sum()
    missing_head_cat = dataframe_1.isnull().head() 

    missing_count_column_float = dataframe_2.isnull().sum()
    missing_head_float = dataframe_2.isnull().head()
    
    return missing_count_column_cat, missing_count_column_float, missing_head_cat, missing_head_float

missing_count_column_cat, missing_count_column_float, missing_head_cat, missing_head_float\
 = Missing_values_analysis(ED.df_loan_categorical, ED.df_loan_float)

#print(missing_count_column_cat) 

#========================================
#Visualization of missing values patterns
#========================================

def Missing_values_patterns(dataframe1, dataframe2):
    
    matrix_cat = msn.matrix(dataframe1)
    bar_cat = msn.bar(dataframe1)

    matrix_float = msn.matrix(dataframe2)
    bar_float = msn.bar(dataframe2)
    
    return matrix_cat, matrix_float, bar_cat, bar_float

matrix_cat, matrix_float, bar_cat, bar_float = Missing_values_patterns(ED.df_loan_categorical, ED.df_loan_float)
#print(dir(matrix_cat))
#print(type(matrix_cat))
#print(matrix_cat.plot())

#plt.show() # One has to invest in understanding matplotlib