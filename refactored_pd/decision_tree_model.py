
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat

# -----------------------------------------------GLM BINOMIAL----------------------------------------------------


def decision_tree_fit(x_train, y_train, randomstate, ccpalpha):
    
    ''' GLM Binomial fit '''


    clf_dt = DecisionTreeClassifier(random_state=randomstate, ccp_alpha=ccpalpha)
    clf_dt = clf_dt.fit(x_train, y_train)
        
    return clf_dt

# ----------------------------------------------Testing------------------------------------------------------------

if __name__ == "__main__":


    file_path = "./KGB.sas7bdat"
    data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)    
    miss = ImputationCat(df_cat=df_loan_categorical)
    imputer_cat = miss.simple_imputer_mode()

    custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}


    instance = OneHotEncoding(custom_rcParams, imputer_cat, "machine")
    
    x_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]
    y_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[2]
    y_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]
    x_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]

    a = decision_tree_fit(x_train, y_train, randomstate=42, ccpalpha=0)
    
    with open('decision_tree.pkl','wb') as file:
         pickle.dump(a, file)