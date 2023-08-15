

import pandas as pd
import numpy as np
import sys
import pickle
import types
import io
import base64
import statsmodels.api as sm

import pd_download
from class_traintest import OneHotEncoding
from class_base import Base
from class_missing_values import ImputationCat
import class_diagnostics
from class_modelperf import ModelPerfomance

#--------------------------------------------------------------------Data---------------------------------------------------

with open('static/glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

file_path = "static/KGB.sas7bdat"
data_types, df_loan_categorical, df_loan_float = pd_download.data_cleaning(file_path)    
miss = ImputationCat(df_loan_categorical)
imputer_cat = miss.simple_imputer_mode()

custom_rcParams = {"figure.figsize": (15, 10), "axes.labelsize": 12}

instance = OneHotEncoding(custom_rcParams, imputer_cat, "statistics")
x_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]
x_test = sm.add_constant(x_test.values)
y_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]
threshold = 0.47
x_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]

b = class_diagnostics.ResidualsPlot(custom_rcParams, x_test, y_test, threshold)
c = ModelPerfomance(custom_rcParams, x_test, y_test, threshold)

