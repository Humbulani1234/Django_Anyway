
# ==================
# Diagonostics Tests
# ==================
# 
#       Hypothesis Tests and Visual Plots:
#     
#         1. Quantile Residuals - Residuals for Discrete GLMs
#         2. Breush Pagan Test - Heteroskedasticity of Variance
#         3. Normal Residuals Test
#         4. Durbin Watson Test - Test for Errors Serial Correlation
#         5. Leverage Studentized Quantile Residuals
#         6. Partial Residuals Plots
#         7. Cooks Distance Quantile Residuals

# ===========================================
# Residuals definition - Tool for Diagnostics
# ===========================================

# =================
# Diagnostics tests
# =================
 
#from scipy.stats import norm
import GLM_Bino
import ED
import train_test

##
# @package numpy
# @package np
#

import numpy as np

import Model_Perf
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import statsmodels.stats.diagnostic as sd
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

##
# @package scipy
#

#import scipy

#from scipy import stats
from math import *

# Quantile residuals @ 0.47 cut-off

def Quantile_Residuals(function, X_test, Y_test, X_train, Y_train, threshold):

    Quantile_Residuals = []
    
    #res = function(X_train, Y_train)[1]
    predict_probability = Model_Perf.Prediction(function, X_test, X_train, Y_train)
    for i in range(Y_test.shape[0]):

        if predict_probability[i] < threshold: 
            u1 = np.round(np.random.uniform(low=0,high=predict_probability[i]),10)
            Quantile_Residuals.append(norm.ppf(u1))
        
        else:
            u2 = np.round(np.random.uniform(low=predict_probability[i],high=1),10)
            Quantile_Residuals.append(norm.ppf(u2))
    
    Quantile_Residuals_Series = pd.Series(Quantile_Residuals)
    
    return Quantile_Residuals_Series

# d = Quantile_Residuals(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
#              ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.8)
# print(d)

# ===============
# Residuals plot
# ===============

def Plot_Residuals(function, X_test, Y_test, X_train, Y_train, threshold):
    
    Quantile_Residuals_Series = Quantile_Residuals(function, X_test, Y_test, X_train, Y_train, threshold)
    Quantile_Residuals_Series.plot()

    plt.title("Cross Validate", fontsize=15, pad=18)
    plt.xlabel("Alpha",fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize = 14)
    plt.yticks(fontsize=12)
    plt.rcParams["figure.figsize"] = (2.7,2.5)
    plt.rcParams["legend.title_fontsize"] = 7

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)  
    
    return plt.show()

#d = Plot_Residuals(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
#              ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)

# =====================================================
# Breush Pagan Test for Hetereskedasticity of variance
# =====================================================

def Breush_Pagan_Test(function, X_test, Y_test, X_train, Y_train, threshold):
    
    Quantile_Residuals_Series = Quantile_Residuals(function, X_test, Y_test, X_train, Y_train, threshold)
    test = sd.het_breuschpagan(Quantile_Residuals_Series, X_test)
    
    return test

# ===============
# Normality Test
# ===============

def Normal_Residual_Test(function, X_test, Y_test, X_train, Y_train, threshold):
    
    Quantile_Residuals_Series = Quantile_Residuals(function, X_test, Y_test, X_train, Y_train, threshold)

    sm.qqplot(Quantile_Residuals_Series, line='45',scale=1)

    plt.title("Normal Test", fontsize=15, pad=18)
    plt.xlabel("Alpha",fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize = 14)
    plt.yticks(fontsize=12)
    plt.rcParams["figure.figsize"] = (2.7,2.5)
    plt.rcParams["legend.title_fontsize"] = 7

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)  

    #pylab.show()

    normal_test = sm.stats.normaltest(Quantile_Residuals_Series)
    
# d = Normal_Residual_Test(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
#              ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)

# ===========================================================
# Durbin Watson Test for Residuals correlation range(1,5 - 2)
# ===========================================================

def Durbin_Watson_Test(function, X_test, Y_test, X_train, Y_train, threshold):
    
    Quantile_Residuals_Series = Quantile_Residuals(function, X_test, Y_test, X_train, Y_train, threshold)

    test_corr_res = durbin_watson(Quantile_Residuals_Series)
    
    return test_corr_res

# ======================================
# Partial Plots - Residuals vs Features
# ======================================

def Partial_Plots(function, independent, X_test, Y_test, X_train, Y_train, threshold):
    
    Quantile_Residuals_Series = Quantile_Residuals(function, X_test, Y_test, X_train, Y_train, threshold)
    plt.scatter(independent, Quantile_Residuals_Series)

    plt.title("Cross Validate", fontsize=15, pad=18)
    plt.xlabel("Alpha",fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize = 14)
    plt.yticks(fontsize=12)
    plt.rcParams["figure.figsize"] = (2.7,2.5)
    plt.rcParams["legend.title_fontsize"] = 7

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)  
    
    return plt.show()

# d = Partial_Plots(GLM_Bino.GLM_Binomial_fit,train_test.X_test["CHILDREN"], train_test.X_test\
#              ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)

# =======================
# Outliers and Influence
# =======================


def Leverage_Studentized_Quantile_Res(function, X_test, Y_test, X_train, Y_train, threshold):

    res = function(X_train, Y_train)[1]
    Quantile_Residuals_Series = Quantile_Residuals(function, X_test, Y_test, X_train, Y_train, threshold)
    
    hat_matrix = np.round(res.get_hat_matrix_diag(),2)

    lev_stud_res = []

    for i in range(len(Quantile_Residuals_Series)):
        
        lev_stud_res.append(Quantile_Residuals_Series[i]/(sqrt(1-hat_matrix[i])))

    pd.Series(lev_stud_res).plot()

    plt.title("Cross Validate", fontsize=15, pad=18)
    plt.xlabel("Alpha",fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize = 14)
    plt.yticks(fontsize=12)
    plt.rcParams["figure.figsize"] = (2.7,2.5)
    plt.rcParams["legend.title_fontsize"] = 7

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)  
    
    return plt.show()

# f = Leverage_Studentized_Quantile_Res(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
#               ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)


# ================
# Cook's Distance
# ================

def Cooks_Distance_Quantile_Res(function, X_test, Y_test, X_train, Y_train, threshold):
    
    Quantile_Residuals_Series = Quantile_Residuals(function, X_test, Y_test, X_train, Y_train, threshold)
    res = function(X_train, Y_train)[1]
    hat_matrix = res.get_hat_matrix_diag()

    D = []

    for i in range(len(Quantile_Residuals_Series)):
        
        D.append((Quantile_Residuals_Series[i]**2/3000)*(hat_matrix[i]/(1-hat_matrix[i])))

    pd.Series(D).plot()

    plt.title("Cross Validate", fontsize=15, pad=18)
    plt.xlabel("Alpha",fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize = 14)
    plt.yticks(fontsize=12)
    plt.rcParams["figure.figsize"] = (2.7,2.5)
    plt.rcParams["legend.title_fontsize"] = 7

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)  
    
    return plt.show()

# f = Cooks_Distance_Quantile_Res(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
#               ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)

