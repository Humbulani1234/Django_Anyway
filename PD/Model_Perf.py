
# =================
# MODEL ASSESSMENT
# =================

# =======================
# Perfomance measurement
# =======================

# ==========================================
# ROC Curve Analytics and Optimal threshold
# ==========================================
# 
import GLM_Bino
import train_test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

# f = plt.figure(figsize=(3,3))

def ROC_Curve_Analytics(function, X_test, Y_test, X_train, Y_train):
    
    # res = (function(X_train, Y_train))[1]
    # predict_probability = res.predict(X_test)

    # fpr,tpr,thresholds = metrics.roc_curve(Y_test, predict_probability)

    # f, ax = plt.subplots(figsize=(3,3))
    # #fig = plt.figure()
    # # plt.plot(fpr,tpr)
    # #ax = f.add_subplot(1,1,1)
    # out = ax.plot(fpr,tpr)

    # optimal_idx = np.argmax(tpr-fpr)
    # optimal_thres = thresholds[optimal_idx]
    
    # return out, #optimal_thres 

    res = (function(X_train, Y_train))[1]
    predict_probability = res.predict(X_test)

    fpr,tpr,thresholds = metrics.roc_curve(Y_test, predict_probability)
    
    #Roc plot

    plt.plot(fpr,tpr)

    plt.title("ROC Curve", fontsize=12, pad=15)
    plt.xlabel("fpr",fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('tpr', fontsize = 14)
    plt.yticks(fontsize=14)
    plt.rcParams["figure.figsize"] = (2.7,2.5)
    plt.rcParams["legend.title_fontsize"] = 7

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)
    #plt.show()

    optimal_idx = np.argmax(tpr-fpr)
    optimal_thres = thresholds[optimal_idx]
    
    return optimal_thres, #plt.show()


#y = ROC_Curve_Analytics(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.Y_test, train_test.X_train\
#, train_test.Y_train)

#plt.show()

# ========================================
# Prediction Function @ maximal threshold
# ========================================

def Predict(function, X_test, Y_test, X_train, Y_train, threshold):
     
    res = function(X_train, Y_train)[1]
    predict_probability = res.predict(X_test)
    k = predict_probability.values.tolist()
    predict_binary = k.copy()

    for i in range(Y_test.shape[0]):

        if predict_binary[i] < threshold:
            predict_binary[i] = 1
            
        else: 
            predict_binary[i] = 0
        
        predict_binary = pd.Series(predict_binary)

    return predict_binary

#p = (Predict(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.Y_test, train_test.X_train, train_test.Y_train\
#, threshold=0.5))

#======================
#Confusion Matrix Plot
#======================

def Confusion_matrix_plot(function, X_test, Y_test, X_train, Y_train, threshold):
    
    predict_binary = Predict(function, X_test, Y_test, X_train, Y_train, threshold)
    
    z = confusion_matrix(Y_test, predict_binary, labels = [0, 1])
    z_1 = ConfusionMatrixDisplay(z, display_labels = ["No Default", "Yes Default"])
    #z_1.plot()
    #
    plt.title("Confusion Matrix", fontsize=15, pad=18)
    plt.xlabel("Predicted Label",fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('True Label', fontsize = 14)
    plt.yticks(fontsize=12)
    plt.rcParams["figure.figsize"] = (2.7,2.5)
    plt.rcParams["legend.title_fontsize"] = 7

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)

    return #plt.show()

# a = (Confusion_matrix_plot(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.Y_test, train_test.X_train\
#      , train_test.Y_train, threshold=0.5))


def Prediction(function, X_test, X_train, Y_train):
     
    res = function(X_train, Y_train)[1]
    predict_probability = res.predict(X_test)
    #print(predict_probability)
    k = [round(i,10) for i in predict_probability.values.tolist()]
    #print(k)
    predict_binary = k.copy()
    #print(predict_binary)

    return predict_binary

#print(Prediction(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.X_train, train_test.Y_train))