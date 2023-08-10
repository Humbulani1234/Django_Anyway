

"""

    ===================================
    MODEL ALTERNATIVES - Decision Tree
    ===================================

    We investigate Decision Trees as a model alternative to GLM - Binomial

    ==========
    Base Tree
    ==========

    ===============
    Fit a base tree
    ===============


"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle 

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat

# -----------------------------------------------------------------Class DecisionTree------------------------------------------------

# with open('static/decision_tree.pkl','rb') as file:
#         loaded_model = pickle.load(file)

class BaseDecisonTree(OneHotEncoding):

    """ Fit a base tree """

    def __init__(self, custom_rcParams, df_nomiss_cat, which, x_test, y_test,
                  df_loan_float, target, threshold, randomstate):

        super().__init__(custom_rcParams, df_nomiss_cat, which)
        
        self.df_loan_float = df_loan_float
        self.target = target
        self.randomstate = randomstate
        # self.x_train = super().split_xtrain_ytrain(self.df_loan_float, self.target)[0]
        # self.y_train = super().split_xtrain_ytrain(self.df_loan_float, self.target)[2]


    def dt_classification_fit(self, ccpalpha):
        
        ''' DT Classification fit '''
        
        clf_dt = DecisionTreeClassifier(self.randomstate, ccpalpha)
        clf_dt = clf_dt.fit(self.x_train, self.y_train)
        
        return clf_dt

    def dt_binary_prediction(self, ccpalpha):
        
        ''' Base tree prediction '''
        
        # clf_dt = self.function(self.x_train, self.y_train, self.randomstate, self.ccpalpha)
        
        predict_dt = super().dt_classification_fit(ccpalpha).predict(self.x_test)
        predict_dt_series = pd.Series(predict_dt)

        return predict_dt_series

    def dt_confusion_matrix_plot(self, ccpalpha):

        """ Base tree Confusion matrix """

        predict_dt_series = super().dt_binary_prediction(ccpalpha)       
        conf_matrix = confusion_matrix(self.y_test, predict_dt_series, labels = [0, 1])

        conf_matrix_plot = ConfusionMatrixDisplay(conf_matrix, display_labels = ["No Default", "Yes Default"])
        conf_matrix_plot.plot(cmap="Blues", ax=self.axs, values_format="d")       
        conf_matrix_plot.ax_.set_title("Confusion Matrix", fontsize=15, pad=18)
        conf_matrix_plot.ax_.set_xlabel("Predicted Label",fontsize=14)
        conf_matrix_plot.ax_.set_ylabel('True Label', fontsize = 14)

        return self.fig

    def plot_dt(self):

        """ Base tree plot """
        
        plot_tree(super().dt_classification_fit(ccpalpha), filled = True, rounded = True, 
                  class_names = ["No Default", "Yes Default"], feature_names = self.x_train.columns, ax = self.axs)   

        return self.fig  

# --------------------------------------------------------------Class PrunedTree------------------------------------------------------------

class PrunedDecisionTree(BaseDecisonTree):

    def pruning(self, ccpalpha):

        """ Extracting alphas for pruning """
        
        clf_dt = super().dt_classification_fit(ccpalpha)

        path = clf_dt.cost_complexity_pruning_path(self.x_train, self.y_train) 
        ccp_alphas = path.ccp_alphas 
        ccp_alphas = ccp_alphas[:-1] 
        
        return ccp_alphas

    def cross_validate_alphas(self, ccpalpha):
        
        """ Cross validation for best alpha """

        alpha_loop_values = []
        
        ccp_alphas = super().pruning(ccpalpha)

        for ccp_alpha in ccp_alphas:

            clf_dt = DecisionTreeClassifier(random_state=self.randomstate, ccp_alpha=ccp_alpha)
            scores = cross_val_score(clf_dt, self.x_train, self.y_train, cv=5)
            alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
        
        alpha_results = pd.DataFrame(alpha_loop_values, columns=["alpha", "mean_accuracy", "std"])
        alpha_results.plot(ax = self.axs, x = "alpha", y = "mean_accuracy", yerr = "std", marker = "o" , linestyle = "--")
        
        return alpha_results, self.fig

    def ideal_alpha(self, ccpalpha, threshold_1, threshold_2):
        
        """ Extraction of ideal alpha """

        alpha_results = super().cross_validate_alphas(ccpalpha)[0]
        
        ideal_ccp_alpha = alpha_results[(alpha_results["alpha"] > threshold_1) & (alpha_results["alpha"] < threshold_2)]["alpha"]
        ideal_ccp_alpha = ideal_ccp_alpha.values.tolist()
        
        return ideal_ccp_alpha[0]
    
    def dt_pruned_tree(self, ccpalpha, threshold_1, threshold_2):
    

        """ Ideal alpha value for pruning the tree """

        ideal_ccp_alpha = super().ideal_alpha(ccpalpha=0, threshold_1=0.0019, threshold_2=0.0021)

        """ Pruned tree fitting """

        pruned_clf_dt = super().dt_classification_fit(ccpalpha=ideal_ccp_alpha)

        """ Prediction and perfomance analytics """

        pruned_predict_dt = super().dt_binary_prediction(ccpalpha=ideal_ccp_alpha)

        """ Confusion matrix plot """

        pruned_confusion_matrix = super().dt_confusion_matrix_plot(ccpalpha=ideal_ccp_alpha)

        """ Plot final tree """

        pruned_plot_tree = super().plot_dt(ccpalpha=ideal_ccp_alpha)

        return ideal_ccp_alpha, clf_dt_pruned, pruned_predict_dt, pruned_confusion_matrix, pruned_plot_tree

# --------------------------------------------------------Testing--------------------------------------------------------------------------------

# if __name__ == "__main__":

    file_path = "static/KGB.sas7bdat"
    data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)    
    miss = ImputationCat(df_cat=df_loan_categorical)
    imputer_cat = miss.simple_imputer_mode()
    to_view = miss.concatenate_total_df(df_loan_float, imputer_cat)

    #print(to_use)

    # custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

    custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

    instance = OneHotEncoding(custom_rcParams, imputer_cat, "machine")
    #instance.sample_imbalance(df_loan_float, df_loan_float["GB"])
    
    # x_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]
    # y_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[2]
    y_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]
    x_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]

    #pdb.set_trace()

    #pdb.set_trace()

    # Model Perfomance
    
    threshold = 0.47

    # d = BaseDecisonTree(custom_rcParams, imputer_cat, which, x_test, y_test,
    #                     df_loan_float, df_loan_float["GB"], threshold, randomstate, ccpalpha)

    #print(d.dt_classification_fit(ccpalpha = 0))

    # e = PrunedDecisionTree(custom_rcParams, imputer_cat, which, x_test, y_test,
    #                     df_loan_float, df_loan_float["GB"], threshold, randomstate)


