
# ===============================================================================
# ANALYSIS OF REJECT INFERENCE AND FINAL COMPLETE MODEL - through Decision Tree
# ===============================================================================

# ==============
# Data download
# ==============

df_loan_reject = Data_download(file_path = "KGB.sas7bdat")
df_loan_reject

# ==============
# Data Cleaning
# ==============

df_loan_reject_types, df_loan_reject_categorical, df_loan_reject_float = Data_cleaning(df = df_loan_reject)
df_loan_reject_categorical

# ======================================
# Missing values analysis and imputation
# ======================================

df_loan_reject_categorical_mode = Simple_Imputer_mode(df_loan_reject_categorical)
print(df_loan_reject_categorical_mode)

# =================================
# Data Clustering with K-Prototypes
# =================================

K_prototype_cluster = K_Prototypes_Clustering(dataframe, values, cluster_no)

# =========================================================================================
# Prediction with Pruned Decision Tree to determine what would have been Bad/Good customers
# =========================================================================================

df_loan_reject_categorical_encoded = One_Hot_Encoding_Func_Machi(dataframe=df_loan_reject_categorical_mode)
df_loan_reject_categorical_encoded

# ===============================================
# Create Testing Dataframe (the whole dataframe)
# ===============================================

df_loan_reject_total_partition = pd.concat([df_loan_reject_float, df_loan_reject_categorical_encoded], axis = 1)

X_reject, Y_reject = Create_X_Y(df_loan_reject_float, df_loan_reject_categorical_encoded\
                                      , target=df_loan_reject_float["GB"])

X_reject = X_reject.drop(labels=["_freq_"], axis=1)
X_reject

# ================
# Run prediction
# ================

Predict_binary_DT_reject = Predict_binary_DT(DT_Classification_fit, X_reject, Y_reject, X_train, Y_train\
                                              , randomstate=42, ccpalpha=ideal_ccp_alpha)
Predict_binary_DT_reject

# ==========================================
# Final Dataframes - Float and Categorical
# ==========================================

df_loan_reject_total_no_missing = pd.concat([df_loan_reject_float, df_loan_reject_categorical_mode], axis=1)
df_loan_reject_total_no_missing

dataframe_1 = df_loan_total_no_missing
dataframe_1 = pd.concat([df_loan_total_no_missing.drop(labels=["GB"], axis=1),df_loan_total_no_missing["GB"]], axis=1)
print(dataframe_1)


df_loan_reject_total_no_missing= df_loan_reject_total_no_missing.drop(labels=["GB"], axis=1)
dataframe_2_reject = pd.concat([df_loan_reject_total_no_missing, Predict_binary_DT_reject], axis=1)
dataframe_2_reject.rename(columns = {0: "GB"}, inplace=True)
print(dataframe_2_reject)

Final_dataframe_reject = pd.concat([dataframe_1, dataframe_2_reject], axis=0)
print(Final_dataframe_reject)

Final_dataframe_categorical_reject = Final_dataframe_reject.select_dtypes(object)
print(Final_dataframe_categorical_reject)
Final_dataframe_float_reject = Final_dataframe_reject.select_dtypes(float)
print(Final_dataframe_float_reject)

# ===================
# Fit a Decision Tree
# ===================

# =============================================
# Creating independent and dependent variables
# =============================================

Final_dataframe_categorical_reject_encoded = One_Hot_Encoding_Func_Machi(dataframe=Final_dataframe_categorical_reject)
print(Final_dataframe_categorical_reject_encoded)

df_loan_reject_total_partition_final = pd.concat([Final_dataframe_float_reject, Final_dataframe_categorical_reject_encoded]\
                                                  , axis = 1)

X_final, Y_final = Create_X_Y(dataframe_float, dataframe_categorical, target)

X_final = X_final.drop(labels=["_freq_"], axis=1)
X_final

# ===========================================
# Sample partition into Train and Test sets
# ===========================================

X_train_reject, X_test_reject, Y_train_reject, Y_test_reject = Split_Xtrain_Ytrain(dataframe_float\
                                                               , dataframe_categorical,target , testsize, randomstate)

X_train_reject = X_train_reject.drop(labels=["_freq_"], axis=1) # temp, for mach it has to be dropped
print(X_train_reject)

X_test_reject = X_test_reject.drop(labels=["_freq_"], axis=1) # temp
print(X_test_reject)

# ===========================================
# Find ideal alpha through cross validation
# ===========================================

print(Cross_Validate_Alphas(DT_Classification_fit, X_train_reject, Y_train_reject, randomstate=42, ccpalpha=0))

ideal_ccp_alpha = Ideal_Alpha(DT_Classification_fit, X_train_reject, Y_train_reject, threshold_1=0.00115
                              , threshold_2=0.00125, randomstate=42, ccpalpha=0)
print(ideal_ccp_alpha)

# =======================
# Fitting a pruned tree
# =======================

clf_dt_pruned = DT_Classification_fit(X_train_reject, Y_train_reject, randomstate=42, ccpalpha=ideal_ccp_alpha)

# ============================================
# Perfomance and Goodness of fit of the model
# ============================================

# ============
# Prediction
# ============

predict_DT_Series_reject = Predict_binary_DT(DT_Classification_fit, X_test_reject, Y_test_reject, X_train_reject\
                                              , Y_train_reject, randomstate=42, ccpalpha=ideal_ccp_alpha)
predict_DT_Series_reject

# ======================
# Confusion matrix plot
# ======================

Confusion_matrix_plot_reject = Confusion_matrix_plot_DT(DT_Classification_fit, X_train_reject, Y_train_reject,
                                            X_test_reject, Y_test_reject, randomstate=42, ccpalpha=ideal_ccp_alpha)
Confusion_matrix_plot_reject

# ====================
# Decision tree plot
# ====================

Plot_tree_reject = Plot_DT(DT_Classification_fit, X_train_reject, Y_train_reject, randomstate=42, ccpalpha=ideal_ccp_alpha)
Plot_tree_reject
