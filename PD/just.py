# =================
# MODEL DEPLOYMENT
# =================

# =================
# Save with pickle
# =================

# ========
# Options:
# ========

#     1. Cloud deployment
#     2. Build a GUI for the model

# ===============================================================================================================================
import streamlit as st 
import train_test
import Model_Perf
import ED
import matplotlib.pyplot as plt
import GLM_Bino
import warnings
import missing_adhoc
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)

pd.set_option("display.width", 30000)
pd.set_option("display.max_columns", 30000)
pd.set_option("display.max_rows", 30000)
pd.set_option("display.float_format", lambda x: "%.0f" %x)
warnings.filterwarnings("ignore")

# Create headings

st.title("Probability of Default Prediction")
st.subheader("This model will predict the probability of default for a customer")
#st.table(train_test.X_train.head(5))
st.dataframe(train_test.X_test)

#st.subheader("Train Set Score: {}".format (round(train_score,3)))
#st.subheader("Test Set Score: {}".format (round(test_score,3)))

# Plot the confusion matrix

#confusion = confusion_matrix(y_test, y_predict)
confusion = Model_Perf.a[0]
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]


plt.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP],width=0.15)
plt.xticks(fontsize=7)
plt.xlabel("",fontsize=7)
plt.yticks(fontsize=7)
plt.ylabel("",fontsize=7)
#plt.show()
#plt.gca()
#plt.lines()
st.pyplot()
#plt.show()
# Create entries for inputs - Categorical
#def user_input():

# NAME = st.text_input("Name of Customer")
# TITLE = st.selectbox("TITLE", options=['H']) # dropped R
# STATUS = st.selectbox("STATUS",options=['V','U','G','E','T']) #dropped W
# #print(STATUS)
# PRODUCT = st.selectbox('PRODUCT',options=['Radio_TV_Hifi','Furniture_Carpet', 'Dept_Store_Mail', 'Leisure','Cars', 'OT']) # dropped Radio
# RESID = st.selectbox('RESID',options=['Lease','Owner']) # dropped Owner
# NAT = st.selectbox('NAT',options=['German', 'Turkish', 'RS', 'Greek' ,'Yugoslav', 'Italian', 'Other_European','Spanish_Portugue']) #dropped Yugoslavia
# PROF = st.selectbox('PROF',options=['Others','Civil_Service_M' ,'Self_employed_pe', 'Food_Building_Ca','Chemical_Industr'\
# , 'Pensioner' ,'Sea_Vojage_Gast', 'State_Steel_Ind,','Military_Service']) # dropped State_Steel_Ind
# CAR = st.selectbox('CAR',options=['Car', 'Without_Vehicle', 'Car_and_Motor_bi']) # dropped Without_Vehicle
# CARDS = st.selectbox("CARDS",options=['Cheque_card' ,'no_credit_cards', 'Mastercard_Euroc', 'VISA_mybank','VISA_Others'\
# , 'Other_credit_car', 'American_Express']) # dropped Cheque_card

#data = {'TITLE':[TITLE],'STATUS':[STATUS],'PRODUCT':[PRODUCT],'RESID':[RESID],'NAT':[NAT],'PROF':[PROF],'CAR':[CAR]\
#,'CARDS':[CARDS]}

#features = pd.DataFrame(data)

#return features

#input_df = user_input()
#print(input_df)

# Float features

# def user_input():

NAME = st.text_input("Name of Customer")
AGE = st.slider("Enter", 0,100)
# CHILDREN = st.slider("CHILDREN", 1, 10,1)
# PERS_H = st.slider("PERS_H", 1, 10,1)
# TMADD = st.slider("TMADD", 1, 1000,1)
# TMJOB1 = st.slider("TMJOB1", 1, 1000,1)
# TEL = st.slider("TEL", 1, 10,1)
# NMBLOAN = st.slider("NMBLOAN", 1, 10,1)
# FINLOAN = st.slider("FINLOAN", 1, 10,1)
# INCOME = st.slider("INCOME", 1, 1000000,100)
# EC_CARD = st.slider("EC_CARD", 1, 10,1)
# INC = st.slider("INC", 1, 1000000,100)
# INC1 = st.slider("INC1", 1, 10,1)
# BUREAU = st.slider("BUREAU", 1, 10,1)
# LOCATION = st.slider("LOCATION", 1, 10,1)
# LOANS = st.slider("LOANS", 1, 10,1)
# REGN = st.slider("REGN", 1, 10,1)
# DIV = st.slider("DIV", 1, 10,1)
# CASH = st.slider("CASH", 1, 1000000,100)

# Categorical values

# TITLE = st.selectbox("TITLE", options=['H']) # dropped R
# STATUS = st.selectbox("STATUS",options=['V','U','G','E','T']) #dropped W
# PRODUCT = st.selectbox('PRODUCT',options=['Radio_TV_Hifi','Furniture_Carpet', 'Dept_Store_Mail', 'Leisure','Cars', 'OT']) # dropped Radio
# RESID = st.selectbox('RESID',options=['Lease','Owner']) # dropped Owner
# NAT = st.selectbox('NAT',options=['German', 'Turkish', 'RS', 'Greek' ,'Yugoslav', 'Italian', 'Other_European','Spanish_Portugue']) #dropped Yugoslavia
# PROF = st.selectbox('PROF',options=['Others','Civil_Service_M' ,'Self_employed_pe', 'Food_Building_Ca','Chemical_Industr'\
# , 'Pensioner' ,'Sea_Vojage_Gast', 'State_Steel_Ind,','Military_Service']) # dropped State_Steel_Ind
# CAR = st.selectbox('CAR',options=['Car', 'Without_Vehicle', 'Car_and_Motor_bi']) # dropped Without_Vehicle
# CARDS = st.selectbox("CARDS",options=['Cheque_card' ,'no_credit_cards', 'Mastercard_Euroc', 'VISA_mybank','VISA_Others'\
# , 'Other_credit_car', 'American_Express']) # dropped cheque card

agree = st.checkbox('I agree to the terms and conditions')

# if st.button("update"):
# 	age=AGE
# 	st.write(age)
	
# 	status=STATUS
# 	st.write(status)

# print(age)
#print(status)	

# age = AGE
# status = STATUS

button_clicked = st.button('Submit')

def update_variables():
    if button_clicked:
        print(f"Name: {NAME}, age: {AGE}")

	# st.write('updated', STATUS)
	# st.write('updated', AGE)
	# st.write('updated', NAME)
	#Use the values of the input widgets to perform some action
	#print('STATUS:', STATUS)
	#print('AGE:', AGE)
update_variables()
#print(AGE)
#print(NAME)

# print(STATUS)
# print(AGE)
# print(NAME)
# 	data = {'AGE':[AGE], 'CHILDREN':[CHILDREN], 'PERS_H':[PERS_H], 'TMADD':[TMADD]}
# 	features = pd.DataFrame(data)

# input_df = user_input()
# print(input_df)

# uploaded_file = st.sidebar.file_uploader("test.csv", type=['csv'])
# if uploaded_file is not None:
# 	input_df = pd.read_csv(uploaded_file)
# else:
# 	def user_input():

        
# 		AGE = st.slider("AGE", 1, 100,1)
# 		CHILDREN = st.slider("CHILDREN", 1, 10,1)
# 		PERS_H = st.slider("PERS_H", 1, 10,1)
# 		TMADD = st.slider("TMADD", 1, 1000,1)
# 		TMJOB1 = st.slider("TMJOB1", 1, 1000,1)
# 		TEL = st.slider("TEL", 1, 10,1)
# 		NMBLOAN = st.slider("NMBLOAN", 1, 10,1)
# 		FINLOAN = st.slider("FINLOAN", 1, 10,1)
# 		INCOME = st.slider("INCOME", 1, 1000000,100)
# 		EC_CARD = st.slider("EC_CARD", 1, 10,1)
# 		INC = st.slider("INC", 1, 1000000,100)
# 		INC1 = st.slider("INC1", 1, 10,1)
# 		BUREAU = st.slider("BUREAU", 1, 10,1)
# 		LOCATION = st.slider("LOCATION", 1, 10,1)
# 		LOANS = st.slider("LOANS", 1, 10,1)
# 		REGN = st.slider("REGN", 1, 10,1)
# 		DIV = st.slider("DIV", 1, 10,1)
# 		CASH = st.slider("CASH", 1, 1000000,100)

# 		# Categorical values

# 		TITLE = st.selectbox("TITLE", options=['H']) # dropped R
# 		STATUS = st.selectbox("STATUS",options=['V','U','G','E','T']) #dropped W
# 		PRODUCT = st.selectbox('PRODUCT',options=['Radio_TV_Hifi','Furniture_Carpet', 'Dept_Store_Mail', 'Leisure','Cars', 'OT']) # dropped Radio
# 		RESID = st.selectbox('RESID',options=['Lease','Owner']) # dropped Owner
# 		NAT = st.selectbox('NAT',options=['German', 'Turkish', 'RS', 'Greek' ,'Yugoslav', 'Italian', 'Other_European','Spanish_Portugue']) #dropped Yugoslavia
# 		PROF = st.selectbox('PROF',options=['Others','Civil_Service_M' ,'Self_employed_pe', 'Food_Building_Ca','Chemical_Industr'\
# 		, 'Pensioner' ,'Sea_Vojage_Gast', 'State_Steel_Ind,','Military_Service']) # dropped State_Steel_Ind
# 		CAR = st.selectbox('CAR',options=['Car', 'Without_Vehicle', 'Car_and_Motor_bi']) # dropped Without_Vehicle
# 		CARDS = st.selectbox("CARDS",options=['Cheque_card' ,'no_credit_cards', 'Mastercard_Euroc', 'VISA_mybank','VISA_Others'\
# 		, 'Other_credit_car', 'American_Express']) # dropped cheque card

# 		data = {'AGE':[AGE], 'CHILDREN':[CHILDREN], 'PERS_H':[PERS_H], 'TMADD':[TMADD]}
# 		features = pd.DataFrame(data)

# 		return features

# 	input_df = user_input()


# Input the values for prediction purposes

#input_data = scaler.transform([[sex , age, f_class , s_class, t_class]])
#prediction = Model_Perf.Prediction(GLM_Bino.GLM_Binomial_fit,train_test.X_test.iloc[20], train_test.X_train, train_test.Y_train)
#print(prediction)
#predict_probability = model.predict_proba(input_data)

# Displaying our prediction

#if prediction[0] == 1:
#st.subheader('Passenger {} would have survived with a probability of {}'.format(NAME , prediction))
#else:
	#st.subheader('Passenger {} would not have survived with a probability of {}%'.format(name, round(predict_probability[0][0]*100 , 3)))
#print(STATUS)
#print(AGE)