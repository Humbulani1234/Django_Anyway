
import streamlit as st 
import train_test
import Model_Perf
import ED
import matplotlib.pyplot as plt
import GLM_Bino
import warnings
import missing_adhoc
import pandas as pd
from PIL import Image

# Float

st.set_option('deprecation.showPyplotGlobalUse', False)

pd.set_option("display.width", 30000)
pd.set_option("display.max_columns", 30000)
pd.set_option("display.max_rows", 30000)
pd.set_option("display.float_format", lambda x: "%.0f" %x)
warnings.filterwarnings("ignore")

st.title("Probability of Default Prediction")
image = Image.open('pngegg.png')
st.image(image,use_column_width=True)
st.subheader("This model will predict the probability of default for a customer")

dataset_name = st.sidebar.selectbox('Select dataset', ('KGB', 'Pass'),key=29)
classifier_name = st.sidebar.selectbox('Select classifier', ('Logistic', 'Decision'),key=30)
visualization_name=st.sidebar.selectbox('Select Visuals', ('confusion','Cross_tab','Pivot'), key=31)

def get_dataset(name):
	data=None
	if name=='KGB':
		data = train_test.X_test
	else:
		pass
	return data

data = get_dataset(dataset_name)
st.dataframe(data)
st.write('Shape of dataframe:', data.shape)

def get_data(name):

    data=None

    if name=='Cross_tab':
        missing_adhoc.Categorical_missingness_Crosstab_Plot(ED.df_loan_categorical["RESID"], ED.df_loan_float["GB"])
        st.pyplot()

    elif name=='confusion':

        confusion = Model_Perf.a
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

# missing_adhoc.Categorical_missingness_Pivot_Plot(ED.df_loan_categorical["RESID"], ED.df_loan_float["GB"])
# st.pyplot()


    else:
        missing_adhoc.Categorical_missingness_Pivot_Plot(ED.df_loan_categorical["RESID"], ED.df_loan_float["GB"])
        st.pyplot()

    return data

get_data(visualization_name)

#missing_adhoc.Categorical_missingness_Crosstab_Plot(ED.df_loan_categorical["RESID"], ED.df_loan_float["GB"])

#st.pyplot()

# Model_Perf.Confusion_matrix_plot(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.Y_test, train_test.X_train\
#     ,train_test.Y_train, threshold=0.5)

# st.pyplot()

# confusion = Model_Perf.a[0]
# FN = confusion[1][0]
# TN = confusion[0][0]
# TP = confusion[1][1]
# FP = confusion[0][1]


# plt.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP],width=0.15)
# plt.xticks(fontsize=7)
# plt.xlabel("",fontsize=7)
# plt.yticks(fontsize=7)
# plt.ylabel("",fontsize=7)
# #plt.show()
# #plt.gca()
# #plt.lines()
# st.pyplot()

# missing_adhoc.Categorical_missingness_Pivot_Plot(ED.df_loan_categorical["RESID"], ED.df_loan_float["GB"])
# st.pyplot()

# def add_parameter():

#     if name_of_clf == 'Logistic':

#def add_parameters():

    #global NAME
    #global inputs


NAME = st.sidebar.text_input("Customer", key=90)
AGE = st.sidebar.slider("Enter", 0,100,key=10)
CHILDREN = st.sidebar.slider("CHILDREN", 0, 10,key=11)
PERS_H = st.sidebar.slider("PERS_H", 0, 10,key=12)
TMADD = st.sidebar.slider("TMADD", 0, 1000,key=13)
TMJOB1 = st.sidebar.slider("TMJOB1", 0, 1000,key=14)
TEL = st.sidebar.slider("TEL", 1, 10,key=15)
NMBLOAN = st.sidebar.slider("NMBLOAN", 0, 10,key=16)
FINLOAN = st.sidebar.slider("FINLOAN", 0, 10,key=17)
INCOME = st.sidebar.slider("INCOME", 1, 1000000,100,key=18)
EC_CARD = st.sidebar.slider("EC_CARD", 1, 10,1,key=19)
INC = st.sidebar.slider("INC", 1, 1000000,100,key=20)
INC1 = st.sidebar.slider("INC1", 1, 10,1,key=21)
BUREAU = st.sidebar.slider("BUREAU", 1, 10,1,key=22)
LOCATION = st.sidebar.slider("LOCATION", 1, 10,1,key=23)
LOANS = st.sidebar.slider("LOANS", 1, 10,1,key=24)
REGN = st.sidebar.slider("REGN", 1, 10,1,key=25)
DIV = st.sidebar.slider("DIV", 1, 10,1,key=26)
CASH = st.sidebar.slider("CASH", 1, 1000000,100,key=27)        

# Categorical
# 
TITLE = st.sidebar.selectbox("TITLE", options=['H','R'], key=2)
STATUS = st.sidebar.selectbox("STATUS",options=['V','U','G','E','T','W'], key=3)
PRODUCT = st.sidebar.selectbox('PRODUCT',options=['Radio_TV_Hifi','Furniture_Carpet', 'Dept_Store_Mail', 'Leisure','Cars', 'OT'], key=4) # dropped Radio
RESID = st.sidebar.selectbox('RESID',options=['Lease','Owner'], key=5) # dropped Owner
NAT = st.sidebar.selectbox('NAT',options=['German', 'Turkish', 'RS', 'Greek' ,'Yugoslav', 'Italian', 'Other_European','Spanish_Portugue'], key=6) #dropped Yugoslavia
PROF = st.sidebar.selectbox('PROF',options=['Others','Civil_Service_M' ,'Self_employed_pe', 'Food_Building_Ca','Chemical_Industr'\
, 'Pensioner' ,'Sea_Vojage_Gast', 'State_Steel_Ind,','Military_Service'], key=7) # dropped State_Steel_Ind
CAR = st.sidebar.selectbox('CAR',options=['Car', 'Without_Vehicle', 'Car_and_Motor_bi'], key=8) # dropped Without_Vehicle
CARDS = st.sidebar.selectbox("CARDS",options=['Cheque_card' ,'no_credit_cards', 'Mastercard_Euroc', 'VISA_mybank','VISA_Others'\
, 'Other_credit_car', 'American_Express'], key=9) # dropped cheque card    


#button_clicked = st.button('Submit', key=28)  
 
button_clicked = st.button('Submit', key=28)    

def update_variables():    

    #global STATUS
    #global CHILDREN 
    #global inputs 
    #add_parameters()

    # button_clicked = st.button('Submit', key=28)   

    if button_clicked:    

        V, U, G, E, T = 0,0,0,0,0    

        if STATUS == 'V':
            V=1
        elif STATUS == 'U':
            U=1
        elif STATUS == 'G':
            G=1
        elif STATUS == 'E':
            E=1
        elif STATUS=='T':
            T=1
        else:
            V, U, G, E, T = 0,0,0,0,0    

        #     

        H = 0    

        if TITLE=='H':
            H = 1
        else:
            H=0
        #
        
        Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0    

        if PRODUCT=='Furniture_Carpet':
	        Furniture_Carpet=1
        elif PRODUCT=='Dept_Store_Mail':
	        Dept_Store_Mail=1
        elif PRODUCT=='Leisure':
	        Leisure=1
        elif PRODUCT=='Cars':
	        Cars=1
        elif PRODUCT=='OT':
	        OT=1
        else:
            Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0    

	    #
	    
        Lease = 0    

        if RESID=='Lease':
	        Lease=1    

        else:
            Lease=0
	    #
	    
        German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0    

        if NAT=='German':
            German=1
        elif NAT=='Turkish':
            Turkish=1        
        elif NAT=='RS':
            RS=1
        elif NAT=='Greek':
            Greek=1
        elif NAT=='Italian':
            Italian=1
        elif NAT=='Other_European':
            Other_European=1
        elif NAT=='Spanish_Portugue':
            Spanish_Portugue=1
        else:
            German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0    




        #
        
        Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
        ,Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0    

        if PROF=='Others':
            Others=1
        elif PROF=='Civil_Service_M':
            Civil_Service_M=1
        elif PROF=='Self_employed_pe':
            Self_employed_pe=1
        elif PROF=='Food_Building_Ca':
            Food_Building_Ca=1
        elif PROF=='Chemical_Industr':
            Chemical_Industr=1
        elif PROF=='Pensioner':
            Pensioner=1
        elif PROF=='Sea_Vojage_Gast':
            Sea_Vojage_Gast=1
        elif PROF=='Military_Service':
            Military_Service=1
        else:
            Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
            ,Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0    


        #
        
        Car,Car_and_Motor_bi= 0,0    

        if CAR=='Car':
            Car=1
        elif CAR=='Car_and_Motor_bi':
            Car_and_Motor_bi=1
        else:
            Car,Car_and_Motor_bi= 0,0    

        #
        
        no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
        ,Other_credit_car, American_Express = 0,0,0,0,0,0    

        if CARDS=='no_credit_cards':
            no_credit_cards=1
        elif CARDS=='Mastercard_Euroc':
            Mastercard_Euroc=1
        elif CARDS == 'VISA_mybank':
            VISA_mybank=1
        elif CARDS=='VISA_Others':
            VISA_Others=1
        elif CARDS=='Other_credit_car':
            Other_credit_car=1
        elif CARDS=='American_Express':
            American_Express=1
        else:
            no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
            ,Other_credit_car, American_Express = 0,0,0,0,0,0    


        
        inputs1 = [H,V, U, G, E, T,Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT,Lease,German, Turkish, RS, Greek ,Italian\
        , Other_European, Spanish_Portugue,Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
        , Pensioner ,Sea_Vojage_Gast, Military_Service,Car,Car_and_Motor_bi,no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
        , Other_credit_car, American_Express]
        
        inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, LOCATION, LOANS\
        , REGN, DIV, CASH]    

        list_ = inputs2 + inputs1

        #if list_ is None:
            #inputs = train_test.X_test.iloc[0]
            #print(type(inputs))
        #else:
        inputs = pd.Series(list_)  
        #except ValueError:
            #print('Press Submit')
        prediction = Model_Perf.Prediction(GLM_Bino.GLM_Binomial_fit,inputs, train_test.X_train, train_test.Y_train)    

        st.subheader('Customer {} probability of default is: {}'.format(NAME , prediction))
        st.success('Successfully executed the model')

    #try:        
        #return inputs
    #except ValueError:
        #Bprint('Press Submit')


        # print(type(inputs))
        # print(inputs)
        # #print(train_test.X_test.iloc[0])    

        # prediction = Model_Perf.Prediction(GLM_Bino.GLM_Binomial_fit,inputs, train_test.X_train, train_test.Y_train)    

        # st.subheader('Customer {} probability of default is: {}'.format(NAME , prediction))
        # st.success('Successfully executed the model')
        # print(list_)    

    #print(f"Name: {NAME}, predict: {prediction}")    

update_variables()


# def show_inputs(name1):

#     #data=None
#     if name1=='Logistic':
#         add_parameters()
        
#     else:
#         pass

# show_inputs(classifier_name)

# def get_prediction(name2):

#     #data=None
#     if name2=='Logistic':
#         data = update_variables()
#         print(type(data))
#         prediction = Model_Perf.Prediction(GLM_Bino.GLM_Binomial_fit,data, train_test.X_train, train_test.Y_train)    

#         st.subheader('Customer {} probability of default is: {}'.format(NAME , prediction))
#         st.success('Successfully executed the model')

#     else:
#         pass


# get_prediction(classifier_name)

# # int(V)
# #print(train_test.X_test.iloc[0].tolist())
# # else:
# # pass

# # p = add_parameter(classifier_name)