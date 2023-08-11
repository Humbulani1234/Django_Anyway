
import pandas as pd
import numpy as np
import pickle
import sys

sys.path.append('/home/humbulani/New/django_project/refactored_pd')

from .forms import In, Si
from django.shortcuts import render, redirect
from django.contrib import messages

from class_decision_tree import DecisionTree
from class_missing_values import ImputationCat
from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning

# -------------------------------------------------------------------Defined Variables-------------------------------------------------------

file_path = "static/KGB.sas7bdat"
data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)    
miss = ImputationCat(df_cat=df_loan_categorical)
imputer_cat = miss.simple_imputer_mode()
to_view = miss.concatenate_total_df(df_loan_float, imputer_cat)

custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

instance = OneHotEncoding(custom_rcParams, imputer_cat, "machine")

x_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]
y_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[2]
y_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]
x_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]

# Model Perfomance

threshold = 0.47
randomstate = 42
ccpalpha = 0
threshold_1=0.0019
threshold_2=0.0021

d = DecisionTree(custom_rcParams, imputer_cat, "machine", y_test,
                    df_loan_float, df_loan_float["GB"], threshold, randomstate)

# d = class_decision_tree.BaseDecisonTree(custom_rcParams, imputer_cat, which, x_test, y_test,
#                     df_loan_float, df_loan_float["GB"], threshold, randomstate, ccpalpha)

# print(d.dt_classification_fit(ccpalpha = 0))

# e = class_decision_tree.PrunedDecisionTree(custom_rcParams, imputer_cat, which, x_test, y_test,
#                     df_loan_float, df_loan_float["GB"], threshold, randomstate)

# with open('static/decision_tree.pkl','rb') as file:
#         loaded_model = pickle.load(file)

# -------------------------------------------------------------------------------Views-----------------------------------------------------

def confusion_decision(request):

     return render (request, 'confusion_decision.html')


def decision_tree(request):

     return render (request, 'decision_tree.html')


def cross_validate(request):

     return render (request, 'cross_validate.html')


def tree(request):

    answer = ""

    if request.method == 'POST':
        form = In(request.POST)
        side_bar = Si(request.POST)
        if form.is_valid():

        	# Float features
        	
            NAME = form.cleaned_data.get("NAME")
            AGE = form.cleaned_data.get("AGE")
            CHILDREN = form.cleaned_data.get("CHILDREN")
            PERS_H = form.cleaned_data.get("PERS_H")
            TMADD = form.cleaned_data.get("TMADD")
            TMJOB1 = form.cleaned_data.get("TMJOB1")
            TEL = form.cleaned_data.get("TEL")
            NMBLOAN = form.cleaned_data.get("NMBLOAN")
            FINLOAN = form.cleaned_data.get("FINLOAN")
            INCOME = form.cleaned_data.get("INCOME")
            EC_CARD = form.cleaned_data.get("EC_CARD")
            INC = form.cleaned_data.get("INC")
            INC1 = form.cleaned_data.get("INC1")
            BUREAU = form.cleaned_data.get("BUREAU")
            LOCATION = form.cleaned_data.get("LOCATION")
            LOANS = form.cleaned_data.get("LOANS")
            REGN = form.cleaned_data.get("REGN")
            DIV = form.cleaned_data.get("DIV")
            CASH = form.cleaned_data.get("CASH")
            
    	    
    	    # Categorical features
    	    # 
            TITLE = form.cleaned_data.get("TITLE")
            R,H = 0,0
            if TITLE == 'H':
    	        H=1
    	        # list_.append(H)
            else:
    	        R=0
    	        # list_.append(H)
            #input_ = [H]
            #
            STATUS = form.cleaned_data.get("STATUS")

            W,V, U, G, E, T = 0,0,0,0,0,0    

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
                W = 0 

            PRODUCT = form.cleaned_data.get("PRODUCT") 

            Radio_TV_Hifi, Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0,0    

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
                Radio_TV_Hifi = 0   

            RESID = form.cleaned_data.get("RESID")

            Owner,Lease = 0,0    

            if RESID=='Lease':
    	        Lease=1    

            else:
                Owner=0

            NAT = form.cleaned_data.get("NAT")

            Yugoslav,German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0,0    

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
                Yugoslav = 1 

            PROF = form.cleaned_data.get("PROF")  

            State_Steel_Ind,Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
            ,Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0,0    

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
                State_Steel_Ind = 1 

            CAR = form.cleaned_data.get("CAR")   

            Without_Vehicle,Car,Car_and_Motor_bi= 0,0,0    

            if CAR=='Car':
                Car=1
            elif CAR=='Car_and_Motor_bi':
                Car_and_Motor_bi=1
            else:
                Without_Vehicle= 1    

            Cheque_card,no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
            ,Other_credit_car, American_Express = 0,0,0,0,0,0,0  

            CARDS = form.cleaned_data.get("CARDS")  

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
                Cheque_card = 1  



            # inputs1 = [H, R, E, G, T, U, V, W,Radio_TV_Hifi, Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT,Owner, Lease\
            # ,Yugoslav, German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue,Others, Civil_Service_M ,State_Steel_Ind, Self_employed_pe\
            # , Food_Building_Ca, Chemical_Industr, Pensioner ,Sea_Vojage_Gast, Military_Service,Without_Vehicle, Car,Car_and_Motor_bi\
            # ,Cheque_card, no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others, Other_credit_car, American_Express]

            inputs1 = [H, R, E, G, T, U, V, W, Cars, Dept_Store_Mail, Furniture_Carpet, Leisure, OT, Radio_TV_Hifi, Lease, Owner  
            , German, Greek, Italian, Other_European, RS, Spanish_Portugue, Turkish, Yugoslav, Chemical_Industr,  Civil_Service_M 
            , Food_Building_Ca, Military_Service, Others, Pensioner, Sea_Vojage_Gast, Self_employed_pe, State_Steel_Ind  
            , Car, Car_and_Motor_bi, Without_Vehicle, American_Express, Cheque_card, Mastercard_Euroc, Other_credit_car, VISA_Others  
            , VISA_mybank, no_credit_cards]
            
            inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, LOCATION, LOANS\
            , REGN, DIV, CASH]    

            list_ = inputs2 + inputs1

            inputs = np.array([list_]).reshape(1,-1)           
            answer = d.dt_pruned_tree(0, inputs, x_test, y_test, ccpalpha, threshold_1, threshold_2)[2]  
            print(answer)

    else:
        form = In()
        side_bar = Si()

    return render(request, 'decision.html', {'form':form, 'answer':answer})
