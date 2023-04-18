from django.shortcuts import render

from django.shortcuts import render, redirect
from django.contrib import messages
import sys
import os
sys.path.append('/home/humbulani/New/django_project/')
#sys.path.append('/home/humbulani/PD/django_project/django_project')
print(sys.path)
from .forms import In, Si

from django.templatetags.static import static
url = static('KGB.sas7bdat')
print(os.getcwd())

# from django_project.settings import BASE_DIR
# path = '/home/humbulani/PD/django_project/deploy/static/KGB.sas7bdat'

#import streamlit as st 
import ED
# df_loan = ED.Data_download(file_path = path)
# print(df_loan)
import train_test
#import Model_Perf
import matplotlib.pyplot as plt
#import GLM_Bino
import warnings
import missing_adhoc
import pandas as pd
#from PIL import Image
#import clustering
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#import Diagnostics
from scipy.stats import norm
import pylab
#import statsmodels.stats.diagnostic as sd
#from statsmodels.stats.stattools import durbin_watson
#import statsmodels.api as sm
import scipy
from scipy import stats
from math import *
from sklearn.tree import DecisionTreeClassifier
import train_test1
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import numpy as np
import Decision_tree
#

# def result(request):

#     return render(request, 'results.html', {'answer': answer})


def tree(request):

    global AGE, TITLE, list_1, answer
    AGE = 8
    TITLE = 6
    list_=[]
    answer=''


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



            inputs1 = [R, H, W, V, U, G, E, T,Radio_TV_Hifi, Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT,Owner, Lease\
            ,Yugoslav, German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue,Others, Civil_Service_M ,State_Steel_Ind, Self_employed_pe\
            , Food_Building_Ca, Chemical_Industr, Pensioner ,Sea_Vojage_Gast, Military_Service,Without_Vehicle, Car,Car_and_Motor_bi\
            ,Cheque_card, no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others, Other_credit_car, American_Express]
            
            inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, LOCATION, LOANS\
            , REGN, DIV, CASH]    

            list_ = inputs2 + inputs1

            inputs = pd.Series(list_)  
            
            answer = Decision_tree.Predict_binary_DT(Decision_tree.DT_Classification_fit, inputs.values.reshape(1,-1), train_test1.Y_test\
                         , train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=Decision_tree.ideal_ccp_alpha)    


            #return redirect('deployment') 

    else:
        form = In()
        side_bar = Si()

    return render(request, 'decision.html', {'form':form,'side_bar':side_bar,'answer':answer})

    
	#inputs(request.POST)

# def Sides(request):

#     if request.method == 'POST':
#         side_bar = Side(request.POST)
#         # if side_bar.is_valid():



#     else:
#         side_bar = Side()


#     return render(request, 'side.html', {'side_bar':side_bar,'AGE':AGE,'TITLE':TITLE,'list_':list_,'answer':answer})


