from django.shortcuts import render, redirect
from django.contrib import messages
import sys
import os
from pathlib import Path

sys.path.append('/home/humbulani/New/django_project/PD')

from .forms import Inputs, Side

from django.templatetags.static import static

import Model_Perf
import pandas as pd
import numpy as np

def inputs(request):

    global AGE, TITLE, list_1, answer
    AGE = 8
    TITLE = 6
    list_=[]
    answer=''


    if request.method == 'POST':
        form = Inputs(request.POST)
        side_bar = Side(request.POST)
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
            H = 0
            if TITLE == 'H':
    	        H=1
    	        # list_.append(H)
            else:
    	        H=0
    	        # list_.append(H)
            #input_ = [H]
            #
            STATUS = form.cleaned_data.get("STATUS")

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

            PRODUCT = form.cleaned_data.get("PRODUCT") 

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

            RESID = form.cleaned_data.get("RESID")

            Lease = 0    

            if RESID=='Lease':
    	        Lease=1    

            else:
                Lease=0

            NAT = form.cleaned_data.get("NAT")

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

            PROF = form.cleaned_data.get("PROF")  

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

            CAR = form.cleaned_data.get("CAR")   

            Car,Car_and_Motor_bi= 0,0    

            if CAR=='Car':
                Car=1
            elif CAR=='Car_and_Motor_bi':
                Car_and_Motor_bi=1
            else:
                Car,Car_and_Motor_bi= 0,0    

            no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
            ,Other_credit_car, American_Express = 0,0,0,0,0,0  

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
                no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
                ,Other_credit_car, American_Express = 0,0,0,0,0,0  



            inputs1 = [H,V, U, G, E, T,Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT,Lease,German, Turkish, RS, Greek ,Italian\
             , Other_European, Spanish_Portugue,Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
             , Pensioner ,Sea_Vojage_Gast, Military_Service,Car,Car_and_Motor_bi,no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
             , Other_credit_car, American_Express]
            
            inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, LOCATION, LOANS\
             , REGN, DIV, CASH]    

            list_ = inputs2 + inputs1

            inputs = pd.Series(list_)  
            
            answer = np.round(Model_Perf.Prediction(GLM_Bino.GLM_Binomial_fit,inputs, train_test.X_train, train_test.Y_train),10)   


            #return redirect('deployment') 

    else:
        form = Inputs()
        side_bar = Side()

    return render(request, 'features.html', {'form':form,'side_bar':side_bar,'answer':answer})


# def Sides(request):

#     if request.method == 'POST':
#         side_bar = Side(request.POST)
#         # if side_bar.is_valid():



#     else:
#         side_bar = Side()


#     return render(request, 'side.html', {'side_bar':side_bar,'AGE':AGE,'TITLE':TITLE,'list_':list_,'answer':answer})

