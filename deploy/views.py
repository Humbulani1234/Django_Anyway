import tracemalloc
tracemalloc.start()

import pandas as pd
import numpy as np
import sys
import pickle
import types
import io
import base64
import statsmodels.api as sm

from django.shortcuts import render, redirect
from django.contrib import messages
from pathlib import Path
from django.template import loader
from django.http import HttpResponse

from .forms import Inputs, Side

sys.path.append('/home/humbulani/New/django_project/refactored_pd')

import pd_download
from class_traintest import OneHotEncoding
from class_base import Base
from class_missing_values import ImputationCat
import class_diagnostics
from class_modelperf import ModelPerfomance

#-------------------------------------------------------------------Defined variables-----------------------------------------------------------

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
print(x_train.columns)

b = class_diagnostics.ResidualsPlot(custom_rcParams, x_test, y_test, threshold)
c = ModelPerfomance(custom_rcParams, x_test, y_test, threshold)

def image_generator(f):

    buffer = io.BytesIO()
    f.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return image_base64

#------------------------------------------------------------------ Performance measures---------------------------------------------------------

def roc(request):

    f = c.roc_curve_analytics()
    image_base64 = image_generator(f)                 

    return render (request, 'roc.html', {'image_base64':image_base64})

def confusion_logistic(request):

    f = c.confusion_matrix_plot()
    image_base64 = image_generator(f)

    return render (request, 'confusion_logistic.html', {'image_base64':image_base64})

#-------------------------------------------------------------------Model Diagnostics------------------------------------------------------------

def normal_plot(request):

    f = b.plot_normality_quantile()
    image_base64 = image_generator(f)

    return render (request, 'normal_plot.html', {'image_base64':image_base64})

def residuals(request):

    f = b.plot_quantile_residuals()
    image_base64 = image_generator(f)

    return render (request, 'residuals.html', {'image_base64':image_base64})

def partial(request):

    f = b.partial_plots_quantile()
    image_base64 = image_generator(f)

    return render (request, 'partial.html', {'image_base64':image_base64})

def student(request):

    f = b.plot_lev_stud_quantile()
    image_base64 = image_generator(f)

    return render (request, 'student.html', {'image_base64':image_base64})

def cooks(request):

    f = b.plot_cooks_dis_quantile()
    image_base64 = image_generator(f)

    return render (request, 'cooks.html', {'image_base64':image_base64})

#-------------------------------------------------------------------Home and Models------------------------------------------------------------------

def home(request):

    return render(request, 'home.html')

def inputs(request):

    #print(request.COOKIES)

    answer = ""

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
            
            TITLE = form.cleaned_data.get("TITLE")

            H = 0

            if TITLE == 'H':
                H=1
    
            else:
                H=0
            
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

            Cheque_card, Mastercard_Euroc, VISA_mybank,VISA_Others\
            ,Other_credit_car, American_Express = 0,0,0,0,0,0  

            CARDS = form.cleaned_data.get("CARDS")  

            if CARDS=='Cheque_card':
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
                Cheque_card, Mastercard_Euroc, VISA_mybank,VISA_Others\
                ,Other_credit_car, American_Express = 0,0,0,0,0,0  

            # inputs1 = [ H,V, U, G, E, T,Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT,Lease,German, Turkish, RS, Greek ,Italian
            #           , Other_European, Spanish_Portugue,Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr
            #           , Pensioner ,Sea_Vojage_Gast, Military_Service,Car,Car_and_Motor_bi,no_credit_cards, Mastercard_Euroc, VISA_mybank
            #           , VISA_Others, Other_credit_car, American_Express ] 

            inputs1 = [H, E, G, T, U, V, Cars, Dept_Store_Mail, Furniture_Carpet, Leisure, OT, Lease, German, Greek, 
            Italian, Other_European, RS, Spanish_Portugue, Turkish, Chemical_Industr, Civil_Service_M, 
            Food_Building_Ca, Military_Service, Others, Pensioner, Sea_Vojage_Gast, Self_employed_pe, Car, 
            Car_and_Motor_bi, American_Express, Cheque_card, Mastercard_Euroc, Other_credit_car, VISA_Others, VISA_mybank]
            
            inputs2 = [ 1, CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, 
                        LOCATION, LOANS, REGN, DIV, CASH ]    

            list_ = inputs2 + inputs1
            inputs = np.array(list_).reshape(1,-1)
            answer = np.array(loaded_model.predict(inputs.reshape(1,-1)))
            answer = "{: .10f}".format(answer[0])

    else:

        form = Inputs()
        side_bar = Side()

    return render(request, 'features.html', {'form':form, 'answer':answer})

print(tracemalloc.get_traced_memory())
tracemalloc.stop()

# ------------------------------------------------------------------Consider----------------------------------------------------------
