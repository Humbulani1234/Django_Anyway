from django import forms
#from django.contrib.auth.models import User
from django.db.models.fields import BLANK_CHOICE_DASH

class Inputs(forms.Form):

    # Float categories
    # 
	NAME = forms.CharField()
	AGE = forms.FloatField()
	CHILDREN = forms.FloatField()
	PERS_H =  forms.FloatField()
	TMADD = forms.FloatField()
	TMJOB1 = forms.FloatField() 
	TEL =  forms.FloatField()
	NMBLOAN = forms.FloatField() 
	FINLOAN = forms.FloatField()
	INCOME =  forms.FloatField()
	EC_CARD = forms.FloatField()
	INC =  forms.FloatField()
	INC1 = forms.FloatField()
	BUREAU = forms.FloatField() 
	LOCATION = forms.FloatField()
	LOANS = forms.FloatField()
	REGN =forms.FloatField()
	DIV = forms.FloatField() 
	CASH = forms.FloatField() 

	# Categorical categories     

	TITLE = forms.CharField(label='TITLE', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('H','H'), ('R','R')]))
	STATUS = forms.CharField(label='STATUS', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('V','V'), ('U','U'),('G','G')\
		,('E','E'),('T','T'),('W','W')]))
	PRODUCT = forms.CharField(label='PRODUCT', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('Radio_TV_Hifi','Radio_TV_Hifi')\
		,('Furniture_Carpet','Furniture_Carpet'), ('Dept_Store_Mail','Dept_Store_Mail'), ('Leisure','Leisure')\
		,('Cars','Cars'), ('OT','OT')]))
	RESID = forms.CharField(label='RESID', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('Lease','Lease'),('Owner','Owner')]))
	NAT = forms.CharField(label='NAT', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('German','German'), ('Turkish','Turkish')\
		, ('RS','RS'), ('Greek','Greek'),('Yugoslav','Yugoslav'), ('Italian','Italian'), ('Other_European','Other_European')\
		,('Spanish_Portugue','Spanish_Portugue')]))
	PROF = forms.CharField(label='PROF', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('Others','Others'),('Civil_Service_M','Civil_Service_M')\
	 ,('Self_employed_pe','Self_employed_pe'), ('Food_Building_Ca','Food_Building_Ca'),('Chemical_Industr','Chemical_Industr')\
	 , ('Pensioner','Pensioner'),('Sea_Vojage_Gast','Sea_Vojage_Gast'), ('State_Steel_Ind,','State_Steel_Ind,')\
	 ,('Military_Service','Military_Service')]))
	CAR = forms.CharField(label='CAR', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('Car','Car'), ('Without_Vehicle','Without_Vehicle')\
		, ('Car_and_Motor_bi','Car_and_Motor_bi')]))
	CARDS = forms.CharField(label='CARDS', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('Cheque_card','Cheque_card')\
	 ,('no_credit_cards','no_credit_cards'), ('Mastercard_Euroc','Mastercard_Euroc'), ('VISA_mybank','VISA_mybank')\
	 ,('VISA_Others','VISA_Others'), ('Other_credit_car','Other_credit_car'), ('American_Express','American_Express')]))

class Side(forms.Form):

	Diagnostics = forms.CharField(label='Diagnostics', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('H','H'), ('R','R')]))
	Datasets = forms.CharField(label='Datasets', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('H','H'), ('R','R')]))
	Visualizations = forms.CharField(label='Visualizations', widget=forms.Select(choices=BLANK_CHOICE_DASH+[('H','H'), ('R','R')]))



