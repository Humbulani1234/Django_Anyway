H = 0

if TITLE=='H':
    H == 1
#else:
    #H == 1

V, U, G, E, T = 0,0,0,0,0

if STATUS == 'V':
	V==1
elif STATUS == 'U':
	U==1
elif STATUS == 'G':
	G==1
elif STATUS == 'E':
	E==1
else:
	T==1

Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0

if PRODUCT=='Furniture_Carpet':
	Furniture==1
elif PRODUCT=='Dept_Store_Mail':
	Dept_Store_Mail==1
elif PRODUCT=='Leisure':
	Leisure==1
elif PRODUCT=='Cars':
	Cars==1
else:
	OT==1

Lease = 0

if RESID=='Lease':
	Lease==1

German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0

if NAT=='German':
	German==1
elif NAT=='Turkish':
	Turkish==1
elif NAT=='RS':
	RS==1
elif NAT=='Greek':
	Greek==1
elif NAT=='Italian':
	Italian==1
elif NAT=='Other_European':
	Other_European==1
else:
	Spanish_Portugue==1

Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
, Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0

if PROF=='Others':
	Others==1
elif PROF=='Civil_Service_M':
	Civil_Service_M==1
elif PROF=='Self_employed_pe':
	Self_employed_pe==1
elif PROF=='Food_Building_Ca':
	Food_Building_Ca==1
elif PROF=='Chemical_Industr':
	Chemical_Industr==1
elif PROF=='Pensioner':
	Pensioner==1
elif PROF=='Sea_Vojage_Gast':
	Sea_Vojage_Gast==1
else:
	Military_Service==1

Car,Car_and_Motor_bi= 0,0

if CAR=='Car':
	Car==1
else:
	Car_and_Motor_bi==1


no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
, Other_credit_car, American_Express = 0,0,0,0,0,0

if CARDS=='no_credit_cards':
	no_credit_cards==1
elif CARDS=='Mastercard_Euroc':
	Mastercard_Euroc==1
elif CARDS=='VISA_mybank':
	VISA_mybank==1
elif CARDS=='VISA_Others':
	VISA_Others==1
elif CARDS=='Other_credit_car':
	Other_credit_car==1
else:
	American_Express==1

inputs1 = [H,V, U, G, E, T,Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT,Lease,German, Turkish, RS, Greek ,Italian\
, Other_European, Spanish_Portugue,Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
, Pensioner ,Sea_Vojage_Gast, Military_Service,Car,Car_and_Motor_bi,no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
, Other_credit_car, American_Express]

#inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, LOCATION, LOANS\
#, REGN, DIV, CASH]

# inputs = inputs1 + inputs2
# print(len(inputs))
# print(len(train_test.X_test.columns.tolist()))
# print(train_test.X_test)

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