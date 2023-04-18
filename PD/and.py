import streamlit as st 

NAME = st.text_input("Name of Customer")
AGE = st.slider("Enter", 0,100)
STATUS = st.selectbox("STATUS",options=['V','U','G','E','T'])


button_clicked = st.button('Submit')

def update_variables():

    global STATUS

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
	    		 else:
	    		     T=1

    print(f"Name: {NAME}, age: {AGE} , state:{STATUS}")

update_variables()