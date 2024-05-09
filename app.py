import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.markdown('<div class="full-page">', unsafe_allow_html=True)

with st.container():
    # Title
    st.title("Predictive Maintenance of Robotic Systems")

    # Collect user inputs
    types=[]
    input_values=[]
    air_temp=st.text_input("Enter Air Temperature(K): ")
    process_temp=st.text_input("Enter Process Temperature(K): ")
    rot_speed=st.text_input("Enter Rotational Speed(rpm): ")
    tor=st.text_input("Enter the Torque(Nm): ")
    tool=st.text_input("Enter Tool Wear(min): ")
    twf=st.text_input("Enter Tool Wear Failure(0/1): ")
    hdf=st.text_input("Enter Heat Dissipation Failure(0/1): ")
    pwf=st.text_input("Enter Power Failure(0/1): ")
    osf=st.text_input("Enter Overstrain Failure(0/1): ")
    rnf=st.text_input("Enter Random Failure(0/1): ")
    typ=st.text_input("Enter Type(H/L/M): ")
    if typ=='H':
       types=[1,0,0]
    elif typ=='L':
       types=[0,1,0]
    elif typ=='M':
       types=[0,0,1] 
    input_values=[air_temp,process_temp,rot_speed,tor,tool,twf,hdf,pwf,osf,rnf]+types
    # Button to trigger prediction
    if st.button("Predict"):
        # Convert input values to an array (assuming numeric inputs)
        input_array = np.array([float(val) for val in input_values]).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(input_array)
        if prediction==1:
           st.write("Machine Failure")
        else:
           st.write("Machine is Safe for Use") 
        

st.markdown('</div>', unsafe_allow_html=True)
