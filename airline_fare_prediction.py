import streamlit as st
import requests
import pandas as pd
from datetime import datetime, date
from src import functions

refund_df = pd.DataFrame({})
coach_df = pd.DataFrame({})
airline_df = pd.DataFrame({})
smit_df = pd.DataFrame({})

kwargs = {}

def validate_data(origin_airport, destination_airport, departure_date, departure_time, cabin_type, search_date):
    if not(origin_airport):
        st.error('You have not entered the origin airport code')
        return
    
    if not(destination_airport):
        st.error('You have not entered the destination airport code')
        return
    
    if origin_airport == destination_airport:
        st.error('Destination airport cannot be same as origin airport')
        return
    
    if departure_date<=search_date:
        st.error('Departure date cannot be today or in the past')
        st.warning('Please enter a search date for testing purpose. This option will be disabled in production')
        return
    
    display_output(origin_airport, destination_airport, departure_date, departure_time, cabin_type, search_date)

def display_output(origin_airport, destination_airport, departure_date, departure_time, cabin_type, search_date):
    departure_datetime = datetime.combine(departure_date, departure_time)
    search_datetime = datetime.combine(search_date, departure_time)
    df_refund, df_layover, df_airline, df_fare = functions.predict(origin_airport, destination_airport, departure_datetime, cabin_type, search_datetime)
    tab1, tab2, tab3, tab4 = st.tabs(["Average fare prediction", "Airline fare predictions", "Layover fare predictions", "Refundable vs Non-refundable fare prediction"])
    with tab1:
        st.dataframe(df_fare, hide_index=True)

    with tab2:
        st.dataframe(df_airline, hide_index=True)

    with tab3:
        st.write('This model is in progress and will be available soon')
        #st.dataframe(df_layover, hide_index=True)

    with tab4:
        st.dataframe(df_refund, hide_index=True)
    
# Title
st.header('Data Product with Machine Learning')

with st.container():
    cn1_col1, cn1_col2, cn1_col3 = st.columns(3)
    with cn1_col1:
        origin_airport = st.text_input('Origin airport', placeholder='Starting airport code')
    with cn1_col2:
        destination_airport = st.text_input('Destination airport', placeholder='Ending airport code')
    with cn1_col3:
        cabin_options = ['coach', 'premium coach', 'business', 'first']
        cabin_type = st.selectbox('Cabin type', options=cabin_options, placeholder='Select a cabin')

with st.container():
    cn2_col1, cn2_col2, cn2_col3 = st.columns(3)
    
    with cn2_col1:
        departure_date = st.date_input('Departure date')
    with cn2_col2:
        departure_time = st.time_input('Departure time',step=60)
    with cn2_col3:
        search_date = st.date_input('Search date')
        
with st.container():
    predict_button = st.button('Predict', type='primary')

if predict_button:
    validate_data(origin_airport, destination_airport, departure_date, departure_time, cabin_type, search_date)

