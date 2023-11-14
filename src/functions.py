from datetime import datetime
import time
import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
    
fare_model = load('models/average_fare_model.joblib')
airline_model = load('models/airline_model.joblib')
refund_model = load('models/refund_model.joblib')
layover_model = load('models/layover_model3.9.joblib')

def predict(origin_airport, destination_airport, departure_datetime, cabin_type, search_datetime=None):
    if not(search_datetime):
        search_datetime = datetime.now()
    data = {
        'searchDate': search_datetime,
        'flightDate': departure_datetime,
        'startingAirport': origin_airport,
        'destinationAirport': destination_airport,
        'segmentsCabinCode': cabin_type
    }
    df = pd.DataFrame([data])
    df = flightDate_process(df)

    df_refund = df.copy()
    df_layover = df.copy()
    df_airline = df.copy()
    df_fare = df.copy()

    with st.spinner('Loading the predictions...'):

        df_refund = refund_predict(df_refund)
        df_layover = layover_predict(df_layover)
        df_airline = airline_predict(df_airline)
        df_fare = fare_predict(df_fare)

    return df_refund, df_layover, df_airline, df_fare

def flightDate_process(df):
    df['flightDate'] = pd.to_datetime(df['flightDate'])
    
    df['flightYear'] = df['flightDate'].dt.year
    df['flightMonth'] = df['flightDate'].dt.month
    df['flightWeek'] = df['flightDate'].dt.isocalendar().week
    df['flightDay'] = df['flightDate'].dt.day
    df['departureHour'] = df['flightDate'].dt.hour
    df['departureMinutes'] = df['flightDate'].dt.minute
    return df
    
def airline_process(df):
    df['searchDate'] = pd.to_datetime(df['searchDate'])
    df['searchGap'] = (df['flightDate']-df['searchDate']).dt.days

    cabin_weights = {'coach': 1, 'premium coach': 2, 'business': 3, 'first': 4}
    df['cabinWeight'] = df['segmentsCabinCode'].replace(cabin_weights)

    df_airline_tt = pd.read_csv('data/airline_timetable.csv')
    df = df.join(df_airline_tt.set_index(['startingAirport', 'destinationAirport', 'departureHour']), on=['startingAirport', 'destinationAirport', 'departureHour'])
    df.reset_index(drop=True, inplace=True)

    df = pivot_airline(df, 'segmentsAirlineName')
    
    unique_airlines = pivot_airline(df_airline_tt, 'segmentsAirlineName', find_unique=True)
    unavailable_airlines = set(unique_airlines) - set(df.columns)
    
    for airline in unavailable_airlines:
       df[airline] = 0

    return df

def pivot_airline(df, col_name, find_unique=False):
    df_split = pd.DataFrame({})
    df_split[col_name] = df[col_name].str.split('\|\|')
    df_split.reset_index(drop=True, inplace=True)
    df_exploded = df_split.explode(col_name, ignore_index=False)
    if find_unique:
        unique_airlines = df_exploded[col_name].unique()
        return unique_airlines
    pivot_table = df_exploded.pivot_table(index=df_exploded.index, columns=col_name, aggfunc='size', fill_value=0)
    df = pd.concat([df, pivot_table], axis=1)
    return df

def airline_predict(df):
    display_cols = ['segmentsAirlineName', 'predictedFare']
    df = airline_process(df)
    df['predictedFare'] = airline_model.predict(df).round(2)
    return df[display_cols]

# Mahjabeen's code start
#Create a function to return the features to be sent for the prediction
def format_features(Origin_Airport,Destination_Airport,Cabin_Type,departure_datetime:datetime):
    return {
       'Origin_Airport': [Origin_Airport],
        'Destination_Airport': [Destination_Airport],
        'segmentsCabinCode': [Cabin_Type],
        'Departure_Year': [departure_datetime.year],
        'Departure_Month': [departure_datetime.month],
        'Departure_Day': [departure_datetime.day],
        'Departure_Hour': [departure_datetime.hour],
        'Departure_Minute': [departure_datetime.minute],
       'isNonStop': 1,
       'layover': 0
    }

# Create a function to make predictions
def make_prediction(Origin_Airport, Destination_Airport,departure_datetime, Cabin_Type):
      features = format_features(
        Origin_Airport,
        Destination_Airport,
        Cabin_Type,
        departure_datetime
      )
       
      input = pd.DataFrame(features)
      prediction = layover_model.predict(input)
      return prediction
selected_columns1 = ['segmentsCabinCode']
df_Cabin_Type = pd.read_csv('data/raw/Final_cleaned_Dataset7.csv', usecols=selected_columns1)

# Create a list of unique cabin types
cabin_types = ['Select Cabin Type'] + df_Cabin_Type['segmentsCabinCode'].unique().tolist()

def layover_predict(df):
    display_cols = ['isNonStop', 'predictedFare']
    df['isNonStop'] = 0
    df = pd.concat([df, df.assign(isNonStop=1)], ignore_index=True)
    df['predictedFare'] = layover_model.predict(df).round(2)
    return df[display_cols]

def refund_predict(df):
    display_cols = ['isRefundable', 'predictedFare']
    df['isRefundable'] = 0

    df = pd.concat([df, df.assign(isRefundable=1)], ignore_index=True)
    df['predictedFare'] = refund_model.predict(df).round(2)
    return df[display_cols]

def fare_predict(df):
    display_cols = ['predictedFare']
    prediction = fare_model.predict(df)
    df['predictedFare'] = round(prediction[0], 2)
    return df[display_cols]
