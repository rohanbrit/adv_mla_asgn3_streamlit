from datetime import datetime
import time
import streamlit as st
import pandas as pd
from joblib import load
    
airline_model = load('models/airline_gb_pipe.joblib')
fare_model = load('models/flight_fare_model.joblib')

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

def airline_process(df):
    df['searchDate'] = pd.to_datetime(df['searchDate'])
    df['flightDate'] = pd.to_datetime(df['flightDate'])
    
    df['flightYear'] = df['flightDate'].dt.year
    df['flightMonth'] = df['flightDate'].dt.month
    df['flightWeek'] = df['flightDate'].dt.isocalendar().week
    df['departureHour'] = df['flightDate'].dt.hour
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
    #display_cols = ['segmentsAirlineName','startingAirport', 'destinationAirport', 'flightDate', 'segmentsCabinCode', 'predictedFare']
    display_cols = ['segmentsAirlineName', 'predictedFare']
    df = airline_process(df)
    df['predictedFare'] = airline_model.predict(df).round(2)
    return df[display_cols]

def layover_predict(df):
    #time.sleep(1)
    return df

def refund_predict(df):
    #time.sleep(1)
    return df

def fare_predict(df):
    display_cols = ['predictedFare']
    prediction = fare_model.predict(df)
    df['predictedFare'] = round(prediction[0], 2)
    return df[display_cols]
