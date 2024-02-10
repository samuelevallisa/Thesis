import pandas as pd
import datetime as dt
from datetime import datetime
import numpy as np
import math
import torch
import torch.nn as nn

def make_data_windows(df_input, df_target, time_window_train,time_window_trgt, time_horizon,stride):  
    
  dates=df_input.index.date
  hours=df_input.index.hour
  last_date = dates[-1]
  last_hour = hours[-1]
  first_hour = hours[0]
  dates=np.unique(dates)
  hours=np.unique(hours)
  inputs=[]
  target_vector=[]
  dec_input=[]

  window_start_hour=first_hour
  i=0
  while True:
    #j=0
    #while True:
    date_input_start_str=dates[i].strftime('%Y-%m-%d')
    #window_start_hour=#hours[j]
    window_start_hour = window_start_hour%24
    window_end_hour = (window_start_hour+time_window_train+time_horizon+time_window_trgt)%24
    if window_start_hour < 10:
        window_start_hour_str = ' 0'+str(window_start_hour)+':00:00'
    else:
        window_start_hour_str = ' '+str(window_start_hour)+':00:00'
    date_input_start_str+=window_start_hour_str
    #convert the date and time string of the input start to datetime 
    datetime_input_start = datetime.strptime(date_input_start_str, '%Y-%m-%d %H:%M:%S')

    #add time_window hours so to get date and time of input end 
    delta_hours = dt.timedelta(hours=time_window_train - 1)
    datetime_input_end = datetime_input_start + delta_hours
    delta_hours_dec_in = dt.timedelta(hours=time_window_trgt-1)
    
    #add time_horizon hours so to get date and time of target
    delta_hours_target = dt.timedelta(hours=time_horizon)
    datetime_dec_in_start = datetime_input_end + delta_hours_target
    datetime_dec_in_end = datetime_dec_in_start+ delta_hours_dec_in
    datetime_target_start = datetime_dec_in_start + dt.timedelta(hours=1)
    datetime_target_end = datetime_dec_in_end + dt.timedelta(hours=1)
    date_target = datetime_target_end.date()

    datetime_input_start_str = datetime_input_start.strftime('%Y-%m-%d %H:%M:%S')
    datetime_input_end_str = datetime_input_end.strftime('%Y-%m-%d %H:%M:%S')
    datetime_dec_in_end_str = datetime_dec_in_end.strftime('%Y-%m-%d %H:%M:%S')
    datetime_dec_in_start_str = datetime_dec_in_start.strftime('%Y-%m-%d %H:%M:%S')
    datetime_target_start_str = datetime_target_start.strftime('%Y-%m-%d %H:%M:%S')
    datetime_target_end_str = datetime_target_end.strftime('%Y-%m-%d %H:%M:%S')

    in_seq = df_input.loc[ datetime_input_start_str : datetime_input_end_str] #.to_numpy()
    dec_in_seq = df_target.loc[ datetime_dec_in_start_str : datetime_dec_in_end_str]
    #dec_in_seq.values[0]=0.0
    target_seq = df_target.loc[ datetime_target_start_str : datetime_target_end_str]
    inputs.append(in_seq)
    dec_input.append(dec_in_seq)
    target_vector.append(target_seq)
    
        # if window_start_hour>=last_hour:
        #   break
        # j+=1
        # if date_target >= last_date and window_end_hour>=last_hour :
        #   break
    if date_target >= last_date and window_end_hour>=last_hour :
      break
    if window_start_hour+stride>23:
      i+=1
    window_start_hour+=stride
  encoder_input=np.stack(inputs)
  decoder_input=np.stack(dec_input)
  target=np.stack(target_vector)
  for i in range(len(decoder_input)):
      decoder_input[i][0]=0.0
  return encoder_input,decoder_input,target


def Data_Norm(dataset,feature_list):
    
    for name in feature_list:
        min = np.min(dataset[name])
        max = np.max(dataset[name])
        dataset[name]=(dataset[name]-min)/()
    return dataset 

def Data_creation(df):
    #df = pd.read_csv(data_dir1)
    #indexNames = df[ (df['Active_Energy_Delivered_Received'] <= 0.1)].index
    #df.drop(indexNames , inplace=True)
    #df.Timestamp = df.Timestamp.astype(np.datetime64) #set 'Timestamp' to np.datetime type
    #df = df.set_index('Timestamp') # set 'Timestamp' column as index
    #df = df.resample('H').first() #Fill all the missing timesteps with Nan 
    #df = df.interpolate()


    #covariates = pd.read_csv(data_dir2)
    #covariates.index=pd.to_datetime(covariates['Timestamp'],format="%Y/%m/%d")
    df.index=pd.to_datetime(df['time'],format="%Y/%m/%d")
    df['time']=df.index
    df['hour']=df.index.hour
    #covariates['min']=covariates.index.minute
    #covariates['dayofweek']=covariates.index.day_of_week
    df['month']=df.index.month
    df['dayofyear']=df.index.day_of_year
    df.time = df.time.astype(np.datetime64) #set 'Timestamp' to np.datetime type
    df = df.set_index('time') # set 'Timestamp' column as index
    # df = df.resample('H').first() #Fill all the missing timesteps with Nan 
    # df = df.interpolate()

    #df_total = pd.concat([df,covariates],axis=1)
    # df_total=df_total.rename(columns={"Wind_Speed":"WindSpeed",
    #             'Weather_Temperature_Celsius':'Temperature',
    #             'Global_Horizontal_Radiation': 'GHR',
    #             'Wind_Direction':'WindDir',
    #             'Daily_Rainfall' : 'DailyRain',
    #             'Max_Wind_Speed':'MaxWindSpeed',
    #             'Air_Pressure':'AirPressure',
    #             'Hail_Accumulation':'Hail_Accumulation'
    #             })
    df = df.drop(columns=['precipitation (mm)', 'rain (mm)',
       'snowfall (cm)','shortwave_radiation (W/m²)', 'direct_radiation (W/m²)',
       'diffuse_radiation (W/m²)', 'direct_normal_irradiance (W/m²)','soil_temperature_0_to_7cm (°C)',
       'soil_temperature_7_to_28cm (°C)', 'soil_temperature_28_to_100cm (°C)',
       'soil_temperature_100_to_255cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)',
       'soil_moisture_7_to_28cm (m³/m³)', 'soil_moisture_28_to_100cm (m³/m³)',
       'soil_moisture_100_to_255cm (m³/m³)','pressure_msl (hPa)','surface_pressure (hPa)'
       ,'vapor_pressure_deficit (kPa)','et0_fao_evapotranspiration (mm)', 'month','relativehumidity_2m (%)'
       ,'apparent_temperature (°C)'])

    train_data = df[df.index <= '2021-04-30 23:00:00' ] #circa il 90% del dataset
    test_data = df[df.index > '2021-04-30 23:00:00']

    feature_list=['temperature_2m (°C)', 'dewpoint_2m (°C)',
       'cloudcover (%)', 'cloudcover_low (%)',
       'cloudcover_mid (%)', 'cloudcover_high (%)','hour',
       'dayofyear']
    train_data = Data_Norm(train_data,feature_list=feature_list)
    test_data = Data_Norm(test_data,feature_list=feature_list)

    return train_data,test_data
