#***********************************************************************************************
# Project : AudioShield : Leveraging Machine Learning to Detect Deepfake Voices
# Task    : Model Deployment and Prediction 
#***********************************************************************************************
#==============================================================================================
# 1. Import the required libraries
#==============================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import calendar
from datetime import datetime
import datetime
import os 
import pickle
import streamlit as st

import tensorflow as tf
import keras
from keras.models import load_model
from sklearn import preprocessing 

import librosa, librosa.display
import librosa
import librosa.display
import IPython.display as ipd
#==================================================================================================

#----------------------------------------------------------------------------
# 2. Load the saved ML model & Variables
#----------------------------------------------------------------------------
model = load_model('Omdena_DeepfakeAudio_Classification_CNN.h5')   # trained ML model

SAMPLE_RATE = pickle.load(open('sample_rate.pkl','rb'))       # Sample rate  
DURATION = pickle.load(open('duration.pkl','rb'))             # Duration of audio clips in seconds to be considered 
N_MELS = pickle.load(open('n_mels.pkl','rb'))                 # Number of Mel frequency bins
HOP_LENGTH = pickle.load(open('hop_length.pkl','rb'))         # the number of samples between successive frames
max_time_steps = pickle.load(open('max_time_steps.pkl','rb')) # maximum time steps

#----------------------------------------------------------------------------
# 3. Creating a function for input data transformation & forecasting
#----------------------------------------------------------------------------
def Forecasting(file_path):

    X_new = []

    # Load audio file 
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Extract Mel spectrogram 
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < max_time_steps:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_time_steps]

    X_new.append(mel_spectrogram)

    # Convert list to numpy array
    X_new = np.array(X_new)

    # Predict for testing data
    y_pred_new = model.predict(X_new)

    # Convert probabilities to predicted classes
    y_pred_new_class = np.argmax(y_pred_new, axis=1)

    if y_pred_new_class==1:
        prediction ="Bonafide"
    else:
        prediction ="Spoof"
        
    return prediction
    
#========================================================================================================
# Streamlit related
#========================================================================================================

#-----------------------------------------------------------------------------------------------
# 1. Add logos
#-----------------------------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.image("omdena.png", width=200)
   
#-----------------------------------------------------------------------------------------------
# 2. Titel & Goal 
#-----------------------------------------------------------------------------------------------
st.header('Project: :blue[Leveraging AI to Detect Deepfake Voices]')
st.subheader("Deepfake Audio Detection:", divider='rainbow')

#-----------------------------------------------------------------------------------------------
# 3. Take input data file from user
#-----------------------------------------------------------------------------------------------
st.write("Please upload the audio file to be detected: ")

inputdata = st.file_uploader("upload audio file")

if inputdata is not None:
    file_path = inputdata

#-------------------------------------------------------------------------------------------------------
if st.button('Predict',type="primary"):
    
    # 1. Forecasting 
    prediction_ = Forecasting(file_path)
    
    #===========================================================================
    st.subheader("Prediction Result:", divider='rainbow')
    
    prediction_ = str(prediction_)
    
    if prediction_=="Bonafide":        
        st.write(f"Audio file is :  :green[{prediction_}]")
    else:
        st.write(f"Audio file is :  :red[{prediction_}]")
        