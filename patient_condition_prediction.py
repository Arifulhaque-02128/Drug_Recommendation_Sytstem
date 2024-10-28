# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import numpy as np
import pickle
import streamlit as slit
#from streamlit_option_menu import option_menu
#import pandas as pd
from utils import top_drugs_extractor




#loading the saved model
patient_condition_model = pickle.load(open('C:/Users/User/Desktop/ML_Projects/Classify_patient_condition_using_drug_reviews/patient_condition_prediction/patient_condition_model.sav', 'rb'))
bigram_tfidf_vectorizer = pickle.load(open('C:/Users/User/Desktop/ML_Projects/Classify_patient_condition_using_drug_reviews/patient_condition_prediction/bigram_tfidf_vect.sav', 'rb'))



def patient_condition_prediction (review) :
    vectorizer = bigram_tfidf_vectorizer.transform(review)
    pred = patient_condition_model.predict(vectorizer)
    
    return pred

def main() :
    slit.markdown(
    "<h1 style='text-align: center; color: #00c8ff;'>Patient Condition and Drug Recommendation</h1>",
    unsafe_allow_html=True)
    
    drug_review = slit.text_area('Enter Medical Issue for getting Drug Recommendation')
    
    #code for prediction
    patient_condition = ''
    
    # creating a button for prediction
    
    if slit.button('Predict') : 
        patient_condition = patient_condition_prediction([drug_review])
        
        slit.markdown(
        "<h5>Predicted Medical Condition : </h5>",
        unsafe_allow_html=True)
        slit.success(f"{patient_condition[0]}")
        
        
        slit.markdown(
        "<h5>Top 3 Recommended Drugs : </h5>",
        unsafe_allow_html=True)
        for drug in top_drugs_extractor(patient_condition[0]) : 
            slit.success(f"{drug}")
        
        
        
    
        
        


if __name__ == '__main__' :
    main()
