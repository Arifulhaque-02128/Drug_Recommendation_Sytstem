# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 00:03:23 2024

@author: User
"""

import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('C:/Users/User/Desktop/ML_Projects/Classify_patient_condition_using_drug_reviews/patient_condition_prediction/drugs_data.csv')

def top_drugs_extractor(condition):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst