#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:46:36 2024

@author: amynguyen
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
s = pd.read_csv("social_media_usage.csv")

# Define function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Data prep
ss = pd.DataFrame({
    'sm_li': clean_sm(s['web1h']),  # Target variable: LinkedIn user
    'income': np.where(s['income'] <= 9, s['income'], np.nan),  # Income
    'education': np.where(s['educ2'] <= 8, s['educ2'], np.nan),  # Education
    'parent': clean_sm(s['par']),  # Parent (binary)
    'married': clean_sm(s['marital']),  # Married (binary)
    'female': clean_sm(s['gender']),  # Female (binary)
    'age': np.where(s['age'] <= 98, s['age'], np.nan)  # Age
}).dropna()  # Drop missing values

# Feature selection
X = ss[["income", "education", "parent", "married", "female", "age"]]
y = ss["sm_li"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, stratify=y, test_size=0.2, random_state=38
)

# Train logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Streamlit UI
st.title("LinkedIn User Prediction")

# Input form
st.header("Enter User Information")
income = st.slider("Income Level (1-9)", min_value=1, max_value=9, value=5)
education = st.slider("Education Level (1-8)", min_value=1, max_value=8, value=3)
parent = st.selectbox("Parent?", ["No", "Yes"])
married = st.selectbox("Married?", ["No", "Yes"])
female = st.selectbox("Female?", ["No", "Yes"])
age = st.slider("Age", min_value=18, max_value=98, value=33)

# Convert inputs to model format
parent = 1 if parent == "Yes" else 0
married = 1 if married == "Yes" else 0
female = 1 if female == "Yes" else 0

person = [income, education, parent, married, female, age]

# Make prediction
predicted_class = lr.predict([person])[0]
probs = lr.predict_proba([person])

# Output prediction
st.subheader("Prediction")
if predicted_class == 1:
    st.success("This is a LinkedIn user.")
else:
    st.error("Not a LinkedIn user")

# Display probabilities
st.subheader("Prediction Probabilities")
st.write(f"Probability of being a LinkedIn user: {probs[0][1]:.2f}")
st.write(f"Probability of not being a LinkedIn user: {probs[0][0]:.2f}")



# pwd

# cd /Users/amynguyen/Downloads/Georgetown/Programming_II/Final_Project
# streamlit run Nguyen_Final.py
# ctrl c