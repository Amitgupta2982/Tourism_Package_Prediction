This repository implements a complete Machine Learning + MLOps workflow for predicting whether a customer will purchase a tourism package.

Business Objective

"Visit with Us" aims to automate and improve customer targeting for their new Wellness Tourism Package.

The objective is to:

1. Predict the likelihood of a customer purchasing the package before contacting them

2. Optimize marketing campaigns through targeted customer identification

3. Reduce manual effort and eliminate inconsistency in customer prioritization

4. Implement a scalable, repeatable, automated MLOps pipeline

5. Ensure continuous model improvement via CI/CD
   
Dataset Description

The dataset contains customer demographics and interaction data with 20 features:

Customer Details:

CustomerID, Age, Gender, MaritalStatus, CityTier, Occupation, Designation,MonthlyIncome, NumberOfPersonVisiting, NumberOfChildrenVisiting.
NumberOfTrips, Passport, OwnCar, PreferredPropertyStar.

Sales Interaction Data:

TypeofContact, DurationOfPitch, ProductPitched, NumberOfFollowups.PitchSatisfactionScore.

Target Variable:

ProdTaken (0: No purchase, 1: Purchase)

MLOps Pipeline Architecture 

Below is the complete automated pipeline implemented in this project:

1. Data Registration
   
tore raw dataset on HuggingFace Dataset Hub

Enable data versioning, reproducibility, and controlled access

Ensure downstream steps always fetch the correct dataset version

3. Data Preparation
   
Load data from HuggingFace Hub

Perform:

Missing value handling

Column cleanup (remove identifiers)

Encoding categorical variables

Feature engineering (optional)

Train/Test split (80/20)

Upload processed datasets back to HuggingFace

4. Model Building & Experimentation (Development Mode)
   
Model: XGBoost Classifier

Hyperparameter tuning using GridSearchCV

Track:

Parameters

Metrics

Artifacts using MLflow

Evaluate performance on train & test sets

5. Model Registration (Production Mode)
   
Save best-performing model as .joblib

Push model file to HuggingFace Model Hub

Model becomes versioned, reproducible, and accessible for deployment

6. Deployment
    
Streamlit Web App

User-friendly interface for real-time predictions

Runs inside a Docker container

HuggingFace Spaces Deployment

Automated hosting of:

Dockerfile

Streamlit app

Requirements

CI/CD ensures the app updates whenever new code is pushed

7. CI/CD Pipeline (GitHub Actions)

Automated steps include:

Dataset registration

Data preprocessing

Model training

Model evaluation

Model publishing to HuggingFace



Project Structure 


Tourism_Package_Prediction/
│
├── data/
│   └── tourism.csv
│
├── model_building/
│   ├── data_register.py
│   ├── prep.py
│   └── train.py
│
├── deployment/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
│
├── hosting/
│   └── hosting.py
│
└── requirements.txt

