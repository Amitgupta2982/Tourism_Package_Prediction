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

1. Data Registration
   
Upload original dataset to HuggingFace Hub

Establish data versioning and accessibility

3. Data Preparation
   
Load data from HuggingFace Hub

Clean and handle missing values

Feature engineering (income categories, age groups)

Encode categorical variables

Split into train/test sets (80/20)

Upload processed datasets to HuggingFace

5. Model Building & Experimentation
   
XGBoost

Hyperparameter tuning with GridSearchCV
MLflow experiment tracking
Model evaluation and comparison
Register best model to HuggingFace Model Hub

7. Deployment
   
Containerized deployment with Docker
Streamlit web application for predictions
Automated deployment to HuggingFace Spaces

9. CI/CD Pipeline
    
GitHub Actions workflow automation
Automated testing and deployment
Continuous integration on main branch updates
