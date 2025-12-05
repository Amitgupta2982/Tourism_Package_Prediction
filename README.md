This repository implements a complete Machine Learning + MLOps workflow for predicting whether a customer will purchase a tourism package.

Project Overview

"Visit with Us" tourism company's MLOps pipeline for predicting customer purchase likelihood of the Wellness Tourism Package. This end-to-end machine learning solution automates customer targeting through data-driven predictions.

Business Objective

Build an automated system that:

Predicts whether customers will purchase the Wellness Tourism Package
Optimizes marketing campaigns through targeted customer identification
Implements scalable MLOps practices for continuous model improvement
Reduces manual effort and improves campaign performance
Dataset Description
The dataset contains customer demographics and interaction data with 20 features:

Customer Details:

CustomerID, Age, Gender, MaritalStatus, CityTier, Occupation, Designation
MonthlyIncome, NumberOfPersonVisiting, NumberOfChildrenVisiting
NumberOfTrips, Passport, OwnCar, PreferredPropertyStar
Sales Interaction Data:

TypeofContact, DurationOfPitch, ProductPitched, NumberOfFollowups
PitchSatisfactionScore
Target Variable:

ProdTaken (0: No purchase, 1: Purchase)
MLOps Pipeline Architecture
1. Data Registration
Upload original dataset to HuggingFace Hub
Establish data versioning and accessibility
2. Data Preparation
Load data from HuggingFace Hub
Clean and handle missing values
Feature engineering (income categories, age groups)
Encode categorical variables
Split into train/test sets (80/20)
Upload processed datasets to HuggingFace
3. Model Building & Experimentation
Train multiple ML algorithms:
Decision Tree
Random Forest
Gradient Boosting
XGBoost
AdaBoost
Hyperparameter tuning with GridSearchCV
MLflow experiment tracking
Model evaluation and comparison
Register best model to HuggingFace Model Hub
4. Deployment
Containerized deployment with Docker
Streamlit web application for predictions
Automated deployment to HuggingFace Spaces
5. CI/CD Pipeline
GitHub Actions workflow automation
Automated testing and deployment
Continuous integration on main branch updates
