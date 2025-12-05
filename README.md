This repository contains a complete Machine Learning + MLOps pipeline for predicting whether a customer will purchase a tourism package.
The project includes:

✔ Model building with experimentation tracking using MLflow
✔ Hyperparameter tuning using GridSearchCV
✔ Model registration in the Hugging Face Model Hub
✔ Deployment using Streamlit + Docker + HuggingFace Spaces
✔ Automated training pipeline using GitHub Actions

Tourism_Package_Prediction/
│
├── data/
│   ├── tourism.csv
│   ├── Xtrain.csv
│   ├── Xtest.csv
│   ├── ytrain.csv
│   └── ytest.csv
│
├── model_building/
│   ├── dev_experiment.ipynb
│   └── train.py      ← Production model training script
│
├── deployment/
│   ├── app.py        ← Streamlit UI
│   ├── Dockerfile
│   ├── requirements.txt
│   └── host_to_hf.py ← Upload deployment files to HuggingFace Space
│
├── .github/
│   └── workflows/
│       └── pipeline.yml  ← CI/CD automation
│
└── README.md


Model Building & Experimentation Tracking

✔ Development Environment

The development notebook performs:

Data cleaning and preprocessing

Label encoding of categorical data

Feature scaling

Hyperparameter tuning using GridSearchCV

MLflow experiment tracking


Production Training Pipeline

The production script train.py:

Loads train & test data from Hugging Face Dataset Hub
Builds preprocessing (scaling + one-hot encoding)
 Trains XGBoost using hyperparameter tuning
 Logs evaluation metrics to MLflow
 Saves the best model
Uploads it to Hugging Face Model Hub

Model Hub Location

🔗 https://huggingface.co/Amitgupta2982/Tourism-Package-Model

Saving the best model as best_xgboost_tourism_dev.pkl

