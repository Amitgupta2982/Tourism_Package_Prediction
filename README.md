# 🌍 Tourism Package Purchase Prediction – End-to-End MLOps Project

This repository implements a complete **Machine Learning + MLOps workflow** for predicting whether a customer will purchase a tourism package.

---

##  Business Objective

"Visit with Us" aims to automate and improve customer targeting for their new **Wellness Tourism Package**.

The objectives are:

1.  Predict the likelihood of a customer purchasing the package before contacting them  
2.  Optimize marketing campaigns through targeted customer identification  
3.  Reduce manual effort and remove inconsistency in customer prioritization  
4.  Implement a scalable, repeatable, automated MLOps pipeline  
5.  Ensure continuous model improvement via CI/CD  

---

##  Dataset Description

The dataset contains **customer demographics + interaction features** (20 total):

###  **Customer Details**
- CustomerID  
- Age  
- Gender  
- MaritalStatus  
- CityTier  
- Occupation  
- Designation  
- MonthlyIncome  
- NumberOfPersonVisiting  
- NumberOfChildrenVisiting  
- NumberOfTrips  
- Passport  
- OwnCar  
- PreferredPropertyStar  

###  **Sales Interaction Details**
- TypeofContact  
- DurationOfPitch  
- ProductPitched  
- NumberOfFollowups  
- PitchSatisfactionScore  

###  **Target Variable**
- **ProdTaken** → `1 = Purchased`, `0 = Not Purchased`

---

##  MLOps Pipeline Architecture

Raw Data → HF Dataset Hub
→ Data Prep (cleaning, encoding, splits)
→ MLflow Training (GridSearchCV + XGBoost)
→ Best Model Saved + Uploaded to Hugging Face
→ Dockerized Streamlit App
→ Deployed to Hugging Face Spaces
→ Automated CI/CD via GitHub Actions

---

##  **1. Data Registration (HuggingFace Hub)**

 Upload raw dataset  
 Enable dataset versioning  

---

##  **2. Data Preparation**

Performed tasks:

- Remove unnecessary columns  
- Encode categorical values  
- Train/Test split  
- Upload prepared datasets to HuggingFace  


---

##  **3. Model Building & Experiment Tracking**

Algorithm used:

- **XGBoost Classifier**

Includes:

- Hyperparameter tuning (`GridSearchCV`)
- MLflow experiment tracking
- Classification performance evaluation
- Best model registration to HuggingFace Model Hub  


---

##  **4. Deployment (Streamlit + Docker + HuggingFace Spaces)**

The application:

- Loads model directly from HuggingFace  
- Collects user inputs  
- Generates real-time predictions  

Deployment files:


Live App Link 🔗:  
 **https://huggingface.co/spaces/Amitgupta2982/Tourism-Package-App**

---

##  **5. CI/CD Pipeline (GitHub Actions)**

Automated steps:

 Dataset registration  
 Data preparation  
 Model training + MLflow logging  
 Model deployment  
 Push to HuggingFace Space  

---

## 📁 **Project Folder Structure**

```plaintext
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


Important Links
Component	Link
🗂 GitHub Repository	https://github.com/Amitgupta2982/Tourism_Package_Prediction

🤗 HuggingFace Space	https://huggingface.co/spaces/Amitgupta2982/Tourism-Package-App

HuggingFace Model Hub	https://huggingface.co/Amitgupta2982/Tourism-Package-Model

Conclusion

This project demonstrates a fully automated MLOps workflow for tourism package prediction:

✔ Automated data ingestion
✔ Dataset versioning
✔ MLflow experiment tracking
✔ Best model registry
✔ Containerized deployment
✔ CI/CD automation
✔ Production-ready prediction system







