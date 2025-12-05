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

## 🏗️ MLOps Pipeline Architecture

