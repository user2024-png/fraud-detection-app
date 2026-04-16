# Credit Card Fraud Detection System

## Description

This project develops a machine learning system to detect fraudulent credit card transactions. It applies data preprocessing, feature scaling, and classification algorithms to identify suspicious transactions. The final model is deployed using Streamlit, enabling users to perform real-time fraud predictions through an interactive web application.


## Objectives

* Build an accurate fraud detection model
* Compare multiple machine learning algorithms
* Evaluate performance using appropriate metrics
* Deploy the model as a web-based application



## Dataset

The dataset used is the Credit Card Fraud Detection dataset from Kaggle. It contains anonymized features (V1–V28), along with Time, Amount, and Class labels where Class indicates fraudulent or normal transactions. Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download



## Technologies Used

* Python
* Jupyter Notebook
* Scikit-learn
* Pandas and NumPy
* Matplotlib
* Streamlit



## Machine Learning Models

The following models were implemented:

* Logistic Regression
* Decision Tree
* Random Forest

Random Forest was selected as the final model due to its better performance in terms of precision, recall, and F1-score.



## Features

* Data preprocessing and scaling
* Exploratory Data Analysis (EDA)
* Model training and evaluation
* Performance visualization
* Real-time fraud prediction using Streamlit


## Deployment

The application is deployed using Streamlit Community Cloud and can be accessed via a web browser. Users can input transaction details and receive instant predictions about whether a transaction is fraudulent or not.



## Conclusion

This project demonstrates the effectiveness of machine learning techniques in detecting fraudulent transactions. The deployed system provides a practical solution for real-time fraud detection with high accuracy and reliability.
