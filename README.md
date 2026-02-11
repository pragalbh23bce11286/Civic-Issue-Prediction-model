# Civic-Issue-Prediction-model

This project implements a machine learning model to predict the estimated time (in days) required to resolve civic issues based on complaint severity, category, historical frequency, and required resources. The goal is to help civic authorities prioritize issues, allocate resources efficiently, and improve response times.

# Features Used:

Severity_Score – Numeric score representing the seriousness of the issue

Complaint_Category – Type of civic issue (e.g., Drainage, Roads, Electricity)

Historical_Frequency – Number of similar complaints reported previously

Required_Resources – Resources needed for resolution (e.g., Workers, Machinery)

Target Variable:

Estimated_Resolution_Time_Days – Number of days required to resolve the issue

# Model Overview:

Algorithm: Random Forest Regressor

Problem Type: Regression

Frameworks Used: Scikit-learn, Pandas, NumPy

The model uses a preprocessing pipeline that:

Standardizes numerical features using StandardScaler

Encodes categorical features using OneHotEncoder

Combines preprocessing and model training using Pipeline

# Project Workflow

Load the dataset from a CSV file

Remove rows with missing values

Split data into training and testing sets (80/20 split)

Preprocess numerical and categorical features

Train a Random Forest regression model

Predict resolution time for unseen data
