# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('Training Dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Handling missing values
# Impute missing values for numerical features with the mean
num_features = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
imputer = SimpleImputer(strategy='mean')
data[num_features] = imputer.fit_transform(data[num_features])

# Impute missing values for categorical features with the most frequent value
cat_features = ['Gender', 'Married', 'Dependents', 'Self_Employed']
imputer = SimpleImputer(strategy='most_frequent')
data[cat_features] = imputer.fit_transform(data[cat_features])

# Encode categorical variables
# Label Encoding for binary categorical features
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Married'] = le.fit_transform(data['Married'])
data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])  # Target variable

# One-Hot Encoding for other categorical features
data = pd.get_dummies(data, columns=['Dependents', 'Education', 'Property_Area'], drop_first=True)

# Feature Scaling
scaler = StandardScaler()
scaled_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
data[scaled_features] = scaler.fit_transform(data[scaled_features])

# Splitting the dataset into training and testing sets
X = data.drop('Loan_Status', axis=1)  # Features
y = data['Loan_Status']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the train and test sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
