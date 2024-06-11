import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline

# Load sample dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# Assume the dataset has the following columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Display the first few rows of the dataset
print("Initial DataFrame:")
print(df.head())

# 1. Handling Missing Values
# For demonstration, we'll introduce some missing values
df.loc[0, 'sepal_length'] = np.nan
df.loc[2, 'petal_width'] = np.nan

# Define numerical and categorical columns
num_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
cat_features = ['species']

# Impute missing values
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# Split the data into training and test sets
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Transform the test data
X_test_preprocessed = preprocessor.transform(X_test)

# Convert the preprocessed data back to a DataFrame for inspection
X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=preprocessor.get_feature_names_out())
X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=preprocessor.get_feature_names_out())

print("\nPreprocessed Training DataFrame:")
print(X_train_preprocessed_df.head())

print("\nPreprocessed Test DataFrame:")
print(X_test_preprocessed_df.head())

# 2. Feature Engineering
# Example: Create a new feature "sepal_area" from "sepal_length" and "sepal_width"
df['sepal_area'] = df['sepal_length'] * df['sepal_width']

# Example: Create interaction features
df['petal_length_width_ratio'] = df['petal_length'] / df['petal_width']

print("\nDataFrame with New Features:")
print(df.head())

# Example Pipeline including Feature Engineering
feature_engineering = Pipeline(steps=[
    ('feature_eng', FunctionTransformer(lambda x: x.assign(sepal_area=x['sepal_length'] * x['sepal_width'],
                                                           petal_length_width_ratio=x['petal_length'] / x['petal_width']), validate=False))
])

# Apply the feature engineering pipeline
X_train_fe = feature_engineering.fit_transform(X_train)
X_test_fe = feature_engineering.transform(X_test)

print("\nFeature Engineered Training DataFrame:")
print(X_train_fe.head())

print("\nFeature Engineered Test DataFrame:")
print(X_test_fe.head())
