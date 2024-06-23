import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')

# Step 1: Display the first few rows of the dataset
print("Initial Data:")
print(titanic.head())

# Step 2: EDA
print("\nDataset Information:")
titanic.info()

print("\nDescriptive Statistics:")
print(titanic.describe())

print("\nMissing Values:")
print(titanic.isnull().sum())

# Step 3: Handling Missing Values
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
titanic.drop(columns=['deck'], inplace=True)
titanic.dropna(subset=['embark_town', 'alive'], inplace=True)

# Step 4: Feature Transformation and Engineering
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
titanic.drop(columns=['sibsp', 'parch', 'class', 'who', 'adult_male', 'embark_town', 'alive'], inplace=True)
titanic['fare'] = titanic['fare'].apply(lambda x: np.log(x + 1))

# Step 5: Normalization
numerical_features = ['age', 'fare', 'family_size']
scaler = StandardScaler()
titanic[numerical_features] = scaler.fit_transform(titanic[numerical_features])

# Step 6: Encoding Categorical Variables
titanic = pd.get_dummies(titanic, columns=['sex', 'embarked', 'pclass'], drop_first=True)

# Display the first few rows of the preprocessed dataset
print("\nPreprocessed Data:")
print(titanic.head())
