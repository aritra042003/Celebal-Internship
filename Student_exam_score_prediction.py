import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  # For handling missing values

# Define functions for data preprocessing
def preprocess_data(df):
  # Handle missing values (replace with appropriate strategy if needed)
  imputer = SimpleImputer(strategy='mean')
  df = pd.DataFrame(imputer.fit_transform(df))

  

  return df

# Load data from CSV files
df_mat = pd.read_csv("student-mat.csv")
df_por = pd.read_csv("student-por.csv")

# Preprocess data (optional, call the function on each dataframe)
df_mat = preprocess_data(df_mat.copy())
df_por = preprocess_data(df_por.copy())

# Combine dataframes (optional, ensure they have the same structure)
# df = pd.concat([df_mat, df_por])

# Define features and target variable (replace with actual names)
features = ['Hours Studied', 'Previous Exam Score', 'Attendance']
target = 'Final Grade'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_mat[features], df_mat[target], test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)


print("Predicted exam scores:", y_pred)
print("Actual exam scores:", y_test)

# Example: Print coefficients to see how features affect score
print("Model coefficients:", model.coef_)
