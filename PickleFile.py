import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 1. Load the data
df = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')

# 2. Data Cleaning (Handling Missing Values)
# Categorical: Fill with most frequent value (Mode)
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Numerical: Fill with Median
num_cols = ['LoanAmount', 'Loan_Amount_Term']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# 3. Feature Engineering
# Combine incomes to create a stronger predictor
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# 4. Encoding Categorical Variables
# We create a dictionary to store an individual encoder for every column.
# This ensures that when a user types "Male" in your app, the app knows exactly 
# what number (e.g., 1) the model expects.
encoders = {}
cols_to_encode = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']

for col in cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le # Store the fitted encoder

# Encode the Target variable (Loan_Status: Y=1, N=0)
le_status = LabelEncoder()
df['Loan_Status'] = le_status.fit_transform(df['Loan_Status'])
encoders['Loan_Status'] = le_status

# 5. Model Training
# We drop Loan_ID (irrelevant) and Loan_Status (the target) from X
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X, y)

# 6. Serialization (Saving to .pkl)
# We save the model object
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)

# We save the dictionary of encoders so the app can "reverse engineer" text inputs
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)