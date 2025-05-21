import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

# Load the dataset
df = pd.read_csv('/Users/mantimokone/Downloads/insurance.csv')
df.dropna(inplace=True)

# Convert 'smoker' to binary (yes = 1, no = 0)
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# One-hot encode 'region'
x = df[['bmi', 'age', 'region']]
x = pd.get_dummies(x, drop_first=True)

y = df['smoker']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict and evaluate
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
dump(model, 'insurance_model.joblib')
print("Model saved successfully!")
