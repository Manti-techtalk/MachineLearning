import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

# Load dataset
df = pd.read_csv('/Users/mantimokone/Downloads/tested 2.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Check what columns exist
print(df.columns)

# Select features and target
x = df[['Pclass', 'PassengerId', 'Age']]
y = df['Survived']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict
predictions = model.predict(x_test)
print(predictions)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")

# Save model
dump(model, 'titanic_model.joblib')
