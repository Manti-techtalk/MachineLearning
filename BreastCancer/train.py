from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

# prep the dataset
data = load_breast_cancer()
x = data.data
y = data.target

#split the data into training and testing sets

x_training, x_testing, y_training,y_testing = train_test_split(x,y,test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_training,y_training)

dump(model,"cancer_model.joblib")
