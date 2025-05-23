import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#imporrt the dataset
data = pd.read_csv('/Users/mantimokone/Downloads/heart_disease_uci.csv')

#clean the dataset
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

#expore the dataset
print("_____Head of the dataset_____")
print(data.head(10))
print("_____Describe of the dataset_____")
print(data.describe())
print("_____Info of the dataset_____")
print(data.info())
print("_____Shape of the dataset_____")
print(data.shape)
print("_____Columns of the dataset_____")
print(data.columns)

#features and target
x = data[['age','ca','chol']].values
y = data['num'].values


print("_____Features_____")
print(x)
print("_____Target_____")
print(y)
#train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#fit the model
model.fit(x_train, y_train)
#predict the model
y_pred = model.predict(x_test)

#evaluate the model
print("_____Classification Report_____")
print(classification_report(y_test, y_pred))
print("_____Confusion Matrix_____")
print(confusion_matrix(y_test, y_pred))

#save the model
dump(model, 'heart_disease_model.joblib')
print("_____Model Saved_____")