from sklearn.datasets import load_iris # load the dataset
from sklearn.model_selection import train_test_split #for splitting the dataset
from sklearn.linear_model import LogisticRegression
from joblib import dump #for saving the model


iris = load_iris()
x = iris.data
y = iris.target

#splitting the data into trainng and testing set

x_train , x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


#chooseing and training the model

model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

#save the model for later use

dump(model,'model.joblib')

#testing the model

accuracy = model.score(x_test,y_test)
print(f"Model accuracy: {accuracy*100:.2f}%")