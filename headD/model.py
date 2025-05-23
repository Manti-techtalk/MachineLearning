import pandas as pd


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
