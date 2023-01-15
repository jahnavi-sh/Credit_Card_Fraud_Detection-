#workflow - 
#1. load credit card data 
#2. data preprocessing 
#3. data analysis and exploration 
#4. train test split 
#5. binary classification model - logistic regression algorithm 
#6. model evaluation - accuracy score 

#import libraries 
#linear algebra - constructing matrices 
import numpy as np 

#data preprocessing and analysis 
import pandas as pd 

#model training and evaluation 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load data 
credit_card_data = pd.read_csv(r'credit_card_data.csv')

#view data
#view first 5 rows of the dataset 
credit_card_data.head()
#the dataset has following columns 
#1. Time
#2. Since credit card data is sensitive information. therefore rest of the data is encoded (V1 - V28) using 
#   principal component analysis method and numerical data has been provided  
#3. Amount 
#4. Class

#data exploration to get familiar with the dataset 
#view last 5 rows of the dataset 
credit_card_data.tail()

#view the statistical measures of the data 
credit_card_data.describe()

#view missing values 
credit_card_data.isnull().sum()
#there are no missing values in the dataset 

#check distribution of legit transaction and fraud transaction 
credit_card_data['Class'].value_counts()
#this data is highly unbalanced 
#0-normal 
#1-fraud

#there are 2,84,315 legitimate transactions and 492 fradulent transactions 

#separate data 
legit = credit_card_data[credit_card_data.Class==0]
fraud = credit_card_data[credit_card_data.Class==1]

#statistical measures of data 
legit.Amount.describe()
fraud.Amount.describe()

#compare values for both transaction 
credit_card_data.groupby('Class').mean()

#under sampling
#build sample dataset from original dataset with similar distribution of normal transaction and fraudulent transaction 
legit_sample = legit.sample(n=492)

#concatenating two dataframes
new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset.head()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

#split data into features and target 
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

#train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#train model 
model = LogisticRegression()
model.fit(X_train, Y_train)

#evaluate model - accuracy score 
#training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#accuracy score is 94%

#test data 
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#accuracy score is 93%