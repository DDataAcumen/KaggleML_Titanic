# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:03:50 2020

@author: DHRUV
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 19:10:35 2020

@author: DHRUV
"""

#Survival Rates for Males and Females
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset = pd.read_csv('Titanic_train.csv')
dataset.info()
from pandas.plotting import scatter_matrix
scatter_matrix(dataset, diagonal = 'kde')

Noofmales = dataset[dataset['Sex']=='male']
Noofmales = Noofmales.reset_index()
Noofmales = Noofmales.drop(columns =['index'])
Nooffemales = dataset[dataset['Sex'] =='female']
Nooffemales = Nooffemales.reset_index()
Nooffemales = Nooffemales.drop(columns =['index'])

SurvivedMales = Noofmales[Noofmales['Survived'] == 1]
SurvivedMales = SurvivedMales.reset_index()
SurvivedMales = SurvivedMales.drop(columns =['index'])
rate_survivedmales = len(SurvivedMales['Survived'])/len(Noofmales['Survived'])*100

SurvivedFemales = Nooffemales[Nooffemales['Survived'] == 1]
SurvivedFemales = SurvivedFemales.reset_index()
SurvivedFemales = SurvivedFemales.drop(columns =['index'])
rate_survivedfemales = len(SurvivedFemales['Survived'])/len(Nooffemales['Survived'])*100

#Survival Based on Passenger Class
classification_m = SurvivedMales.groupby('Pclass')['Survived'].sum()
classification_f = SurvivedFemales.groupby('Pclass')['Survived'].sum()

classification_m = pd.DataFrame(classification_m)
classification_m = classification_m.reset_index()
sns.barplot(classification_m.Pclass, classification_m.Survived)
plt.title('Survived Males across Classes')

classification_f = pd.DataFrame(classification_f)
classification_f = classification_f.reset_index()
sns.barplot(classification_f.Pclass, classification_f.Survived)
plt.title('Survived Females across Classes')


#Survival as per Age 
Age_Classification_M = SurvivedMales.groupby('Age')['Survived'].value_counts()
Age_Classification_M = Age_Classification_M.reset_index('Age')
Age_Classification_M.columns = ['Age_inyears', 'Survival_Frequency']
#Age_Classification_M = Age_Classification_M.reindex(columns = ['Survival_Frequency', 'Age_inyears'])
#Age_Classification_M = Age_Classification_M.transpose()
sns.barplot(Age_Classification_M.Age_inyears, Age_Classification_M.Survival_Frequency, palette = 'cubehelix')
plt.xticks(rotation=99)

sns.pointplot(Age_Classification_M.Age_inyears, Age_Classification_M.Survival_Frequency, palette = 'Spectral')
plt.xticks(rotation=99)

Age_Classification_F = SurvivedFemales.groupby('Age')['Survived'].value_counts()
Age_Classification_F = Age_Classification_F.reset_index('Age')
Age_Classification_F.columns = ['Age_inyears', 'Survival_Frequency']

sns.barplot(Age_Classification_F.Age_inyears, Age_Classification_F.Survival_Frequency, palette = 'cubehelix')
plt.xticks(rotation=99)

sns.pointplot(Age_Classification_F.Age_inyears, Age_Classification_F.Survival_Frequency, palette = 'Spectral')
plt.xticks(rotation=99)

# Fare 
dataset['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.title('Fare Allocation')
plt.xlabel('Amount of Fare')
plt.ylabel('No ofPeople')



dataset1 = pd.read_csv('Kaggle_test.csv')

#Prediction Using Decision Tree Classifier
X_train = dataset.drop('Survived', axis=1)
X_train = X_train.drop('PassengerId',axis=1)
Y_train = dataset["Survived"]
Y_train = pd.DataFrame(Y_train)
#Y_train = Y_train.reset_index()
#Y_train.columns = ['PassengerId', 'Survived']

X_test  = dataset1.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis=1)
X_test  = X_test.reindex(columns= ['Sex','Pclass','Age', 'Fare'])
X_train.shape, Y_train.shape, X_test.shape

#Replacing M/F by 0 and 1
X_train['Sex'].replace(['male','female'],[0,1],inplace=True)
X_test['Sex'].replace(['male','female'],[0,1],inplace=True)


#Filling Null Values with the Mean Values
X_test.mean()
X_test = X_test.fillna(X_test.mean())


#Dropping Null from X_test Set
#X_test = X_test.dropna(subset=['Age'])
#X_test = X_test.dropna(subset=['Fare'])

X_test.Fare = X_test.Fare.astype(int)
X_test.Age = X_test.Age.astype(int)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_test = decision_tree.predict(X_test)
Y_test = pd.DataFrame(Y_test)
Y_test.columns = ['Survived']
#score_decision_tree = round(decision_tree.score(X_test, Y_test) *100, 2)
#score_decision_tree


#Making Minor Adjustments
#X_test = X_test.reset_index()
#X_test.columns = ['Passenger_No', 'Sex', 'Pclass', 'Age', 'Fare']
Y_test = Y_test.reset_index()
Y_test.columns = ['PassengerId', 'Survived']
Y_test['PassengerId'] = dataset1['PassengerId'].values

#Exporting Our Predicted Set into Dot Csv File  
Y_test.to_csv('Titanic Predicted Survivors', index=True)



