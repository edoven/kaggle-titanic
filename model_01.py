import numpy as np
import pandas as pd
import sklearn
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.grid_search
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier 
import scipy as sp
import math
import random
import csv



## TRAIN: FEATURES TRANSFORMATIONS ##


def updatedAges():
	class2meanAge = dict(train[['Pclass', 'Age']].groupby('Pclass').mean().reset_index().values)

	trainAges = []
	for i in range(len(train)):
	    age = train.Age.values[i]
	    if math.isnan(age):
	        pclass = train.Pclass.values[i]
	        age = class2meanAge[pclass]
	    trainAges.append(age) 
	train['Age'] = trainAges

	testAges = []
	for i in range(len(test)):
	    age = test.Age.values[i]
	    if math.isnan(age):
	        pclass = test.Pclass.values[i]
	        age = class2meanAge[pclass]
	    testAges.append(age) 
	test['Age'] = testAges  



def updateEmbarked():
	mostCommonEmbarked = train['Embarked'].dropna().mode().values[0]

	train['Embarked'] = train['Embarked'].fillna(mostCommonEmbarked)
	train['Embarked'] = train['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

	test['Embarked'] = test['Embarked'].fillna(mostCommonEmbarked)
	test['Embarked'] = test['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)



def updateTitles():
	titles = ['Mr.', 'Miss.', 'Master.', 'Mrs.', 'Don.', 'Rev.', 'Dr.', 'Mme.', 'Ms.','Major.', 'Lady.', 'Sir.', 'Mlle.', 'Col.', 'Capt.','Countess.','Jonkheer.', 'Dona.']

	trainTitles = []
	for name in train.Name.values:
	    for title in titles:
	        if title in name:
	            trainTitles.append(titles.index(title))
	            break
	train['Title'] = trainTitles

	testTitles = []
	for name in test.Name.values:
	    for title in titles:
	        if title in name:
	            testTitles.append(titles.index(title))
	            break
	test['Title'] = testTitles



def updatedHasCabin():
	train['HasCabin'] = [0 if type(cabin)==float else 1 for cabin in train.Cabin.values]
	test['HasCabin'] = [0 if type(cabin)==float else 1 for cabin in test.Cabin.values]



def updatedCabineLetter():
	train['CabinLetter'] = [0 if type(cabin)==float else cabin[0] for cabin in train.Cabin.values]
	cabinLetters = list(set(train['CabinLetter']))
	train['CabinLetter'] = [cabinLetters.index(cabinLetter) for cabinLetter in train['CabinLetter']]  

	test['CabinLetter'] = [0 if type(cabin)==float else cabin[0] for cabin in test.Cabin.values]
	test['CabinLetter'] = [cabinLetters.index(cabinLetter) for cabinLetter in test['CabinLetter']] 


def updateCabnIds():
	cabinsIds = list(set(train['Cabin']))  
	train['Cabin'] = [0 if type(cabin)==float else cabinsIds.index(cabin) for cabin in train.Cabin.values]

	test['Cabin'] = [0 if type(cabin)==float else (cabinsIds.index(cabin) if cabin in cabinsIds else len(cabinsIds)) for cabin in test.Cabin.values]


def updateSex():
	train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

	test['Sex'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


def updatedParentesisInName():
	train['NameWithParentesis'] = [1 if '(' in name else 0 for name in train['Name'].values]
	test['NameWithParentesis'] = [1 if '(' in name else 0 for name in test['Name'].values]


def updateFare():
	#TEST HAS A null FARE!!
	for id in test[test.Fare.isnull()].PassengerId.values:
	    pclass = test[test.PassengerId==id].Pclass.values[0]
	    trainPclassFares = train[train.Pclass==pclass]['Fare'].values
	    testPclassFares = test[test.Fare.notnull()][test.Pclass==pclass]['Fare'].values
	    classMeanFare = np.mean(list(trainPclassFares)+list(testPclassFares))
	    index = test[test.PassengerId==id].index.values[0]
	    test.loc[index, 'Fare'] = classMeanFare








#CREATE A POOL OF CLASSIFIERS
def getClassifiers():
	classifiersPool = []

	for i in range(100):  
	    print i,
	    classifier = RandomForestClassifier(n_estimators = random.choice([50,100,500]), 
	                                    	max_features=random.choice([None, 'log2', 'auto']), 
	                                    	min_samples_split=random.choice([10,2]), 
	                                    	min_samples_leaf=random.choice([2,1]))
	    classifier.fit(train_X,train_Y.values)
	    classifiersPool.append(classifier)
	print
	return classifiersPool




#GET PREDICTIONS
def getPredictions(classifiers, test_X):
	totalPredictions = []
	for classifier in classifiers:
	    predictions = classifier.predict(test_X)
	    totalPredictions.append(predictions)

    
	finalPredictions = []
	for i in range(len(test_X)):
	    values = [prediction[i] for prediction in totalPredictions]
	    finalPredictions.append( int(np.round(np.mean(values))) )
	return finalPredictions







print '1 - Loading datasets'
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


print '2 - Features engineering'
updatedAges()
updateEmbarked()
updateTitles()
updatedHasCabin()
updatedCabineLetter()
updateCabnIds()
updateSex()
updatedParentesisInName()
updateFare()

print '3 - Updating datasets'
inputColumns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Cabin', 'HasCabin', 'CabinLetter', 'Title', 'NameWithParentesis']
train_X = train[inputColumns]
train_Y = train['Survived']
test_X = test[inputColumns]

print '4 - Building classifiers'
classifiers = getClassifiers()

print '5 - Building predictions'
predictions = getPredictions(classifiers, test_X)

print '6 - Writing results'
submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": predictions})
submission.to_csv("prova.csv", index=False)