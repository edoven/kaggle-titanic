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




print '####################'
print '1 - Loading datasets'
print '####################'

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
tot = train.copy()
tot = tot.append(test).reset_index(drop=True)



print '########################'
print '2 - Features engineering'
print '########################'

#AGES
class2meanAge = dict(tot[['Pclass', 'Age']].groupby('Pclass').mean().reset_index().values)
newAges = []
for i in range(len(tot)):
    age = tot.Age.values[i]
    if math.isnan(age):
        pclass = tot.Pclass.values[i]
        age = class2meanAge[pclass]
    newAges.append(age) 
tot['Age'] = newAges


#ISCHILD
tot['IsChild'] = [1 if age<=16 else 0 for age in tot['Age'].values]


#PARENTLESS CHILD
isChildValues = list(tot['IsChild'].values)
parchValues = list(tot['Parch'].values)
parentlessChild = []
for i in range(len(tot)):
	if (isChildValues[i]==1) & (parchValues[i]==0):
			parentlessChild.append(1)
	else:
		parentlessChild.append(0)
tot['ParentlessChild'] = parentlessChild


#EMBARKED
mostCommonEmbarked = tot['Embarked'].dropna().mode().values[0]
tot['Embarked'] = tot['Embarked'].fillna(mostCommonEmbarked)
tot['Embarked'] = tot['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)



#TITLES
titles = ['Mr.', 'Miss.', 'Master.', 'Mrs.', 'Don.', 'Rev.', 'Dr.', 'Mme.', 'Ms.','Major.', 'Lady.', 'Sir.', 'Mlle.', 'Col.', 'Capt.','Countess.','Jonkheer.', 'Dona.']

peopleTitles = []
for name in tot.Name.values:
    for title in titles:
        if title in name:
            peopleTitles.append(titles.index(title))
            break
tot['Title'] = peopleTitles


#HASCABIN
tot['HasCabin'] = [0 if type(cabin)==float else 1 for cabin in tot.Cabin.values]


#CABIN LETTER
tot['CabinLetter'] = [0 if type(cabin)==float else cabin[0] for cabin in tot.Cabin.values]
cabinLetters = list(set(tot['CabinLetter']))
tot['CabinLetter'] = [cabinLetters.index(cabinLetter) for cabinLetter in tot['CabinLetter']]  


#CABIN (str to int)
cabinsIds = list(set(tot['Cabin']))  
tot['Cabin'] = [0 if type(cabin)==float else cabinsIds.index(cabin) for cabin in tot.Cabin.values]


#SEX (str to int)
tot['Sex'] = tot['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


#NAME WITH PARENTESIS
tot['NameWithParentesis'] = [1 if '(' in name else 0 for name in tot['Name'].values]


#FARE (TEST HAS A nan FARE)
for id in tot[tot.Fare.isnull()].PassengerId.values:
    pclass = tot[tot.PassengerId==id].Pclass.values[0]
    trainPclassFares = train[train.Pclass==pclass]['Fare'].values
    testPclassFares = tot[tot.Fare.notnull()][tot.Pclass==pclass]['Fare'].values
    classMeanFare = np.mean(list(trainPclassFares)+list(testPclassFares))
    index = tot[tot.PassengerId==id].index.values[0]
    tot.loc[index, 'Fare'] = classMeanFare




print '#####################'
print '3 - Updating datasets'
print '#####################'

train = tot.head(len(train))
test = tot.tail(len(test))

columnsToRemove = ['PassengerId', 'Survived', 'Name', 'Ticket']
inputColumns = train.columns - columnsToRemove
train_X = train[inputColumns]
train_Y = train['Survived']
test_X = test[inputColumns]




print '########################'
print '4 - Building classifiers'
print '########################'

classifiers = []
for i in range(10):  
    classifier = RandomForestClassifier(n_estimators = random.choice([50,100,500]), 
                                    	max_features=random.choice([None, 'log2', 'auto']), 
                                    	min_samples_split=random.choice([10,2]), 
                                    	min_samples_leaf=random.choice([2,1]))
    classifier.fit(train_X,train_Y.values)
    classifiers.append(classifier)

subsetClassifiers = []
for i in range(10):  
    classifier = RandomForestClassifier(n_estimators = random.choice([50,100,500]), 
                                        max_features=random.choice([None, 'log2', 'auto']), 
                                        min_samples_split=random.choice([10,2]), 
                                        min_samples_leaf=random.choice([2,1]))
    classifier.fit(train[['Sex','Pclass', 'Age']], train_Y.values)
    subsetClassifiers.append(classifier)
print




print '########################'
print '5 - Building predictions'
print '########################'

totalPredictions = []
for classifier in classifiers:
    predictions = classifier.predict(test_X)
    totalPredictions.append(predictions)
for classifier in subsetClassifiers:
    predictions = classifier.predict(test[['Sex','Pclass', 'Age']])
    totalPredictions.append(predictions)

finalPredictions = []
for i in range(len(test_X)):
    values = [prediction[i] for prediction in totalPredictions]
    finalPredictions.append( int(np.round(np.mean(values))) )




print '###################'
print '6 - Writing results'
print '###################'

submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": finalPredictions})
submission.to_csv("submissions/model04.csv", index=False)