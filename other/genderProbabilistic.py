# Tutorial 1: Gender Based Model (0.76555) - by Myles O'Neill
import numpy as np
import pandas as pd
import pylab as plt
import random

# (1) Import the Data into the Script
train = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )

# (2) Create the submission file with passengerIDs from the test file
submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})

zerosMale = len(train[train.Sex=='male'][train.Survived==0])
onesMale = len(train[train.Sex=='male'][train.Survived==1])
probZeroMale = zerosMale/float(zerosMale+onesMale)
probOneMale = onesMale/float(zerosMale+onesMale)
posMale = [0]*(int(probZeroMale*1000)) + [1]*(int(probOneMale*1000))

def maleProb(): 
    return random.choice(posMale) 


zerosFemale = len(train[train.Sex=='female'][train.Survived==0])
onesFemale = len(train[train.Sex=='female'][train.Survived==1])
probZeroFemale = zerosFemale/float(zerosFemale+onesFemale)
probOneFemale = onesFemale/float(zerosFemale+onesFemale)
posFemale = [0]*(int(probZeroFemale*1000)) + [1]*(int(probOneFemale*1000))

    
def femaleProb():
    return random.choice(posFemale)

# (3) Fill the Data for the survived column, all females live (1) all males die (0)
submission.Survived = [int(np.round(np.mean([femaleProb() for _ in range(10)]))) if x == 'female' else int(np.round(np.mean([maleProb() for _ in range(10)]))) for x in test['Sex']]

# (4) Create final submission file
submission.to_csv("genderProb.csv", index=False)