__author__ = 'Xing'

import pandas as pd
import numpy as np
import re

#加resampling



df=pd.read_csv('train.csv')
df.Age=df.Age.fillna(28.000000) #fill na with median
df=df.fillna(0)
df['child']=df['Age'].apply(lambda d:1 if d<=15 else 0)
df['Title']=df['Name'].apply(lambda d:re.split(',|\.',d)[1][1:])
df['Title']=df['Title'].apply(lambda d:'Mlle' if d in ['Mlle','Mme'] else d)
df['Title']=df['Title'].apply(lambda d:'Sir' if d in ['Capt', 'Don', 'Major', 'Sir'] else d)
df['Title']=df['Title'].apply(lambda d:'Lady' if d in ['Dona', 'Lady', 'the Countess', 'Jonkheer'] else d)
df['FamilySize']=df.Parch+df.SibSp+1

cols=set(df.columns)
cols = cols-set(['PassengerId','Survived','Name','Age'])

#resampling
# import random
# df1=pd.DataFrame()
# randomIndex=[random.randint(0,len(df)-1) for i in range(1300)]
# for index in randomIndex:
#     df1=df1.append(df.ix[index])
# df=df.append(df1)



x = df.ix[:,cols] #input
y = df['Survived'].values #output
from sklearn.feature_extraction import DictVectorizer #in python, the input should be float, so one-hot encoding is needed
v = DictVectorizer()
x_vec = v.fit_transform(x.to_dict(outtype='records')).toarray()
from sklearn.cross_validation import train_test_split


train_x, test_x, train_y, test_y = train_test_split(x_vec, y)


from sklearn import cross_validation
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import datetime
estimators = {}

estimators['tree'] = tree.DecisionTreeClassifier()
estimators['forest_100'] = RandomForestClassifier(n_estimators = 100)
estimators['svm_linear'] = svm.LinearSVC()
estimators['gbdt']=GradientBoostingClassifier(n_estimators=200)
for k in estimators.keys():
    start_time = datetime.datetime.now()
    print ('----%s----' % k)
    model=estimators[k]
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    print("%s Score: %0.2f" % (k, model.score(test_x, test_y)))
    scores = cross_validation.cross_val_score(model, test_x, test_y, cv=5)
    print("%s Cross Avg. Score: %0.2f (+/- %0.2f)" % (k, scores.mean(), scores.std() * 2))
    end_time = datetime.datetime.now()
    time_spend = end_time - start_time
    print("%s Time: %0.2f" % (k, time_spend.total_seconds()))


test=pd.read_csv('test.csv')

test.Age=test.Age.fillna(28.000000) #fill na with median
test=test.fillna(0)
test['child']=test['Age'].apply(lambda d:1 if d<=15 else 0)
test['Title']=test['Name'].apply(lambda d:re.split(',|\.',d)[1][1:])
test['Title']=test['Title'].apply(lambda d:'Mlle' if d in ['Mlle','Mme'] else d)
test['Title']=test['Title'].apply(lambda d:'Sir' if d in ['Capt', 'Don', 'Major', 'Sir'] else d)
test['Title']=test['Title'].apply(lambda d:'Lady' if d in ['Dona', 'Lady', 'the Countess', 'Jonkheer'] else d)
test['FamilySize']=test.Parch+test.SibSp+1
# cols = ['Pclass', 'Title', 'Sex', 'Age', 'FamilySize', 'Ticket', 'Fare', 'Cabin', 'Embarked']
test_input = test.ix[:,cols] #input
test_d = test_input.to_dict(outtype='records')
test_vec = v.transform(test_d).toarray()

for k in estimators.keys():
    model=estimators[k]
    model.fit(x_vec,y)
    pred = estimators[k].predict(test_vec)
    test['Survived'] = pred
    test.to_csv(k + '_2.csv', columns=['PassengerId','Survived'], index=False)