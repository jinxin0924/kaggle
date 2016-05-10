__author__ = 'Xing'
import pandas as pd
import numpy as np
import re
#more feature engineering
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

def data_processing(df):

    def replace_titles(x):
        title=x['Title']
        if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme','Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms','Miss']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title =='':
            if x['Sex']=='Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    def Name(df):
        df['Title']=df['Name'].apply(lambda d:re.split(',|\.',d)[1][1:])
        df['Title']=df.apply(replace_titles, axis=1)
        df = df.drop(['Name'], axis=1)
        return df

    def Age(df):
        df['AgeFill']=df['Age']
        mean_ages = np.zeros(4)
        mean_ages[0]=np.average(df[df['Title'] == 'Miss']['Age'].dropna())
        mean_ages[1]=np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
        mean_ages[2]=np.average(df[df['Title'] == 'Mr']['Age'].dropna())
        mean_ages[3]=np.average(df[df['Title'] == 'Master']['Age'].dropna())
        df.loc[ (df.Age.isnull()) & (df.Title == 'Miss') ,'AgeFill'] = mean_ages[0]
        df.loc[ (df.Age.isnull()) & (df.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]
        df.loc[ (df.Age.isnull()) & (df.Title == 'Mr') ,'AgeFill'] = mean_ages[2]
        df.loc[ (df.Age.isnull()) & (df.Title == 'Master') ,'AgeFill'] = mean_ages[3]
        df['AgeCat']=df['AgeFill']
        df.loc[ (df.AgeFill<=10) ,'AgeCat'] = 'child'
        df.loc[ (df.AgeFill>60),'AgeCat'] = 'aged'
        df.loc[ (df.AgeFill>10) & (df.AgeFill <=30) ,'AgeCat'] = 'adult'
        df.loc[ (df.AgeFill>30) & (df.AgeFill <=60) ,'AgeCat'] = 'senior'
        df = df.drop(['Age'], axis=1)
        return df

    def Gender(df):
        df['Gender'] = df['Sex'].apply(lambda d: 1 if d=='female' else 0)
        df = df.drop(['Sex'], axis=1)
        return df

    def FamilySize(df):
        df['Family_Size']=df.Parch+df.SibSp+1
        df['Family']=df['SibSp']*df['Parch']
        return df

    def Ticket(df):
        def getTicketPrefix(ticket):
            match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
            if match:
                return match.group()
            else:
                return 'U'
        def getTicketNumber(ticket):
            match = re.compile("([\d]+$)").search(ticket)
            if match:
                return int(match.group())
            else:
                return 0
        df['TicketPrefix'] = df['Ticket'].map(lambda x: getTicketPrefix(x.upper()))
        df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('[\.?\/?]','',x))
        df['TicketPrefix'] = df['TicketPrefix'].map(lambda x:re.sub('STON','SOTON',x))
        df['TicketNumber'] = df['Ticket'].map( lambda x: getTicketNumber(x) )
        df['TicketNumberStart']=df['TicketNumber'].apply(lambda d:str(d)[0])
        return df

    def Fare(df):
        df.loc[ (df.Fare.isnull())&(df.Pclass==1),'Fare'] =np.median(df[df['Pclass'] == 1]['Fare'].dropna())
        df.loc[ (df.Fare.isnull())&(df.Pclass==2),'Fare'] =np.median( df[df['Pclass'] == 2]['Fare'].dropna())
        df.loc[ (df.Fare.isnull())&(df.Pclass==3),'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())
        df['Fare_Per_Person']=df['Fare']/df['Family_Size']
        return df

    def Embarked(df):
        df.Embarked = df.Embarked.fillna('S')
        return df

    def cross(df):
        df['AgeClass']=df['AgeFill']*df['Pclass']
        df['ClassFare']=df['Pclass']*df['Fare_Per_Person']
        return df

    def Cabin(df):
        df.loc[ (df.Cabin.isnull()),'Cabin']='Null'
        return df

    df=Name(df)
    df=Age(df)
    df=Gender(df)
    df=FamilySize(df)
    df=Ticket(df)
    df=Fare(df)
    df=Embarked(df)
    df=cross(df)
    df=Cabin(df)
    return df

seed=0
MODEL_PATH="./Model/"
SUBMISSION_PATH="./Submission/"

df=pd.read_csv('train.csv')
cols=set(df.columns)
cols = cols-set(['PassengerId','Survived'])
x = df.ix[:,cols] #input
y = df['Survived'].values #output
x=data_processing(x)

v = DictVectorizer()
x_vec = v.fit_transform(x.to_dict(outtype='records')).toarray()
from sklearn.cross_validation import train_test_split
##select a train and test set
train_x, test_x, train_y, test_y = train_test_split(x_vec, y,test_size=0.2,random_state=seed)

from sklearn import cross_validation
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from time import time
from sklearn.grid_search import GridSearchCV
from pprint import pprint

clf=RandomForestClassifier(random_state=seed)
parameters = {'n_estimators': [300, 500], 'max_features': [4,5,'auto']}
grid_search = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=1, cv=10, verbose=20, scoring='accuracy')

print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(train_x,train_y)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

#DUMP THE BEST MODEL
from sklearn.externals import joblib
model_file=MODEL_PATH+'model-rf.pkl'
joblib.dump(grid_search.best_estimator_, model_file)


###########################
clf = joblib.load(model_file)
df=pd.read_csv('test.csv')
x = df.ix[:,cols] #input
x=data_processing(x)
x_d = x.to_dict(outtype='records')
x_vec = v.transform(x_d).toarray()
pred = clf.predict(x_vec)
df['Survived'] = pred
subFile=SUBMISSION_PATH+'RF.csv'
df.to_csv(subFile, columns=['Survived', 'PassengerId'], index=False)