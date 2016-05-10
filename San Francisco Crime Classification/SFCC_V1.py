__author__ = 'Xing'
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from time import time
from sklearn.grid_search import GridSearchCV
from pprint import pprint
from copy import deepcopy
from sklearn.metrics import log_loss

def one_hot_dataframe(data, cols,vec, replace=False,test=False):
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    if not test:
        vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    else:
        vecData = pd.DataFrame(vec.transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    if not test:
        return data,vec
    return data

def pre_count_featurizer(trainDF):
    global logodds
    global logoddsPA
    global addresses
    global categories
    global default_logodds
    global C_counts
    global A_counts
    global A_C_counts
    logodds={}
    logoddsPA={}
    addresses=sorted(trainDF["Address"].unique())
    categories=sorted(trainDF["Category"].unique())
    C_counts=trainDF.groupby(["Category"]).size()
    A_C_counts=trainDF.groupby(["Address","Category"]).size()
    A_counts=trainDF.groupby(["Address"]).size()
    MIN_CAT_COUNTS=2
    default_logodds=np.log(C_counts/len(trainDF))-np.log(1.0-C_counts/float(len(trainDF)))
    for addr in addresses:
        PA=A_counts[addr]/float(len(trainDF))
        logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
        logodds[addr]=deepcopy(default_logodds)
        for cat in A_C_counts[addr].keys():
            if (A_C_counts[addr][cat]>MIN_CAT_COUNTS) and A_C_counts[addr][cat]<A_counts[addr]:
                PA=A_C_counts[addr][cat]/float(A_counts[addr])
                logodds[addr][categories.index(cat)]=np.log(PA)-np.log(1.0-PA)
        logodds[addr]=pd.Series(logodds[addr])
        logodds[addr].index=range(len(categories))

def FeatureEngineering(df,test=False):
    df['Year'] = df['Dates'].map(lambda x: str(x.year))
    # df['Week'] = df['Dates'].map(lambda x: str(x.week))
    df['Month'] = df['Dates'].map(lambda x:str(x.month))
    df['Hour'] = df['Dates'].map(lambda x: str(x.hour))
    df['Day'] = df['Dates'].map(lambda x:str(x.day))

    # cols=set(df.columns)
    # # cols = cols-set(['Dates','Category','Descript','Resolution'])
    # x = df.ix[:,cols] #input

    district = pd.get_dummies(df.PdDistrict,prefix='Dis')
    dof=pd.get_dummies(df.DayOfWeek,prefix='Dof')
    month=pd.get_dummies(df.Month,prefix='Month')
    year=pd.get_dummies(df.Year,prefix='Year')
    hour=pd.get_dummies(df.Hour,prefix='Hour')
    day=pd.get_dummies(df.Day,prefix='Day')

    X_scaled = preprocessing.scale(df.X)
    Y_scaled = preprocessing.scale(df.Y)
    #count featurizer
    if test==False:
        pre_count_featurizer(df)

    address_features=df["Address"].apply(lambda x: logodds[x])
    address_features.columns=["logodds"+str(x) for x in range(len(address_features.columns))]

    train=pd.concat([district, dof, month,year,hour,day,address_features,df.Address], axis=1)#merge

    train['IsInterection']=train["Address"].apply(lambda x: 1 if "/" in x else 0)
    train["logoddsPA"]=train["Address"].apply(lambda x: logoddsPA[x])

    train['X']=X_scaled
    train['Y']=Y_scaled

    train=train.drop('Address',axis=1)

    return train

seed=0
MODEL_PATH="./Model/"
SUBMISSION_PATH="./Submission/"

df = pd.read_csv('input/train.csv', parse_dates=['Dates'])
x = FeatureEngineering(df)

le_crime = preprocessing.LabelEncoder()
y = le_crime.fit_transform(df.Category)

# train_x, test_x, train_y, test_y = train_test_split(
#     x, y, test_size=0.4, random_state=0)
#########EXPLORE THE BEST MODEL############
###RF
# clf=RandomForestClassifier(random_state=seed)
# parameters = {'n_estimators': [300, 500], 'max_features': ['auto']}
# RF_grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, verbose=20, scoring='accuracy')
#
# print("parameters:")
# pprint(parameters)
# t0 = time()
# RF_grid_search.fit(x,y)
# print("done in %0.3fs" % (time() - t0))
# print()
#
# print("Best score: %0.3f" % RF_grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = RF_grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))

###LR
# clf=LogisticRegression(random_state=seed)
# parameters = {'penalty': ['l1', 'l2'], 'C': [0.01,0.1,1]}
# LR_grid_search = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=1, cv=3, verbose=20, scoring='accuracy')
#
# print("parameters:")
# pprint(parameters)
# t0 = time()
# LR_grid_search.fit(x,y)
# print("done in %0.3fs" % (time() - t0))
# print()
#
# print("Best score: %0.3f" % LR_grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = LR_grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
#




#DUMP THE BEST MODEL
# from sklearn.externals import joblib
# model_file=MODEL_PATH+'model-rf.pkl'
# joblib.dump(grid_search.best_estimator_, model_file)

##########SUBMISSION######
# predicted = clf.predict_proba(test)
# result=pd.DataFrame(predicted, columns=le_crime.classes_)
# result.to_csv(SUBMISSION_PATH+'first_try.csv', index = True, index_label = 'Id')

#####model fit and test######
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
clf=LogisticRegression(penalty='l1',C=0.01,random_state=seed)
clf.fit(train_x,train_y)
pred=clf.predict_proba(test_x)
from sklearn.metrics import log_loss
print('log loss %4f' %log_loss(test_y,pred))

test= pd.read_csv('input/test.csv', parse_dates=['Dates'])
addresses=sorted(df["Address"].unique())
new_addresses=sorted(test["Address"].unique())
new_A_counts=test.groupby("Address").size()
only_new=set(new_addresses+addresses)-set(addresses)
only_old=set(new_addresses+addresses)-set(new_addresses)
in_both=set(new_addresses).intersection(addresses)
for addr in only_new:
    PA=new_A_counts[addr]/float(len(test)+len(df))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    logodds[addr]=deepcopy(default_logodds)
    logodds[addr].index=range(len(categories))
for addr in in_both:
    PA=(A_counts[addr]+new_A_counts[addr])/float(len(test)+len(df))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)

test_fe=FeatureEngineering(test,test=True)
predicted = clf.predict_proba(test_fe)
result=pd.DataFrame(predicted, columns=le_crime.classes_)
result.to_csv(SUBMISSION_PATH+'Second_Address_try.csv', index = True, index_label = 'Id')
