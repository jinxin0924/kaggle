__author__ = 'Xing'
df.describe()
df['Sex'].unique()


%matplotlib inline
import matplotlib.pyplot as plt

#histogram
fig =  plt.figure()
# ax = fig.add_subplot(111)
plt.hist(df['Fare'], bins = 10, range = (df['Fare'].min(),df['Fare'].max()))
plt.title('Fare distribution')
plt.xlabel('Fare')
plt.ylabel('Count of Passengers')

#boxplot
df.boxplot(column='Fare', by = 'Pclass')

#categorical feature distribution
df.PdDistrict.value_counts().plot(kind='bar', figsize=(8,10))



#group by and plot
temp1 = df.groupby('Pclass').Survived.count()
temp2 = df.groupby('Pclass').Survived.sum()/df.groupby('Pclass').Survived.count()
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers by Pclass")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Pclass')
ax2.set_ylabel('Probability of Survival')
ax2.set_title("Probability of survival by class")


#group by and plot
temp3 = pd.crosstab([df.Pclass, df.Sex], df.Survived.astype(bool))
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

#group by and plot
hourly_events = train[['Hour','event']].groupby(['Hour']).count().reset_index()
hourly_events.plot(kind='bar', figsize=(6, 6))

#group by and size
trainDF.groupby(["Category"]).size()

#convert string to datetime
pd.to_datetime('2015-05-13 23:53:00', format='%Y-%m-%d %H:%M:%S') 


#one hot encode
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
enc.transform([[0, 1, 1]]).toarray()

#auc
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)

