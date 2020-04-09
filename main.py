import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

a=pd.read_csv("./cci.csv")
a=a.drop(columns=['2016','Month'])
x=list(a.columns)
y=list(a.iloc[12,:])


for i in range(len(y)):
    y[i] = float(y[i])
    
for i in range(len(x)):
    x[i] = int(x[i])

test_x=x.pop()
test_y=y.pop()

x_mean=np.mean(x)
y_mean=np.mean(y)
l=len(x)
numr=0
denm=0
for i in range(l):
    
    numr+=(x[i]-x_mean)*(y[i]-y_mean)
    denm+=(x[i]-x_mean)**2
grad=numr/denm
const=y_mean-(grad*x_mean) 
# print(const)

_y=[0]*len(x)

for i in range(len(x)):
    
    _y[i]=grad*x[i]+const
pred_y=pd.Series(_y)
plt.subplot(1,2,1)
plt.plot(x,pred_y,color='red')
plt.scatter(x,y)

pred_y=grad*test_x+const
print((pred_y/test_y)*100)
print(x,y)

startingYear = 2020
predictYears = []
predictCCIValues = []
for i in range(10):
    predictCCIValues.append(grad*startingYear + const)
    predictYears.append(startingYear)
    startingYear+=1
    
print("Predicted Data")
print(predictCCIValues)
print(predictYears)

plt.subplot(1,2,2)
plt.scatter(predictYears, predictCCIValues)

# k means clustering
import numpy as np

k=3

centroid={i+1:[np.random.randint(0,2449) ,np.random.randint(0,2449) ] 
            for i in range(k)}
# centroid

plt.scatter(x,y,color='k')
colormap={1:'r',2:'g',3:'b'}
for i in centroid.keys():
    plt.scatter(*centroid,color=colormap[i])

plt.show()
centroid



def assigment(a,centroid):
    k_x=pd.Series(x)
    k_y=pd.Series(y)
    for i in centroid.keys():
        a['distance_from{}'.format(i)]=np.sqrt((k_x-centroid[i][0])**2 + 
                                           (k_y-centroid[i][1])**2 )
        centroid_dist=['distance_from{}'.format(i) for i in centroid.keys()]
        a['closest']=a.loc[:,centroid_dist].idxmin(axis=1)
        a['closest']=a['closest'].map(lambda kx: int(kx.lstrip('distance_from')))
        a['color']=a['closest'].map(lambda kx:colormap[kx])    
    return a
k_x=pd.Series(x)
k_y=pd.Series(y)
a['average']=k_y
a['years']=k_x
assigment(a,centroid)
a


# logistic regression
from sklearn.model_selection import train_test_split
log_x=a.drop(columns=['color','closest','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007', '2008','2009','2010','2011','2012','2013','2014','2015'])
log_y=a['closest']

x_train,x_test,y_train,Y_test=train_test_split(log_x,log_y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
log_x


predictions=logmodel.predict(x_test)
from sklearn.metrics import classification_report
classification_report(Y_test,predictions)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,predictions)
from sklearn.metrics import  accuracy_score
accuracy_score(Y_test,predictions)