# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 23:22:18 2020

@author: ASUS
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import accuracy_score

toilet_data = pd.read_excel("File 7 - Group7.xlsx", header = 0, sep = " ")
toilet=toilet_data

print(toilet.columns)

print(toilet["Flush volume"].value_counts())






#4.203382217526121e-25   Assumption 1
# =============================================================================
toilet["Blue11"]=np.sqrt(toilet['Blue LED 1\nPhotodiode 1'])
toilet["Blue12"]=np.sqrt(toilet['Blue LED 1\nPhotodiode 2'])
toilet["Blue21"]=np.sqrt(toilet['Blue LED 2\nPhotodiode 1'])
toilet["Blue22"]=np.sqrt(toilet['Blue LED 2\nPhotodiode 2'])
toilet["Green11"]=np.sqrt(toilet['Green LED 1\nPhotodiode 1'])
toilet["Green12"]=np.sqrt(toilet['Green LED 1\nPhotodiode 2'])
toilet["Green21"]=np.sqrt(toilet['Green LED 2\nPhotodiode 1'])
toilet["Green22"]=np.sqrt(toilet['Green LED 2\nPhotodiode 2'])
toilet["Red11"]=np.sqrt(toilet['Red LED 1\nPhotodiode 1'])
toilet["Red12"]=np.sqrt(toilet['Red LED 1\nPhotodiode 2'])
toilet["Red21"]=np.sqrt(toilet['Red LED 2\nPhotodiode 1'])
toilet["Red22"]=np.sqrt(toilet['Red LED 2\nPhotodiode 2'])
X=toilet[["Blue11", "Blue12", "Blue21", "Blue22", "Green11", "Green12", "Green21", "Green22", "Red11", "Red12", "Red21", "Red22"]]
X = sm.add_constant(X)
Y=toilet[["Flush volume"]]
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
bins = [-100,0.75,1.7,2.15,2.6,3.05,3.55,4,4.45,4.95,5.4,5.85,100]

toilet.loc[:,'FlushV_int']=pd.cut(toilet['Flush volume'],bins,
                                  labels=[0,1,2,3,4,5,6,7,8,9,10,11])
Y2=toilet.loc[:,'FlushV_int']
ypred=results.predict(X)
toilet.loc[:,"Ypred"]=ypred

toilet.loc[:,'Flushpr_int']=pd.cut(ypred,bins,labels=[0,1,2,3,4,5,6,7,8,9,10,11])
X2=toilet['Flushpr_int']
results = confusion_matrix(Y2,X2)
y_pred = pd.Series(toilet['Flushpr_int'])
y_true = pd.Series(toilet["Flush volume"])
fchi=pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
fchi=stats.chi2_contingency(fchi)
print(fchi)
# =============================================================================

#5.629189891557796e-21  Assumption 2
# =============================================================================
toilet["Blue11"]=np.log(toilet['Blue LED 1\nPhotodiode 1']+1)
toilet["Blue12"]=np.log(toilet['Blue LED 1\nPhotodiode 2']+1)
toilet["Blue21"]=np.log(toilet['Blue LED 2\nPhotodiode 1']+1)
toilet["Blue22"]=np.log(toilet['Blue LED 2\nPhotodiode 2']+1)
toilet["Green11"]=np.log(toilet['Green LED 1\nPhotodiode 1']+1)
toilet["Green12"]=np.log(toilet['Green LED 1\nPhotodiode 2']+1)
toilet["Green21"]=np.log(toilet['Green LED 2\nPhotodiode 1']+1)
toilet["Green22"]=np.log(toilet['Green LED 2\nPhotodiode 2']+1)
toilet["Red11"]=np.log(toilet['Red LED 1\nPhotodiode 1']+1)
toilet["Red12"]=np.log(toilet['Red LED 1\nPhotodiode 2']+1)
toilet["Red21"]=np.log(toilet['Red LED 2\nPhotodiode 1']+1)
toilet["Red22"]=np.log(toilet['Red LED 2\nPhotodiode 2']+1)
X=toilet[["Blue11", "Blue12", "Blue21", "Blue22", "Green11", "Green12", "Green21", "Green22", "Red11", "Red12", "Red21", "Red22"]]
X = sm.add_constant(X)
Y=toilet[["Flush volume"]]
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
bins = [-100,0.75,1.7,2.15,2.6,3.05,3.55,4,4.45,4.95,5.4,5.85,100]

toilet.loc[:,'FlushV_int']=pd.cut(toilet['Flush volume'],bins,
                                  labels=[0,1,2,3,4,5,6,7,8,9,10,11])
Y2=toilet.loc[:,'FlushV_int']
ypred=results.predict(X)
toilet.loc[:,"Ypred"]=ypred

toilet.loc[:,'Flushpr_int']=pd.cut(ypred,bins,labels=[0,1,2,3,4,5,6,7,8,9,10,11])
X2=toilet['Flushpr_int']
results = confusion_matrix(Y2,X2)

y_pred = pd.Series(toilet['Flushpr_int'])
y_true = pd.Series(toilet["Flush volume"])
fchi=pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
fchi=stats.chi2_contingency(fchi)
print(fchi)
# =============================================================================

#4.62526748964767e-22   Assumption 3
# =============================================================================
toilet["Blue11"]=toilet['Blue LED 1\nPhotodiode 1']
toilet["Blue12"]=toilet['Blue LED 1\nPhotodiode 2']
toilet["Blue21"]=toilet['Blue LED 2\nPhotodiode 1']
toilet["Blue22"]=toilet['Blue LED 2\nPhotodiode 2']
toilet["Green11"]=toilet['Green LED 1\nPhotodiode 1']
toilet["Green12"]=toilet['Green LED 1\nPhotodiode 2']
toilet["Green21"]=toilet['Green LED 2\nPhotodiode 1']
toilet["Green22"]=toilet['Green LED 2\nPhotodiode 2']
toilet["Red11"]=toilet['Red LED 1\nPhotodiode 1']
toilet["Red12"]=toilet['Red LED 1\nPhotodiode 2']
toilet["Red21"]=toilet['Red LED 2\nPhotodiode 1']
toilet["Red22"]=toilet['Red LED 2\nPhotodiode 2']
X=toilet[["Blue11", "Blue12", "Blue21", "Blue22", "Green11", "Green12", "Green21", "Green22", "Red11", "Red12", "Red21", "Red22"]]
X = sm.add_constant(X)
Y=toilet[["Flush volume"]]
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
bins = [-100,0.75,1.7,2.15,2.6,3.05,3.55,4,4.45,4.95,5.4,5.85,100]

toilet.loc[:,'FlushV_int']=pd.cut(toilet['Flush volume'],bins,
                                  labels=[0,1,2,3,4,5,6,7,8,9,10,11])
Y2=toilet.loc[:,'FlushV_int']
ypred=results.predict(X)
toilet.loc[:,"Ypred"]=ypred

toilet.loc[:,'Flushpr_int']=pd.cut(ypred,bins,labels=[0,1,2,3,4,5,6,7,8,9,10,11])
X2=toilet['Flushpr_int']
results = confusion_matrix(Y2,X2)
y_pred = pd.Series(toilet['Flushpr_int'])
y_true = pd.Series(toilet["Flush volume"])
fchi=pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
fchi=stats.chi2_contingency(fchi)
print(fchi)
# =============================================================================

#2.549091996368603e-13    Assumption 4
# =============================================================================
toilet["Blue11"]=toilet['Blue LED 1\nPhotodiode 1']**2
toilet["Blue12"]=toilet['Blue LED 1\nPhotodiode 2']**2
toilet["Blue21"]=toilet['Blue LED 2\nPhotodiode 1']**2
toilet["Blue22"]=toilet['Blue LED 2\nPhotodiode 2']**2
toilet["Green11"]=toilet['Green LED 1\nPhotodiode 1']**2
toilet["Green12"]=toilet['Green LED 1\nPhotodiode 2']**2
toilet["Green21"]=toilet['Green LED 2\nPhotodiode 1']**2
toilet["Green22"]=toilet['Green LED 2\nPhotodiode 2']**2
toilet["Red11"]=toilet['Red LED 1\nPhotodiode 1']**2
toilet["Red12"]=toilet['Red LED 1\nPhotodiode 2']**2
toilet["Red21"]=toilet['Red LED 2\nPhotodiode 1']**2
toilet["Red22"]=toilet['Red LED 2\nPhotodiode 2']**2
X=toilet[["Blue11", "Blue12", "Blue21", "Blue22", "Green11", "Green12", "Green21", "Green22", "Red11", "Red12", "Red21", "Red22"]]
X = sm.add_constant(X)
Y=toilet[["Flush volume"]]
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
bins = [-100,0.75,1.7,2.15,2.6,3.05,3.55,4,4.45,4.95,5.4,5.85,100]

toilet.loc[:,'FlushV_int']=pd.cut(toilet['Flush volume'],bins,
                                  labels=[0,1,2,3,4,5,6,7,8,9,10,11])
Y2=toilet.loc[:,'FlushV_int']
ypred=results.predict(X)
toilet.loc[:,"Ypred"]=ypred

toilet.loc[:,'Flushpr_int']=pd.cut(ypred,bins,labels=[0,1,2,3,4,5,6,7,8,9,10,11])
X2=toilet['Flushpr_int']
results = confusion_matrix(Y2,X2)
y_pred = pd.Series(toilet['Flushpr_int'])
y_true = pd.Series(toilet["Flush volume"])
fchi=pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
fchi=stats.chi2_contingency(fchi)
print(fchi)