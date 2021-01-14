# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:40:05 2020

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

while (1):
    
    #xn=[0.6049757,0.80858354,0.257002,0.63890724,0.5909667,0.59050965,
    #0.61059362,0.10776922,0.33897334,0.31753335,0.67646697,0.49213317]#
    xn=np.random.rand(12)
    # 0.4095061179987186 8.742664186153026e-31
    # 0.3704170514493971 5.861962592836711e-33
    #[0.6049757  0.80858354 0.257002   0.63890724 0.5909667  0.59050965
    #0.61059362 0.10776922 0.33897334 0.31753335 0.67646697 0.49213317] 3.326956159301724e-42
    
    #your model
    toilet["Blue11"]=toilet['Blue LED 1\nPhotodiode 1']**xn[0]
    toilet["Blue12"]=toilet['Blue LED 1\nPhotodiode 2']**xn[1]
    toilet["Blue21"]=toilet['Blue LED 2\nPhotodiode 1']**xn[2]
    toilet["Blue22"]=toilet['Blue LED 2\nPhotodiode 2']**xn[3]
    toilet["Green11"]=toilet['Green LED 1\nPhotodiode 1']**xn[4]
    toilet["Green12"]=toilet['Green LED 1\nPhotodiode 2']**xn[5]
    toilet["Green21"]=toilet['Green LED 2\nPhotodiode 1']**xn[6]
    toilet["Green22"]=toilet['Green LED 2\nPhotodiode 2']**xn[7]
    toilet["Red11"]=toilet['Red LED 1\nPhotodiode 1']**xn[8]
    toilet["Red12"]=toilet['Red LED 1\nPhotodiode 2']**xn[9]
    toilet["Red21"]=toilet['Red LED 2\nPhotodiode 1']**xn[10]
    toilet["Red22"]=toilet['Red LED 2\nPhotodiode 2']**xn[11]
    
    
    X=toilet[[ 'Blue11', 'Blue12','Blue21', 'Blue22', 'Green11', 'Green12',
          'Green21', 'Green22', 'Red11','Red12', 'Red21', 'Red22']]
    X = sm.add_constant(X)
    Y=toilet[['Flush volume']]
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
    if fchi[1]<3.326956159301724e-42:
        print(xn, fchi[1])
        break
    
sns.heatmap(results, annot=True,  fmt='', cmap='Blues')
