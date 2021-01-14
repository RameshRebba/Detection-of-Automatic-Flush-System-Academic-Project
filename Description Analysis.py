#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:50:27 2020

@author: dorian
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import scipy.stats as stats

data = pd.read_excel('/Users/dorian/Desktop/Data Science/Group 7/File 7 - Group7 - Dorian.xlsx')
ml_data=data[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2','Case of flush']] 
features=ml_data[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]  
target=ml_data['Case of flush']

#Analysis of Case of flush

cof0=ml_data[ml_data['Case of flush']==0]
f0=cof0[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f0)
plt.show()

cof1=ml_data[ml_data['Case of flush']==1]
f1=cof1[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f1)
plt.show()

cof2=ml_data[ml_data['Case of flush']==2]
f2=cof2[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f2)
plt.show()

cof3=ml_data[ml_data['Case of flush']==3]
f3=cof3[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f3)
plt.show()

cof4=ml_data[ml_data['Case of flush']==4]
f4=cof4[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f4)
plt.show()

cof5=ml_data[ml_data['Case of flush']==5]
f5=cof5[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f5)
plt.show()

cof6=ml_data[ml_data['Case of flush']==6]
f6=cof6[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f6)
plt.show()

cof7=ml_data[ml_data['Case of flush']==7]
f7=cof7[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f7)
plt.show()

cof8=ml_data[ml_data['Case of flush']==8]
f8=cof8[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f8)
plt.show()

cof9=ml_data[ml_data['Case of flush']==9]
f9=cof9[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f9)
plt.show()

cof10=ml_data[ml_data['Case of flush']==10]
f10=cof10[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f10)
plt.show()

cof11=ml_data[ml_data['Case of flush']==11]
f11=cof11[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(f1)
plt.show()

#Sum of all values by color

#Blue

blue=ml_data[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2']]
sns.distplot(blue,color='b')
plt.show()
print(stats.shapiro(blue))

blue['agg']=blue[list(blue.columns)].sum(axis=1)
blue_agg=blue['agg']
ml_data['Blue']=blue_agg

print(ml_data['Blue'].describe())

mean_blue=ml_data.groupby('Case of flush')['Blue'].mean()
median_blue=ml_data.groupby('Case of flush')['Blue'].median()
min_blue=ml_data.groupby('Case of flush')['Blue'].min()
max_blue=ml_data.groupby('Case of flush')['Blue'].max()
print(mean_blue)
print(median_blue)
print(min_blue)
print(max_blue)

fig, ax = plt.subplots()
sns.regplot(target,ml_data['Blue'],color='b')
ax.set_xticks(range(0,11))
plt.show()

sns.distplot(ml_data['Blue'],color='b')
plt.show()

#Red

red=ml_data[['Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]
sns.distplot(red,color='r')
plt.show()
print(stats.shapiro(red))

red['agg']=red[list(red.columns)].sum(axis=1)
red_agg=red['agg']
ml_data['Red']=red_agg

print(ml_data['Red'].describe())

mean_red=ml_data.groupby('Case of flush')['Red'].mean()
median_red=ml_data.groupby('Case of flush')['Red'].median()
min_red=ml_data.groupby('Case of flush')['Red'].min()
max_red=ml_data.groupby('Case of flush')['Red'].max()
print(mean_red)
print(median_red)
print(min_red)
print(max_red)

fig, ax = plt.subplots()
sns.regplot(target,ml_data['Red'],color='red')
ax.set_xticks(range(0,11))
plt.show()

sns.distplot(ml_data['Red'],color='r')
plt.show()

#Green

green=ml_data[['Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2']]
sns.distplot(green,color='g')
plt.show()
print(stats.shapiro(green))

green['agg']=green[list(green.columns)].sum(axis=1)
green_agg=green['agg']
ml_data['Green']=green_agg

lognormal=np.random.lognormal(size=122)
sns.distplot(lognormal)
plt.show()

print(ml_data['Green'].describe())

mean_green=ml_data.groupby('Case of flush')['Green'].mean()
median_green=ml_data.groupby('Case of flush')['Green'].median()
min_green=ml_data.groupby('Case of flush')['Green'].min()
max_green=ml_data.groupby('Case of flush')['Green'].max()
print(mean_green)
print(median_green)
print(min_green)
print(max_green)

fig, ax = plt.subplots()
sns.regplot(target,ml_data['Green'],color='green')
ax.set_xticks(range(0,11))
plt.show()

sns.distplot(ml_data['Green'],color='g')
plt.show()

#Displot of the sum of colors

ml_data['sum']=ml_data.apply(lambda row: row.Blue + row.Green + row.Red, axis=1)
print(ml_data['sum'].describe())
sns.distplot(ml_data['sum'])

#Relation plot color - Case of flush

fig, ax = plt.subplots()
sns.regplot(target,ml_data['sum'])
ax.set_xticks(range(0,11))
plt.show()

#Correlation

ml=ml_data[["Case of flush", "sum"]].dropna()
corrMatrix = ml.corr()
print(corrMatrix)
print(stats.pearsonr(ml['Case of flush'], ml['sum']))
#We reject H0 -> strong negative correlation 

print(stats.pearsonr(ml_data['Case of flush'], ml_data['Red']))
#We reject H0 -> strong negative correlation 
print(stats.pearsonr(ml_data['Case of flush'], ml_data['Blue']))
#We reject H0 -> strong negative correlation 
print(stats.pearsonr(ml_data['Case of flush'], ml_data['Green']))
#We reject H0 -> strong negative correlation 

#Number of trials per case of flush

count=ml_data['Case of flush'].value_counts()
print(count)
sns.countplot(data=ml_data,x='Case of flush',order=ml_data['Case of flush'].value_counts().index)
plt.show()

#Description group by Case of flush

mean_cof=ml_data.groupby('Case of flush')['sum'].mean()
median_cof=ml_data.groupby('Case of flush')['sum'].median()
min_cof=ml_data.groupby('Case of flush')['sum'].min()
max_cof=ml_data.groupby('Case of flush')['sum'].max()
print(mean_cof)
print(median_cof)
print(min_cof)
print(max_cof)


#Importance of Feature

color=ml_data[['Blue','Red','Green']]
X=color
y=target
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
