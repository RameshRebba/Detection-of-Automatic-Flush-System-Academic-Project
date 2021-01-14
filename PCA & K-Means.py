#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:22:44 2020

@author: dorian
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from kneed import KneeLocator
from mlxtend.plotting import plot_pca_correlation_graph
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



data = pd.read_excel('/Users/dorian/Desktop/Data Science/Group 7/File 7 - Group7 - Dorian.xlsx')
ml_data=data[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2','Case of flush']] 
features=ml_data[['Blue LED 1 Photodiode 1','Blue LED 1 Photodiode 2','Green LED 1 Photodiode 1','Green LED 1 Photodiode 2','Red LED 1 Photodiode 1','Red LED 1 Photodiode 2','Blue LED 2 Photodiode 1','Blue LED 2 Photodiode 2','Green LED 2 Photodiode 1','Green LED 2 Photodiode 2','Red LED 2 Photodiode 1','Red LED 2 Photodiode 2']]  
target=ml_data['Case of flush']


numeric_df=features.apply(lambda x: np.log(x+1) if np.issubdtype(x.dtype,np.number)else x)
print(np.log(ml_data['Blue LED 1 Photodiode 1']))
#PCA

#Standardize features

x_norm = StandardScaler().fit_transform(ml_data)
np.mean(x_norm),np.std(x_norm)
feat_cols = ['feature'+str(i) for i in range(x_norm.shape[1])]
normalised_features = pd.DataFrame(x_norm,columns=feat_cols)


#Creation of the Components

n_components=12
pca_cof = PCA(n_components)
principalComponents_cof = pca_cof.fit_transform(x_norm)
print('Explained variation per principal component: {}'.format(pca_cof.explained_variance_ratio_))

#Optimal number of components: 4 (95% of dataset explained)

 

reduced = principalComponents_cof

#Append the principle components for each entry to the dataframe

for i in range(0, n_components):
    ml_data['PC' + str(i + 1)] = reduced[:, i]

print(ml_data.head())

#Scree plot

ind = np.arange(0, n_components)
(fig, ax) = plt.subplots(figsize=(8, 6))
sns.pointplot(x=ind, y=pca_cof.explained_variance_ratio_)
ax.set_title('Scree plot')
ax.set_xticks(ind)
ax.set_xticklabels(ind)
ax.set_xlabel('Component Number')
ax.set_ylabel('Explained Variance')
plt.show()

#Show the Case of Flush in terms of the first two PCs

g = sns.lmplot('PC1','PC2',hue='Case of flush',data=ml_data,fit_reg=False,scatter=True)
plt.show()

#Correlation Circle
features_name=['Blue 1-1','Blue 1-2','Green 1-1','Green 1-2','Red 1-1','Red 1-2','Blue 2-1','Blue 2-2','Green 2-1','Green 2-2','Red 2-1','Red 2-2','Case of flush']
fig, correlation_matrix=plot_pca_correlation_graph(x_norm,features_name,dimensions=(1,2,3,4))
plt.show()


#K-means

#Elbow Method - SSE = Sum Squared Error

sse=[]
for k in range(1,15):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(x_norm)
    sse.append(kmeans.inertia_)

kl=KneeLocator(range(1,15),sse,curve='convex',direction='decreasing')
nbr_cluster=kl.elbow
KneeLocator.plot_knee(kl)
plt.show()
print(nbr_cluster)

#Print the minimal SSE

km=KMeans(n_clusters=4)
km.fit(x_norm)
print(km.inertia_)

# Plotting the cluster centers and the data points on a 2D plane

plt.scatter(ml_data['PC1'],ml_data['PC2'])
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='x')
plt.xlabel('DIM 1')
plt.ylabel('DIM 2')   
plt.title('Data points and cluster centroids')
plt.show()

#Creation of Pipelines 

preprocessor=Pipeline([("scaler",StandardScaler()),("pca",PCA(n_components=4,random_state=42))])
clusterer=Pipeline([("kmeans",KMeans(n_clusters=nbr_cluster,random_state=42))])  
pipe=Pipeline([("preprocessor",preprocessor),("clusterer",clusterer)])
pipe.fit(ml_data)

pcadf=pd.DataFrame(pipe['preprocessor'].transform(ml_data),columns=['DIM 1','DIM 2','DIM 3','DIM 4'])
pcadf["predicted_cluster"]=pipe['clusterer']['kmeans'].labels_
pcadf['true_label']=ml_data['Case of flush']

#Scatter Plot of the clusters

scat=sns.scatterplot("DIM 1","DIM 2",data=pcadf,hue='true_label',style='predicted_cluster',s=50)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.0)
plt.show()

#Predictions

#GLM - All Dataset

X0=ml_data[['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12']]
X0 = sm.add_constant(X0)

Y=ml_data['Case of flush']

model0 = sm.GLM(Y, X0)
results0 = model0.fit()
print(results0.summary())

ypred0=results0.predict(X0)
print(ypred0)
print(ypred0.round())
predicted_class=ypred0.round()

ml_data["Bpred"]=predicted_class
Z=ml_data["Bpred"]
results = confusion_matrix(Y,Z)

print(results)
print(sns.heatmap(results, annot=True,  fmt='', cmap='Blues'))
print(classification_report(Y, Z))

#Logistic - Part of the Dataset

X_train, X_test, y_train, y_test = train_test_split(X0, Y, test_size = 0.4,random_state=42)
model = LogisticRegression(multi_class='multinomial', penalty='none').fit(X_train, y_train)
preds = model.predict(X_test)
print(preds)

print('Accuracy Score:', accuracy_score(y_test, preds)) 
print(classification_report(y_test, preds))



#VIF

vif_data=pd.DataFrame()
vif_data['feature']=pcadf.columns

vif_data['VIF']=[variance_inflation_factor(pcadf.values,i)
                     for i in range(len(pcadf.columns))]
print(vif_data)

#Correlation Matrix

corrMatrix=normalised_features.corr()
print(corrMatrix)





