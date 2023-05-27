#Librairies
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import  SelectFromModel
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, auc,  roc_auc_score,f1_score, balanced_accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB

import statsmodels.api as sm

pd.set_option('display.max_columns', None) #afficher ttes les cols
sns.set_theme( palette="pastel") 

#Import
os.chdir("./data")
path=os.getcwd()

df=pd.read_csv(path+"\current.csv",sep=",",decimal=".")
df1=pd.read_csv(path+"\\USRECD.csv",sep=",")
NYSE=pd.read_csv(path+"\BOGZ1FL073164003Q.csv") # NYSE
Rate=pd.read_csv(path+"\BOGZ1FL073164013Q.csv") # Dow Jones
dft=df.drop([0,1,259]) #dft= df transformed; actor, transform, last line
dft.reset_index(drop=True, inplace=True)

#Préprocessing

#transformations des données
for col in df.columns: 
    if df[col].iloc[1]==2:
        dft[col]=dft[col].diff()
    if df[col].iloc[1]==3:
        dft[col]=dft[col].diff(2)
    if df[col].iloc[1]==4:
        dft[col]=dft[col].apply(np.log)
    if df[col].iloc[1]==5:
        dft[col]=dft[col].apply(np.log).diff()
    if df[col].iloc[1]==6:
        dft[col]=dft[col].apply(np.log).diff(2)
    if df[col].iloc[1]==7:
        dft[col] = ((dft[col]/ (dft[col].shift(1)))-1).diff()  


# appliquer le format date

dft["sasdate"]=pd.to_datetime(dft["sasdate"])
dft['sasdate'] = dft['sasdate'].dt.strftime('%d/%m/%Y')

df1['DATE'] = pd.to_datetime(df1['DATE'])
df1['DATE'] = df1['DATE'] + pd.DateOffset(months=2) # ajoute 2 mois 
df1['DATE'] = df1['DATE'].dt.strftime('%d/%m/%Y') # met la date dans le bon format

NYSE['DATE'] = pd.to_datetime(NYSE['DATE'])
NYSE['DATE'] = NYSE['DATE'] + pd.DateOffset(months=2) # ajoute 2 mois 
NYSE['DATE'] = NYSE['DATE'].dt.strftime('%d/%m/%Y')

Rate['DATE'] = pd.to_datetime(Rate['DATE'])
Rate['DATE'] = Rate['DATE'] + pd.DateOffset(months=2) # ajoute 2 mois 
Rate['DATE'] = Rate['DATE'].dt.strftime('%d/%m/%Y')

# ajouter les variables et supprimer les variables inutiles
merged_df = pd.merge(df1, dft, left_on='DATE', right_on='sasdate')
merged_df= pd.merge(merged_df, Rate, how = 'left',on="DATE")# Rate : 01/12/1970 à 01/12/2022 : BOGZ1FL073164013Q	
merged_df= pd.merge(merged_df, NYSE, how = 'left',on="DATE")# NYSE : 01/12/1945	à 01/12/2022 : BOGZ1FL073164003Q	
merged_df=merged_df.drop(columns=["sasdate"])
merged_df.drop(columns=["ACOGNOx","COMPRMS", "OPHMFG","ULCMFG","DRIWCIL","USEPUINDXM","CUSR0000SEHC"], inplace=True)
dft = merged_df

dft['DATE'] = pd.to_datetime(dft['DATE'])
dft['BOGZ1FL073164003Q'] = pd.to_numeric(dft['BOGZ1FL073164003Q'])

#imputation des valeurs manquantes par la moyenne en fonction de la période
date_ranges = [
    ("1959-01-03", "1969-01-03"),
    ("1969-01-03", "1979-01-03"),
    ("1979-01-03", "1989-01-03"),
    ("1989-01-03", "1999-01-03"),
    ("1999-01-03", "2009-01-03"),
    ("2009-01-03", "2019-01-03"),
    ("2019-01-03", "2022-01-12")]

dft_list = []

for start_date, end_date in date_ranges:
    dft_temp = dft.loc[(dft['DATE'] >= start_date) & (dft['DATE'] < end_date)].copy(deep=True)
    dft_temp.fillna(dft_temp.mean(), inplace=True)
    dft_list.append(dft_temp)

dft = pd.concat(dft_list)
dft.fillna(dft.median(), inplace=True)
