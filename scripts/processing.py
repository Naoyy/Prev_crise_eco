from itertools import combinations
import matplotlib.pyplot as plt

import pandas as pd

import scripts.preprocessing as pp
import seaborn as sns
from sklearn.metrics import f1_score
import statsmodels.api as sm

dft=pp.dft

Xpropre,Ypropre=dft.drop(columns=["USRECD","DATE"]), dft["USRECD"]
max_lags = 10
lags_optimal = []

#estimer le meilleur lag au sens du pseudo R2
for variable in Xpropre.columns:
    optimal_lag = 0
    X=sm.add_constant(Xpropre[variable]) #initialiser le modèle de base (lag= 0)
    model = sm.Probit(Ypropre, X).fit()
    estrella_max = model.prsquared**((-2/len(Xpropre))*model.llf)     
    
    # Parcourir différents lags
    for lag in range(1, max_lags+1):
        
        X_temp= X.shift(lag)

        temp_df=pd.concat([Ypropre,X_temp], axis=1)
        temp_df.dropna(subset=temp_df.columns, how='any', inplace=True)
        temp_df.reset_index(drop=True, inplace=True)

        X_temp= temp_df.drop(columns=["USRECD"])
        y_temp= temp_df["USRECD"]

        model = sm.Probit(y_temp, X_temp).fit()
        estrella = model.prsquared**((-2/len(X_temp))*model.llf)  

        if ( estrella > estrella_max ):
            optimal_lag = lag
            estrella_max = estrella
     
    # Ajouter le lag optimal 
    lags_optimal.append(optimal_lag)
# Stocker nos lags optimaux pour chaque variable
lags_optimal = dict(zip(Xpropre.columns, lags_optimal))

# Créer les variables laggées en fonction des lags optimaux
for col in Xpropre.columns:
    if lags_optimal[col]!=0:
        dft[f"{col} lag {lags_optimal[col]}"]= dft[col].shift(lags_optimal[col])
    
dft.dropna(subset=dft.columns, how='any', inplace=True)
dft.reset_index(drop=True, inplace=True)

Xpropre,Ypropre=dft.drop(columns=["USRECD","DATE"]), dft["USRECD"]

# Classer nos variables par R2
classement=[]
student=[]
for col in Xpropre.columns : # parcourt chaque colonne 
    X = sm.add_constant(Xpropre[col]) # ajoute constante au modele
    model = sm.Probit(Ypropre, X).fit()
    
    estrella = model.prsquared**((-2/len(Xpropre))*model.llf) 
    t= model.pvalues[1]<0.01 #t-stat inférieur à 1%
    classement= classement + [{"name":col,"r2": estrella,"t-stat à 1%":t}] 
classement.sort(key=lambda x: x.get('r2'),reverse=True) #trie en fonction du r2


split =pd.to_datetime('01/03/1995') # date du split

train_df = dft[dft['DATE'] < split]
test_df = dft[dft['DATE'] >= split]

X_train, y_train = train_df.drop(columns=['DATE','USRECD']),train_df["USRECD"]
X_test,y_test=test_df.drop(columns=['DATE','USRECD']),test_df["USRECD"]

# récupérer 10 variables parmi 20
variables_30=[variable['name'] for variable in classement[:20]if variable['name'] not in ["CPF3MTB3Mx lag 1","PRFIx lag 1","CPF3MTB3Mx",
                                                                                             "UMCSENTx","TB3SMFFM lag 1","PRFIx","AAAFFM lag 3",
                                                                                             "IPDMAT","PNFIx","DMANEMP"]]

# Stocker la meilleur combinaison au sens du F1
meilleure_combinaison_F1= []
meilleur_F1 = 0

combinaisons_possibles = []

for i in range(1,len(variables_30)+1) : 

    # test toutes les combinaisons possibles
    combinaisons_i = combinations(variables_30, i)
    
    # rajoute les combi possibles
    combinaisons_possibles.extend(combinaisons_i)

for i in range(len(combinaisons_possibles)) : 

    X_train_F1=sm.add_constant(X_train[list(combinaisons_possibles[i])]) # ajoute la constante que par rapport au modèle testé
    probit_model_F1=sm.Probit(y_train,X_train_F1)
    result_F1=probit_model_F1.fit()

    X_test_F1=sm.add_constant(X_test[list(combinaisons_possibles[i])])
    pred_F1 = result_F1.predict(X_test_F1)

    f1 = f1_score(y_test, pred_F1.apply(lambda x: 1 if x>0.5 else 0))

    if f1 > meilleur_F1 :
        meilleur_F1 = f1
        meilleure_combinaison_F1 = probit_model_F1.exog_names
