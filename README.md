# Prévision de crises économiques

Dans le cadre du cours de Séries Temporelles, nous travaillons sur le sujet suivant:

Il a été noté par Mishkin et Estrella (1995) qu’une inversion de la courbe des taux avait un pouvoir prédictif concernant l’apparition de crises économiques. Cette relation au départ purement empirique a ensuite été théorisée et étendue pour prendre en compte d’autres variables explicatives. A l’aide de modèles logit/probit, sur données américaines et en se servant des données du NBER (datation des crises), vous testerez la pertinence de ces travaux, en in et out sample, en utilisant la courbe ROC pour analyser le contenu prédictif, en tentant d’enrichir le modèle,….. Techniques : Modèles logit/probit, simulations, courbe ROC

## Télécharger le projet
 Pour explorer le contenu du projet:

Créer un environnement virtuel 

```
py -m pip install virtualenv
py -m venv [Nom De L'environnement virtuel souhaité]
.\[Nom De L'environnement virtuel souhaité]\Scripts\activate
pip install jupyter
pip install -r requirements.txt
```

On peut ensuite accéder et faire marcher les notebooks.

## Organisation du projet
Ce dossier est composé de plusieurs élements:
- 3 notebooks: 
    - Processing (sélection des lags, sélection de variables)
    - Modelling and Challenge (Modèle Probit, Prolongement et Applications de modèles ML et autres méthodes)
- data: 
    - external: données de bases externes comme la variable de récession, le NYSE, Dow Jones et données trimestrielles
- reports:
    - Rapport écrit de notre analyse
    - Papier de référence d'Estrella et Mishkin
    - 2020-005: descriptif de nos variables à partir de la page 38
- scripts: 
    - pré processing (types de données, gestion des valeurs manquantes)
    - processing 
    
