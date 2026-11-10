# Analyse et prédiction des coûts d'assurance médicale

Ce projet analyse les coûts d'une assurance santé et construit un modèle pour les prédire.

---

## 1. Situation

Une assurance santé paie les soins de ses clients. Mais le coût change beaucoup d'un client à l'autre.

Nous avons un tableau de 1338 clients avec ces informations :

- l'âge
- le sexe
- le BMI (indice de masse corporelle)
- le nombre d'enfants
- fumeur ou non
- la région
- le coût médical (`charges`)

Le problème : on ne sait pas bien **pourquoi** certains clients coûtent plus cher. Et on ne peut pas **estimer le coût** d'un nouveau client.

---

## 2. Tâche

Mon travail avait deux buts :

1. **Comprendre** ce qui fait monter les coûts.
2. **Prédire** le coût médical d'un nouveau client avec un modèle.

Pour savoir si le modèle est bon, je mesure ses erreurs avec quatre chiffres : **MAE**, **MSE**, **RMSE** et **R²**.

---

## 3. Action

Voici ce que j'ai fait, étape par étape.

### a) Explorer les données (EDA)

- Je regarde les variables en nombres (âge, BMI), les variables en catégories (sexe, fumeur, région) et le nombre d'enfants.
- Je fais des graphiques pour voir comment les valeurs sont réparties.
- Je cherche les liens entre les variables et le coût.
- Je repère les valeurs très hautes ou très basses (valeurs aberrantes).

### b) Préparer les données

- Je mets les variables en nombres à la même échelle (standardisation).
- Je transforme les catégories en nombres (encodage).
- Je crée un **pipeline** : il fait toutes ces étapes tout seul sur de nouvelles données.

### c) Entraîner plusieurs modèles

J'ai testé :

- Régression linéaire
- Régression polynomiale

Je les compare sur le jeu de test avec **MSE** et **R²**.

### d) Prédire et vérifier

- Je prédis les coûts du jeu de test avec le meilleur modèle.
- Je trace un graphique **valeurs réelles vs valeurs prédites**. Plus les points sont proches de la ligne, meilleur est le modèle.

### e) Vérifier la régression linéaire

Pour que la régression linéaire soit fiable, je vérifie quelques règles :

| Ce qu'on regarde       | La règle                 | Ce que ça veut dire                                          |
| ---------------------- | ------------------------- | ------------------------------------------------------------- |
| X et Y                 | Linéarité               | le lien entre la variable et le coût ressemble à une droite |
| X et X                 | Pas de multicolinéarité | les variables ne disent pas la même chose entre elles        |
| Les erreurs (résidus) | Indépendance             | une erreur ne dépend pas d'une autre                         |
| Les erreurs (résidus) | Homoscédasticité        | les erreurs ont la même taille partout                       |
| Les erreurs (résidus) | Normalité                | les erreurs suivent une forme de cloche                       |

---

## 4. Résultat

Résultats de la **régression polynomiale** sur le jeu de test :

| Mesure | Valeur        | En clair                                                               |
| ------ | ------------- | ---------------------------------------------------------------------- |
| MAE    | 2 788,40      | en moyenne, le modèle se trompe d'environ 2 790 $                     |
| MSE    | 20 927 496,53 | erreur au carré (difficile à lire seule, on regarde plutôt le RMSE) |
| RMSE   | 4 574,66      | erreur typique, qui compte plus les grosses erreurs                    |
| R²    | 0,8652        | le modèle explique environ 87 % des changements de coût              |

- Le modèle prédit bien : il explique près de 87 % des variations du coût.
- L'erreur moyenne est d'environ 2 790 $. Le coût moyen d'un client est d'environ 13 270 $. L'erreur est donc raisonnable, mais pas nulle.
- Le RMSE est plus grand que le MAE. Cela veut dire que le modèle fait quelques grosses erreurs, surtout sur les clients très chers.
