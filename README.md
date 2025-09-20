# Analyse des Coûts Médicaux

Analyse exploratoire et prédiction des coûts d’assurance médicale.

---

## Introduction

- Ce projet vise à **analyser et prédire les coûts d’assurance médicale** à partir de données démographiques et médicales.  
- L’objectif est de **comprendre les facteurs influençant les coûts** et de construire un modèle capable de **prévoir les dépenses médicales pour de nouveaux individus**.  
- Le projet suit un **workflow complet de Data Science** : exploration, prétraitement, modélisation, évaluation et prédiction.

---

## Objectifs

- **Analyse exploratoire (EDA)** : comprendre la structure des données, étudier les relations entre variables et identifier les valeurs aberrantes.  
- **Prétraitement des données** : nettoyer les données, encoder les variables catégorielles, normaliser si nécessaire et préparer un dataset prêt pour le modèle.  
- **Entraînement du modèle** : tester différents modèles de régression (linéaires, RandomForest, XGBoost, LightGBM) pour prédire les coûts.  
- **Évaluation du modèle** : mesurer la performance avec des métriques comme **MSE** et **R²**, et visualiser les résultats.  
- **Prédiction** : utiliser le modèle entraîné pour estimer les coûts pour de nouveaux individus.  

---

## Méthodologie

1. **Chargement et exploration des données**  
   - Analyse des variables continues, catégorielles et de comptage.  
   - Visualisations pour comprendre les distributions et relations.  

2. **Prétraitement**  
   - Standardisation des variables continues.  
   - Encodage des variables catégorielles.  
   - Pipeline complet pour transformer automatiquement les données.  

3. **Entraînement des modèles**  
   - Modèles testés : LinearRegression, Ridge, Lasso, RandomForest, SVR, GradientBoosting, XGBoost.  
   - Évaluation avec **MSE** et **R²** sur le jeu de test.  

4. **Optimisation du modèle**  
   - Hyperparamètres optimisés avec **GridSearchCV** sur XGBoost.  
   - Validation croisée pour choisir les meilleurs paramètres.  

5. **Prédiction et visualisation**  
   - Prédictions sur le jeu de test avec le meilleur modèle.  
   - Comparaison valeurs réelles vs prédites via scatter plot.  

---

## Résultats

- Le **meilleur modèle** : XGBoost avec R² ≈ 0.88 et MSE ≈ 11 784 586.  
- Les prédictions sont proches des valeurs réelles, ce qui montre que le modèle est **fiable et robuste**.  
- Pipeline sauvegardé pour **réutilisation future ou déploiement**.

---

## Utilisation

1. Cloner le projet :
```bash
git clone <repo_url>
cd project
pip install -r requirements.txt
