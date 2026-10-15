# 🏆 CAN 2025 – Prédiction du vainqueur par Machine Learning

Projet de data science appliqué au football africain : prédire le résultat des matchs avec un **Random Forest**, puis simuler **1 000 fois** la CAN 2025 (Monte Carlo) pour estimer la probabilité de victoire de chaque sélection.


---

## 🎯 Objectif

Construire un pipeline complet, de la donnée brute à la simulation du tournoi :

1. Nettoyer et fusionner plusieurs sources de données.
2. Construire des variables explicatives (forme, historique CAN, confrontations directes…).
3. Entraîner un modèle qui prédit l'issue d'un match : victoire (W), nul (D) ou défaite (L).
4. Utiliser ce modèle pour simuler le tournoi complet et en déduire des probabilités de titre.

## 🔄 Pipeline

```
Données brutes → Nettoyage/Fusion → 20 features → Split chronologique
      → Preprocessing → Random Forest → Simulation Monte Carlo (1 000 tournois)
```

## 📊 Données

Fusion de **5 sources** en une base de **8 297 matchs** (2010 – 2024) :

| Source | Contenu |
|--------|---------|
| African national football 2010–2024 | Historique des matchs internationaux africains |
| Matchs de CAN | Résultats détaillés des éditions de la CAN |
| Classement FIFA | Rang des sélections |
| Statistiques d'équipes | Statistiques agrégées par sélection |
| Champions | Palmarès de la CAN |

## 🔧 Features (20 variables)

| Groupe | Variables |
|--------|-----------|
| Force & forme | Différence de classement FIFA, points et différence de buts sur les 5 derniers matchs |
| Historique CAN | Taux de victoire en CAN, nombre de titres |
| Confrontations directes | Nombre de H2H, taux de victoire H2H |
| Contexte | Pays hôte, phase du tournoi |
| Fatigue | Jours de repos depuis le dernier match |
| Variables dérivées | Écart de forme, écart de performance en CAN, dominance H2H, avantage de titres |

## 🤖 Modélisation

- **Split chronologique** (pour éviter toute fuite de données temporelle) :
  - Train : 2010–2020 (6 139 matchs)
  - Validation : 2021–2022 (1 343 matchs)
  - Test : 2023–2024 (815 matchs)
- **Preprocessing** : `StandardScaler` sur les features, `LabelEncoder` sur la cible.
- **Modèle** : Random Forest régularisé (profondeur limitée, feuilles minimales élevées) après un premier modèle en surapprentissage, avec Grid Search comparatif (243 fits).

### Résultats (jeu de test)

| Métrique | Valeur |
|----------|--------|
| Accuracy train / validation / test | 52,2 % / 51,5 % / 52,6 % |
| F1-macro | 0,504 |
| F1 par classe | W : 0,62 · L : 0,54 · D : 0,35 |

Le surapprentissage a été éliminé : l'écart train/test est quasi nul, contre plus de 20 points pour la première version.

## 🎲 Simulation Monte Carlo

Le tournoi est simulé 1 000 fois à partir du **tableau officiel des 1/8 de finale** (16 équipes qualifiées), avec prédiction ML de chaque match, génération de scores et gestion des tirs au but.

### Top 5 des probabilités de titre

| Rang | Équipe | Probabilité de titre |
|------|--------|---------------------|
| 1 | Égypte | 36,8 % |
| 2 | Algérie | 12,8 % |
| 3 | Sénégal | 12,6 % |
| 4 | Nigeria | 8,1 % |
| 5 | Maroc | 7,2 % |

## ⚠️ Limites

- Avec 3 classes, la classe majoritaire (victoire) représente environ 50 % des matchs du test : le modèle est donc seulement légèrement au-dessus d'un classifieur naïf, et les nuls restent difficiles à prédire.
- La confiance moyenne des prédictions est faible (~45 %) : les probabilités de titre sont des **estimations**, pas des certitudes.
- Les données incluent des matchs amicaux et de qualification, dont l'enjeu diffère d'un match de phase finale.
- Une compétition à élimination directe laisse une grande part au hasard (tirs au but, blessures, suspensions), non modélisée ici.

## 🚀 Lancer le projet

```bash
git clone https://github.com/Hasnae112/NOM_DU_REPO.git
cd NOM_DU_REPO
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook CAN_2025_FINAL_Restructure.ipynb
```

> Le notebook a été développé sous Google Colab : adapte les chemins `/content/...` à ton dossier local, et place les fichiers CSV dans un dossier `data/`.

## 🗂️ Structure

```
├── CAN_2025_FINAL_Restructure.ipynb   # Notebook principal
├── data/                              # Jeux de données (CSV)
├── models/                            # Modèle et transformers sauvegardés (.pkl)
└── README.md
```

## 🛠️ Stack

Python · Pandas · NumPy · Scikit-Learn · Matplotlib · Seaborn

## 👤 Auteure

**Hasnae Elballaoui** – Élève-ingénieure en IA et Transformation Digitale (ENSAH)
[LinkedIn](https://linkedin.com/in/hasnae-elballaoui-312209256) · [GitHub](https://github.com/Hasnae112)
