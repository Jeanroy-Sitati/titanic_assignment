# Titanic Survival Prediction – Feature Engineering Assignment

## Overview
This project covers the full data preparation pipeline for the Titanic dataset:
data cleaning, feature engineering, and feature selection. The goal is to produce
a clean, information-rich dataset ready for survival prediction modelling.

## Project Structure
```
titanic_assignment/
├── data/
│   ├── train.csv               ← raw dataset
│   ├── train_cleaned.csv       ← after Part 1 (cleaning)
│   ├── train_engineered.csv    ← after Part 2 (feature engineering)
│   └── train_selected.csv      ← after Part 3 (feature selection)
│
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb   ← full walkthrough with plots
│
├── scripts/
│   ├── data_cleaning.py        ← Part 1 logic
│   ├── feature_engineering.py  ← Part 2 logic
│   └── feature_selection.py    ← Part 3 logic
│
├── README.md
└── requirements.txt
```

## How to Run

### Option 1 – Jupyter Notebook (recommended for full exploration)
```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook Titanic_Feature_Engineering.ipynb
```

### Option 2 – Run scripts in order
```bash
pip install -r requirements.txt

# from project root
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
```
Each script reads from `data/` and writes its output back to `data/`.

## Approach

### Part 1 – Data Cleaning
| Column | Strategy | Reason |
|--------|----------|--------|
| Age (~20% missing) | Median impute grouped by Pclass+Sex | Group median is more accurate than a global one |
| Fare (<1% missing) | Global median impute | Only 1 row missing |
| Embarked (<1% missing) | Mode impute | 2–3 rows; Southampton dominates |
| Cabin (~77% missing) | Extract deck letter + binary indicator; drop raw | Too sparse to impute; deck letter retains signal |

Outlier handling: Fare is right-skewed with a few extreme luxury tickets. Capped at the 99th percentile.

### Part 2 – Features Engineered
- **FamilySize** = SibSp + Parch + 1
- **IsAlone** = 1 if FamilySize == 1
- **Title** – extracted from Name, rare titles bucketed into "Rare"
- **Deck** – first letter of Cabin (already done in cleaning)
- **AgeGroup** – Child (0–12) / Teen (13–17) / Adult (18–60) / Senior (60+)
- **FarePerPerson** = Fare / FamilySize
- **Fare_log**, **Age_log**, **FarePerPerson_log** – log1p transforms
- **Pclass_x_Fare** – interaction term
- One-hot encoding: Sex, Embarked, Title, Deck, AgeGroup

### Part 3 – Feature Selection
1. Dropped features with pairwise correlation > 0.90 (redundancy)
2. Ranked remaining features by Random Forest importance
3. Kept features with importance ≥ 1% + a few domain-knowledge must-keeps (Pclass, Age_log, FamilySize)

## Key Findings
- **Sex** is the single strongest predictor ("women and children first" was real)
- **Pclass** and **Fare_log** are nearly as important – class determined access to lifeboats
- **Title** is surprisingly powerful – it encodes gender, age, and class in one feature
- Solo travellers had notably lower survival rates than those travelling with 2–4 family members
- Deck information (even as a binary HasCabin flag) adds signal – proximity to lifeboats mattered
- The final selected feature set has ~20 columns, down from 30+ after OHE, with minimal redundancy
