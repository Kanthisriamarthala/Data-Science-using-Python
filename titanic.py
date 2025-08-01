import sys
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def drop_irrelevant(df):
    return df.drop(columns=['passengerid', 'name', 'ticket', 'cabin'], errors='ignore')

def impute_age(df):
    # ✅ Fix index alignment using transform()
    df['age'] = df['age'].fillna(df.groupby('pclass')['age'].transform('median'))
    return df

def preprocess(df):
    df = clean_column_names(df)
    df = drop_irrelevant(df)
    df = impute_age(df)
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})

    df = pd.get_dummies(df, columns=['embarked', 'pclass'], prefix=['embarked', 'pclass'])
    return df

def descriptive(df):
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe(include='all').T)
    print("\nMissing values per column:\n", df.isnull().sum())

def plot_eda(df, outdir="eda_plots"):
    os.makedirs(outdir, exist_ok=True)
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        sns.histplot(df[col], kde=True)
        plt.savefig(f"{outdir}/{col}_hist.png"); plt.clf()
        sns.boxplot(x=df[col])
        plt.savefig(f"{outdir}/{col}_box.png"); plt.clf()
    if len(num_cols) > 1:
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        plt.savefig(f"{outdir}/corr.png"); plt.clf()
    print(f"EDA plots saved in: {outdir}/")

def model(df):
    X = df.drop(columns=['survived'])
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 200], 'min_samples_split': [2, 5]},
        cv=3, n_jobs=-1
    )
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_

    preds = clf.predict(X_test)
    print("\n=== MODEL METRICS ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
    print("\nTop 10 Feature Importances:\n", feat_imp.sort_values(ascending=False).head(10))
    return feat_imp

def main():
    if len(sys.argv) != 2:
        print("Usage: python titanic.py titanic.csv")
        return
    df = pd.read_csv(sys.argv[1])
    df = preprocess(df)
    descriptive(df)
    plot_eda(df)
    model(df)

if __name__ == "__main__":
    main()
