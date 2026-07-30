#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraîne le modèle Top 3 (gradient boosting calibré) sur tout l'historique
disponible et le SAUVEGARDE dans modele_top3.joblib, pour que l'application
Streamlit puisse prédire sans réentraîner.

À relancer quand tu veux réactualiser le modèle (après un backfill, etc.).

Dépendances : pip install pandas numpy scikit-learn psycopg2-binary joblib
Usage : python entrainer_modele.py
"""

import os
import sys
import getpass
import warnings

import numpy as np
import pandas as pd

from backtest_roi import NUM_FEATURES, CAT_FEATURES, prepare_xy
from build_features import compute_features, SQL_LOAD

DB_DSN = dict(
    dbname=os.environ.get("PGDATABASE", "pmu_trot"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", ""),
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
)
# Modèle INDÉPENDANT (aucune cote) : indispensable pour prédire la VEILLE, quand
# les cotes n'existent pas encore. On forme notre proba, puis on la compare au
# marché le lendemain.
FEATURES = [f for f in NUM_FEATURES if f not in ("cote_depart", "cote_rang")] + CAT_FEATURES
FICHIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modele_top3.joblib")


def main():
    import psycopg2
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    if not DB_DSN.get("password"):
        DB_DSN["password"] = getpass.getpass(
            f"Mot de passe PostgreSQL (utilisateur {DB_DSN['user']}) : ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn = psycopg2.connect(**DB_DSN)
        df_raw = pd.read_sql(SQL_LOAD, conn)     # tables brutes (pas features_partant)
    conn.close()
    df = compute_features(df_raw)                # features calculées EN MÉMOIRE
    df = df[df["cible"].notna()].copy()
    print(f"{len(df)} partants avec résultat pour l'entraînement.")

    df["date_course"] = pd.to_datetime(df["date_course"])
    X = prepare_xy(df, FEATURES, CAT_FEATURES)
    y = df["cible"].astype(int)

    # Contrôle rapide : AUC sur un hold-out temporel (le dernier mois)
    cutoff = df["date_course"].max() - pd.Timedelta(days=30)
    m_tr = df["date_course"] < cutoff
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          max_depth=4, l2_regularization=1.0, random_state=42)
    ctrl = CalibratedClassifierCV(base, method="isotonic", cv=3)
    ctrl.fit(X[m_tr.values], y[m_tr.values])
    if (~m_tr).sum() > 100:
        auc = roc_auc_score(y[~m_tr.values], ctrl.predict_proba(X[~m_tr.values])[:, 1])
        print(f"Contrôle AUC (hold-out dernier mois) : {auc:.3f}")

    # Modèle FINAL : entraîné sur TOUT l'historique
    base_f = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                            max_depth=4, l2_regularization=1.0, random_state=42)
    model = CalibratedClassifierCV(base_f, method="isotonic", cv=3)
    model.fit(X, y)

    joblib.dump({"model": model, "columns": list(X.columns),
                 "features": FEATURES, "cat": CAT_FEATURES}, FICHIER)
    print(f"Modèle sauvegardé : {FICHIER}")


if __name__ == "__main__":
    main()
