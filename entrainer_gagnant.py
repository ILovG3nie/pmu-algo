#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraîne le modèle GAGNANT (arrivé 1er), calibré, sur tout l'historique, et le
sauvegarde dans modele_gagnant.joblib. Même logique que entrainer_modele.py
(modèle INDÉPENDANT, sans cote), mais cible = victoire au lieu de Top 3.

Sert à l'app pour calculer la value GAGNANT en plus de la value PLACÉ, et choisir
le meilleur des deux marchés.

À relancer en même temps que entrainer_modele.py (après un backfill / changement
de features).

Dépendances : pip install pandas numpy scikit-learn psycopg2-binary joblib
Usage : python entrainer_gagnant.py
"""

import os
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
FEATURES = [f for f in NUM_FEATURES if f not in ("cote_depart", "cote_rang")] + CAT_FEATURES
FICHIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modele_gagnant.joblib")


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
    df["date_course"] = pd.to_datetime(df["date_course"])
    y = (df["place"] == 1).astype(int)           # 'place' vient de SQL_LOAD
    print(f"{len(df)} partants | {int(y.sum())} gagnants ({100*y.mean():.1f}%).")

    X = prepare_xy(df, FEATURES, CAT_FEATURES)

    cutoff = df["date_course"].max() - pd.Timedelta(days=30)
    m_tr = (df["date_course"] < cutoff).values
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          max_depth=4, l2_regularization=1.0, random_state=42)
    ctrl = CalibratedClassifierCV(base, method="isotonic", cv=3)
    ctrl.fit(X[m_tr], y[m_tr])
    if (~m_tr).sum() > 100:
        auc = roc_auc_score(y[~m_tr], ctrl.predict_proba(X[~m_tr])[:, 1])
        print(f"Contrôle AUC gagnant (hold-out dernier mois) : {auc:.3f}")

    base_f = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                            max_depth=4, l2_regularization=1.0, random_state=42)
    model = CalibratedClassifierCV(base_f, method="isotonic", cv=3)
    model.fit(X, y)
    joblib.dump({"model": model, "columns": list(X.columns),
                 "features": FEATURES, "cat": CAT_FEATURES}, FICHIER)
    print(f"Modèle gagnant sauvegardé : {FICHIER}")


if __name__ == "__main__":
    main()
