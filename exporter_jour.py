#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporte les PREDICTIONS d'un ou PLUSIEURS jours dans predictions_jour.csv
(carte facon Geny + heure de depart), lu par l'application cloud.

A lancer en LOCAL apres scraper_pmu.py + build_features.py :
    python exporter_jour.py 2026-07-13 2026-07-14     (un ou plusieurs jours)
    python exporter_jour.py                            (aujourd'hui seul)

Dependances : pip install pandas numpy scikit-learn psycopg2-binary joblib requests
"""

import os
import sys
import getpass
import datetime
import warnings

import numpy as np
import pandas as pd
import requests

from backtest_roi import prepare_xy

DB_DSN = dict(
    dbname=os.environ.get("PGDATABASE", "pmu_trot"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", ""),
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
)
MODELE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modele_top3.joblib")
MODELE_G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modele_gagnant.joblib")
SORTIE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_jour.csv")
BASE_API = "https://online.turfinfo.api.pmu.fr/rest/client/1/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (perso)"}

COLS = ["date_course", "numero_reunion", "numero_course", "hippodrome", "nom_prix",
        "heure", "heure_depart_ms", "type_depart", "distance_m",
        "numero", "cheval", "sexe", "age", "driver", "entraineur",
        "ferrure", "distance_partant_m", "recul_m", "gains_carriere_eur",
        "musique", "cote_depart", "cote_finale", "p_top3", "p_gagne",
        "premier_d4", "premier_dp", "premier_da",
        "h_meilleure_reduc_hist", "mus_moy_pos5",
        "h_taux_top3_meme_ferrure_hist", "h_nb_meme_ferrure_hist",
        "drv_taux_top3_hist", "cd_taux_top3_hist", "cd_nb_hist",
        "h_top3_sur_hippo_hist", "h_nb_sur_hippo_hist"]

SQL = """
    SELECT f.*, ch.nom AS cheval, dr.nom AS driver, e.nom AS entraineur,
           h.nom AS hippodrome, c.numero_reunion, c.numero_course, c.nom_prix,
           p.musique, p.distance_partant_m
    FROM features_partant f
    JOIN courses c       ON c.id = f.course_id
    JOIN chevaux ch      ON ch.id = f.cheval_id
    LEFT JOIN drivers dr ON dr.id = f.driver_id
    LEFT JOIN entraineurs e ON e.id = f.entraineur_id
    JOIN hippodromes h   ON h.id = f.hippodrome_id
    JOIN partants p      ON p.id = f.partant_id
    WHERE f.date_course = %s
    ORDER BY c.numero_reunion, c.numero_course, f.numero
"""


def fetch_heures(date):
    d = datetime.date.fromisoformat(date)
    try:
        prog = requests.get(f"{BASE_API}/{d.strftime('%d%m%Y')}", headers=HEADERS,
                            timeout=20).json().get("programme", {})
    except Exception:
        return {}
    m = {}
    for reu in prog.get("reunions", []):
        nr = reu.get("numOfficiel")
        for cse in reu.get("courses", []):
            h = cse.get("heureDepart")
            if h:
                m[(nr, cse.get("numOrdre"))] = int(h)
    return m


def main():
    import psycopg2
    import joblib
    args = sys.argv[1:] or [datetime.date.today().isoformat()]
    dates = []
    for a in args:
        dd = a.strip()[:10]
        try:
            datetime.date.fromisoformat(dd)
            dates.append(dd)
        except ValueError:
            sys.exit(f"Date invalide (format AAAA-MM-JJ) : {a!r}")
    if not DB_DSN.get("password"):
        DB_DSN["password"] = getpass.getpass(
            f"Mot de passe PostgreSQL (utilisateur {DB_DSN['user']}) : ")
    d = joblib.load(MODELE)
    dg = joblib.load(MODELE_G) if os.path.exists(MODELE_G) else None
    if dg is None:
        print("  (modele_gagnant.joblib absent : p_gagne laissé vide. "
              "Lance entrainer_gagnant.py pour la value gagnant.)")

    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn = psycopg2.connect(**DB_DSN)
        for date in dates:
            df = pd.read_sql(SQL, conn, params=(date,))
            if df.empty:
                print(f"  {date} : aucune course en base (ignore).")
                continue
            X = prepare_xy(df, d["features"], d["cat"]).reindex(columns=d["columns"], fill_value=0)
            df["p_top3"] = d["model"].predict_proba(X)[:, 1]
            if dg is not None:
                Xg = prepare_xy(df, dg["features"], dg["cat"]).reindex(columns=dg["columns"], fill_value=0)
                df["p_gagne"] = dg["model"].predict_proba(Xg)[:, 1]
            else:
                df["p_gagne"] = np.nan
            heures = fetch_heures(date)
            df["heure_depart_ms"] = df.apply(
                lambda r: heures.get((r["numero_reunion"], r["numero_course"])), axis=1)
            df["heure"] = df["heure_depart_ms"].apply(
                lambda ms: datetime.datetime.fromtimestamp(ms / 1000).strftime("%H:%M")
                if pd.notna(ms) else "")
            frames.append(df.reindex(columns=COLS))
            print(f"  {date} : {len(df)} partants.")
    conn.close()
    if not frames:
        sys.exit("Aucune donnee exportee.")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(SORTIE, index=False, encoding="utf-8")
    print(f"\nTotal : {len(out)} partants sur {len(frames)} jour(s) -> {SORTIE}")
    print("Commite ce fichier sur GitHub pour mettre a jour l'appli cloud.")


if __name__ == "__main__":
    main()
