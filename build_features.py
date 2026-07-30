#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construction de la table `features_partant` (cible : placé Top 3).

Principe ANTI-FUITE : pour chaque partant, toute variable "historique" n'utilise
que des courses STRICTEMENT antérieures (dans le temps) à la course en cours.
Concrètement, les statistiques glissantes sont décalées (shift) pour exclure la
ligne courante ; l'ordre chronologique est (date_course, course_id, numero).

Deux familles de variables :
  - snapshots pré-course fournis par l'API (déjà sans fuite) : âge, gains,
    nb de courses/victoires/places en carrière, musique, cotes ;
  - variables glissantes calculées sur les données déjà collectées :
    forme du cheval, taux de placé du driver / entraîneur / tandem,
    performance sur l'hippodrome, jours depuis la dernière course, record glissant.

Dépendances : pip install pandas numpy psycopg2-binary
Usage :       python build_features.py
"""

import os
import sys
import re
import getpass

import numpy as np
import pandas as pd

DB_DSN = dict(
    dbname=os.environ.get("PGDATABASE", "pmu_trot"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", ""),
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
)

SQL_LOAD = """
SELECT p.id AS partant_id, c.id AS course_id, c.date_course,
       c.discipline, c.type_depart, c.distance_m, c.hippodrome_id,
       h.sens_corde, h.tour_de_piste_m, h.surface, h.longueur_ligne_droite_m,
       p.cheval_id, p.driver_id, p.entraineur_id,
       p.numero, p.corde, p.recul_m, p.ferrure,
       ch.sexe, ch.origine,
       EXTRACT(YEAR FROM age(c.date_course, ch.date_naissance))::int AS age,
       p.gains_carriere_eur, p.nb_courses_avant, p.nb_victoires_avant,
       p.nb_places_avant, p.gains_annee_eur, p.musique,
       p.cote_depart, p.cote_finale,
       r.place, COALESCE(r.disqualifie, false) AS disqualifie,
       r.reduction_km_cs,
       COALESCE(r.place_top3, false) AS cible
FROM partants p
JOIN courses c      ON c.id = p.course_id
JOIN hippodromes h  ON h.id = c.hippodrome_id
JOIN chevaux ch     ON ch.id = p.cheval_id
LEFT JOIN resultats r ON r.partant_id = p.id;
"""

# ---------------------------------------------------------------------------
# Parsing de la musique (forme récente codée, ex. "1a2aDa0a(25)6a5m")
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"(\d+|[A-Za-z])([am])")

def parse_musique(musique, n=5):
    """Retourne (moy_pos_n, nb_incidents_n, nb_perf) à partir de la musique.

    - chiffre 1..9 = place ; 0 = non placé (traité comme 10) ;
    - lettre (D, T, A, R, ...) = incident/disqualification (pénalité 11, comptée).
    Ne lit que de l'information passée -> sans fuite.
    """
    if not musique or not isinstance(musique, str):
        return (np.nan, 0, 0)
    perfs = []          # (position_numerique, est_incident)
    for val, _allure in _TOKEN.findall(musique):
        if val.isdigit():
            pos = int(val)
            perfs.append((10 if pos == 0 else pos, False))
        else:
            perfs.append((11, True))     # incident
    if not perfs:
        return (np.nan, 0, 0)
    last = perfs[:n]                     # la musique est déjà du + récent au + ancien
    positions = [p for p, _ in last]
    nb_inc = sum(1 for _, inc in last if inc)
    return (float(np.mean(positions)), nb_inc, len(perfs))


def _taux_glissant(df, cle, cible_col):
    """Taux moyen de `cible_col` sur les courses antérieures, par `cle` (groupby)."""
    g = df.groupby(cle)
    nb_avant = g.cumcount()                         # nb d'occurrences avant (0-based)
    somme_avant = g[cible_col].cumsum() - df[cible_col]
    taux = somme_avant / nb_avant.replace(0, np.nan)
    return nb_avant, taux


def compute_features(df):
    """Calcule toutes les variables. df = sortie brute de SQL_LOAD."""
    df = df.copy()
    df["date_course"] = pd.to_datetime(df["date_course"])
    df = df.sort_values(["date_course", "course_id", "numero"]).reset_index(drop=True)

    df["cible_i"] = df["cible"].astype(int)
    df["disq_i"] = df["disqualifie"].astype(int)

    # --- Snapshots carrière (API, pré-course) ------------------------------
    nbc = df["nb_courses_avant"].replace(0, np.nan)
    df["taux_victoire_carriere"] = df["nb_victoires_avant"] / nbc
    df["taux_place_carriere"] = df["nb_places_avant"] / nbc

    # --- Combinaisons intra-course (pré-course) ----------------------------
    gc = df.groupby("course_id")
    df["nb_partants"] = gc["partant_id"].transform("count")            # taille du champ
    df["cote_rang"] = gc["cote_finale"].rank(method="min")             # 1 = favori du marché
    df["ratio_gains_course"] = df["gains_carriere_eur"] / nbc          # gains moyens / course

    # --- Profil de piste (regroupe les hippodromes qui se ressemblent) -----
    def _bucket_tour(t):
        if pd.isna(t):
            return "NA"
        if t < 1150:
            return "petite"      # < 1150 m
        if t < 1400:
            return "moyenne"     # 1150-1400 m
        return "grande"          # >= 1400 m
    corde = df["sens_corde"].fillna("NA").astype(str)
    surf = df["surface"].fillna("NA").astype(str)
    df["profil_piste"] = (corde + "|" + df["tour_de_piste_m"].apply(_bucket_tour)
                          + "|" + surf)
    df["ligne_droite_m"] = df["longueur_ligne_droite_m"]

    # --- Musique -----------------------------------------------------------
    mus = df["musique"].apply(lambda s: pd.Series(parse_musique(s),
                              index=["mus_moy_pos5", "mus_nb_incidents5", "mus_nb_perf"]))
    df = pd.concat([df, mus], axis=1)

    # --- Hiérarchie intra-course SANS cote (force/faiblesse du favori) ------
    # "Où se situe ce cheval dans le peloton ?" et "le podium est-il verrouillé
    #  par un favori dominant, ou la course est-elle ouverte ?" — calculé sur des
    #  signaux pré-course (gains/course, forme, taux de placé carrière) : donc
    #  anti-fuite ET sans cote. Permet au modèle de baisser les outsiders quand un
    #  favori écrase le peloton (2 favoris = ~1 place de podium restante).
    gc2 = df.groupby("course_id")
    df["rang_ratio_gains"] = gc2["ratio_gains_course"].rank(method="min", ascending=False)
    df["rang_forme"] = gc2["mus_moy_pos5"].rank(method="min", ascending=True)        # 1 = meilleure forme
    df["rang_taux_place"] = gc2["taux_place_carriere"].rank(method="min", ascending=False)
    # écart au meilleur "gagne-pain" du peloton (0 = leader, -> +grand = loin derrière)
    leader = gc2["ratio_gains_course"].transform("max")
    df["ecart_leader_gains"] = (leader - df["ratio_gains_course"]) / leader.replace(0, np.nan)
    # domination : part de la "qualité" (gains/course) détenue par les 2 meilleurs
    # du peloton -> proche de 1 = 2 favoris qui verrouillent le podium (peu de place
    # pour les outsiders), plus bas = course ouverte. Même valeur pour toute la course.
    def _dom(s):
        v = s.dropna().sort_values(ascending=False)
        tot = v.sum()
        if len(v) < 2 or tot == 0:
            return np.nan
        return float(v.iloc[:2].sum() / tot)
    df["dom_course"] = gc2["ratio_gains_course"].transform(_dom)
    # nb de concurrents crédibles (taux de placé carrière >= 40 %) dans le peloton
    df["nb_chevaux_forts"] = gc2["taux_place_carriere"].transform(
        lambda s: int((s >= 0.40).sum()))

    # --- Forme du cheval (glissant, antérieur) -----------------------------
    gch = df.groupby("cheval_id")
    df["h_nb_courses_hist"] = gch.cumcount()
    df["h_cum_top3"] = gch["cible_i"].cumsum() - df["cible_i"]
    df["h_taux_top3_hist"] = df["h_cum_top3"] / df["h_nb_courses_hist"].replace(0, np.nan)
    df["h_cum_disq"] = gch["disq_i"].cumsum() - df["disq_i"]
    df["h_taux_disq_hist"] = df["h_cum_disq"] / df["h_nb_courses_hist"].replace(0, np.nan)
    df["h_jours_depuis_derniere"] = gch["date_course"].diff().dt.days
    df["h_meilleure_reduc_hist"] = gch["reduction_km_cs"].transform(
        lambda s: s.shift().cummin())

    # --- Ferrure : changement + premières déferrures (anti-fuite, métier) --
    prev_ferrure = gch["ferrure"].shift()                       # ferrure course précédente
    df["changement_ferrure"] = (prev_ferrure.notna() & (df["ferrure"] != prev_ferrure)).astype(int)
    # nb d'occurrences de chaque code AVANT la course courante (par cheval)
    deja = {}
    for code in ["D4", "P4", "DA", "DP", "PA", "PP"]:
        ind = (df["ferrure"] == code).astype(int)
        deja[code] = ind.groupby(df["cheval_id"]).cumsum() - ind
    f = df["ferrure"]
    df["premier_d4"] = ((f == "D4") & (deja["D4"] == 0)).astype(int)
    df["premier_p4"] = ((f == "P4") & (deja["P4"] == 0)).astype(int)
    # partiels : premier seulement si JAMAIS déferré/plaqué des 4 auparavant
    df["premier_dp"] = ((f == "DP") & (deja["DP"] == 0) & (deja["D4"] == 0)).astype(int)
    df["premier_da"] = ((f == "DA") & (deja["DA"] == 0) & (deja["D4"] == 0)).astype(int)
    df["premier_pp"] = ((f == "PP") & (deja["PP"] == 0) & (deja["P4"] == 0)).astype(int)
    df["premier_pa"] = ((f == "PA") & (deja["PA"] == 0) & (deja["P4"] == 0)).astype(int)

    # --- Driver / entraîneur / tandems (glissant) --------------------------
    df["drv_nb_hist"], df["drv_taux_top3_hist"] = _taux_glissant(df, "driver_id", "cible_i")
    df["ent_nb_hist"], df["ent_taux_top3_hist"] = _taux_glissant(df, "entraineur_id", "cible_i")
    df["cd_nb_hist"], df["cd_taux_top3_hist"] = _taux_glissant(
        df, ["cheval_id", "driver_id"], "cible_i")
    df["dh_nb_hist"], df["dh_taux_top3_hist"] = _taux_glissant(
        df, ["driver_id", "hippodrome_id"], "cible_i")

    # --- Perf passée du cheval SELON la ferrure du jour (anti-fuite) --------
    # "ce cheval réussit-il avec cette ferrure ?" (déferré transforme ou pas)
    df["h_nb_meme_ferrure_hist"], df["h_taux_top3_meme_ferrure_hist"] = _taux_glissant(
        df, ["cheval_id", "ferrure"], "cible_i")

    # --- Win sur le parcours : placé déjà sur cet hippodrome ---------------
    gchh = df.groupby(["cheval_id", "hippodrome_id"])
    df["h_nb_sur_hippo_hist"] = gchh.cumcount()
    df["h_top3_sur_hippo_hist"] = gchh["cible_i"].cumsum() - df["cible_i"]

    # --- Spécialiste des CONDITIONS : perf passée du cheval dans des courses
    #     aux MÊMES caractéristiques que celle du jour (anti-fuite, glissant).
    #     Complète la musique (forme brute) par "ce cheval réussit-il DANS CE
    #     type de course ?" : même distance, même mode de départ, même piste.
    def _bucket_dist(d):
        if pd.isna(d):
            return "NA"
        if d < 2100:
            return "sprint"      # < 2100 m
        if d < 2700:
            return "moyen"       # 2100-2700 m
        return "long"            # >= 2700 m
    df["_dist_bucket"] = df["distance_m"].apply(_bucket_dist)
    df["h_nb_meme_dist_hist"], df["h_taux_top3_meme_dist_hist"] = _taux_glissant(
        df, ["cheval_id", "_dist_bucket"], "cible_i")
    df["h_nb_meme_depart_hist"], df["h_taux_top3_meme_depart_hist"] = _taux_glissant(
        df, ["cheval_id", "type_depart"], "cible_i")
    df["h_nb_meme_profil_hist"], df["h_taux_top3_meme_profil_hist"] = _taux_glissant(
        df, ["cheval_id", "profil_piste"], "cible_i")

    # nettoyage colonnes intermédiaires
    df = df.drop(columns=["h_cum_top3", "h_cum_disq", "_dist_bucket"])
    return df


FEATURE_COLS = [
    "partant_id", "course_id", "date_course", "cheval_id", "driver_id",
    "entraineur_id", "hippodrome_id", "discipline", "type_depart",
    "sens_corde", "surface", "tour_de_piste_m", "ligne_droite_m", "profil_piste",
    "distance_m", "numero", "corde", "recul_m", "ferrure", "sexe", "origine",
    "age", "gains_carriere_eur", "gains_annee_eur",
    "nb_courses_avant", "nb_victoires_avant", "nb_places_avant",
    "taux_victoire_carriere", "taux_place_carriere",
    "nb_partants", "cote_rang", "ratio_gains_course",
    "rang_ratio_gains", "rang_forme", "rang_taux_place",
    "ecart_leader_gains", "dom_course", "nb_chevaux_forts",
    "mus_moy_pos5", "mus_nb_incidents5", "mus_nb_perf",
    "cote_depart", "cote_finale",
    "h_nb_courses_hist", "h_taux_top3_hist", "h_taux_disq_hist",
    "h_jours_depuis_derniere", "h_meilleure_reduc_hist",
    "changement_ferrure", "premier_d4", "premier_p4",
    "premier_dp", "premier_da", "premier_pp", "premier_pa",
    "drv_nb_hist", "drv_taux_top3_hist", "ent_nb_hist", "ent_taux_top3_hist",
    "cd_nb_hist", "cd_taux_top3_hist", "dh_nb_hist", "dh_taux_top3_hist",
    "h_nb_sur_hippo_hist", "h_top3_sur_hippo_hist",
    "h_nb_meme_ferrure_hist", "h_taux_top3_meme_ferrure_hist",
    "h_nb_meme_dist_hist", "h_taux_top3_meme_dist_hist",
    "h_nb_meme_depart_hist", "h_taux_top3_meme_depart_hist",
    "h_nb_meme_profil_hist", "h_taux_top3_meme_profil_hist",
    "cible",
]

DDL = """
DROP TABLE IF EXISTS features_partant;
CREATE TABLE features_partant (
    partant_id              INTEGER PRIMARY KEY,
    course_id               INTEGER,
    date_course             DATE,
    cheval_id               INTEGER,
    driver_id               INTEGER,
    entraineur_id           INTEGER,
    hippodrome_id           INTEGER,
    discipline              TEXT,
    type_depart             TEXT,
    sens_corde              TEXT,
    surface                 TEXT,
    tour_de_piste_m         INTEGER,
    ligne_droite_m          INTEGER,
    profil_piste            TEXT,
    distance_m              INTEGER,
    numero                  SMALLINT,
    corde                   SMALLINT,
    recul_m                 INTEGER,
    ferrure                 TEXT,
    sexe                    TEXT,
    origine                 TEXT,
    age                     INTEGER,
    gains_carriere_eur      NUMERIC(12,2),
    gains_annee_eur         NUMERIC(12,2),
    nb_courses_avant        INTEGER,
    nb_victoires_avant      INTEGER,
    nb_places_avant         INTEGER,
    taux_victoire_carriere  REAL,
    taux_place_carriere     REAL,
    nb_partants             INTEGER,
    cote_rang               REAL,
    ratio_gains_course      NUMERIC(12,2),
    rang_ratio_gains        REAL,
    rang_forme              REAL,
    rang_taux_place         REAL,
    ecart_leader_gains      REAL,
    dom_course              REAL,
    nb_chevaux_forts        INTEGER,
    mus_moy_pos5            REAL,
    mus_nb_incidents5       INTEGER,
    mus_nb_perf             INTEGER,
    cote_depart             NUMERIC(8,2),
    cote_finale             NUMERIC(8,2),
    h_nb_courses_hist       INTEGER,
    h_taux_top3_hist        REAL,
    h_taux_disq_hist        REAL,
    h_jours_depuis_derniere INTEGER,
    h_meilleure_reduc_hist  INTEGER,
    changement_ferrure      INTEGER,
    premier_d4              INTEGER,
    premier_p4              INTEGER,
    premier_dp              INTEGER,
    premier_da              INTEGER,
    premier_pp              INTEGER,
    premier_pa              INTEGER,
    drv_nb_hist             INTEGER,
    drv_taux_top3_hist      REAL,
    ent_nb_hist             INTEGER,
    ent_taux_top3_hist      REAL,
    cd_nb_hist              INTEGER,
    cd_taux_top3_hist       REAL,
    dh_nb_hist              INTEGER,
    dh_taux_top3_hist       REAL,
    h_nb_sur_hippo_hist     INTEGER,
    h_top3_sur_hippo_hist   INTEGER,
    h_nb_meme_ferrure_hist  INTEGER,
    h_taux_top3_meme_ferrure_hist REAL,
    h_nb_meme_dist_hist     INTEGER,
    h_taux_top3_meme_dist_hist REAL,
    h_nb_meme_depart_hist   INTEGER,
    h_taux_top3_meme_depart_hist REAL,
    h_nb_meme_profil_hist   INTEGER,
    h_taux_top3_meme_profil_hist REAL,
    cible                   BOOLEAN
);
"""

def main():
    import psycopg2
    import psycopg2.extras
    if not DB_DSN.get("password"):
        DB_DSN["password"] = getpass.getpass(
            f"Mot de passe PostgreSQL (utilisateur {DB_DSN['user']}) : ")
    conn = psycopg2.connect(**DB_DSN)
    df = pd.read_sql(SQL_LOAD, conn)
    print(f"{len(df)} partants chargés.")
    feats = compute_features(df)[FEATURE_COLS]

    # Option cloud : ne stocker que les N derniers jours (les features sont
    # calculées sur TOUT l'historique en mémoire, mais on n'écrit qu'une petite
    # fenêtre récente -> évite de saturer une base cloud à quota serré).
    jours = os.environ.get("FEATURES_JOURS")
    if jours:
        d = pd.to_datetime(feats["date_course"])
        seuil = d.max() - pd.Timedelta(days=int(jours))
        feats = feats[d >= seuil]
        print(f"Stockage limité aux {jours} derniers jours : {len(feats)} lignes.")

    # conversions pour insertion (NaN -> None, types Python natifs)
    feats = feats.astype(object).where(pd.notnull(feats), None)
    rows = [tuple(r) for r in feats.to_numpy()]

    cur = conn.cursor()
    cur.execute(DDL)
    cols = ",".join(FEATURE_COLS)
    psycopg2.extras.execute_values(
        cur, f"INSERT INTO features_partant ({cols}) VALUES %s", rows, page_size=1000)
    conn.commit()
    cur.close()
    conn.close()
    print(f"Table features_partant créée : {len(rows)} lignes, "
          f"{len(FEATURE_COLS)} colonnes.")

if __name__ == "__main__":
    main()
