#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest ROI + analyse de variance (value betting).

Objectif : juger un modèle par le RENDEMENT NET simulé (pas par l'AUC), avec la
variance chiffrée, pour distinguer un vrai edge d'un coup de chance.

Méthode :
  - validation TEMPORELLE glissante (walk-forward) : on entraîne sur le passé,
    on parie sur le bloc suivant, on décale ;
  - calibration des probabilités (isotone) : indispensable pour que la value ait
    un sens (sinon les longshots créent de la fausse value) ;
  - value/EV : pour chaque partant, edge = p_modèle * cote - 1 ; on ne parie que
    si edge > seuil et que la cote est dans la plage autorisée (mise fixe) ;
  - les cotes PMU (rapports mutuels) incluent DÉJÀ le prélèvement : le rapport est
    ce qui est réellement payé, donc gain = mise * cote (rien à retrancher) ;
  - variance : ROI global + par an, IC bootstrap, drawdown maximal, Monte-Carlo
    de bankroll (mise fixe et Kelly).

MARCHÉ : ce script parie sur le GAGNANT (cote_finale = rapport SIMPLE_GAGNANT,
succès = arrivé 1er), car ce sont les cotes dont on dispose. Le PLACÉ (Top 3)
utilisera exactement le même code dès qu'on aura collecté les rapports "placé".

Dépendances : pip install pandas numpy scikit-learn psycopg2-binary
Usage :
    python backtest_roi.py --seuil 0.05
    python backtest_roi.py --seuil 0.10 --cote-max 15      # éviter les outsiders
"""

import os
import sys
import getpass
import argparse
import warnings

import numpy as np
import pandas as pd

DB_DSN = dict(
    dbname=os.environ.get("PGDATABASE", "pmu_trot"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", ""),
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
)

NUM_FEATURES = [
    "distance_m", "numero", "corde", "recul_m", "age",
    "gains_carriere_eur", "gains_annee_eur",
    "nb_courses_avant", "nb_victoires_avant", "nb_places_avant",
    "taux_victoire_carriere", "taux_place_carriere",
    "mus_moy_pos5", "mus_nb_incidents5", "mus_nb_perf",
    "h_nb_courses_hist", "h_taux_top3_hist", "h_taux_disq_hist",
    "h_jours_depuis_derniere", "h_meilleure_reduc_hist",
    "drv_nb_hist", "drv_taux_top3_hist", "ent_nb_hist", "ent_taux_top3_hist",
    "cd_nb_hist", "cd_taux_top3_hist", "dh_nb_hist", "dh_taux_top3_hist",
    "h_nb_sur_hippo_hist", "h_top3_sur_hippo_hist", "cote_depart",
    "nb_partants", "cote_rang", "ratio_gains_course",
    "rang_ratio_gains", "rang_forme", "rang_taux_place",
    "ecart_leader_gains", "dom_course", "nb_chevaux_forts",
    "tour_de_piste_m", "ligne_droite_m",
    "changement_ferrure", "premier_d4", "premier_p4",
    "premier_dp", "premier_da", "premier_pp", "premier_pa",
    "h_nb_meme_ferrure_hist", "h_taux_top3_meme_ferrure_hist",
    "h_nb_meme_dist_hist", "h_taux_top3_meme_dist_hist",
    "h_nb_meme_depart_hist", "h_taux_top3_meme_depart_hist",
    "h_nb_meme_profil_hist", "h_taux_top3_meme_profil_hist",
]
CAT_FEATURES = ["ferrure", "sexe", "origine", "type_depart", "discipline",
                "hippodrome_id", "sens_corde", "surface", "profil_piste"]


# ===========================================================================
#  Fonctions PURES (variance / bankroll) — testables sans base ni sklearn
# ===========================================================================
def bet_profit(odds, won, stake=1.0):
    """Profit net d'un pari mutuel : gain = mise*cote si gagné, sinon -mise."""
    return stake * (odds - 1.0) if won else -stake


def roi(profits, stakes):
    profits, stakes = np.asarray(profits, float), np.asarray(stakes, float)
    s = stakes.sum()
    return float(profits.sum() / s) if s > 0 else float("nan")


def bootstrap_roi_ci(profits, stakes, n_boot=2000, alpha=0.05, seed=42):
    """Intervalle de confiance du ROI par ré-échantillonnage des paris."""
    profits, stakes = np.asarray(profits, float), np.asarray(stakes, float)
    m = len(profits)
    if m == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, m, m)
        rois[i] = profits[idx].sum() / stakes[idx].sum()
    return (float(np.quantile(rois, alpha / 2)),
            float(np.quantile(rois, 1 - alpha / 2)))


def max_drawdown(cumulative_pnl):
    """Pire repli pic-à-creux d'une courbe de P&L cumulé."""
    c = np.asarray(cumulative_pnl, float)
    if c.size == 0:
        return 0.0
    peak = np.maximum.accumulate(c)
    return float(np.max(peak - c))


def kelly_fraction(p, odds):
    """Fraction de Kelly pour un pari : (p*cote - 1) / (cote - 1), bornée à [0,1]."""
    if odds <= 1:
        return 0.0
    f = (p * odds - 1.0) / (odds - 1.0)
    return float(min(max(f, 0.0), 1.0))


def simulate_bankroll(bets, mode="flat", stake=1.0, kelly_frac=0.25,
                      bankroll0=100.0, ruin_level=0.0):
    """Simule l'évolution d'une bankroll sur une séquence ordonnée de paris.

    bets : liste de dicts {odds, won, p} dans l'ordre chronologique.
    mode : 'flat' (mise fixe) ou 'kelly' (fraction de Kelly).
    Retourne (bankroll_finale, courbe, ruine_bool).
    """
    bk = bankroll0
    courbe = [bk]
    ruine = False
    for b in bets:
        if bk <= ruin_level:
            ruine = True
            courbe.append(bk)
            continue
        if mode == "kelly":
            mise = bk * kelly_frac * kelly_fraction(b["p"], b["odds"])
        else:
            mise = min(stake, bk)
        bk += bet_profit(b["odds"], b["won"], mise)
        courbe.append(bk)
    if bk <= ruin_level:
        ruine = True
    return float(bk), courbe, ruine


# ===========================================================================
#  Backtest (scikit-learn) — walk-forward
# ===========================================================================
def prepare_xy(df, features, cat_features):
    return pd.get_dummies(df[features], columns=cat_features, dummy_na=False)


def walk_forward_bets(df, features, cat_features, seuil,
                      cote_min=0.0, cote_max=None, calibrer=True):
    """Entraîne mois par mois (walk-forward) et renvoie les paris value hors-échantillon.

    - calibrer : enrobe le modèle d'une calibration isotone (probabilités fiables) ;
    - cote_min / cote_max : ne parie que dans cette plage de cotes.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV

    df = df.copy()
    df["date_course"] = pd.to_datetime(df["date_course"])
    df["mois"] = df["date_course"].dt.to_period("M")
    mois = sorted(df["mois"].unique())
    paris = []
    for i in range(1, len(mois)):
        tr = df[df["mois"] < mois[i]]
        te = df[df["mois"] == mois[i]]
        if len(tr) < 500 or te.empty:
            continue
        Xtr = prepare_xy(tr, features, cat_features)
        Xte = prepare_xy(te, features, cat_features).reindex(columns=Xtr.columns, fill_value=0)
        ytr = tr["gagnant"].astype(int)
        base = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=42)
        model = CalibratedClassifierCV(base, method="isotonic", cv=3) if calibrer else base
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        te = te.assign(p=p)
        te = te[te["cote_finale"].notna()]
        if cote_min:
            te = te[te["cote_finale"] >= cote_min]
        if cote_max:
            te = te[te["cote_finale"] <= cote_max]
        te = te.assign(edge=te["p"] * te["cote_finale"] - 1.0)
        sel = te[te["edge"] > seuil]
        for _, r in sel.iterrows():
            paris.append(dict(date=r["date_course"], odds=float(r["cote_finale"]),
                              p=float(r["p"]), won=bool(r["gagnant"])))
    return pd.DataFrame(paris)


def rapport(paris, mise=1.0, kelly_frac=0.25):
    if paris.empty:
        print("Aucun pari value sélectionné (seuil trop haut, plage de cotes trop "
              "étroite, ou données trop peu nombreuses).")
        return
    paris = paris.sort_values("date").reset_index(drop=True)
    paris["stake"] = mise
    paris["profit"] = [bet_profit(o, w, mise) for o, w in zip(paris["odds"], paris["won"])]

    r = roi(paris["profit"], paris["stake"])
    lo, hi = bootstrap_roi_ci(paris["profit"].values, paris["stake"].values)
    dd = max_drawdown(paris["profit"].cumsum().values)
    n, nw = len(paris), int(paris["won"].sum())

    print("\n==================== BACKTEST ROI (marché GAGNANT) ====================")
    print(f"Paris value : {n}  |  gagnés : {nw} ({100*nw/n:.1f}%)  "
          f"|  cote moyenne : {paris['odds'].mean():.2f}")
    print(f"ROI global  : {100*r:+.2f}%   (IC 95% bootstrap : "
          f"{100*lo:+.2f}% à {100*hi:+.2f}%)")
    if lo > 0:
        verdict = "edge SIGNIFICATIF (IC entièrement au-dessus de 0)"
    elif hi < 0:
        verdict = "stratégie PERDANTE significative (IC entièrement sous 0)"
    else:
        verdict = "non distinguable de zéro (IC chevauche 0)"
    print(f"Verdict variance : {verdict}")
    print(f"Drawdown max : {dd:.1f} unités de mise")

    paris["annee"] = pd.to_datetime(paris["date"]).dt.year
    print("\nROI par année :")
    for an, g in paris.groupby("annee"):
        print(f"  {an} : {len(g):5d} paris, ROI {100*roi(g['profit'], g['stake']):+.2f}%")

    bets = paris[["odds", "won", "p"]].to_dict("records")
    rng = np.random.default_rng(0)
    finals_flat, finals_kelly, ruines = [], [], 0
    for _ in range(500):
        seq = [bets[k] for k in rng.permutation(len(bets))]
        bf, _, _ = simulate_bankroll(seq, "flat", stake=mise, bankroll0=100)
        bk, _, ruine = simulate_bankroll(seq, "kelly", kelly_frac=kelly_frac, bankroll0=100)
        finals_flat.append(bf); finals_kelly.append(bk); ruines += int(ruine)
    print(f"\nMonte-Carlo bankroll (départ 100, {len(bets)} paris, 500 tirages) :")
    print(f"  Mise fixe  : médiane {np.median(finals_flat):.0f} "
          f"(5e-95e pct : {np.quantile(finals_flat,0.05):.0f}–{np.quantile(finals_flat,0.95):.0f})")
    print(f"  Kelly {kelly_frac:.2f} : médiane {np.median(finals_kelly):.0f} "
          f"(5e-95e pct : {np.quantile(finals_kelly,0.05):.0f}–{np.quantile(finals_kelly,0.95):.0f})")
    print(f"  Probabilité de ruine (Kelly) : {100*ruines/500:.1f}%")
    print("======================================================================")
    print("Note : backtest sur la cote FINALE (optimiste, non connue au pari).")
    print("Test réaliste : rejouer en sélectionnant sur cote_depart (cote du matin).")


def main():
    import psycopg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--seuil", type=float, default=0.05, help="edge minimal (0.05 = +5%)")
    ap.add_argument("--mise", type=float, default=1.0)
    ap.add_argument("--kelly", type=float, default=0.25, help="fraction de Kelly")
    ap.add_argument("--cote-min", type=float, default=0.0, help="cote minimale pariable")
    ap.add_argument("--cote-max", type=float, default=None, help="cote maximale pariable (ex. 15)")
    ap.add_argument("--sans-calibration", action="store_true", help="desactive la calibration")
    args = ap.parse_args()

    if not DB_DSN.get("password"):
        DB_DSN["password"] = getpass.getpass(
            f"Mot de passe PostgreSQL (utilisateur {DB_DSN['user']}) : ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn = psycopg2.connect(**DB_DSN)
        df = pd.read_sql("""
            SELECT f.*, (r.place = 1) AS gagnant
            FROM features_partant f
            JOIN resultats r ON r.partant_id = f.partant_id
        """, conn)
    conn.close()
    df["gagnant"] = df["gagnant"].fillna(False)
    print(f"{len(df)} partants chargés ; {int(df['gagnant'].sum())} gagnants.")

    paris = walk_forward_bets(df, NUM_FEATURES + CAT_FEATURES, CAT_FEATURES, args.seuil,
                              cote_min=args.cote_min, cote_max=args.cote_max,
                              calibrer=not args.sans_calibration)
    rapport(paris, mise=args.mise, kelly_frac=args.kelly)


if __name__ == "__main__":
    main()
