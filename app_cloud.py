#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Streamlit CLOUD (mot de passe) — lit predictions_jour.csv.
Onglets : Prédictions & Value (par course), Carte façon Geny, Top 5 value 2h.
Cotes en direct via l'API PMU. Aucune base ni modèle en ligne.

Mot de passe : défaut "cagnes2026". Sur Streamlit Cloud, mets dans les Secrets :
    app_password = "ton_mot_de_passe"

Dépendances (requirements.txt) : streamlit, pandas, numpy, requests
"""

import os
import time
import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_jour.csv")
BASE_API = "https://online.turfinfo.api.pmu.fr/rest/client/1/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (perso)"}

st.set_page_config(page_title="Value Trot PMU", layout="wide")


def acces_autorise():
    try:
        vrai = st.secrets.get("app_password", None)
    except Exception:
        vrai = None
    vrai = vrai or os.environ.get("APP_PASSWORD") or "cagnes2026"
    if st.session_state.get("ok"):
        return True
    p = st.text_input("🔒 Mot de passe d'accès", type="password")
    if p == vrai:
        st.session_state["ok"] = True
        return True
    if p:
        st.error("Mot de passe incorrect.")
    return False


def win_probs_from_odds(cotes):
    inv = 1.0 / np.asarray(cotes, float)
    s = inv.sum()
    return inv / s if s > 0 else inv


def harville_top3(win_probs):
    p = np.asarray(win_probs, float)
    n = len(p)
    P1 = p.copy(); P2 = np.zeros(n); P3 = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            dj = 1.0 - p[j]
            if dj <= 1e-12:
                continue
            P2[i] += p[j] * p[i] / dj
            for m in range(n):
                if m == i or m == j:
                    continue
                djm = 1.0 - p[j] - p[m]
                if djm > 1e-12:
                    P3[i] += p[j] * (p[m] / dj) * (p[i] / djm)
    return P1 + P2 + P3


def fmt_gains(v):
    return f"{int(v):,} €".replace(",", " ") if pd.notna(v) else ""


def fmt_restant(sec):
    if sec is None or pd.isna(sec):
        return ""
    if sec < 0:
        return "parti"
    h, m = int(sec // 3600), int((sec % 3600) // 60)
    return f"{h}h{m:02d}" if h else f"{m} min"


def calc_signaux(g):
    """Signaux par cheval, comparés au peloton. Deux niveaux : normal, et FORT
    (préfixé 🔴, marque la ligne comme 'fort' pour l'afficher en rouge)."""
    g = g.copy()
    rec = g["h_meilleure_reduc_hist"] if "h_meilleure_reduc_hist" in g else pd.Series(dtype=float)
    best = rec.min() if len(rec) and rec.notna().any() else np.nan
    med = rec.median() if len(rec) and rec.notna().any() else np.nan

    def tags(r):
        s, fort = [], False
        cd, cf = r.get("cote_depart"), r.get("cote_finale")
        drift = (cd / cf) if (pd.notna(cd) and pd.notna(cf) and cf) else np.nan
        if pd.notna(drift):
            if drift >= 1.40:
                s.append("🔴🔥steam++"); fort = True
            elif drift >= 1.15:
                s.append("🔥steam")
            elif drift <= 0.85:
                s.append("↘dérive")
        if r.get("premier_d4") == 1:
            s.append("1er déf.4")
        elif r.get("premier_dp") == 1 or r.get("premier_da") == 1:
            s.append("1er déf.")
        rr = r.get("h_meilleure_reduc_hist")
        if pd.notna(rr) and pd.notna(best):
            if rr <= best * 1.002 and pd.notna(med) and rr <= med * 0.985:
                s.append("🔴RECORD"); fort = True          # bien meilleur que le peloton
            elif rr <= best * 1.005:
                s.append("⏱record")
        nf, tf = (r.get("h_nb_meme_ferrure_hist") or 0), (r.get("h_taux_top3_meme_ferrure_hist") or 0)
        if nf >= 5 and tf >= 0.70:
            s.append("🔴ferrage"); fort = True
        elif nf >= 3 and tf >= 0.50:
            s.append("ferrage✓")
        mp = r.get("mus_moy_pos5")
        if pd.notna(mp):
            if mp <= 1.8:
                s.append("🔴forme"); fort = True
            elif mp <= 3:
                s.append("forme+")
        dv = r.get("drv_taux_top3_hist") or 0
        if dv >= 0.55:
            s.append("🔴driver"); fort = True
        elif dv >= 0.40:
            s.append("driver+")
        cn, ct = (r.get("cd_nb_hist") or 0), (r.get("cd_taux_top3_hist") or 0)
        if cn >= 5 and ct >= 0.70:
            s.append("🔴tandem"); fort = True
        elif cn >= 3 and ct >= 0.50:
            s.append("tandem✓")
        nbh = r.get("h_nb_sur_hippo_hist") or 0
        th = (r.get("h_top3_sur_hippo_hist") or 0) / nbh if nbh else 0
        if nbh >= 3 and th >= 0.70:
            s.append("🔴hippo"); fort = True
        elif nbh >= 2 and th >= 0.50:
            s.append("hippo✓")
        return pd.Series([" ".join(s), fort])

    g[["sig", "fort"]] = g.apply(tags, axis=1)
    return g


def ajoute_value(df, marge):
    df = df.copy()
    df["p_marche"] = np.nan
    for _, idx in df.groupby(["date_course", "numero_reunion", "numero_course"]).groups.items():
        cotes = df.loc[idx, "cote_finale"]
        if cotes.notna().sum() >= 4:
            df.loc[idx, "p_marche"] = harville_top3(
                win_probs_from_odds(cotes.fillna(cotes.max()).values))
    df["cote_pivot"] = (1.0 / df["p_top3"]).round(2)
    df["value"] = df["p_top3"] / df["p_marche"] - 1.0
    df["VALUE_ok"] = (df["value"] > marge) & (df["p_marche"] >= 0.10)
    if "heure_depart_ms" in df.columns:
        df["restant"] = df["heure_depart_ms"] / 1000.0 - time.time()
    else:
        df["restant"] = np.nan
    return df


@st.cache_data(ttl=120)
def charger():
    return pd.read_csv(CSV)


def fetch_cotes_live(date_str, numR, numC):
    d = datetime.date.fromisoformat(str(date_str)[:10])
    url = f"{BASE_API}/{d.strftime('%d%m%Y')}/R{int(numR)}/C{int(numC)}/participants"
    try:
        data = requests.get(url, headers=HEADERS, timeout=15).json()
    except Exception:
        return {}
    res = {}
    for p in data.get("participants", []):
        c = (p.get("dernierRapportDirect") or p.get("dernierRapportReference") or {}).get("rapport")
        if c:
            res[p.get("numPmu")] = float(c)
    return res


# ---------------------------------------------------------------------------
st.title("🏇 Value — courses de trot")
if not acces_autorise():
    st.stop()

try:
    df = charger()
except Exception:
    st.error("predictions_jour.csv introuvable (à générer avec exporter_jour.py "
             "puis commiter dans le dépôt).")
    st.stop()

dates = sorted(df["date_course"].astype(str).unique())
date_sel = st.sidebar.selectbox("Jour", dates, index=len(dates) - 1) if len(dates) > 1 else dates[0]
st.sidebar.markdown(f"**Courses du {date_sel}**")
marge = st.sidebar.slider("Marge de value mini", 0.0, 0.5, 0.15, 0.05)
top_n = st.sidebar.slider("Nombre de value à afficher (onglet 2h)", 3, 20, 5)
fenetre_h = st.sidebar.slider("Fenêtre de départ (heures)", 1, 6, 2)
df = ajoute_value(df, marge)

onglet_top, onglet_course, onglet_carte = st.tabs(
    [f"⏱️ Top {top_n} value ({fenetre_h}h)", "🎯 Prédictions & Value", "📋 Carte (Geny)"])

# ---- Onglet Top 5 value des 2 prochaines heures ---------------------------
with onglet_top:
    st.subheader(f"Top {top_n} value — départs dans les {fenetre_h} prochaines heures")
    cand = df[df["VALUE_ok"] & df["restant"].between(0, fenetre_h * 3600)].copy()
    if cand.empty:
        st.info(f"Aucune value crédible sur une course partant dans les {fenetre_h} h "
                "(ou heures de départ absentes du fichier — régénère l'export).")
    else:
        cand = cand.sort_values("value", ascending=False).head(top_n)
        top = pd.DataFrame({
            "Course": "R" + cand["numero_reunion"].astype(str) + "C" + cand["numero_course"].astype(str),
            "N°": cand["numero"].astype("Int64"),
            "Cheval": cand["cheval"],
            "Hippodrome": cand["hippodrome"],
            "Heure": cand["heure"],
            "Value %": (100 * cand["value"]).round(0),
            "Départ dans": cand["restant"].apply(fmt_restant),
        })
        st.dataframe(top, hide_index=True, use_container_width=True)

# ---- Sélecteur de course (pour les 2 autres onglets) ----------------------
dfj = df[df["date_course"].astype(str) == date_sel]
reunions = dfj[["numero_reunion", "numero_course", "hippodrome", "nom_prix"]].drop_duplicates().reset_index(drop=True)
labels = [f"R{r.numero_reunion}C{r.numero_course} — {r.hippodrome}" for _, r in reunions.iterrows()]
i = st.sidebar.selectbox("Course", options=range(len(reunions)), format_func=lambda k: labels[k])
sel = reunions.iloc[i]
g = dfj[(dfj.numero_reunion == sel.numero_reunion) & (dfj.numero_course == sel.numero_course)].copy()

if st.sidebar.button("🔄 Rafraîchir les cotes en direct"):
    live = fetch_cotes_live(date_sel, sel.numero_reunion, sel.numero_course)
    if live:
        g["cote_finale"] = g["numero"].map(live).fillna(g["cote_finale"])
        cf = g["cote_finale"]
        if cf.notna().sum() >= 4:
            g["p_marche"] = harville_top3(win_probs_from_odds(cf.fillna(cf.max()).values))
            g["value"] = g["p_top3"] / g["p_marche"] - 1.0
            g["VALUE_ok"] = (g["value"] > marge) & (g["p_marche"] >= 0.10)
        st.sidebar.success(f"Cotes mises à jour ({len(live)}).")
    else:
        st.sidebar.warning("Cotes live indisponibles.")

g = g.sort_values("p_top3", ascending=False).reset_index(drop=True)
g["rang"] = g.index + 1
g = calc_signaux(g)

with onglet_course:
    st.subheader(f"R{sel.numero_reunion}C{sel.numero_course} — {sel.hippodrome} "
                 f"— {sel.get('nom_prix', '') or ''}  ({g['heure'].iloc[0] if len(g) else ''})")
    aff = pd.DataFrame({
        "Rg": g["rang"], "N°": g["numero"].astype("Int64"),
        "Cheval": g["cheval"], "Driver": g["driver"],
        "P(Top3)": (100 * g["p_top3"]).round(1),
        "Cote pivot placé": g["cote_pivot"], "Cote": g["cote_finale"],
        "Value %": (100 * g["value"]).round(0),
        "✔": np.where(g["VALUE_ok"], "✅", ""), "Signaux": g["sig"],
    })
    forts = g["fort"].values
    sty = aff.style.apply(
        lambda col: ["color:#d00; font-weight:700" if forts[k] else "" for k in range(len(col))],
        subset=["Signaux"])
    st.dataframe(sty, hide_index=True, use_container_width=True)
    st.markdown(
        "**Légende des signaux** (aide à la lecture, à croiser avec ton œil) :\n"
        "- **🔥steam** : cote qui raccourcit (argent tardif souvent informé) · **↘dérive** : cote qui monte\n"
        "- **⏱record** : meilleur chrono passé du peloton (ou à 0,5 %) · **forme+** : place moyenne ≤ 3 sur ses dernières sorties\n"
        "- **ferrage✓** : ≥ 50 % de placé avec la ferrure du jour (≥ 3 courses) · **1er déf.4 / 1er déf.** : premier déferré (peut transformer)\n"
        "- **driver+** : driver ≥ 40 % de placé · **tandem✓** : couple cheval×driver ≥ 50 % (≥ 3 courses) · **hippo✓** : ≥ 50 % de placé sur cet hippodrome\n"
        "- **✅** (colonne ✔) : value crédible (modèle > marché au-dessus de ta marge, hors purs outsiders)\n"
        "- **Cote pivot placé** = 1/P(Top3) : la cote placé minimale pour que le pari soit intéressant.")

with onglet_carte:
    carte = g.sort_values("numero").copy()
    tab = pd.DataFrame({
        "N°": carte["numero"].astype("Int64"),
        "Cheval": carte["cheval"],
        "S/A": carte["sexe"].astype(str) + carte["age"].astype("Int64").astype(str),
        "Driver": carte["driver"],
        "Entraîneur": carte["entraineur"],
        "Déf.": carte["ferrure"],
        "Dist.": carte["distance_partant_m"].astype("Int64"),
        "Gains": carte["gains_carriere_eur"].apply(fmt_gains),
        "Musique": carte["musique"],
        "Cote": carte["cote_finale"],
        "P(Top3)": (100 * carte["p_top3"]).round(1),
    })
    st.dataframe(tab, hide_index=True, use_container_width=True)
    st.caption("Déf. = déferrage (D4/DA/DP/P4…). P(Top3) = probabilité de placer (modèle).")
