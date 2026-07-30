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
            mv = (drift - 1) * 100
            if drift >= 1.40:
                s.append(f"🔴🔥steam++ ({mv:+.0f}%)"); fort = True
            elif drift >= 1.15:
                s.append(f"🔥steam ({mv:+.0f}%)")
            elif drift <= 0.85:
                s.append(f"↘dérive ({mv:+.0f}%)")
        if r.get("premier_d4") == 1:
            s.append("1er déf.4")
        elif r.get("premier_dp") == 1 or r.get("premier_da") == 1:
            s.append("1er déf.")
        rr = r.get("h_meilleure_reduc_hist")
        if pd.notna(rr) and pd.notna(best):
            if rr <= best * 1.002 and pd.notna(med) and rr <= med * 0.985:
                s.append("🔴RECORD (top peloton)"); fort = True   # bien meilleur que le peloton
            elif rr <= best * 1.005:
                s.append("⏱record")
        nf, tf = (r.get("h_nb_meme_ferrure_hist") or 0), (r.get("h_taux_top3_meme_ferrure_hist") or 0)
        if nf >= 5 and tf >= 0.70:
            s.append(f"🔴ferrage ({tf*100:.0f}%)"); fort = True
        elif nf >= 3 and tf >= 0.50:
            s.append(f"ferrage✓ ({tf*100:.0f}%)")
        mp = r.get("mus_moy_pos5")
        if pd.notna(mp):
            if mp <= 1.8:
                s.append(f"🔴forme (pos {mp:.1f})"); fort = True
            elif mp <= 3:
                s.append(f"forme+ (pos {mp:.1f})")
        dv = r.get("drv_taux_top3_hist") or 0
        if dv >= 0.55:
            s.append(f"🔴driver ({dv*100:.0f}%)"); fort = True
        elif dv >= 0.40:
            s.append(f"driver+ ({dv*100:.0f}%)")
        cn, ct = (r.get("cd_nb_hist") or 0), (r.get("cd_taux_top3_hist") or 0)
        if cn >= 5 and ct >= 0.70:
            s.append(f"🔴tandem ({ct*100:.0f}%)"); fort = True
        elif cn >= 3 and ct >= 0.50:
            s.append(f"tandem✓ ({ct*100:.0f}%)")
        nbh = r.get("h_nb_sur_hippo_hist") or 0
        th = (r.get("h_top3_sur_hippo_hist") or 0) / nbh if nbh else 0
        if nbh >= 3 and th >= 0.70:
            s.append(f"🔴hippo ({th*100:.0f}%)"); fort = True
        elif nbh >= 2 and th >= 0.50:
            s.append(f"hippo✓ ({th*100:.0f}%)")
        return pd.Series([" ".join(s), fort])

    g[["sig", "fort"]] = g.apply(tags, axis=1)
    return g


def ajoute_value(df, marge, cote_max=12.0):
    df = df.copy()
    df["p_marche"] = np.nan
    for _, idx in df.groupby(["date_course", "numero_reunion", "numero_course"]).groups.items():
        cotes = df.loc[idx, "cote_finale"]
        if cotes.notna().sum() >= 4:
            df.loc[idx, "p_marche"] = harville_top3(
                win_probs_from_odds(cotes.fillna(cotes.max()).values))
    df["cote_pivot"] = (1.0 / df["p_top3"]).round(2)
    # --- value PLACÉ (modèle Top 3 vs marché placé implicite) --------------
    df["value_place"] = df["p_top3"] / df["p_marche"] - 1.0
    ok_place = ((df["value_place"] > marge) & (df["p_marche"] >= 0.12)
                & (df["cote_finale"] <= cote_max))
    # --- value GAGNANT (proba de gagner × cote gagnant), si dispo ----------
    if "p_gagne" in df.columns and df["p_gagne"].notna().any():
        df["value_gagnant"] = df["p_gagne"] * df["cote_finale"] - 1.0
        ok_gagnant = ((df["value_gagnant"] > marge) & (df["p_gagne"] >= 0.08)
                      & (df["cote_finale"] <= cote_max * 2))
    else:
        df["value_gagnant"] = np.nan
        ok_gagnant = pd.Series(False, index=df.index)
    # --- on garde le MEILLEUR pari crédible des deux marchés ---------------
    vg = df["value_gagnant"].where(ok_gagnant, other=-np.inf)
    vp = df["value_place"].where(ok_place, other=-np.inf)
    df["type_pari"] = np.where(ok_gagnant & (vg >= vp), "Gagnant", "Placé")
    df["VALUE_ok"] = ok_place | ok_gagnant
    best = np.maximum(vg, vp)
    df["value"] = np.where(np.isfinite(best), best, df["value_place"])
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


def voisins_meme_reunion(reunions, cur):
    """Index de la course précédente / suivante DANS LA MÊME réunion (ou None)."""
    R = reunions.iloc[cur]["numero_reunion"]
    idxs = [k for k in range(len(reunions)) if reunions.iloc[k]["numero_reunion"] == R]
    k = idxs.index(cur)
    prev = idxs[k - 1] if k > 0 else None
    nxt = idxs[k + 1] if k < len(idxs) - 1 else None
    return prev, nxt


def barre_nav(titre, reunions, cur, prefixe):
    """Titre à gauche + flèches ◀ ▶ en haut à droite (course précédente/suivante
    de la même réunion). Synchronisé avec le sélecteur de la barre latérale."""
    prev, nxt = voisins_meme_reunion(reunions, cur)
    c1, c2, c3 = st.columns([8, 1, 1])
    c1.subheader(titre)
    if c2.button("◀", key=f"{prefixe}_prev", disabled=prev is None,
                 use_container_width=True, help="Course précédente (même réunion)"):
        st.session_state["cur"] = prev
        st.rerun()
    if c3.button("▶", key=f"{prefixe}_next", disabled=nxt is None,
                 use_container_width=True, help="Course suivante (même réunion)"):
        st.session_state["cur"] = nxt
        st.rerun()


def mise_conseillee(value, p_top3, value_ok):
    """Mise en UNITÉS (1 à 3), seulement sur une value crédible, dosée par la
    confiance (proba de placer) et l'ampleur de la value. C'est une aide au
    DOSAGE, pas un système gagnant (le placé reste négatif net de prélèvement)."""
    if not value_ok:
        return 0
    u = 1
    if value >= 0.25:          # value nette
        u += 1
    if p_top3 >= 0.45:         # confiance (proba de placer élevée)
        u += 1
    return min(u, 3)


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
cote_max_value = st.sidebar.slider("Cote gagnant max pour signaler ✅", 5, 30, 12)
top_n = st.sidebar.slider("Nombre de value à afficher (onglet 2h)", 3, 20, 5)
fenetre_h = st.sidebar.slider("Fenêtre de départ (heures)", 1, 6, 2)
df = ajoute_value(df, marge, cote_max_value)

onglet_top, onglet_plan, onglet_course, onglet_carte = st.tabs(
    [f"⏱️ Top {top_n} value ({fenetre_h}h)", "🎫 Plan du jour",
     "🎯 Prédictions & Value", "📋 Carte (Geny)"])

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
            "Value placé %": (100 * cand["value_place"]).round(0),
            "Value gagnant %": (100 * cand["value_gagnant"]).round(0),
            "Mise": cand.apply(lambda r: f"{mise_conseillee(r['value'], r['p_top3'], True)}u", axis=1),
            "Départ dans": cand["restant"].apply(fmt_restant),
        })
        st.dataframe(top, hide_index=True, use_container_width=True, column_config={
            "Value placé %": st.column_config.NumberColumn(format="%+.0f"),
            "Value gagnant %": st.column_config.NumberColumn(format="%+.0f")})

# ---- Sélecteur de course (pour les 2 autres onglets) ----------------------
dfj = df[df["date_course"].astype(str) == date_sel]
reunions = (dfj[["numero_reunion", "numero_course", "hippodrome", "nom_prix"]]
            .drop_duplicates()
            .sort_values(["numero_reunion", "numero_course"])
            .reset_index(drop=True))
labels = [f"R{r.numero_reunion}C{r.numero_course} — {r.hippodrome}" for _, r in reunions.iterrows()]

# index de course courant, PARTAGÉ entre le sélecteur latéral et les flèches
if st.session_state.get("date_prev") != date_sel:
    st.session_state["cur"] = 0                      # reset si on change de jour
    st.session_state["date_prev"] = date_sel
cur = min(st.session_state.get("cur", 0), len(reunions) - 1)
i_sb = st.sidebar.selectbox("Course", options=range(len(reunions)),
                            index=cur, format_func=lambda k: labels[k])
if i_sb != cur:                                      # l'utilisateur a changé via le menu
    cur = i_sb
st.session_state["cur"] = cur
sel = reunions.iloc[cur]

# ---- Onglet Plan du jour (toutes les values de la journée) ----------------
with onglet_plan:
    st.subheader(f"🎫 Plan de jeu — {date_sel}")
    vals = dfj[dfj["VALUE_ok"]].copy()
    if vals.empty:
        st.info("Aucune value crédible aujourd'hui avec tes réglages "
                "(baisse la marge ou monte la cote max dans la barre latérale).")
    else:
        cnt = vals.groupby(["numero_reunion", "numero_course"]).size()
        vals["n_val"] = [int(cnt[(r.numero_reunion, r.numero_course)]) for r in vals.itertuples()]
        vals["mise_u"] = vals.apply(
            lambda r: mise_conseillee(r["value"], r["p_top3"], True), axis=1)
        ncourses, nmulti = len(cnt), int((cnt >= 2).sum())
        ntickets, nunites = len(vals), int(vals["mise_u"].sum())
        st.markdown(f"**{ntickets} tickets** sur **{ncourses} courses** "
                    f"(dont **{nmulti}** à plusieurs values) · total **{nunites} unités** "
                    f"— soit {nunites}× ta mise de base.")
        vals = vals.sort_values(["n_val", "value"], ascending=[False, False])
        plan = pd.DataFrame({
            "Course": "R" + vals["numero_reunion"].astype(str) + "C" + vals["numero_course"].astype(str),
            "Hippo": vals["hippodrome"],
            "Heure": vals["heure"] if "heure" in vals else "",
            "N°": vals["numero"].astype("Int64"),
            "Cheval": vals["cheval"],
            "Cote": vals["cote_finale"],
            "P(Top3)": (100 * vals["p_top3"]).round(1),
            "Value placé %": (100 * vals["value_place"]).round(0),
            "Value gagnant %": (100 * vals["value_gagnant"]).round(0),
            "Values/course": vals["n_val"].astype(int),
            "Mise": vals["mise_u"].astype(int).astype(str) + "u",
        })
        st.dataframe(plan, hide_index=True, use_container_width=True, column_config={
            "P(Top3)": st.column_config.NumberColumn(format="%.1f"),
            "Cote": st.column_config.NumberColumn(format="%.1f"),
            "Value placé %": st.column_config.NumberColumn(format="%+.0f"),
            "Value gagnant %": st.column_config.NumberColumn(format="%+.0f")})
        st.caption(
            "**Mise** (1-3 u) = value crédible dosée par la confiance (P(Top3)) : "
            "3 u = value nette ET forte proba de placer. Aide au dosage, pas un "
            "système gagnant. **Values/course ≥ 2** = course à surveiller "
            "(plusieurs overlays, ou modèle très en désaccord avec le marché — "
            "à croiser à l'œil, parfois signe d'une course ouverte).")
g = dfj[(dfj.numero_reunion == sel.numero_reunion) & (dfj.numero_course == sel.numero_course)].copy()

if st.sidebar.button("🔄 Rafraîchir les cotes en direct"):
    live = fetch_cotes_live(date_sel, sel.numero_reunion, sel.numero_course)
    if live:
        g["cote_finale"] = g["numero"].map(live).fillna(g["cote_finale"])
        cf = g["cote_finale"]
        if cf.notna().sum() >= 4:
            g["p_marche"] = harville_top3(win_probs_from_odds(cf.fillna(cf.max()).values))
            g["value_place"] = g["p_top3"] / g["p_marche"] - 1.0
            okp = ((g["value_place"] > marge) & (g["p_marche"] >= 0.12)
                   & (g["cote_finale"] <= cote_max_value))
            if "p_gagne" in g.columns and g["p_gagne"].notna().any():
                g["value_gagnant"] = g["p_gagne"] * g["cote_finale"] - 1.0
                okg = ((g["value_gagnant"] > marge) & (g["p_gagne"] >= 0.08)
                       & (g["cote_finale"] <= cote_max_value * 2))
            else:
                g["value_gagnant"] = np.nan
                okg = pd.Series(False, index=g.index)
            vg = g["value_gagnant"].where(okg, other=-np.inf)
            vp = g["value_place"].where(okp, other=-np.inf)
            g["type_pari"] = np.where(okg & (vg >= vp), "Gagnant", "Placé")
            g["VALUE_ok"] = okp | okg
            best = np.maximum(vg, vp)
            g["value"] = np.where(np.isfinite(best), best, g["value_place"])
        st.sidebar.success(f"Cotes mises à jour ({len(live)}).")
    else:
        st.sidebar.warning("Cotes live indisponibles.")

g["rang"] = g["p_top3"].rank(ascending=False, method="min").astype("Int64")  # rang = ordre modèle
g = calc_signaux(g)
g = g.sort_values("cote_finale", na_position="last").reset_index(drop=True)  # affichage trié par cote

with onglet_course:
    titre = (f"R{sel.numero_reunion}C{sel.numero_course} — {sel.hippodrome} "
             f"— {sel.get('nom_prix', '') or ''}  ({g['heure'].iloc[0] if len(g) else ''})")
    barre_nav(titre, reunions, cur, "course")
    aff = pd.DataFrame({
        "Rg": g["rang"], "N°": g["numero"].astype("Int64"),
        "Cheval": g["cheval"], "Driver": g["driver"],
        "Cote": g["cote_finale"],
        "P(Top3)": (100 * g["p_top3"]).round(1),
        "Cote pivot placé": g["cote_pivot"],
        "Value placé %": (100 * g["value_place"]).round(0),
        "Value gagnant %": (100 * g["value_gagnant"]).round(0),
        "✔": np.where(g["VALUE_ok"], "✅", ""), "Signaux": g["sig"],
    })
    st.dataframe(aff, hide_index=True, use_container_width=True, column_config={
        "P(Top3)": st.column_config.NumberColumn(format="%.1f"),
        "Cote pivot placé": st.column_config.NumberColumn(format="%.2f"),
        "Cote": st.column_config.NumberColumn(format="%.1f"),
        "Value placé %": st.column_config.NumberColumn(format="%+.0f"),
        "Value gagnant %": st.column_config.NumberColumn(format="%+.0f"),
    })
    st.markdown(
        "**Légende des signaux** — le nombre entre parenthèses = **taux de placé "
        "historique** (% sur 100) derrière le signal, pour en jauger le poids :\n"
        "- **driver+ (48%)** : le driver place 48 % du temps · **tandem✓ (72%)** : le couple "
        "cheval×driver place 72 % · **ferrage✓ (65%)** : 65 % de placé avec cette ferrure · "
        "**hippo✓ (60%)** : 60 % de placé sur cet hippodrome\n"
        "- **🔥steam (+18%)** : la cote a raccourci de 18 % (argent tardif souvent informé) · "
        "**↘dérive (−x%)** : la cote monte · **forme+ (pos 2.4)** : position moyenne récente\n"
        "- **⏱record / 🔴RECORD** : meilleur (ou quasi) chrono du peloton · **1er déf.4 / 1er déf.** : premier déferré (peut transformer)\n"
        "- Un signal **🔴 rouge** = version forte (seuils élevés) · **✅** (colonne ✔) : au moins un des deux marchés offre une value crédible\n"
        "- ⚠️ ces % sont **descriptifs** (historique), pas une garantie : sur petit "
        "échantillon ils sont bruités, et le marché en price déjà l'essentiel\n"
        "- **Value placé %** vs **Value gagnant %** : la value sur chaque marché — à toi de choisir "
        "(le gagnant paie plus mais sort moins souvent ; au backtest il perd moins que le placé)\n"
        "- **Cote pivot placé** = 1/P(Top3) : la cote placé minimale pour que le pari placé soit intéressant.")

with onglet_carte:
    barre_nav(f"R{sel.numero_reunion}C{sel.numero_course} — {sel.hippodrome}",
              reunions, cur, "geny")
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
    st.dataframe(tab, hide_index=True, use_container_width=True, column_config={
        "P(Top3)": st.column_config.NumberColumn(format="%.1f"),
        "Cote": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Déf. = déferrage (D4/DA/DP/P4…). P(Top3) = probabilité de placer (modèle).")
