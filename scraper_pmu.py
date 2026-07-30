#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scraper PMU — courses de TROT, backfill par plage de dates (version généralisée).

Source : API publique non officielle PMU
    https://online.turfinfo.api.pmu.fr/rest/client/1/programme/<DDMMYYYY>
    .../programme/<DDMMYYYY>/R<r>/C<c>/participants

Ne garde que les réunions en France et les courses de trot (attelé / monté),
insère dans la base PostgreSQL "pmu_trot". Idempotent (UPSERT) : peut être
relancé sans créer de doublons. Par défaut, saute les jours déjà collectés
(reprise), sauf --force.

Dépendances : pip install requests psycopg2-binary

Exemples :
    python scraper_pmu.py 2025-01-01 2025-12-31        # une année
    python scraper_pmu.py 2024-01-01 2024-12-31 --pause 0.4
    python scraper_pmu.py 2025-05-01 2025-05-31 --force  # re-collecte forcée
"""

import os
import sys
import time
import getpass
import argparse
import datetime as dt

import requests

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit("Installe d'abord les dependances : pip install requests psycopg2-binary")

DB_DSN = dict(
    dbname=os.environ.get("PGDATABASE", "pmu_trot"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", ""),
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
)

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/1/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (recherche perso PMU trot)"}

# ---------------------------------------------------------------------------
# Mappings vers le schema (types ENUM)
# ---------------------------------------------------------------------------
def map_sexe(v):
    return {"MALES": "M", "FEMELLES": "F", "HONGRES": "H"}.get(v)

def map_origine(race):
    if race == "TROTTEUR FRANCAIS":
        return "FR"
    if race == "TROTTEUR ETRANGER":
        return "EUROPEENNE"
    return "INCONNU"

def map_ferrure(deferre):
    """Traduit la valeur API 'deferre' vers la nomenclature trot (D4/DA/DP/P4/PA/PP/…)."""
    if not deferre:
        return "FERRE"
    s = deferre.upper()
    if s == "DEFERRE_ANTERIEURS_POSTERIEURS":
        return "D4"
    if s == "PROTEGE_ANTERIEURS_POSTERIEURS":
        return "P4"
    ant_def  = "DEFERRE_ANTERIEURS" in s
    post_def = ("DEFERRE_POSTERIEURS" in s) or ("DEFERRRE_POSTERIEURS" in s)
    ant_pro  = "PROTEGE_ANTERIEURS" in s
    post_pro = "PROTEGE_POSTERIEURS" in s
    if ant_def and post_def:   return "D4"
    if ant_pro and post_pro:   return "P4"
    if ant_def and post_pro:   return "DA_PP"
    if ant_pro and post_def:   return "PA_DP"
    if ant_def:                return "DA"
    if post_def:               return "DP"
    if ant_pro:                return "PA"
    if post_pro:               return "PP"
    return "INCONNU"

def map_type_depart(categorie_particularite):
    s = categorie_particularite or ""
    return "AUTOSTART" if "AUTOSTART" in s else "VOLTE"

def cents_to_eur(v):
    return round(v / 100.0, 2) if v is not None else None

def annee_naissance(date_course, age):
    if age is None:
        return None
    try:
        return dt.date(date_course.year - int(age), 1, 1)
    except Exception:
        return None

def parse_rapports_place(data):
    """Extrait {numero_cheval: cote_place} depuis la reponse rapports-definitifs."""
    res = {}
    if not isinstance(data, list):
        return res
    for pari in data:
        if pari.get("typePari") != "SIMPLE_PLACE":
            continue
        for r in pari.get("rapports", []):
            comb, div = r.get("combinaison"), r.get("dividendePourUnEuro")
            if comb is None or div is None:
                continue
            try:
                res[int(comb)] = round(div / 100.0, 2)
            except (ValueError, TypeError):
                continue
    return res

def get_json(url, pause_retry=5, essais=3):
    for k in range(essais):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 204 or not r.content:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if k == essais - 1:
                raise
            time.sleep(pause_retry)

# ---------------------------------------------------------------------------
# Upsert des referentiels (cle = nom)
# ---------------------------------------------------------------------------
class Upserter:
    def __init__(self, cur):
        self.cur = cur
        self.cache = {"eleveurs": {}, "proprietaires": {}, "entraineurs": {},
                      "drivers": {}, "chevaux": {}, "hippodromes": {}}

    def reset(self):
        """Purge les caches (à appeler après un rollback : les id insérés puis
        annulés ne doivent plus être réutilisés)."""
        for d in self.cache.values():
            d.clear()

    def _simple(self, table, nom):
        if not nom:
            return None
        c = self.cache[table]
        if nom in c:
            return c[nom]
        self.cur.execute(f"SELECT id FROM {table} WHERE nom = %s", (nom,))
        row = self.cur.fetchone()
        if row:
            c[nom] = row[0]; return row[0]
        self.cur.execute(f"INSERT INTO {table} (nom) VALUES (%s) RETURNING id", (nom,))
        c[nom] = self.cur.fetchone()[0]
        return c[nom]

    def eleveur(self, n):      return self._simple("eleveurs", n)
    def proprietaire(self, n): return self._simple("proprietaires", n)
    def entraineur(self, n):   return self._simple("entraineurs", n)

    def driver(self, nom, specialite):
        if not nom:
            return None
        c = self.cache["drivers"]
        if nom in c:
            return c[nom]
        self.cur.execute("SELECT id FROM drivers WHERE nom = %s", (nom,))
        row = self.cur.fetchone()
        if row:
            c[nom] = row[0]; return row[0]
        self.cur.execute(
            "INSERT INTO drivers (nom, specialite) VALUES (%s, %s) RETURNING id",
            (nom, specialite))
        c[nom] = self.cur.fetchone()[0]
        return c[nom]

    def hippodrome(self, code, nom, sens_corde, type_piste):
        key = code or nom
        c = self.cache["hippodromes"]
        if key in c:
            return c[key]
        self.cur.execute("SELECT id FROM hippodromes WHERE code = %s OR nom = %s",
                         (code, nom))
        row = self.cur.fetchone()
        if row:
            c[key] = row[0]; return row[0]
        self.cur.execute(
            """INSERT INTO hippodromes (code, nom, sens_corde, type_piste)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            (code, nom, sens_corde, type_piste))
        c[key] = self.cur.fetchone()[0]
        return c[key]

    def cheval(self, nom, sexe=None, date_naissance=None, origine=None,
               pere=None, mere=None, eleveur_id=None, proprietaire_id=None):
        if not nom:
            return None
        c = self.cache["chevaux"]
        key = (nom, date_naissance)
        if key in c:
            return c[key]
        # 1) correspondance EXACTE (nom, date) si la date est connue
        if date_naissance is not None:
            self.cur.execute(
                "SELECT id FROM chevaux WHERE nom = %s AND date_naissance = %s",
                (nom, date_naissance))
            row = self.cur.fetchone()
            if row:
                self.cur.execute(
                    """UPDATE chevaux SET
                           sexe = COALESCE(sexe, %s),
                           origine = CASE WHEN origine='INCONNU' THEN %s ELSE origine END,
                           eleveur_id = COALESCE(eleveur_id, %s),
                           proprietaire_id = COALESCE(proprietaire_id, %s)
                       WHERE id = %s""",
                    (sexe, origine or "INCONNU", eleveur_id, proprietaire_id, row[0]))
                c[key] = row[0]
                return row[0]
        # 2) sinon, une ligne du meme nom ; on prend d'abord une ligne SANS date a enrichir
        self.cur.execute(
            "SELECT id, date_naissance FROM chevaux WHERE nom = %s "
            "ORDER BY (date_naissance IS NOT NULL), id LIMIT 1", (nom,))
        row = self.cur.fetchone()
        if row:
            rid, rdate = row
            if date_naissance is not None and rdate is None:
                # dater la ligne sans date : sans risque (aucune (nom,date), cf. etape 1)
                self.cur.execute(
                    """UPDATE chevaux SET date_naissance=%s, sexe=COALESCE(sexe,%s),
                           origine=CASE WHEN origine='INCONNU' THEN %s ELSE origine END,
                           eleveur_id=COALESCE(eleveur_id,%s),
                           proprietaire_id=COALESCE(proprietaire_id,%s)
                       WHERE id=%s""",
                    (date_naissance, sexe, origine or "INCONNU",
                     eleveur_id, proprietaire_id, rid))
            else:
                self.cur.execute(
                    """UPDATE chevaux SET sexe=COALESCE(sexe,%s),
                           origine=CASE WHEN origine='INCONNU' THEN %s ELSE origine END,
                           eleveur_id=COALESCE(eleveur_id,%s),
                           proprietaire_id=COALESCE(proprietaire_id,%s)
                       WHERE id=%s""",
                    (sexe, origine or "INCONNU", eleveur_id, proprietaire_id, rid))
            c[key] = rid
            return rid
        # 3) insertion
        pere_id = self.cheval(pere) if pere else None
        mere_id = self.cheval(mere) if mere else None
        self.cur.execute(
            """INSERT INTO chevaux
                   (nom, sexe, date_naissance, origine, pere_id, mere_id,
                    eleveur_id, proprietaire_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (nom, sexe, date_naissance, origine or "INCONNU",
             pere_id, mere_id, eleveur_id, proprietaire_id))
        c[key] = self.cur.fetchone()[0]
        return c[key]

# ---------------------------------------------------------------------------
# Traitement d'une course
# ---------------------------------------------------------------------------
def traiter_course(cur, up, date_course, reunion, course, pause):
    num_reunion = reunion["numOfficiel"]
    num_course = course["numOrdre"]
    hip = reunion["hippodrome"]

    sens_corde = {"CORDE_GAUCHE": "GAUCHE", "CORDE_DROITE": "DROITE"}.get(course.get("corde"))
    hippo_id = up.hippodrome(hip.get("code"), hip.get("libelleCourt"),
                             sens_corde, course.get("parcours"))
    type_depart = map_type_depart(course.get("categorieParticularite"))
    discipline = course.get("discipline")

    cur.execute(
        """INSERT INTO courses
               (date_course, heure_depart, hippodrome_id, numero_reunion,
                numero_course, nom_prix, discipline, type_depart, distance_m,
                allocation_eur, categorie, cond_gains, nb_partants)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (date_course, hippodrome_id, numero_course)
           DO UPDATE SET nb_partants = EXCLUDED.nb_partants
           RETURNING id""",
        (date_course, None, hippo_id, num_reunion, num_course,
         course.get("libelle"), discipline, type_depart, course.get("distance"),
         cents_to_eur(course.get("montantPrix")),
         course.get("categorieParticularite"), course.get("conditions"),
         course.get("nombreDeclaresPartants")))
    course_id = cur.fetchone()[0]

    url = (f"{BASE}/{date_course.strftime('%d%m%Y')}"
           f"/R{num_reunion}/C{num_course}/participants")
    data = get_json(url)
    time.sleep(pause)
    if not data:
        return 0
    n = 0
    for p in data.get("participants", []):
        if p.get("statut") != "PARTANT":
            continue
        elv_id = up.eleveur(p.get("eleveur") or None)
        prop_id = up.proprietaire(p.get("proprietaire") or None)
        cheval_id = up.cheval(
            p["nom"], map_sexe(p.get("sexe")),
            annee_naissance(date_course, p.get("age")),
            map_origine(p.get("race")),
            p.get("nomPere"), p.get("nomMere"), elv_id, prop_id)
        driver_id = up.driver(p.get("driver"),
                              "MONTE" if discipline == "MONTE" else "ATTELE")
        entr_id = up.entraineur(p.get("entraineur") or None)

        cote_ref = (p.get("dernierRapportReference") or {}).get("rapport")
        cote_dir = (p.get("dernierRapportDirect") or {}).get("rapport")
        handicap = p.get("handicapDistance")
        recul = (handicap - course["distance"]) if (handicap and course.get("distance")) else 0
        gp = p.get("gainsParticipant") or {}

        cur.execute(
            """INSERT INTO partants
                   (course_id, cheval_id, numero, corde, distance_partant_m,
                    recul_m, ferrure, driver_id, entraineur_id, gains_carriere_eur,
                    nb_courses_avant, nb_victoires_avant, nb_places_avant,
                    gains_annee_eur, gains_annee_prec_eur, musique,
                    cote_depart, cote_finale)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (course_id, numero) DO UPDATE SET
                    corde=EXCLUDED.corde, distance_partant_m=EXCLUDED.distance_partant_m,
                    recul_m=EXCLUDED.recul_m, ferrure=EXCLUDED.ferrure,
                    driver_id=EXCLUDED.driver_id, entraineur_id=EXCLUDED.entraineur_id,
                    gains_carriere_eur=EXCLUDED.gains_carriere_eur,
                    nb_courses_avant=EXCLUDED.nb_courses_avant,
                    nb_victoires_avant=EXCLUDED.nb_victoires_avant,
                    nb_places_avant=EXCLUDED.nb_places_avant,
                    gains_annee_eur=EXCLUDED.gains_annee_eur,
                    gains_annee_prec_eur=EXCLUDED.gains_annee_prec_eur,
                    musique=EXCLUDED.musique,
                    cote_depart=EXCLUDED.cote_depart, cote_finale=EXCLUDED.cote_finale
               RETURNING id""",
            (course_id, cheval_id, p.get("numPmu"), p.get("placeCorde"),
             handicap, recul, map_ferrure(p.get("deferre")), driver_id, entr_id,
             cents_to_eur(gp.get("gainsCarriere")),
             p.get("nombreCourses"), p.get("nombreVictoires"), p.get("nombrePlaces"),
             cents_to_eur(gp.get("gainsAnneeEnCours")),
             cents_to_eur(gp.get("gainsAnneePrecedente")), p.get("musique"),
             cote_ref, cote_dir))
        partant_id = cur.fetchone()[0]

        incident = p.get("incident")
        disq = bool(incident and "DISQUALIFIE" in incident)
        ordre = p.get("ordreArrivee")
        place = ordre if (ordre and not disq) else None
        cur.execute(
            """INSERT INTO resultats
                   (partant_id, place, disqualifie, motif_disqual, reduction_km_cs)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (partant_id) DO NOTHING""",
            (partant_id, place, disq, incident if disq else None,
             p.get("reductionKilometrique")))
        n += 1

    cur.execute("""INSERT INTO sources_collecte (course_id, source, url)
                   VALUES (%s, 'PMU', %s)""", (course_id, url))

    # Rapports PLACÉ (marché cible) collectés dans la même passe.
    # Isolés par SAVEPOINT : un échec ici (réseau, placé non offert sur petit
    # champ) ne fait pas perdre les données de la course.
    url_rp = (f"{BASE}/{date_course.strftime('%d%m%Y')}"
              f"/R{num_reunion}/C{num_course}/rapports-definitifs")
    cur.execute("SAVEPOINT sp_place")
    try:
        for numero, cote in parse_rapports_place(get_json(url_rp)).items():
            cur.execute("""UPDATE resultats SET rapport_place = %s
                           WHERE partant_id = (SELECT id FROM partants
                                               WHERE course_id = %s AND numero = %s)""",
                        (cote, course_id, numero))
        cur.execute("""INSERT INTO sources_collecte (course_id, source, url)
                       VALUES (%s, 'RAPPORTS_PLACE', %s)""", (course_id, url_rp))
        cur.execute("RELEASE SAVEPOINT sp_place")
        time.sleep(pause)
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT sp_place")
    return n

# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
def jour_deja_collecte(cur, jour):
    cur.execute("SELECT 1 FROM courses WHERE date_course = %s LIMIT 1", (jour,))
    return cur.fetchone() is not None

def connexion():
    if not DB_DSN.get("password"):
        DB_DSN["password"] = getpass.getpass(
            f"Mot de passe PostgreSQL (utilisateur {DB_DSN['user']}) : ")
    try:
        return psycopg2.connect(**DB_DSN)
    except psycopg2.OperationalError as e:
        msg = str(e).strip() or "(aucun detail fourni par le serveur)"
        print("\n--- Echec de connexion a PostgreSQL ---\nDetail :", msg)
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Backfill PMU trot (plage de dates).")
    ap.add_argument("debut", help="date de début (YYYY-MM-DD)")
    ap.add_argument("fin", help="date de fin incluse (YYYY-MM-DD)")
    ap.add_argument("--pause", type=float, default=0.7, help="pause (s) entre requêtes")
    ap.add_argument("--force", action="store_true",
                    help="re-collecte les jours déjà présents en base")
    args = ap.parse_args()

    debut = dt.date.fromisoformat(args.debut)
    fin = dt.date.fromisoformat(args.fin)
    if fin < debut:
        sys.exit("La date de fin est antérieure à la date de début.")

    conn = connexion()
    conn.autocommit = False
    cur = conn.cursor()
    up = Upserter(cur)

    jour = debut
    total_courses = total_partants = jours_sautes = 0
    print(f"Backfill du {debut} au {fin} (pause {args.pause}s, "
          f"reprise {'OFF' if args.force else 'ON'})")
    while jour <= fin:
        if not args.force and jour_deja_collecte(cur, jour):
            jours_sautes += 1
            jour += dt.timedelta(days=1)
            continue
        try:
            prog = get_json(f"{BASE}/{jour.strftime('%d%m%Y')}")
        except Exception as e:
            print(f"[{jour}] erreur programme : {e}")
            jour += dt.timedelta(days=1)
            continue
        reunions = (prog or {}).get("programme", {}).get("reunions", []) if prog else []
        nb_c = 0
        for reu in reunions:
            if (reu.get("pays") or {}).get("code") != "FRA":
                continue
            for cse in reu.get("courses", []):
                if cse.get("discipline") not in ("ATTELE", "MONTE"):
                    continue
                try:
                    nb = traiter_course(cur, up, jour, reu, cse, args.pause)
                    total_partants += nb; nb_c += 1; total_courses += 1
                except Exception as e:
                    conn.rollback()
                    up.reset()      # purge le cache : les id annulés ne doivent pas resservir
                    print(f"[{jour}] R{reu['numOfficiel']}C{cse['numOrdre']} erreur : {e}")
                else:
                    conn.commit()
        if nb_c:
            print(f"[{jour}] {nb_c} courses de trot")
        time.sleep(args.pause)
        jour += dt.timedelta(days=1)

    cur.close(); conn.close()
    print(f"\nTermine : {total_courses} courses, {total_partants} partants inseres."
          f" ({jours_sautes} jours deja collectes, sautes)")

if __name__ == "__main__":
    main()
