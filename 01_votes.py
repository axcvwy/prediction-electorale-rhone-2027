
import pandas as pd
import unicodedata
from pathlib import Path


INPUT_FILE = Path("Data_Source/candidats_results.csv")
OUTPUT_FILE = Path("Data_Cleaned/votes_perc_blocs_communes_dep69.csv")


TARGET_IDS = {
    "2012_pres_t1",
    "2017_pres_t1",
    "2022_pres_t1",
}


COLS_TO_KEEP = [
    "id_election",
    "code_departement",
    "code_commune",
    "voix",
    "nom",
]


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().upper()
    x = unicodedata.normalize("NFKD", x).encode("ASCII", "ignore").decode("utf-8")
    return " ".join(x.split())


BLOC_MAPPING = {
    "ARTHAUD": "gauche_radicale",
    "POUTOU": "gauche_radicale",

    "HOLLANDE": "gauche",
    "HAMON": "gauche",
    "HIDALGO": "gauche",
    "JOLY": "gauche",
    "JADOT": "gauche",
    "ROUSSEL": "gauche",
    "MELENCHON": "gauche",

    "MACRON": "centre",
    "BAYROU": "centre",
    "LASSALLE": "centre",

    "SARKOZY": "droite",
    "FILLON": "droite",
    "PECRESSE": "droite",

    "LE PEN": "droite_radicale",
    "ZEMMOUR": "droite_radicale",

    "DUPONT-AIGNAN": "souverainiste_autres",
    "ASSELINEAU": "souverainiste_autres",
    "CHEMINADE": "souverainiste_autres",
}


def main():
    df = pd.read_csv(INPUT_FILE, sep=";", dtype=str)

    df = df[
        (df["id_election"].isin(TARGET_IDS)) &
        (df["code_departement"].astype(str).str.strip() == "69")
    ].copy()

    df = df[COLS_TO_KEEP].copy()

    df["code_commune"] = (
        df["code_commune"]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(5)
    )

    df["nom"] = df["nom"].map(normalize_text)
    df["bloc"] = df["nom"].map(BLOC_MAPPING)

    df["voix"] = pd.to_numeric(df["voix"], errors="coerce").fillna(0)

    df["annee_election"] = (
        df["id_election"]
        .str.extract(r"(\d{4})")[0]
        .astype(int)
    )

    df = df[df["bloc"].notna()].copy()

    votes_bloc = (
        df.groupby(["annee_election", "code_commune", "bloc"], as_index=False)["voix"]
        .sum()
    )

    totaux = (
        votes_bloc.groupby(["annee_election", "code_commune"], as_index=False)["voix"]
        .sum()
        .rename(columns={"voix": "voix_total_exprimes"})
    )

    votes_bloc = votes_bloc.merge(
        totaux,
        on=["annee_election", "code_commune"],
        how="left"
    )

    votes_bloc["pct_bloc"] = (
        votes_bloc["voix"] / votes_bloc["voix_total_exprimes"] * 100
    )

    votes_wide = (
        votes_bloc.pivot_table(
            index=["annee_election", "code_commune"],
            columns="bloc",
            values="pct_bloc",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    votes_wide.columns.name = None

    votes_wide = votes_wide.rename(columns={
        col: f"bloc_{col}"
        for col in votes_wide.columns
        if col not in ["annee_election", "code_commune"]
    })

    bloc_cols = sorted([c for c in votes_wide.columns if c.startswith("bloc_")])

    votes_final = votes_wide[
        ["annee_election", "code_commune"] + bloc_cols
    ].sort_values(["annee_election", "code_commune"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    votes_final.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(votes_final))
    print("Bloc columns:", bloc_cols)


if __name__ == "__main__":
    main()