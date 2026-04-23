import pandas as pd

f2012 = pd.read_csv("Data_Source/base-cc-filosofi-2012.csv", sep=";", dtype={"CODGEO": str})
f2017 = pd.read_csv("Data_Source/base-cc-filosofi-2017.csv", sep=";", dtype={"CODGEO": str})
f2021 = pd.read_csv("Data_Source/FILO2021_DEC_COM.csv", sep=";", dtype={"CODGEO": str})

r2012 = (
    f2012[["CODGEO", "LIBGEO", "NBMENFISC12", "NBPERSMENFISC12", "MED12"]]
    .rename(columns={
        "CODGEO": "code_commune",
        "LIBGEO": "nom_commune",
        "NBMENFISC12": "nb_menages_fiscaux",
        "NBPERSMENFISC12": "nb_personnes_menages_fiscaux",
        "MED12": "revenu_median"
    })
)
r2012["annee"] = 2012

r2017 = (
    f2017[["CODGEO", "LIBGEO", "NBMENFISC17", "NBPERSMENFISC17", "MED17"]]
    .rename(columns={
        "CODGEO": "code_commune",
        "LIBGEO": "nom_commune",
        "NBMENFISC17": "nb_menages_fiscaux",
        "NBPERSMENFISC17": "nb_personnes_menages_fiscaux",
        "MED17": "revenu_median"
    })
)
r2017["annee"] = 2017

r2021 = (
    f2021[["CODGEO", "LIBGEO", "NBMEN21", "NBPERS21", "Q221"]]
    .rename(columns={
        "CODGEO": "code_commune",
        "LIBGEO": "nom_commune",
        "NBMEN21": "nb_menages_fiscaux",
        "NBPERS21": "nb_personnes_menages_fiscaux",
        "Q221": "revenu_median"
    })
)
r2021["annee"] = 2021

revenus_communes = pd.concat([r2012, r2017, r2021], ignore_index=True)

revenus_communes = revenus_communes[
    [
        "annee",
        "code_commune",
        "nom_commune",
        "nb_menages_fiscaux",
        "nb_personnes_menages_fiscaux",
        "revenu_median"
    ]
]

revenus_communes["code_commune"] = revenus_communes["code_commune"].astype(str).str.strip()
revenus_communes["nom_commune"] = revenus_communes["nom_commune"].astype(str).str.strip()

revenus_communes = revenus_communes[
    revenus_communes["code_commune"].str.startswith("69", na=False)
]

for col in ["nb_menages_fiscaux", "nb_personnes_menages_fiscaux", "revenu_median"]:
    revenus_communes[col] = pd.to_numeric(revenus_communes[col], errors="coerce")

revenus_communes = revenus_communes.dropna(
    subset=["nb_menages_fiscaux", "nb_personnes_menages_fiscaux", "revenu_median"],
    how="all"
)

revenus_communes = revenus_communes.drop_duplicates(
    subset=["annee", "code_commune"]
).sort_values(["annee", "code_commune"])

revenus_communes.to_csv(
    "Data_Cleaned/revenus_communes_dep69.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)