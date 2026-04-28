import pandas as pd

eta = pd.read_csv("Data_Source/etablissements-et-effectifs-salaries.csv", sep=";", dtype={
    "Code département": str,
    "Code commune": str
})

eta = eta[eta["Code département"].astype(str).str.zfill(2) == "69"]

cols = [
    "Nom commune",
    "Code commune",
    "Nombre d'établissements 2012",
    "Effectifs salariés 2012",
    "Nombre d'établissements 2017",
    "Effectifs salariés 2017",
    "Nombre d'établissements 2022",
    "Effectifs salariés 2022"
]

eta = eta[cols].rename(columns={
    "Nom commune": "nom_commune",
    "Code commune": "code_commune",
    "Nombre d'établissements 2012": "nb_etablissements_2012",
    "Effectifs salariés 2012": "effectifs_salaries_2012",
    "Nombre d'établissements 2017": "nb_etablissements_2017",
    "Effectifs salariés 2017": "effectifs_salaries_2017",
    "Nombre d'établissements 2022": "nb_etablissements_2022",
    "Effectifs salariés 2022": "effectifs_salaries_2022"
})

eta["code_commune"] = eta["code_commune"].astype(str).str.strip()
eta["nom_commune"] = eta["nom_commune"].astype(str).str.strip()

num_cols = [
    "nb_etablissements_2012", "effectifs_salaries_2012",
    "nb_etablissements_2017", "effectifs_salaries_2017",
    "nb_etablissements_2022", "effectifs_salaries_2022"
]

for col in num_cols:
    eta[col] = pd.to_numeric(eta[col], errors="coerce")

eta_agg = eta.groupby(["code_commune", "nom_commune"], as_index=False)[num_cols].sum(min_count=1)

eta_2012 = eta_agg[["code_commune", "nom_commune", "nb_etablissements_2012", "effectifs_salaries_2012"]].rename(columns={
    "nb_etablissements_2012": "nb_etablissements",
    "effectifs_salaries_2012": "effectifs_salaries"
})
eta_2012["annee"] = 2012

eta_2017 = eta_agg[["code_commune", "nom_commune", "nb_etablissements_2017", "effectifs_salaries_2017"]].rename(columns={
    "nb_etablissements_2017": "nb_etablissements",
    "effectifs_salaries_2017": "effectifs_salaries"
})
eta_2017["annee"] = 2017

eta_2022 = eta_agg[["code_commune", "nom_commune", "nb_etablissements_2022", "effectifs_salaries_2022"]].rename(columns={
    "nb_etablissements_2022": "nb_etablissements",
    "effectifs_salaries_2022": "effectifs_salaries"
})
eta_2022["annee"] = 2022

eta_long = pd.concat([eta_2012, eta_2017, eta_2022], ignore_index=True)

eta_long = eta_long[
    ["annee", "code_commune", "nom_commune", "nb_etablissements", "effectifs_salaries"]
]

eta_long = eta_long.dropna(
    subset=["nb_etablissements", "effectifs_salaries"],
    how="all"
)

eta_long = eta_long.drop_duplicates(
    subset=["annee", "code_commune"]
).sort_values(["annee", "code_commune"])

eta_long.to_csv("Data_Cleaned/etablissements_salaries_communes_dep69.csv", sep=";", index=False, encoding="utf-8-sig")