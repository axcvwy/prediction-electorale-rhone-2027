import pandas as pd

def clean_code_commune(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(5)
    )

def clean_nom_commune(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

votes = pd.read_csv("Data_Cleaned/votes_perc_blocs_communes_dep69.csv", sep=";")
chomage = pd.read_csv("Data_Cleaned/chomage_communes_dep69.csv", sep=";")
population = pd.read_csv("Data_Cleaned/population_communes_dep69.csv", sep=";")
revenus = pd.read_csv("Data_Cleaned/revenus_communes_dep69.csv", sep=";")
etab = pd.read_csv("Data_Cleaned/etablissements_salaries_communes_dep69.csv", sep=";")


votes["code_commune"] = clean_code_commune(votes["code_commune"])
votes["annee_election"] = pd.to_numeric(votes["annee_election"], errors="coerce").astype("Int64")

for df in [chomage, population, revenus, etab]:
    df["code_commune"] = clean_code_commune(df["code_commune"])
    df["nom_commune"] = clean_nom_commune(df["nom_commune"])
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce").astype("Int64")


for col in ["taux_chomage"]:
    if col in chomage.columns:
        chomage[col] = pd.to_numeric(chomage[col], errors="coerce")

for col in ["population"]:
    if col in population.columns:
        population[col] = pd.to_numeric(population[col], errors="coerce")

for col in ["nb_menages_fiscaux", "nb_personnes_menages_fiscaux", "revenu_median"]:
    if col in revenus.columns:
        revenus[col] = pd.to_numeric(revenus[col], errors="coerce")

for col in ["nb_etablissements", "effectifs_salaries"]:
    if col in etab.columns:
        etab[col] = pd.to_numeric(etab[col], errors="coerce")

bloc_cols = [c for c in votes.columns if c.startswith("bloc_")]
for col in bloc_cols:
    votes[col] = pd.to_numeric(votes[col], errors="coerce")

# YEAR MAPPING
map_chomage = {2011: 2012, 2016: 2017, 2022: 2022}
map_population = {2013: 2012, 2017: 2017, 2022: 2022}
map_revenus = {2012: 2012, 2017: 2017, 2021: 2022}
map_etab = {2012: 2012, 2017: 2017, 2022: 2022}

chomage_m = chomage[chomage["annee"].isin(map_chomage.keys())].copy()
chomage_m["annee_election"] = chomage_m["annee"].map(map_chomage)
chomage_m = chomage_m[["annee_election", "code_commune", "nom_commune", "taux_chomage"]]

population_m = population[population["annee"].isin(map_population.keys())].copy()
population_m["annee_election"] = population_m["annee"].map(map_population)
population_m = population_m[["annee_election", "code_commune", "nom_commune", "population"]]

revenus_m = revenus[revenus["annee"].isin(map_revenus.keys())].copy()
revenus_m["annee_election"] = revenus_m["annee"].map(map_revenus)
revenus_m = revenus_m[[
    "annee_election",
    "code_commune",
    "nom_commune",
    "nb_menages_fiscaux",
    "nb_personnes_menages_fiscaux",
    "revenu_median"
]]

etab_m = etab[etab["annee"].isin(map_etab.keys())].copy()
etab_m["annee_election"] = etab_m["annee"].map(map_etab)
etab_m = etab_m[[
    "annee_election",
    "code_commune",
    "nom_commune",
    "nb_etablissements",
    "effectifs_salaries"
]]

# COMMUNE REFERENCE
communes_ref = pd.concat([
    chomage_m[["code_commune", "nom_commune"]],
    population_m[["code_commune", "nom_commune"]],
    revenus_m[["code_commune", "nom_commune"]],
    etab_m[["code_commune", "nom_commune"]],
], ignore_index=True)

communes_ref = (
    communes_ref.dropna()
    .drop_duplicates(subset=["code_commune", "nom_commune"])
    .groupby("code_commune", as_index=False)
    .first()
)

# MERGE
dataset = votes.merge(communes_ref, on="code_commune", how="left")
dataset = dataset.merge(
    chomage_m[["annee_election", "code_commune", "taux_chomage"]],
    on=["annee_election", "code_commune"],
    how="left"
)
dataset = dataset.merge(
    population_m[["annee_election", "code_commune", "population"]],
    on=["annee_election", "code_commune"],
    how="left"
)
dataset = dataset.merge(
    revenus_m[[
        "annee_election",
        "code_commune",
        "nb_menages_fiscaux",
        "nb_personnes_menages_fiscaux",
        "revenu_median"
    ]],
    on=["annee_election", "code_commune"],
    how="left"
)
dataset = dataset.merge(
    etab_m[[
        "annee_election",
        "code_commune",
        "nb_etablissements",
        "effectifs_salaries"
    ]],
    on=["annee_election", "code_commune"],
    how="left"
)


bloc_cols = sorted([c for c in dataset.columns if c.startswith("bloc_")])

final_cols = [
    "annee_election",
    "code_commune",
    "nom_commune",
    "taux_chomage",
    "population",
    "nb_menages_fiscaux",
    "nb_personnes_menages_fiscaux",
    "revenu_median",
    "nb_etablissements",
    "effectifs_salaries",
] + bloc_cols

dataset = dataset[final_cols].sort_values(["annee_election", "code_commune"])

# QUALITY CHECK
duplicates = dataset.duplicated(subset=["annee_election", "code_commune"]).sum()
print("duplicates:", duplicates)

null_rates = (dataset.isna().mean() * 100).round(2)
print(null_rates)

socio_cols = [
    "taux_chomage",
    "population",
    "nb_menages_fiscaux",
    "nb_personnes_menages_fiscaux",
    "revenu_median",
    "nb_etablissements",
    "effectifs_salaries"
]

dataset = dataset.dropna(subset=socio_cols, how="all")

dataset.to_csv("Data_Model/dataset_modele.csv", sep=";", index=False, encoding="utf-8-sig")