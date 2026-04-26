import pandas as pd

pop = pd.read_csv("Data_Source/population_municipale_communes_france.csv", sep=";", dtype={"codgeo": str, "dep": str})

pop = pop[pop["dep"].astype(str).str.zfill(2) == "69"]

pop = pop[["codgeo", "libgeo", "p13_pop", "p17_pop", "p22_pop"]].rename(columns={
    "codgeo": "code_commune",
    "libgeo": "nom_commune"
})

pop_long = pop.melt(
    id_vars=["code_commune", "nom_commune"],
    value_vars=["p13_pop", "p17_pop", "p22_pop"],
    var_name="annee_source",
    value_name="population"
)

annee_map = {
    "p13_pop": 2013,
    "p17_pop": 2017,
    "p22_pop": 2022
}

pop_long["annee"] = pop_long["annee_source"].map(annee_map)

pop_long["code_commune"] = pop_long["code_commune"].astype(str).str.strip()
pop_long["nom_commune"] = pop_long["nom_commune"].astype(str).str.strip()
pop_long["population"] = pd.to_numeric(pop_long["population"], errors="coerce")

pop_long = pop_long.dropna(subset=["annee", "code_commune", "population"])

pop_long = pop_long[["annee", "code_commune", "nom_commune", "population"]]

pop_long = pop_long.drop_duplicates(
    subset=["annee", "code_commune"]
).sort_values(["annee", "code_commune"])

pop_long.to_csv("Data_Cleaned/population_communes_dep69.csv", sep=";", index=False, encoding="utf-8-sig")