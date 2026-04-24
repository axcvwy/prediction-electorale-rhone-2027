import pandas as pd

chomage = pd.read_csv("Data_Source/taux_chomage_communes.csv", sep=",", dtype={"code_com": str})

chomage = chomage.rename(columns={
    "code_com": "code_commune",
    "nom_territoire": "nom_commune",
    "valeur": "taux_chomage"
})

chomage = chomage[
    ["annee", "code_commune", "nom_commune", "taux_chomage"]
]

chomage["code_commune"] = chomage["code_commune"].astype(str).str.strip()
chomage["nom_commune"] = chomage["nom_commune"].astype(str).str.strip()
chomage["annee"] = pd.to_numeric(chomage["annee"], errors="coerce")
chomage["taux_chomage"] = pd.to_numeric(chomage["taux_chomage"], errors="coerce")

chomage = chomage[
    chomage["code_commune"].str.startswith("69", na=False)
]

chomage = chomage.dropna(subset=["annee", "code_commune", "taux_chomage"])

chomage = chomage.drop_duplicates(
    subset=["annee", "code_commune"]
).sort_values(["annee", "code_commune"])

chomage.to_csv("Data_Cleaned/chomage_communes_dep69.csv", sep=";", index=False, encoding="utf-8-sig")