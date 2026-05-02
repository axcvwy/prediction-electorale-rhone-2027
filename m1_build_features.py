import pandas as pd
import numpy as np

INPUT_FILE = "Data_Model/dataset_modele.csv"
OUTPUT_FILE = "Data_Model/dataset_features.csv"

df = pd.read_csv(INPUT_FILE, sep=";")

df["code_commune"] = (
    df["code_commune"]
    .astype(str)
    .str.strip()
    .str.replace(".0", "", regex=False)
    .str.zfill(5)
)

df["annee_election"] = pd.to_numeric(df["annee_election"], errors="coerce").astype("Int64")
df = df.sort_values(["code_commune", "annee_election"]).copy()

bloc_cols = [c for c in df.columns if c.startswith("bloc_")]
socio_cols = [
    "taux_chomage",
    "population",
    "nb_menages_fiscaux",
    "nb_personnes_menages_fiscaux",
    "revenu_median",
    "nb_etablissements",
    "effectifs_salaries"
]

id_cols = ["code_commune", "nom_commune"]
years = [2012, 2017, 2022]

frames = []
for year in years:
    temp = df[df["annee_election"] == year].copy()
    keep_cols = id_cols + socio_cols + bloc_cols
    temp = temp[keep_cols]

    rename_map = {
        col: f"{col}_{year}"
        for col in keep_cols
        if col not in id_cols
    }
    temp = temp.rename(columns=rename_map)
    frames.append(temp)

wide = frames[0]
for temp in frames[1:]:
    wide = wide.merge(temp, on=["code_commune", "nom_commune"], how="outer")

# Bloc deltas
for col in bloc_cols:
    col_2012 = f"{col}_2012"
    col_2017 = f"{col}_2017"
    col_2022 = f"{col}_2022"

    if col_2012 in wide.columns and col_2017 in wide.columns:
        wide[f"{col}_delta_12_17"] = wide[col_2017] - wide[col_2012]

    if col_2017 in wide.columns and col_2022 in wide.columns:
        wide[f"{col}_delta_17_22"] = wide[col_2022] - wide[col_2017]

# Socio-economic deltas
for col in socio_cols:
    col_2012 = f"{col}_2012"
    col_2017 = f"{col}_2017"
    col_2022 = f"{col}_2022"

    if col_2012 in wide.columns and col_2017 in wide.columns:
        wide[f"{col}_delta_12_17"] = wide[col_2017] - wide[col_2012]

    if col_2017 in wide.columns and col_2022 in wide.columns:
        wide[f"{col}_delta_17_22"] = wide[col_2022] - wide[col_2017]

# Optional ratios
for year in years:
    col_sal = f"effectifs_salaries_{year}"
    col_etab = f"nb_etablissements_{year}"
    col_pers = f"nb_personnes_menages_fiscaux_{year}"
    col_men = f"nb_menages_fiscaux_{year}"

    if col_sal in wide.columns and col_etab in wide.columns:
        wide[f"salaries_par_etablissement_{year}"] = np.where(
            wide[col_etab].fillna(0) != 0,
            wide[col_sal] / wide[col_etab],
            np.nan
        )

    if col_pers in wide.columns and col_men in wide.columns:
        wide[f"personnes_par_menage_fiscal_{year}"] = np.where(
            wide[col_men].fillna(0) != 0,
            wide[col_pers] / wide[col_men],
            np.nan
        )

# Winner labels for analysis
for year in years:
    year_bloc_cols = [f"{c}_{year}" for c in bloc_cols if f"{c}_{year}" in wide.columns]

    if year_bloc_cols:
        mask_has_data = wide[year_bloc_cols].notna().any(axis=1)

        winners = pd.Series(index=wide.index, dtype="object")

        winners.loc[mask_has_data] = (
            wide.loc[mask_has_data, year_bloc_cols]
            .idxmax(axis=1)
            .str.replace(f"_{year}", "", regex=False)
        )

        wide[f"bloc_gagnant_{year}"] = winners

        print(f"rows with no bloc data in {year}: {(~mask_has_data).sum()}")

wide = wide.sort_values(["code_commune"]).reset_index(drop=True)

wide.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8-sig")

print("rows:", len(wide))
print("cols:", len(wide.columns))
print("output:", OUTPUT_FILE)