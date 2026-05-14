import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


INPUT_FILE = Path("Data_Model/predictions_2027_results.csv")
OUTPUT_DIR = Path("Data_Viz")

OUTPUT_RANK_COMMUNES = OUTPUT_DIR / "classement_blocs_communes_gagnees_2027.png"
OUTPUT_RANK_SCORES = OUTPUT_DIR / "classement_blocs_scores_moyens_2027.png"
OUTPUT_COMPARE_2022_2027 = OUTPUT_DIR / "comparaison_blocs_communes_2022_2027.png"
OUTPUT_REPARTITION_2027 = OUTPUT_DIR / "repartition_gagnants_2027.png"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_FILE, sep=";")


# Nettoyage des colonnes dupliquées éventuelles
df = df.loc[:, ~df.columns.duplicated()].copy()


# Détection des colonnes de scores prédites pour 2027
pred_cols = [
    c for c in df.columns
    if c.startswith("bloc_")
    and c.endswith("_pred_2027")
    and c != "bloc_gagnant_pred_2027"
]

bloc_names = [c.replace("_pred_2027", "") for c in pred_cols]


# Vérification des colonnes minimales nécessaires
required_cols = ["code_commune", "nom_commune", "bloc_gagnant_2022", "bloc_gagnant_pred_2027"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise ValueError(f"Colonnes manquantes : {missing_required}")


# Calcul du nombre de communes gagnées par bloc en 2022 et en 2027
wins_2022 = (
    df["bloc_gagnant_2022"]
    .value_counts(dropna=False)
    .rename_axis("bloc")
    .reset_index(name="nb_communes_gagnees_2022")
)

wins_2027 = (
    df["bloc_gagnant_pred_2027"]
    .value_counts(dropna=False)
    .rename_axis("bloc")
    .reset_index(name="nb_communes_gagnees_2027")
)

summary_bloc = wins_2022.merge(wins_2027, on="bloc", how="outer").fillna(0)
summary_bloc["nb_communes_gagnees_2022"] = summary_bloc["nb_communes_gagnees_2022"].astype(int)
summary_bloc["nb_communes_gagnees_2027"] = summary_bloc["nb_communes_gagnees_2027"].astype(int)
summary_bloc["evolution_nb_communes"] = (
    summary_bloc["nb_communes_gagnees_2027"] - summary_bloc["nb_communes_gagnees_2022"]
)

summary_bloc = summary_bloc.sort_values(
    ["nb_communes_gagnees_2027", "bloc"],
    ascending=[False, True]
).reset_index(drop=True)


# Calcul du classement par communes gagnées
nb_total = len(df)

rank_communes = summary_bloc[["bloc", "nb_communes_gagnees_2027"]].copy()
rank_communes["part_communes_gagnees_2027"] = (
    rank_communes["nb_communes_gagnees_2027"] / nb_total * 100
)
rank_communes["rang"] = range(1, len(rank_communes) + 1)
rank_communes = rank_communes[
    ["rang", "bloc", "nb_communes_gagnees_2027", "part_communes_gagnees_2027"]
]


# Calcul du classement par score moyen prédit
avg_scores = []
for bloc in bloc_names:
    col = f"{bloc}_pred_2027"
    avg_scores.append({
        "bloc": bloc,
        "score_moyen_pred_2027": df[col].mean(),
        "score_min_pred_2027": df[col].min(),
        "score_max_pred_2027": df[col].max()
    })

rank_scores = pd.DataFrame(avg_scores).sort_values(
    "score_moyen_pred_2027",
    ascending=False
).reset_index(drop=True)

rank_scores["rang"] = range(1, len(rank_scores) + 1)
rank_scores = rank_scores[[
    "rang",
    "bloc",
    "score_moyen_pred_2027",
    "score_min_pred_2027",
    "score_max_pred_2027"
]]


# Calcul des changements de bloc gagnant entre 2022 et 2027
switches = df[df["bloc_gagnant_2022"] != df["bloc_gagnant_pred_2027"]].copy()
nb_switches = len(switches)
share_switches = (nb_switches / nb_total * 100) if nb_total > 0 else 0


# Détermination du bloc en tête en 2027
top_bloc_communes = None
top_bloc_communes_nb = None
if not rank_communes.empty:
    top_bloc_communes = rank_communes.iloc[0]["bloc"]
    top_bloc_communes_nb = int(rank_communes.iloc[0]["nb_communes_gagnees_2027"])

top_bloc_score = None
top_bloc_score_value = None
if not rank_scores.empty:
    top_bloc_score = rank_scores.iloc[0]["bloc"]
    top_bloc_score_value = float(rank_scores.iloc[0]["score_moyen_pred_2027"])


# Construction d'un tableau des communes les plus favorables à chaque bloc
top_rows = []
for bloc in bloc_names:
    score_col = f"{bloc}_pred_2027"
    temp = pd.DataFrame({
        "code_commune": df["code_commune"],
        "nom_commune": df["nom_commune"],
        "bloc": bloc,
        "score_bloc_pred_2027": df[score_col]
    })
    temp = temp.sort_values("score_bloc_pred_2027", ascending=False).head(5)
    top_rows.append(temp)

top_communes_bloc = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()


# Fonction d'annotation des barres
def annotate_bars(ax, fmt="{:.1f}", percent=False):
    for bar in ax.patches:
        value = bar.get_width() if bar.get_width() != 0 else bar.get_height()
        if np.isnan(value):
            continue

        if bar.get_width() > 0:
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            label = fmt.format(value)
            if percent:
                label += "%"
            ax.annotate(
                label,
                (x, y),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9
            )
        else:
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            label = fmt.format(y)
            if percent:
                label += "%"
            ax.annotate(
                label,
                (x, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )


# Palette simple et stable par bloc
color_map = {
    "bloc_gauche_radicale": "#B22222",
    "bloc_gauche": "#E15759",
    "bloc_centre": "#4E79A7",
    "bloc_droite": "#F28E2B",
    "bloc_droite_radicale": "#7F3C8D",
    "bloc_souverainiste_autres": "#59A14F"
}


def bloc_color(bloc):
    return color_map.get(bloc, "#808080")


# Graphique du classement par nombre de communes gagnées
rank_communes_plot = rank_communes.sort_values("nb_communes_gagnees_2027", ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(
    rank_communes_plot["bloc"],
    rank_communes_plot["nb_communes_gagnees_2027"],
    color=[bloc_color(b) for b in rank_communes_plot["bloc"]]
)

ax.set_title("Classement des blocs par nombre de communes gagnées en 2027")
ax.set_xlabel("Nombre de communes gagnées")
ax.set_ylabel("Bloc")
ax.grid(axis="x", alpha=0.3)
annotate_bars(ax, fmt="{:.0f}")
plt.tight_layout()
plt.savefig(OUTPUT_RANK_COMMUNES, dpi=300, bbox_inches="tight")
plt.close()


# Graphique du classement par score moyen prédit
rank_scores_plot = rank_scores.sort_values("score_moyen_pred_2027", ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(
    rank_scores_plot["bloc"],
    rank_scores_plot["score_moyen_pred_2027"],
    color=[bloc_color(b) for b in rank_scores_plot["bloc"]]
)

ax.set_title("Classement des blocs par score moyen prédit en 2027")
ax.set_xlabel("Score moyen prédit (%)")
ax.set_ylabel("Bloc")
ax.grid(axis="x", alpha=0.3)
annotate_bars(ax, fmt="{:.2f}", percent=True)
plt.tight_layout()
plt.savefig(OUTPUT_RANK_SCORES, dpi=300, bbox_inches="tight")
plt.close()


# Graphique de comparaison 2022 / 2027 en nombre de communes gagnées
compare_plot = summary_bloc.sort_values("nb_communes_gagnees_2027", ascending=False).reset_index(drop=True)
x = np.arange(len(compare_plot))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 7))
bars1 = ax.bar(
    x - width / 2,
    compare_plot["nb_communes_gagnees_2022"],
    width,
    label="2022",
    color="#BDBDBD"
)
bars2 = ax.bar(
    x + width / 2,
    compare_plot["nb_communes_gagnees_2027"],
    width,
    label="2027 prédit",
    color=[bloc_color(b) for b in compare_plot["bloc"]]
)

ax.set_title("Comparaison du nombre de communes gagnées par bloc : 2022 vs 2027")
ax.set_xlabel("Bloc")
ax.set_ylabel("Nombre de communes")
ax.set_xticks(x)
ax.set_xticklabels(compare_plot["bloc"], rotation=20, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(OUTPUT_COMPARE_2022_2027, dpi=300, bbox_inches="tight")
plt.close()


# Graphique de répartition des gagnants en 2027
fig, ax = plt.subplots(figsize=(10, 8))
colors = [bloc_color(b) for b in rank_communes["bloc"]]

wedges, texts, autotexts = ax.pie(
    rank_communes["nb_communes_gagnees_2027"],
    labels=rank_communes["bloc"],
    autopct="%1.1f%%",
    startangle=90,
    colors=colors
)

ax.set_title("Répartition des communes gagnées par bloc en 2027")
ax.axis("equal")
plt.tight_layout()
plt.savefig(OUTPUT_REPARTITION_2027, dpi=300, bbox_inches="tight")
plt.close()


# Restitution console
print("ANALYSE DES RÉSULTATS 2027")
print(f"Nombre total de communes analysées : {nb_total}")
print(f"Nombre de communes avec changement de bloc gagnant : {nb_switches}")
print(f"Part des communes avec changement : {share_switches:.2f}%")

if top_bloc_communes is not None:
    print(f"Bloc en tête en nombre de communes gagnées : {top_bloc_communes} ({top_bloc_communes_nb} communes)")

if top_bloc_score is not None:
    print(f"Bloc en tête en score moyen prédit : {top_bloc_score} ({top_bloc_score_value:.2f}%)")

print("\n CLASSEMENT PAR COMMUNES GAGNÉES ")
print(rank_communes.to_string(index=False))

print("\n CLASSEMENT PAR SCORE MOYEN PRÉDIT ")
print(rank_scores.to_string(index=False))

print("\n ÉVOLUTION 2022 / 2027 ")
print(summary_bloc.to_string(index=False))

if not top_communes_bloc.empty:
    print("\n COMMUNES LES PLUS FAVORABLES PAR BLOC ")
    print(top_communes_bloc.to_string(index=False))

print("\nGraphiques enregistrés :")
print("-", OUTPUT_RANK_COMMUNES)
print("-", OUTPUT_RANK_SCORES)
print("-", OUTPUT_COMPARE_2022_2027)
print("-", OUTPUT_REPARTITION_2027)