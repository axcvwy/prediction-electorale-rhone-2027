import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

try:
    from sklearn.metrics import root_mean_squared_error
    HAS_RMSE_FUNC = True
except ImportError:
    HAS_RMSE_FUNC = False


INPUT_FILE = Path("Data_Model/dataset_features.csv")
OUTPUT_DIR = Path("Data_Viz")

OUTPUT_GLOBAL_PNG = OUTPUT_DIR / "compare_models_global_metrics.png"
OUTPUT_WINNERS_PNG = OUTPUT_DIR / "compare_models_winner_metrics.png"
OUTPUT_PER_BLOC_PNG = OUTPUT_DIR / "compare_models_mae_per_bloc.png"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_FILE, sep=";")


# Colonnes cibles des blocs en 2022
target_cols = sorted([
    c for c in df.columns
    if c.startswith("bloc_")
    and c.endswith("_2022")
    and not c.startswith("bloc_gagnant_")
])

bloc_bases = [c[:-5] for c in target_cols]


# Colonnes de variables explicatives disponibles avant 2022
feature_cols = []
for c in df.columns:
    if c in ["code_commune", "nom_commune"]:
        continue
    if c.startswith("bloc_gagnant_"):
        continue
    if c.endswith("_2022"):
        continue
    if c.endswith("_2012") or c.endswith("_2017") or c.endswith("_delta_12_17"):
        feature_cols.append(c)


required_input_cols = [f"{b}_2012" for b in bloc_bases] + [f"{b}_2017" for b in bloc_bases]
required_target_cols = target_cols

model_df = df.dropna(subset=required_input_cols + required_target_cols).copy()

X = model_df[feature_cols].copy()
y = model_df[target_cols].copy()
meta = model_df[["code_commune", "nom_commune"]].copy()


# Même séparation apprentissage / test pour tous les modèles
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=0.2, random_state=42
)


def normalize_predictions(pred):
    pred = np.clip(pred, 0, None)
    row_sums = pred.sum(axis=1, keepdims=True)
    pred = np.divide(pred, row_sums, out=np.zeros_like(pred), where=row_sums != 0) * 100
    return pred


def compute_rmse(y_true, y_pred):
    if HAS_RMSE_FUNC:
        return root_mean_squared_error(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = normalize_predictions(pred)

    overall_mae = mean_absolute_error(y_test.values, pred)
    overall_rmse = compute_rmse(y_test.values, pred)
    overall_r2 = r2_score(y_test.values, pred, multioutput="uniform_average")

    actual_winner = np.array(bloc_bases)[np.argmax(y_test.values, axis=1)]
    pred_winner = np.array(bloc_bases)[np.argmax(pred, axis=1)]
    winner_accuracy = accuracy_score(actual_winner, pred_winner)

    sorted_pred = np.sort(pred, axis=1)
    top2_margin = sorted_pred[:, -1] - sorted_pred[:, -2]

    global_row = {
        "model": model_name,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_r2": overall_r2,
        "winner_accuracy": winner_accuracy,
        "winner_margin_mean": float(np.mean(top2_margin)),
        "winner_margin_median": float(np.median(top2_margin)),
        "winner_margin_min": float(np.min(top2_margin)),
        "winner_margin_max": float(np.max(top2_margin)),
    }

    per_bloc_rows = []
    for i, col in enumerate(target_cols):
        per_bloc_rows.append({
            "model": model_name,
            "target": col,
            "mae": mean_absolute_error(y_test[col], pred[:, i]),
            "rmse": compute_rmse(y_test[col], pred[:, i]),
            "r2": r2_score(y_test[col], pred[:, i]),
            "pred_mean": float(np.mean(pred[:, i])),
            "actual_mean": float(np.mean(y_test[col])),
        })

    winner_row = {
        "model": model_name,
        "winner_accuracy": winner_accuracy,
        "nb_test_rows": len(y_test),
        "nb_correct_winners": int(np.sum(actual_winner == pred_winner)),
        "nb_wrong_winners": int(np.sum(actual_winner != pred_winner)),
    }

    return global_row, per_bloc_rows, winner_row


models = {
    "lasso": MultiOutputRegressor(
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LassoCV(
                alphas=np.logspace(-2, 1, 15),
                cv=5,
                max_iter=100000,
                tol=1e-3,
                random_state=42
            ))
        ])
    ),

    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "gradient_boosting": MultiOutputRegressor(
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ))
        ])
    ),
}


global_rows = []
per_bloc_rows = []
winner_rows = []

for model_name, model in models.items():
    global_row, bloc_rows, winner_row = evaluate_model(
        model_name, model, X_train, X_test, y_train, y_test
    )
    global_rows.append(global_row)
    per_bloc_rows.extend(bloc_rows)
    winner_rows.append(winner_row)


df_global = pd.DataFrame(global_rows).sort_values(
    by=["winner_accuracy", "overall_mae"],
    ascending=[False, True]
).reset_index(drop=True)

df_per_bloc = pd.DataFrame(per_bloc_rows).sort_values(
    by=["target", "mae"]
).reset_index(drop=True)

df_winners = pd.DataFrame(winner_rows).sort_values(
    by=["winner_accuracy", "nb_correct_winners"],
    ascending=[False, False]
).reset_index(drop=True)


print("COMPARAISON GLOBALE DES MODÈLES")
print(df_global.to_string(index=False))

print("\nPRÉCISION DES GAGNANTS")
print(df_winners.to_string(index=False))

print("\nMAE PAR BLOC")
print(df_per_bloc.to_string(index=False))


# Graphique des métriques globales
plot_df = df_global.copy()
x = np.arange(len(plot_df))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, plot_df["overall_mae"], width, label="MAE")
ax.bar(x, plot_df["overall_rmse"], width, label="RMSE")
ax.bar(x + width, plot_df["overall_r2"], width, label="R2")

ax.set_title("Comparaison globale des modèles")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["model"])
ax.set_ylabel("Valeur")
ax.legend()
ax.grid(axis="y", alpha=0.3)

for i, v in enumerate(plot_df["overall_mae"]):
    ax.text(i - width, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

for i, v in enumerate(plot_df["overall_rmse"]):
    ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

for i, v in enumerate(plot_df["overall_r2"]):
    ax.text(i + width, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_GLOBAL_PNG, dpi=300, bbox_inches="tight")
plt.close()


# Graphique des métriques de gagnant
plot_df = df_global.copy()
x = np.arange(len(plot_df))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width / 2, plot_df["winner_accuracy"], width, label="Winner accuracy")
ax.bar(x + width / 2, plot_df["winner_margin_mean"], width, label="Marge moyenne top 2")

ax.set_title("Comparaison des gagnants prédits")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["model"])
ax.set_ylabel("Valeur")
ax.legend()
ax.grid(axis="y", alpha=0.3)

for i, v in enumerate(plot_df["winner_accuracy"]):
    ax.text(i - width / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

for i, v in enumerate(plot_df["winner_margin_mean"]):
    ax.text(i + width / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_WINNERS_PNG, dpi=300, bbox_inches="tight")
plt.close()


# Graphique des MAE par bloc
pivot_mae = df_per_bloc.pivot(index="target", columns="model", values="mae")
pivot_mae = pivot_mae[[c for c in ["lasso", "random_forest", "gradient_boosting"] if c in pivot_mae.columns]]

x = np.arange(len(pivot_mae.index))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))

for i, model_name in enumerate(pivot_mae.columns):
    ax.bar(x + (i - 1) * width, pivot_mae[model_name].values, width, label=model_name)

ax.set_title("MAE par bloc et par modèle")
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_2022", "") for c in pivot_mae.index], rotation=20, ha="right")
ax.set_ylabel("MAE")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PER_BLOC_PNG, dpi=300, bbox_inches="tight")
plt.close()


best_model = df_global.iloc[0]["model"]

print("\nModèle retenu :", best_model)
print("\nVisualisations enregistrées dans :", OUTPUT_DIR)
print("-", OUTPUT_GLOBAL_PNG.name)
print("-", OUTPUT_WINNERS_PNG.name)
print("-", OUTPUT_PER_BLOC_PNG.name)