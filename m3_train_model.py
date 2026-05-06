import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

try:
    from sklearn.metrics import root_mean_squared_error
    HAS_RMSE_FUNC = True
except ImportError:
    HAS_RMSE_FUNC = False


INPUT_FILE = Path("Data_Model/dataset_features.csv")

OUTPUT_METRICS = Path("Data_Model/model_metrics_results.csv")
OUTPUT_TEST_PRED = Path("Data_Model/test_predictions_2022.csv")
OUTPUT_ALL_PRED_2022 = Path("Data_Model/predictions_all_2022.csv")
OUTPUT_FORECAST_2027 = Path("Data_Model/predictions_2027_results.csv")
OUTPUT_IMPORTANCE = Path("Data_Model/model_coefficients.csv")


df = pd.read_csv(INPUT_FILE, sep=";")


# Détection des colonnes cibles des blocs en 2022
target_cols = sorted([
    c for c in df.columns
    if c.startswith("bloc_")
    and c.endswith("_2022")
    and not c.startswith("bloc_gagnant_")
])

bloc_bases = [c[:-5] for c in target_cols]


# Construction de la liste des variables explicatives disponibles avant 2022
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


# Vérification des colonnes minimales nécessaires à l'entraînement
required_input_cols = [f"{b}_2012" for b in bloc_bases] + [f"{b}_2017" for b in bloc_bases]
required_target_cols = target_cols

model_df = df.dropna(subset=required_input_cols + required_target_cols).copy()

X = model_df[feature_cols].copy()
y = model_df[target_cols].copy()
meta = model_df[["code_commune", "nom_commune"]].copy()


# Séparation apprentissage / test pour évaluer le modèle sur un jeu non vu
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=0.2, random_state=42
)


# Fonction de renormalisation des prédictions pour obtenir des parts positives totalisant 100
def normalize_predictions(pred):
    pred = np.clip(pred, 0, None)
    row_sums = pred.sum(axis=1, keepdims=True)
    pred = np.divide(pred, row_sums, out=np.zeros_like(pred), where=row_sums != 0) * 100
    return pred


# Fonction compatible avec différentes versions de scikit-learn
def compute_rmse(y_true, y_pred):
    if HAS_RMSE_FUNC:
        return root_mean_squared_error(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Pipeline de modélisation avec imputation, standardisation et Lasso multi-sorties
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", MultiOutputRegressor(
        LassoCV(
            alphas=np.logspace(-2, 1, 15),
            cv=5,
            max_iter=100000,
            tol=1e-3,
            random_state=42,
            selection="random"
        ),
        n_jobs=-1
    ))
])


# Entraînement sur l'échantillon d'apprentissage
pipeline.fit(X_train, y_train)


# Prédiction sur l'échantillon de test pour évaluer la performance en 2022
y_pred = pipeline.predict(X_test)
y_pred = normalize_predictions(y_pred)


# Calcul des métriques par bloc
metrics_rows = []

for i, col in enumerate(target_cols):
    mae = mean_absolute_error(y_test[col], y_pred[:, i])
    rmse = compute_rmse(y_test[col], y_pred[:, i])
    r2 = r2_score(y_test[col], y_pred[:, i])

    metrics_rows.append({
        "target": col,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })


# Calcul des métriques globales
overall_mae = mean_absolute_error(y_test.values, y_pred)
overall_rmse = compute_rmse(y_test.values, y_pred)
overall_r2 = r2_score(y_test.values, y_pred, multioutput="uniform_average")

actual_winner = np.array(bloc_bases)[np.argmax(y_test.values, axis=1)]
pred_winner = np.array(bloc_bases)[np.argmax(y_pred, axis=1)]
winner_accuracy = accuracy_score(actual_winner, pred_winner)

metrics_df = pd.DataFrame(metrics_rows)
summary_df = pd.DataFrame([{
    "target": "OVERALL",
    "mae": overall_mae,
    "rmse": overall_rmse,
    "r2": overall_r2,
    "winner_accuracy": winner_accuracy
}])

metrics_out = pd.concat([metrics_df, summary_df], ignore_index=True)
metrics_out.to_csv(OUTPUT_METRICS, sep=";", index=False, encoding="utf-8-sig")


# Export des prédictions de test commune par commune
test_pred_df = meta_test.reset_index(drop=True).copy()
y_test_reset = y_test.reset_index(drop=True)

for i, col in enumerate(target_cols):
    test_pred_df[f"actual_{col}"] = y_test_reset[col]
    test_pred_df[f"pred_{col}"] = y_pred[:, i]

test_pred_df["actual_winner_2022"] = actual_winner
test_pred_df["pred_winner_2022"] = pred_winner
test_pred_df.to_csv(OUTPUT_TEST_PRED, sep=";", index=False, encoding="utf-8-sig")


# Réentraînement sur toutes les données disponibles avant la simulation 2027
pipeline.fit(X, y)


# Export des prédictions 2022 sur l'ensemble des communes pour Power BI
pred_all_2022 = pipeline.predict(X)
pred_all_2022 = normalize_predictions(pred_all_2022)

all_pred_df = meta.reset_index(drop=True).copy()
y_reset = y.reset_index(drop=True)

for i, col in enumerate(target_cols):
    all_pred_df[f"actual_{col}"] = y_reset[col]
    all_pred_df[f"pred_{col}"] = pred_all_2022[:, i]

all_pred_df["actual_winner_2022"] = np.array(bloc_bases)[np.argmax(y.values, axis=1)]
all_pred_df["pred_winner_2022"] = np.array(bloc_bases)[np.argmax(pred_all_2022, axis=1)]

all_pred_df.to_csv(OUTPUT_ALL_PRED_2022, sep=";", index=False, encoding="utf-8-sig")


# Export des coefficients du modèle Lasso pour chaque cible
model = pipeline.named_steps["model"]
coef_matrix = np.column_stack([est.coef_ for est in model.estimators_])

coef_df = pd.DataFrame(coef_matrix, index=feature_cols, columns=target_cols)
coef_df.index.name = "feature"
coef_df["mean_abs_coef"] = coef_df.abs().mean(axis=1)
coef_df = coef_df.sort_values("mean_abs_coef", ascending=False)

coef_df.to_csv(OUTPUT_IMPORTANCE, sep=";", index=True, encoding="utf-8-sig")


# Construction des variables d'entrée de la simulation 2027
forecast_required = [f"{b}_2017" for b in bloc_bases] + [f"{b}_2022" for b in bloc_bases]
forecast_df = df.dropna(subset=forecast_required).copy()

X_2027 = pd.DataFrame(index=forecast_df.index)

for col in feature_cols:
    if col.endswith("_2012"):
        src = col[:-5] + "_2017"
    elif col.endswith("_2017"):
        src = col[:-5] + "_2022"
    elif col.endswith("_delta_12_17"):
        src = col.replace("_delta_12_17", "_delta_17_22")
    else:
        src = None

    if src in forecast_df.columns:
        X_2027[col] = forecast_df[src]
    else:
        X_2027[col] = np.nan


# Prédiction des scores de blocs simulés pour 2027
pred_2027 = pipeline.predict(X_2027)
pred_2027 = normalize_predictions(pred_2027)

forecast_out = forecast_df[["code_commune", "nom_commune"]].copy()

for i, bloc in enumerate(bloc_bases):
    forecast_out[f"{bloc}_pred_2027"] = pred_2027[:, i]

forecast_out["bloc_gagnant_2022"] = np.array(bloc_bases)[
    np.argmax(forecast_df[target_cols].values, axis=1)
]

forecast_out["bloc_gagnant_pred_2027"] = np.array(bloc_bases)[
    np.argmax(pred_2027, axis=1)
]

forecast_out["score_gagnant_pred_2027"] = pred_2027.max(axis=1)

forecast_out.to_csv(OUTPUT_FORECAST_2027, sep=";", index=False, encoding="utf-8-sig")


print("training rows:", len(model_df))
print("test rows:", len(X_test))
print("overall MAE:", round(overall_mae, 4))
print("overall RMSE:", round(overall_rmse, 4))
print("overall R2:", round(overall_r2, 4))
print("winner accuracy (2022 holdout):", round(winner_accuracy, 4))
print("saved:", OUTPUT_METRICS)
print("saved:", OUTPUT_TEST_PRED)
print("saved:", OUTPUT_ALL_PRED_2022)
print("saved:", OUTPUT_FORECAST_2027)
print("saved:", OUTPUT_IMPORTANCE)