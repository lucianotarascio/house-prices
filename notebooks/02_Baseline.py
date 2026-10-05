import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------
# Cargar y Limpiar Datos
# ---------------------------------------------------------
train = pd.read_csv("train.csv")

# Eliminación de outliers atípicos identificados en el EDA
train = train.drop(
    train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 200000)].index
)

# Separar X e y (transformación logarítmica de la variable objetivo)
X = train.drop(columns=["Id", "SalePrice"])
y_log = np.log1p(train["SalePrice"])

# ---------------------------------------------------------
# Definir Tipos de Columnas
# ---------------------------------------------------------
# Columnas donde 'NA' significa ausencia de la característica
none_cols = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

# Columnas categóricas nominales estándar
freq_cols = [
    c for c in X.select_dtypes(include=["object"]).columns if c not in none_cols
]

# Columnas numéricas
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# ---------------------------------------------------------
# Construir Transformers y Pipeline de Preprocesamiento
# ---------------------------------------------------------
num_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

none_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

freq_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("none", none_transformer, none_cols),
        ("freq", freq_transformer, freq_cols),
    ]
)

# ---------------------------------------------------------
# 4. Definir el Pipeline Completo con el Modelo (Ridge)
# ---------------------------------------------------------
baseline_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=10.0))]
)

# ---------------------------------------------------------
# Evaluación mediante Validación Cruzada (5-Fold CV)
# ---------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluamos con MSE negativo y aplicamos la raíz cuadrada para obtener el RMSE sobre log(SalePrice)
neg_mse_scores = cross_val_score(
    baseline_pipeline,
    X,
    y_log,
    scoring="neg_mean_squared_error",
    cv=kf,
    n_jobs=-1,
)

rmse_scores = np.sqrt(-neg_mse_scores)

print("=" * 40)
print("   RESULTADOS DEL MODELO BASELINE (Ridge)")
print("=" * 40)
print(f"RMSE por fold: {np.round(rmse_scores, 4)}")
print(
    f"RMSE Promedio: {rmse_scores.mean():.4f} (std: +/- {rmse_scores.std():.4f})"
)
print("=" * 40)
