import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBRegressor

from sklearn import set_config

# Forza a Scikit-Learn a mantener los nombres de columnas como Pandas DataFrames
set_config(transform_output="pandas")

# ---------------------------------------------------------
# 1. Mapeos Ordinales Explícitos
# ---------------------------------------------------------
ORDINAL_MAPS = {
    "ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "PoolQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
}


# ---------------------------------------------------------
# 2. Advanced Feature Engineering Pipeline
# ---------------------------------------------------------
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, skew_threshold=0.75):
        self.skew_threshold = skew_threshold
        self.skewed_feats_ = []

    def fit(self, X, y=None):
        X_temp = X.copy()
        num_cols = X_temp.select_dtypes(include=[np.number]).columns
        skewness = X_temp[num_cols].apply(lambda x: skew(x.dropna()))
        self.skewed_feats_ = skewness[
            abs(skewness) > self.skew_threshold
        ].index.tolist()
        return self

    def transform(self, X):
        X_out = X.copy()

        # Mapeo ordinal
        for col, mapping in ORDINAL_MAPS.items():
            if col in X_out.columns:
                X_out[col] = (
                    X_out[col].fillna("None").map(mapping).fillna(0).astype(int)
                )

        # Totales combinados
        X_out["TotalSF"] = (
            X_out["TotalBsmtSF"].fillna(0)
            + X_out["1stFlrSF"].fillna(0)
            + X_out["2ndFlrSF"].fillna(0)
        )
        X_out["TotalPorchSF"] = (
            X_out["OpenPorchSF"].fillna(0)
            + X_out["EnclosedPorch"].fillna(0)
            + X_out["3SsnPorch"].fillna(0)
            + X_out["ScreenPorch"].fillna(0)
            + X_out["WoodDeckSF"].fillna(0)
        )
        X_out["TotalBaths"] = (
            X_out["FullBath"].fillna(0)
            + 0.5 * X_out["HalfBath"].fillna(0)
            + X_out["BsmtFullBath"].fillna(0)
            + 0.5 * X_out["BsmtHalfBath"].fillna(0)
        )

        # Antigüedad y métricas relativas
        X_out["HouseAge"] = X_out["YrSold"] - X_out["YearBuilt"]
        X_out["RemodAge"] = X_out["YrSold"] - X_out["YearRemodAdd"]
        X_out["IsRemodeled"] = (
            X_out["YearRemodAdd"] != X_out["YearBuilt"]
        ).astype(int)

        # Interacciones clave
        X_out["Qual_x_TotalSF"] = X_out["OverallQual"] * X_out["TotalSF"]

        # Indicadores binarios
        X_out["HasBasement"] = (X_out["TotalBsmtSF"] > 0).astype(int)
        X_out["HasGarage"] = (X_out["GarageArea"] > 0).astype(int)
        X_out["HasFireplace"] = (X_out["Fireplaces"] > 0).astype(int)

        # Corrección de skewness
        for col in self.skewed_feats_:
            if col in X_out.columns and col != "YrSold":
                min_val = X_out[col].min()
                if min_val >= 0:
                    X_out[col] = np.log1p(X_out[col])

        return X_out


# ---------------------------------------------------------
# 3. Carga de Datos
# ---------------------------------------------------------
train = pd.read_csv("/kaggle/input/competitions/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/competitions/house-prices-advanced-regression-techniques/test.csv")

# Eliminación de outliers atípicos en train únicamente
train = train.drop(
    train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 200000)].index
)

X_train = train.drop(columns=["Id", "SalePrice"])
y_log = np.log1p(train["SalePrice"])
X_test = test.drop(columns=["Id"])
test_ids = test["Id"]

# ---------------------------------------------------------
# 4. Definición del Preprocesador
# ---------------------------------------------------------
fe_temp = AdvancedFeatureEngineer()
X_trans = fe_temp.fit_transform(X_train)

none_cols = [
    "Alley",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageType",
    "GarageFinish",
    "Fence",
    "MiscFeature",
]
freq_cols = [
    c
    for c in X_trans.select_dtypes(include=["object"]).columns
    if c not in none_cols
]
num_cols = X_trans.select_dtypes(include=[np.number]).columns.tolist()

num_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
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
# 5. Definición de Modelos Individuales
# ---------------------------------------------------------
models = {
    "Lasso": LassoCV(
        alphas=np.logspace(-4, -1, 30), cv=5, max_iter=10000, random_state=42
    ),
    "ElasticNet": ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9],
        alphas=np.logspace(-4, -1, 30),
        cv=5,
        max_iter=10000,
        random_state=42,
    ),
    "LGBM": LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.015,
        max_depth=4,
        num_leaves=15,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=1000,
        learning_rate=0.015,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    ),
}

# ---------------------------------------------------------
# 6. Evaluación de Modelos con K-Fold CV
# ---------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 55)
print("   EVALUACIÓN DE MODELOS INDIVIDUALES EN 5-FOLD CV")
print("=" * 55)

fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(
        steps=[
            ("fe", AdvancedFeatureEngineer()),
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    scores = cross_val_score(
        pipe,
        X_train,
        y_log,
        scoring="neg_mean_squared_error",
        cv=kf,
        n_jobs=-1,
    )
    rmse = np.sqrt(-scores)
    print(f"{name:12s} -> RMSE CV: {rmse.mean():.4f} (+/- {rmse.std():.4f})")

    # Entrenar en todo el conjunto de entrenamiento para la inferencia posterior
    pipe.fit(X_train, y_log)
    fitted_pipelines[name] = pipe

print("=" * 55)

# ---------------------------------------------------------
# 7. Generación de Predicciones y Blending Ponderado
# ---------------------------------------------------------
# Generar predicciones en escala logarítmica sobre el conjunto de test
preds_lasso = fitted_pipelines["Lasso"].predict(X_test)
preds_elastic = fitted_pipelines["ElasticNet"].predict(X_test)
preds_lgbm = fitted_pipelines["LGBM"].predict(X_test)
preds_xgb = fitted_pipelines["XGBoost"].predict(X_test)

# Blending ponderado (Darle más peso a Lasso/LGBM según el CV)
final_log_preds = (
    0.35 * preds_lasso
    + 0.15 * preds_elastic
    + 0.25 * preds_lgbm
    + 0.25 * preds_xgb
)

# Invertir la transformación logarítmica log1p -> expm1
final_preds = np.expm1(final_log_preds)
