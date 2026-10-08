import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# ---------------------------------------------------------
# 1. Transformer Personalizado para Feature Engineering
# ---------------------------------------------------------
class HousePricesFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No aprende parámetros, solo transforma

    def transform(self, X):
        X_out = X.copy()
        
        # Áreas combinadas
        X_out["TotalSF"] = (
            X_out["TotalBsmtSF"].fillna(0) + 
            X_out["1stFlrSF"].fillna(0) + 
            X_out["2ndFlrSF"].fillna(0)
        )
        X_out["TotalPorchSF"] = (
            X_out["OpenPorchSF"].fillna(0) + 
            X_out["EnclosedPorch"].fillna(0) + 
            X_out["3SsnPorch"].fillna(0) + 
            X_out["ScreenPorch"].fillna(0) + 
            X_out["WoodDeckSF"].fillna(0)
        )
        
        # Conteo total de baños
        X_out["TotalBaths"] = (
            X_out["FullBath"].fillna(0) + 
            0.5 * X_out["HalfBath"].fillna(0) + 
            X_out["BsmtFullBath"].fillna(0) + 
            0.5 * X_out["BsmtHalfBath"].fillna(0)
        )
        
        # Antigüedad
        X_out["HouseAge"] = X_out["YrSold"] - X_out["YearBuilt"]
        X_out["RemodAge"] = X_out["YrSold"] - X_out["YearRemodAdd"]
        X_out["IsRemodeled"] = (X_out["YearRemodAdd"] != X_out["YearBuilt"]).astype(int)
        
        # Calidad combinada
        X_out["OverallScore"] = X_out["OverallQual"] * X_out["OverallCond"]
        
        # Indicadores de existencia (Presencia vs Ausencia)
        X_out["HasBasement"] = (X_out["TotalBsmtSF"] > 0).astype(int)
        X_out["HasGarage"] = (X_out["GarageArea"] > 0).astype(int)
        X_out["HasFireplace"] = (X_out["Fireplaces"] > 0).astype(int)
        
        return X_out

# ---------------------------------------------------------
# 2. Carga y Limpieza Base
# ---------------------------------------------------------
df_train = pd.read_csv("train.csv")

# Eliminación de outliers identificados en el EDA
df_train = df_train.drop(
    df_train[(df_train["GrLivArea"] > 4000) & (df_train["SalePrice"] < 200000)].index
)

X = df_train.drop(columns=["Id", "SalePrice"])
y_log = np.log1p(df_train["SalePrice"])

# Aplicamos Feature Engineering inicial para actualizar las listas de columnas
fe = HousePricesFeatureEngineer()
X_fe = fe.transform(X)

# ---------------------------------------------------------
# 3. Clasificación de Columnas
# ---------------------------------------------------------
none_cols = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
]

freq_cols = [
    c for c in X_fe.select_dtypes(include=["object"]).columns if c not in none_cols
]

num_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()

# ---------------------------------------------------------
# 4. Pipeline de Preprocesamiento Avanzado
# ---------------------------------------------------------
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())  # RobustScaler es más estable ante outliers residuales que StandardScaler
])

none_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

freq_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("none", none_transformer, none_cols),
    ("freq", freq_transformer, freq_cols)
])

# ---------------------------------------------------------
# 5. Pipeline Completo: Feature Engineering + Scaling + Lasso
# ---------------------------------------------------------
improved_pipeline = Pipeline(steps=[
    ("feature_engineering", HousePricesFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("model", LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=42))
])

# ---------------------------------------------------------
# 6. Evaluación de la Mejora
# ---------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores_neg_mse = cross_val_score(
    improved_pipeline,
    X,
    y_log,
    scoring="neg_mean_squared_error",
    cv=kf,
    n_jobs=-1
)

rmse_improved = np.sqrt(-scores_neg_mse)

print("=" * 45)
print("   RESULTADOS CON FEATURE ENGINEERING + LASSO")
print("=" * 45)
print(f"RMSE por fold: {np.round(rmse_improved, 4)}")
print(f"RMSE Promedio: {rmse_improved.mean():.4f} (std: +/- {rmse_improved.std():.4f})")
print("=" * 45)
