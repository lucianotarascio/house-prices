import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# ---------------------------------------------Inicio
df_train = pd.read_csv('train.csv')

df_train.info()
df_train.describe().T

df_train["MSSubClass"] = df_train["MSSubClass"].astype(str)

#----------------------------------------------Graph SalePrice
# Configuración visual
sns.set_theme(style="whitegrid")

# Distribución original vs logarítmica
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df_train["SalePrice"], kde=True, ax=axes[0])
axes[0].set_title(f"Original SalePrice (Skewness: {df_train['SalePrice'].skew():.2f})")

sale_price_log = np.log1p(df_train["SalePrice"])
sns.histplot(sale_price_log, kde=True, ax=axes[1], color="green")
axes[1].set_title(f"log1p(SalePrice) (Skewness: {sale_price_log.skew():.2f})")

plt.tight_layout()
plt.show()


#---------------------------------Valores Nulos

# Porcentaje de valores nulos por columna
missing_series = df_train.isnull().sum()
missing_pct = (missing_series / len(df_train)) * 100
missing_df = (
    pd.DataFrame({"Missing Count": missing_series, "Percentage (%)": missing_pct})
    .query("`Missing Count` > 0")
    .sort_values(by="Percentage (%)", ascending=False)
)

print(missing_df)


#-----------------------------------Dispersión entre superficie y precio

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df_train["GrLivArea"], y=np.log1p(df_train["SalePrice"]), alpha=0.7
)
plt.axvline(x=4000, color="red", linestyle="--", label="Umbral de atipicidad área")
plt.title("GrLivArea vs log(SalePrice)")
plt.legend()
plt.show()

#------------------------------------Cálculo de correlaciones

numeric_cols = df_train.select_dtypes(include=[np.number]).columns
correlations = (
    df_train[numeric_cols]
    .assign(log_SalePrice=np.log1p(df_train["SalePrice"]))
    .corr(method="spearman")["log_SalePrice"]
    .sort_values(ascending=False)
)

print("Top 10 variables más correlacionadas con log(SalePrice):")
print(correlations.head(11))

#--------------------------------Ordenar barrios por precio mediano
sorted_neighborhoods = (
    df_train.assign(log_Price=np.log1p(df_train["SalePrice"]))
    .groupby("Neighborhood")["log_Price"]
    .median()
    .sort_values()
    .index
)

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_train.assign(log_Price=np.log1p(df_train["SalePrice"])),
    x="Neighborhood",
    y="log_Price",
    order=sorted_neighborhoods,
)
plt.xticks(rotation=45, ha="right")
plt.title("Distribución de log(SalePrice) por Barrio (Ordenado por Mediana)")
plt.tight_layout()
plt.show()


#------------------------------------Categoría de varianza baja
low_variance_cols = []
for col in df_train.select_dtypes(include=["object"]).columns:
    top_pct = df_train[col].value_counts(normalize=True).iloc[0]
    if top_pct > 0.95:
        low_variance_cols.append((col, round(top_pct * 100, 2)))

print("Variables categóricas con varianza casi nula (>95% un solo valor):")
print(low_variance_cols)

# Ejemplos típicos en este dataset: 'Utilities', 'Street', 'Condition2'





