# Predicción de precios de casas (Ames, Iowa)

Modelo de regresión que predice el precio de venta de viviendas a partir de 79 características (superficie, calidad, ubicación, antigüedad, etc.), con foco en un proceso reproducible, sin *data leakage* y con análisis de errores e interpretabilidad.

> **Resultado principal:** el mejor modelo (ensamble de `[MODELO A]` + `[MODELO B]`) logra un **RMSE de `[X.XXX]`** en validación cruzada (sobre el log del precio), frente a 0.1148 del baseline.

---

## 1. Problema y objetivo

Estimar `SalePrice` de una vivienda a partir de sus atributos. Un modelo así puede servir como referencia para tasaciones, detección de propiedades mal valuadas o apoyo a compradores y vendedores.

- **Dataset:** [House Prices: Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (~1.460 filas de entrenamiento, 79 variables).
- **Métrica:** RMSE sobre `log(SalePrice)` (la misma que usa Kaggle), que penaliza por igual los errores relativos en casas baratas y caras.

---

## 2. Resultados

Validación cruzada de 5 folds (`KFold`, `random_state=42`). Menor es mejor.

| Modelo | CV RMSE (log) | Desvío entre folds |
|---|---|---|
| Baseline: media (`DummyRegressor`) | `[X.XXX]` | `[X.XXX]` |
| Regresión lineal (variables básicas) | `[X.XXX]` | `[X.XXX]` |
| Ridge | `[X.XXX]` | `[X.XXX]` |
| Lasso | `[X.XXX]` | `[X.XXX]` |
| Random Forest | `[X.XXX]` | `[X.XXX]` |
| LightGBM (afinado con Optuna) | `[X.XXX]` | `[X.XXX]` |
| **Ensamble final** | **`[X.XXX]`** | `[X.XXX]` |

**Score público en Kaggle:** `[X.XXXXX]` (se reporta solo como referencia; la selección de modelos se hizo con validación cruzada).

---

## 3. Hallazgos clave del análisis exploratorio

- `SalePrice` tiene una distribución asimétrica con cola larga a la derecha, por lo que se entrena sobre `log1p(SalePrice)`.
- En muchas columnas (`PoolQC`, `Alley`, `GarageType`, `BsmtQual`, etc.) el valor nulo **significa "no tiene"**, no un dato perdido. Se imputan como categoría `"None"` según el diccionario de datos.
- Se detectaron `[N]` outliers claros (casas con `GrLivArea` > 4000 y precio anormalmente bajo) que se `[eliminaron / mantuvieron]` por `[motivo]`.
- Las variables más correlacionadas con el precio son `OverallQual`, `GrLivArea`, `TotalBsmtSF` y `GarageCars`.
- `[Agregar 1 o 2 hallazgos propios]`

---

## 4. Decisiones de preprocesamiento

Todo el preprocesamiento vive dentro de un `Pipeline` de scikit-learn, de modo que se ajusta **solo con los datos de entrenamiento de cada fold** (sin *data leakage*).

| Tipo de variable | Tratamiento |
|---|---|
| Numéricas | Imputación con mediana; `log1p` en las de fuerte asimetría; escalado para modelos lineales |
| Categóricas nominales | Imputación con `"None"` + `OneHotEncoder(handle_unknown="ignore")` |
| Categóricas ordinales (calidades Ex/Gd/TA/Fa/Po) | `OrdinalEncoder` con el orden correcto |

### Feature engineering

| Variable nueva | Definición | Impacto en CV |
|---|---|---|
| `TotalSF` | Sótano + planta baja + primer piso | `[mejoró / sin cambio]` |
| `TotalBath` | Baños completos + 0,5 × medios baños | `[mejoró / sin cambio]` |
| `HouseAge` | Año de venta − año de construcción | `[mejoró / sin cambio]` |
| `Remodeled` | Booleano (remodelada o no) | `[mejoró / sin cambio]` |
| `HasGarage`, `HasPool`, `HasFireplace` | Indicadores binarios | `[mejoró / sin cambio]` |

---

## 5. Modelado

1. **Baselines** para tener una referencia mínima.
2. **Modelos lineales regularizados** (Ridge, Lasso, ElasticNet).
3. **Modelos de árboles** (Random Forest, LightGBM).
4. **Optimización de hiperparámetros** con Optuna (`[N]` trials, optimizando CV RMSE).
5. **Ensamble** por promedio de `[MODELO A]` y `[MODELO B]`, elegidos por ser de familias distintas (errores poco correlacionados).

Las predicciones se transforman de vuelta a la escala original con `expm1`.

---

## 6. Interpretabilidad y análisis de errores

- **Importancia de variables (SHAP):** los factores que más influyen en el precio son `[VAR 1]`, `[VAR 2]` y `[VAR 3]`. `[Comentar si coincide con la intuición del negocio.]`
- **Residuos:** el modelo se equivoca más en `[casas de mayor precio / casas atípicas / cierto barrio]`. Posible causa: `[hipótesis]`.
- **Qué probaría para mejorar:** `[ej. variables de ubicación más finas, tratamiento específico de casas de lujo, más datos]`.

---

## 7. Qué funcionó y qué no

**Funcionó**
- `[ej. Imputar nulos según el diccionario de datos redujo el RMSE en X]`
- `[ej. Ensamble Lasso + LightGBM]`

**No funcionó**
- `[ej. Crear interacciones polinómicas de todas las variables: más ruido, sin mejora]`
- `[ej. Eliminar outliers adicionales empeoró el score en Kaggle]`

---

## 8. Limitaciones y próximos pasos

- El dataset corresponde a una sola ciudad y un período acotado (2006–2010), por lo que el modelo **no generaliza** a otros mercados ni épocas sin reentrenarse.
- No incorpora información de contexto (tasas de interés, inflación, ubicación geográfica precisa).
- **Próximos pasos:** `[probar CatBoost, stacking, exponer el modelo con una API (FastAPI), desplegar una demo]`.

---

## 9. Cómo reproducirlo

```bash
# 1. Clonar el repositorio
git clone https://github.com/[USUARIO]/house-prices.git
cd house-prices

# 2. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Descargar los datos desde Kaggle y colocarlos en data/
#    (train.csv y test.csv; no se incluyen en el repo por licencia)
```

Ejecutar los notebooks en este orden:

1. `notebooks/01_eda.ipynb`: análisis exploratorio
2. `notebooks/02_baseline.ipynb`: modelos de referencia
3. `notebooks/03_preprocesamiento_features.ipynb`: pipeline y feature engineering
4. `notebooks/04_modelos.ipynb`: comparación y optimización
5. `notebooks/05_ensamble_interpretacion.ipynb`: ensamble, SHAP y análisis de errores

### Estructura del repositorio

```
house-prices/
├── data/            # datos (no versionados)
├── notebooks/       # análisis paso a paso
├── src/             # funciones reutilizables (preprocesamiento, features)
├── requirements.txt
└── README.md
```

---

## 10. Tecnologías

Python · pandas · NumPy · scikit-learn · LightGBM · Optuna · SHAP · matplotlib · seaborn

---

## Autor

**[Tu nombre]** · [LinkedIn](https://linkedin.com/in/[usuario]) · [GitHub](https://github.com/[usuario])
