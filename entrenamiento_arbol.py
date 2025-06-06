import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Definir rutas
base_dir = os.path.dirname(__file__)
static_dir = os.path.abspath(os.path.join(base_dir, "..", "static"))
data_path = os.path.join(base_dir, "..", "data", "SECTORES_CRITICOS_DE_SINIESTRALIDAD_VIAL_20250514.csv")
modelo_path = os.path.join(base_dir, "..", "modelo_ml", "modelo.pkl")

# Crear carpeta static si no existe
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Cargar datos
df = pd.read_csv(data_path)

# Selección y codificación de variables
X = df[["GiZScore", "GiPValue", "Latitud", "Longitud"]].copy()
le_tramo = LabelEncoder()
X["Tramo"] = le_tramo.fit_transform(df["Tramo"].astype(str))
le_mpio = LabelEncoder()
X["Municipio"] = le_mpio.fit_transform(df["Municipio"].astype(str))
y = df["Fallecidos"]

# Entrenar modelo
model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X, y)

# Guardar modelo y labelencoders
with open(modelo_path, "wb") as f:
    pickle.dump({"model": model, "le_tramo": le_tramo, "le_mpio": le_mpio}, f)

# Importancia de variables
importancias = model.feature_importances_
variables = X.columns

plt.figure(figsize=(8,4))
plt.barh(variables, importancias)
plt.xlabel("Importancia")
plt.title("Importancia de variables para el riesgo operativo")
plt.tight_layout()
plt.savefig(os.path.join(static_dir, "importancia_variables.png"))
plt.close()

# Predicciones sobre los datos de entrenamiento
y_pred = model.predict(X)

# Gráfica: valores reales vs predichos
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, alpha=0.7)
plt.xlabel("Fallecidos reales")
plt.ylabel("Fallecidos predichos")
plt.title("Árbol de decisión: Reales vs Predichos")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.tight_layout()
plt.savefig(os.path.join(static_dir, "prediccion_arbol.png"))
plt.close()

print("Modelo y gráficas guardados correctamente en la carpeta static.")