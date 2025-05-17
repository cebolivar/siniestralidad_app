import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import plt
import os

# Cargar los datos
csv_path = os.path.join(os.path.dirname(__file__), "data", "RIESGO OPERATIVO PARA MIEMBROS DE LA FUERZA PUBLICA.csv")
df = pd.read_csv(csv_path)

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

# Guardar modelo
model_path = os.path.join(os.path.dirname(__file__), "modelo.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Importancia de variables
importancias = model.feature_importances_
variables = X.columns

plt.figure(figsize=(8,4))
plt.barh(variables, importancias)
plt.xlabel("Importancia")
plt.title("Importancia de variables para el riesgo operativo")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "importancia_variables.png"))
plt.close()

# Gráfica: valores reales vs predichos
y_pred = model.predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, alpha=0.7)
plt.xlabel("Fallecidos reales")
plt.ylabel("Fallecidos predichos")
plt.title("Árbol de decisión: Reales vs Predichos")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "prediccion_arbol.png"))
plt.close()

print("Modelo, importancia de variables y gráfica de predicción guardados.")