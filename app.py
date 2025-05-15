from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = "tu_clave_secreta"  # Cambia esto por una clave secreta segura en producción

# Cargar datos
data_path = os.path.join(os.path.dirname(__file__), "data", "SECTORES_CRITICOS_DE_SINIESTRALIDAD_VIAL_20250514.csv")
df = pd.read_csv(data_path)

# Cargar modelo
modelo_path = os.path.join(os.path.dirname(__file__), "modelo_ml", "modelo.pkl")
with open(modelo_path, "rb") as f:
    model = pickle.load(f)

# Entrenar LabelEncoders desde los datos
le_tramo = LabelEncoder()
le_tramo.fit(df["Tramo"].astype(str))
le_mpio = LabelEncoder()
le_mpio.fit(df["Municipio"].astype(str))

@app.route("/")
def index():
    columns = df.columns
    data = df.head(50).to_dict(orient="records")  # Muestra las primeras 50 filas
    return render_template("index.html", data=data, columns=columns)

@app.route("/seleccionar", methods=["POST"])
def seleccionar():
    fila = {
        "score": request.form["score"],
        "pvalue": request.form["pvalue"],
        "latitud": request.form["latitud"],
        "longitud": request.form["longitud"],
        "tramo": request.form["tramo"],
        "municipio": request.form["municipio"]
    }
    seleccionados = session.get("seleccionados", [])
    seleccionados.append(fila)
    session["seleccionados"] = seleccionados
    return redirect(url_for("index"))

@app.route("/prediccion", methods=["GET", "POST"])
def prediccion():
    if request.method == "POST":
        try:
            score = float(request.form["score"])
            pvalue = float(request.form["pvalue"])
            latitud = float(request.form["latitud"])
            longitud = float(request.form["longitud"])
            tramo = request.form["tramo"]
            municipio = request.form["municipio"]
            tramo_encoded = le_tramo.transform([tramo])[0]
            municipio_encoded = le_mpio.transform([municipio])[0]
            features = [score, pvalue, latitud, longitud, tramo_encoded, municipio_encoded]
            pred = model.predict([features])[0]
            # Puedes guardar el resultado en session si lo necesitas en /analisis
            session["prediccion_result"] = pred
            return redirect(url_for("analisis"))
        except Exception:
            return render_template("resultado.html", pred=None, error="Por favor, completa todos los campos correctamente.")
    # Si es GET, rellena el formulario con los datos recibidos
    return render_template(
        "prediccion.html",
        score=request.args.get("score", ""),
        pvalue=request.args.get("pvalue", ""),
        latitud=request.args.get("latitud", ""),
        longitud=request.args.get("longitud", ""),
        tramo=request.args.get("tramo", ""),
        municipio=request.args.get("municipio", "")
    )

@app.route("/resultado")
def resultado():
    return render_template("resultado.html", pred=None)

@app.route("/seleccionados")
def seleccionados():
    seleccionados = session.get("seleccionados", [])
    return render_template("seleccionados.html", seleccionados=seleccionados)

@app.route("/eliminar_seleccionado/<int:idx>", methods=["POST"])
def eliminar_seleccionado(idx):
    seleccionados = session.get("seleccionados", [])
    if 0 <= idx < len(seleccionados):
        seleccionados.pop(idx)
        session["seleccionados"] = seleccionados
    return redirect(url_for("seleccionados"))

@app.route("/analisis")
def analisis():
    return render_template("analisis.html")