from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

# Cargar el modelo
modelo = tf.keras.models.load_model("C:\\UTN\\Desarrollo de Software\\Parcial\\Parcial_IA\\X-MEN.h5")

app = Flask(__name__)
CORS(app)

def convertirADN(base):
    """Convierte una base de ADN a un número entero."""
    conversion = {'A': 2, 'T': 3, 'C': 4, 'G': 5}
    return conversion.get(base, 0)  # 0 si la base no es válida

def ajustar_matriz(matriz, tamaño_fijo=10):
    """Ajusta la matriz de ADN al tamaño fijo de 10x10 usando padding."""
    secuencia_numerica = []
    
    for fila in matriz:
        # Convertir cada base de ADN a su valor numérico y aplicar padding en la fila si es necesario
        secuencia_numerica_fila = [convertirADN(base) for base in fila]
        secuencia_numerica_fila += [0] * (tamaño_fijo - len(secuencia_numerica_fila))
        secuencia_numerica.append(secuencia_numerica_fila)

    # Aplicar padding en las filas para que la matriz sea de tamaño 10x10
    while len(secuencia_numerica) < tamaño_fijo:
        secuencia_numerica.append([0] * tamaño_fijo)

    return secuencia_numerica

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Asegúrate de que 'data' tiene la estructura esperada
    print("Datos recibidos:", data)

    # Procesar la secuencia de ADN
    matriz = data[0]['secuencia']  # Accede a la primera secuencia en la lista
    
    # Ajustar la matriz al tamaño fijo de 10x10
    secuencia_numerica = ajustar_matriz(matriz)
    secuencia_numerica = np.array([secuencia_numerica])  # Añade la dimensión del batch

    print("Secuencia numérica ajustada para predicción:", secuencia_numerica)

    # Realizar la predicción
    prediccion = modelo.predict(secuencia_numerica)
    print("Predicción cruda del modelo:", prediccion)
    
    # Determinar si es mutante o no basado en el umbral de 0.5
    etiqueta_actualizada = int((prediccion > 0.5).astype("int32")[0][0])
    es_mutante = "Mutante" if etiqueta_actualizada == 1 else "No mutante"

    # Imprimir si la secuencia es mutante o no
    print(f"Resultado de la predicción: {es_mutante}")

    # Devolver la respuesta como JSON, actualizando la etiqueta y el resultado en texto
    return jsonify([{
        'secuencia': matriz,
        'etiqueta': etiqueta_actualizada,
    }])

if __name__ == '__main__':
    app.run(port=5000)
