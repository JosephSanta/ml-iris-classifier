# usar el modelo guardado para predecir una flor nueva 

import sys # para leer argumentos de linea de comandos
import pandas as pd # manejo de tablas de datos
import joblib # cargar el modelo entrenado

# importo lo que necesito de common.py
from common import FEATURE_COLUMNS, MODEL_PATH

def main():

    # leo los argumentos de linea de comandos
    # el primer argumento es el nombre del script (predict.py)
    # los siguientes argumentos son las características de la flor a predecir

    args = sys.argv[1:] # leo todos los argumentos menos el primero

    if len(args) != len(FEATURE_COLUMNS):

        print(f"Uso: python {sys.argv[0]} " + " ".join([f"<{col}>" for col in FEATURE_COLUMNS]))
        print(f"Ejemplo: python {sys.argv[0]} 5.1 3.5 1.4 0.2")
        sys.exit(1) #Sal del programa con error


    #los datos vienen como strings (texto)
    # convierto los argumentos a float
    try:
        features = [float(arg) for arg in args]
    except ValueError:
        print("Error: todas las características deben ser números.")
        sys.exit(1)

    # creo un DataFrame con las características
    input_data = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    # cargo el modelo entrenado
    loaded = joblib.load(MODEL_PATH)
    model = loaded["model"]
    target_names = loaded["target_names"]
    


    # hago la predicción
    prediction = model.predict(input_data)

    # muestro el resultado
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

    
    predicted_species = species_map.get(prediction[0], "desconocida")

    print(f"La especie predicha es: {predicted_species}")

if __name__ == "__main__":
    main()
