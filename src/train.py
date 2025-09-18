#Entrenamos un modelo simple (LOGISTIC REGRESSION)
#con el dataset de IRIS
# Este archivo se usa cuando quiero entrenar el modelo con datos conocidos


#Librerias que voy a usar


from sklearn.datasets import load_iris #dataset de ejemplo (flores iris)
import pandas as pd #Manejo de tablas de datos (como excel)
from sklearn.model_selection import train_test_split #divdir datos en entrenamiento y prueba
from sklearn.linear_model import LogisticRegression # modelo que vamos a usar (clasificacion)
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
#metricas para saber que tan bueno es el modelo


import joblib # guardar/ cargar el modelo entrenado


# archivo donde guardamos cosas compartidas (columnas, ruta del modelo)

from common import FEATURE_COLUMNS, MODEL_PATH

def main():


    #Primero 
    #cargo el dataset de IRIS (viene dentro de sklearn, no necesito CSV)

    iris = load_iris() #cargo el dataset

    #segungo
    #convierto el dataset a un DataFrame de pandas (tabla de datos)
    # es mas facil trabajar con tablas de pandas que con arrays de numpy

    df = pd.DataFrame(data= iris.data, columns= FEATURE_COLUMNS) #creo la tabla con los datos y nombres de columnas
    df["species"] = iris.target #agrego la columna de especies (lo que quiero predecir)

    #Tercero
    #separo los datos en X (entradas) y y (salidas)
    #X son las columnas que el modelo usara para aprender
    #y es la columna que el modelo tiene que predecir

    X = df[FEATURE_COLUMNS] #todas las columnas menos species
    y = df["species"] #solo la columna species

    # Cuarto
    # divido los datos en entrenamiento y prueba
    # 80% para entrenar el modelo
    # 20% para probar el modelo

    # train_test_split → siempre que hagas machine learning vas a dividir los datos
    #train_test_split divide los datos en 4 partes
    # stratify=y → mantiene las proporciones de cada clase (muy recomendable en clasificación)


    """
    
    X_train = preguntas para estudiar.
    y_train = respuestas de esas preguntas.
    X_test = preguntas nuevas en el examen.
    y_test = respuestas correctas del examen.
    y_pred = lo que respondió el modelo en el examen.
    
    
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 , stratify=y)

    # Quinto
    # creo el modelo
    # Si quiero probar otros modelos, este es el lugar

    model = LogisticRegression(max_iter=200) #creo el modelo (puedo cambiar parametros si quiero)

    # Sexto
    # entreno el modelo con los datos de entrenamiento

    model.fit(X_train, y_train) #entreno el modelo

    # Septimo
    # pruebo el modelo con los datos de prueba

    # .predict(X_test) → el modelo intenta adivinar las especies de las flores de prueba
    # y_pred = lo que el modelo contestó
    # luego comparamos con y_test (las respuestas reales)

    y_pred = model.predict(X_test) #el modelo predice las especies de las flores de prueba

    # acuracy_score → porcentaje de aciertos del modelo
    #es bueno cuando las clases estan balanceadas (igual cantidad de cada clase)
    # mal si no es 50/50/50


    print("Accuracy:", accuracy_score(y_test, y_pred)) #imprimo el porcentaje de aciertos

    # classification_report → reporte detallado de precision, recall, f1-score
    print("\nReporte por clase:\n", classification_report(y_test, y_pred , target_names= iris.target_names)) #imprimo el reporte detallado

    #tabla de aciertos y errores
    #lo usamos para ver en que se equivoca el modelo
    #si confusion_matrix es grande (muchas clases) es dificil de interpretar

    # confusion_matrix → matriz de confusión
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))


    # Octavo
    # guardo el modelo entrenado en un archivo para usarlo despues

    bundle ={"model":model , "target_names": iris.target_names}
    joblib.dump(bundle, MODEL_PATH) #guardo el modelo en un archivo
    print(f"Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    main()