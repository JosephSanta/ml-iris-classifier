#Esto es una la libreta con datos compartidos 
#Si algo es unico de un archivo lo ponemos en ese archivo
#si algo se repite mas de 1 archivo , y casi nunca cambia lo ponemos aqui


#columnas (x): el orden importa 
#El modelo aprende con columnas en cierto orden

FEATURE_COLUMNS = ["sepal_length (cm)", "sepal_width (cm)", "petal_length (cm)", "petal_width (cm)"]

#Archivo donde guardaremos el modelo entrenado

MODEL_PATH = "models/model.joblib"
