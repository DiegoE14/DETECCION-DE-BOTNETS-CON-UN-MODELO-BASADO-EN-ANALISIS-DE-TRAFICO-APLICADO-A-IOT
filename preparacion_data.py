import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#Se carga Data 
print("--------- Importando Data... ---------\n")
trafico=pd.read_csv("Data/trafico_iot.csv")
botnet=pd.read_csv("Data/botnet.csv")
print("--------- ¡¡Importanción exitosa!! ---------\n")

# Visualización de los datos
print("--------- Data de trafico IoT ---------\n")
print(trafico)
print("--------- Data de Botnets ---------\n")
print(botnet)

#Visualización de las dimensiones de cada matriz (filas y columnas)
print("\nDimensiones de la data trafico IoT: ",trafico.shape)
print("Dimensiones de la data botnet :",botnet.shape)

#Se agrega columna 'clasificacion' para etiquetar como anomalo (0) o normal (1):
#trafico de IoT (trafico benigno) = 1
#botnets = 0
trafico['clasificacion']=1
botnet['clasificacion']=0

#Se concatena los registros de los dos datasets (trafico y botnet) en uno solo ('dataset')
#por filas ('axis=0')
dataset=pd.concat([trafico,botnet],axis=0)
print("--------- Data dataset ---------\n")
print(trafico)
print("\nDimensiones de la data dataset: ",dataset.shape)

#Barajamos la data de dataset
dataset=shuffle(dataset)

#Se extrae la columna 'claisifcacion' de la data de dataset
#y se la guarda en la variable 'clasificacion'
clasificacion=dataset['clasificacion']

#Se borra la columna clasificacion de la data del 'dataset'
#fila por fila (axis=1)
dataset=dataset.drop(['clasificacion'],axis=1)

#Se elimina las columnas apartir de la columna 29 en adelante debido a que
#mostró valores atípicos bastante obvios y tipos de datos sobreajustados.
dataset_final=dataset.iloc[:,:28]
print("\nDimensiones de dataset1 : ",dataset_final.shape)

#Se concatena la variable clasificacion en una sola dimensión
clasificacion=np.array(clasificacion).flatten()
print("\nDimensiones de 'clasificacion' : ",clasificacion.shape)

#Entrenamiento y pruebas del dataset
#test_size = porcentaje de los datos tomados para las pruebas o testeo
#random_state = Se evita la aleatoridad en el conjunto de datos 
Xtrain,Xtest,Ytrain,Ytest=train_test_split(dataset_final,clasificacion,test_size=0.2,random_state=993)

print("\nPreparacion de la data completada")