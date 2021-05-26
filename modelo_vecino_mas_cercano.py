from preparacion_data import Xtrain,Ytrain
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

#Entrenamiento de vecino mas cercano
print("\n\tVECINO MAS CERCANO\n")
print("\n\tEntrenamiento Vecino mas Cercano\n")
KNN=KNeighborsClassifier()
KNN.fit(Xtrain,Ytrain)
dump(KNN,'modelo_vecino_mas_cercano_entrenado.joblib')
print("\n\tEntrenamiento vecino mas cercano completado\n")

