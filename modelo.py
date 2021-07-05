from preparacion_data import Xtrain,Ytrain
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

#Entrenamiento del modelo
print("\n\tEntrenamiento del modelo\n")
KNN=KNeighborsClassifier()
KNN.fit(Xtrain,Ytrain)
dump(KNN,'modelo_entrenado.joblib')
print("\n\tEntrenamiento completado\n")

