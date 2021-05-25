from preparacion_data import Xtrain,Ytrain
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

#Entrenamiento de vecino mas cercano
print("\n\tVECINO MAS CERCANO\n")
print("\n\tEntrenamiento Vecino mas Cercano\n")

kf = KFold(n_splits=5)

KNN=KNeighborsClassifier()
KNN.fit(Xtrain,Ytrain)

score = KNN.score(Xtrain,Ytrain)
dump(KNN,'modelo_vecino_mas_cercano_entrenado.joblib')

print("\n\tEntrenamiento Regresión Vecino mas cercano\n")
print("Metrica del modelo", score)
scores = cross_val_score(KNN, Xtrain, Ytrain, cv=kf, scoring="accuracy")
print("Metricas cross_validation", scores)
print("Media de cross_validation", scores.mean())