from preparacion_data import Xtrain,Xtest,Ytrain,Ytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#Entrenamiento de vecino mas cercano
print("\n\tVECINO MAS CERCANO\n")
print("\n\tEntrenamiento Vecino mas Cercano\n")
KNN=KNeighborsClassifier()
KNN.fit(Xtrain,Ytrain)

#Pruebas o testeo de Naive Bayes
Prediccion_KNN=KNN.predict(Xtest)
print("Puntuación del modelo con Vecino mas Cercano: ",KNN.score(Xtest,Ytest)*100)
print("Exactitud de Vecino mas Cercano: ",accuracy_score(Prediccion_KNN,Ytest))
print("Precisión de Vecino mas Cercano: ",precision_score(Prediccion_KNN,Ytest))
print("Recordar de Vecino mas Cercano: ",recall_score(Prediccion_KNN,Ytest))
print("F-Measure de Vecino mas Cercano: ",f1_score(Prediccion_KNN,Ytest))

#Grafica del modelo
plt.title("Vecino mas Cercano",
    fontdict={
        'family': 'serif',
        'color' : 'darkblue',
        'weight': 'bold',
        'size'  : 18
    })
Ytest_p=Ytest[:100]
Grafica=Prediccion_KNN[:100]
plt.xlabel('X(Time->)')
plt.ylabel('0 para anomalo 1 para normal')
plt.plot(Ytest_p,c='b',label="Data de prueba")
plt.plot(Grafica,c='r',label="Predicción del ataque")
plt.legend(loc='upper left')
plt.show()

