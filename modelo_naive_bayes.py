from preparacion_data import Xtrain,Xtest,Ytrain,Ytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB


#Entrenamiento de Naive Bayes
print("\n\tNAIVE BAYES\n")
print("\n\tEntrenamiento Naive Bayes\n")
naive_bayes=GaussianNB()
naive_bayes.fit(Xtrain,Ytrain)

#Pruebas o testeo de Naive Bayes
Prediccion_naive_bayes=naive_bayes.predict(Xtest)
print("Puntuación del modelo con Naive Bayes: ",naive_bayes.score(Xtest,Ytest)*100)
print("Exactitud de Naive Bayes: ",accuracy_score(Prediccion_naive_bayes,Ytest))
print("Precisión de Naive Bayes: ",precision_score(Prediccion_naive_bayes,Ytest))
print("Recordar de Naive Bayes: ",recall_score(Prediccion_naive_bayes,Ytest))
print("F-Measure de Naive Bayes: ",f1_score(Prediccion_naive_bayes,Ytest))

#Grafica del modelo
plt.title("Naive Bayes",
    fontdict={
        'family': 'serif',
        'color' : 'darkblue',
        'weight': 'bold',
        'size'  : 18
    })
Ytest_p=Ytest[:100]
Grafica=Prediccion_naive_bayes[:100]
plt.xlabel('X(Time->)')
plt.ylabel('0 para anomalo 1 para normal')
plt.plot(Ytest_p,c='b',label="Data de prueba")
plt.plot(Grafica,c='r',label="Predicción del ataque")
plt.legend(loc='upper left')
plt.show()

