from preparacion_data import Xtrain,Xtest,Ytrain,Ytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

#Entrenamiento de Regresion Logistica
#Solver = Decidir que solucionador usar para usar el modelo
#max_iter = Numero de iteraciones para ajustar el modelo
print("\n\tREGRESIÓN LOGISTICA\n")
print("\n\tEntrenamiento Regresión Logistica\n")
regresion_logistica=LogisticRegression(solver='lbfgs',max_iter=1000)
regresion_logistica.fit(Xtrain,Ytrain)

#Pruebas o testeo de Regresion Logistica
Prediccion_regresion_logistica=regresion_logistica.predict(Xtest)
print("Puntuación del modelo con Regresión logistica: ",regresion_logistica.score(Xtest,Ytest)*100)
print("Exactitud de Regresión logistica: ",accuracy_score(Prediccion_regresion_logistica,Ytest))
print("Precisión de Regresión logistica: ",precision_score(Prediccion_regresion_logistica,Ytest))
print("Recordar de Regresión logistica: ",recall_score(Prediccion_regresion_logistica,Ytest))
print("F-Measure de Regresión logistica: ",f1_score(Prediccion_regresion_logistica,Ytest))

#Reporte
print("Reporte")
print(classification_report(Ytest, Prediccion_regresion_logistica, labels=[0, 1]))

#Grafica del modelo
plt.title("Regresión Logistica",
    fontdict={
        'family': 'serif',
        'color' : 'darkblue',
        'weight': 'bold',
        'size'  : 18
    })
Ytest_p=Ytest[:100]
Grafica=Prediccion_regresion_logistica[:100]
plt.xlabel('X(Time->)')
plt.ylabel('0 para anomalo 1 para normal')
plt.plot(Ytest_p,c='b',label="Data de prueba")
plt.plot(Grafica,c='r',label="Predicción del ataque")
plt.legend(loc='upper left')
plt.show()


