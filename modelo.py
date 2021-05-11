from preparacion_data import Xtrain,Xtest,Ytrain,Ytest
from sklearn.metrics import accuracy_score
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
prediction_regresion_logistica=regresion_logistica.predict(Xtest)
print(regresion_logistica.score(Xtest,Ytest)*100)
print("Accuracy de Regresión logistica: ",accuracy_score(prediction_regresion_logistica,Ytest))

Ytest_p=Ytest[:100]
prediction1_p=prediction_regresion_logistica[:100]
plt.xlabel('X(Time->)')
plt.ylabel('0 para anomalo 1 para normal')
plt.plot(Ytest_p,c='b',label="Test Data")
plt.plot(prediction1_p,c='r',label="Predicted Attack")
plt.legend(loc='upper left')
plt.show()

