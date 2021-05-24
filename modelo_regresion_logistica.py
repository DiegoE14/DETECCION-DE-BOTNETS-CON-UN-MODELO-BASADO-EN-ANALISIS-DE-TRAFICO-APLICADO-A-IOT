from preparacion_data import Xtrain,Ytrain
from sklearn.linear_model import LogisticRegression
from joblib import dump

#Entrenamiento de Regresion Logistica
#Solver = Decidir que solucionador usar para usar el modelo
#max_iter = Numero de iteraciones para ajustar el modelo
print("\n\tREGRESIÓN LOGISTICA\n")
print("\n\tEntrenamiento Regresión Logistica\n")
regresion_logistica=LogisticRegression(solver='lbfgs',max_iter=2000)
regresion_logistica.fit(Xtrain,Ytrain)
dump(regresion_logistica,'modelo_regresion_logistica_entrenado.joblib')
print("\n\tEntrenamiento Regresión Logistica completado\n")