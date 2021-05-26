from sklearn import metrics
from preparacion_data import Xtest,Ytest,Xtrain,Ytrain
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,roc_curve,auc,plot_confusion_matrix
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt2
from matplotlib import pyplot as plt3
from joblib import load
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

regresion_logistica = load('modelo_regresion_logistica_entrenado.joblib') 

#Pruebas o testeo de Regresion Logistica
Prediccion_regresion_logistica=regresion_logistica.predict(Xtest)
score_pred = metrics.accuracy_score(Ytest, Prediccion_regresion_logistica)

#Validacion Cruzada
kf = KFold(n_splits=5)
score = regresion_logistica.score(Xtrain,Ytrain)
print("\nValidacion Cruzada\n")
print("Metrica del modelo", score)
print("Metrica en Test", score_pred)
scores = cross_val_score(regresion_logistica, Xtrain, Ytrain, cv=kf, scoring="accuracy")
print("Metricas cross_validation", scores)
print("Media de cross_validation", scores.mean())

#Variables
print("\nMetricas Calculadas\n")
print("Exactitud de Regresión logistica: ",accuracy_score(Prediccion_regresion_logistica,Ytest))
print("Precisión de Regresión logistica: ",precision_score(Prediccion_regresion_logistica,Ytest))
print("Recordar de Regresión logistica: ",recall_score(Prediccion_regresion_logistica,Ytest))
print("F-Measure de Regresión logistica: ",f1_score(Prediccion_regresion_logistica,Ytest))

#Reporte
print("\n\tReporte\n")
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

#Grafica CURVA ROC y AUC
logns_probs = [0 for _ in range(len(Ytest))]
loglr_probs = regresion_logistica.predict(Xtest)

logns_fpr,logns_tpr, umbral = roc_curve(Ytest,logns_probs)
auc_logns = auc (logns_fpr,logns_tpr)
loglr_fpr,loglr_tpr, umbral = roc_curve(Ytest,loglr_probs)
auc_log = auc (loglr_fpr,loglr_tpr)

plt2.figure(figsize=(5,5),dpi=100)
plt2.plot(logns_fpr,logns_tpr, linestyle ='--',label = 'Sin entrenar (auc = %0.2f)' %
auc_logns)
plt2.plot(loglr_fpr,loglr_tpr, marker ='.',label = 'Regreción Logistica (auc = %0.2f)' %
auc_log)
plt2.xlabel('Tasa de Falsos Positivos')
plt2.ylabel('Tasa de Verdaderos Positivos')
plt2.title('Curva ROC Regresión Logistica')
plt2.legend()
plt2.show()

#Matrix de confusión
plot_confusion_matrix(regresion_logistica, Xtest, Ytest)
plt3.title('Matriz de Confusión Regresión Logistica')
plt3.show()
