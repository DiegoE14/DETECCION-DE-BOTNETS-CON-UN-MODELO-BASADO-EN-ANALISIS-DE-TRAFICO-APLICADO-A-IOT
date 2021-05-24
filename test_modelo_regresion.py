from preparacion_data import Xtest,Ytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,roc_curve,auc,plot_confusion_matrix
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt2
from matplotlib import pyplot as plt3
from joblib import load

regresion_logistica = load('modelo_regresion_logistica_entrenado.joblib') 

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
plt2.title('Curva ROC')
plt2.legend()
plt2.show()

#Matrix de confusión
plot_confusion_matrix(regresion_logistica, Xtest, Ytest)
plt3.title('Matriz de Confusión')
plt3.show()
