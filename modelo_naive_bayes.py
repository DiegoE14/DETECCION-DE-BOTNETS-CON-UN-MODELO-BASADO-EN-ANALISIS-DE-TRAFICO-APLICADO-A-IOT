from preparacion_data import Xtrain,Xtest,Ytrain,Ytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_confusion_matrix
from matplotlib import pyplot as plt1
from matplotlib import pyplot as plt2
from matplotlib import pyplot as plt3
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

#Reporte
print("Reporte")
print(classification_report(Ytest, Prediccion_naive_bayes, labels=[0, 1]))

#Grafica curvas roc
ns_probs = [0 for _ in range(len(Ytest))]
lr_probs = naive_bayes.predict_proba(Xtest)
lr_probs = lr_probs[:, 1]
ns_fpr, ns_tpr, _ = roc_curve(Ytest, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(Ytest, lr_probs)
plt1.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenar')
plt1.plot(lr_fpr, lr_tpr, marker='.', label='Naive Bayes')
plt1.xlabel('Tasa de Falsos Positivos')
plt1.ylabel('Tasa de Verdaderos Positivos')
plt1.legend()
plt1.show()

#Grafica matriz de confusión
plot_confusion_matrix(naive_bayes, Xtest, Ytest)  
plt2.show() 

#Grafica del modelo
plt3.title("Naive Bayes",
    fontdict={
        'family': 'serif',
        'color' : 'darkblue',
        'weight': 'bold',
        'size'  : 18
    })
Ytest_p=Ytest[:100]
Grafica=Prediccion_naive_bayes[:100]
plt3.xlabel('X(Time->)')
plt3.ylabel('0 para anomalo 1 para normal')
plt3.plot(Ytest_p,c='b',label="Data de prueba")
plt3.plot(Grafica,c='r',label="Predicción del ataque")
plt3.legend(loc='upper left')
plt3.show()
