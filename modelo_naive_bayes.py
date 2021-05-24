from preparacion_data import Xtrain,Ytrain
from sklearn.naive_bayes import GaussianNB
from joblib import dump

#Entrenamiento de Naive Bayes
print("\n\tNAIVE BAYES\n")
print("\n\tEntrenamiento Naive Bayes\n")
naive_bayes=GaussianNB()
naive_bayes.fit(Xtrain,Ytrain)
dump(naive_bayes,'modelo_naive_bayes_entrenado.joblib')
print("\n\tEntrenamiento Naive Bayes completado\n")