from tkinter import * #Carga módulo tk (widgets estándar)
from tkinter import ttk #Carga ttk (para widgets nuevos 8.5 o más)
from tkinter.filedialog import askopenfilename
from tkinter.font import BOLD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_confusion_matrix
from test_modelo import Prediccion_KNN,Ytest, KNN, Xtest
from matplotlib import pyplot as plt1
from matplotlib import pyplot as plt2

interfaz = Tk()

ancho_ventana = 580
alto_ventana = 200
x_ventana = interfaz.winfo_screenwidth() // 2 - ancho_ventana // 2
y_ventana = interfaz.winfo_screenheight() // 2 - alto_ventana // 2

def matriz():
    plot_confusion_matrix(KNN, Xtest, Ytest)
    plt1.title('Matriz de Confusión')
    plt1.show()

def grafica():
    plt2.title("Modelo",
    fontdict={
        'family': 'serif',
        'color' : 'darkblue',
        'weight': 'bold',
        'size'  : 18
    })
    Ytest_p=Ytest[:100]
    Grafica=Prediccion_KNN[:100]
    plt2.xlabel('X(Time->)')
    plt2.ylabel('0 para anomalo 1 para normal')
    plt2.plot(Ytest_p,c='b',label="Data de prueba")
    plt2.plot(Grafica,c='r',label="Predicción del ataque")
    plt2.legend(loc='upper left')
    plt2.show()

posicion = str(ancho_ventana) + "x" + str(alto_ventana) + "+" + str(x_ventana) + "+" + str(y_ventana)
interfaz.geometry(posicion)
interfaz.resizable(width=False, height=False)
interfaz.configure(bg= '#204056')
interfaz.title('DETECCION DE BOTNETS CON UN MODELO BASADO EN ANALISIS DE TRAFICO APLICADO A IOT')

frame = Frame(interfaz, width=300)
frame.configure(bg= '#204056')

titulo = Label(frame, text='Estadisticas del modelo')
titulo.grid(row=0, column=0, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))
titulo.configure(bg= '#204056', fg= 'white', font=('Verdana', 12, BOLD))

Exactitud = Label(frame, text='Exactitud: ' + str(accuracy_score(Prediccion_KNN,Ytest)))
Exactitud.grid(row=1, column=0, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))
Exactitud.configure(bg= '#204056', fg= 'white', font=('Verdana', 12))

Precisión = Label(frame, text='Precisión: ' + str(precision_score(Prediccion_KNN,Ytest)))
Precisión.grid(row=1, column=1, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))
Precisión.configure(bg= '#204056', fg= 'white', font=('Verdana', 12))

Recordar = Label(frame, text='Recordar: ' + str(recall_score(Prediccion_KNN,Ytest)))
Recordar.grid(row=2, column=0, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))
Recordar.configure(bg= '#204056', fg= 'white', font=('Verdana', 12))

F_Measure = Label(frame, text='F-Measure: ' + str(f1_score(Prediccion_KNN,Ytest)))
F_Measure.grid(row=2, column=1, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))
F_Measure.configure(bg= '#204056', fg= 'white', font=('Verdana', 12))

Matriz = Button(frame, text='Matriz de confusión', command=matriz)
Matriz.grid(row=3, column=0, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))

Grafica = Button(frame, text='Grafica del modelo', command=grafica)
Grafica.grid(row=3, column=1, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))


frame.grid()

interfaz.mainloop()