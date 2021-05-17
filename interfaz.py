from tkinter import * #Carga módulo tk (widgets estándar)
from tkinter import ttk #Carga ttk (para widgets nuevos 8.5 o más)
from tkinter.filedialog import askopenfilename

interfaz = Tk()

ancho_ventana = 450
alto_ventana = 200
x_ventana = interfaz.winfo_screenwidth() // 2 - ancho_ventana // 2
y_ventana = interfaz.winfo_screenheight() // 2 - alto_ventana // 2

posicion = str(ancho_ventana) + "x" + str(alto_ventana) + "+" + str(x_ventana) + "+" + str(y_ventana)
interfaz.geometry(posicion)
interfaz.resizable(width=False, height=False)
interfaz.configure(bg= '#204056')
interfaz.title('DETECCION DE BOTNETS CON UN MODELO BASADO EN ANALISIS DE TRAFICO APLICADO A IOT')

frame = Frame(interfaz, width=300)
frame.configure(bg= '#204056')

label = Label(frame, text='Cargar Dataset (.csv): ')
label.grid(row=0, column=0, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))
label.configure(bg= '#204056', fg= 'white', font=('Verdana', 12))

v = StringVar(frame, value='.csv')
entry = Entry(frame, textvariable = v)
entry.grid(row=0, column=1, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))

button = Button(frame, text='Browse')
button.bind('<1>', lambda e: v.set(askopenfilename().split('/')[-1]))
button.grid(row=0, column=2, rowspan=1, columnspan=1, padx = 10, pady=10)

label2 = Label(frame, text='Cargar Dataset (.csv): ')
label2.grid(row=1, column=0, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))
label2.configure(bg= '#204056', fg= 'white', font=('Verdana', 12))

v2 = StringVar(frame, value='.csv')
entry2 = Entry(frame, textvariable = v)
entry2.grid(row=1, column=1, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))

button2 = Button(frame, text='Browse')
button2.bind('<1>', lambda e: v.set(askopenfilename().split('/')[-1]))
button2.grid(row=1, column=2, rowspan=1, columnspan=1, padx = 10, pady=10)


machineLabel = Label(frame, text='Algoritmo: ')
machineLabel.grid(row=2, column=0, padx=10, pady=10, sticky=(W, ))
machineLabel.configure(bg= '#204056', fg= 'white', font=('Verdana', 12))

combo = ttk.Combobox(frame)
combo['values'] = sorted(['Naive Bayes', 'Regresión Logisitcia', 'Vecino mas Cercano'])
combo.grid(row=2, column=1, padx=10, pady=10)

v2 = StringVar(frame, value='Accuracy: ')
resultLabel = Label(frame, textvariable=v2)

calButton = Button(frame, text='Go')
calButton.bind('<1>', lambda e: callSuitable(combo.get(), v2))
calButton.grid(row=2, column=2, sticky = (E, W), padx=10, pady=10)


resultLabel.grid(row=3, pady=10, padx=10, columnspan=3, sticky=(W, ))

frame.grid()

interfaz.mainloop()