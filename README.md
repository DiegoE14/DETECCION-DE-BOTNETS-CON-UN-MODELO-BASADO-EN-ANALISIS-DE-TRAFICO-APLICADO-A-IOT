# DETECCION-DE-BOTNETS-CON-UN-MODELO-BASADO-EN-ANALISIS-DE-TRAFICO-APLICADO-A-IOT
Proyecto de grado para obtener titulo de ingeniero de sistemas de la universidad Cesmag - Colombia

Propetarios: 
Diego Giovanny Enriquez Guevara - diegogeg21@gmail.com
Oscar Alexander Rodriguez Insuasty - rodriguezinsuastyoscar@gmail.com

Versionado: 
- Python 3.9.2
- pandas 1.2.4
- numpy 1.19.5
- matplotlib 3.4.2
- scikit-learn 0.24.2

Estructura del proyecto: 
- Carpeta Data (contiene datasets a trabajar)
    - botnet.csv (datos de botnets Mirai y BASHLITE)
    - trafico_iot.csv (datos del trafico generado en IoT)
- interfaz.py (La interfaz donde el usuario carga y visualiza el modelo)
- modelo.py (Se cargan todos los algoritmos a tratar para el modelo)
- preparacion_data.py (Preparación del modelo e importación de los datos)

Algoritmos de aprendizaje supervisado (Machine Learning)
- Regresión Logistica
- Preceptor
- Arboles de desicion
- Naive Bayes
- KNN Classifier