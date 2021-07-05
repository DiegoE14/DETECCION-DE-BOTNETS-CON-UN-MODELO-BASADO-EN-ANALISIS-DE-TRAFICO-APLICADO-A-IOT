# DETECCION-DE-BOTNETS-CON-UN-MODELO-BASADO-EN-ANALISIS-DE-TRAFICO-APLICADO-A-IOT
Proyecto de grado para obtener titulo de ingeniero de sistemas de la universidad Cesmag - Colombia

Autores: 
Diego Giovanny Enriquez Guevara - diegogeg21@gmail.com
Oscar Alexander Rodriguez Insuasty - rodriguezinsuastyoscar@gmail.com

Versionado: 
- Python 3.9.2
- pandas 1.2.4
- numpy 1.19.5
- matplotlib 3.4.2
- scikit-learn 0.24.2
- joblib 1.0.1

Estructura del proyecto: 
- Data (contiene datasets a trabajar)
    - botnet.csv (datos de botnets Mirai y BASHLITE)
    - trafico_iot.csv (datos del trafico generado en IoT)
- interfaz.py (Interfaz sencilla que funciona como dashboard)
- modelo_entrenado.py (Se guarda el entrenamiento del modelo)
- modelo.py (Se realiza entrenamiento y pruebas)
- preparacion_data.py (Preparación del modelo e importación de los datos)
- test_modelo.py (Se realizan pruebas y graficas para ver su eficacia)

Algoritmo de aprendizaje supervisado (Machine Learning) utilizado:
- Naive Bayes