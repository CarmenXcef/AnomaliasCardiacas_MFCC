import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

# Funciones para clasificar utilizando: Regresión logística, SVM y Random Forest.
def rLog(train, valid):
    inicio = train.find('_') + 1
    fin = train.rfind('.')
    ventana = train[inicio:fin]
    print("--------------------REGRESIÓN LOGÍSTICA--------------------")
    print("ventanas de: ", ventana)

    # Datos de entrenamiento y prueba, omitiendo la columna que contiene el nombre del audio,
    # esto se utiliza únicamente con los datasets de los MFCC:
    # datos_train = pd.read_csv(train, usecols=lambda col: col != 'audio')
    # datos_test = pd.read_csv(valid, usecols=lambda col: col != 'audio')

    # Si el dataset es de diferencias, se utiliza de la siguiente manera:
    datos_train = pd.read_csv(train)
    datos_test = pd.read_csv(valid)

    # Variables de entrada y salida, entrenamiento
    X_train = datos_train.iloc[:, :-1]
    y_train = datos_train.iloc[:, -1]
    # Variables de entrada y salida, prueba
    X_test = datos_test.iloc[:, :-1]
    y_test = datos_test.iloc[:, -1]

    # Regresión logística
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, y_train)
    #Predicción
    y_pred = lr.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_txt = 'Accuracy: %.3f' % accuracy
    print(accuracy_txt)

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, zero_division=1)
    report_txt = "Reporte de clasificación:\n" + report
    print(report_txt)

    nombre_archivo = 'resultados_' + ventana + '_rLog.txt'
    # Guardar en un archivo de texto
    with open(nombre_archivo, 'w') as file:
        file.write("--------------------REGRESIÓN LOGÍSTICA--------------------\n")
        file.write("ventanas de: " + ventana + '\n')
        file.write(accuracy_txt + '\n')
        file.write(report_txt)

def supvm(train, valid):
    inicio = train.find('_') + 1
    fin = train.rfind('.')
    ventana = train[inicio:fin]
    print("---------------------------SVM---------------------------")
    print("ventanas de: ", ventana)
    # Datos de entrenamiento y prueba, omitiendo la columna que contiene el nombre del audio,
    # esto se utiliza únicamente con los datasets de los MFCC:
    # datos_train = pd.read_csv(train, usecols=lambda col: col != 'audio')
    # datos_test = pd.read_csv(valid, usecols=lambda col: col != 'audio')

    # Si el dataset es de diferencias, se utiliza de la siguiente manera:
    datos_train = pd.read_csv(train)
    datos_test = pd.read_csv(valid)

    # Variables de entrada y salida, entrenamiento
    X_train = datos_train.iloc[:, :-1]
    y_train = datos_train.iloc[:, -1]

    # Variables de entrada y salida, prueba
    X_test = datos_test.iloc[:, :-1]
    y_test = datos_test.iloc[:, -1]

    # SVM
    clf = svm.SVC(kernel='linear')
    # Entrenar el clasificador SVM
    clf.fit(X_train, y_train)

    # Predicción con los datos de prueba
    y_pred = clf.predict(X_test)

    # Precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_txt = 'Accuracy: %.3f' % accuracy
    print(accuracy_txt)

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, zero_division=1)
    report_txt = "Reporte de clasificación:\n" + report
    print(report_txt)

    nombre_archivo = 'resultados_' + ventana + '_svm.txt'
    # Guardar en un archivo de texto
    with open(nombre_archivo, 'w') as file:
        file.write("---------------------------SVM---------------------------\n")
        file.write("ventanas de: "+ ventana+ '\n')
        file.write(accuracy_txt + '\n')
        file.write(report_txt)

def randomF(train, valid):
    inicio = train.find('_') + 1
    fin = train.rfind('.')
    ventana = train[inicio:fin]
    print("------------------------RANDOM FOREST------------------------")
    print("ventanas de: ", ventana)
    # Datos de entrenamiento y prueba, omitiendo la columna que contiene el nombre del audio,
    # esto se utiliza únicamente con los datasets de los MFCC:
    # datos_train = pd.read_csv(train, usecols=lambda col: col != 'audio')
    # datos_test = pd.read_csv(valid, usecols=lambda col: col != 'audio')

    # Si el dataset es de diferencias, se utiliza de la siguiente manera:
    datos_train = pd.read_csv(train)
    datos_test = pd.read_csv(valid)

    # Variables de entrada y salida, entrenamiento
    X_train = datos_train.iloc[:, :-1]
    y_train = datos_train.iloc[:, -1]

    # Variables de entrada y salida, prueba
    X_test = datos_test.iloc[:, :-1]
    y_test = datos_test.iloc[:, -1]

    # Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Entrenar el clasificador
    rf_classifier.fit(X_train, y_train)

    # Realizar predicciones en los datos de prueba
    y_pred = rf_classifier.predict(X_test)

    # Precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_txt = 'Accuracy: %.3f' % accuracy
    print(accuracy_txt)

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, zero_division=1)
    report_txt = "Reporte de clasificación:\n" + report
    print(report_txt)

    nombre_archivo = 'resultados_' + ventana + '_rf.txt'
    # Guardar en un archivo de texto
    with open(nombre_archivo, 'w') as file:
        file.write("------------------------RANDOM FOREST-------------------------\n")
        file.write("ventanas de: "+ ventana+ '\n')
        file.write(accuracy_txt + '\n')
        file.write(report_txt)

