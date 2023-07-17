import os
from glob import glob
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import csv
import pandas as pd

# Función para extraer MFCCs:
def mfcc_extract(folder, ft, winlen, nmfcc, reference):
    is_first_iteration = True
    file = r"" + folder + "/*.wav"

    with open(ft, "w", newline="") as k:
        the_writter = csv.writer(k)
        features_title = ["audio"]


        # Cargar el archivo CSV con la información de la clase de cada audio
        with open(reference, "r") as f:
            reader = csv.DictReader(f)
            clases_dict = {row["audio"]: row["clase"] for row in reader}

        # for f in glob(r"Audios/audios_3301/training-20s/*.wav"):
        for f in glob(file):

            file_name = os.path.splitext(os.path.basename(f))[0]  # Obtiene el nombre base del archivo sin la extensión
            clase = clases_dict.get(file_name,"-1")  # Obtiene la clase correspondiente al nombre del archivo, o "-1" si no se encuentra

            (rate, sig) = wav.read(f)
            sig = sig / 32767.0
            sig = np.round(sig, decimals=4)
            # print(file_name, clase, " rate: ",rate, ", longitud sig: ", len(sig))

            chunks = int(len(sig) / (rate*winlen))
            # print("ventanas: ", chunks)
            if is_first_iteration:
                for i in range(1, chunks + 1):
                    for j in range(1, nmfcc + 1):
                        features_title.append(f"mfcc{j}_{i}")
                features_title.append("class")
                the_writter.writerow(features_title)
                is_first_iteration = False

            print(file_name, " rate: ", rate, ", longitud sig: ", len(sig), " ventanas: ", chunks)
            # print("mfcc: ", nmfcc, "len: ",winlen)

            limite = int((rate*winlen))
            # print(limite)
            mfccs = []
            for g in range(0, (chunks * limite), limite):
                # print(g)
                shorten_audio = sig[g: g + limite]
                audio_array = np.array(shorten_audio)

                # print(audio_array)

                mfcc_feat = mfcc(audio_array, samplerate=rate, numcep=nmfcc, winlen=winlen, winstep=0.01, preemph=0.97,
                                 nfilt=20, lowfreq=5, highfreq=500, ceplifter=22, appendEnergy=False)

                mfccs.append(mfcc_feat)

            mfccs = np.array(mfccs)
            mfccs = mfccs.reshape(-1, 16)
            mfccs = mfccs.T.tolist()

            features_array = [file_name]
            for i in range(len(mfccs)):
                features_array += mfccs[i]

            features_array += [clase]
            the_writter.writerow(features_array)

# Función para obtener las diferencias:
def dif_1(dataset, data_dif):
    df = pd.read_csv(dataset)

    # Obtener los nombres de las columnas del dataframe
    columnas = df.columns.tolist()
    mfccs = set()

    # Identificar los MFCC presentes en las columnas
    for columna in columnas:
        if columna.startswith('mfcc'):
            # print(columna)
            mfccs.add(columna.split('_')[0])


    # Ordenar los MFCC
    mfcc_list = sorted(mfccs, key=lambda x: int(x[4:]))
    # print("MFCC ordenados:", mfcc_list)

    # Diccionario para almacenar las diferencias
    diferencias = {}

    # Calcular las diferencias para cada MFCC
    for mfcc in mfccs:
        mfcc_columnas = []
        ventana = 0

        # Buscar las columnas correspondientes al MFCC actual
        while f'{mfcc}_{int(ventana) + 1}' in columnas:
            mfcc_columnas.append(f'{mfcc}_{ventana}')
            ventana += 1

    # print("ventanas: ", ventana)

    # Calcular las diferencias para cada MFCC
    for mfcc in mfcc_list:
        for v in range(1, int(ventana)):
            # Se construyen los nombres de las columnas correspondientes
            columna_ventana1 = f'{mfcc}_{v}'
            columna_ventana2 = f'{mfcc}_{int(v) + 1}'
            # Se calcula la diferencia
            diferencia = df[columna_ventana2] - df[columna_ventana1]
            # Se construye un nombre para esa diferencia y se almacena en el diccionario
            nombre_diferencia = f'Dif{int(v) + 1}-{v}_{mfcc}'
            diferencias[nombre_diferencia] = diferencia

    # Crear un DataFrame con las diferencias del diccionario
    diferencias_df = pd.DataFrame(diferencias)
    df_final = pd.concat([diferencias_df, df['class']], axis=1)

    print("DataFrame de diferencias:")
    print(df_final)
    df_final.to_csv(data_dif, index=False)

# Función para obtener max, min, prom de diferencias:
def diferencias(data_mfcc, data_dif, dataset):
    # Cargar el archivo de diferencias y mfcc
    diferencias_df = pd.read_csv(data_dif)
    mfcc_df = pd.read_csv(data_mfcc)

    # Obtener los nombres de las columnas del dataframe
    columnas = mfcc_df.columns.tolist()
    mfccs = set()

    # Identificar los MFCC presentes en las columnas
    for columna in columnas:
        if columna.startswith('mfcc'):
            # print(columna)
            mfccs.add(columna.split('_')[0])

    # print(len(mfccs))

    for mfcc in mfccs:
        mfcc_columnas = []
        ventana = 0

        # Buscar las columnas correspondientes al MFCC actual
        while f'{mfcc}_{int(ventana) + 1}' in columnas:
            mfcc_columnas.append(f'{mfcc}_{ventana}')
            ventana += 1

    # print("ventanas: ", ventana)


    # DataFrame para almacenar los resultados
    resultados_df = pd.DataFrame()



    # Recorrer los MFCC 1 al 16:
    for mfcc in range(1, len(mfccs)+1):
        # Lista que contiene los nombres de las columnas correspondientes al MFCC actual
        mfcc_columnas = [f"Dif{i + 2}-{i + 1}_mfcc{mfcc}" for i in range(1, int(ventana)-1)]
        # print(mfcc_columnas)
        # Obtener los valores mínimos y máximos por fila
        # Da como resultado una serie que contiene los valores mínimos y máximos por fila
        min_val = diferencias_df[mfcc_columnas].min(axis=1)
        max_val = diferencias_df[mfcc_columnas].max(axis=1)

        # Calcular el promedio por fila
        prom_mfcc = diferencias_df[mfcc_columnas].mean(axis=1)

        # Agregar las series de valores mínimos, máximos y promedio al DataFrame de resultados
        resultados_df[f"min_mfcc{mfcc}"] = min_val
        resultados_df[f"max_mfcc{mfcc}"] = max_val
        resultados_df[f"prom_mfcc{mfcc}"] = prom_mfcc

    # Reordenar las columnas: primero valores mínimos, luego máximos y por último los promedios
    columnas_ordenadas = []
    for mfcc in range(1, len(mfccs)+1):
        columnas_ordenadas.append(f"max_mfcc{mfcc}")
    for mfcc in range(1, len(mfccs)+1):
        columnas_ordenadas.append(f"min_mfcc{mfcc}")
    for mfcc in range(1, len(mfccs)+1):
        columnas_ordenadas.append(f"prom_mfcc{mfcc}")
    resultados_df = resultados_df[columnas_ordenadas]

    df_final = pd.concat([resultados_df, mfcc_df['class']], axis=1)

    print("DataFrame:")
    print(df_final)
    # Guardar los resultados en un archivo CSV
    df_final.to_csv(dataset, index=False)

