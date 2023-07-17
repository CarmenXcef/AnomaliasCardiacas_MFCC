import MFCC
import Classification

# Extracción de MFCC:
# ruta audios, dataset para almacenar MFCC, longitud de ventana, número de mfcc, archivo de referencia de clases
MFCC.mfcc_extract("Audios/Training", "Datasets/MFCC/MFCCT_5seg.csv", 5, 16, "Audios/Training/REFERENCE1.csv")
MFCC.mfcc_extract("Audios/Validation", "Datasets/MFCC/MFCCV_5seg.csv", 5, 16, "Audios/Validation/REFERENCE.csv")
"""
Entrenamiento: 
MFCC.mfcc_extract("Audios/Training", "Datasets/MFCC/MFCCT_cuartoCiclo.csv", 0.2, 16, "Audios/Training/REFERENCE1.csv")
MFCC.mfcc_extract("Audios/Training", "Datasets/MFCC/MFCCT_30ms.csv", 0.3, 16, "Audios/Training/REFERENCE1.csv")
MFCC.mfcc_extract("Audios/Training", "Datasets/MFCC/MFCCT_medioCiclo.csv", 0.4, 16, "Audios/Training/REFERENCE1.csv")
MFCC.mfcc_extract("Audios/Training", "Datasets/MFCC/MFCCT_ciclo.csv", 0.8, 16, "Audios/Training/REFERENCE1.csv")

Prueba:
MFCC.mfcc_extract("Audios/Validation", "Datasets/MFCC/MFCCV_cuartoCiclo.csv", 0.2, 16, "Audios/Validation/REFERENCE.csv")
MFCC.mfcc_extract("Audios/Validation", "Datasets/MFCC/MFCCV_30ms.csv", 0.3, 16, "Audios/Validation/REFERENCE.csv")
MFCC.mfcc_extract("Audios/Validation", "Datasets/MFCC/MFCCV_medioCiclo.csv", 0.4, 16, "Audios/Validation/REFERENCE.csv")
MFCC.mfcc_extract("Audios/Validation", "Datasets/MFCC/MFCCV_ciclo.csv", 0.8, 16, "Audios/Validation/REFERENCE.csv")
"""


# Cálculo de diferencias
# ruta del dataset con los MFCC, ruta en la que se va a guardar el archivo
MFCC.dif_1("Datasets/MFCC/MFCCT_5seg.csv", "Datasets/Diferencias/diferenciasT_5seg.csv")
MFCC.dif_1("Datasets/MFCC/MFCCV_5seg.csv", "Datasets/Diferencias/diferenciasV_5seg.csv")
"""
MFCC.dif_1("Datasets/MFCC/MFCCV_cuartoCiclo.csv", "Datasets/Diferencias/diferenciasV_200ms.csv")
MFCC.dif_1("Datasets/MFCC/MFCCV_medioCiclo.csv", "Datasets/Diferencias/diferenciasV_400ms.csv")
MFCC.dif_1("Datasets/MFCC/MFCCV_30ms.csv", "Datasets/Diferencias/diferenciasV_30ms.csv")
MFCC.dif_1("Datasets/MFCC/MFCCV_ciclo.csv", "Datasets/Diferencias/diferenciasV_800ms.csv")

MFCC.dif_1("Datasets/MFCC/MFCCT_30ms.csv", "Datasets/Diferencias/diferenciasT_30ms.csv")
MFCC.dif_1("Datasets/MFCC/MFCCT_cuartoCiclo.csv", "Datasets/Diferencias/diferenciasT_200ms.csv")
MFCC.dif_1("Datasets/MFCC/MFCCT_medioCiclo.csv", "Datasets/Diferencias/diferenciasT_400ms.csv")
MFCC.dif_1("Datasets/MFCC/MFCCT_ciclo.csv", "Datasets/Diferencias/diferenciasT_800ms.csv")
"""

# Cálculo de mínimo, máximo y promedio de diferencias
# ruta del dataset con los MFCC, ruta del dataset con las diferencias, ruta en la que se va a guardar el archivo
MFCC.diferencias("Datasets/MFCC/MFCCT_5seg.csv", "Datasets/Diferencias/diferenciasT_5seg.csv", "Datasets/Train_5seg.csv")
MFCC.diferencias("Datasets/MFCC/MFCCV_5seg.csv", "Datasets/Diferencias/diferenciasV_5seg.csv", "Datasets/Validation_5seg.csv")
"""
diferencias("Datasets/MFCC/MFCCT_cuartoCiclo.csv", "Datasets/Diferencias/diferenciasT_200ms.csv", "Datasets/Train_200ms.csv")
diferencias("Datasets/MFCC/MFCCT_medioCiclo.csv", "Datasets/Diferencias/diferenciasT_400ms.csv", "Datasets/Train_400ms.csv")
diferencias("Datasets/MFCC/MFCCT_30ms.csv", "Datasets/Diferencias/diferenciasT_30ms.csv", "Datasets/Train_30ms.csv")
diferencias("Datasets/MFCC/MFCCT_ciclo.csv", "Datasets/Diferencias/diferenciasT_800ms.csv", "Datasets/Train_800ms.csv")

diferencias("Datasets/MFCC/MFCCV_cuartoCiclo.csv", "Datasets/Diferencias/diferenciasV_200ms.csv", "Datasets/Validation_200ms.csv")
diferencias("Datasets/MFCC/MFCCV_medioCiclo.csv", "Datasets/Diferencias/diferenciasV_400ms.csv", "Datasets/Validation_400ms.csv")
diferencias("Datasets/MFCC/MFCCV_30ms.csv", "Datasets/Diferencias/diferenciasV_30ms.csv", "Datasets/Validation_30ms.csv")
diferencias("Datasets/MFCC/MFCCV_ciclo.csv", "Datasets/Diferencias/diferenciasV_800ms.csv", "Datasets/Validation_800ms.csv")

"""


# Para clasificar utilizando los datasets que contienen las diferencias:
Classification.rLog('Datasets/Train/Train_5seg.csv', 'Datasets/Validation/Validation_5seg.csv')
Classification.supvm('Datasets/Train/Train_5seg.csv', 'Datasets/Validation/Validation_5seg.csv')
Classification.randomF('Datasets/Train/Train_5seg.csv', 'Datasets/Validation/Validation_5seg.csv')
"""
Classification.rLog('Datasets/Train/Train_800ms.csv', 'Datasets/Validation/Validation_800ms.csv')
Classification.rLog('Datasets/Train/Train_400ms.csv', 'Datasets/Validation/Validation_400ms.csv')
Classification.rLog('Datasets/Train/Train_200ms.csv', 'Datasets/Validation/Validation_200ms.csv')
Classification.rLog('Datasets/Train/Train_30ms.csv', 'Datasets/Validation/Validation_30ms.csv')
"""

# Para clasificar utilizando los datasets que contienen los MFCC:
Classification.rLog('Datasets/MFCC/MFCCT_5seg.csv', 'Datasets/MFCC/MFCCV_5seg.csv')
Classification.supvm('Datasets/MFCC/MFCCT_5seg.csv', 'Datasets/MFCC/MFCCV_5seg.csv')
Classification.randomF('Datasets/MFCC/MFCCT_5seg.csv', 'Datasets/MFCC/MFCCV_5seg.csv')
"""
Classification.rLog('Datasets/MFCC/MFCCT_ciclo.csv', 'Datasets/MFCC/MFCCV_ciclo.csv')
Classification.rLog('Datasets/MFCC/MFCCT_medioCiclo.csv', 'Datasets/MFCC/MFCCV_medioCiclo.csv')
Classification.rLog('Datasets/MFCC/MFCCT_cuartoCiclo.csv', 'Datasets/MFCC/MFCCV_cuartoCiclo.csv')
Classification.rLog('Datasets/MFCC/MFCCT_30ms.csv', 'Datasets/MFCC/MFCCV_30ms.csv')
"""