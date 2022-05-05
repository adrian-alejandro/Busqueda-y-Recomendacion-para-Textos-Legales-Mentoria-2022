#!/usr/bin/env python
# coding: utf-8

# # Universidad Nacional de Córdoba - Facultad de Matemática, Astronomía, Física y Computación
#
# ### Diplomatura en Ciencia de Datos, Aprendizaje Automático y sus Aplicaciones 2022
# Búsqueda y Recomendación para Textos Legales
#
# Mentor: Jorge E. Pérez Villella
#
# # Práctico Análisis y Visualización
#
# Integrantes:
# * Fernando Agustin Cardellino
# * Adrian Zelaya
#

# ### Objetivos:
#
# * Generar un corpus con todos los documentos.
#
# * Dividir el corpus en tokens, graficar el histograma de frecuencia de palabras demostrando la ley Zipf.
#
# * Analizar palabras más frecuentes y menos frecuentes. Seleccionar 5 documentos de cada fuero y realizar el mismo análisis. ¿Se repiten las palabras?
#
# * Hacer lo mismo con n-gramas.
#
# * Visualizar la frecuencia de palabras en una nube de palabras.
#
# * Elaborar una breve conclusión de lo encontrado
#
# Fecha de Entrega: 20 de mayo de 2022

# ## Actividades
# ### Generar un corpus

# In[31]:


import spacy
import matplotlib
import os
import pandas as pd

nlp = spacy.load("es_core_news_sm")

CURR_DIR = os.getcwd()  # Gets current directory


def getListOfFiles(dirName, return_dir=False):
    # create a list of file and sub directories
    # names in the given directory
    files = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for file in files:
        # Create full path
        fullPath = dirName + "\\" + file
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            if return_dir:
                allFiles.append(fullPath.split("\\")[-1])
            else:
                allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


# armar función para obtener subcarpetas de Documentos

filesDir = f"{CURR_DIR}\Documentos"

corpus = {}

fueros = getListOfFiles(filesDir, return_dir=True)

print(fueros)

corpus_list = []

palabras = []

i = 0
for filename in getListOfFiles(filesDir):
    file_name = filename.split("\\")[-1]
    fuero = filename.split("\\")[-2]
    if fuero not in corpus.keys():
        corpus[fuero] = {}

    # Creamos este diccionario para luego utilizarlo para armar un dataframe
    corpus_dict = {'fuero': fuero,
                   'documento': file_name}

    with open(filename, encoding='utf-8') as file:
        file_text = file.read()
        corpus[fuero][file_name] = nlp(file_text)
        corpus_dict['texto'] = nlp(file_text)
        corpus_list.append(corpus_dict)
        for token in nlp(file_text):
            palabras.append(token)
    # remover esta sección (testing)
    i += 1
    if i > 2:
        break
# [{'fuero': FAMILIA, 'documento': filename, 'texto': nlp(file_text)}]
# [{'column':value, 'column2':value}, {'column':value, 'column2':value}]

corpus_df = pd.DataFrame(corpus_list)
palabras_df = pd.DataFrame([{'palabra': str(x).lower()} for x in palabras])
# print(corpus_df.head())


p_df = palabras_df.groupby(['palabra'])['palabra'].count()

p_df[p_df > 50].plot(kind='bar')
# palabras_df['count'] = palabras_df.groupby(['palabra'])['palabra'].count().values#.plot.bar()
# palabras_df.head(50)
