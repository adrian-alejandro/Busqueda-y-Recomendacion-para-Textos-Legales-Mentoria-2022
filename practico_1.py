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

# Start with loading all necessary libraries
import numpy as np
import pandas as pd
import os
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
import pickle
#import nltk
#nltk.download('stopwords')  # para bajar las stopwords


nlp = spacy.load("es_core_news_sm")

CURR_DIR = os.getcwd()  # Gets current directory
STOPWORDS_ES = stopwords.words('spanish')


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

### Primer punto
# armar función para obtener subcarpetas de Documentos

filesDir = f"{CURR_DIR}\Documentos"

fueros = getListOfFiles(filesDir, return_dir=True)


def get_palabras(files_path, fuero_name=None, ngrams=False):
    corpus = {}

    corpus_list = []

    palabras = []

    i = 0
    for filename in getListOfFiles(files_path):
        file_name = filename.split("\\")[-1]
        fuero = fuero_name if fuero_name is not None else filename.split("\\")[-2]
        if fuero not in corpus.keys():
            corpus[fuero] = {}

        # Creamos este diccionario para luego utilizarlo para armar un dataframe
        corpus_dict = {'fuero': fuero,
                       'documento': file_name}

        with open(filename, encoding='utf-8') as file:
            file_text = file.read()

            nlp_doc = nlp(file_text)
            corpus[fuero][file_name] = nlp_doc
            corpus_dict['texto'] = nlp_doc
            corpus_list.append(corpus_dict)
            if ngrams:
                nlp_doc = nlp_doc.noun_chunks
            for token in nlp_doc:
                if ngrams:
                    palabras.append(token)
                elif token.is_alpha:  # si es sólo alfabético
                    palabras.append(token)
        # remover esta sección (testing)
        i += 1
        if i > 5:
            break
    return palabras, corpus
# TODO: correr todos los documentos y guardar los objetos en formato pkl, así no es necesario correr el proceso entero
# TODO: hacer limpieza de los espacios

# [{'fuero': FAMILIA, 'documento': filename, 'texto': nlp(file_text)}]
# [{'column':value, 'column2':value}, {'column':value, 'column2':value}]

#corpus_df = pd.DataFrame(corpus_list)

def save_to_pickle(obj, filename):
    """
    TODO: [E111] Pickling a token is not supported, because tokens are only views of the parent Doc and can't exist on
    their own. A pickled token would always have to include its Doc and Vocab, which has practically no advantage over
    pickling the parent Doc directly. So instead of pickling the token, pickle the Doc it belongs to.
    :param obj:
    :param filename:
    :return:
    """

    file_path = f"{CURR_DIR}\\{filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'rb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def get_conteo_palabras(palabras):
    palabras_df = pd.DataFrame([{'palabra': str(x).lower()} for x in palabras])
    # print(corpus_df.head())

    return palabras_df.groupby(['palabra'])['palabra'].count().sort_values(ascending=False)


palabras, corpus_dic = get_palabras(filesDir)

p_df = get_conteo_palabras(palabras)

print(p_df.head(5))
print(p_df.index)

#save_to_pickle(palabras, f"pickles\\palabras_ALL_DOCS.pkl")
#save_to_pickle(corpus_dic, f"pickles\\corpus_ALL_DOCS.pkl")
#save_to_pickle(p_df, f"pickles\\frecuencia_palabras_ALL_DOCS.pkl")

def show_histogram(dataframe, threshold=1):
    dataframe[dataframe > threshold].plot(kind='bar')
    plt.show()



# palabras_df['count'] = palabras_df.groupby(['palabra'])['palabra'].count().values#.plot.bar()
# palabras_df.head(50)

### segundo punto
# TODO: ponerlo más lindo
#print(p_df.head())

def show_zipf(dataframe):
    rank_palabras = [x + 1 for x in range(len(dataframe))]
    
    sns.scatterplot(x=rank_palabras,
                    y=dataframe).set(xscale="log",
                                yscale="log",
                                ylabel='log(Frecuencia)',
                                xlabel='log(Orden)')
    
    plt.show()


# Tercer punto >> análisis de más y menos frecuente del corputs +  Seleccionar 5 documentos de cada fuero y realizar el mismo análisis.
def comparar_frecuencias_palabras(dataframe, description=None):
    if description:
        print(description)
    print(dataframe.head(50))  # La mayoría son todas stopwords, a partir de la 20ma empiezan a haber palabras propias del ámbito jurídico
    print(dataframe.tail(50))  # palabras normales pero donde la mayoría comparte una base en común (e.g. ordenadas, ordenado, ordenando) >> con lematización, se podría disminuir esto
    return dataframe.head(50).index.values

fueros_lista = []

# Analizamos por fueros
for fuero in ['FAMILIA', 'LABORAL', 'MENORES', 'PENAL']:
    filesDir = f"{CURR_DIR}\\Documentos\\{fuero}"
    palabras_fuero, corpus_dic_fuero = get_palabras(filesDir, fuero_name=fuero)

    palabras_df = get_conteo_palabras(palabras_fuero)

    # Convertimos las palabras más frecuentes en un conjunto, a modo de poder utilizar la propiedad intersección (de conjuntos)
    fueros_lista.append(
        set(comparar_frecuencias_palabras(palabras_df, description=fuero))
    )

    #save_to_pickle(palabras_fuero, f"pickles\\palabras_FUERO_{fuero}.pkl")
    #save_to_pickle(corpus_dic_fuero, f"pickles\\corpus_FUERO_{fuero}.pkl")
    #save_to_pickle(palabras_df, f"pickles\\frecuencia_palabras_FUERO_{fuero}.pkl")

fueros_intersection = fueros_lista[0].intersection(*[x for x in fueros_lista[1:]])

# las palabras frecuentes que se repiten son stopwords
print(fueros_intersection)


## Análisis n-grams


def get_n_grams(tokens, n):
    return [' '.join([token.text for token in tokens[i:i+n]]) for i in range(len(tokens) - n + 1)]


chunks = get_palabras(filesDir, fuero_name=None, ngrams=True)

#palabras = get_palabras(filesDir, fuero_name=None)

for n in range(2, 4):
    n_grams = get_n_grams(palabras, n)
    #print(n_grams)
    ngrams_count = get_conteo_palabras(n_grams)

    #save_to_pickle(n_grams, f"pickles\\tokens_{n}-grams.pkl")
    #save_to_pickle(ngrams_count, f"pickles\\frecuencia_{n}-grams.pkl")

    show_histogram(ngrams_count)
    show_zipf(ngrams_count)

# Word-cloud:
#https://www.datacamp.com/tutorial/wordcloud-python


def generar_wordcloud(img_name, stopwords, tokens, output_path):

    image_name = f"img/{img_name}.jpg"


    # Generate a word cloud image
    mask = np.array(Image.open(image_name))
    text = ' '.join(
        [token.text for token in tokens])
    wordcloud_law = WordCloud(
        stopwords=stopwords,
        background_color="white",
        mode="RGBA",
        max_words=1000,
        mask=mask).generate(text)

    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[7, 7])
    plt.imshow(wordcloud_law.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")

    # store to file
    plt.savefig(output_path, format="png")

    plt.show()


img_name = "legal-icon-png"
output_name = f"img/{img_name}_wordcloud.png"

generar_wordcloud(img_name, STOPWORDS_ES, palabras, output_name)