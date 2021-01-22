import nltk
from nltk.book import text1
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.util import bigrams
import plotly.express as px
from nltk.util import ngrams

##Obtener colocaciones de forma rapida y eficiente sin usar el grafico de dispersion

from nltk.collocations import *
bigram_measure = nltk.collocations.BigramAssocMeasures() ##importar metodos de nltk para usar metricas. incluso pmi
finder = BigramCollocationFinder.from_words(text1)  ##Metodo que ayuda a encontrar las colocaciones

finder.apply_freq_filter(20)
print(finder.nbest(bigram_measure.pmi, 10))


##Mismo ejercicio pero en español

nltk.download('cess_esp')
corpus = nltk.corpus.cess_esp.sents()

##cambiar corpus de lista de listas a una lista plana
flatten_corpus = [w for l in corpus for w in l]

##ver colocaciones de corpus en español

finder = BigramCollocationFinder.from_documents(corpus)
finder.apply_freq_filter(10)
#print(finder.nbest(bigram_measure.pmi, 10))