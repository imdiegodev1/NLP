import nltk
from nltk.book import text1
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.util import bigrams
import plotly.express as px
from nltk.util import ngrams

##Crear bigramas
mb_bigrams = list(bigrams(text1))
print(mb_bigrams)

##Crear frecuencia de bigramas
fdist = FreqDist(mb_bigrams)

##Empezar a crear filtros para los bigramas (palabras con mas de 3 caracteres)
threshold = 2
filtered_bigrams = [bigram for bigram in mb_bigrams if len(bigram[0])>threshold and len(bigram[1])>threshold]

##FreqDist para saber las frecuencias de palabras
filtered_bigram_dist = FreqDist(filtered_bigrams)

##trabajar con trigramas
md_trigrams = list(ngrams(text1, 3))
fdist_trigrams = FreqDist (md_trigrams)
print(fdist_trigrams.most_common(20))

###############################
######Colocaciones##########

filtered_words = [word for word in text1 if len(word) > threshold]
filtered_words_dist = FreqDist(filtered_words)

##crear un dataframe
df = pd.DataFrame()
df['bi_grams'] = list(set(filtered_bigrams))

##Separar las palabras del bigrama cada una en su columna
df['word_0'] = df['bi_grams'].apply(lambda x: x[0])
df['word_1'] = df['bi_grams'].apply(lambda x: x[1])

##crear columnas para calcular la metrica PMI
df['bi_gram_freq'] = df['bi_grams'].apply(lambda x: filtered_bigram_dist[x])
df['word_0_freq'] = df['word_0'].apply(lambda x: filtered_words_dist[x])
df['word_1_freq'] = df['word_1'].apply(lambda x: filtered_words_dist[x])

##Calcular PMI
df['PMI'] = df[['bi_gram_freq','word_0_freq','word_1_freq']].apply(lambda x: np.log2(x.values[0]/(x.values[1]*x.values[2])), axis =1)

df.sort_values(by = 'PMI',ascending = False)

##Grafico de dispersion para colocaciones
df['log(bi_gram_freq)'] = df['bi_gram_freq'].apply(lambda x: np.log2(x))

fig = px.scatter(x= df['PMI'].values,y= df['log(bi_gram_freq)'],color= df['PMI']+df['log(bi_gram_freq)'],hover_name = df['bi_grams'].values ,width= 600,height= 600,labels={'x': 'PMI', 'y': 'log (bigram frequency'})

fig.show()
print(df)