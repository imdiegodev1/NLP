import nltk
from nltk.book import text1
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np


##see the tokenization
tokenize = text1.tokens[:10]

##count how many tokens it has
count_tokens = len(text1)

##Construir vocabulario con funcion set (palabras sin repetir)
vocabulary = sorted(set(text1))

##Riquesa lexica (palabras unicas)/(total palabras)
rl = len(vocabulary)/len(text1)
##print(rl)

##como deberia definirse una funcion de riquesa lexica

def riquesa_lexica (texto):
    vocabulary = sorted(set(text1))
    return len(vocabulary)/len(texto)

rl2 = riquesa_lexica(text1)

##funcion para saber q porcentaje de texto una palabra consume en todo el texto
def porcentaje_palabra (palabra,texto):
    return 100*texto.count(palabra)/len(texto)

pp = porcentaje_palabra('monster',text1)

##construir un diccionario que cuente la ocurrencia de cada palabra
fdist = FreqDist(text1)
print (fdist)

##Saber el top 20 de palabras en el texto
common_words = fdist.most_common(20)
#print(common_words)
##graficar un histograma con el numero de palagras y ocurrencias
#plot = fdist.plot(20)

##Saber una palabra cuantas veces se repite
ocurrency = fdist['monster']

##crear un diccionario de palabras de longitud mayor
long_words = [palabra for palabra in text1 if len(palabra)>5]  ##filtro fino, solo palabras con mas de 5 letras
vocabulario_filtrado = sorted(set(long_words))  ##Ordenar de forma alfabetica y seleccionar elementos sin q se repitan

##Conteo de palabras de longitud mayor y con frecuencia mayor a 10
palabras_interesantes = [(palabra, fdist[palabra]) for palabra in set(text1) if len(palabra)>5 and fdist[palabra] > 10]

##Convertir lista a un elemento numpy (es una libreria muy util)        ##Al final esto es lo mismo que hace el freqdist
##Primero defino el tipo de dato que voy a utilizar
dtypes = [('word','S10'),('frequency', int)]
##combierto el elemento palabras_interesantes a un objeto np usando los tipos de datos
palabras_interesantes = np.array(palabras_interesantes, dtype = dtypes)
##Organizar palabras interesantes de forma alfabetica
palabras_interesantes = np.sort(palabras_interesantes, order = 'frequency')

##graficar elemento np usando libreria matplotlist
top_words=20
x = np.arange(len(palabras_interesantes[-top_words:]))
y = [freq[1] for freq in palabras_interesantes[-top_words:]]
plt.figure(figsize = (10,5))
plt.plot(x,y)
plt.xticks(x, [str(freq[0]) for freq in palabras_interesantes[-top_words:]], rotation = 'vertical')
plt.grid(True)
plt.show()
#print(palabras_interesantes)