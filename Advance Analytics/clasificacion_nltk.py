import nltk, random
#nltk.download('names')
from nltk.corpus import names

##primer atributo es extraer la ultima letra de los nombres
def atributos(palabra):
    return {'ultima letra': palabra[-1]}

#obtencion de la lista de nombres y juntarlas
tagset = [(name,'Male') for name in names.words('male.txt')] + [(name,'Female') for name in names.words('female.txt')]

##Mezclar las dos listas
random.shuffle(tagset)

#print(tagset)


fset = [(atributos(n), g) for (n, g) in tagset]

##Determinar con que voy a entrenar mi modelo y con que lo voy a probar
train, test = fset[500:], fset[:500]

##Entrenar modeloc on naive bayes
classifier = nltk.NaiveBayesClassifier.train(train)

#print(classifier.classify(atributos('rosalba')))

##que porcentaje de accurancy tiene este modelo
#print(nltk.classify.accuracy(classifier, test))

##probar con nuevos atributos

##La funcion utiliza entonces como atributos primera, ultima letra, 
##numero de veces que se repite una letra y si tiene o no alguna letra
def mas_atributos(nombre):
    atrib = {}
    atrib['primera_letra'] = nombre[0].lower()
    atrib['ultima_letra'] = nombre[-1].lower()

    for letra in 'abcdefghijklmnopqrstuvwxyz':
        atrib['count {}'.format(letra)] = nombre.lower().count(letra)
        atrib['has {}'.format(letra)] = (letra in nombre.lower())

    return atrib

#print(mas_atributos('jhon'))

fset = [(mas_atributos(n), g) for (n, g) in tagset]

train, test = fset[500:], fset[:500]

##entrenamiento del modelo con muchos mas atributos
classifier2 = nltk.NaiveBayesClassifier.train(train)

##calculo de accurancy del nuevo modelo
#print(nltk.classify.accuracy(classifier2, test))



###Ejercicio clasificacion de emails
import pandas as pd
import numpy as np
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize

df = pd.read_csv(r'C:/Users/diego/python_platzi/NLTK/NLP_2/dataset_mails/datasets/email/csv/spam-apache.csv', names = ['clase', 'contenido'])

df['tokens'] = df['contenido'].apply(lambda x: word_tokenize(x))
#print(df['tokens'].values[0])

##Tokenizacion de los emails
all_words = nltk.FreqDist([w for tokenlist in df['tokens'].values for w in tokenlist])
top_words = all_words.most_common(200)
##print(top_words)

def documento_atributo(document):
    document_words = set(document)
    atrib = {}
    for word in top_words:
        atrib['contains({})'.format(word)] = (word in document_words)
    return atrib

#print(documento_atributo(df['tokens'].values[0]))

fset = [(documento_atributo(texto), clase) for texto, clase in zip(df['tokens'].values, df['clase'].values)]
#print(fset)
random.shuffle(fset)
train, test = fset[:200], fset[200:]

classifier = nltk.NaiveBayesClassifier.train(train)

#print(nltk.classify.accuracy(classifier,test))

##crear mas atributos. hay una funcion que me puede ayudar a ver q atributos uso el modelo
print(classifier.show_most_informative_features(5))