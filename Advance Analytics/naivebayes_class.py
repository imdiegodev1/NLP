import math
import os
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
tokenizer = Tokenizer(nlp.vocab)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = []
clases = []

###Obtencion de la data para entrenar el modelo

##lectura de spam genera una lista de todos los archivos dentro de la carpeta corpus1/spam
for file in os.listdir('C:/Users/diego/python_platzi/NLTK/NLP_2/corpus1/spam'):
    with open ('C:/Users/diego/python_platzi/NLTK/NLP_2/corpus1/spam/' + file, encoding='latin-1') as f:
        data.append(f.read())
        clases.append('spam')

#print(clases)

##Lectura de data ham
for file in os.listdir('C:/Users/diego/python_platzi/NLTK/NLP_2/corpus1/ham'):
    with open ('C:/Users/diego/python_platzi/NLTK/NLP_2/corpus1/ham/' + file, encoding='latin-1') as f:
        data.append(f.read())
        clases.append('ham')

###Uso de spacy para entrenar el modelo

##probando el tokenizador de spacy
# #print([t.text for t in tokenizer(data[0])])

##recordar que el modelo de bayes utiliza la simplificacion logaritmica
## para evitar casos atipicos (extremos) se utiliza el suavizado de laplace

##Como es un clase nos apoyamos de la POO
##creamos entonces la clase del clasificador

class NaiveBayesClassifier():
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)

    def tokenize (self, doc):
        return [t.text.lower() for t in tokenizer(doc)]

    def word_count(self, words):
        wordCount = {}
        for w in words:
            if w in wordCount.keys():
                wordCount[w] += 1
            else:
                wordCount[w] = 1
        return wordCount
    
    def fit(self, data, clases):
        n = len(data)
        self.unique_clases = set(clases)
        self.vocab = set()
        self.classCount = {}                ##Conteo de cada categoria
        self.log_classPriorProb = {}        ##Logaritmo de la p de la clase
        self.wordConditionalCounts = {}     ##Conteo de la condicion dada x categoria observe una palabra en el documento de esa categoria

        ##conteo de clases
        for c in clases:
            if c in self.classCount.keys():
                self.classCount[c] += 1
            else:
                self.classCount[c] = 1
        
        ##Calculo de la probabilidad de la clase
        for c in self.classCount.keys():
            self.log_classPriorProb[c] = math.log(self.classCount[c]/n)
            self.wordConditionalCounts[c] = {}
        ##Calculo de p condicional (dado una clase esa palabra ocurra)
        for text, c in zip(data, clases):
            counts = self.word_count(self.tokenize(text))
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.wordConditionalCounts[c]:
                    self.wordConditionalCounts[c] [word] = 0.0
                self.wordConditionalCounts[c] [word] += count

    def predict(self, data):
        results = []
        for text in data:
            words = set(self.tokenize(text))
            scoreProb = {}
            for word in words:
                if word not in self.vocab: continue #si palabra no en vocabulario ignorar
                ##suavizado de laplace
                for c in self.unique_clases:
                    log_wordClassProb = math.log((self.wordConditionalCounts[c].get(word, 0.0) + 1) / (self.classCount[c] + len(self.vocab)))
                    scoreProb[c] = scoreProb.get(c, self.log_classPriorProb[c]) + log_wordClassProb
            
            arg_maxprob = np.argmax(np.array(list(scoreProb.values())))
            results.append(list(scoreProb.keys())[arg_maxprob])
        return results


data_train, data_test, clases_train, clases_test = train_test_split(data, clases, test_size = 0.10, random_state=42)

##Entrenar el modelo
classifier = NaiveBayesClassifier()
classifier.fit(data_train, clases_train)

##Probar el modelo
clases_predict = classifier.predict(data_test)

##verificar su accuracy
print(accuracy_score(clases_test, clases_predict))

##Verificar precision
print(precision_score(clases_test, clases_predict, average = None, zero_division=1))

##Verificar recall
print(recall_score(clases_test, clases_predict, average = None, zero_division=1))