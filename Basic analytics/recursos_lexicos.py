import nltk
#nltk.download('book')
from nltk.book import *
from nltk.corpus import stopwords

##vocabulario
vocab = sorted(set(text1))

##frecuencia de palabras
word_freq = FreqDist(text1)

##Trabajar con stopwords(hay q definir el idioma)
print(stopwords.words('english'))

def stopwords_percentage(text):
    
    stopwd = stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwd]
    return len(content)/len(text)

relevant_words = stopwords_percentage(text1)
print(relevant_words)