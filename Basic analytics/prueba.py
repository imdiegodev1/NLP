import nltk
from nltk.book import text1
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.util import bigrams
import plotly.express as px
from nltk.util import ngrams
from nltk.corpus import stopwords
ll = ['vamos', 'a', 'ver', 'a', 'que', 'tipo', 'de', 'moneda', 'vamos', 'a', 'convertir']
fdist = FreqDist(ll)
#print(fdist.most_common(2))

ll2 = ['hola', 'hola', 'mundo', 'tengo' , 'frio', 'si', 'si']
#print(sorted(set(ll2)))

print(stopwords.words('spanish'))