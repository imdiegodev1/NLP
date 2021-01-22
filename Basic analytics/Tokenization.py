import nltk
import re

##cadena de texto con salto de linea
string1 = 'esto es \n cadena de texto'

##Cadena de texto definida solo como archivo plano
string2 = (r'esta es \n cadena de texto')

#print(string1)
#print(string2)

##Proceso de tokenizacion
string = """ Cuando sea el rey del mundo (imaginaba él en su cabeza) no tendré que  preocuparme por estas bobadas. Era solo un niño de 7 años, pero pensaba que podría ser cualquier cosa que su imaginación le permitiera visualizar en su cabeza ...""" 

##caso 1. tokenizar por espacios vacios
print(re.split(r' ',string))

##caso2. tokenizar usando expresiones regulares
print(re.split(r'[ \t\n]+',string))

##caso 3. si solo estoy tokenizando texto
print(re.split(r'[ \W\t\n]+',string))

##Caso 4, usando regexp_tokenize. Metodo de nltk
##Primero defino una exprecion regular compleja
pattern = r'''(?x)                  # Flag para iniciar el modo verbose
              (?:[A-Z]\.)+            # Hace match con abreviaciones como U.S.A.
              | \w+(?:-\w+)*         # Hace match con palabras que pueden tener un guión interno
              | \$?\d+(?:\.\d+)?%?  # Hace match con dinero o porcentajes como $15.5 o 100%
              | \.\.\.              # Hace match con puntos suspensivos
              | [][.,;"'?():-_`]    # Hace match con signos de puntuación
'''
tokenization = nltk.regexp_tokenize(string,pattern)
print(tokenization)