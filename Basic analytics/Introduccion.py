import nltk
import re   ##Regular Expresion library

#Call corpus string cess_esp
corpus = nltk.corpus.cess_esp.sents()

##concatenate all the list inside the list cess_esp
flatten = [w for l in corpus for w in l]

##See only the first n elements
#print (flatten [:100])

##use regular expresions. add a certein words that  satisfied a regular expresion search pattern
##This time the idea is search all the elements on the list that contains "es"
arr = [w for w in flatten if re.search('es', w)]
#print(arr[:20])

##Using more complicate search patterns "es"=the string contains
##"es$"=at the end of string    "^es"=at the beggining
arr = [w for w in flatten if re.search('es$',w )]
#print(arr[:20])

##Range in this case from a-z. really usefull '^[ghi]'=character is at the beggining of the string
arr = [w for w in flatten if re.search('^[ghi]',w )]
##print(arr[:10])

##Using clausuras * and *+
arr = [w for w in flatten if re.search('^(no)+',w )]
##print(arr[:20])
##print(len(corpus))
