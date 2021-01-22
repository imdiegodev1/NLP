from nltk.corpus import swadesh

##Que hay dentro de swadesh. que idiomas
print(swadesh.fileids())

##Saber que palabras tiene swadesh en ingles
print (swadesh.words('en'))

##objeto que me ayuda a definir un diccionario par atraducir las palabras del frances al español
fr2es = swadesh.entries(['fr','en'])
print(fr2es)

##crear diccionario de frances a ingles
translate = dict(fr2es)
print(translate['chien'])  ##traducir la plabra chien en frances al ingles
