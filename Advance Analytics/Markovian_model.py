from conllu import parse_incr
import nltk
import numpy as np

WordList = []
data_file = open('C:/Users/diego/python_platzi/NLTK/NLP_2/AnCora/UD_Spanish-AnCora/es_ancora-ud-dev.conllu', 'r', encoding="utf-8")

#for tokenlist in parse_incr(data_file):
#    print(tokenlist.serialize())

tagCountDict = {} 
emissionDict = {}
transitionDict = {}

tagtype = 'upos'

# Calculando conteos (pre-probabilidades)
for tokenlist in parse_incr(data_file):
    prevtag = None
    for token in tokenlist:

        #C(tag)
        tag = token[tagtype]
        if tag in tagCountDict.keys():
            tagCountDict [tag] += 1
        else:
            tagCountDict [tag] = 1
        
        #C(word\tag) -> probabilidad emision
        wordtag = token['form'].lower()+'|'+token[tagtype]
        if wordtag in emissionDict.keys():
            emissionDict[wordtag] = emissionDict[wordtag] + 1
        else:
            emissionDict[wordtag] = 1
        
        #C(tag|tag_previo)
        if prevtag is None:
            prevtag = tag
            continue
        transitiontags = tag+ '|'+ prevtag
        if transitiontags in transitionDict.keys():
            transitionDict[transitiontags] = transitionDict[transitiontags]+1
        else:
            transitionDict[transitiontags] = 1
        prevtag = tag


transitionProbDict = {} # matriz A
emissionProbDict = {} # matriz B

# transition Probabilities 
for key in transitionDict.keys():
  tag, prevtag = key.split('|')
  if tagCountDict[prevtag]>0:
    transitionProbDict[key] = transitionDict[key]/(tagCountDict[prevtag])
  else:
    print(key)

# emission Probabilities 
for key in emissionDict.keys():
  word, tag = key.split('|')
  if emissionDict[key]>0:
    emissionProbDict[key] = emissionDict[key]/tagCountDict[tag]
  else:
    print(key)

transitionProbDict['ADJ|ADJ']
#emissionProbDict

np.save('transitionHMM.npy', transitionProbDict)
np.save('emissionHMM.npy', emissionProbDict)
transitionProbdict = np.load('transitionHMM.npy', allow_pickle='TRUE').item()
transitionProbDict['ADJ|ADJ']