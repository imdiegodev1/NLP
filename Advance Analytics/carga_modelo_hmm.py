import numpy as np
from conllu import parse_incr
import nltk
#nltk.download('treebank')
from nltk.corpus import treebank
from nltk import word_tokenize
from nltk.tag import hmm

transitionProbdict = np.load('transitionHMM.npy', allow_pickle='TRUE').item()
emissionProbdict = np.load('emissionHMM.npy', allow_pickle='TRUE').item()

# identificamos las categorias gramaticales 'upos' unicas en el corpus
stateSet = set([w.split('|')[1] for w in list(emissionProbdict.keys())])
#print(stateSet)

# enumeramos las categorias con numeros para asignar a 
# las columnas de la matriz de Viterbi
tagStateDict = {}
for i, state in enumerate(stateSet):
  tagStateDict[state] = i
#print(tagStateDict)

##Calcular distribucion inicial de estado latente
##Estados iniciales en viterbi es la probabilidad de la primera palabra
WordList = []
data_file = open('C:/Users/diego/python_platzi/NLTK/NLP_2/AnCora/UD_Spanish-AnCora/es_ancora-ud-dev.conllu', 'r', encoding="utf-8")
count = 0
initTagStateProb = {}
for tokenlist in parse_incr(data_file):
    count += 1
    tag = tokenlist[0]['upos']
    if tag in initTagStateProb.keys():
        initTagStateProb[tag] += 1
    else:
        initTagStateProb[tag] = 1

for key in initTagStateProb.keys():
    initTagStateProb[key] /= count

#print(initTagStateProb)

##verificar que la suma de la p es 1
verificacion = np.array([initTagStateProb[k] for k in initTagStateProb.keys()]).sum()
#print(verificacion)

##Hasta aqui todo esta listo para generar el algoritmo de viterbi. buscar entre todos los caminos el mas posible
##y por lo tanto la distribucion de etiquetas mas probables.



def ViterbiMatrix (secuencia, transitionProbdict = transitionProbdict, emissionProbdict = emissionProbdict, tagStateDict = tagStateDict, initTagStateProb = initTagStateProb):
    ##El primer paso es determinar las p de la primera columna
    
    seq = word_tokenize(secuencia)
    viterbiProb = np.zeros((17, len(seq)))
    
    for key in tagStateDict.keys():
        tag_row = tagStateDict[key]
        word_tag = seq[0].lower() + '|'+ key
        if word_tag in emissionProbdict.keys():
            viterbiProb[tag_row, 0] = initTagStateProb[key]*emissionProbdict[word_tag]
    #return viterbiProb

    ##Computo de las p de las demas columnas
    for col in range(1, len(seq)):
        for key in tagStateDict.keys():
            tag_row = tagStateDict[key]
            word_tag = seq[col].lower() + '|'+ key
            if word_tag in emissionProbdict.keys():
                possible_probs = []
                for key2 in tagStateDict.keys():
                    tag_row2 = tagStateDict[key2]
                    tag_prevtag = key + '|'+ key2
                    if tag_prevtag in transitionProbdict.keys():
                        if viterbiProb[tag_row2, col-1]>0:
                            possible_probs.append(viterbiProb[tag_row2, col-1]*transitionProbdict[tag_prevtag]*emissionProbdict[word_tag])
                ##Escoger ahora el maximo de los elementos por columna
                viterbiProb[tag_row, col] = max(possible_probs)
    return viterbiProb

matrix = ViterbiMatrix('el mundo es pequeño')
#print(matrix)

##Determinar etiquetas a partir de matriz de viterbi

def ViterbiTags (secuencia, transitionProbdict = transitionProbdict, emissionProbdict = emissionProbdict, tagStateDict = tagStateDict, initTagStateProb = initTagStateProb):
    ##El primer paso es determinar las p de la primera columna
    
    seq = word_tokenize(secuencia)
    viterbiProb = np.zeros((17, len(seq)))
    
    for key in tagStateDict.keys():
        tag_row = tagStateDict[key]
        word_tag = seq[0].lower() + '|'+ key
        if word_tag in emissionProbdict.keys():
            viterbiProb[tag_row, 0] = initTagStateProb[key]*emissionProbdict[word_tag]
    #return viterbiProb

    ##Computo de las p de las demas columnas
    for col in range(1, len(seq)):
        for key in tagStateDict.keys():
            tag_row = tagStateDict[key]
            word_tag = seq[col].lower() + '|'+ key
            if word_tag in emissionProbdict.keys():
                possible_probs = []
                for key2 in tagStateDict.keys():
                    tag_row2 = tagStateDict[key2]
                    tag_prevtag = key + '|'+ key2
                    if tag_prevtag in transitionProbdict.keys():
                        if viterbiProb[tag_row2, col-1]>0:
                            possible_probs.append(viterbiProb[tag_row2, col-1]*transitionProbdict[tag_prevtag]*emissionProbdict[word_tag])
                ##Escoger ahora el maximo de los elementos por columna
                viterbiProb[tag_row, col] = max(possible_probs)
    
    ##Construccion de la secuencia de etiquetas (palabra y tag)
    res = []
    for i, p in enumerate(seq):
        for tag in tagStateDict.keys():
            if tagStateDict[tag] == np.argmax(viterbiProb[:, i]):
                res.append((p, tag))
    return res

vector = ViterbiTags('el mundo es pequeño')
#print(vector)


##Ahora se peude hacer un entrenamiento directo con NLTK

train_data = treebank.tagged_sents()[:3900]
#print(train_data)

tagger = hmm.HiddenMarkovModelTrainer().train_supervised(train_data)

test = tagger.tag('Pierre Vinken will get old'.split())
print(test)

check = tagger.evaluate(train_data)
print(check)