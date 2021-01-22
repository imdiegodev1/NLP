##import wordnet in spanish
import nltk
##nltk.download('omw')
from nltk.corpus import wordnet as wn

##librerias para graficar
import networkx as nx
import matplotlib.pyplot as plt

#synsets = grupo de sinonimos
##consultar un synset
ss = wn.synsets('carro', lang = 'spa')

##explore synset
for syn in ss:
    ##Consultar mas informacion dentro de ss. info de elementos de la lista
    print(syn.name(), ' : ', syn.definition())
    for name in syn.lemma_names():
        ##consultar ya como tal las palabras dentro de cada elemento de la lista de ss
        print(' * ', name)

##construir o agrupar informaciòn para crear el grafo (no es el grafo)
def clousure_graph(synset, fn):
    seen = set()
    graph = nx.DiGraph()
    labels = {}
    def recurse(s):
        if not s in seen:
            seen.add(s)
            labels[s.name] = s.name().split('.')[0]
            graph.add_node(s.name)
            for s1 in fn(s):
                graph.add_node(s1.name)
                graph.add_edge(s.name, s1.name)
                recurse(s1)
    recurse(synset)
    return graph, labels

##funcion para graficar la informaciòn organizada
def draw_text_graph(G, labels):
    plt.figure(figsize=(18, 12))
    pos = nx.planar_layout(G, scale=18)
    nx.draw_networkx_nodes(G, pos, node_color="red", linewidths=0, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=20, labels=labels)
    nx.draw_networkx_edges(G, pos)
    plt.xticks([])
    plt.yticks([])
    plt.show()

##obtener hiponimo
ss_hypo = ss[0].hyponyms()
    ##graficarlos
G, labels = clousure_graph(ss[0], fn = lambda s: s.hyponyms())
#print(draw_text_graph(G, labels))

##obtener hyperonimos
ss_hyper = ss[0].hyponyms()
    ##graficarlos
G2, labels2 = clousure_graph(ss[0], fn = lambda s: s.hypernyms())
#print(draw_text_graph(G2, labels2))

##Similitud semantica
def show_syns(word):
    ss = wn.synsets(word, lang = 'spa')
    for syn in ss:
        print(syn.name(), ' : ', syn.definition())
        for name in syn.lemma_names():
            print(' * ', name)
    return ss

##prueba para la funcion show_syns, aqui tenemos ejemplo 3 palabras
ss2 = show_syns('gato')
ss3 = show_syns('animal')
ss4 = show_syns('perro')

##al hacer analisis de similitud semantica debemos definir que definicion vamos a tomar
##del grupo de sinonimos
perro = ss4[0]
animal = ss3[0]
gato = ss2[0]

##determinar la similitud semantica entre dos palabras
print(animal.path_similarity(perro))
print(gato.path_similarity(perro))