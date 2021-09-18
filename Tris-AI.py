# --- TO DO LIST ---
#
# Vettorializzare la backpropagation
# Suddividere il progetto in più file
# Aggiungere training data
# Dare i training data a pacchetti
# Testare diverse procedure di scelta dei training data
# Testare altre cost function (Cross entropy?)
# Migliorare l'interazione con l'utente da terminale
# Migliorare l'inizializzazione del network
# Implementare regolarizzazione (?)
# Implementare softmaxing (?)
# Testare diverse dimensioni del network
# Testare differenti step size
# Testare differenti iterazioni totali di training
# Cambiare il percorso file dei training data
# Semplificare la raccolta di training data
# Renderla una rete convolutiva
# Cambiare e**x con un'exp
# Aggiungere la possibilità di cambiare la profondità
# Capire come funziona la lambda nel sorting
# Riscrivere le variabili e i commenti in inglese
# Mostrare il costo sul terminale aggiornandolo

import numpy as np
from random import randrange

# Parametri
it=35000 # Iterazioni di training
s=0.15 # Step size
size1=4 # Dimensioni layer intermedi
size2=4

# Funzioni matematiche utili
def sigmoid(x):
    e=2.7182818284590
    return 1/(1+e**(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

# Classe del Network 
class Network(object):
    
    # Inizializzazione del Network
    def __init__(self):
        self.li=[0 for i in range (18)] # Layer
        self.l1=[0 for i in range (size1)]
        self.l2=[0 for i in range (size2)]
        self.lo=[0 for i in range (9)]
        self.expected=[0 for i in range (9)] # Output corretto
        self.p1=[0 for i in range (size1)] # Copia dei layer, per tenere il risultato prima di applicare la sigmoide
        self.p2=[0 for i in range (size2)]
        self.po=[0 for i in range (9)]
        self.w1=[[0.5 for i in range (size1)] for j in range (18)] # Pesi
        self.w2=[[0.5 for i in range (size2)] for j in range (size1)]
        self.wo=[[0.5 for i in range (9)] for j in range (size2)]
        self.b1=[0 for i in range (size1)] # Bias
        self.b2=[0 for i in range (size2)]
        self.bo=[0 for i in range (9)]

    # Calcola il network in base a pesi, bias e input
    def calcola_network(self):
        self.p1=np.dot(self.li,self.w1)+self.b1
        self.l1=sigmoid(self.p1)
        self.p2=np.dot(self.l1,self.w2)+self.b2
        self.l2=sigmoid(self.p2)
        self.po=np.dot(self.l2,self.wo)+self.bo
        self.lo=sigmoid(self.po)
        return

    # Cost function quadratica
    def costo(self):
        return np.linalg.norm(self.lo-self.expected)

    # Back propagation
    def back_propagation(self):
        do=np.multiply((self.lo-self.expected),d_sigmoid(self.po))
        d2=np.multiply(np.dot(self.wo,do),d_sigmoid(self.p2))
        d1=np.multiply(np.dot(self.w2,d2),d_sigmoid(self.p1))
        
        dc_b1=d1
        dc_b2=d2
        dc_bo=do
        
        self.b1=self.b1-s*dc_b1
        self.b2=self.b2-s*dc_b2
        self.bo=self.bo-s*dc_bo
        
        dc_w1=[0 for i in range (18)]
        dc_w2=[0 for i in range (size1)]
        dc_wo=[0 for i in range (size2)]
        
        for i in range (18):
            dc_w1[i]=self.li[i]*d1
            self.w1[i]=self.w1[i]-s*self.li[i]*d1
        
        for i in range (size1):
            dc_w2[i]=self.l1[i]*d2
            self.w2[i]=self.w2[i]-s*self.l1[i]*d2
            
        for i in range (size2):
            dc_wo[i]=self.l2[i]*do
            self.wo[i]=self.wo[i]-s*self.l2[i]*do

        return

    # Training
    def train(self, it):
        for i in range (0,it):
            n=randrange(12)
            network.li=data[n][0]
            network.expected=data[n][1]
            network.calcola_network()
            network.back_propagation()
            print(n, network.costo())

# Caricamento dei training data
data=[[[0 for i in range (18)],[0 for j in range (0,9)]] for k in range (12)]
f=open('data','r')
d=f.read()
for i in range (12):
        data[i][0]=list(map(int, d[27*i:27*i+18]))
        data[i][1]=list(map(int, d[27*i+18:27*(i+1)]))
f.close()

# Test del Network
network=Network()
network.train(it)
while True:
    m=int(input("\nChoose data: "))
    network.li=data[m][0]
    network.calcola_network()
    l=sorted(range(len(network.lo)), key=lambda k: network.lo[k])
    l.reverse()
    print("\n L'output layer è:\n", network.lo, "\n\n", "L'ordine di preferenza delle mosse è:\n", l, "\n")

# Mossa corretta in base all'input:
#  0 -> 4
#  1 -> 2
#  2 -> 8
#  3 -> 0
#  4 -> 6
#  5 -> 0
#  6 -> 6
#  7 -> 2
#  8 -> 3
#  9 -> 1
# 10 -> 7
# 11 -> 8
