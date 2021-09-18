# --- TO DO LIST ---
#
# Creare più parametri
# Vettorializzare la backpropagation
# Aggiungere training data
# Ottimizzare la procedura di training
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
# Cambiare e**x con un exp

import numpy as np
from random import randrange

e=2.7182818284590
s=0.15 # Step size

# Funzioni matematiche utili
def sigmoid(x):
    return 1/(1+e**(-x))
def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class Network(object):
    
    # Inizializzazione del Network
    def __init__(self):
        self.li=[0 for i in range (0,18)] # Layer
        self.l1=[0 for i in range (0,4)]
        self.l2=[0 for i in range (0,4)]
        self.lo=[0 for i in range (0,9)]
        self.expected=[0 for i in range (0,9)] # Output corretto
        self.p1=[0 for i in range (0,4)] # Copia dei layer, per tenere il risultato prima di applicare la sigmoide
        self.p2=[0 for i in range (0,4)]
        self.po=[0 for i in range (0,9)]
        self.w1=[[0.5 for i in range (0,4)] for j in range (0,18)] # Pesi
        self.w2=[[0.5 for i in range (0,4)] for j in range (0,4)]
        self.wo=[[0.5 for i in range (0,9)] for j in range (0,4)]
        self.b1=[0 for i in range (0,4)] # Bias
        self.b2=[0 for i in range (0,4)]
        self.bo=[0 for i in range (0,9)]

    # Calcola il network in base a pesi, bias e input
    def calcola_output(self):
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
        
        dc_w1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        dc_w2=[0,0,0,0]
        dc_wo=[0,0,0,0]
        
        for i in range (0,18):
            dc_w1[i]=self.li[i]*d1
            self.w1[i]=self.w1[i]-s*self.li[i]*d1
        
        for i in range (0,4):
            dc_w2[i]=self.l1[i]*d2
            self.w2[i]=self.w2[i]-s*self.l1[i]*d2
            
        for i in range (0,4):
            dc_wo[i]=self.l2[i]*do
            self.wo[i]=self.wo[i]-s*self.l2[i]*do

        return

# Creazione Network
network=Network()

# Caricamento dei training data
data=[[[0 for i in range (0,18)],[0 for j in range (0,9)]] for k in range (0,12)]
f=open('data','r')
d=f.read()
for i in range (0,12):
        data[i][0]=list(map(int, d[27*i:27*i+18]))
        data[i][1]=list(map(int, d[27*i+18:27*(i+1)]))
f.close()

# Procedura di Training
for i in range (0,35000):
    n=randrange(12)
    network.li=data[n][0]
    network.expected=data[n][1]
    network.calcola_output()
    network.back_propagation()
    print(network.costo())

# Test del Network
while True:
    print()
    m=int(input("Choose data: "))
    network.li=data[m][0]
    print()
    print("L'output layer è:")
    network.calcola_output()
    print(network.lo)
    print()
    print("L'ordine di preferenza delle mosse è:")
    m=sorted(range(len(network.lo)), key=lambda k: network.lo[k])
    m.reverse()
    print(m)
    print()

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
