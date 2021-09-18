# --- TO DO LIST ---
#
# Oggettificare il network
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

import numpy as np
from random import randrange

e=2.7182818284590
s=0.15 # Step size

# Inizializzazione del Network
l1=[0 for i in range (0,4)] # Layer (senza input layer)
l2=[0 for i in range (0,4)]
lo=[0 for i in range (0,9)]
p1=[0 for i in range (0,4)] # Copia dei layer, per tenere il risultato prima di applicare la sigmoide
p2=[0 for i in range (0,4)]
po=[0 for i in range (0,9)]
w1=[[0.5 for i in range (0,4)] for j in range (0,18)] # Pesi
w2=[[0.5 for i in range (0,4)] for j in range (0,4)]
wo=[[0.5 for i in range (0,9)] for j in range (0,4)]
b1=[0 for i in range (0,4)] # Bias
b2=[0 for i in range (0,4)]
bo=[0 for i in range (0,9)]

# Funzioni matematiche utili
def sigmoid(x):
    return 1/(1+e**(-x))
def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

# Calcola il network in base a pesi, bias e input
def calcola_output(li,w1,w2,wo,b1,b2,bo):
    p1=np.dot(li,w1)+b1
    l1=sigmoid(p1)
    p2=np.dot(l1,w2)+b2
    l2=sigmoid(p2)
    po=np.dot(l2,wo)+bo
    lo=sigmoid(po)
    return [p1,p2,po,l1,l2,lo]

# Cost function quadratica
def costo(lo,expected):
    return np.linalg.norm(lo-expected)

# Backpropagation
def back_propagation(li,l1,l2,w1,w2,wo,b1,b2,bo,expected,p1,p2,po):
    do=np.multiply((lo-expected),d_sigmoid(po))
    d2=np.multiply(np.dot(wo,do),d_sigmoid(p2))
    d1=np.multiply(np.dot(w2,d2),d_sigmoid(p1))
    
    dc_b1=d1
    dc_b2=d2
    dc_bo=do
    
    b1=b1-s*dc_b1
    b2=b2-s*dc_b2
    bo=bo-s*dc_bo
    
    dc_w1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    dc_w2=[0,0,0,0]
    dc_wo=[0,0,0,0]
    
    for i in range (0,18):
        dc_w1[i]=li[i]*d1
        w1[i]=w1[i]-s*li[i]*d1
    
    for i in range (0,4):
        dc_w2[i]=l1[i]*d2
        w2[i]=w2[i]-s*l1[i]*d2
        
    for i in range (0,4):
        dc_wo[i]=l2[i]*do
        wo[i]=wo[i]-s*l2[i]*do

    return [w1,w2,wo,b1,b2,bo]

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
    [p1,p2,po,l1,l2,lo]=calcola_output(data[n][0],w1,w2,wo,b1,b2,bo)
    [w1,w2,wo,b1,b2,bo]=back_propagation(data[n][0],l1,l2,w1,w2,wo,b1,b2,bo,data[n][1],p1,p2,po)
    print(costo(calcola_output(data[n][0],w1,w2,wo,b1,b2,bo)[5], data[n][1]))

# Test del Network
while True:
    print()
    m=int(input("Choose data: "))
    print()
    print("L'output layer è:")
    o=calcola_output(data[m][0],w1,w2,wo,b1,b2,bo)[5]
    print(o)
    print()
    print("L'ordine di preferenza delle mosse è:")
    m=sorted(range(len(o)), key=lambda k: o[k])
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
