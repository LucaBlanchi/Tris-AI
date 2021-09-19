from random import randrange
import numpy as np
import Utilities

# Classe del Network 
class Network(object):
    
    # Inizializzazione del Network
    def __init__(self,size1,size2):
        self.size1=size1
        self.size2=size2
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
        self.l1=Utilities.sigmoid(self.p1)
        self.p2=np.dot(self.l1,self.w2)+self.b2
        self.l2=Utilities.sigmoid(self.p2)
        self.po=np.dot(self.l2,self.wo)+self.bo
        self.lo=Utilities.sigmoid(self.po)
        return

    # Cost function quadratica
    def costo(self):
        return np.linalg.norm(self.lo-self.expected)

    # Back propagation
    def back_propagation(self,s):
        do=np.multiply((self.lo-self.expected),Utilities.d_sigmoid(self.po))
        d2=np.multiply(np.dot(self.wo,do),Utilities.d_sigmoid(self.p2))
        d1=np.multiply(np.dot(self.w2,d2),Utilities.d_sigmoid(self.p1))
        
        dc_b1=d1
        dc_b2=d2
        dc_bo=do
        
        self.b1=self.b1-s*dc_b1
        self.b2=self.b2-s*dc_b2
        self.bo=self.bo-s*dc_bo
        
        dc_w1=[0 for i in range (18)]
        dc_w2=[0 for i in range (self.size1)]
        dc_wo=[0 for i in range (self.size2)]
        
        for i in range (18):
            dc_w1[i]=self.li[i]*d1
            self.w1[i]=self.w1[i]-s*self.li[i]*d1
        
        for i in range (self.size1):
            dc_w2[i]=self.l1[i]*d2
            self.w2[i]=self.w2[i]-s*self.l1[i]*d2
            
        for i in range (self.size2):
            dc_wo[i]=self.l2[i]*do
            self.wo[i]=self.wo[i]-s*self.l2[i]*do

        return

    # Training
    def train(self, it, data, s):
        for i in range (0,it):
            n=randrange(12)
            self.li=data[n][0]
            self.expected=data[n][1]
            self.calcola_network()
            self.back_propagation(s)
            print(n, self.costo())
