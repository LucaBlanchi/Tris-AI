import Network
import Utilities

# Parametri
it=35000 # Iterazioni di training
s=0.15 # Step size
size1=4 # Dimensioni layer intermedi
size2=4

# Caricamento dei training data
data=[[[0 for i in range (18)],[0 for j in range (0,9)]] for k in range (12)]
f=open('data','r')
d=f.read()
for i in range (12):
        data[i][0]=list(map(int, d[27*i:27*i+18]))
        data[i][1]=list(map(int, d[27*i+18:27*(i+1)]))
f.close()

# Test del Network
network=Network.Network(size1,size2)
network.train(it,data,s)
while True:
    m=int(input("\nChoose data: "))
    network.li=data[m][0]
    network.calcola_network()
    l=sorted(range(len(network.lo)), key=lambda k: network.lo[k])
    l.reverse()
    print("\n L'output layer è:\n", network.lo, "\n\n", "L'ordine di preferenza delle mosse è:\n", l, "\n")
