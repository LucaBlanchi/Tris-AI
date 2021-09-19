e=2.7182818284590

def sigmoid(x):
    return 1/(1+e**(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x)) 
