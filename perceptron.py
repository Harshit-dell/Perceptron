import numpy as np

# This is a simple implementation of a perceptron using numpy

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
        return x*(1-x)
    
traning_data =np.array([[0,0, 1], 
                        [0,0, 1],
                        [1,0, 0], 
                        [1,0, 1]])


traning_output = np.array([[0,0,1,1]]).T

np.random.seed(1)

synaptic_weight=2*np.random.random((3,1))-1

print("Initial synaptic weights: ")
print(synaptic_weight)

for iteration in range(100000):
        
        input_layer=traning_data
        
        output=sigmoid(np.dot(input_layer,synaptic_weight))
 
        error=traning_output -output
        
        adjustment=error*sigmoid_derivative(output)
        
        synaptic_weight += np.dot(input_layer.T, adjustment)
        
        
print('Synaptic weights after training: ')
print(synaptic_weight)        
        

print('Output after training: ')        
print(output)