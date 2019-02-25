import numpy
import random

class Network(object) :

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y,1) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y,x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a) :
        for b, w in zip(self.biases, self.weights) :
            a = sigmoid(numpy.dot(w,a) + b)
        return a

    def stochasticGradientAlgorithm(self, training_data, epochs, mini_batch_size, eta, test_data = None) :
        training_data = list(training_data)
        training_data_size = len(training_data)

        if(test_data) :
            test_data = list(test_data)
            test_data_size = len(test_data)

        for i in range(epochs) :
            random.shuffle(training_data)

            mini_batches = [training_data[k: k + mini_batch_size]
                            for k in range(0, training_data_size, mini_batch_size)]
            for mini_batch in mini_batches :
                self.update_rule(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(i,self.evaluate(test_data),test_data_size));
            else:
                
                print("Epoch {} complete".format(i))

    def update_rule(self, mini_batch, eta) :
        partial_b = [numpy.zeros(b.shape) for b in self.biases]
        partial_w = [numpy.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch :
            delta_partial_b, delta_partial_w = self. backprop(x,y)
            partial_b = [pb + dpb
                        for pb, dpb in zip(partial_b, delta_partial_b)]
            partial_w = [pw + dpw
                        for pw, dpw in zip(partial_w, delta_partial_w)]
            self.weights = [w - (eta/len(mini_batch)) * cW
                            for w, cW in zip(self.weights, partial_w)]
            self.biases = [b - (eta/len(mini_batch)) * cB
                            for b, cB in zip(self.biases, partial_b)]

    def backprop(self, x, y):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(numpy.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def printBias(self):
        print(self.biases)

def sigmoid(z) :
    return 1.0 / (1.0 + numpy.exp(-z))

def sigmoid_prime(z) :
    return sigmoid(z) * (1-sigmoid(z))
