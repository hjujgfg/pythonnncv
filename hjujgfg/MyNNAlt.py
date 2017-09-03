import numpy as np
import random
import math
import matplotlib.pyplot as plt


def activate(output):
    return 1.0 / (1.0 + np.exp(-output))

def output_derivative(value):
    return value * (1.0 - value)

vectorizedDerivative = np.vectorize(output_derivative)
vectorizedActivation = np.vectorize(activate)

class Layer:
    def __init__(self, neuron_number, prev_neuron_number):
        self.weights = np.matrix(np.ones([neuron_number, prev_neuron_number]))
        self.weights = np.multiply(self.weights, np.random.random_sample([neuron_number, prev_neuron_number]))
        self.biases = np.multiply(np.ones([neuron_number, 1]), np.random.random_sample([neuron_number, 1]))
        self.outputs = None
        self.deltas = None
        self.derivative_w = None
        self.derivative_b = None
        self.delta_W = np.matrix(np.zeros([neuron_number, prev_neuron_number]))
        self.delta_b = np.zeros([neuron_number, 1])

    def applyWeights(self, input):
        self.zeds = self.weights * input + self.biases

    def activate(self, input):
        self.applyWeights(input)
        self.outputs = vectorizedActivation(self.zeds)

    def calc_derivatives(self, prev_outputs):
        self.derivative_w = self.deltas * prev_outputs.T
        self.derivative_b = self.deltas

    def update_delta(self):
        self.delta_W = self.delta_W + self.derivative_w
        self.delta_b = self.delta_b + self.derivative_b

    def update_weights(self, alpha, lamb, train_size):
        self.weights = self.weights - alpha * ((1/train_size) * self.delta_W - lamb * self.weights)
        self.biases = self.biases - alpha * ((1/train_size) * self.delta_b)
        self.delta_W = self.delta_W * 0
        self.delta_b = self.delta_b * 0



    def __str__(self):
        return "Weights: \n" \
               "" + str(self.weights) + \
               "\nBiases: " + str(self.biases) + \
               "\nOutputs: " + str(self.outputs) + \
               "\nDeltas: " + str(self.deltas) + \
               "\nDerivative W: " + str(self.derivative_w) + \
               "\nDerivative b: " + str(self.derivative_b)




class NN:
    def __init__(self, dimensions):
        self.layers = []
        prev = dimensions[0]
        i = 0
        self.input_size = dimensions[0]
        self.output_size = dimensions[-1]
        for dim in dimensions:
            if i == 0:
                pass
            else:
                self.layers.append(Layer(dim, prev))
            prev = dim
            i += 1

    def run(self, input):
        prev = np.array(input).reshape(len(input), 1)
        self.input = prev
        for l in self.layers:
            l.activate(prev)
            prev = l.outputs

    def run_res(self, input):
        self.run(input)
        return self.out()

    def back(self, expected):
        res = np.array(expected).reshape(len(expected), 1)
        i = 0
        for l in reversed(self.layers):
            if i == 0:
                l.deltas = -np.multiply((res - l.outputs), vectorizedDerivative(l.outputs))
            else:
                l.deltas = np.multiply((prev.weights.T * prev.deltas), vectorizedDerivative(l.outputs))
            prev = l
            i += 1
        prev = self.input
        for l in self.layers:
            l.calc_derivatives(prev)
            l.update_delta()

    def out(self):
        return self.layers[-1].outputs

    def update_weights(self, alpha, lamb, train_size):
        for l in self.layers:
            l.update_weights(alpha, lamb, train_size)

    def __str__(self):
        res = ("START OF NN\nNeural network\n" +
               "----------------\n" +
               "Input size: " + str(self.input_size) +
               "\nOutput size: " + str(self.output_size))
        for l in self.layers:
            res = res + "\n" + str(l)
        res = res + "\n END OF NN"
        return res

def train(net, alpha = 0.5, lamb = 0.05, epochs=500, train_number = 1000):
    errors = []
    func = []
    for epoch in range(epochs):
        sum_error = 0
        for j in range(train_number):

            inp = [j]
            out = calc_line(j)
            func.append(out)

            net.run(inp)
            net.back(out)
            sum_error += np.sum((net.out() - out) ** 2)
        net.update_weights(alpha, lamb, train_number)
        print('>epoch=%d, lrate=%.3f, error=%.8f, acutally trained:%.3f' % (epoch, alpha, sum_error, train_number))
        errors.append(sum_error)
        if len(errors) > 1 and errors[epoch] == errors[epoch - 1]:
            break
    plot(errors)
    plot(func)
    return errors

def test(net, number = 1000):
    results = []
    for i in range(number):
        inp = [i]
        res = net.run_res(inp)
        results.append(res[0,0])
    return results

def calc_sin(input):
    return [math.sin(input * math.pi / 180)]

def calc_log(input):
    return [math.log10(input + 10)]

def calc_line(input):
    return [input]

def plot(arr):
    plt.plot(arr)
    plt.show()


