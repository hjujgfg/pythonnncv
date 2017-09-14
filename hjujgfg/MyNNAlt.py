import numpy as np
import random
import math
import matplotlib.pyplot as plt


def activate(output):
    return 1.0 / (1.0 + np.exp(-output))

def linearActivation(output):
    return np.sum(output)
def linearActivationDerivative(ouput):
    return

def output_derivative(value):
    return value * (1.0 - value)

vectorizedDerivative = np.vectorize(output_derivative)
vectorizedActivation = np.vectorize(activate)

class Layer:
    def __init__(self, neuron_number, prev_neuron_number):
        self.weights = np.matrix(np.ones([neuron_number, prev_neuron_number]))
        self.weights = np.multiply(self.weights, np.random.random_sample([neuron_number, prev_neuron_number])) * 0.001
        self.biases = np.multiply(np.ones([neuron_number, 1]), np.random.random_sample([neuron_number, 1])) * 0.001
        self.outputs = None
        self.deltas = None
        self.derivative_w = None
        self.derivative_b = None
        self.delta_W = np.matrix(np.zeros([neuron_number, prev_neuron_number]))
        self.delta_b = np.zeros([neuron_number, 1])

    def applyWeights(self, input):
        self.zeds = self.weights * input + self.biases

    def activate(self, input, isLast):
        self.applyWeights(input)
        self.outputs = vectorizedActivation(self.zeds)
        #attempt to use linear function for last layer
        #if not isLast:
        #    self.outputs = vectorizedActivation(self.zeds)
        #else:
        #    self.outputs = linearActivation(self.zeds)

    def calc_derivatives(self, prev_outputs):
        #print "Self deltas: \n" + str(self.deltas)
        #print "\nPrev out: \n" + str(prev_outputs)
        self.derivative_w = self.deltas * prev_outputs.T
        self.derivative_b = self.deltas

    def update_delta(self):
        self.delta_W = self.delta_W + self.derivative_w
        self.delta_b = self.delta_b + self.derivative_b

    def update_weights(self, alpha, lamb, train_size):
        #print "weights before update: " + str(self.weights)
        self.weights = self.weights - alpha * ((1/train_size) * self.delta_W - lamb * self.weights)
        #print "weights after update: " + str(self.weights)
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
            l.activate(prev, l == self.layers[-1])
            prev = l.outputs

    def run_res(self, input):
        self.run(input)
        return self.out()

    def back(self, expected):
        res = np.array(expected).reshape(len(expected), 1)
        i = 0
        for l in reversed(self.layers):
            #print "Layer #" + str(len(self.layers) - i)
            if i == 0:
                l.deltas = np.multiply((l.outputs - res), vectorizedDerivative(l.outputs))
            else:
                l.deltas = np.multiply((prev.weights.T * prev.deltas), vectorizedDerivative(l.outputs))
            prev = l
            i += 1
        prev = self.input
        for l in self.layers:
            l.calc_derivatives(prev)
            l.update_delta()
            prev = l.outputs

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

def train(net, alpha = 0.3, lamb = 0.003, epochs=300, train_number = 360):
    errors = []
    func = []
    train_set = generate_set(train_number, calc_sin)
    #normalization_result = mean_normalize(train_set['inputs'])
    #inputs = normalization_result['normalized']
    #outputs = train_set['outputs']
    inputs = train_set['inputs']
    outputs = train_set['outputs']
    for epoch in range(epochs):
        sum_error = 0
        func = []
        for j in range(train_number):
            #rand = random.random() * 360
            #val = inputs[j]
            #inp = [val]
            #out = [(outputs[j] - normalization_result['mean']) / normalization_result['denominator']]

            inp = [inputs[j] / 360]
            out = [scale_value(outputs[j], -1, 1, 0, 1)]

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

def scale_value(value, old_min, old_max, new_min, new_max):
    return ((value - old_min) / (old_max - old_min)) * ((new_max - new_min) + new_min)

def mean_normalize(array):
    """won't be used right now"""
    mean = 0
    max = -999999
    min = 999999
    for i in array:
        mean += i
        if max < i:
            max = i
        if min > i:
            min = i
    denominator = max - min
    mean = mean / len(array)
    print "Mean:%.3f, min:%.3f, max:%.3f" % (mean, min, max)
    new_array = (np.array(array) - mean) * (1/float(denominator))
    return {'normalized':new_array.tolist(), 'mean': mean, 'denominator':denominator}


def generate_set(size, func):
    inputs = []
    outputs = []
    for i in range(0, size):
        input = i #random.random() * 360
        output = func(input)[0]
        inputs.append(input)
        outputs.append(output)
    return {'inputs': inputs, 'outputs': outputs}

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


