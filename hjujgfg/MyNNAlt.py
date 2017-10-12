import numpy as np
import random
import math
import matplotlib.pyplot as plt
import plotter as myPlot


def activate(output):
    return 1.0 / (1.0 + np.exp(-output))

def activateRelu(output):
    return max([0, output])

def linearActivation(output):
    return np.array([[np.sum(output)]])

def linearActivationDerivative(ouput):
    return 0

def output_derivative(value):
    return float(value) * (1.0 - value)

def output_derivativeRelu(value):
    if (value > 0):
        return 1
    else:
        return 0

vectorizedDerivative = np.vectorize(output_derivative)
vectorizedActivation = np.vectorize(activate)

vectorizedActivationRelu = np.vectorize(activateRelu)
vectorizedDerivativeRelu = np.vectorize(output_derivativeRelu)


class Layer:
    def __init__(self, neuron_number, prev_neuron_number):
        self.weights = np.matrix(np.ones([neuron_number, prev_neuron_number]))
        self.weights = np.multiply(self.weights, np.random.random_sample([neuron_number, prev_neuron_number]))
        self.biases = np.multiply(np.ones([neuron_number, 1]), np.random.random_sample([neuron_number, 1]))
        self.outputs = None
        self.error_terms = None
        self.derivative_w = None
        self.derivative_b = None
        self.delta_W = np.matrix(np.zeros([neuron_number, prev_neuron_number]))
        self.delta_b = np.zeros([neuron_number, 1])

    def applyWeights(self, input):
        self.zeds = self.weights * input + self.biases

    def activate(self, input, isLast):
        self.applyWeights(input)
        #self.outputs = vectorizedActivationRelu(self.zeds)
        #attempt to use linear function for last layer
        if not isLast:
            self.outputs = vectorizedActivationRelu(self.zeds)
        else:
            self.outputs = linearActivation(self.zeds)

    def calc_derivatives(self, prev_outputs):
        #print "Self deltas: \n" + str(self.deltas)
        #print "\nPrev out: \n" + str(prev_outputs)
        self.derivative_w = self.error_terms * prev_outputs.T
        self.derivative_b = self.error_terms

    def update_delta(self):
        self.delta_W = self.delta_W + self.derivative_w
        self.delta_b = self.delta_b + self.derivative_b

    def update_weights(self, alpha, lamb, train_size):
        #print "weights before update: " + str(self.weights)
        self.weights = self.weights - (alpha * (((1/float(train_size)) * self.delta_W) + (lamb * self.weights)))
        #print "weights after update: " + str(self.weights)
        self.biases = self.biases - (alpha * ((1/float(train_size)) * self.delta_b))
        self.delta_W = self.delta_W * 0
        self.delta_b = self.delta_b * 0

    def pretrain(self):
        self.error_terms = None
        self.derivative_w = None
        self.derivative_b = None
        self.delta_W = np.matrix(np.zeros(self.weights.shape))
        self.delta_b = np.zeros([self.weights.shape[0], 1])


    def __str__(self):
        return "\nWeights shape " + str(self.weights.shape) + \
               "\nBiases shape " + str(self.biases.shape) + \
               "\nerror_terms shape " + (str(self.error_terms.shape) if self.error_terms is not None else "None") + \
               "\nDerivative W shape " + (str(self.derivative_w.shape) if self.derivative_w is not None else "None") + \
               "\nDerivative b shape " + (str(self.derivative_b.shape) if self.derivative_b is not None else "None") + \
               "\nOutputs shape " + (str(self.outputs.shape) if self.outputs is not None else "None") + \
               "\nWeights: \n" + \
               "" + str(self.weights) + \
               "\nBiases: " + str(self.biases) + \
               "\nOutputs: " + str(self.outputs) + \
               "\nerror_terms: " + str(self.error_terms) + \
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
            #print "layer --- " + str(i)
            #print l.weights.shape
            if i == 0:
                #l.error_terms = - np.multiply((res - l.outputs), vectorizedDerivativeRelu(l.zeds))
                #for linear
                l.error_terms = - np.multiply(( res - l.outputs), np.ones([len(l.outputs), 1]))
            else:
                l.error_terms = np.multiply((prev.weights.T * prev.error_terms), vectorizedDerivativeRelu(l.zeds))
            #print l.deltas
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

    def pretrain(self):
        for l in self.layers:
            l.pretrain()

    def __str__(self):
        res = ("START OF NN\nNeural network\n" +
               "----------------\n" +
               "Input size: " + str(self.input_size) +
               "\nOutput size: " + str(self.output_size))
        res = res + "\n Printing layers: " + str(len(self.layers))
        i = 0
        for l in self.layers:
            res = res + "\n-----------------------\n"
            res = res + "\n Layer #" + str(i) + str(l)
            i += 1
        res = res + "\n END OF NN"
        return res

def train(net, alpha = 0.3, lamb = 0.003, epochs=300, train_number = 360, show_plots=False):
    errors = []
    func = []
    train_set = generate_set(train_number, calc_sin)
    #normalization_result = mean_normalize(train_set['inputs'])
    #inputs = normalization_result['normalized']
    #outputs = train_set['outputs']
    inputs = train_set['inputs']
    outputs = train_set['outputs']
    net.pretrain()
    #plt.plot([])
    #plt.show()
    #myPlot.init()
    plt.axis([0, train_number, 0, 1])
    plt.ion()
    plt.show()

    for epoch in range(epochs):
        sum_error = 0
        func = []
        net_res = []
        for j in range(train_number):
            #rand = random.random() * 360
            #val = inputs[j]
            #inp = [val]
            #out = [(outputs[j] - normalization_result['mean']) / normalization_result['denominator']]

            inp = [float(inputs[j]) / train_number]
            out = [scale_value(outputs[j], -1, 1, 0, 1)]
            #out = [outputs[j]]

            func.append(out)

            net.run(inp)
            net.back(out)
            sum_error += np.sum((net.out() - out) ** 2)
            net_res.append(net.out()[0,0])
        #myPlot.plot(func)
        #plt.cla()
        plt.plot(net_res)
        #error = (1 / float(train_number)) * 0.5 * sum_error
        error = 0.5 * sum_error
        errors.append(error)
        if len(errors) > 1 and errors[epoch] >= errors[epoch - 1]:
                print('>epoch=%d, lrate=%.3f, error=%.8f, acutally trained:%.3f' % (epoch, alpha, error, train_number))
                print 'stopping without weights update!'
                break
        net.update_weights(alpha, lamb, train_number)
        #res = test(net, train_number)
        #if epoch == 0:
            #plt.plot(func)
            #plt.plot(res)
        #else:
            #plt.plot([])
            #plt.plot(res)
            #plt.draw()
        print('>epoch=%d, lrate=%.3f, error=%.8f, acutally trained:%.3f' % (epoch, alpha, error, train_number))
    if show_plots:
        plot(errors)
        plot(func)
        return func


def scale_value(value, old_min, old_max, new_min, new_max):
    return ( float(value - old_min) / ( float(old_max - old_min) / (new_max - new_min) ) ) + new_min

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
        #input = random.random() * 360
        output = func(input)[0]
        inputs.append(input)
        outputs.append(output)
    return {'inputs': inputs, 'outputs': outputs}

def test(net, number = 100):
    results = []
    test = generate_set(number, calc_sin)
    for i in range(number):
        inp = [float(test['inputs'][i]) / float(number)]
        #print inp
        res = net.run_res(inp)
        results.append(res[0,0])
        #results.append(scale_value(res[0,0], 0, 1, 0, 360))
        #results.append(scale_value(res, 0, 1, -1, 1))
    plot(results)
    return results

def calc_sin(input):
    return [math.sin(input * math.pi / 180)]

def calc_log(input):
    return [math.log10(input + 10)]

def calc_line(input):
    return [input]

def calc_ident(input):
    return [input]

def plot(arr):
    plt.plot(arr)
    plt.show()

def plot_two(arr1, arr2):
    plt.plot(arr1)
    plt.plot(arr2)
    plt.show()

def build_logical_training_set(func):
    inputs = []
    inputs.append([0., 0.])
    inputs.append([0., 1.])
    inputs.append([1., 0.])
    inputs.append([1., 1.])
    outputs = []
    for inp in inputs:
        outputs.append(func(inp[0], inp[1]))
    return {"inputs":inputs, "outputs":outputs}


def calc_and(a, b):
    return [float(a and b)]

def calc_or(a, b):
    return [float(a or b)]


def train_logical(net, epochs=10, alpha=0.3, lamb=0.03, func = calc_and):
    train_set = build_logical_training_set(func)
    errors = []
    for epoch in range(epochs):
        iterator = 0
        sum_error = 0
        for j in train_set['inputs']:
            net.run(j)
            out = train_set['outputs'][iterator]
            net.back(out)
            print net.layers[0].weights
            iterator += 1
            sum_error += np.sum((net.out() - out) ** 2)
        error = sum_error
        errors.append(error)
        if len(errors) > 1 and errors[epoch] >= errors[epoch - 1]:
            print('>epoch=%d, lrate=%.3f, error=%.8f, actually trained:%.3f' % (epoch, alpha, error, iterator))
            print 'stopping without weights update!'
            break
        net.update_weights(alpha, lamb, iterator)
        print('>epoch=%d, lrate=%.3f, error=%.8f, acutally trained:%.3f' % (epoch, alpha, error, iterator))


def test_and(net, func = calc_and):
    set = build_logical_training_set(func)
    for inp in set['inputs']:
        res = net.run_res(inp)
        print "Input: ", inp
        print "Calculated: ", res
        print "Expected: ", func(inp[0], inp[1])