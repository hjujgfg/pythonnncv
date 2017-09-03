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

def create_nn(dimensions):
    nn = {}
    nn['layers'] = []
    prev = dimensions[0]
    i = 0
    nn['input_size'] = prev
    nn['output_size'] = dimensions[-1]
    for dim in dimensions:
        if i != 0:
            layer = {}
            weights = 0
            #if i != 0:
            weights = np.arange((prev + 1) * dim).reshape(dim, prev + 1)
            prev = dim
            #else:
            #    weights = np.arange(prev * dim).reshape(dim, prev)
            weights = np.matrix(weights)
            weights = np.multiply(weights, random.random())
            layer['weights'] = weights
            layer['delta_weights'] = weights * 0
            nn['layers'].append(layer)
        i += 1
    print_net(nn)
    return nn

def activate_layer(input, weights):
    return weights * input

def scale_vector(vector):
    min = np.amin(vector)
    result = {}
    if len(vector) != 1:
        offseted = vector - min
    else:
        offseted = vector
    max = np.amax(offseted)
    result['input_min'] = min
    result['input_max'] = max
    if max != 0:
        scaled = offseted * (1 / max)
    else:
        scaled = offseted
    result['array'] = scaled
    return result

def unscale_vector(vec, net):
    unscaled = vec * net['input_max']
    unshifted = unscaled + net['input_min']
    return unshifted

def forward_propagate(input, net):
    temp = scale_vector(np.array([input]))
    net['input_min'] = temp['input_min']
    net['input_max'] = temp['input_max']
    input = np.append(temp['array'].T, [[1]], axis=0)
    vectorizedActivation = np.vectorize(activate)
    layerNumber = 0
    net['input'] = input
    for l in net['layers']:
        outputs = vectorizedActivation(activate_layer(input, l['weights']))
        l['outputs'] = outputs
        input = np.append(l['outputs'], [[1]], axis=0)
        layerNumber += 1

def back_propagate(expected, net):
    counter = 0
    prev = 0
    for l in reversed(net['layers']):
        errors = 0
        toDerivate = 0;
        if counter == 0:
            test = np.array([expected]).reshape(len(expected), 1)
            if (net['input_max'] != 0) :
                test = (test - net['input_min']) * (1 / net['input_max'])
            else:
                test = (test - net['input_min'])
            errors = l['outputs'] - test
            errors = np.append(errors, [[0]], axis=0)
        else:
            errors = next['weights'].T * next['deltas'][:-1]
        toDerivate = np.append(l['outputs'], [[1]], axis=0) #we add bias unit, which outputs always 1
        derivatives = vectorizedDerivative(toDerivate)
        l['deltas'] = np.multiply(errors, derivatives)
        counter += 1
        next = l

def get_outputs(net):
    return unscale_vector(net['layers'][-1]['outputs'], net)

def train_encoder(net, alpha = 0.5, lamb = 0.05, epochs=500, train_number = 1000):
    errors = []
    for epoch in range(0, epochs):
        sum_error = 0
        actually_trained = 0
        for j in range(1, train_number):
            #inp = create_random_vector(net['input_size'])
            #out = inp

            inp = [j]
            out = calc_sin(j)

            #function_to_train.append(out)
            #print("Input: ", inp)
            #print("Output: ", out)
            forward_propagate(inp, net)
            back_propagate(out, net)
            compute_delta_weights(net)
            sum_error += np.sum( (get_outputs(net) - np.array(out)) ** 2 )
            actually_trained += 1
            if sum_error > 10000000:
                break
        update_weights(net, alpha, lamb, actually_trained)
        print('>epoch=%d, lrate=%.3f, error=%.8f, acutally trained:%.3f' % (epoch, alpha, sum_error, actually_trained))
        errors.append(sum_error)
        if epoch % 100 == 0 and alpha >= 0.15:
            pass
            #alpha -= 0.05
        if len(errors) > 1 and errors[epoch] == errors[epoch - 1]:
            break
    return errors

def test_nn(net, number):
    outputs = []
    for i in range(0, number):
        inp = [i]
        forward_propagate(inp, net)
        outputs.append(get_outputs(net)[0,0])
    return outputs


def create_input(prev):
    return np.array([prev + 1])

def calc_sin(input):
    return [math.sin(input * math.pi / 180)]

def calc_log(input):
    return [math.log10(input + 10)]

def compute_delta_weights(net):
    prev_out = net['input']
    for l in net['layers']:
        l['delta_weights'] = l['delta_weights'] + l['deltas'][:-1] * prev_out.T
        prev_out = np.append(l['outputs'], [[1]], axis=0)

def update_weights(net, alpha, lambd, train_number):
    for l in net['layers']:
        l['weights'] = l['weights'] - alpha * ((1/train_number) * l['delta_weights'] + lambd * l['weights'])
        l['delta_weights'] = l['delta_weights'] * 0

def create_random_vector(size):
    vec = []
    for i in range(size):
        vec.append(random.random())
    return vec

def print_net(net):
    counter = 0
    for i in net['layers']:
        print ("Layer #", counter)
        print (i)
        print ("\n")
        counter += 1

def print_output(net):
    print (net['layers'][-1]['outputs'])

def print_info(net):
    print("Number of levels: ", len(net['layers']))
    print("-----------------\n")
    counter = 0;
    if 'input' in net:
        print ("input was: ")
        print (net['input'])
    else:
        print ("No inputs yet")
    for i in net['layers']:
        print("Layer #", counter)
        print("Weights size: ", i['weights'].shape)
        print('Weights: ')
        print(i['weights'])
        if 'outputs' in i:
            print("outputs shape: ", i['outputs'].shape)
            print("outputs: ")
            print(i['outputs'])
        else:
            print("No outputs for layer yet")
        if 'deltas' in i:
            print("deltas shape: ", i['deltas'].shape)
            print("Deltas: ")
            print(i['deltas'])
        else:
            print("No deltas for layer yet")
        if 'delta_weights' in i:
            print("Delta weights: ")
            print(i['delta_weights'])
        else:
            print("no delta weights")
        counter += 1

def plot(arr):
    plt.plot(arr)
    plt.show()