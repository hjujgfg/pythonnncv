import numpy as np
import random

def activate(output):
    return 1.0 / (1.0 + np.exp(-output))
def output_derivative(value):
    return value * (1.0 - value)

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
    offeseted = vector - min
    max = np.amax(offeseted)
    result['input_min'] = min
    result['input_max'] = max
    scaled = offeseted * (1 / max)
    result['array'] = scaled
    return result

def unscale_vector(vec, net):
    unscaled = vec * net['input_max']
    unshifted = unscaled + net['input_min']
    return unshifted

def forward_propagate(input, net):
    #print ("Attempting to forward propagate on inputs: ", input)
    temp = scale_vector(np.array([input]))
    net['input_min'] = temp['input_min']
    net['input_max'] = temp['input_max']
    input = np.append(temp['array'].T, [[1]], axis=0)
    #print ("Constructed input: ", input)
    #print ('prepared the following array: \n', input)
    vectorizedActivation = np.vectorize(activate)
    layerNumber = 0
    net['input'] = input
    for l in net['layers']:
        outputs = vectorizedActivation(activate_layer(input, l['weights']))
        l['outputs'] = outputs
        input = np.append(l['outputs'], [[1]], axis=0)
        #print ("calculated outputs for layer #", layerNumber)
        #print ("\n", l['outputs'])
        #print ("input vector for next layer: \n", input)
        layerNumber += 1
    ##print_net(net)

def back_propagate(expected, net):
    counter = 0
    prev = 0
    for l in reversed(net['layers']):
        errors = 0
        toDerivate = 0;
        if counter == 0:
            test = np.array([expected]).reshape(len(expected), 1)
            test = (test - net['input_min']) * (1 / net['input_max'])
            #print("\ntest array: ", test)
            #print("\nOutputs: ", l['outputs'])
            errors = test - l['outputs']
            errors = np.append(errors, [[1]], axis=0)
            #toDerivate = l['outputs']
        else:
            errors = next['weights'].T * next['deltas'][:-1]
        toDerivate = np.append(l['outputs'], [[1]], axis=0) #we add bias unit, which outputs always 1
        #print ("calculated errors for layer #", len(net['layers']) - counter)
        #print ("\n", errors)
        vectorizedDerivative = np.vectorize(output_derivative)
        derivatives = vectorizedDerivative(toDerivate)
        #print("\nderivatives: \n")
        #print(derivatives)
        l['deltas'] = np.multiply(errors, derivatives)
        #print ('\ncalculated deltas for layer #', len(net['layers']) - counter)
        #print ('\n', l['deltas'])
        counter += 1
        next = l

def get_outputs(net):
    return unscale_vector(net['layers'][-1]['outputs'], net)

def train_encoder(net, alpha = 0.5, lamb = 0.02, epochs=500, train_number = 500):
    for epoch in range(0, epochs):
        sum_error = 0
        for j in range(train_number):
            inp = create_random_vector(net['input_size'])
            out = inp
            forward_propagate(inp, net)
            back_propagate(out, net)
            compute_delta_weights(net)
            sum_error += np.sum( (get_outputs(net) - out) ** 2 )
        update_weights(net, alpha, lamb, train_number)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, alpha, sum_error))
        if epoch % 50 == 0 and alpha >= 0.15:
            alpha -= 0.05


def compute_delta_weights(net):
    prev_out = net['input']
    for l in net['layers']:
        l['delta_weights'] = l['delta_weights'] + l['deltas'][:-1] * prev_out.T
        prev_out = np.append(l['outputs'], [[1]], axis=0)

def update_weights(net, alpha, lambd, train_number):
    for l in net['layers']:
        l['weights'] = l['weights'] - alpha * (1/train_number * l['delta_weights'] + lambd * l['weights'])
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
    print("\nNumber of levels: ", len(net['layers']))
    print("\n-----------------\n")
    counter = 0;
    for i in net['layers']:
        print("Layer #", counter)
        print("\nWeights size: ", i['weights'].shape)
        if 'outputs' in i:
            print("\noutputs: ", i['outputs'].shape)
        else:
            print("\nNo outputs for layer yet")
        if 'deltas' in i:
            print("\ndeltas: ", i['deltas'].shape)
        else:
            print("\nNo deltas for layer yet")
        counter += 1