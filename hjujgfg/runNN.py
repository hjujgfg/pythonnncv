import MyNNAlt as nn

net = nn.NN([1,10,1])

#net.run([1.,1.])
#print net
#net.back([1])
#print net
#net.update_weights(3, 0.03, 1)

nn.train(net)