import numpy as np
import matplotlib.pyplot as plt



def init():
    plt.axis([0, 10, 0, 1])
    plt.ion()

def plot(arr):
    plt.cla()
    plt.plot(arr)