import matplotlib.pyplot as plt
import numpy as np
import time

def do_stuff():
    x = np.linspace(0, 10 * np.pi, 100)
    y = np.sin(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'b-')

    plt.show()
    for phase in np.linspace(0, 10 * np.pi, 100):
        line1.set_ydata(np.sin(0.5 * x + phase))
        fig.canvas.draw()
        time.sleep(4)