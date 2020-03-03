import matplotlib.pyplot as plt
import numpy as np

def match(x):
    return np.floor(4 + np.log2(np.sqrt(x) / 224))

if __name__ == '__main__':
    areas = np.arange(1000, 100000, 1000)
    ks = match(areas)
    plt.plot(areas, ks)
    plt.show()
