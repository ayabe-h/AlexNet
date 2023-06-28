import numpy as np
import matplotlib.pyplot as plt

loss=np.loadtxt('./model/loss.txt')

N=len(loss)

epoch=[i for i in range(N)]

plt.plot(epoch, loss)

plt.savefig('./figs/loss.png')

plt.show()